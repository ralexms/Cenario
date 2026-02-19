import torch
from transformers import pipeline, TextIteratorStreamer, BitsAndBytesConfig
import gc
import os
import threading

class Summarizer:
    """Handles summarization and action point extraction using a local LLM."""

    # Overhead tokens for chat template (system + user role markers, special tokens)
    _TEMPLATE_OVERHEAD = 200
    # Default safe context budget in tokens (leaves room for generation + overhead)
    _DEFAULT_CONTEXT_BUDGET = 12000
    # Overlap between chunks in tokens for context continuity
    _CHUNK_OVERLAP = 200

    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cuda", quantization="4"):
        self.model_id = model_id
        self.device = device
        self.quantization = quantization  # "4", "8", or "none"
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return

        print(f"Loading summarization model: {self.model_id}...")
        try:
            # Set PYTORCH_CUDA_ALLOC_CONF to avoid fragmentation
            # Note: This must be set before PyTorch is initialized to be effective.
            # It is now set in gui/app.py, but keeping it here as a fallback/reminder doesn't hurt,
            # though it likely won't take effect if torch is already loaded.
            if self.device == "cuda":
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            use_cuda = self.device == "cuda" and torch.cuda.is_available()

            if use_cuda and self.quantization == "4":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_id,
                    model_kwargs={"quantization_config": quantization_config},
                    device_map="auto",
                )
            elif use_cuda and self.quantization == "8":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_id,
                    model_kwargs={"quantization_config": quantization_config},
                    device_map="auto",
                )
            elif use_cuda:
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_id,
                    model_kwargs={"torch_dtype": torch.float16},
                    device=0,
                )
            else:
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_id,
                    model_kwargs={"torch_dtype": torch.float16},
                    device=-1,
                )
            print("Summarization model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Ensure cleanup on failure
            self.unload_model()
            raise

    def unload_model(self):
        """Free the model and release GPU memory aggressively."""
        if self.pipe is not None:
            # Delete sub-objects first so their VRAM is freed before the pipe
            if hasattr(self.pipe, 'model'):
                del self.pipe.model
            if hasattr(self.pipe, 'tokenizer'):
                del self.pipe.tokenizer
            del self.pipe
            self.pipe = None

        # Double gc.collect breaks reference cycles from C extensions
        gc.collect()
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("Summarization model unloaded, GPU memory freed")

    # ---- Helper methods ----

    def count_tokens(self, text):
        """Count tokens for a text string using the model's tokenizer."""
        return len(self.pipe.tokenizer.encode(text, add_special_tokens=False))

    def _get_context_budget(self, max_new_tokens):
        """Return safe chunk size in tokens, accounting for generation and overhead."""
        if hasattr(self.pipe.model, 'config') and hasattr(self.pipe.model.config, 'max_position_embeddings'):
            model_max = self.pipe.model.config.max_position_embeddings
        else:
            model_max = 32768  # Conservative default

        # Budget = model context window - generation tokens - template overhead - safety margin
        budget = model_max - max_new_tokens - self._TEMPLATE_OVERHEAD - 512
        # Clamp to our default budget to stay within VRAM limits
        return min(budget, self._DEFAULT_CONTEXT_BUDGET)

    def _split_text_into_chunks(self, text, chunk_token_size, overlap_tokens=None):
        """Split text into chunks respecting newline boundaries with token overlap."""
        if overlap_tokens is None:
            overlap_tokens = self._CHUNK_OVERLAP

        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = self.count_tokens(line + '\n')

            # If a single line exceeds the chunk size, add it as its own chunk
            if line_tokens > chunk_token_size:
                if current_chunk_lines:
                    chunks.append('\n'.join(current_chunk_lines))
                    current_chunk_lines = []
                    current_tokens = 0
                chunks.append(line)
                continue

            if current_tokens + line_tokens > chunk_token_size and current_chunk_lines:
                chunks.append('\n'.join(current_chunk_lines))
                # Overlap: keep trailing lines from previous chunk
                overlap_lines = []
                overlap_count = 0
                for prev_line in reversed(current_chunk_lines):
                    prev_tokens = self.count_tokens(prev_line + '\n')
                    if overlap_count + prev_tokens > overlap_tokens:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_count += prev_tokens
                current_chunk_lines = overlap_lines
                current_tokens = overlap_count

            current_chunk_lines.append(line)
            current_tokens += line_tokens

        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            if chunk_text.strip():
                chunks.append(chunk_text)

        # Filter out empty chunks
        return [c for c in chunks if c.strip()]

    def _get_system_prompt(self):
        """Return the standard system prompt for summarization."""
        return (
            "You are a helpful assistant that summarizes meeting transcripts. "
            "The transcript was generated by automatic speech recognition and may contain "
            "errors such as misspelled words, incorrect names, or garbled phrases. "
            "Use surrounding context to infer the correct meaning when something doesn't make sense. "
            "Do not reproduce transcription errors in your summary. "
            "Your task is to analyze the provided transcript and generate a structured markdown output "
            "containing a summary of the meeting and a list of action points."
        )

    def _build_user_prompt(self, text, detail_level):
        """Build the user prompt for a given detail level."""
        if detail_level == "detailed":
            return (f"Here is the meeting transcript:\n\n{text}\n\n"
                    "Please provide:\n1. A detailed summary of the discussion, covering key points discussed "
                    "and arguments, as well as possible problems or difficulties.\n"
                    "2. A comprehensive list of action points (tasks, decisions, or follow-ups) with assigned owners if mentioned.")
        elif detail_level == "comprehensive":
            return (f"Here is the meeting transcript:\n\n{text}\n\n"
                    "Please provide:\n1. A comprehensive summary of the discussion, including context, key decisions, "
                    "and nuances, as well as possible problems or difficulties.\n"
                    "2. A detailed list of action points (tasks, decisions, or follow-ups).\n"
                    "3. Any open questions or unresolved issues.\nBe as detailed and thorough as possible.")
        else:  # concise
            return (f"Here is the meeting transcript:\n\n{text}\n\n"
                    "Please provide:\n1. A concise summary of the discussion.\n"
                    "2. A list of action points (tasks, decisions, or follow-ups).")

    def _summarize_single(self, system_prompt, user_prompt, max_new_tokens, stream_callback=None):
        """Run a single generation pass. Returns the generated text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        thread = None
        if stream_callback:
            streamer = TextIteratorStreamer(self.pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = dict(
                text_inputs=messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False,
                streamer=streamer
            )

            thread = threading.Thread(target=self.pipe, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                stream_callback(new_text)

            thread.join()
            return generated_text
        else:
            outputs = self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False
            )
            return outputs[0]["generated_text"]

    def _free_kv_cache(self):
        """Free KV cache VRAM between generation passes."""
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Chunked summarization (map-reduce) ----

    def summarize_chunked(self, text, detail_level="concise", max_new_tokens=1024, stream_callback=None):
        """Map-reduce chunked summarization for long transcripts."""
        self.load_model()

        context_budget = self._get_context_budget(max_new_tokens)
        input_tokens = self.count_tokens(text)
        system_prompt = self._get_system_prompt()

        # Determine chunk size: context budget minus the overhead of the user prompt template
        # (the "Here is the meeting transcript..." wrapper is ~50 tokens)
        chunk_content_budget = context_budget - 60

        chunks = self._split_text_into_chunks(text, chunk_content_budget)
        num_chunks = len(chunks)

        if stream_callback:
            stream_callback(f"\n[Chunked mode: {input_tokens} tokens, split into {num_chunks} chunk(s)]\n\n")

        print(f"Chunked summarization: {input_tokens} tokens -> {num_chunks} chunks (budget: {context_budget} tokens/chunk)")

        # If only 1 chunk, skip reduce phase
        if num_chunks == 1:
            user_prompt = self._build_user_prompt(chunks[0], detail_level)
            return self._summarize_single(system_prompt, user_prompt, max_new_tokens, stream_callback)

        # --- MAP phase: summarize each chunk ---
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if stream_callback:
                header = f"\n--- Summarizing chunk {i+1}/{num_chunks} ---\n\n"
                stream_callback(header)

            chunk_prompt = (
                f"Here is part {i+1} of {num_chunks} of a meeting transcript:\n\n{chunk}\n\n"
                "Please provide a detailed summary of this section, covering all key points, "
                "decisions, and action items mentioned."
            )

            try:
                summary = self._summarize_single(system_prompt, chunk_prompt, max_new_tokens, stream_callback)
                chunk_summaries.append(summary)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM during chunk {i+1}/{num_chunks}")
                raise RuntimeError(
                    f"GPU Out of Memory while summarizing chunk {i+1}/{num_chunks}. "
                    "Try a smaller model or enable quantization."
                )

            # Free KV cache between chunks
            self._free_kv_cache()

        # --- REDUCE phase: combine chunk summaries into final summary ---
        combined_summaries = "\n\n".join(
            f"## Summary of Part {i+1}\n{s}" for i, s in enumerate(chunk_summaries)
        )

        # Check if combined summaries fit in context budget; if not, do hierarchical reduce
        combined_tokens = self.count_tokens(combined_summaries)
        if combined_tokens > context_budget - 100:
            combined_summaries = self._hierarchical_reduce(
                chunk_summaries, system_prompt, max_new_tokens, context_budget, stream_callback
            )

        if stream_callback:
            stream_callback("\n\n--- Generating final summary ---\n\n")

        reduce_prompt = (
            f"Below are summaries of {num_chunks} consecutive parts of a single meeting transcript. "
            "Combine them into one unified, coherent summary.\n\n"
            f"{combined_summaries}\n\n"
            "Please provide:\n"
        )

        if detail_level == "detailed":
            reduce_prompt += (
                "1. A detailed summary of the full discussion, covering key points discussed "
                "and arguments, as well as possible problems or difficulties.\n"
                "2. A comprehensive list of action points (tasks, decisions, or follow-ups) with assigned owners if mentioned."
            )
        elif detail_level == "comprehensive":
            reduce_prompt += (
                "1. A comprehensive summary of the full discussion, including context, key decisions, "
                "and nuances, as well as possible problems or difficulties.\n"
                "2. A detailed list of action points (tasks, decisions, or follow-ups).\n"
                "3. Any open questions or unresolved issues.\nBe as detailed and thorough as possible."
            )
        else:
            reduce_prompt += (
                "1. A concise summary of the full discussion.\n"
                "2. A list of action points (tasks, decisions, or follow-ups)."
            )

        self._free_kv_cache()
        return self._summarize_single(system_prompt, reduce_prompt, max_new_tokens, stream_callback)

    def _hierarchical_reduce(self, summaries, system_prompt, max_new_tokens, context_budget, stream_callback):
        """Reduce summaries in pairs when they exceed the context budget."""
        current = list(summaries)
        iteration = 0
        while len(current) > 1:
            iteration += 1
            reduced = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    pair_text = f"Summary A:\n{current[i]}\n\nSummary B:\n{current[i+1]}"
                else:
                    reduced.append(current[i])
                    continue

                if stream_callback:
                    stream_callback(f"\n--- Reducing pair {i//2 + 1} (iteration {iteration}) ---\n")

                pair_prompt = (
                    "Combine the following two partial meeting summaries into one coherent summary "
                    "preserving all key points, decisions, and action items:\n\n"
                    f"{pair_text}"
                )

                result = self._summarize_single(system_prompt, pair_prompt, max_new_tokens, stream_callback)
                reduced.append(result)
                self._free_kv_cache()

            current = reduced

            # Check if result fits in budget now
            combined_tokens = self.count_tokens("\n\n".join(current))
            if combined_tokens <= context_budget - 100:
                break

        return "\n\n".join(f"## Summary of Part {i+1}\n{s}" for i, s in enumerate(current))

    # ---- Main entry point ----

    def summarize(self, text, detail_level="concise", max_new_tokens=1024, stream_callback=None, chunking="auto"):
        """
        Summarize the text.

        Args:
            text: The transcript text.
            detail_level: "concise", "detailed", or "comprehensive".
            max_new_tokens: Max tokens to generate.
            stream_callback: Optional function(text_chunk) for real-time updates.
            chunking: "auto" (chunk if input exceeds safe threshold) or "always" (force chunking).
        """
        thread = None
        try:
            self.load_model()

            input_tokens = self.count_tokens(text)
            context_budget = self._get_context_budget(max_new_tokens)
            needs_chunking = (chunking == "always") or (chunking == "auto" and input_tokens > context_budget)

            if stream_callback:
                mode_str = "chunked" if needs_chunking else "single-pass"
                reason = ""
                if chunking == "always":
                    reason = " (forced)"
                elif needs_chunking:
                    reason = f" ({input_tokens} tokens > {context_budget} budget)"
                stream_callback(f"[{input_tokens} tokens, mode: {mode_str}{reason}]\n\n")

            print(f"Summarize: {input_tokens} tokens, budget: {context_budget}, chunking: {chunking}, needs_chunking: {needs_chunking}")

            if needs_chunking:
                return self.summarize_chunked(text, detail_level, max_new_tokens, stream_callback)

            # Single-pass summarization
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_user_prompt(text, detail_level)
            return self._summarize_single(system_prompt, user_prompt, max_new_tokens, stream_callback)

        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory Error caught in summarize()")
            # Wait for any generation thread to finish so it releases references
            if stream_callback and thread is not None:
                thread.join(timeout=5)
            self.unload_model()
            raise RuntimeError("GPU Out of Memory. Try a smaller model, enable quantization, or use chunked mode for long transcripts.")
        except Exception as e:
            print(f"Summarization error: {e}")
            if stream_callback and thread is not None:
                thread.join(timeout=5)
            self.unload_model()
            raise
