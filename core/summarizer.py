import torch
from transformers import pipeline, TextIteratorStreamer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import gc
import os
import threading
import math

class StopSignalCriteria(StoppingCriteria):
    def __init__(self, stop_event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()

class Summarizer:
    """Handles summarization and action point extraction using a local LLM."""

    # Overhead tokens for chat template (system + user role markers, special tokens)
    _TEMPLATE_OVERHEAD = 200
    # Overlap between chunks in tokens for context continuity
    _CHUNK_OVERLAP = 200
    # Fraction of free VRAM to use for KV cache (keep 15% headroom)
    _VRAM_USAGE_FRACTION = 0.85
    # Fallback budget when VRAM estimation isn't possible (CPU mode)
    _FALLBACK_CONTEXT_BUDGET = 16000

    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cuda", quantization="4"):
        self.model_id = model_id
        self.device = device
        self.quantization = quantization  # "4", "8", or "none"
        self.pipe = None
        self.chunk_summaries = None  # Populated during chunked summarization
        self.stop_event = threading.Event()

    def stop(self):
        """Signal the summarizer to stop processing."""
        self.stop_event.set()

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

    def _estimate_kv_bytes_per_token(self):
        """Estimate KV cache memory per token from model config."""
        config = self.pipe.model.config

        num_layers = getattr(config, 'num_hidden_layers', 24)
        # GQA models have fewer KV heads than attention heads
        num_kv_heads = getattr(config, 'num_key_value_heads',
                               getattr(config, 'num_attention_heads', 16))
        hidden_size = getattr(config, 'hidden_size', 1024)
        num_heads = getattr(config, 'num_attention_heads', 16)
        head_dim = hidden_size // num_heads

        # KV cache: 2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes
        # KV cache is typically stored in float16 (2 bytes) regardless of weight quantization
        dtype_bytes = 2
        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        return bytes_per_token

    def _estimate_activation_bytes_per_token(self):
        """Estimate peak activation memory per token during the forward pass.

        During prefill, all input tokens are processed at once. The peak memory
        per token comes from the FFN intermediate buffers (gate + up projections
        for gated/SwiGLU architectures) which are the largest transient tensors.
        """
        config = self.pipe.model.config
        hidden_size = getattr(config, 'hidden_size', 1024)
        intermediate_size = getattr(config, 'intermediate_size', 4 * hidden_size)
        dtype_bytes = 2  # float16

        # Peak FFN activation: gate_proj and up_proj outputs alive simultaneously
        # Plus attention Q projection and residual buffers (~hidden_size)
        bytes_per_token = (2 * intermediate_size + hidden_size) * dtype_bytes
        return bytes_per_token

    def _estimate_fixed_overhead(self):
        """Estimate fixed VRAM overhead that is NOT proportional to sequence length.

        For quantized models (4/8-bit), bitsandbytes dequantizes weight matrices
        to fp16 on the fly during computation. The peak occurs in the FFN where
        gate_proj and up_proj may both be dequantized simultaneously. This cost
        is per-layer (not accumulated) but is a large fixed allocation.
        Also accounts for CUDA workspace, cuBLAS handles, and other runtime overhead.
        """
        config = self.pipe.model.config
        hidden_size = getattr(config, 'hidden_size', 1024)
        intermediate_size = getattr(config, 'intermediate_size', 4 * hidden_size)
        dtype_bytes = 2  # fp16

        overhead = 0

        # Dequantization buffers: peak is 2 FFN projections (gate + up) at fp16
        if self.quantization in ("4", "8"):
            overhead += 2 * intermediate_size * hidden_size * dtype_bytes

        # CUDA context, cuBLAS workspace, kernel launch buffers, allocator fragmentation
        overhead += 64 * 1024 * 1024  # 64 MiB flat

        return overhead

    def _get_context_budget(self, max_new_tokens):
        """Return safe input token budget based on VRAM and model context window."""
        config = self.pipe.model.config
        model_max = getattr(config, 'max_position_embeddings', 32768)

        # Maximum tokens the model supports for input
        model_budget = model_max - max_new_tokens - self._TEMPLATE_OVERHEAD - 512

        # On GPU: estimate how many tokens fit in free VRAM
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                free_vram, _ = torch.cuda.mem_get_info()
                fixed_overhead = self._estimate_fixed_overhead()
                usable_vram = free_vram * self._VRAM_USAGE_FRACTION - fixed_overhead
                usable_vram = max(usable_vram, 0)
                kv_bytes_per_token = self._estimate_kv_bytes_per_token()
                activation_bytes_per_token = self._estimate_activation_bytes_per_token()
                # KV cache persists across all layers; activation is peak per-layer
                # but spans all tokens during prefill. Both scale linearly with seq_len.
                total_bytes_per_token = kv_bytes_per_token + activation_bytes_per_token
                # Total sequence = input + generated tokens; both consume KV cache
                vram_budget = int(usable_vram / total_bytes_per_token) - max_new_tokens - self._TEMPLATE_OVERHEAD
                vram_budget = max(vram_budget, 1024)  # floor at 1K tokens

                budget = min(model_budget, vram_budget)
                print(f"Context budget: free_vram={free_vram/1e6:.0f}MB, "
                      f"fixed_overhead={fixed_overhead/1e6:.0f}MB, "
                      f"usable_vram={usable_vram/1e6:.0f}MB, "
                      f"kv={kv_bytes_per_token}B/tok, "
                      f"activation={activation_bytes_per_token}B/tok, "
                      f"total={total_bytes_per_token}B/tok, "
                      f"vram_budget={vram_budget}, model_budget={model_budget}, "
                      f"final={budget}")
                return budget
            except Exception as e:
                print(f"VRAM estimation failed ({e}), using model budget")
                return min(model_budget, self._FALLBACK_CONTEXT_BUDGET)

        # CPU: no VRAM constraint, just respect model context window
        return model_budget

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
                    "Please provide:\n1. A detailed summary of the discussion, covering key points discussed"
                    "and arguments, as well as possible problems or difficulties in a few paragraphs.\n"
                    "2. A detailed list of action points (tasks, decisions, or follow-ups) with assigned owners if mentioned.")
        elif detail_level == "comprehensive":
            return (f"Here is the meeting transcript:\n\n{text}\n\n"
                    "Please provide:\n1. A comprehensive summary of the full discussion, including context, key decisions, "
                    "and nuances, as well as possible problems or difficulties in a few paragraphs.\n"
                    "2. A detailed list of action points (tasks, decisions, or follow-ups).\n"
                    "3. Any open questions or unresolved issues.\n"
                    "4. A list of key questions asked during the meeting along with their proposed answers if any.")
        else:  # concise
            return (f"Here is the meeting transcript:\n\n{text}\n\n"
                    "Please provide:\n1. A concise summary of the discussion in a few paragraphs.\n"
                    "2. A list of action points (tasks, decisions, or follow-ups).")

    def _summarize_single(self, system_prompt, user_prompt, max_new_tokens, stream_callback=None):
        """Run a single generation pass. Returns the generated text."""
        if self.stop_event.is_set():
            raise RuntimeError("Summarization stopped by user")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        stopping_criteria = StoppingCriteriaList([StopSignalCriteria(self.stop_event)])

        gen_params = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.05,
            return_full_text=False,
            stopping_criteria=stopping_criteria,
        )

        if stream_callback:
            streamer = TextIteratorStreamer(self.pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = dict(text_inputs=messages, streamer=streamer, **gen_params)

            # Capture exceptions from the generation thread so they propagate
            # to the main thread — TextIteratorStreamer does not do this.
            thread_error = [None]

            def _generate():
                try:
                    self.pipe(**generation_kwargs)
                except BaseException as e:
                    thread_error[0] = e
                    # Unblock the streamer iterator by sending the stop signal
                    streamer.text_queue.put(streamer.stop_signal)

            thread = threading.Thread(target=_generate)
            thread.start()

            generated_text = ""
            for new_text in streamer:
                if self.stop_event.is_set():
                    break
                generated_text += new_text
                stream_callback(new_text)

            thread.join()

            if thread_error[0] is not None:
                raise thread_error[0]
            
            if self.stop_event.is_set():
                raise RuntimeError("Summarization stopped by user")

            return generated_text
        else:
            outputs = self.pipe(messages, **gen_params)
            if self.stop_event.is_set():
                raise RuntimeError("Summarization stopped by user")
            return outputs[0]["generated_text"]

    def _free_kv_cache(self):
        """Free KV cache VRAM between generation passes."""
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_reduce_prompt(self, combined_summaries, num_parts, detail_level):
        """Build the reduce-phase prompt that merges chunk summaries."""
        prompt = (
            f"Below are summaries of {num_parts} consecutive parts of a single meeting transcript. "
            "Combine them into one unified, coherent summary.\n\n"
            f"{combined_summaries}\n\n"
            "Please provide:\n"
        )

        if detail_level == "detailed":
            prompt += (
                "1. A detailed summary of the full discussion, covering key points discussed "
                "and arguments, as well as possible problems or difficulties.\n"
                "2. A comprehensive list of action points (tasks, decisions, or follow-ups) with assigned owners if mentioned."
            )
        elif detail_level == "comprehensive":
            prompt += (
                "1. A comprehensive summary of the full discussion, including context, key decisions, "
                "and nuances, as well as possible problems or difficulties.\n"
                "2. A detailed list of action points (tasks, decisions, or follow-ups).\n"
                "3. Any open questions or unresolved issues.\nBe as detailed and thorough as possible."
            )
        else:
            prompt += (
                "1. A concise summary of the full discussion.\n"
                "2. A list of action points (tasks, decisions, or follow-ups)."
            )

        return prompt

    # ---- Chunked summarization (map-reduce) ----

    def summarize_chunked(self, text, detail_level="concise", max_new_tokens=None, stream_callback=None, num_chunks=None):
        """Map-reduce chunked summarization for long transcripts."""
        if max_new_tokens is None:
            max_new_tokens = self._MAX_NEW_TOKENS.get(detail_level, 2048)
        self.load_model()

        context_budget = self._get_context_budget(max_new_tokens)
        input_tokens = self.count_tokens(text)
        system_prompt = self._get_system_prompt()

        # Determine chunk size: context budget minus the overhead of the user prompt template
        # (the "Here is the meeting transcript..." wrapper is ~50 tokens)
        chunk_content_budget = context_budget - 60

        if num_chunks is not None:
             # Account for overlap: each chunk boundary adds overlap tokens,
             # so effective new content per chunk is (chunk_size - overlap).
             # To produce exactly num_chunks: chunk_size = (T + (N-1)*O) / N
             overlap = self._CHUNK_OVERLAP
             tokens_per_chunk = math.ceil((input_tokens + (num_chunks - 1) * overlap) / num_chunks)
             
             # When user forces num_chunks, we respect it up to the model's hard limit,
             # ignoring the VRAM estimation which might be too conservative.
             config = self.pipe.model.config
             model_max = getattr(config, 'max_position_embeddings', 32768)
             model_hard_limit = model_max - max_new_tokens - self._TEMPLATE_OVERHEAD - 512 - 60

             # We use the smaller of the calculated size or the hard model limit
             chunk_size = min(tokens_per_chunk, model_hard_limit)
             chunks = self._split_text_into_chunks(text, chunk_size)
        else:
             chunks = self._split_text_into_chunks(text, chunk_content_budget)
             # Add one extra chunk as safety margin — VRAM estimation is inherently
             # imprecise (fragmentation, runtime buffers, driver overhead), so
             # slightly over-chunking is cheaper than hitting OOM.
             if len(chunks) > 1:
                 target_chunks = len(chunks) + 1
                 overlap = self._CHUNK_OVERLAP
                 safer_size = math.ceil((input_tokens + (target_chunks - 1) * overlap) / target_chunks)
                 safer_size = min(safer_size, chunk_content_budget)
                 chunks = self._split_text_into_chunks(text, safer_size)

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
            if self.stop_event.is_set():
                raise RuntimeError("Summarization stopped by user")

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

        # Store chunk summaries for export
        self.chunk_summaries = list(chunk_summaries)

        # --- REDUCE phase: combine chunk summaries into final summary ---
        combined_summaries = "\n\n".join(
            f"## Summary of Part {i+1}\n{s}" for i, s in enumerate(chunk_summaries)
        )

        # Build the full reduce prompt so we can check total token count
        reduce_prompt = self._build_reduce_prompt(combined_summaries, num_chunks, detail_level)

        # Check FULL prompt tokens (system + user + template overhead) against budget
        system_tokens = self.count_tokens(system_prompt)
        total_input_tokens = system_tokens + self.count_tokens(reduce_prompt) + self._TEMPLATE_OVERHEAD
        print(f"Reduce phase: total_input_tokens={total_input_tokens}, context_budget={context_budget}")

        if total_input_tokens > context_budget:
            # Compute how many tokens the summaries can occupy
            summaries_budget = context_budget - system_tokens - self._TEMPLATE_OVERHEAD - 200  # 200 for reduce instructions
            combined_summaries = self._hierarchical_reduce(
                chunk_summaries, system_prompt, max_new_tokens, summaries_budget, stream_callback
            )
            reduce_prompt = self._build_reduce_prompt(combined_summaries, num_chunks, detail_level)

        if stream_callback:
            stream_callback("\n\n--- Generating final summary ---\n\n")

        self._free_kv_cache()
        return self._summarize_single(system_prompt, reduce_prompt, max_new_tokens, stream_callback)

    def _hierarchical_reduce(self, summaries, system_prompt, max_new_tokens, context_budget, stream_callback):
        """Reduce summaries in pairs when they exceed the context budget."""
        current = list(summaries)
        iteration = 0
        while len(current) > 1:
            if self.stop_event.is_set():
                raise RuntimeError("Summarization stopped by user")

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

    # Max output tokens per detail level
    _MAX_NEW_TOKENS = {
        "concise": 1024,
        "detailed": 2048,
        "comprehensive": 4096,
    }

    def summarize(self, text, detail_level="concise", max_new_tokens=None, stream_callback=None, chunking="auto", num_chunks=None):
        """
        Summarize the text.

        Args:
            text: The transcript text.
            detail_level: "concise", "detailed", or "comprehensive".
            max_new_tokens: Max tokens to generate. If None, scales with detail_level.
            stream_callback: Optional function(text_chunk) for real-time updates.
            chunking: "auto" (chunk if input exceeds safe threshold), "always" (force chunking), or "never".
            num_chunks: Optional number of chunks to split the text into (if chunking is "always").
        """
        if max_new_tokens is None:
            max_new_tokens = self._MAX_NEW_TOKENS.get(detail_level, 2048)
        self.chunk_summaries = None
        self.stop_event.clear()
        try:
            self.load_model()

            input_tokens = self.count_tokens(text)
            context_budget = self._get_context_budget(max_new_tokens)
            
            if chunking == "never":
                needs_chunking = False
            else:
                needs_chunking = (chunking == "always") or (chunking == "auto" and input_tokens > context_budget)

            if stream_callback:
                mode_str = "chunked" if needs_chunking else "single-pass"
                reason = ""
                if chunking == "always":
                    reason = " (forced)"
                elif chunking == "never":
                    reason = " (forced single-pass)"
                elif needs_chunking:
                    reason = f" ({input_tokens} tokens > {context_budget} budget)"
                stream_callback(f"[{input_tokens} tokens, mode: {mode_str}{reason}]\n\n")

            print(f"Summarize: {input_tokens} tokens, budget: {context_budget}, chunking: {chunking}, needs_chunking: {needs_chunking}")

            if needs_chunking:
                return self.summarize_chunked(text, detail_level, max_new_tokens, stream_callback, num_chunks=num_chunks if chunking == "always" else None)

            # Single-pass summarization — with OOM fallback to chunked mode
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_user_prompt(text, detail_level)
            try:
                return self._summarize_single(system_prompt, user_prompt, max_new_tokens, stream_callback)
            except torch.cuda.OutOfMemoryError:
                if chunking == "never":
                    print("Single-pass OOM — chunking disabled, re-raising error")
                    raise RuntimeError("GPU Out of Memory. Try enabling chunking or using a smaller model.")

                print("Single-pass OOM — falling back to chunked mode")
                self._free_kv_cache()
                if stream_callback:
                    stream_callback("\n\n[Single-pass ran out of GPU memory, retrying with chunked mode...]\n\n")
                return self.summarize_chunked(text, detail_level, max_new_tokens, stream_callback)

        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory Error caught in summarize()")
            self.unload_model()
            raise RuntimeError("GPU Out of Memory even in chunked mode. Try a smaller model or enable quantization.")
        except Exception as e:
            print(f"Summarization error: {e}")
            self.unload_model()
            raise
