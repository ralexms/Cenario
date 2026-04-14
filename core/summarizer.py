import json
import math
import os
import threading
import gc
import re

import requests
import torch
from transformers import pipeline, TextIteratorStreamer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList


TRANSFORMERS_SUMMARIZATION_MODELS = [
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen 2.5 0.5B (Fast, Low VRAM)",
        "description": "Good for quick summaries on modest hardware.",
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen 2.5 1.5B (Balanced)",
        "description": "Balanced speed and summary quality.",
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "label": "Qwen 2.5 3B",
        "description": "Higher quality with higher VRAM cost.",
    },
    {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "label": "Llama 3.2 3B (Good quality)",
        "description": "Good quality for general meeting notes.",
    },
    {
        "id": "microsoft/Phi-3.5-mini-instruct",
        "label": "Phi 3.5 Mini 3.8B (Best quality)",
        "description": "Default built-in option.",
        "default": True,
    },
]


def get_transformers_summary_models():
    return [dict(model) for model in TRANSFORMERS_SUMMARIZATION_MODELS]


def _lmstudio_base_url():
    return os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234").rstrip("/")


def _lmstudio_headers():
    token = os.environ.get("LM_STUDIO_API_TOKEN", "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _format_bytes(size_bytes):
    if not isinstance(size_bytes, (int, float)) or size_bytes <= 0:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return None


def _lmstudio_context_length(model):
    loaded_instances = model.get("loaded_instances") or []
    if loaded_instances:
        config = loaded_instances[0].get("config") or {}
        if config.get("context_length"):
            return config["context_length"]
    return model.get("max_context_length")


def list_lmstudio_models(timeout=2):
    """Return locally available LLM models exposed by LM Studio."""
    url = f"{_lmstudio_base_url()}/api/v1/models"
    try:
        response = requests.get(url, headers=_lmstudio_headers(), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"LM Studio server is not reachable at {_lmstudio_base_url()}. "
            "Start the LM Studio Developer server first."
        ) from exc
    except ValueError as exc:
        raise RuntimeError("LM Studio returned an invalid model list response.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("LM Studio returned an unexpected model list payload.")

    models = []
    for model in payload.get("models", []):
        if model.get("type") != "llm":
            continue

        key = model.get("key")
        if not key:
            continue

        context_length = _lmstudio_context_length(model)
        quant = model.get("quantization") or {}
        quant_name = quant.get("name")
        params = model.get("params_string")
        display_name = model.get("display_name") or key
        label_bits = [display_name]
        if params:
            label_bits.append(params)
        if quant_name:
            label_bits.append(quant_name)

        models.append({
            "id": key,
            "label": " | ".join(label_bits),
            "description": model.get("description") or "",
            "publisher": model.get("publisher"),
            "architecture": model.get("architecture"),
            "format": model.get("format"),
            "params": params,
            "quantization": quant_name,
            "size_bytes": model.get("size_bytes"),
            "size_label": _format_bytes(model.get("size_bytes")),
            "context_length": context_length,
            "loaded": bool(model.get("loaded_instances")),
            "reasoning": ((model.get("capabilities") or {}).get("reasoning") or {}),
        })

    if not models:
        raise RuntimeError(
            "LM Studio is running but did not report any local LLM models."
        )

    models.sort(key=lambda item: (not item["loaded"], item["label"].lower()))
    if models:
        models[0]["default"] = True
    return models

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

    # Max output tokens per detail level (used for MAP phase and custom prompt path)
    _MAX_NEW_TOKENS = {
        "concise": 1024,
        "detailed": 2048,
        "comprehensive": 4096,
    }

    # Per-section output token limits for multi-section summarization
    _SECTION_TOKENS = {
        "concise":       {"summary": 768,  "action_points": 256},
        "detailed":      {"summary": 1536, "action_points": 512},
        "comprehensive": {"summary": 2048, "action_points": 512, "open_questions": 512, "qa": 1024},
    }

    # Ordered (section_key, markdown_heading) pairs per detail level
    _SECTIONS = {
        "concise": [
            ("summary", None),
            ("action_points", "## Action Points"),
        ],
        "detailed": [
            ("summary", None),
            ("action_points", "## Action Points"),
        ],
        "comprehensive": [
            ("summary", None),
            ("action_points", "## Action Points"),
            ("open_questions", "## Open Questions"),
            ("qa", "## Key Questions & Answers"),
        ],
    }

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
                    llm_int8_enable_fp32_cpu_offload=True,
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
            "Do not reproduce transcription errors in your summary."
        )

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

    # ---- Multi-section summarization ----

    def _build_section_prompt(self, content_prefix, section, detail_level):
        """Build a focused single-task prompt for one output section."""
        if section == "summary":
            if detail_level == "concise":
                instruction = (
                    "Write a concise summary of this meeting in 2-3 prose paragraphs. "
                    "Do not use bullet points or lists. Cover the main topics and outcomes."
                )
            elif detail_level == "detailed":
                instruction = (
                    "Write a detailed summary of this meeting in several prose paragraphs. "
                    "Do not use bullet points or lists. Cover all key topics, arguments, "
                    "decisions, problems raised, and important context."
                )
            else:  # comprehensive
                instruction = (
                    "Write a comprehensive summary of this meeting in several detailed prose paragraphs. "
                    "Do not use bullet points or lists. Cover all topics, full context, "
                    "key decisions and their rationale, and any difficulties raised."
                )
        elif section == "action_points":
            instruction = (
                "List every concrete task, decision, or follow-up from this meeting as a bulleted list. "
                "Include the assigned owner next to each item if mentioned. "
                "If there are none, don't write any assigned person or institution."
            )
        elif section == "open_questions":
            instruction = (
                "List any unresolved issues or open questions from this meeting as a bulleted list."
            )
        elif section == "qa":
            instruction = (
                "List the key questions raised during this meeting along with their answers as a bulleted list. "
                "Format each item as: '- Q: [question] / A: [answer]'."
            )
        else:
            raise ValueError(f"Unknown section: {section}")

        return content_prefix + instruction

    def _summarize_sections(self, content_prefix, detail_level, stream_callback):
        """Run a separate focused generation pass for each output section and combine."""
        system_prompt = self._get_system_prompt()
        parts = []

        for section, heading in self._SECTIONS[detail_level]:
            if self.stop_event.is_set():
                raise RuntimeError("Summarization stopped by user")

            max_tok = self._SECTION_TOKENS[detail_level][section]
            prompt = self._build_section_prompt(content_prefix, section, detail_level)

            if heading and stream_callback:
                stream_callback(f"\n\n{heading}\n")

            text = self._summarize_single(system_prompt, prompt, max_tok, stream_callback)
            parts.append(f"{heading}\n{text.strip()}" if heading else text.strip())
            self._free_kv_cache()

        return "\n\n".join(parts)

    # ---- Chunked summarization (map-reduce) ----

    def summarize_chunked(self, text, detail_level="concise", max_new_tokens=None, stream_callback=None, num_chunks=None, custom_prompt=None):
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
            content_prefix = f"Here is the meeting transcript:\n\n{chunks[0]}\n\n"
            if custom_prompt:
                return self._summarize_single(system_prompt, content_prefix + custom_prompt, max_new_tokens, stream_callback)
            return self._summarize_sections(content_prefix, detail_level, stream_callback)

        # --- MAP phase: summarize each chunk with separate per-section prompts ---
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if self.stop_event.is_set():
                raise RuntimeError("Summarization stopped by user")

            if stream_callback:
                header = f"\n--- Summarizing chunk {i+1}/{num_chunks} ---\n\n"
                stream_callback(header)

            content_prefix = f"Here is part {i+1} of {num_chunks} of a meeting transcript:\n\n{chunk}\n\n"

            try:
                chunk_result = self._summarize_sections(content_prefix, detail_level, stream_callback)
                chunk_summaries.append(chunk_result)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM during chunk {i+1}/{num_chunks}")
                raise RuntimeError(
                    f"GPU Out of Memory while summarizing chunk {i+1}/{num_chunks}. "
                    "Try a smaller model or enable quantization."
                )

        # Store chunk summaries for export
        self.chunk_summaries = list(chunk_summaries)

        # --- REDUCE phase ---
        combined_summaries = "\n\n".join(
            f"## Summary of Part {i+1}\n{s}" for i, s in enumerate(chunk_summaries)
        )

        reduce_prefix = (
            f"Below are summaries of {num_chunks} consecutive parts of a single meeting transcript. "
            f"Read all of them carefully.\n\n{combined_summaries}\n\n"
        )

        # Check if combined summaries fit within the context budget
        system_tokens = self.count_tokens(system_prompt)
        total_input_tokens = system_tokens + self.count_tokens(reduce_prefix) + self._TEMPLATE_OVERHEAD + 100
        print(f"Reduce phase: total_input_tokens≈{total_input_tokens}, context_budget={context_budget}")

        if total_input_tokens > context_budget:
            # Compress summaries until they fit
            summaries_budget = context_budget - system_tokens - self._TEMPLATE_OVERHEAD - 300
            combined_summaries = self._hierarchical_reduce(
                chunk_summaries, system_prompt, max_new_tokens, summaries_budget, stream_callback
            )
            reduce_prefix = (
                f"Below are summaries of {num_chunks} consecutive parts of a single meeting transcript. "
                f"Read all of them carefully.\n\n{combined_summaries}\n\n"
            )

        if stream_callback:
            stream_callback("\n\n--- Generating final summary ---\n\n")

        self._free_kv_cache()

        if custom_prompt:
            return self._summarize_single(system_prompt, reduce_prefix + custom_prompt, max_new_tokens, stream_callback)
        return self._summarize_sections(reduce_prefix, detail_level, stream_callback)

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

    def summarize(self, text, detail_level="concise", max_new_tokens=None, stream_callback=None, chunking="auto", num_chunks=None, custom_prompt=None):
        """
        Summarize the text.

        Args:
            text: The transcript text.
            detail_level: "concise", "detailed", or "comprehensive".
            max_new_tokens: Max tokens to generate. If None, scales with detail_level.
            stream_callback: Optional function(text_chunk) for real-time updates.
            chunking: "auto" (chunk if input exceeds safe threshold), "always" (force chunking), or "never".
            num_chunks: Optional number of chunks to split the text into (if chunking is "always").
            custom_prompt: Optional custom prompt to override default instructions.
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
                return self.summarize_chunked(text, detail_level, max_new_tokens, stream_callback,
                                              num_chunks=num_chunks if chunking == "always" else None,
                                              custom_prompt=custom_prompt)

            # Single-pass summarization — with OOM fallback to chunked mode
            system_prompt = self._get_system_prompt()
            content_prefix = f"Here is the meeting transcript:\n\n{text}\n\n"
            try:
                if custom_prompt:
                    return self._summarize_single(system_prompt, content_prefix + custom_prompt, max_new_tokens, stream_callback)
                return self._summarize_sections(content_prefix, detail_level, stream_callback)
            except torch.cuda.OutOfMemoryError:
                if chunking == "never":
                    print("Single-pass OOM — chunking disabled, re-raising error")
                    raise RuntimeError("GPU Out of Memory. Try enabling chunking or using a smaller model.")

                print("Single-pass OOM — falling back to chunked mode")
                self._free_kv_cache()
                if stream_callback:
                    stream_callback("\n\n[Single-pass ran out of GPU memory, retrying with chunked mode...]\n\n")
                return self.summarize_chunked(text, detail_level, max_new_tokens, stream_callback,
                                              custom_prompt=custom_prompt)

        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory Error caught in summarize()")
            self.unload_model()
            raise RuntimeError("GPU Out of Memory even in chunked mode. Try a smaller model or enable quantization.")
        except Exception as e:
            print(f"Summarization error: {e}")
            self.unload_model()
            raise


class LMStudioSummarizer(Summarizer):
    """Summarization backend that delegates generation to a local LM Studio server."""

    _REQUEST_TIMEOUT = (10, 600)

    def __init__(self, model_id, device="cpu", quantization="none"):
        super().__init__(model_id=model_id, device=device, quantization=quantization)
        self.model_info = None
        self.context_length = 8192
        self._active_response = None

    def stop(self):
        super().stop()
        if self._active_response is not None:
            try:
                self._active_response.close()
            except Exception:
                pass

    def load_model(self):
        if self.model_info is not None:
            return

        print(f"Loading LM Studio model metadata: {self.model_id}...")
        models = list_lmstudio_models(timeout=5)
        for model in models:
            if model["id"] == self.model_id:
                self.model_info = model
                self.context_length = model.get("context_length") or 8192
                print("LM Studio model metadata loaded.")
                return

        raise RuntimeError(
            f"LM Studio model not found: {self.model_id}. "
            "Refresh the model list after loading it in LM Studio."
        )

    def unload_model(self):
        self._active_response = None
        self.model_info = None

    def count_tokens(self, text):
        """Approximate token count conservatively when tokenizer access is remote."""
        if not text:
            return 0
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return 0
        return max(1, math.ceil(len(normalized) / 3.2))

    def _get_context_budget(self, max_new_tokens):
        model_budget = self.context_length - max_new_tokens - self._TEMPLATE_OVERHEAD - 512
        return max(1024, model_budget)

    def _free_kv_cache(self):
        return

    def _build_request_body(self, system_prompt, user_prompt, max_new_tokens, stream):
        body = {
            "model": self.model_id,
            "system_prompt": system_prompt,
            "input": user_prompt,
            "stream": stream,
            "temperature": 0.6,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "max_output_tokens": max_new_tokens,
            "context_length": self.context_length,
        }

        reasoning = (self.model_info or {}).get("reasoning") or {}
        allowed_reasoning = reasoning.get("allowed_options") or []
        if "off" in allowed_reasoning:
            body["reasoning"] = "off"

        return body

    def _extract_stream_text(self, event_name, payload):
        if not isinstance(payload, dict):
            return ""

        for key in ("content", "text", "delta"):
            value = payload.get(key)
            if isinstance(value, str):
                return value

        if event_name == "message":
            return payload.get("content", "") or ""

        if event_name == "message.delta":
            delta = payload.get("delta")
            if isinstance(delta, dict):
                return delta.get("content", "") or ""

        return ""

    def _stream_chat(self, body, stream_callback):
        generated_text = ""
        event_name = None
        url = f"{_lmstudio_base_url()}/api/v1/chat"

        try:
            with requests.post(
                url,
                headers=_lmstudio_headers(),
                json=body,
                stream=True,
                timeout=self._REQUEST_TIMEOUT,
            ) as response:
                response.raise_for_status()
                self._active_response = response
                for raw_line in response.iter_lines(decode_unicode=True):
                    if self.stop_event.is_set():
                        raise RuntimeError("Summarization stopped by user")
                    if raw_line is None:
                        continue

                    line = raw_line.strip()
                    if not line:
                        event_name = None
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue

                    payload = json.loads(data)
                    text = self._extract_stream_text(event_name, payload)
                    if not text:
                        continue

                    generated_text += text
                    stream_callback(text)
        finally:
            self._active_response = None

        if self.stop_event.is_set():
            raise RuntimeError("Summarization stopped by user")
        return generated_text

    def _summarize_single(self, system_prompt, user_prompt, max_new_tokens, stream_callback=None):
        if self.stop_event.is_set():
            raise RuntimeError("Summarization stopped by user")

        body = self._build_request_body(system_prompt, user_prompt, max_new_tokens, bool(stream_callback))

        try:
            if stream_callback:
                return self._stream_chat(body, stream_callback)

            response = requests.post(
                f"{_lmstudio_base_url()}/api/v1/chat",
                headers=_lmstudio_headers(),
                json=body,
                timeout=self._REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            self._active_response = None
            raise RuntimeError(
                f"LM Studio request failed. Check that the local server is running at {_lmstudio_base_url()}."
            ) from exc
        except ValueError as exc:
            self._active_response = None
            raise RuntimeError("LM Studio returned an invalid inference response.") from exc
        finally:
            self._active_response = None

        if self.stop_event.is_set():
            raise RuntimeError("Summarization stopped by user")

        for key in ("text", "content", "output"):
            value = payload.get(key)
            if isinstance(value, str):
                return value

        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        raise RuntimeError("LM Studio returned a response without generated text.")
