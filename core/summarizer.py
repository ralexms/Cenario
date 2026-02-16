import torch
from transformers import pipeline, TextIteratorStreamer
import gc
import os
import threading

class Summarizer:
    """Handles summarization and action point extraction using a local LLM."""

    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cuda"):
        self.model_id = model_id
        self.device = device
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return

        print(f"Loading summarization model: {self.model_id}...")
        try:
            # Check if CUDA is available
            device_id = 0 if self.device == "cuda" and torch.cuda.is_available() else -1
            
            # Set PYTORCH_CUDA_ALLOC_CONF to avoid fragmentation
            if self.device == "cuda":
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            self.pipe = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.float16},
                device=device_id
            )
            print("Summarization model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Ensure cleanup on failure
            self.unload_model()
            raise

    def unload_model(self):
        """Free the model and release GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Summarization model unloaded, GPU memory freed")

    def summarize(self, text, detail_level="concise", max_new_tokens=1024, stream_callback=None):
        """
        Summarize the text.
        
        Args:
            text: The transcript text.
            detail_level: "concise", "detailed", or "comprehensive".
            max_new_tokens: Max tokens to generate.
            stream_callback: Optional function(text_chunk) for real-time updates.
        """
        try:
            self.load_model()

            system_prompt = "You are a helpful assistant that summarizes meeting transcripts. Your task is to analyze the provided transcript and generate a structured output containing a summary and a list of action points."
            
            if detail_level == "detailed":
                user_prompt = f"Here is the meeting transcript:\n\n{text}\n\nPlease provide:\n1. A detailed summary of the discussion, covering key points and arguments.\n2. A comprehensive list of action points (tasks, decisions, or follow-ups) with assigned owners if mentioned."
            elif detail_level == "comprehensive":
                user_prompt = f"Here is the meeting transcript:\n\n{text}\n\nPlease provide:\n1. A comprehensive summary of the discussion, including context, key decisions, and nuances.\n2. A detailed list of action points (tasks, decisions, or follow-ups).\n3. Any open questions or unresolved issues."
            else: # concise
                user_prompt = f"Here is the meeting transcript:\n\n{text}\n\nPlease provide:\n1. A concise summary of the discussion.\n2. A list of action points (tasks, decisions, or follow-ups)."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            streamer = None
            if stream_callback:
                streamer = TextIteratorStreamer(self.pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                # Run generation in a separate thread so we can consume the streamer
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

        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory Error caught in summarize()")
            self.unload_model()
            raise RuntimeError("GPU Out of Memory. Try a smaller model or shorter transcript.")
        except Exception as e:
            print(f"Summarization error: {e}")
            self.unload_model()
            raise
