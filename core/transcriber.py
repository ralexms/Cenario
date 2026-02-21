from faster_whisper import WhisperModel
import torch
import gc
import wave
import numpy as np
import tempfile
import os
import time
import difflib
import re
import shutil


# Directory for caching CTranslate2-converted models
_CT2_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cenario", "ct2-models")


def _ensure_ct2_model(model_id):
    """If model_id is a HuggingFace repo that isn't in CTranslate2 format, convert it.

    Standard faster-whisper model names (tiny, base, etc.) are returned as-is.
    HuggingFace repo IDs (containing '/') are checked for a cached CTranslate2
    conversion.  If none exists, the model is downloaded and converted using
    ctranslate2's TransformersConverter.

    Returns the model_size_or_path string to pass to WhisperModel().
    """
    # Standard model names — handled natively by faster-whisper
    if "/" not in model_id:
        return model_id

    # Build a stable cache path from the repo id
    safe_name = model_id.replace("/", "--")
    ct2_dir = os.path.join(_CT2_CACHE_DIR, safe_name)
    model_bin = os.path.join(ct2_dir, "model.bin")

    # Already converted?
    if os.path.isfile(model_bin) and os.path.getsize(model_bin) > 0:
        print(f"Using cached CTranslate2 model: {ct2_dir}")
        return ct2_dir

    # Clean up partial/failed conversion
    if os.path.exists(ct2_dir):
        print(f"Cleaning up incomplete cache: {ct2_dir}")
        shutil.rmtree(ct2_dir)

    # Convert from HuggingFace transformers format
    print(f"Converting '{model_id}' to CTranslate2 format (first-time only)...")
    from ctranslate2.converters import TransformersConverter

    class _CompatConverter(TransformersConverter):
        """Work around dtype/torch_dtype mismatch between ctranslate2 and transformers."""
        def load_model(self, model_class, model_name_or_path, **kwargs):
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            return model_class.from_pretrained(model_name_or_path, **kwargs)

    os.makedirs(ct2_dir, exist_ok=True)

    # Only copy files that actually exist in the repo (some fine-tuned
    # models ship without tokenizer.json)
    from huggingface_hub import list_repo_files
    repo_files = set(list_repo_files(model_id))
    want_copy = ["tokenizer.json", "preprocessor_config.json"]
    copy_files = [f for f in want_copy if f in repo_files]

    converter = _CompatConverter(model_id, copy_files=copy_files)
    converter.convert(ct2_dir, quantization="float16", force=True)

    # If tokenizer.json was missing, generate it from the model's tokenizer
    if not os.path.isfile(os.path.join(ct2_dir, "tokenizer.json")):
        print("Generating tokenizer.json from model tokenizer...")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.save_pretrained(ct2_dir)

    print(f"Conversion complete — saved to {ct2_dir}")
    return ct2_dir


def _split_audio_chunks(audio_file, chunk_minutes, overlap_seconds=10):
    """Split a WAV file into overlapping chunks as temp WAV files.

    Returns list of (file_path, chunk_start_sec, chunk_end_sec) tuples.
    If the file is shorter than one chunk, returns [(audio_file, 0.0, duration)]
    with no temp files created.
    """
    with wave.open(audio_file, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        all_frames = wf.readframes(n_frames)

    duration = n_frames / framerate
    chunk_sec = chunk_minutes * 60

    if duration <= chunk_sec:
        return [(audio_file, 0.0, duration)]

    chunks = []
    pos = 0.0
    while pos < duration:
        end = min(pos + chunk_sec, duration)
        start_frame = int(pos * framerate)
        end_frame = int(end * framerate)
        bytes_per_frame = n_channels * sampwidth
        chunk_bytes = all_frames[start_frame * bytes_per_frame:end_frame * bytes_per_frame]

        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(chunk_bytes)

        chunks.append((tmp_path, pos, end))

        # Next chunk starts overlap_seconds before where this one ended
        next_pos = end - overlap_seconds
        if next_pos <= pos:
            break
        pos = next_pos

    return chunks


def _deduplicate_overlap(prev_segments, new_segments, overlap_start, overlap_end):
    """Filter new_segments to remove duplicates that fall in the overlap zone.

    Two-pass check: timestamp proximity (midpoints within 2s) then text
    similarity (SequenceMatcher ratio > 0.8). Segments starting after
    overlap_end are always kept.
    """
    if not prev_segments or not new_segments:
        return new_segments

    kept = []
    for seg in new_segments:
        mid = (seg['start'] + seg['end']) / 2
        # Segments clearly past the overlap zone — always keep
        if mid > overlap_end:
            kept.append(seg)
            continue

        # Check against previous segments for duplicates
        is_dup = False
        for prev in prev_segments:
            prev_mid = (prev['start'] + prev['end']) / 2
            if abs(mid - prev_mid) < 2.0:
                ratio = difflib.SequenceMatcher(
                    None, prev['text'].strip(), seg['text'].strip()
                ).ratio()
                if ratio > 0.8:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(seg)

    return kept


class Transcriber:
    """Handles speech-to-text transcription using faster-whisper"""

    def __init__(self, device=None, compute_type=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.current_model_size = None

    def load_model(self, model_size="small"):
        if self.current_model_size == model_size and self.model is not None:
            print(f"Model '{model_size}' already loaded")
            return

        # Free previous model first to avoid holding two on GPU
        if self.model is not None:
            self.unload_model()

        # Convert HuggingFace models to CTranslate2 format if needed
        model_path = _ensure_ct2_model(model_size)

        # Try loading with fallback: chosen compute -> cuda/int8 -> cpu/int8
        attempts = [(self.device, self.compute_type)]
        if self.device == "cuda" and self.compute_type != "int8":
            attempts.append(("cuda", "int8"))
        if self.device == "cuda":
            attempts.append(("cpu", "int8"))

        for device, compute_type in attempts:
            try:
                print(f"Loading Whisper model: {model_size} (device={device}, compute={compute_type})...")
                self.model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type
                )
                self.device = device
                self.compute_type = compute_type
                self.current_model_size = model_size
                print(f"Model '{model_size}' loaded on {device} with {compute_type}")
                return
            except ValueError as e:
                print(f"Failed ({device}/{compute_type}): {e}")
                continue

        raise RuntimeError(f"Could not load Whisper model '{model_size}' with any device/compute combination")

    def unload_model(self):
        """Free the Whisper model and release GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.current_model_size = None

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("Model unloaded, GPU memory freed")

    def transcribe(self, audio_file, model_size="small", language=None,
                   on_segment=None, beam_size=5, vad_filter=False):
        self.load_model(model_size)

        print(f"Transcribing {audio_file}...")
        kwargs = {'beam_size': beam_size, 'vad_filter': vad_filter}
        if language:
            kwargs['language'] = language
        segments, info = self.model.transcribe(audio_file, **kwargs)

        result = {
            'language': info.language,
            'segments': []
        }

        for segment in segments:
            seg = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            }
            result['segments'].append(seg)
            if on_segment:
                on_segment(seg)

        return result

    def transcribe_chunked(self, audio_file, model_size="small", language=None,
                           on_segment=None, beam_size=5, vad_filter=False,
                           chunk_minutes=10, overlap_seconds=10,
                           on_progress=None):
        """Transcribe a long audio file in chunks to avoid OOM.

        Splits audio into overlapping segments, transcribes each independently
        (with model reload between chunks to reset CTranslate2 memory pool),
        and merges results with deduplication.
        """
        chunks = _split_audio_chunks(audio_file, chunk_minutes, overlap_seconds)

        # Short file — fall through to regular transcribe
        if len(chunks) == 1 and chunks[0][0] == audio_file:
            return self.transcribe(audio_file, model_size, language=language,
                                   on_segment=on_segment, beam_size=beam_size,
                                   vad_filter=vad_filter)

        total_chunks = len(chunks)
        all_segments = []
        detected_language = language  # pin after chunk 0
        temp_files = [c[0] for c in chunks]

        try:
            for idx, (chunk_file, chunk_start, chunk_end) in enumerate(chunks):
                print(f"Transcribing chunk {idx + 1}/{total_chunks} "
                      f"({chunk_start:.0f}s - {chunk_end:.0f}s)...")

                if on_progress:
                    start_m = int(chunk_start) // 60
                    start_s = int(chunk_start) % 60
                    end_m = int(chunk_end) // 60
                    end_s = int(chunk_end) % 60
                    on_progress({
                        'type': 'chunk_progress',
                        'chunk': idx + 1,
                        'total': total_chunks,
                        'start': chunk_start,
                        'end': chunk_end,
                        'label': (f"Transcribing chunk {idx + 1}/{total_chunks} "
                                  f"({start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d})")
                    })

                # Unload model between chunks to reset CTranslate2 memory pool
                if idx > 0:
                    self.unload_model()

                # Collect segments for this chunk with adjusted timestamps.
                # For the first chunk all segments are emitted immediately.
                # For later chunks, segments clearly past the overlap zone are
                # emitted immediately; overlap-zone segments are held for
                # deduplication before emission.
                chunk_segments = []
                _is_first_chunk = (idx == 0)
                _overlap_end = chunk_start + overlap_seconds

                def _chunk_cb(seg, _offset=chunk_start,
                              _first=_is_first_chunk,
                              _ov_end=_overlap_end):
                    adjusted = {
                        'start': seg['start'] + _offset,
                        'end': seg['end'] + _offset,
                        'text': seg['text']
                    }
                    chunk_segments.append(adjusted)
                    # Emit in real-time when safe to do so (no dedup needed)
                    if on_segment:
                        mid = (adjusted['start'] + adjusted['end']) / 2
                        if _first or mid > _ov_end:
                            on_segment(adjusted)

                result = self.transcribe(chunk_file, model_size,
                                         language=detected_language,
                                         on_segment=_chunk_cb,
                                         beam_size=beam_size,
                                         vad_filter=vad_filter)

                # Pin language from first chunk
                if idx == 0:
                    detected_language = result.get('language', detected_language)

                if idx == 0:
                    # All segments already emitted in real-time
                    all_segments.extend(chunk_segments)
                else:
                    # Deduplicate overlap zone; non-overlap segments were
                    # already emitted in real-time so only emit survivors
                    # that fall inside the overlap zone here.
                    overlap_end = chunk_start + overlap_seconds
                    survived = _deduplicate_overlap(
                        all_segments, chunk_segments, chunk_start, overlap_end
                    )
                    for s in survived:
                        all_segments.append(s)
                        s_mid = (s['start'] + s['end']) / 2
                        if on_segment and s_mid <= overlap_end:
                            on_segment(s)

        finally:
            # Clean up temp files (skip original audio_file)
            for f in temp_files:
                if f != audio_file:
                    try:
                        os.unlink(f)
                    except OSError:
                        pass

        return {
            'language': detected_language or '',
            'segments': all_segments
        }

    def _diarize(self, audio_file, hf_token=None, speaker_prefix="",
                 on_progress=None):
        """Run diarization on an audio file and return the result.

        Args:
            audio_file: Path to audio file
            hf_token: HuggingFace token (required first time)
            speaker_prefix: Prefix for speaker labels (e.g. "REMOTE_")
            on_progress: Callback for diarization progress events
        """
        from pyannote.audio import Pipeline

        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)

        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )

        if self.device == "cuda":
            diarization_pipeline.to(torch.device("cuda"))
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            diarization_pipeline.to(torch.device("mps"))

        # Build progress hook if callback provided
        hook = None
        if on_progress:
            try:
                from pyannote.audio import Inference
                from pyannote.audio.pipelines.utils import PipelineModel

                def _hook(step_name, step_artefact, file=None,
                          total=None, completed=None):
                    if completed is not None and total is not None and total > 0:
                        on_progress({
                            'type': 'diarize_progress',
                            'step': step_name,
                            'completed': completed,
                            'total': total,
                            'prefix': speaker_prefix,
                        })

                hook = _hook
            except Exception:
                pass  # graceful fallback if hook API not available

        diarization = diarization_pipeline(audio_file, hook=hook)
        return diarization, speaker_prefix

    def _assign_speakers(self, transcription, diarization, speaker_prefix=""):
        """Assign speaker labels to transcription segments."""
        for segment in transcription['segments']:
            segment_mid = (segment['start'] + segment['end']) / 2

            speaker = None
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                if turn.start <= segment_mid <= turn.end:
                    speaker = speaker_prefix + speaker_label
                    break

            segment['speaker'] = speaker if speaker else 'UNKNOWN'

    def transcribe_stereo(self, audio_file, model_size="small", hf_token=None,
                          language=None, on_progress=None,
                          beam_size=5, vad_filter=False, stereo_mode="joint",
                          chunk_minutes=0):
        """
        Transcribe stereo file by splitting channels, with optional diarization.

        Args:
            audio_file: Path to stereo WAV file
            model_size: Whisper model to use
            hf_token: HuggingFace token for diarization (omit to skip diarization)
            language: Force language code (None for auto-detect)
            on_progress: Callback for progress events
            beam_size: Beam size for decoding (higher = better but slower)
            vad_filter: Enable voice activity detection filtering
            stereo_mode: "joint" (default) or "separate"

        Returns:
            dict with transcription results
        """
        if stereo_mode == "joint":
            if hf_token:
                return self.transcribe_with_diarization(
                    audio_file, model_size=model_size, hf_token=hf_token,
                    language=language, on_progress=on_progress,
                    beam_size=beam_size, vad_filter=vad_filter,
                    chunk_minutes=chunk_minutes
                )
            else:
                if on_progress:
                    on_progress({'type': 'status', 'status': 'transcribing'})

                def seg_cb(seg):
                    if on_progress:
                        on_progress({'type': 'segment', **seg})

                print("Transcribing stereo file (joint)...")
                if chunk_minutes > 0:
                    def _progress_cb(evt):
                        if on_progress:
                            on_progress(evt)
                    result = self.transcribe_chunked(
                        audio_file, model_size, language=language,
                        on_segment=seg_cb, beam_size=beam_size,
                        vad_filter=vad_filter, chunk_minutes=chunk_minutes,
                        on_progress=_progress_cb)
                else:
                    result = self.transcribe(audio_file, model_size, language=language,
                                             on_segment=seg_cb, beam_size=beam_size,
                                             vad_filter=vad_filter)
                
                if on_progress:
                    on_progress({'type': 'transcription_done', 'result': result})
                
                return result

        import wave
        import numpy as np
        import os

        self.load_model(model_size)

        with wave.open(audio_file, 'rb') as wav:
            if wav.getnchannels() != 2:
                raise ValueError("Audio file must be stereo")
            frames = wav.readframes(wav.getnframes())
            sample_rate = wav.getframerate()

        stereo_data = np.frombuffer(frames, dtype=np.int16)
        left = stereo_data[0::2]
        right = stereo_data[1::2]

        temp_left = "temp_left.wav"
        temp_right = "temp_right.wav"

        for data, filepath in [(left, temp_left), (right, temp_right)]:
            with wave.open(filepath, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(data.tobytes())

        if on_progress:
            on_progress({'type': 'status', 'status': 'transcribing',
                         'detail': 'left channel (meeting audio)'})

        def left_cb(seg):
            if on_progress:
                on_progress({'type': 'segment', 'channel': 'left', **seg})

        print("Transcribing left channel (meeting audio)...")
        if chunk_minutes > 0:
            result_left = self.transcribe_chunked(
                temp_left, model_size, language=language,
                on_segment=left_cb, beam_size=beam_size,
                vad_filter=vad_filter, chunk_minutes=chunk_minutes,
                on_progress=on_progress)
        else:
            result_left = self.transcribe(temp_left, model_size, language=language,
                                          on_segment=left_cb, beam_size=beam_size,
                                          vad_filter=vad_filter)

        if on_progress:
            on_progress({'type': 'status', 'status': 'transcribing',
                         'detail': 'right channel (microphone)'})

        def right_cb(seg):
            if on_progress:
                on_progress({'type': 'segment', 'channel': 'right', **seg})

        print("Transcribing right channel (your microphone)...")
        if chunk_minutes > 0:
            result_right = self.transcribe_chunked(
                temp_right, model_size, language=language,
                on_segment=right_cb, beam_size=beam_size,
                vad_filter=vad_filter, chunk_minutes=chunk_minutes,
                on_progress=on_progress)
        else:
            result_right = self.transcribe(temp_right, model_size, language=language,
                                           on_segment=right_cb, beam_size=beam_size,
                                           vad_filter=vad_filter)

        result = {'left_channel': result_left, 'right_channel': result_right}

        # Signal transcription complete before diarization
        if on_progress:
            on_progress({'type': 'transcription_done', 'result': result})

        # Diarize if token provided — wrapped in try/except so transcription is not lost
        if hf_token:
            # Free Whisper model to make room for diarization on GPU
            self.unload_model()

            if on_progress:
                on_progress({'type': 'status', 'status': 'diarizing'})

            def diarize_cb(event):
                if on_progress:
                    on_progress(event)

            try:
                print("Identifying speakers on left channel (meeting audio)...")
                diarization_left, _ = self._diarize(temp_left, hf_token,
                                                    on_progress=diarize_cb)
                self._assign_speakers(result_left, diarization_left,
                                      speaker_prefix="REMOTE_")

                print("Identifying speakers on right channel (microphone)...")
                diarization_right, _ = self._diarize(temp_right, hf_token,
                                                     speaker_prefix="LOCAL_",
                                                     on_progress=diarize_cb)
                self._assign_speakers(result_right, diarization_right,
                                      speaker_prefix="LOCAL_")
            except Exception as e:
                print(f"Diarization failed: {e}")
                if on_progress:
                    on_progress({'type': 'warning',
                                 'message': f'Diarization failed: {e}. Transcription saved without speaker labels.'})

        os.remove(temp_left)
        os.remove(temp_right)

        return result

    def transcribe_live(self, capture, model_size="small", chunk_seconds=5,
                        stop_event=None, on_segment=None, language=None):
        """
        Live transcription from an active AudioCapture session.

        Args:
            capture: AudioCapture instance with active recording
            model_size: Whisper model size
            chunk_seconds: Seconds of audio per chunk
            stop_event: threading.Event to signal stop
            on_segment: callback(text, start, end) for each segment
            language: Force language code (None for auto-detect)
        """
        self.load_model(model_size)
        elapsed = 0.0

        while not stop_event.is_set():
            stop_event.wait(timeout=chunk_seconds)
            if stop_event.is_set():
                break

            chunk = capture.get_live_chunk()
            if chunk is None or len(chunk) < capture.sample_rate:
                continue

            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                with wave.open(tmp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(capture.sample_rate)
                    wf.writeframes(chunk.tobytes())

                kwargs = {'beam_size': 3}
                if language:
                    kwargs['language'] = language
                segments, _ = self.model.transcribe(tmp_path, **kwargs)
                for seg in segments:
                    start = elapsed + seg.start
                    end = elapsed + seg.end
                    text = seg.text.strip()
                    if text and on_segment:
                        on_segment(text, start, end)
            except Exception as e:
                print(f"Live transcription error: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            elapsed += len(chunk) / capture.sample_rate

    def transcribe_with_diarization(self, audio_file, model_size="small",
                                     hf_token=None, language=None,
                                     on_progress=None, beam_size=5,
                                     vad_filter=False, chunk_minutes=0):
        """
        Transcribe with speaker diarization.

        Args:
            audio_file: Path to audio file
            model_size: Whisper model size
            hf_token: HuggingFace token (required first time)
            language: Force language code (None for auto-detect)
            on_progress: Callback for progress events
            beam_size: Beam size for decoding (higher = better but slower)
            vad_filter: Enable voice activity detection filtering
            chunk_minutes: When > 0, use chunked transcription

        Returns:
            dict with transcription + speaker labels
        """
        if on_progress:
            on_progress({'type': 'status', 'status': 'transcribing'})

        def seg_cb(seg):
            if on_progress:
                on_progress({'type': 'segment', **seg})

        print("Transcribing...")
        if chunk_minutes > 0:
            transcription = self.transcribe_chunked(
                audio_file, model_size, language=language,
                on_segment=seg_cb, beam_size=beam_size,
                vad_filter=vad_filter, chunk_minutes=chunk_minutes,
                on_progress=on_progress)
        else:
            transcription = self.transcribe(audio_file, model_size, language=language,
                                            on_segment=seg_cb, beam_size=beam_size,
                                            vad_filter=vad_filter)

        # Signal transcription complete before diarization
        if on_progress:
            on_progress({'type': 'transcription_done', 'result': transcription})

        # Free Whisper model to make room for diarization on GPU
        self.unload_model()

        # Diarize — wrapped in try/except so transcription is not lost
        if on_progress:
            on_progress({'type': 'status', 'status': 'diarizing'})

        def diarize_cb(event):
            if on_progress:
                on_progress(event)

        try:
            print("Identifying speakers...")
            diarization, _ = self._diarize(audio_file, hf_token,
                                           on_progress=diarize_cb)
            self._assign_speakers(transcription, diarization)
            print("Finished!")
        except Exception as e:
            print(f"Diarization failed: {e}")
            if on_progress:
                on_progress({'type': 'warning',
                             'message': f'Diarization failed: {e}. Transcription saved without speaker labels.'})

        return transcription
