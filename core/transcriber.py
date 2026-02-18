from faster_whisper import WhisperModel
import torch
import gc
import wave
import numpy as np
import tempfile
import os
import time


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

        # Try loading with fallback: cuda/float16 -> cuda/int8 -> cpu/int8
        attempts = [(self.device, self.compute_type)]
        if self.device == "cuda" and self.compute_type == "float16":
            attempts.append(("cuda", "int8"))
        if self.device == "cuda":
            attempts.append(("cpu", "int8"))

        for device, compute_type in attempts:
            try:
                print(f"Loading Whisper model: {model_size} (device={device}, compute={compute_type})...")
                self.model = WhisperModel(
                    model_size,
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
                          beam_size=5, vad_filter=False, stereo_mode="joint"):
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
                    beam_size=beam_size, vad_filter=vad_filter
                )
            else:
                if on_progress:
                    on_progress({'type': 'status', 'status': 'transcribing'})

                def seg_cb(seg):
                    if on_progress:
                        on_progress({'type': 'segment', **seg})

                print("Transcribing stereo file (joint)...")
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
                                     vad_filter=False):
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

        Returns:
            dict with transcription + speaker labels
        """
        if on_progress:
            on_progress({'type': 'status', 'status': 'transcribing'})

        def seg_cb(seg):
            if on_progress:
                on_progress({'type': 'segment', **seg})

        print("Transcribing...")
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
