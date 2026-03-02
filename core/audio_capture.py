# core/audio_capture.py

import platform
import subprocess
import wave
import json
import threading
import numpy as np

PLATFORM = platform.system()  # 'Linux', 'Windows', 'Darwin'

if PLATFORM == 'Windows':
    import sounddevice as sd
    try:
        import pyaudiowpatch as pyaudio
        _HAS_PYAUDIOWPATCH = True
    except ImportError:
        _HAS_PYAUDIOWPATCH = False
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from comtypes import CLSCTX_ALL
        from ctypes import cast, POINTER
        _HAS_PYCAW = True
    except ImportError:
        _HAS_PYCAW = False
elif PLATFORM == 'Darwin':
    import sounddevice as sd


class _LoopbackCapture:
    """Thread-based WASAPI loopback capture via pyaudiowpatch.

    Uses blocking reads in a daemon thread instead of PortAudio callbacks
    to avoid native segfaults when closing a stream while audio is active.
    Exposes start/stop/close/active matching the sounddevice API so the
    rest of AudioCapture can treat it like any other stream.
    """

    def __init__(self, pa_instance, device_idx, target_rate, buffer_list, lock):
        dev = pa_instance.get_device_info_by_index(device_idx)
        self._channels = dev['maxInputChannels']
        self._native_rate = int(dev['defaultSampleRate'])
        self._target_rate = target_rate
        self._buffer_list = buffer_list
        self._lock = lock
        self._stop_event = threading.Event()
        self._thread = None
        self.active = False

        self._stream = pa_instance.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._native_rate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=1024,
            start=False,
        )

    def start(self):
        self.active = True
        self._stop_event.clear()
        self._stream.start_stream()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.active:
            return
        self._stop_event.set()
        
        # Wait for the reader thread to exit.
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
            
        # We do NOT call self._stream.stop_stream() here.
        # Calling stop_stream() on a WASAPI loopback stream that might still
        # have pending data or be in a delicate state can cause crashes.
        # We rely on close() to implicitly stop the stream.
            
        self.active = False

    def close(self):
        self.stop()
        try:
            self._stream.close()
        except Exception:
            pass

    def _reader(self):
        chunk = 1024
        channels = self._channels
        native_rate = self._native_rate
        target_rate = self._target_rate
        import time

        # Initialize timing for silence insertion
        start_time = time.time()
        total_samples_out = 0
        
        # Threshold for silence insertion (e.g., 100ms)
        # If we are behind by more than this, we insert silence.
        # This handles the case where WASAPI loopback provides no data (silence).
        silence_threshold = int(0.1 * target_rate)

        while not self._stop_event.is_set():
            # 1. Read available data
            try:
                avail = self._stream.get_read_available()
            except Exception:
                avail = 0
            
            # If data is available, read it
            if avail > 0:
                # Read up to a reasonable amount to avoid massive allocations if buffer got full
                read_frames = min(avail, 4096)
                try:
                    data = self._stream.read(read_frames, exception_on_overflow=False)
                    
                    arr = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
                    mono = _downmix_to_mono_int16(arr)

                    # Resample if needed
                    if native_rate != target_rate and len(mono) > 0:
                        n_out = int(len(mono) * target_rate / native_rate)
                        if n_out > 0:
                            mono = np.interp(
                                np.linspace(0, len(mono) - 1, n_out),
                                np.arange(len(mono)),
                                mono.astype(np.float64),
                            ).astype(np.int16)
                    
                    if len(mono) > 0:
                        with self._lock:
                            self._buffer_list.append(mono.tobytes())
                        total_samples_out += len(mono)
                        
                except Exception:
                    pass
            
            # 2. Check synchronization with wall clock
            current_time = time.time()
            elapsed = current_time - start_time
            expected_samples = int(elapsed * target_rate)
            
            deficit = expected_samples - total_samples_out
            
            if deficit > silence_threshold:
                # We are falling behind (likely due to silence in loopback).
                # Insert silence to catch up.
                # We fill up to the expected amount.
                silence_samples = deficit
                silence_bytes = b'\x00' * (silence_samples * 2)
                
                with self._lock:
                    self._buffer_list.append(silence_bytes)
                
                total_samples_out += silence_samples
            
            # Sleep briefly to avoid busy loop
            time.sleep(0.005)


def _downmix_to_mono_int16(indata):
    """Downmix multi-channel int16 input to mono, avoiding overflow."""
    if indata.shape[1] == 1:
        return indata[:, 0]
    return indata.astype(np.int32).mean(axis=1).astype(np.int16)


class AudioCapture:
    """Handles audio recording — PulseAudio on Linux, WASAPI/sounddevice on Windows"""

    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording_process = None
        self._recording_mode = None  # 'mono' or 'stereo'
        # Buffered recording state
        self._buffer1 = []
        self._buffer2 = []
        self._buffer1_lock = threading.Lock()
        self._buffer2_lock = threading.Lock()
        self._live_read_pos1 = 0
        self._live_read_pos2 = 0
        self._reader_threads = []
        self._stop_readers = threading.Event()
        # Windows: sounddevice stream references
        self._stream1 = None
        self._stream2 = None
        self._pyaudio = None  # pyaudiowpatch instance (loopback only)

    # ------------------------------------------------------------------ #
    #  Source listing                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_sources():
        """List all available audio sources (platform-aware)."""
        if PLATFORM == 'Windows':
            return AudioCapture._list_sources_windows()
        if PLATFORM == 'Darwin':
            return AudioCapture._list_sources_mac()
        return AudioCapture._list_sources_linux()

    @staticmethod
    def _list_sources_linux():
        """List PulseAudio sources."""
        try:
            result = subprocess.run(
                ['pactl', '-f', 'json', 'list', 'sources'],
                capture_output=True,
                text=True,
                check=True
            )
            sources = json.loads(result.stdout)
            return sources
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Error listing sources: {e}")
            return []

    @staticmethod
    def _list_sources_mac():
        """List audio input devices on macOS via sounddevice.

        Regular microphones get the prefix 'input_<idx>'.
        Virtual loopback devices (BlackHole, Soundflower, etc.) get 'loopback_<idx>'.
        To capture system audio you need a virtual audio driver such as
        BlackHole (https://github.com/ExistentialAudio/BlackHole).
        """
        devices = sd.query_devices()
        default_input_name = None
        try:
            default_input_name = sd.query_devices(kind='input')['name']
        except Exception:
            pass

        _LOOPBACK_KEYWORDS = ('blackhole', 'loopback', 'soundflower', 'virtual')
        sources = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                name = d['name']
                state = 'RUNNING' if name == default_input_name else 'IDLE'
                is_loopback = any(kw in name.lower() for kw in _LOOPBACK_KEYWORDS)
                prefix = 'loopback' if is_loopback else 'input'
                sources.append({
                    'name': f"{prefix}_{i}",
                    'description': name,
                    'state': state,
                    'index': i,
                    'default_samplerate': d['default_samplerate'],
                })
        return sources

    @staticmethod
    def _list_sources_windows():
        """List audio devices via sounddevice (inputs) and pyaudiowpatch (loopback)."""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        # Find WASAPI host API index
        wasapi_idx = None
        for i, api in enumerate(hostapis):
            if 'WASAPI' in api['name']:
                wasapi_idx = i
                break

        # Determine default device names so we can mark them RUNNING
        default_input_name = None
        default_output_name = None
        try:
            default_input_name = sd.query_devices(kind='input')['name']
        except Exception:
            pass
        try:
            default_output_name = sd.query_devices(kind='output')['name']
        except Exception:
            pass

        sources = []

        # Input devices (microphones) — via sounddevice
        for i, d in enumerate(devices):
            if wasapi_idx is not None and d['hostapi'] != wasapi_idx:
                continue
            if d['max_input_channels'] > 0:
                state = 'RUNNING' if d['name'] == default_input_name else 'IDLE'
                sources.append({
                    'name': f"input_{i}",
                    'description': d['name'],
                    'state': state,
                    'index': i,
                    'default_samplerate': d['default_samplerate'],
                })

        # Loopback devices — via pyaudiowpatch
        if _HAS_PYAUDIOWPATCH:
            sources += AudioCapture._list_loopback_pyaudio(default_output_name)

        return sources

    @staticmethod
    def _list_loopback_pyaudio(default_output_name):
        """Discover WASAPI loopback devices via pyaudiowpatch."""
        sources = []
        p = pyaudio.PyAudio()
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            p.terminate()
            return sources

        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if not dev.get('isLoopbackDevice', False):
                continue
            name = dev['name']
            state = 'RUNNING' if name == default_output_name else 'IDLE'
            
            # Clean up description
            desc = name
            if '[Loopback]' in desc:
                desc = desc.replace('[Loopback]', '').strip()
            if not desc.endswith('(Loopback)'):
                desc = f"{desc} (Loopback)"
                
            sources.append({
                'name': f"loopback_{i}",
                'description': desc,
                'state': state,
                'index': i,
                'default_samplerate': dev['defaultSampleRate'],
                # Store native config for stream opening
                '_loopback_channels': dev['maxInputChannels'],
            })
        p.terminate()
        return sources

    # ------------------------------------------------------------------ #
    #  Windows helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_windows_source(source_name):
        """
        Parse a Windows source name into (device_index, is_loopback).
        E.g. 'loopback_5' -> (5, True), 'input_3' -> (3, False).
        """
        if source_name.startswith('loopback_'):
            return int(source_name.split('_', 1)[1]), True
        elif source_name.startswith('input_'):
            return int(source_name.split('_', 1)[1]), False
        else:
            raise ValueError(f"Unknown Windows source format: {source_name}")

    @staticmethod
    def _wasapi_extra(is_loopback):
        """Return WasapiSettings for WASAPI devices with auto sample rate conversion."""
        return sd.WasapiSettings(exclusive=False, auto_convert=True)

    @staticmethod
    def _downmix_to_mono_int16(indata):
        """Downmix multi-channel int16 input to mono, avoiding overflow."""
        return _downmix_to_mono_int16(indata)

    def _make_sd_callback(self, buffer_list, lock):
        """Create a sounddevice callback that appends mono int16 bytes to a buffer."""
        def callback(indata, frames, time_info, status):
            mono = AudioCapture._downmix_to_mono_int16(indata)
            with lock:
                buffer_list.append(mono.tobytes())
        return callback

    def _open_sd_stream(self, device_idx, is_loopback, buffer_list, lock):
        """Open and return a sounddevice InputStream (mic input only)."""
        stream = sd.InputStream(
            device=device_idx,
            channels=1,
            samplerate=self.sample_rate,
            dtype='int16',
            callback=self._make_sd_callback(buffer_list, lock),
            extra_settings=self._wasapi_extra(is_loopback),
        )
        return stream

    def _open_loopback_stream(self, device_idx, buffer_list, lock):
        """Open a WASAPI loopback stream via pyaudiowpatch.

        Returns a _LoopbackCapture wrapper that uses thread-based blocking
        reads instead of PortAudio callbacks (avoids native crash on close).
        """
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()
        return _LoopbackCapture(
            self._pyaudio, device_idx, self.sample_rate, buffer_list, lock,
        )

    @staticmethod
    def _safe_close_stream(stream):
        """Stop and close a stream (sounddevice or pyaudio), ignoring errors."""
        if stream is None:
            return
        # sounddevice streams have .active; pyaudio streams have .is_active()
        try:
            is_active = stream.active if hasattr(stream, 'active') else stream.is_active()
            if is_active:
                if hasattr(stream, 'stop_stream'):
                    stream.stop_stream()  # pyaudio
                else:
                    stream.stop()  # sounddevice
        except Exception as e:
            print(f"Warning: stream stop failed: {e}")
        try:
            stream.close()
        except Exception as e:
            print(f"Warning: stream close failed: {e}")

    @staticmethod
    def _start_stream(stream):
        """Start a stream (sounddevice or pyaudio)."""
        if hasattr(stream, 'start_stream'):
            stream.start_stream()  # pyaudio
        else:
            stream.start()  # sounddevice

    def _cleanup_pyaudio(self):
        """Terminate the PyAudio instance if no streams are active."""
        if self._pyaudio is not None and self._stream1 is None and self._stream2 is None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None

    @staticmethod
    def _mute_outputs():
        """Mute default output device. Returns state for _restore_outputs."""
        if not _HAS_PYCAW:
            return []
        saved = []
        try:
            speakers = AudioUtilities.GetSpeakers()
            interface = speakers.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            was_muted = volume.GetMute()
            if not was_muted:
                volume.SetMute(1, None)
            saved.append((volume, was_muted))
        except Exception:
            pass
        return saved

    @staticmethod
    def _restore_outputs(saved):
        """Restore mute state saved by _mute_outputs."""
        for volume, was_muted in saved:
            try:
                if not was_muted:
                    volume.SetMute(0, None)
            except Exception:
                pass

    def _close_loopbacks_with_mute(self, streams):
        """Close WASAPI loopback streams, muting outputs first to prevent crash.

        When audio is actively playing through speakers, closing a WASAPI
        loopback stream can cause a PortAudio segfault.  Briefly muting the
        output drains the loopback buffer so the close is safe.
        """
        import time
        # Stop reader threads (but leave underlying PA streams open)
        for s in streams:
            s.stop()
        saved_mute = self._mute_outputs()
        try:
            time.sleep(1)
            for s in streams:
                try:
                    s._stream.close()
                except Exception:
                    pass
        finally:
            self._restore_outputs(saved_mute)

    # ------------------------------------------------------------------ #
    #  Mac (CoreAudio / sounddevice) helpers                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_mac_source(source_name):
        """Parse a Mac source name into (device_index, is_loopback).

        'loopback_5' -> (5, True),  'input_3' -> (3, False).
        """
        if source_name.startswith('loopback_'):
            return int(source_name.split('_', 1)[1]), True
        elif source_name.startswith('input_'):
            return int(source_name.split('_', 1)[1]), False
        else:
            raise ValueError(f"Unknown Mac source format: {source_name}")

    def _open_mac_stream(self, device_idx, buffer_list, lock):
        """Open a sounddevice InputStream for macOS (no WASAPI settings)."""
        stream = sd.InputStream(
            device=device_idx,
            channels=1,
            samplerate=self.sample_rate,
            dtype='int16',
            callback=self._make_sd_callback(buffer_list, lock),
        )
        return stream

    def _record_mac(self, source_name, duration, output_file):
        try:
            device_idx, _ = self._parse_mac_source(source_name)
            chunks = []
            lock = threading.Lock()

            stream = self._open_mac_stream(device_idx, chunks, lock)
            stream.start()

            import time
            time.sleep(duration)

            stream.stop()
            stream.close()

            audio_data = b''.join(chunks)
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)

            return True
        except Exception as e:
            print(f"Recording error: {e}")
            return False

    def _record_stereo_mac(self, source1_name, source2_name, duration, output_file):
        try:
            idx1, _ = self._parse_mac_source(source1_name)
            idx2, _ = self._parse_mac_source(source2_name)

            buf1, buf2 = [], []
            lock1, lock2 = threading.Lock(), threading.Lock()

            stream1 = self._open_mac_stream(idx1, buf1, lock1)
            stream2 = self._open_mac_stream(idx2, buf2, lock2)
            stream1.start()
            stream2.start()

            import time
            time.sleep(duration)

            stream1.stop(); stream1.close()
            stream2.stop(); stream2.close()

            channel1 = np.frombuffer(b''.join(buf1), dtype=np.int16)
            channel2 = np.frombuffer(b''.join(buf2), dtype=np.int16)

            min_len = min(len(channel1), len(channel2))
            stereo = np.empty((min_len * 2,), dtype=np.int16)
            stereo[0::2] = channel1[:min_len]
            stereo[1::2] = channel2[:min_len]

            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(stereo.tobytes())

            return True
        except Exception as e:
            print(f"Stereo recording error: {e}")
            return False

    def _start_mono_mac(self, source_name, output_file):
        try:
            device_idx, _ = self._parse_mac_source(source_name)

            self._buffer1 = []
            self._live_read_pos1 = 0
            self._stream1 = self._open_mac_stream(device_idx, self._buffer1, self._buffer1_lock)
            self._stream2 = None
            self.process1 = None
            self.process2 = None
            self.output_file = output_file
            self.recording_process = True
            self._recording_mode = 'mono'
            self._reader_threads = []

            self._stream1.start()
            print(f"Recording started (mono) -> {output_file}")
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def _stop_mono_mac(self):
        if self._stream1 is None:
            print("No recording in progress")
            return False

        saved = False
        try:
            with self._buffer1_lock:
                audio_data = b''.join(self._buffer1)

            samples = np.frombuffer(audio_data, dtype=np.int16)
            with wave.open(self.output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(samples.tobytes())

            saved = True
            print("Recording stopped and saved")
        except Exception as e:
            print(f"Error saving recording: {e}")

        self._safe_close_stream(self._stream1)
        self._stream1 = None
        self._stream2 = None
        self.recording_process = None
        self._recording_mode = None
        self._buffer1 = []
        return saved

    def _start_stereo_mac(self, source1_name, source2_name, output_file):
        try:
            idx1, _ = self._parse_mac_source(source1_name)
            idx2, _ = self._parse_mac_source(source2_name)

            self._buffer1 = []
            self._buffer2 = []
            self._live_read_pos1 = 0
            self._live_read_pos2 = 0

            self._stream1 = self._open_mac_stream(idx1, self._buffer1, self._buffer1_lock)
            self._stream2 = self._open_mac_stream(idx2, self._buffer2, self._buffer2_lock)
            self.process1 = None
            self.process2 = None
            self.output_file = output_file
            self.recording_process = True
            self._recording_mode = 'stereo'
            self._reader_threads = []

            self._stream1.start()
            self._stream2.start()
            print(f"Recording started -> {output_file}")
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def _stop_stereo_mac(self):
        if self._stream1 is None:
            print("No recording in progress")
            return False

        saved = False
        try:
            with self._buffer1_lock:
                audio_data1 = b''.join(self._buffer1)
            with self._buffer2_lock:
                audio_data2 = b''.join(self._buffer2)

            channel1 = np.frombuffer(audio_data1, dtype=np.int16)
            channel2 = np.frombuffer(audio_data2, dtype=np.int16)

            if len(channel1) == 0 and len(channel2) > 0:
                channel1 = np.zeros(len(channel2), dtype=np.int16)
            elif len(channel2) == 0 and len(channel1) > 0:
                channel2 = np.zeros(len(channel1), dtype=np.int16)

            min_len = min(len(channel1), len(channel2))
            stereo = np.empty((min_len * 2,), dtype=np.int16)
            stereo[0::2] = channel1[:min_len]
            stereo[1::2] = channel2[:min_len]

            with wave.open(self.output_file, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(stereo.tobytes())

            saved = True
            print("Recording stopped and saved")
        except Exception as e:
            print(f"Error saving recording: {e}")

        self._safe_close_stream(self._stream1)
        self._safe_close_stream(self._stream2)
        self._stream1 = None
        self._stream2 = None
        self.recording_process = None
        self._recording_mode = None
        self._buffer1 = []
        self._buffer2 = []
        return saved

    # ------------------------------------------------------------------ #
    #  Blocking recording (fixed duration)                                 #
    # ------------------------------------------------------------------ #

    def record(self, source_name, duration, output_file):
        """Record audio from a source to WAV file (blocking)."""
        if PLATFORM == 'Windows':
            return self._record_windows(source_name, duration, output_file)
        if PLATFORM == 'Darwin':
            return self._record_mac(source_name, duration, output_file)
        return self._record_linux(source_name, duration, output_file)

    def _record_linux(self, source_name, duration, output_file):
        cmd = [
            'parec',
            '--device', source_name,
            '--channels', str(self.channels),
            '--rate', str(self.sample_rate),
            '--format', 's16le'
        ]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            import time
            time.sleep(duration)
            process.terminate()

            audio_data, _ = process.communicate()

            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)

            return True
        except Exception as e:
            print(f"Recording error: {e}")
            return False

    def _record_windows(self, source_name, duration, output_file):
        try:
            device_idx, is_loopback = self._parse_windows_source(source_name)
            chunks = []
            lock = threading.Lock()

            if is_loopback and _HAS_PYAUDIOWPATCH:
                stream = self._open_loopback_stream(device_idx, chunks, lock)
            else:
                stream = self._open_sd_stream(
                    device_idx, is_loopback, chunks, lock
                )
            self._start_stream(stream)

            import time
            time.sleep(duration)

            self._safe_close_stream(stream)
            self._cleanup_pyaudio()

            audio_data = b''.join(chunks)
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)

            return True
        except Exception as e:
            print(f"Recording error: {e}")
            return False

    def record_stereo(self, source1_name, source2_name, duration, output_file):
        """
        Record from two sources simultaneously and combine into stereo WAV.

        Args:
            source1_name: First audio source (e.g., meeting audio) -> Left channel
            source2_name: Second audio source (e.g., microphone) -> Right channel
            duration: Recording duration in seconds
            output_file: Output WAV file path

        Returns:
            True if successful, False otherwise
        """
        if PLATFORM == 'Windows':
            return self._record_stereo_windows(source1_name, source2_name, duration, output_file)
        if PLATFORM == 'Darwin':
            return self._record_stereo_mac(source1_name, source2_name, duration, output_file)
        return self._record_stereo_linux(source1_name, source2_name, duration, output_file)

    def _record_stereo_linux(self, source1_name, source2_name, duration, output_file):
        cmd1 = [
            'parec',
            '--device', source1_name,
            '--channels', '1',
            '--rate', str(self.sample_rate),
            '--format', 's16le'
        ]

        cmd2 = [
            'parec',
            '--device', source2_name,
            '--channels', '1',
            '--rate', str(self.sample_rate),
            '--format', 's16le'
        ]

        try:
            process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
            process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE)

            import time
            time.sleep(duration)

            process1.terminate()
            process2.terminate()

            audio_data1, _ = process1.communicate()
            audio_data2, _ = process2.communicate()

            channel1 = np.frombuffer(audio_data1, dtype=np.int16)
            channel2 = np.frombuffer(audio_data2, dtype=np.int16)

            min_len = min(len(channel1), len(channel2))
            channel1 = channel1[:min_len]
            channel2 = channel2[:min_len]

            stereo = np.empty((min_len * 2,), dtype=np.int16)
            stereo[0::2] = channel1
            stereo[1::2] = channel2

            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(stereo.tobytes())

            return True

        except Exception as e:
            print(f"Stereo recording error: {e}")
            return False

    def _record_stereo_windows(self, source1_name, source2_name, duration, output_file):
        try:
            idx1, lb1 = self._parse_windows_source(source1_name)
            idx2, lb2 = self._parse_windows_source(source2_name)

            buf1 = []
            buf2 = []
            lock1 = threading.Lock()
            lock2 = threading.Lock()

            if lb1 and _HAS_PYAUDIOWPATCH:
                stream1 = self._open_loopback_stream(idx1, buf1, lock1)
            else:
                stream1 = self._open_sd_stream(idx1, lb1, buf1, lock1)
            if lb2 and _HAS_PYAUDIOWPATCH:
                stream2 = self._open_loopback_stream(idx2, buf2, lock2)
            else:
                stream2 = self._open_sd_stream(idx2, lb2, buf2, lock2)

            self._start_stream(stream1)
            self._start_stream(stream2)

            import time
            time.sleep(duration)

            # Stop mic first, then loopback
            self._safe_close_stream(stream2)
            self._safe_close_stream(stream1)
            self._cleanup_pyaudio()

            with lock1:
                audio1 = b''.join(buf1)
            with lock2:
                audio2 = b''.join(buf2)

            channel1 = np.frombuffer(audio1, dtype=np.int16)
            channel2 = np.frombuffer(audio2, dtype=np.int16)

            min_len = min(len(channel1), len(channel2))
            channel1 = channel1[:min_len]
            channel2 = channel2[:min_len]

            stereo = np.empty((min_len * 2,), dtype=np.int16)
            stereo[0::2] = channel1
            stereo[1::2] = channel2

            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(stereo.tobytes())

            return True

        except Exception as e:
            print(f"Stereo recording error: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Linux: parec reader thread                                          #
    # ------------------------------------------------------------------ #

    def _reader_thread(self, process, buffer_list, lock, chunk_size=4096):
        """Read from a parec process stdout into a buffer list."""
        try:
            while not self._stop_readers.is_set():
                data = process.stdout.read(chunk_size)
                if not data:
                    break
                with lock:
                    buffer_list.append(data)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Non-blocking mono recording                                         #
    # ------------------------------------------------------------------ #

    def start_recording_mono(self, source_name, output_file):
        """
        Start mono recording from a single source in background (non-blocking).

        Args:
            source_name: Audio source (e.g., microphone)
            output_file: Output WAV file path

        Returns:
            True if started successfully, False otherwise
        """
        if self.recording_process is not None:
            print("Recording already in progress")
            return False

        if PLATFORM == 'Windows':
            return self._start_mono_windows(source_name, output_file)
        if PLATFORM == 'Darwin':
            return self._start_mono_mac(source_name, output_file)
        return self._start_mono_linux(source_name, output_file)

    def _start_mono_linux(self, source_name, output_file):
        cmd = [
            'parec',
            '--device', source_name,
            '--channels', '1',
            '--rate', str(self.sample_rate),
            '--format', 's16le'
        ]

        try:
            self._buffer1 = []
            self._live_read_pos1 = 0
            self._stop_readers.clear()

            self.process1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            self.process2 = None
            self.output_file = output_file
            self.recording_process = True
            self._recording_mode = 'mono'

            t1 = threading.Thread(
                target=self._reader_thread,
                args=(self.process1, self._buffer1, self._buffer1_lock),
                daemon=True,
            )
            t1.start()
            self._reader_threads = [t1]

            print(f"Recording started (mono) -> {output_file}")
            return True

        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def _start_mono_windows(self, source_name, output_file):
        try:
            device_idx, is_loopback = self._parse_windows_source(source_name)

            self._buffer1 = []
            self._live_read_pos1 = 0

            if is_loopback and _HAS_PYAUDIOWPATCH:
                self._stream1 = self._open_loopback_stream(
                    device_idx, self._buffer1, self._buffer1_lock
                )
            else:
                self._stream1 = self._open_sd_stream(
                    device_idx, is_loopback, self._buffer1, self._buffer1_lock
                )
            self._stream2 = None
            self.process1 = None
            self.process2 = None
            self.output_file = output_file
            self.recording_process = True
            self._recording_mode = 'mono'
            self._reader_threads = []

            self._start_stream(self._stream1)

            print(f"Recording started (mono) -> {output_file}")
            return True

        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def stop_recording_mono(self):
        """
        Stop mono background recording and save to file.

        Returns:
            True if stopped and saved successfully, False otherwise
        """
        if PLATFORM == 'Windows':
            return self._stop_mono_windows()
        if PLATFORM == 'Darwin':
            return self._stop_mono_mac()
        return self._stop_mono_linux()

    def _stop_mono_linux(self):
        if not hasattr(self, 'process1') or self.process1 is None:
            print("No recording in progress")
            return False

        try:
            self.process1.terminate()

            self._stop_readers.set()
            for t in self._reader_threads:
                t.join(timeout=5)

            with self._buffer1_lock:
                audio_data = b''.join(self._buffer1)

            samples = np.frombuffer(audio_data, dtype=np.int16)

            with wave.open(self.output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(samples.tobytes())

            # Cleanup
            self.process1 = None
            self.process2 = None
            self.recording_process = None
            self._recording_mode = None
            self._buffer1 = []
            self._reader_threads = []

            print(f"Recording stopped and saved")
            return True

        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False

    def _stop_mono_windows(self):
        if self._stream1 is None:
            print("No recording in progress")
            return False

        saved = False
        try:
            # Save buffer FIRST — before touching the stream — so audio is
            # persisted even if PortAudio crashes during stop/close.
            with self._buffer1_lock:
                audio_data = b''.join(self._buffer1)

            samples = np.frombuffer(audio_data, dtype=np.int16)

            with wave.open(self.output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(samples.tobytes())

            saved = True
            print("Recording stopped and saved")

        except Exception as e:
            print(f"Error saving recording: {e}")

        if isinstance(self._stream1, _LoopbackCapture):
            self._close_loopbacks_with_mute([self._stream1])
        else:
            self._safe_close_stream(self._stream1)

        # Unconditional cleanup
        self._stream1 = None
        self._stream2 = None
        self.recording_process = None
        self._recording_mode = None
        self._buffer1 = []
        self._cleanup_pyaudio()

        return saved

    # ------------------------------------------------------------------ #
    #  Non-blocking stereo recording                                       #
    # ------------------------------------------------------------------ #

    def start_recording_stereo(self, source1_name, source2_name, output_file):
        """
        Start stereo recording in background (non-blocking).
        Uses reader threads (Linux) or sounddevice callbacks (Windows)
        to accumulate audio buffers for live preview.

        Returns:
            True if started successfully, False otherwise
        """
        if self.recording_process is not None:
            print("Recording already in progress")
            return False

        if PLATFORM == 'Windows':
            return self._start_stereo_windows(source1_name, source2_name, output_file)
        if PLATFORM == 'Darwin':
            return self._start_stereo_mac(source1_name, source2_name, output_file)
        return self._start_stereo_linux(source1_name, source2_name, output_file)

    def _start_stereo_linux(self, source1_name, source2_name, output_file):
        cmd1 = [
            'parec',
            '--device', source1_name,
            '--channels', '1',
            '--rate', str(self.sample_rate),
            '--format', 's16le'
        ]

        cmd2 = [
            'parec',
            '--device', source2_name,
            '--channels', '1',
            '--rate', str(self.sample_rate),
            '--format', 's16le'
        ]

        try:
            self._buffer1 = []
            self._buffer2 = []
            self._live_read_pos1 = 0
            self._live_read_pos2 = 0
            self._stop_readers.clear()

            self.process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
            self.process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE)
            self.output_file = output_file
            self.recording_process = True
            self._recording_mode = 'stereo'

            t1 = threading.Thread(
                target=self._reader_thread,
                args=(self.process1, self._buffer1, self._buffer1_lock),
                daemon=True,
            )
            t2 = threading.Thread(
                target=self._reader_thread,
                args=(self.process2, self._buffer2, self._buffer2_lock),
                daemon=True,
            )
            t1.start()
            t2.start()
            self._reader_threads = [t1, t2]

            print(f"Recording started -> {output_file}")
            return True

        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def _start_stereo_windows(self, source1_name, source2_name, output_file):
        try:
            idx1, lb1 = self._parse_windows_source(source1_name)
            idx2, lb2 = self._parse_windows_source(source2_name)

            self._buffer1 = []
            self._buffer2 = []
            self._live_read_pos1 = 0
            self._live_read_pos2 = 0

            # Source 1 (monitor/loopback): use pyaudiowpatch if available
            if lb1 and _HAS_PYAUDIOWPATCH:
                self._stream1 = self._open_loopback_stream(
                    idx1, self._buffer1, self._buffer1_lock
                )
            else:
                self._stream1 = self._open_sd_stream(
                    idx1, lb1, self._buffer1, self._buffer1_lock
                )
            # Source 2 (mic): always sounddevice
            if lb2 and _HAS_PYAUDIOWPATCH:
                self._stream2 = self._open_loopback_stream(
                    idx2, self._buffer2, self._buffer2_lock
                )
            else:
                self._stream2 = self._open_sd_stream(
                    idx2, lb2, self._buffer2, self._buffer2_lock
                )
            self.process1 = None
            self.process2 = None
            self.output_file = output_file
            self.recording_process = True
            self._recording_mode = 'stereo'
            self._reader_threads = []

            self._start_stream(self._stream1)
            self._start_stream(self._stream2)

            print(f"Recording started -> {output_file}")
            return True

        except Exception as e:
            print(f"Error starting recording: {e}")
            return False

    def get_live_chunk(self):
        """
        Get new audio since last call, mixed to mono.
        Used by live transcription for preview.
        In mono mode, reads only buffer1. In stereo mode, mixes both buffers.

        Returns:
            numpy array of int16 mono samples, or None if no new data
        """
        if self._recording_mode == 'mono':
            with self._buffer1_lock:
                raw1 = b''.join(self._buffer1[self._live_read_pos1:])
                new_pos1 = len(self._buffer1)

            if not raw1:
                return None

            self._live_read_pos1 = new_pos1
            return np.frombuffer(raw1, dtype=np.int16)

        # Stereo mode: mix both channels
        with self._buffer1_lock:
            raw1 = b''.join(self._buffer1[self._live_read_pos1:])
            new_pos1 = len(self._buffer1)

        with self._buffer2_lock:
            raw2 = b''.join(self._buffer2[self._live_read_pos2:])
            new_pos2 = len(self._buffer2)

        if not raw1 and not raw2:
            return None

        ch1 = np.frombuffer(raw1, dtype=np.int16) if raw1 else np.array([], dtype=np.int16)
        ch2 = np.frombuffer(raw2, dtype=np.int16) if raw2 else np.array([], dtype=np.int16)

        if len(ch1) == 0 and len(ch2) == 0:
            return None

        min_len = min(len(ch1), len(ch2)) if len(ch1) > 0 and len(ch2) > 0 else max(len(ch1), len(ch2))
        if len(ch1) > 0 and len(ch2) > 0:
            mono = ((ch1[:min_len].astype(np.int32) + ch2[:min_len].astype(np.int32)) // 2).astype(np.int16)
        elif len(ch1) > 0:
            mono = ch1
        else:
            mono = ch2

        self._live_read_pos1 = new_pos1
        self._live_read_pos2 = new_pos2

        return mono

    def stop_recording_stereo(self):
        """
        Stop background recording and save to file.

        Returns:
            True if stopped and saved successfully, False otherwise
        """
        if PLATFORM == 'Windows':
            return self._stop_stereo_windows()
        if PLATFORM == 'Darwin':
            return self._stop_stereo_mac()
        return self._stop_stereo_linux()

    def _stop_stereo_linux(self):
        if not hasattr(self, 'process1') or self.process1 is None:
            print("No recording in progress")
            return False

        try:
            self.process1.terminate()
            self.process2.terminate()

            self._stop_readers.set()
            for t in self._reader_threads:
                t.join(timeout=5)

            # Combine buffered data
            with self._buffer1_lock:
                audio_data1 = b''.join(self._buffer1)
            with self._buffer2_lock:
                audio_data2 = b''.join(self._buffer2)

            channel1 = np.frombuffer(audio_data1, dtype=np.int16)
            channel2 = np.frombuffer(audio_data2, dtype=np.int16)

            min_len = min(len(channel1), len(channel2))
            channel1 = channel1[:min_len]
            channel2 = channel2[:min_len]

            stereo = np.empty((min_len * 2,), dtype=np.int16)
            stereo[0::2] = channel1
            stereo[1::2] = channel2

            with wave.open(self.output_file, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(stereo.tobytes())

            # Cleanup
            self.process1 = None
            self.process2 = None
            self.recording_process = None
            self._recording_mode = None
            self._buffer1 = []
            self._buffer2 = []
            self._reader_threads = []

            print(f"Recording stopped and saved")
            return True

        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False

    def _stop_stereo_windows(self):
        if self._stream1 is None:
            print("No recording in progress")
            return False

        saved = False
        try:
            # Save buffers FIRST — before touching streams — so audio is
            # persisted even if PortAudio crashes during stop/close.
            with self._buffer1_lock:
                audio_data1 = b''.join(self._buffer1)
            with self._buffer2_lock:
                audio_data2 = b''.join(self._buffer2)

            channel1 = np.frombuffer(audio_data1, dtype=np.int16)
            channel2 = np.frombuffer(audio_data2, dtype=np.int16)

            # Pad missing channel with silence (WASAPI loopback produces
            # no data when nothing is playing through speakers).
            if len(channel1) == 0 and len(channel2) > 0:
                channel1 = np.zeros(len(channel2), dtype=np.int16)
            elif len(channel2) == 0 and len(channel1) > 0:
                channel2 = np.zeros(len(channel1), dtype=np.int16)

            min_len = min(len(channel1), len(channel2))
            channel1 = channel1[:min_len]
            channel2 = channel2[:min_len]

            stereo = np.empty((min_len * 2,), dtype=np.int16)
            stereo[0::2] = channel1
            stereo[1::2] = channel2

            with wave.open(self.output_file, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(stereo.tobytes())

            saved = True
            print("Recording stopped and saved")

        except Exception as e:
            print(f"Error saving recording: {e}")

        # Close non-loopback streams (mic) first
        lb_streams = []
        for s in [self._stream2, self._stream1]:
            if isinstance(s, _LoopbackCapture):
                lb_streams.append(s)
            else:
                self._safe_close_stream(s)

        # Close loopback streams with mute to prevent WASAPI crash
        if lb_streams:
            self._close_loopbacks_with_mute(lb_streams)

        # Unconditional cleanup
        self._stream1 = None
        self._stream2 = None
        self.recording_process = None
        self._recording_mode = None
        self._buffer1 = []
        self._buffer2 = []
        self._cleanup_pyaudio()

        return saved

    # ------------------------------------------------------------------ #
    #  Source classification for GUI / CLI                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_sources_for_gui():
        """
        Get sources formatted for GUI display

        Returns:
            dict with 'monitors' and 'inputs' lists, each containing:
            - display_name: User-friendly description
            - device_name: Technical name for recording
            - state: RUNNING/SUSPENDED/IDLE
        """
        sources = AudioCapture.list_sources()

        monitors = []
        inputs = []

        for source in sources:
            name = source.get('name', '')
            desc = source.get('description', 'Unknown')
            state = source.get('state', 'UNKNOWN')

            item = {
                'display_name': desc,
                'device_name': name,
                'state': state
            }

            # Linux: PulseAudio monitors contain '.monitor'
            # Windows: loopback sources start with 'loopback_'
            if '.monitor' in name or name.startswith('loopback_'):
                monitors.append(item)
            elif 'input' in name or 'source' in name.lower():
                inputs.append(item)

        return {
            'monitors': monitors,
            'inputs': inputs
        }

    @staticmethod
    def set_source_mute(source_name, muted):
        """Mute or unmute an audio source at the system level.

        Args:
            source_name: PulseAudio source name (Linux) or device index string (Windows/macOS)
            muted: True to mute, False to unmute

        Returns:
            dict with 'success' bool and optional 'error' string
        """
        if PLATFORM == 'Linux':
            try:
                val = '1' if muted else '0'
                subprocess.run(
                    ['pactl', 'set-source-mute', source_name, val],
                    capture_output=True, text=True, check=True
                )
                return {'success': True}
            except subprocess.CalledProcessError as e:
                return {'success': False, 'error': str(e)}
        elif PLATFORM == 'Darwin':
            try:
                val = 'true' if muted else 'false'
                subprocess.run(
                    ['osascript', '-e', f'set volume input volume {0 if muted else 100}'],
                    capture_output=True, text=True, check=True
                )
                return {'success': True}
            except subprocess.CalledProcessError as e:
                return {'success': False, 'error': str(e)}
        elif PLATFORM == 'Windows':
            # Use pycaw to mute the default capture device
            if not _HAS_PYCAW:
                return {'success': False, 'error': 'pycaw not installed'}
            try:
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                from comtypes import CLSCTX_ALL
                from ctypes import cast, POINTER
                import comtypes
                devices = AudioUtilities.GetMicrophone()
                if devices is None:
                    return {'success': False, 'error': 'No microphone found'}
                interface = devices.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                )
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMute(1 if muted else 0, None)
                return {'success': True}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        return {'success': False, 'error': f'Unsupported platform: {PLATFORM}'}

    @staticmethod
    def get_source_mute(source_name):
        """Get the mute state of an audio source.

        Args:
            source_name: PulseAudio source name (Linux) or device index string (Windows/macOS)

        Returns:
            dict with 'muted' bool and 'success' bool
        """
        if PLATFORM == 'Linux':
            try:
                result = subprocess.run(
                    ['pactl', 'get-source-mute', source_name],
                    capture_output=True, text=True, check=True
                )
                muted = 'yes' in result.stdout.lower()
                return {'success': True, 'muted': muted}
            except subprocess.CalledProcessError as e:
                return {'success': False, 'muted': False, 'error': str(e)}
        elif PLATFORM == 'Darwin':
            try:
                result = subprocess.run(
                    ['osascript', '-e', 'input volume of (get volume settings)'],
                    capture_output=True, text=True, check=True
                )
                vol = int(result.stdout.strip())
                return {'success': True, 'muted': vol == 0}
            except Exception as e:
                return {'success': False, 'muted': False, 'error': str(e)}
        elif PLATFORM == 'Windows':
            if not _HAS_PYCAW:
                return {'success': False, 'muted': False, 'error': 'pycaw not installed'}
            try:
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                from comtypes import CLSCTX_ALL
                from ctypes import cast, POINTER
                devices = AudioUtilities.GetMicrophone()
                if devices is None:
                    return {'success': False, 'muted': False, 'error': 'No microphone found'}
                interface = devices.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                )
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                muted = bool(volume.GetMute())
                return {'success': True, 'muted': muted}
            except Exception as e:
                return {'success': False, 'muted': False, 'error': str(e)}
        return {'success': False, 'muted': False, 'error': f'Unsupported platform: {PLATFORM}'}

    @staticmethod
    def find_active_sources():
        """
        Find currently active (RUNNING) sources.
        On Linux, detects actively used PulseAudio sources.
        On Windows, returns the default WASAPI input/output devices.

        Returns:
            dict with 'active_monitor' and 'active_input' (or None if none found)
        """
        data = AudioCapture.get_sources_for_gui()

        active_monitor = None
        active_input = None

        for monitor in data['monitors']:
            if monitor['state'] == 'RUNNING':
                active_monitor = monitor
                break

        for input_src in data['inputs']:
            if input_src['state'] == 'RUNNING':
                active_input = input_src
                break

        return {
            'active_monitor': active_monitor,
            'active_input': active_input
        }
