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

    # ------------------------------------------------------------------ #
    #  Source listing                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_sources():
        """List all available audio sources (platform-aware)."""
        if PLATFORM == 'Windows':
            return AudioCapture._list_sources_windows()
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
    def _list_sources_windows():
        """List audio devices via sounddevice, preferring WASAPI."""
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
        for i, d in enumerate(devices):
            # Only include WASAPI devices when available
            if wasapi_idx is not None and d['hostapi'] != wasapi_idx:
                continue

            # Only consider devices that can provide input
            if d['max_input_channels'] <= 0:
                continue

            # PortAudio WASAPI exposes loopback capture as separate input
            # devices.  PaWasapi_IsLoopback() identifies them.
            try:
                is_loopback = bool(sd._lib.PaWasapi_IsLoopback(i))
            except Exception:
                is_loopback = False

            if is_loopback:
                state = 'RUNNING' if d['name'] == default_output_name else 'IDLE'
                sources.append({
                    'name': f"loopback_{i}",
                    'description': f"{d['name']} (Loopback)",
                    'state': state,
                    'index': i,
                    'default_samplerate': d['default_samplerate'],
                })
            else:
                state = 'RUNNING' if d['name'] == default_input_name else 'IDLE'
                sources.append({
                    'name': f"input_{i}",
                    'description': d['name'],
                    'state': state,
                    'index': i,
                    'default_samplerate': d['default_samplerate'],
                })

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
    def _get_device_channels(device_idx, is_loopback):
        """Query native channel count for a device.

        Loopback devices must be opened with their native input channel
        count — WASAPI rejects a mismatch even with auto_convert.
        Regular input devices are opened mono.
        """
        if is_loopback:
            info = sd.query_devices(device_idx)
            return max(int(info['max_input_channels']), 1)
        return 1  # regular input devices: always open mono

    @staticmethod
    def _downmix_to_mono_int16(indata):
        """Downmix multi-channel int16 input to mono, avoiding overflow.

        Args:
            indata: numpy array of shape (frames, channels), dtype int16
        Returns:
            1-D numpy array of int16 mono samples
        """
        if indata.shape[1] == 1:
            return indata[:, 0]
        # Use int32 intermediate to avoid int16 overflow when summing channels
        mixed = indata.astype(np.int32).mean(axis=1).astype(np.int16)
        return mixed

    def _make_sd_callback(self, buffer_list, lock):
        """Create a sounddevice callback that appends mono int16 bytes to a buffer."""
        def callback(indata, frames, time_info, status):
            mono = AudioCapture._downmix_to_mono_int16(indata)
            with lock:
                buffer_list.append(mono.tobytes())
        return callback

    def _open_sd_stream(self, device_idx, is_loopback, buffer_list, lock):
        """Open and return a sounddevice InputStream for the given device."""
        channels = self._get_device_channels(device_idx, is_loopback)
        stream = sd.InputStream(
            device=device_idx,
            channels=channels,
            samplerate=self.sample_rate,
            dtype='int16',
            callback=self._make_sd_callback(buffer_list, lock),
            extra_settings=self._wasapi_extra(is_loopback),
        )
        return stream

    # ------------------------------------------------------------------ #
    #  Blocking recording (fixed duration)                                 #
    # ------------------------------------------------------------------ #

    def record(self, source_name, duration, output_file):
        """Record audio from a source to WAV file (blocking)."""
        if PLATFORM == 'Windows':
            return self._record_windows(source_name, duration, output_file)
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

            def callback(indata, frames, time_info, status):
                mono = AudioCapture._downmix_to_mono_int16(indata)
                with lock:
                    chunks.append(mono.tobytes())

            channels = self._get_device_channels(device_idx, is_loopback)
            stream = sd.InputStream(
                device=device_idx,
                channels=channels,
                samplerate=self.sample_rate,
                dtype='int16',
                callback=callback,
                extra_settings=self._wasapi_extra(is_loopback),
            )
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

            stream1 = self._open_sd_stream(idx1, lb1, buf1, lock1)
            stream2 = self._open_sd_stream(idx2, lb2, buf2, lock2)

            stream1.start()
            stream2.start()

            import time
            time.sleep(duration)

            stream1.stop()
            stream2.stop()
            stream1.close()
            stream2.close()

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

            self._stream1.start()

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

        # Stop/close stream — errors here must not prevent cleanup
        try:
            if self._stream1.active:
                self._stream1.stop()
        except Exception as e:
            print(f"Warning: stream stop failed: {e}")
        try:
            self._stream1.close()
        except Exception as e:
            print(f"Warning: stream close failed: {e}")

        # Unconditional cleanup
        self._stream1 = None
        self._stream2 = None
        self.recording_process = None
        self._recording_mode = None
        self._buffer1 = []

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

            self._stream1 = self._open_sd_stream(
                idx1, lb1, self._buffer1, self._buffer1_lock
            )
            self._stream2 = self._open_sd_stream(
                idx2, lb2, self._buffer2, self._buffer2_lock
            )
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

        # Stop/close streams — errors here must not prevent cleanup
        for stream in (self._stream1, self._stream2):
            if stream is None:
                continue
            try:
                if stream.active:
                    stream.stop()
            except Exception as e:
                print(f"Warning: stream stop failed: {e}")
            try:
                stream.close()
            except Exception as e:
                print(f"Warning: stream close failed: {e}")

        # Unconditional cleanup
        self._stream1 = None
        self._stream2 = None
        self.recording_process = None
        self._recording_mode = None
        self._buffer1 = []
        self._buffer2 = []

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
