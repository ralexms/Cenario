# core/audio_capture.py

import subprocess
import wave
import json
import threading
import numpy as np


class AudioCapture:
    """Handles audio recording from PulseAudio sources"""

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

    @staticmethod
    def list_sources():
        """List all available PulseAudio sources"""
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

    def record(self, source_name, duration, output_file):
        """Record audio from a PulseAudio source to WAV file"""
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

    def record_stereo(self, source1_name, source2_name, duration, output_file):
        """
        Record from two sources simultaneously and combine into stereo WAV

        Args:
            source1_name: First audio source (e.g., meeting audio) -> Left channel
            source2_name: Second audio source (e.g., microphone) -> Right channel
            duration: Recording duration in seconds
            output_file: Output WAV file path

        Returns:
            True if successful, False otherwise
        """
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

    def stop_recording_mono(self):
        """
        Stop mono background recording and save to file.

        Returns:
            True if stopped and saved successfully, False otherwise
        """
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

    def start_recording_stereo(self, source1_name, source2_name, output_file):
        """
        Start stereo recording in background (non-blocking).
        Uses reader threads to accumulate audio buffers for live preview.

        Returns:
            True if started successfully, False otherwise
        """
        if self.recording_process is not None:
            print("Recording already in progress")
            return False

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

    @staticmethod
    def get_sources_for_gui():
        """
        Get sources formatted for GUI display

        Returns:
            dict with 'monitors' and 'inputs' lists, each containing:
            - display_name: User-friendly description
            - device_name: Technical name for recording
            - state: RUNNING/SUSPENDED
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

            if '.monitor' in name:
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
        Find currently active (RUNNING) sources

        Returns:
            dict with 'active_monitor' and 'active_input' (or None if none running)
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
