import subprocess
import wave
import time


def check_capture(source_name, duration=10, output_file="test_capture.wav"):
    """Test recording from a PulseAudio source"""

    print(f"Recording {duration}s from: {source_name}")
    print(f"Output: {output_file}")
    print("Make sure audio is playing...")
    time.sleep(2)

    cmd = [
        'parec',
        '--device', source_name,
        '--channels', '1',
        '--rate', '16000',
        '--format', 's16le'
    ]

    try:
        # Start recording
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        time.sleep(duration)
        process.terminate()

        # Get the raw audio data
        audio_data, _ = process.communicate()

        # Write as WAV file
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_data)

        print(f"\n✓ Recording saved to {output_file}")
        print(f"Size: {len(audio_data)} bytes")
        print("Play it back with: aplay test_capture.wav")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Test with your USB headset monitor (source #7)
    # check_capture("alsa_output.usb-Generic_USB_Audio_201604140001-00.analog-stereo-headset.monitor")
    check_capture("alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_3__sink.monitor")