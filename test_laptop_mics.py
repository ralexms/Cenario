# test_laptop_mics.py

from core.audio_capture import AudioCapture
import time

recorder = AudioCapture()

print("=== Testing Laptop Microphones ===\n")

# Test mic 1: Headphones Stereo Microphone
print("Test 1: Recording from 'Headphones Stereo Microphone'")
print("Speak into your laptop for 5 seconds...")
time.sleep(2)

recorder.record(
    source_name="alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__source",
    duration=5,
    output_file="laptop_mic1.wav"
)
print("✓ Saved to laptop_mic1.wav\n")

time.sleep(2)

# Test mic 2: Digital Microphone
print("Test 2: Recording from 'Digital Microphone'")
print("Speak into your laptop for 5 seconds...")
time.sleep(2)

recorder.record(
    source_name="alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_6__source",
    duration=5,
    output_file="laptop_mic2.wav"
)
print("✓ Saved to laptop_mic2.wav\n")

print("=== Listen to both ===")
print("aplay laptop_mic1.wav")
print("aplay laptop_mic2.wav")
print("\nWhich one captured your voice?")