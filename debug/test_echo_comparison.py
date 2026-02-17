# test_echo_comparison.py

from core.audio_capture import AudioCapture
import time

recorder = AudioCapture()

print("=== Echo Cancellation Test ===\n")
print("Instructions:")
print("1. Play audio through your MONITOR SPEAKERS")
print("2. Speak into your LAPTOP MICROPHONE while audio plays")
print("3. We'll compare regular vs echo-cancelled mic")
print("\nStarting in 5 seconds...\n")
time.sleep(5)

# Test 1: Regular laptop microphone
print("Recording 10s from REGULAR laptop mic...")
recorder.record(
    source_name="alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_6__source",
    duration=10,
    output_file="../mic_regular.wav"
)
print("✓ Done\n")

time.sleep(2)

# Test 2: Echo-cancelled microphone
print("Recording 10s from ECHO-CANCELLED laptop mic...")
recorder.record(
    source_name="laptop_mic_echo_cancelled",
    duration=10,
    output_file="../mic_echo_cancelled.wav"
)
print("✓ Done\n")

print("=== Compare the results ===")
print("Regular mic:         aplay mic_regular.wav")
print("Echo-cancelled mic:  aplay mic_echo_cancelled.wav")
print("\nThe echo-cancelled version should reduce speaker bleed")