from core.audio_capture import AudioCapture

recorder = AudioCapture()

print("=== Stereo Recording Test ===\n")
print("Setup:")
print("- LEFT channel: Monitor speakers (HDMI output)")
print("- RIGHT channel: Laptop microphone")
print("\nInstructions:")
print("1. Play audio on your monitor speakers")
print("2. Speak into your laptop mic")
print("3. Recording will capture both for 10 seconds")
print("\nStarting in 5 seconds...\n")

import time
time.sleep(5)

# Monitor speakers (meeting audio)
source1 = "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_3__sink.monitor"

# Laptop microphone (your voice)
source2 = "alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_6__source"

print("Recording...")
success = recorder.record_stereo(
    source1_name=source1,
    source2_name=source2,
    duration=10,
    output_file="../stereo_test.wav"
)

if success:
    print("\n✓ Stereo recording complete!")
    print("Play with: aplay stereo_test.wav")
    print("\nLeft channel = monitor audio")
    print("Right channel = your voice")
else:
    print("\n✗ Recording failed")