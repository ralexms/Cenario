from core.audio_capture import AudioCapture

recorder = AudioCapture()

print("Recording 5 seconds from your microphone...")
print("Please speak into your mic!")

recorder.record(
    source_name="alsa_input.usb-Generic_USB_Audio_201604140001-00.analog-stereo-headset",
    duration=5,
    output_file="mic_test.wav"
)

print("✓ Done! Play with: aplay mic_test.wav")