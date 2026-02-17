# test_audio_class.py

from core.audio_capture import AudioCapture


def main():
    # Create recorder instance
    recorder = AudioCapture()

    # Test 1: List sources
    print("=== Available Audio Sources ===")
    sources = recorder.list_sources()
    for i, source in enumerate(sources):
        print(f"{i + 1}. {source.get('description', 'Unknown')}")
        print(f"   Name: {source.get('name', 'Unknown')}")
        print(f"   State: {source.get('state', 'Unknown')}\n")

    # Test 2: Record 5 seconds
    print("\n=== Recording Test ===")
    test_source = "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_3__sink.monitor"

    print(f"Recording 5 seconds from HDMI output...")
    print("Make sure audio is playing!")

    success = recorder.record(test_source, duration=5, output_file="../class_test.wav")

    if success:
        print("✓ Recording successful!")
        print("Play with: aplay class_test.wav")
    else:
        print("✗ Recording failed")


if __name__ == "__main__":
    main()