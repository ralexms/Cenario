# reconfigure_echo_cancel.py

import subprocess


def unload_echo_cancel():
    """Remove existing echo-cancel module"""
    try:
        result = subprocess.run(
            ['pactl', 'list', 'modules', 'short'],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.split('\n'):
            if 'module-echo-cancel' in line:
                module_id = line.split()[0]
                subprocess.run(['pactl', 'unload-module', module_id], check=True)
                print(f"✓ Unloaded old echo-cancel module {module_id}")

    except Exception as e:
        print(f"Error: {e}")


def load_echo_cancel_laptop():
    """Configure echo-cancel for laptop mic + monitor speakers"""

    # Your monitor speakers (HDMI output)
    sink = "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_3__sink"

    # Your laptop's digital microphone
    source = "alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_6__source"

    cmd = [
        'pactl', 'load-module', 'module-echo-cancel',
        'source_name=laptop_mic_echo_cancelled',
        'sink_name=monitor_speakers_echo_cancelled',
        f'source_master={source}',
        f'sink_master={sink}',
        'aec_method=webrtc'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Echo cancellation configured!")
        print(f"  Microphone: Laptop Digital Mic")
        print(f"  Speakers: Monitor HDMI Output")
        print(f"\n  Use source: laptop_mic_echo_cancelled")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")


if __name__ == "__main__":
    print("=== Configuring Echo Cancellation ===\n")