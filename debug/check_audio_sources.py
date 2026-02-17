import subprocess
import json


def list_audio_sources():
    """List all PulseAudio sources"""
    try:
        # Get source list from pactl
        result = subprocess.run(
            ['pactl', '-f', 'json', 'list', 'sources'],
            capture_output=True,
            text=True,
            check=True
        )

        sources = json.loads(result.stdout)

        print(f"Found {len(sources)} audio sources:\n")
        for i, source in enumerate(sources):
            name = source.get('name', 'Unknown')
            desc = source.get('description', 'No description')
            state = source.get('state', 'Unknown')

            print(f"{i + 1}. {desc}")
            print(f"   Name: {name}")
            print(f"   State: {state}")
            print()

        return sources

    except subprocess.CalledProcessError as e:
        print(f"Error running pactl: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing pactl output: {e}")
        return None


if __name__ == "__main__":
    list_audio_sources()