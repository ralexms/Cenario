import subprocess
import json

result = subprocess.run(
    ['pactl', '-f', 'json', 'list', 'sources'],
    capture_output=True,
    text=True,
    check=True
)

sources = json.loads(result.stdout)

print("=== Current Audio Sources ===\n")
for i, source in enumerate(sources):
    name = source.get('name', '')
    desc = source.get('description', '')
    state = source.get('state', 'UNKNOWN')

    # Check if it's an input device (microphone)
    if 'input' in name or 'source' in name.lower():
        print(f"{i + 1}. {desc}")
        print(f"   Name: {name}")
        print(f"   State: {state}")

        # Show if it's the default
        if source.get('default', False):
            print("   [DEFAULT]")
        print()