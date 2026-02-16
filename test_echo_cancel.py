import subprocess
import json


def check_echo_cancel_module():
    """Check if echo cancellation module is loaded"""
    try:
        result = subprocess.run(
            ['pactl', 'list', 'modules', 'short'],
            capture_output=True,
            text=True,
            check=True
        )

        if 'module-echo-cancel' in result.stdout:
            print("✓ Echo cancellation module already loaded")
            return True
        else:
            print("✗ Echo cancellation module not loaded")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error checking modules: {e}")
        return False


def load_echo_cancel():
    """Load echo cancellation module"""
    try:
        # Load with default settings - uses your default sink and source
        result = subprocess.run(
            ['pactl', 'load-module', 'module-echo-cancel'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Echo cancellation module loaded (ID: {result.stdout.strip()})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error loading module: {e}")
        return False


def list_sources():
    """List available sources to find echo-cancelled one"""
    try:
        result = subprocess.run(
            ['pactl', '-f', 'json', 'list', 'sources'],
            capture_output=True,
            text=True,
            check=True
        )
        sources = json.loads(result.stdout)

        print("\n=== Audio Sources ===")
        for source in sources:
            name = source.get('name', '')
            desc = source.get('description', '')
            print(f"- {desc}")
            print(f"  {name}\n")

    except Exception as e:
        print(f"Error listing sources: {e}")


if __name__ == "__main__":
    print("=== Testing Echo Cancellation ===\n")

    if not check_echo_cancel_module():
        print("\nLoading echo cancellation module...")
        load_echo_cancel()

    print("\n" + "=" * 50)
    list_sources()