#!/usr/bin/env bash
# Cenario installer for macOS
# Creates a self-contained installation with venv, app code, model cache, and launchers.
#
# System audio capture requires a virtual audio driver.
# BlackHole (free) is recommended: https://github.com/ExistentialAudio/BlackHole
# Install the 2-channel variant, then use it as the loopback source in Cenario.

set -euo pipefail

# ---- Defaults ----
INSTALL_DIR="$HOME/cenario"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--install-dir <path>]"
            echo ""
            echo "  --install-dir <path>  Installation directory (default: ~/cenario)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---- Helpers ----
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# ---- Detect Apple Silicon vs Intel ----
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
    info "Detected Apple Silicon (arm64) — PyTorch MPS will be available for speaker diarization."
else
    info "Detected Intel Mac (x86_64) — running on CPU."
fi

# ---- Check Python ----
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major="${ver%%.*}"
        minor="${ver##*.}"
        if [[ "$major" -eq 3 && "$minor" -ge 10 ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done
[[ -z "$PYTHON" ]] && error "Python >= 3.10 not found. Install via Homebrew (brew install python@3.11) and try again."
info "Using Python: $PYTHON ($("$PYTHON" --version))"

# ---- Check for venv module ----
if ! "$PYTHON" -c "import venv" &>/dev/null; then
    error "Python venv module not found. Try: brew install python@3.11"
fi

# ---- Check for BlackHole (system audio loopback) ----
if "$PYTHON" -c "import sounddevice as sd; devs = sd.query_devices(); names = [d['name'].lower() for d in devs]; exit(0 if any('blackhole' in n or 'soundflower' in n for n in names) else 1)" 2>/dev/null; then
    info "Virtual audio loopback device detected (BlackHole or Soundflower)."
else
    warn "No virtual audio loopback device found."
    warn "To capture system/meeting audio, install BlackHole 2ch:"
    warn "  https://github.com/ExistentialAudio/BlackHole"
    warn "  or: brew install blackhole-2ch"
    warn "Microphone-only recording will still work without it."
fi

# ---- Create directory structure ----
info "Installing to: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"/{app,models,data}

# ---- Copy app source ----
info "Copying application source..."
rm -rf "$INSTALL_DIR/app/cenario.py" \
       "$INSTALL_DIR/app/updater.py" \
       "$INSTALL_DIR/app/core" \
       "$INSTALL_DIR/app/gui"

cp "$REPO_DIR/cenario.py" "$INSTALL_DIR/app/"
cp "$REPO_DIR/updater.py" "$INSTALL_DIR/app/"
cp -r "$REPO_DIR/core" "$INSTALL_DIR/app/core"
cp -r "$REPO_DIR/gui" "$INSTALL_DIR/app/gui"

find "$INSTALL_DIR/app" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ---- Create / update venv ----
VENV_DIR="$INSTALL_DIR/venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    info "Virtual environment already exists, reusing."
fi

PIP="$VENV_DIR/bin/pip"
PYTHON_VENV="$VENV_DIR/bin/python"

info "Upgrading pip..."
"$PYTHON_VENV" -m pip install --upgrade pip --quiet

# ---- Install PyTorch ----
# On macOS, PyTorch is installed from the standard index.
# Apple Silicon gets MPS support automatically; Intel Macs use CPU.
info "Installing PyTorch (standard index, MPS/CPU)..."
"$PIP" install torch torchaudio --quiet

# ---- Install remaining dependencies ----
info "Installing dependencies..."
"$PIP" install -r "$REPO_DIR/installer/requirements-pip.txt" --quiet

# ---- bitsandbytes: try to install, warn on failure ----
# bitsandbytes 0.43+ supports macOS MPS. If it fails for any reason,
# summarization still works without 4/8-bit quantization.
info "Installing bitsandbytes (optional quantization support)..."
if ! "$PIP" install bitsandbytes==0.49.2 --quiet 2>/dev/null; then
    warn "bitsandbytes failed to install. Quantized model loading unavailable."
    warn "Summarization will still work without quantization."
fi

# ---- Generate launcher scripts ----
info "Creating launcher scripts..."

# GUI launcher
cat > "$INSTALL_DIR/cenario-gui" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="$SCRIPT_DIR/models"
export CENARIO_DATA_DIR="$SCRIPT_DIR/data"
exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/app/gui/app.py" "$@"
LAUNCHER_EOF
chmod +x "$INSTALL_DIR/cenario-gui"

# CLI launcher
cat > "$INSTALL_DIR/cenario" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="$SCRIPT_DIR/models"
export CENARIO_DATA_DIR="$SCRIPT_DIR/data"
exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/app/cenario.py" "$@"
LAUNCHER_EOF
chmod +x "$INSTALL_DIR/cenario"

# ---- Create .env template if not present ----
if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    cat > "$INSTALL_DIR/.env" << 'ENV_EOF'
# Cenario configuration
# Uncomment and set your HuggingFace token to enable speaker diarization:
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ENV_EOF
    info "Created .env template at $INSTALL_DIR/.env"
fi

# ---- Done ----
echo ""
info "Installation complete!"
echo ""
echo "  Install directory:  $INSTALL_DIR"
echo "  Architecture:       $ARCH"
echo ""
echo "  Start the GUI:      $INSTALL_DIR/cenario-gui"
echo "  Use the CLI:        $INSTALL_DIR/cenario --help"
echo ""
echo "  To enable speaker diarization, add your HuggingFace token to:"
echo "    $INSTALL_DIR/.env"
echo ""
echo "  NOTE: System audio capture requires BlackHole (2ch) or Soundflower."
echo "  Install with: brew install blackhole-2ch"
echo "  Then select it as the loopback source in Cenario."
echo ""
