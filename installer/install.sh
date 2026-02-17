#!/usr/bin/env bash
# Cenario installer for Linux
# Creates a self-contained installation with venv, app code, model cache, and launchers.

set -euo pipefail

# ---- Defaults ----
INSTALL_DIR="$HOME/cenario"
CPU_ONLY=0
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--install-dir <path>] [--cpu-only]"
            echo ""
            echo "  --install-dir <path>  Installation directory (default: ~/cenario)"
            echo "  --cpu-only            Skip CUDA detection, install CPU-only PyTorch"
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
[[ -z "$PYTHON" ]] && error "Python >= 3.10 not found. Install Python 3.10+ and try again."
info "Using Python: $PYTHON ($("$PYTHON" --version))"

# ---- Check for venv module ----
if ! "$PYTHON" -c "import venv" &>/dev/null; then
    error "Python venv module not found. Install it (e.g., sudo apt install python3-venv) and try again."
fi

# ---- Check parec (PulseAudio/PipeWire) ----
if ! command -v parec &>/dev/null; then
    warn "parec not found. System audio capture requires PulseAudio or PipeWire."
    warn "Install with: sudo apt install pulseaudio-utils  (or pipewire-pulse)"
fi

# ---- Detect CUDA ----
TORCH_INDEX="cpu"
if [[ "$CPU_ONLY" -eq 0 ]] && command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' || true)
    if [[ -n "$CUDA_VER" ]]; then
        CUDA_MAJOR="${CUDA_VER%%.*}"
        CUDA_MINOR="${CUDA_VER##*.}"
        if [[ "$CUDA_MAJOR" -gt 12 ]] || [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 4 ]]; then
            TORCH_INDEX="cu124"
        elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 1 ]]; then
            TORCH_INDEX="cu121"
        elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
            TORCH_INDEX="cu118"
        else
            warn "CUDA $CUDA_VER detected but too old for GPU PyTorch. Falling back to CPU."
        fi
        [[ "$TORCH_INDEX" != "cpu" ]] && info "Detected CUDA $CUDA_VER -> using PyTorch index: $TORCH_INDEX"
    fi
else
    [[ "$CPU_ONLY" -eq 1 ]] && info "CPU-only mode requested." || info "No NVIDIA GPU detected. Installing CPU-only PyTorch."
fi

# ---- Create directory structure ----
info "Installing to: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"/{app,models,data}

# ---- Copy app source ----
info "Copying application source..."
# Remove old app source (but not user data)
rm -rf "$INSTALL_DIR/app/cenario.py" \
       "$INSTALL_DIR/app/core" \
       "$INSTALL_DIR/app/gui"

cp "$REPO_DIR/cenario.py" "$INSTALL_DIR/app/"
cp -r "$REPO_DIR/core" "$INSTALL_DIR/app/core"
cp -r "$REPO_DIR/gui" "$INSTALL_DIR/app/gui"

# Remove __pycache__ from copied source
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
info "Installing PyTorch ($TORCH_INDEX)..."
if [[ "$TORCH_INDEX" == "cpu" ]]; then
    "$PIP" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
else
    "$PIP" install torch torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_INDEX" --quiet
fi

# ---- Install remaining dependencies ----
info "Installing dependencies..."
"$PIP" install -r "$REPO_DIR/installer/requirements-pip.txt" --quiet

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
echo "  PyTorch variant:    $TORCH_INDEX"
echo ""
echo "  Start the GUI:      $INSTALL_DIR/cenario-gui"
echo "  Use the CLI:        $INSTALL_DIR/cenario --help"
echo ""
echo "  To enable speaker diarization, add your HuggingFace token to:"
echo "    $INSTALL_DIR/.env"
echo ""
