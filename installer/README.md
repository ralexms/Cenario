# Cenario Installer

Self-contained installer for Cenario. Produces a single-folder installation with its own Python virtual environment, model cache, and recordings directory.

## System Requirements

**Linux:**
- Python 3.10+
- `python3-venv` package (e.g., `sudo apt install python3-venv`)
- PulseAudio or PipeWire with `parec` for system audio capture (`sudo apt install pulseaudio-utils`)
- NVIDIA GPU + driver for GPU acceleration (optional — CPU mode works without it)

**Windows:**
- Python 3.10+ from [python.org](https://www.python.org/downloads/) (ensure "Add to PATH" is checked during install)
- NVIDIA GPU + driver for GPU acceleration (optional)

## Installation

### Linux

```bash
# Default install to ~/cenario
bash installer/install.sh

# Custom location
bash installer/install.sh --install-dir /opt/cenario

# Force CPU-only (skip CUDA detection)
bash installer/install.sh --cpu-only
```

### Windows (PowerShell)

```powershell
# Default install to %USERPROFILE%\cenario
.\installer\install.ps1

# Custom location
.\installer\install.ps1 -InstallDir C:\cenario

# Force CPU-only
.\installer\install.ps1 -CpuOnly
```

If PowerShell blocks script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Usage

### GUI (Web Interface)

```bash
# Linux
~/cenario/cenario-gui

# Windows
%USERPROFILE%\cenario\cenario-gui.bat
```

The browser opens automatically. To disable: set `CENARIO_NO_BROWSER=1` in the `.env` file or environment.

### CLI

```bash
# Linux
~/cenario/cenario --help
~/cenario/cenario sources
~/cenario/cenario record

# Windows
%USERPROFILE%\cenario\cenario.bat --help
```

## Installed Folder Structure

```
<install_dir>/
├── app/                    # Application source code
│   ├── cenario.py
│   ├── core/
│   └── gui/
├── venv/                   # Python virtual environment
├── models/                 # HuggingFace model cache (HF_HOME)
├── data/                   # Default recordings output
├── .env                    # HF token and configuration
├── cenario-gui[.bat]       # GUI launcher
└── cenario[.bat]           # CLI launcher
```

## Configuration

Edit `<install_dir>/.env`:

```bash
# Required for speaker diarization:
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional:
# CENARIO_PORT=5000           # Web UI port (default: 5000)
# CENARIO_NO_BROWSER=1        # Don't auto-open browser
# CENARIO_DEBUG=1              # Enable Flask debug mode
```

## CUDA Detection

The installer automatically detects your NVIDIA GPU's CUDA version via `nvidia-smi` and installs the matching PyTorch build:

| CUDA Version | PyTorch Index |
|---|---|
| >= 13.0 | cu130 |
| >= 12.8 | cu128 |
| >= 12.6 | cu126 |
| >= 12.4 | cu124 |
| >= 12.1 | cu121 |
| >= 11.8 | cu118 |
| < 11.8 / No GPU | cpu |

PyTorch pip wheels bundle their own CUDA runtime, so you do **not** need the CUDA toolkit installed.

## Updating

Re-running the installer updates the application code and dependencies while preserving your `models/`, `data/`, and `.env`:

```bash
# Linux
bash installer/install.sh --install-dir ~/cenario

# Windows
.\installer\install.ps1 -InstallDir $env:USERPROFILE\cenario
```

## Troubleshooting

**"No module named venv"** — Install the venv package: `sudo apt install python3-venv`

**"parec not found"** — Install PulseAudio utilities: `sudo apt install pulseaudio-utils`

**No GPU detected despite having NVIDIA GPU** — Ensure NVIDIA drivers are installed and `nvidia-smi` works from your terminal.

**PyTorch warns that your GPU compute capability is unsupported** — Example: `NVIDIA GB10 with CUDA capability sm_121 is not compatible with the current PyTorch installation`. This means the installed torch wheel is too old for that GPU architecture. Reinstall `torch` and `torchaudio` from a newer PyTorch CUDA index.

As of April 9, 2026, the official PyTorch install matrix includes CUDA 12.8 and 13.0 wheels for recent releases, and PyTorch 2.7 introduced Blackwell support.

**Whisper says `This CTranslate2 package was not compiled with CUDA support` on Linux ARM64/AArch64** — This is usually not a model setting issue. On ARM64, `pip install ctranslate2` can still produce a CPU-only wheel even when `torch` can see the GPU.

Validate the installed stack from the Cenario venv:

```bash
~/cenario/venv/bin/python - <<'PY'
import ctranslate2, torch
print("torch.cuda.is_available() =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0) =", torch.cuda.get_device_name(0))
print("ctranslate2.get_cuda_device_count() =", ctranslate2.get_cuda_device_count())
PY
```

If `torch.cuda.is_available()` is `True` but `ctranslate2.get_cuda_device_count()` is `0`, faster-whisper will run on CPU until `ctranslate2` is rebuilt or replaced with a CUDA-enabled build for that machine.

As of April 9, 2026, the CTranslate2 4.7.1 installation docs still describe GPU wheels in terms of CUDA 12.x and cuDNN 8. A CUDA 13-capable driver alone does not make a CPU-only `ctranslate2` wheel use the GPU.

**bitsandbytes errors on Windows** — 4-bit LLM quantization has limited Windows support. Use 8-bit or disable quantization in the summarization settings.

**Models re-downloading each run** — Ensure the launcher scripts are being used (they set `HF_HOME` to the local `models/` directory). Running `python gui/app.py` directly won't set this.
