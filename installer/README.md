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

**bitsandbytes errors on Windows** — 4-bit LLM quantization has limited Windows support. Use 8-bit or disable quantization in the summarization settings.

**Models re-downloading each run** — Ensure the launcher scripts are being used (they set `HF_HOME` to the local `models/` directory). Running `python gui/app.py` directly won't set this.
