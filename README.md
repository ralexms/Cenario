# Cenario

Cenario is a local, privacy-focused meeting transcription and summarization tool designed to run entirely on your own hardware. It captures audio from both your microphone and system audio (for online meetings), transcribes it using OpenAI's Whisper, identifies speakers via diarization, and generates summaries using local LLMs.

## Features

*   **Privacy First**: All processing (recording, transcription, summarization) happens locally on your machine. No audio is sent to the cloud.
*   **Dual-Channel Recording**: Captures your microphone and system audio (meeting participants) separately for clear stereo transcription.
*   **Local Transcription**: Uses `faster-whisper` for high-performance speech-to-text.
*   **Speaker Diarization**: Identifies different speakers in the meeting using `pyannote.audio`.
*   **Local Summarization**: Generates meeting summaries and action points using `Qwen2.5` models, optimized for consumer GPUs (supports 4GB VRAM).
*   **Web Interface**: A clean, dark-mode GUI to manage recordings, view live previews, and run post-processing tasks.
*   **CLI Support**: Full command-line interface for headless operation or automation.
*   **Export Formats**: Exports transcripts to TXT, JSON, SRT, and summaries to Markdown.

![Cenario Interface - Recording](Cenario%20—%20Meeting%20Transcription%20-%20Recording.png)

![Cenario Interface - Post-Processing](Cenario%20—%20Meeting%20Transcription%20-%20Post-Processing.png)

![Cenario Interface - Summarization](Cenario%20—%20Meeting%20Transcription%20-%20Summarization.png)

## Requirements

*   **OS**: Linux (PulseAudio/PipeWire), Windows (WASAPI), or macOS (CoreAudio)
*   **GPU**: NVIDIA GPU with CUDA recommended (4GB+ VRAM for summarization). Apple Silicon MPS is used automatically for diarization on Mac. CPU fallback is available on all platforms.
*   **Python**: 3.10+
*   **macOS only — system audio capture**: macOS does not expose loopback audio natively. To capture meeting audio from your speakers you need a free virtual audio driver. [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) is recommended (`brew install blackhole-2ch`). Microphone-only recording works without it.

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

### macOS

```bash
# Default install to ~/cenario
bash installer/install-mac.sh

# Custom location
bash installer/install-mac.sh --install-dir /opt/cenario
```

> **System audio capture**: Install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) before or after running the installer. The GUI will display a setup banner until it is detected.
> ```bash
> brew install blackhole-2ch
> ```
> After installing BlackHole, restart any active audio applications and click **Check again** in the Cenario banner. Cenario will then list it as a loopback source automatically.

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
# Linux / macOS
~/cenario/cenario-gui

# Windows
%USERPROFILE%\cenario\cenario-gui.bat
```

The browser opens automatically. To disable: set `CENARIO_NO_BROWSER=1` in the `.env` file or environment.

**Workflow**:
1.  **Meeting Setup**: Choose your language and mode (Online or Local).
2.  **Source Selection**: Select your microphone and system monitor (for online meetings).
3.  **Recording**: Start recording. You'll see a live text preview.
4.  **Post-Processing**: After stopping, select the file and run transcription.
6.  **Summarization**: Generate a summary and action points using the local LLM.
7.  **Export**: Download your files.

### Command Line Interface (CLI)

You can also run Cenario from the terminal.

*   **List Audio Sources**:
    ```bash
    python cenario.py sources
    ```

*   **Record a Meeting**:
    ```bash
    python cenario.py record
    ```
    *   Use `--local` for in-person meetings (mic only).
    *   Use `--summarize` to automatically summarize after recording.

*   **Transcribe an Existing File**:
    ```bash
    python cenario.py transcribe recordings/my_meeting.wav --summarize
    ```

## Configuration

*   **Models**: Cenario supports various Whisper model sizes (`tiny`, `base`, `small`, `medium`, `large-v3`). Larger models are more accurate but require more VRAM.
*   **Summarization Models**:
    *   `Qwen/Qwen2.5-0.5B-Instruct`: Fast, very low VRAM (~1.5GB). Good for quick summaries.
    *   `Qwen/Qwen2.5-1.5B-Instruct`: Balanced performance (~3.5GB VRAM).
    *   `Qwen/Qwen2.5-3B-Instruct`: Higher quality (~6.5GB VRAM).

## Troubleshooting

*   **CUDA Out of Memory**: If you encounter OOM errors during summarization, try selecting the **0.5B** model or closing other GPU-intensive applications. The application attempts to handle this gracefully by unloading models when not in use.
*   **Audio Sources (Linux)**: If you don't see your monitor source, ensure you are using PulseAudio or PipeWire and that your recording device is set to "Monitor of..." in your system sound settings.
*   **Audio Sources (Windows)**: System audio capture uses WASAPI loopback. If loopback recording fails, ensure your output device supports shared mode. The default output device is auto-selected.
*   **Audio Sources (macOS)**: System audio capture requires a virtual loopback device. Install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) (`brew install blackhole-2ch`). Once installed, restart any active audio apps and click **Check again** in the Cenario warning banner — it will disappear once BlackHole is detected. Without BlackHole only your microphone is available.
*   **Summarization on Mac (no GPU)**: The local LLM summarizer runs on CPU. Expect slower generation times compared to a CUDA or Apple Silicon MPS GPU. Choosing the **0.5B** model keeps it fast enough for practical use.