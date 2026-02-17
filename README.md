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

*   **OS**: Linux (PulseAudio/PipeWire required for system audio capture)
*   **GPU**: NVIDIA GPU with CUDA support recommended (4GB+ VRAM for summarization).
*   **Python**: 3.8+
*   **System Dependencies**: `portaudio19-dev`, `python3-dev`

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/cenario.git
    cd cenario
    ```

2.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to install PyTorch separately to match your CUDA version)*

## Usage

### Web Interface (GUI)

The easiest way to use Cenario is via the web interface.

1.  Start the server:
    ```bash
    python gui/app.py
    ```
2.  Open your browser and navigate to `http://localhost:5000`.

**Workflow**:
1.  **Meeting Setup**: Choose your language and mode (Online/Stereo or Local/Mono).
2.  **Source Selection**: Select your microphone and system monitor (for online meetings).
3.  **Recording**: Start recording. You'll see a live text preview.
4.  **Post-Processing**: After stopping, select the file and run transcription (with optional diarization).
5.  **Speaker Naming**: (Optional) Rename "SPEAKER_00", "SPEAKER_01" to actual names.
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
*   **Audio Sources**: If you don't see your monitor source, ensure you are using PulseAudio or PipeWire and that your recording device is set to "Monitor of..." in your system sound settings.

## Known Issues

*   **Speaker Diarization Accuracy**: The diarization model does not provide reliable identification of speakers are requires additional time to process while not being really necessary. It has currently been disabled and will probably be removed in the future.
*   **Online Meetings**: Currently it is recommended to use headphones when following an online meeting to prevent the microphone from picking up the system audio, which can cause echo and reduce transcription quality. An echo cancellation feature is planned for a future release.
*   **Model Caching size**: When switching between several models, the program will require to download and cache the models, which will require around 20GB of free space. To prevent this use only the default recommended models.