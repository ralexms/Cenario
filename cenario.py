#!/usr/bin/env python3
# cenario.py — CLI entry point for Cenario meeting transcription

import argparse
import os
import sys
import threading
import wave
from datetime import datetime

from core.audio_capture import AudioCapture
from core.transcriber import Transcriber
from core.exporter import Exporter
from core.summarizer import Summarizer


def _format_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _get_hf_token():
    """Load HuggingFace token from .env file or environment."""
    token = os.environ.get('HF_TOKEN')
    if token:
        return token
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('HF_TOKEN='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    return None


def _select_sources(args):
    """Auto-detect or manually select audio sources."""
    if args.source_monitor and args.source_mic:
        return args.source_monitor, args.source_mic

    active = AudioCapture.find_active_sources()
    monitor = active['active_monitor']
    mic = active['active_input']

    if args.source_monitor:
        monitor = {'device_name': args.source_monitor, 'display_name': args.source_monitor}
    if args.source_mic:
        mic = {'device_name': args.source_mic, 'display_name': args.source_mic}

    if monitor and mic:
        print(f"Monitor: {monitor['display_name']} ({monitor['device_name']})")
        print(f"Mic:     {mic['display_name']} ({mic['device_name']})")
        return monitor['device_name'], mic['device_name']

    # Fallback: interactive selection
    data = AudioCapture.get_sources_for_gui()

    if not monitor:
        monitors = data['monitors']
        if not monitors:
            print("No monitor sources found.")
            sys.exit(1)
        print("\nAvailable monitors:")
        for i, m in enumerate(monitors):
            state = f" [{m['state']}]" if m['state'] == 'RUNNING' else ""
            print(f"  {i + 1}. {m['display_name']}{state}")
        choice = input("Select monitor [1]: ").strip()
        idx = int(choice) - 1 if choice else 0
        monitor = monitors[idx]

    if not mic:
        inputs = data['inputs']
        if not inputs:
            print("No input sources found.")
            sys.exit(1)
        print("\nAvailable inputs:")
        for i, inp in enumerate(inputs):
            state = f" [{inp['state']}]" if inp['state'] == 'RUNNING' else ""
            print(f"  {i + 1}. {inp['display_name']}{state}")
        choice = input("Select input [1]: ").strip()
        idx = int(choice) - 1 if choice else 0
        mic = inputs[idx]

    print(f"\nMonitor: {monitor['display_name']}")
    print(f"Mic:     {mic['display_name']}")
    return monitor['device_name'], mic['device_name']


def _select_mic_source(args):
    """Select mic source only (for local/mono mode)."""
    if args.source_mic:
        print(f"Mic: {args.source_mic}")
        return args.source_mic

    active = AudioCapture.find_active_sources()
    mic = active['active_input']

    if mic:
        print(f"Mic: {mic['display_name']} ({mic['device_name']})")
        return mic['device_name']

    # Fallback: interactive selection
    data = AudioCapture.get_sources_for_gui()
    inputs = data['inputs']
    if not inputs:
        print("No input sources found.")
        sys.exit(1)
    print("\nAvailable inputs:")
    for i, inp in enumerate(inputs):
        state = f" [{inp['state']}]" if inp['state'] == 'RUNNING' else ""
        print(f"  {i + 1}. {inp['display_name']}{state}")
    choice = input("Select input [1]: ").strip()
    idx = int(choice) - 1 if choice else 0
    mic = inputs[idx]

    print(f"\nMic: {mic['display_name']}")
    return mic['device_name']


def _run_summarization(result, base_path):
    """Run summarization on the transcription result."""
    from core.exporter import _merge_stereo_segments
    
    print("\n--- Summarization ---")
    segments = _merge_stereo_segments(result)
    
    # Build full text with speaker labels
    lines = []
    for seg in segments:
        speaker = seg.get('speaker', 'UNKNOWN')
        text = seg.get('text', '').strip()
        lines.append(f"{speaker}: {text}")
    full_text = "\n".join(lines)
    
    if not full_text.strip():
        print("No text to summarize.")
        return

    summarizer = Summarizer()
    try:
        summary = summarizer.summarize(full_text)
        
        print("\n=== Summary ===")
        print(summary)
        print("================\n")
        
        summary_path = base_path + '_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Saved summary: {summary_path}")
        
    except Exception as e:
        print(f"Summarization failed: {e}")
    finally:
        summarizer.unload_model()


def cmd_sources(args):
    """List available audio sources."""
    data = AudioCapture.get_sources_for_gui()

    print("=== Monitors (system audio) ===")
    if not data['monitors']:
        print("  (none)")
    for m in data['monitors']:
        state = m['state']
        print(f"  [{state:10s}] {m['display_name']}")
        print(f"              {m['device_name']}")

    print("\n=== Inputs (microphones) ===")
    if not data['inputs']:
        print("  (none)")
    for inp in data['inputs']:
        state = inp['state']
        print(f"  [{state:10s}] {inp['display_name']}")
        print(f"              {inp['device_name']}")


def cmd_record(args):
    """Record with live preview, post-process, and export."""
    local_mode = getattr(args, 'local', False)

    os.makedirs('recordings', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join('recordings', f'{timestamp}.wav')

    capture = AudioCapture(sample_rate=16000)

    if local_mode:
        mic = _select_mic_source(args)
        input("\nPress Enter to start recording (local/mono)...")
        print()
        if not capture.start_recording_mono(mic, output_file):
            sys.exit(1)
    else:
        monitor, mic = _select_sources(args)
        input("\nPress Enter to start recording...")
        print()
        if not capture.start_recording_stereo(monitor, mic, output_file):
            sys.exit(1)

    stop_event = threading.Event()
    live_thread = None
    live_transcriber = None
    live_model = getattr(args, 'live_model', None) or args.model
    language = getattr(args, 'language', None)

    if not args.no_live:
        live_transcriber = Transcriber()

        def on_segment(text, start, end):
            print(f"  [{_format_time(start)} - {_format_time(end)}] {text}")

        live_thread = threading.Thread(
            target=live_transcriber.transcribe_live,
            args=(capture, live_model, args.chunk, stop_event, on_segment),
            kwargs={'language': language},
            daemon=True,
        )
        live_thread.start()
        print(f"Live preview active (model: {live_model}). Press Enter to stop recording.\n")
    else:
        print("Recording... Press Enter to stop.\n")

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    stop_event.set()
    if live_thread:
        live_thread.join(timeout=10)

    # Free live model from GPU before post-processing
    if live_transcriber:
        live_transcriber.unload_model()

    if local_mode:
        capture.stop_recording_mono()
    else:
        capture.stop_recording_stereo()
    print(f"\nSaved: {output_file}")

    # Post-processing
    print("\n--- Post-processing ---")
    transcriber = Transcriber()
    hf_token = _get_hf_token() if not args.no_diarize else None

    if local_mode:
        if hf_token:
            print("Running transcription with diarization...")
            result = transcriber.transcribe_with_diarization(output_file, model_size=args.model, hf_token=hf_token, language=language, beam_size=args.beam_size, vad_filter=args.vad_filter)
        else:
            if not args.no_diarize:
                print("No HF_TOKEN found, skipping diarization.")
            print("Running transcription...")
            result = transcriber.transcribe(output_file, model_size=args.model, language=language, beam_size=args.beam_size, vad_filter=args.vad_filter)
    else:
        if hf_token:
            print("Running transcription with diarization...")
        else:
            if not args.no_diarize:
                print("No HF_TOKEN found, skipping diarization.")
            print("Running transcription...")
        result = transcriber.transcribe_stereo(output_file, model_size=args.model, hf_token=hf_token, language=language, beam_size=args.beam_size, vad_filter=args.vad_filter)

    # Export
    base_path = os.path.splitext(output_file)[0]
    Exporter.export_all(result, base_path)
    
    if args.summarize:
        transcriber.unload_model()
        _run_summarization(result, base_path)
        
    print("\nDone!")


def cmd_transcribe(args):
    """Transcribe an existing WAV file."""
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    transcriber = Transcriber()
    hf_token = _get_hf_token() if not args.no_diarize else None

    # Detect mono vs stereo
    with wave.open(args.file, 'rb') as wf:
        channels = wf.getnchannels()

    language = getattr(args, 'language', None)

    if channels == 2:
        print("Stereo file detected, transcribing both channels...")
        result = transcriber.transcribe_stereo(args.file, model_size=args.model, hf_token=hf_token, language=language, beam_size=args.beam_size, vad_filter=args.vad_filter)
        # Print merged result
        from core.exporter import _merge_stereo_segments
        segments = _merge_stereo_segments(result)
    else:
        print("Mono file detected...")
        if hf_token:
            result = transcriber.transcribe_with_diarization(args.file, model_size=args.model, hf_token=hf_token, language=language, beam_size=args.beam_size, vad_filter=args.vad_filter)
        else:
            result = transcriber.transcribe(args.file, model_size=args.model, language=language, beam_size=args.beam_size, vad_filter=args.vad_filter)
        segments = result.get('segments', [])

    print(f"\nLanguage: {result.get('language', result.get('left_channel', {}).get('language', 'unknown'))}\n")
    for seg in segments:
        start = _format_time(seg['start'])
        end = _format_time(seg['end'])
        speaker = seg.get('speaker', '')
        prefix = f"{speaker}: " if speaker else ""
        print(f"[{start} - {end}] {prefix}{seg['text'].strip()}")
        
    if args.summarize:
        transcriber.unload_model()
        base_path = os.path.splitext(args.file)[0]
        _run_summarization(result, base_path)


def cmd_export(args):
    """Export an existing WAV file to TXT/JSON/SRT."""
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    transcriber = Transcriber()
    hf_token = _get_hf_token() if not args.no_diarize else None

    language = getattr(args, 'language', None)

    with wave.open(args.file, 'rb') as wf:
        channels = wf.getnchannels()

    if channels == 2:
        result = transcriber.transcribe_stereo(args.file, model_size=args.model, hf_token=hf_token, language=language, beam_size=args.beam_size, vad_filter=args.vad_filter)
    else:
        if hf_token:
            result = transcriber.transcribe_with_diarization(args.file, model_size=args.model, hf_token=hf_token, language=language, beam_size=args.beam_size, vad_filter=args.vad_filter)
        else:
            result = transcriber.transcribe(args.file, model_size=args.model, language=language, beam_size=args.beam_size, vad_filter=args.vad_filter)

    base_path = os.path.splitext(args.file)[0]
    fmt = args.format

    if fmt == 'all':
        Exporter.export_all(result, base_path)
    elif fmt == 'txt':
        Exporter.to_txt(result, base_path + '.txt')
    elif fmt == 'json':
        Exporter.to_json(result, base_path + '.json')
    elif fmt == 'srt':
        Exporter.to_srt(result, base_path + '.srt')
        
    if args.summarize:
        transcriber.unload_model()
        _run_summarization(result, base_path)


def main():
    parser = argparse.ArgumentParser(
        prog='cenario',
        description='Cenario — Meeting transcription with speaker diarization',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Shared flags
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('-m', '--model', default='small', help='Whisper model size (default: small)')
    common.add_argument('-l', '--language', default=None, help='Force language code (e.g. en, pt, es). Default: auto-detect')
    common.add_argument('--no-diarize', action='store_true', help='Skip diarization')
    common.add_argument('--beam-size', type=int, default=5, help='Beam size for decoding (default: 5, higher = better but slower)')
    common.add_argument('--vad-filter', action='store_true', help='Enable voice activity detection filtering')
    common.add_argument('--summarize', action='store_true', help='Summarize the transcription using a local LLM')

    # sources
    subparsers.add_parser('sources', help='List audio sources')

    # record
    p_record = subparsers.add_parser('record', parents=[common], help='Record + live preview + post-process')
    p_record.add_argument('--chunk', type=int, default=5, help='Live preview chunk size in seconds (default: 5)')
    p_record.add_argument('--live-model', default=None, help='Whisper model for live preview (default: same as --model)')
    p_record.add_argument('--no-live', action='store_true', help='Skip live preview')
    p_record.add_argument('--local', action='store_true', help='Local meeting mode (mic only, mono recording)')
    p_record.add_argument('--source-monitor', help='Override monitor source')
    p_record.add_argument('--source-mic', help='Override mic source')

    # transcribe
    p_trans = subparsers.add_parser('transcribe', parents=[common], help='Transcribe existing WAV file')
    p_trans.add_argument('file', help='Path to WAV file')

    # export
    p_export = subparsers.add_parser('export', parents=[common], help='Export WAV to TXT/JSON/SRT')
    p_export.add_argument('file', help='Path to WAV file')
    p_export.add_argument('-f', '--format', default='all', choices=['txt', 'json', 'srt', 'all'],
                          help='Export format (default: all)')

    args = parser.parse_args()

    commands = {
        'sources': cmd_sources,
        'record': cmd_record,
        'transcribe': cmd_transcribe,
        'export': cmd_export,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
