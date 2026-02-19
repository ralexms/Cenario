#!/usr/bin/env python3
# gui/app.py — Flask web interface for Cenario

import os
import sys
import json
import wave
import webbrowser
import threading
import traceback
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response

# Add parent dir to path so we can import core modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Relocatable data directory (recordings output)
DATA_DIR = os.environ.get('CENARIO_DATA_DIR', os.path.join(BASE_DIR, 'recordings'))

from core.audio_capture import AudioCapture
from core.transcriber import Transcriber
from core.exporter import Exporter
from core.summarizer import Summarizer

app = Flask(__name__)

# --- Recording state ---
_capture = None
_transcriber = None  # live transcriber
_live_queue = []  # list-based for reconnection support
_live_queue_lock = threading.Lock()
_stop_event = threading.Event()
_live_thread = None
_recording_mode = None
_output_file = None
_preview_only = False
_live_segments = [] # Store all live segments for export

# --- Post-processing state ---
_post = {
    'status': 'idle',  # idle, transcribing, diarizing, done, error
    'thread': None,
    'events': [],  # all progress events (list for reconnection support)
    'segments': [],  # transcription segments only
    'result': None,  # final transcription result
    'file': None,
    'error': None,
    'duration': 0,  # audio duration in seconds
}

# --- Summarization state ---
_summary = {
    'status': 'idle', # idle, summarizing, done, error
    'thread': None,
    'result': None,
    'error': None,
    'file': None,
    'stream_queue': [],
    'stream_lock': threading.Lock()
}


def _get_hf_token():
    """Load HuggingFace token from .env file or environment."""
    token = os.environ.get('HF_TOKEN')
    if token:
        return token
    # Check .env in BASE_DIR, then parent (install root)
    for env_dir in [BASE_DIR, os.path.dirname(BASE_DIR)]:
        env_path = os.path.join(env_dir, '.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN='):
                        return line.split('=', 1)[1].strip().strip('"').strip("'")
    return None


def _reset_post_state():
    """Reset post-processing state for a new run."""
    _post['status'] = 'idle'
    _post['events'] = []
    _post['segments'] = []
    _post['result'] = None
    _post['error'] = None
    _post['duration'] = 0

def _reset_summary_state():
    """Reset summarization state for a new run."""
    _summary['status'] = 'idle'
    _summary['result'] = None
    _summary['error'] = None
    _summary['file'] = None
    with _summary['stream_lock']:
        _summary['stream_queue'] = []


# ---- Routes ----

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sources')
def api_sources():
    data = AudioCapture.get_sources_for_gui()
    return jsonify(data)


@app.route('/api/recordings')
def api_recordings():
    """List WAV files in recordings directory."""
    rec_dir = DATA_DIR
    if not os.path.exists(rec_dir):
        return jsonify([])
    files = []
    for f in sorted(os.listdir(rec_dir), reverse=True):
        if f.endswith('.wav'):
            path = os.path.join(rec_dir, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            files.append({'name': f, 'path': path, 'size_mb': round(size_mb, 1)})
    return jsonify(files)

@app.route('/api/folder_files')
def api_folder_files():
    """List WAV files in a specific folder."""
    folder = request.args.get('folder', '')
    if not folder or not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify([])
    files = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.wav'):
            path = os.path.join(folder, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            files.append({'name': f, 'path': path, 'size_mb': round(size_mb, 1)})
    return jsonify(files)

@app.route('/api/folders')
def api_folders():
    """List subfolders in the specified base directory (defaults to recordings)."""
    base = request.args.get('base', '').strip()
    if base:
        if os.path.isabs(base):
            rec_dir = base
        else:
            rec_dir = os.path.join(BASE_DIR, base)
    else:
        rec_dir = DATA_DIR
    if not os.path.exists(rec_dir):
        return jsonify([])
    folders = []
    for f in sorted(os.listdir(rec_dir), reverse=True):
        path = os.path.join(rec_dir, f)
        if os.path.isdir(path):
            folders.append({'name': f, 'path': path})
    return jsonify(folders)

@app.route('/api/browse_folder')
def api_browse_folder():
    """Open a folder selection dialog on the server (local machine)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw() # Hide the main window
        root.attributes('-topmost', True) # Bring dialog to front
        
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        root.destroy()
        
        return jsonify({'path': folder_path})
    except Exception as e:
        print(f"Error opening folder dialog: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/record/start', methods=['POST'])
def api_record_start():
    global _capture, _transcriber, _live_thread, _stop_event
    global _recording_mode, _output_file, _preview_only, _live_segments

    if _capture and _capture.recording_process:
        return jsonify({'error': 'Recording already in progress'}), 400

    body = request.get_json(force=True)
    mode = body.get('mode', 'stereo')
    monitor = body.get('monitor')
    mic = body.get('mic')
    live_model = body.get('live_model', 'tiny')
    chunk_seconds = body.get('chunk', 5)
    meeting_name = body.get('meeting_name', '').strip()
    output_folder = body.get('output_folder', '').strip() or 'recordings'
    _preview_only = body.get('preview_only', False)
    language = body.get('language') or None

    if mode == 'stereo' and (not monitor or not mic):
        return jsonify({'error': 'Stereo mode requires both monitor and mic sources'}), 400
    if mode == 'mono' and not mic:
        return jsonify({'error': 'Mono mode requires a mic source'}), 400

    # Build output path
    if os.path.isabs(output_folder):
        base_rec_dir = output_folder
    else:
        base_rec_dir = os.path.join(BASE_DIR, output_folder)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Replace spaces with underscores in meeting name
    safe_meeting_name = meeting_name.replace(' ', '_')
    
    folder_name = f'{safe_meeting_name}_{timestamp}' if safe_meeting_name else f'{timestamp}'
    
    if _preview_only:
        import tempfile
        fd, _output_file = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    else:
        rec_dir = os.path.join(base_rec_dir, folder_name)
        os.makedirs(rec_dir, exist_ok=True)

        filename = f'{folder_name}.wav'
        _output_file = os.path.join(rec_dir, filename)

    _capture = AudioCapture(sample_rate=16000)
    _recording_mode = mode

    if mode == 'mono':
        ok = _capture.start_recording_mono(mic, _output_file)
    else:
        ok = _capture.start_recording_stereo(monitor, mic, _output_file)

    if not ok:
        return jsonify({'error': 'Failed to start recording'}), 500

    # Start live transcription thread
    _stop_event = threading.Event()
    with _live_queue_lock:
        _live_queue.clear()
    
    _live_segments = [] # Clear previous live segments

    _transcriber = Transcriber()

    def on_segment(text, start, end):
        entry = {'text': text, 'start': start, 'end': end}
        with _live_queue_lock:
            _live_queue.append(entry)
            _live_segments.append(entry)

    _live_thread = threading.Thread(
        target=_transcriber.transcribe_live,
        args=(_capture, live_model, chunk_seconds, _stop_event, on_segment),
        kwargs={'language': language},
        daemon=True,
    )
    _live_thread.start()

    return jsonify({
        'status': 'recording', 'file': _output_file,
        'mode': mode, 'preview_only': _preview_only,
    })


@app.route('/api/record/stop', methods=['POST'])
def api_record_stop():
    global _capture, _transcriber, _live_thread, _recording_mode, _output_file, _preview_only

    if not _capture or not _capture.recording_process:
        return jsonify({'error': 'No recording in progress'}), 400

    _stop_event.set()
    if _live_thread:
        _live_thread.join(timeout=10)
        _live_thread = None

    # Free live transcriber model from GPU before post-processing
    if _transcriber:
        _transcriber.unload_model()
        _transcriber = None

    if _recording_mode == 'mono':
        save_ok = _capture.stop_recording_mono()
    else:
        save_ok = _capture.stop_recording_stereo()

    result_file = _output_file
    was_preview = _preview_only
    _capture = None
    _recording_mode = None
    _preview_only = False

    if was_preview and result_file and os.path.exists(result_file):
        os.remove(result_file)
        return jsonify({'status': 'stopped', 'preview_only': True})

    resp = {'status': 'stopped', 'file': result_file}
    if not save_ok:
        resp['warning'] = 'Recording stopped but audio file may not have been saved correctly'
    return jsonify(resp)

@app.route('/api/record/export', methods=['POST'])
def api_record_export():
    """Export the live transcription segments to file(s)."""
    global _live_segments, _output_file
    
    body = request.get_json(force=True)
    formats = body.get('formats', [])
    
    if not _live_segments:
        return jsonify({'error': 'No live transcription data available'}), 400
    
    if not _output_file:
        # Fallback if preview only or no file set
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = os.path.join(DATA_DIR, f'live_export_{timestamp}')
    else:
        base_path = os.path.splitext(_output_file)[0] + '_live'

    # Construct a result object compatible with Exporter
    # Live segments don't have speaker info usually, or just one speaker
    result = {'segments': _live_segments}
    
    exported = []
    try:
        if 'txt' in formats:
            Exporter.to_txt(result, base_path + '.txt')
            exported.append(base_path + '.txt')
        if 'json' in formats:
            Exporter.to_json(result, base_path + '.json')
            exported.append(base_path + '.json')
        if 'srt' in formats:
            Exporter.to_srt(result, base_path + '.srt')
            exported.append(base_path + '.srt')
            
        return jsonify({'status': 'exported', 'files': exported})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/live')
def api_live():
    """SSE stream of live transcription segments. List-based for reconnection."""
    def generate():
        pos = 0
        while True:
            with _live_queue_lock:
                new_items = _live_queue[pos:]
                pos = len(_live_queue)

            for item in new_items:
                yield f"data: {json.dumps(item)}\n\n"

            if _stop_event.is_set():
                # Drain remaining
                with _live_queue_lock:
                    remaining = _live_queue[pos:]
                for item in remaining:
                    yield f"data: {json.dumps(item)}\n\n"
                yield "data: {\"done\": true}\n\n"
                break

            import time
            time.sleep(0.5)
            yield ": keepalive\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---- Post-processing ----

@app.route('/api/postprocess/start', methods=['POST'])
def api_postprocess_start():
    if _post['status'] in ('transcribing', 'diarizing'):
        return jsonify({'error': 'Post-processing already in progress'}), 400

    body = request.get_json(force=True)
    file_path = body.get('file')
    model_size = body.get('model', 'medium') # Default changed to medium
    diarize = body.get('diarize', True)
    language = body.get('language') or None
    beam_size = body.get('beam_size', 5)
    vad_filter = body.get('vad_filter', False)
    stereo_mode = body.get('stereo_mode', 'joint') # Default to joint

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 400

    hf_token = _get_hf_token() if diarize else None
    if diarize and not hf_token:
        diarize = False

    _reset_post_state()
    _post['file'] = file_path

    # Get audio duration for progress calculation
    try:
        with wave.open(file_path, 'rb') as wf:
            channels = wf.getnchannels()
            duration = wf.getnframes() / wf.getframerate()
        _post['duration'] = duration
    except Exception as e:
        return jsonify({'error': f'Cannot read WAV file: {e}'}), 400

    def _emit(event):
        _post['events'].append(event)

    def run():
        try:
            transcriber = Transcriber()

            def on_progress(event):
                evt_type = event.get('type')
                if evt_type == 'segment':
                    _post['segments'].append(event)
                    _emit(event)
                elif evt_type == 'transcription_done':
                    # Save intermediate result before diarization
                    intermediate = event.get('result')
                    if intermediate:
                        _post['result'] = intermediate
                        if diarize:
                            base = os.path.splitext(file_path)[0]
                            try:
                                Exporter.to_txt(intermediate, base + '_transcription.txt')
                                Exporter.to_json(intermediate, base + '_transcription.json')
                                _emit({'type': 'transcription_saved',
                                       'files': [base + '_transcription.txt',
                                                 base + '_transcription.json']})
                            except Exception as e:
                                print(f"Error saving intermediate: {e}")
                    _emit({'type': 'status', 'status': 'transcription_done'})
                elif evt_type == 'status':
                    _post['status'] = event.get('status', _post['status'])
                    _emit(event)
                elif evt_type == 'diarize_progress':
                    _emit(event)
                elif evt_type == 'warning':
                    _emit(event)

            _post['status'] = 'transcribing'
            _emit({'type': 'status', 'status': 'transcribing', 'duration': duration,
                   'channels': channels, 'diarize': diarize})

            if channels == 2:
                result = transcriber.transcribe_stereo(
                    file_path, model_size=model_size,
                    hf_token=hf_token if diarize else None,
                    language=language, on_progress=on_progress,
                    beam_size=beam_size, vad_filter=vad_filter,
                    stereo_mode=stereo_mode)
            else:
                if diarize and hf_token:
                    result = transcriber.transcribe_with_diarization(
                        file_path, model_size=model_size, hf_token=hf_token,
                        language=language, on_progress=on_progress,
                        beam_size=beam_size, vad_filter=vad_filter)
                else:
                    def seg_cb(seg):
                        on_progress({'type': 'segment', **seg})

                    result = transcriber.transcribe(
                        file_path, model_size=model_size, language=language,
                        on_segment=seg_cb, beam_size=beam_size,
                        vad_filter=vad_filter)
                    on_progress({'type': 'transcription_done', 'result': result})

            _post['result'] = result
            _post['status'] = 'done'
            _emit({'type': 'status', 'status': 'done'})

            transcriber.unload_model()

        except Exception as e:
            traceback.print_exc()
            _post['status'] = 'error'
            _post['error'] = str(e)
            _emit({'type': 'error', 'message': str(e)})

    t = threading.Thread(target=run, daemon=True)
    t.start()
    _post['thread'] = t

    return jsonify({'status': 'started', 'file': file_path,
                    'duration': duration, 'channels': channels})


@app.route('/api/postprocess/stream')
def api_postprocess_stream():
    """SSE stream for post-processing progress. Reconnection-safe via list-based events."""
    def generate():
        pos = 0
        while True:
            events = _post['events']
            if pos < len(events):
                for event in events[pos:]:
                    yield f"data: {json.dumps(event)}\n\n"
                pos = len(events)

                # Check if terminal event was sent
                last = events[-1]
                if last.get('status') in ('done', 'error') or last.get('type') == 'error':
                    break
            else:
                import time
                time.sleep(0.5)
                yield ": keepalive\n\n"

                # If post-processing finished while we were waiting
                if _post['status'] in ('done', 'error') and pos >= len(_post['events']):
                    yield f"data: {json.dumps({'type': 'status', 'status': _post['status']})}\n\n"
                    if _post['status'] == 'error' and _post['error']:
                        yield f"data: {json.dumps({'type': 'error', 'message': _post['error']})}\n\n"
                    break

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/postprocess/status')
def api_postprocess_status():
    """Get current post-processing status (non-SSE)."""
    return jsonify({
        'status': _post['status'],
        'file': _post['file'],
        'error': _post.get('error'),
        'segments_count': len(_post['segments']),
        'has_result': _post['result'] is not None,
        'duration': _post['duration'],
    })


@app.route('/api/export', methods=['POST'])
def api_export():
    body = request.get_json(force=True)
    file_path = body.get('file')
    fmt = body.get('format', 'all')

    if not file_path:
        return jsonify({'error': 'No file specified'}), 400
    if not _post['result']:
        return jsonify({'error': 'No transcription result available. Run post-processing first.'}), 400

    base_path = os.path.splitext(file_path)[0]
    exported = []
    result = _post['result']

    if fmt == 'all':
        Exporter.export_all(result, base_path)
        exported = [base_path + ext for ext in ('.txt', '.json', '.srt')]
    elif fmt == 'txt':
        Exporter.to_txt(result, base_path + '.txt')
        exported = [base_path + '.txt']
    elif fmt == 'json':
        Exporter.to_json(result, base_path + '.json')
        exported = [base_path + '.json']
    elif fmt == 'srt':
        Exporter.to_srt(result, base_path + '.srt')
        exported = [base_path + '.srt']

    return jsonify({'status': 'exported', 'files': exported})


@app.route('/api/folder_transcriptions')
def api_folder_transcriptions():
    """List transcription files (.json, .srt, .txt) in a specific folder."""
    folder = request.args.get('folder', '')
    if not folder or not os.path.exists(folder) or not os.path.isdir(folder):
        return jsonify([])
    extensions = {'.json', '.srt', '.txt'}
    files = []
    for f in sorted(os.listdir(folder)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in extensions:
            continue
        path = os.path.join(folder, f)
        if not os.path.isfile(path):
            continue
        size_kb = os.path.getsize(path) / 1024
        files.append({'name': f, 'path': path, 'size_kb': round(size_kb, 1)})
    return jsonify(files)

@app.route('/api/transcriptions')
def api_transcriptions():
    """List available transcription JSON files across known directories."""
    # Scan the default recordings dir plus any custom output folder from the UI
    dirs_to_scan = set()
    default_dir = DATA_DIR
    dirs_to_scan.add(default_dir)

    # Include the folder from the output folder input if provided
    folder = request.args.get('folder', '').strip()
    if folder:
        if os.path.isabs(folder):
            dirs_to_scan.add(folder)
        else:
            dirs_to_scan.add(os.path.join(BASE_DIR, folder))

    # Also include the folder of the last post-processed file
    if _post['file']:
        dirs_to_scan.add(os.path.dirname(_post['file']))

    # Also scan subdirectories of recordings
    if os.path.exists(default_dir):
        for item in os.listdir(default_dir):
            item_path = os.path.join(default_dir, item)
            if os.path.isdir(item_path):
                dirs_to_scan.add(item_path)

    files = []
    seen_paths = set()
    for rec_dir in dirs_to_scan:
        if not os.path.exists(rec_dir):
            continue
        for f in os.listdir(rec_dir):
            if not f.endswith('.json'):
                continue
            path = os.path.join(rec_dir, f)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            # Quick check: must contain transcription data
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    continue
                if 'segments' not in data and 'left_channel' not in data:
                    continue
            except Exception:
                continue
            size_kb = os.path.getsize(path) / 1024
            files.append({'name': f, 'path': path, 'size_kb': round(size_kb, 1)})

    files.sort(key=lambda x: x['name'], reverse=True)
    return jsonify(files)


@app.route('/api/speakers/load', methods=['POST'])
def api_speakers_load():
    """Load a transcription JSON and return unique speakers with sample text."""
    body = request.get_json(force=True)
    file_path = body.get('file')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 400

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            transcription = json.load(f)
    except Exception as e:
        return jsonify({'error': f'Cannot read JSON: {e}'}), 400

    # Collect all segments (handles both mono and stereo formats)
    all_segments = []
    if 'left_channel' in transcription:
        for seg in transcription.get('left_channel', {}).get('segments', []):
            all_segments.append(seg)
        for seg in transcription.get('right_channel', {}).get('segments', []):
            all_segments.append(seg)
    else:
        all_segments = transcription.get('segments', [])

    # Extract unique speakers with sample text
    speakers = {}
    for seg in all_segments:
        spk = seg.get('speaker', '')
        if not spk:
            continue
        if spk not in speakers:
            speakers[spk] = {'label': spk, 'samples': []}
        if len(speakers[spk]['samples']) < 3:
            text = seg.get('text', '').strip()
            if text:
                speakers[spk]['samples'].append(text)

    return jsonify({
        'file': file_path,
        'speakers': list(speakers.values()),
        'segment_count': len(all_segments),
    })


@app.route('/api/speakers/rename', methods=['POST'])
def api_speakers_rename():
    """Rename speakers in a transcription JSON and re-export all formats."""
    body = request.get_json(force=True)
    file_path = body.get('file')
    mapping = body.get('mapping', {})  # {"SPEAKER_00": "Alice", ...}

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 400

    if not mapping:
        return jsonify({'error': 'No speaker mapping provided'}), 400

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            transcription = json.load(f)
    except Exception as e:
        return jsonify({'error': f'Cannot read JSON: {e}'}), 400

    # Apply renaming to all segments
    def rename_segments(segments):
        for seg in segments:
            spk = seg.get('speaker', '')
            if spk in mapping and mapping[spk].strip():
                seg['speaker'] = mapping[spk].strip()

    if 'left_channel' in transcription:
        rename_segments(transcription.get('left_channel', {}).get('segments', []))
        rename_segments(transcription.get('right_channel', {}).get('segments', []))
    else:
        rename_segments(transcription.get('segments', []))

    # Save updated JSON back to same file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(transcription, f, indent=2, ensure_ascii=False)

    # Re-export TXT and SRT with same base name
    # Transcription files are named {base}_transcription.json
    # Export files are named {base}.txt, {base}.json, {base}.srt
    if file_path.endswith('_transcription.json'):
        base_path = file_path[:-len('_transcription.json')]
    else:
        base_path = os.path.splitext(file_path)[0]

    exported = [file_path]
    try:
        Exporter.to_txt(transcription, base_path + '_transcription.txt')
        exported.append(base_path + '_transcription.txt')
    except Exception as e:
        print(f"Error exporting TXT: {e}")

    try:
        Exporter.to_srt(transcription, base_path + '_transcription.srt')
        exported.append(base_path + '_transcription.srt')
    except Exception as e:
        print(f"Error exporting SRT: {e}")

    # Also update the in-memory result if it matches
    if _post['result'] is not None and _post['file']:
        post_base = os.path.splitext(_post['file'])[0]
        if base_path == post_base:
            _post['result'] = transcription

    return jsonify({'status': 'renamed', 'files': exported})


# ---- Summarization ----

@app.route('/api/summarize/start', methods=['POST'])
def api_summarize_start():
    if _summary['status'] == 'summarizing':
        return jsonify({'error': 'Summarization already in progress'}), 400

    body = request.get_json(force=True)
    file_path = body.get('file')
    model_id = body.get('model', 'Qwen/Qwen2.5-0.5B-Instruct')
    detail_level = body.get('detail_level', 'concise')
    quantization = body.get('quantization', '4')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 400

    _reset_summary_state()
    _summary['file'] = file_path
    _summary['status'] = 'summarizing'

    def run():
        summarizer = Summarizer(model_id=model_id, quantization=quantization)
        try:
            # Load transcription based on file type
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    transcription = json.load(f)
                from core.exporter import _merge_stereo_segments
                segments = _merge_stereo_segments(transcription)
                lines = []
                for seg in segments:
                    speaker = seg.get('speaker', 'UNKNOWN')
                    text = seg.get('text', '').strip()
                    lines.append(f"{speaker}: {text}")
                full_text = "\n".join(lines)
            else:
                # .txt and .srt — read as plain text
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
            
            if not full_text.strip():
                raise ValueError("No text to summarize")

            def stream_cb(chunk):
                with _summary['stream_lock']:
                    _summary['stream_queue'].append(chunk)

            # Summarize
            summary_text = summarizer.summarize(full_text, detail_level=detail_level, stream_callback=stream_cb)
            _summary['result'] = summary_text
            
            # Save summary
            if file_path.endswith('_transcription.json'):
                base_path = file_path[:-len('_transcription.json')]
            else:
                base_path = os.path.splitext(file_path)[0]
                
            summary_path = base_path + '_summary.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
                
            _summary['status'] = 'done'
            
        except Exception as e:
            traceback.print_exc()
            _summary['status'] = 'error'
            _summary['error'] = str(e)
        finally:
            summarizer.unload_model()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    _summary['thread'] = t

    return jsonify({'status': 'started', 'file': file_path})

@app.route('/api/summarize/stream')
def api_summarize_stream():
    """SSE stream for summarization text generation."""
    def generate():
        pos = 0
        while True:
            with _summary['stream_lock']:
                new_items = _summary['stream_queue'][pos:]
                pos = len(_summary['stream_queue'])
            
            for item in new_items:
                yield f"data: {json.dumps({'text': item})}\n\n"

            if _summary['status'] in ('done', 'error'):
                # Drain remaining
                with _summary['stream_lock']:
                    remaining = _summary['stream_queue'][pos:]
                for item in remaining:
                    yield f"data: {json.dumps({'text': item})}\n\n"
                
                if _summary['status'] == 'done':
                    yield f"data: {json.dumps({'done': True})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': _summary['error']})}\n\n"
                break

            import time
            time.sleep(0.1)
            yield ": keepalive\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/summarize/export_markdown', methods=['POST'])
def api_summarize_export_markdown():
    body = request.get_json(force=True)
    text = body.get('text')
    file_path = body.get('file') # original transcription file path to derive name

    if not text:
        return jsonify({'error': 'No text to export'}), 400
    
    if file_path:
        if file_path.endswith('_transcription.json'):
            base_path = file_path[:-len('_transcription.json')]
        else:
            base_path = os.path.splitext(file_path)[0]
        md_path = base_path + '_summary.md'
    else:
        # Fallback
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_path = os.path.join(DATA_DIR, f'summary_{timestamp}.md')

    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return jsonify({'status': 'exported', 'file': md_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update', methods=['POST'])
def api_update():
    """Trigger the updater script."""
    try:
        import subprocess
        # Run updater.py in a separate process
        updater_path = os.path.join(BASE_DIR, 'updater.py')
        subprocess.Popen([sys.executable, updater_path], cwd=BASE_DIR)
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('CENARIO_PORT', 5000))
    debug = os.environ.get('CENARIO_DEBUG', '0') == '1'

    if not os.environ.get('CENARIO_NO_BROWSER'):
        def _open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f'http://localhost:{port}')
        threading.Thread(target=_open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True, use_reloader=False)
