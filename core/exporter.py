# core/exporter.py

import json


def _format_time_txt(seconds):
    """Format seconds as MM:SS for plain text output."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _format_time_srt(seconds):
    """Format seconds as HH:MM:SS,mmm for SRT output."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _merge_stereo_segments(transcription):
    """
    Merge left_channel and right_channel segments into a single sorted list.
    Works for both stereo results (dict with left/right) and mono results (dict with segments).
    """
    if 'left_channel' in transcription:
        segments = []
        for seg in transcription['left_channel'].get('segments', []):
            segments.append(seg)
        for seg in transcription['right_channel'].get('segments', []):
            segments.append(seg)
        segments.sort(key=lambda s: s['start'])
        return segments
    else:
        return transcription.get('segments', [])


class Exporter:

    @staticmethod
    def to_txt(transcription, output_path):
        """Export transcription to plain text with speaker labels and timestamps."""
        segments = _merge_stereo_segments(transcription)

        lines = []
        for seg in segments:
            start = _format_time_txt(seg['start'])
            end = _format_time_txt(seg['end'])
            speaker = seg.get('speaker', 'UNKNOWN')
            text = seg.get('text', '').strip()
            lines.append(f"[{start} - {end}] {speaker}: {text}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

        print(f"Exported TXT -> {output_path}")

    @staticmethod
    def to_json(transcription, output_path):
        """Export full structured transcription data to JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)

        print(f"Exported JSON -> {output_path}")

    @staticmethod
    def to_srt(transcription, output_path):
        """Export transcription to SRT subtitle format."""
        segments = _merge_stereo_segments(transcription)

        lines = []
        for i, seg in enumerate(segments, 1):
            start = _format_time_srt(seg['start'])
            end = _format_time_srt(seg['end'])
            speaker = seg.get('speaker', 'UNKNOWN')
            text = seg.get('text', '').strip()
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(f"{speaker}: {text}")
            lines.append("")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Exported SRT -> {output_path}")

    @staticmethod
    def export_all(transcription, base_path):
        """Export to all formats (TXT, JSON, SRT)."""
        Exporter.to_txt(transcription, base_path + '.txt')
        Exporter.to_json(transcription, base_path + '.json')
        Exporter.to_srt(transcription, base_path + '.srt')
