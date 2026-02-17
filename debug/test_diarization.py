# test_diarization.py

import os
from dotenv import load_dotenv
from core.transcriber import Transcriber

load_dotenv()

transcriber = Transcriber(device="cuda", compute_type="float16")

HF_TOKEN = os.environ["HF_TOKEN"]

print("=== Testing Transcription + Diarization ===\n")

result = transcriber.transcribe_with_diarization(
    audio_file="../stereo_test.wav",
    model_size="small",
    hf_token=HF_TOKEN
)

print("\n=== Transcription with Speakers ===")
for segment in result['segments']:
    speaker = segment.get('speaker', 'UNKNOWN')
    text = segment['text']
    start = segment['start']
    end = segment['end']
    print(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")