import os
import time
from dotenv import load_dotenv
from core.transcriber import Transcriber

load_dotenv()

transcriber = Transcriber(device="cuda", compute_type="float16")

HF_TOKEN = os.environ["HF_TOKEN"]

print("=== Testing Stereo Transcription + Diarization ===\n")

audio_file = "stereo_test.wav"

print(f"Transcribing: {audio_file}")
print("Processing left (meeting audio) and right (microphone) separately...\n")

start_time = time.time()
result = transcriber.transcribe_stereo(audio_file, model_size="small", hf_token=HF_TOKEN)
elapsed = time.time() - start_time

print(f"\n✓ Complete in {elapsed:.2f}s\n")

print("=" * 60)
print("LEFT CHANNEL (Meeting Audio)")
print("=" * 60)
for segment in result['left_channel']['segments']:
    speaker = segment.get('speaker', 'UNKNOWN')
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {speaker}: {segment['text']}")

print("\n" + "=" * 60)
print("RIGHT CHANNEL (Your Microphone)")
print("=" * 60)
for segment in result['right_channel']['segments']:
    speaker = segment.get('speaker', 'UNKNOWN')
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {speaker}: {segment['text']}")