from core.transcriber import Transcriber
import time

# Initialize transcriber
transcriber = Transcriber(device="cuda", compute_type="float16")

print("=== Testing Whisper Transcription ===\n")

# Test with the stereo recording we made earlier
audio_file = "harvard.wav"

print(f"Transcribing: {audio_file}")
print("Using 'small' model...\n")

start_time = time.time()
result = transcriber.transcribe(audio_file, model_size="small")
elapsed = time.time() - start_time

print(f"\n✓ Transcription complete in {elapsed:.2f}s\n")
print("=== Results ===")
print(f"Detected language: {result.get('language', 'unknown')}")
print(f"\nTranscript:")
for segment in result['segments']:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")