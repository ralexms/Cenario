import whisperx
import time

# Load model
print("Loading WhisperX model...")
model = whisperx.load_model("small", device="cuda", compute_type="float16")

# Load your audio file
print("Loading audio...")
audio = whisperx.load_audio("harvard.wav")  # Change this path

# Time it
print("Transcribing...")
start = time.time()
result = model.transcribe(audio, batch_size=8)
elapsed = time.time() - start

audio_duration = len(audio) / 16000
speed_ratio = audio_duration / elapsed

print(f"\n=== RESULTS ===")
print(f"Audio duration: {audio_duration:.1f}s")
print(f"Transcription time: {elapsed:.2f}s")
print(f"Speed: {speed_ratio:.1f}x realtime")
print(f"Real-time capable: {'✓ YES' if speed_ratio > 1.0 else '✗ NO'}")

if result.get('segments'):
    print(f"\nTranscript preview:")
    for seg in result['segments'][:3]:  # First 3 segments
        print(f"  [{seg['start']:.1f}s] {seg['text']}")