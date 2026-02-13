import torch
import whisperx

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Recommend model based on VRAM
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if vram_gb >= 10:
    print("\nRecommended Whisper model: large-v2")
elif vram_gb >= 5:
    print("\nRecommended Whisper model: medium")
else:
    print("\nRecommended Whisper model: small")