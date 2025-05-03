from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cuda:0' # Changed from integer 0 to string 'cuda:0'

# Check if CUDA is available
import torch

is_cuda_available = torch.cuda.is_available()
print(f"CUDA available: {is_cuda_available}")

if is_cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {device}")
else:
    print("CUDA is not available. Using CPU for inference.")
    device = 'cpu'  # Set to CPU if CUDA not available
    print(f"Current device: {device}")


text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
model = TTS(language='ZH', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'zh.wav'
model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)