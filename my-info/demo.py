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

# [Chinese]
text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
model = TTS(language='ZH', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'zh.wav'
model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)

# [English]
# text = "The sun sets behind the mountains, casting long shadows across the valley."
# model = TTS(language='EN', device=device)
# speaker_ids = model.hps.data.spk2id
# output_path = 'en.wav'

# # model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed) #[work, but Chinese accent]
# # model.tts_to_file(text, speaker_ids['EN-BR'], output_path, speed=speed) # [work]
# # model.tts_to_file(text, speaker_ids['EN_INDIA'], output_path, speed=speed) # [work, but Indian accent]
# # model.tts_to_file(text, speaker_ids['EN-AU'], output_path, speed=speed) # [work]
# model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=speed) # [work, good. it is American accent]
