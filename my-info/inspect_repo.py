import os
import sys
import torch
import torchaudio

# Make sure we're in the OpenVoice directory
# This path handling makes the script more robust
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the correct class from api.py
from openvoice.api import ToneColorConverter

# Step 1: Define the audio paths
base_audio_path = "1746537433593-melo-1.wav"  # Your MeloTTS output
reference_audio_path = "leijun.wav"  # Your voice sample

# Create output directory if it doesn't exist
os.makedirs("converted", exist_ok=True)

# Step 2: Initialize the tone color converter
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
tone_color_converter = ToneColorConverter(device=device)

# Step 3: Perform voice conversion
print("Converting voice, please wait...")
tone_color_converter.convert(
    source_path=base_audio_path,         # MeloTTS output
    target_path=reference_audio_path,    # Your voice sample
    output_path="converted/my_custom_voice_output.wav"
)

print("Conversion complete! Output saved to: converted/my_custom_voice_output.wav")