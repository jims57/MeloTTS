# import os
# import sys
# import torch
# import torchaudio

# # Make sure we're in the OpenVoice directory
# # This path handling makes the script more robust
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

# # Import the correct class from api.py
# from openvoice.api import ToneColorConverter

# # Step 1: Define the audio paths
# base_audio_path = "1746537433593-melo-1.wav"  # Your MeloTTS output
# reference_audio_path = "leijun.wav"  # Your voice sample

# # Create output directory if it doesn't exist
# os.makedirs("converted", exist_ok=True)

# # Step 2: Initialize the tone color converter
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # Look for config.json in the resources directory
# config_path = os.path.join(current_dir, "resources", "config.json")
# if not os.path.exists(config_path):
#     # Try other potential locations
#     potential_paths = [
#         os.path.join(current_dir, "resources", "config.json"),
#         os.path.join(current_dir, "config.json"),
#         os.path.join(current_dir, "model", "config.json")
#     ]
    
#     for path in potential_paths:
#         if os.path.exists(path):
#             config_path = path
#             break
    
#     if not os.path.exists(config_path):
#         print("Config file not found! Checking available files in resources:")
#         if os.path.exists(os.path.join(current_dir, "resources")):
#             print(os.listdir(os.path.join(current_dir, "resources")))
#         else:
#             print("resources directory not found!")
#         raise FileNotFoundError(f"Could not find config.json in expected locations")

# print(f"Using config from: {config_path}")
# tone_color_converter = ToneColorConverter(device=device, config_path=config_path)

# # Step 3: Perform voice conversion
# print("Converting voice, please wait...")
# tone_color_converter.convert(
#     source_path=base_audio_path,         # MeloTTS output
#     target_path=reference_audio_path,    # Your voice sample
#     output_path="converted/my_custom_voice_output.wav"
# )

# print("Conversion complete! Output saved to: converted/my_custom_voice_output.wav")

# import os
# import torch
# from openvoice import se_extractor
# from openvoice.api import ToneColorConverter

# ckpt_converter = 'checkpoints_v2/converter'
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# output_dir = 'outputs_v2'

# tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
# tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# os.makedirs(output_dir, exist_ok=True)


import os
import torch
from openvoice.api import ToneColorConverter

# Force CPU usage to avoid CUDA library issues
device = "cpu"
output_dir = 'outputs_v2'

# Define the input files
base_audio_path = "1746539324580-melo-1.wav"  # Your MeloTTS output
output_path = f"{output_dir}/converted_voice.wav"

print(f"Using device: {device}")

# Initialize the converter with CPU
ckpt_converter = 'checkpoints_v2/converter'
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("\nLoading pre-extracted speaker embeddings...")

# Use pre-extracted source embedding (default English voice)
# These paths should match those used in the demo notebooks
source_se_path = 'checkpoints_v2/base_speakers/ses/en-default.pth'
if not os.path.exists(source_se_path):
    print(f"Source embedding not found at {source_se_path}")
    print("Looking for alternative embeddings...")
    # Try to find any available embedding
    import glob
    available_embeddings = glob.glob('checkpoints_v2/base_speakers/ses/*.pth')
    if available_embeddings:
        source_se_path = available_embeddings[0]
        print(f"Using alternative source embedding: {source_se_path}")
    else:
        raise FileNotFoundError("No source embeddings found. Please check the paths.")

# Use pre-extracted target embedding (or one from the resources directory)
target_se_path = 'checkpoints_v2/base_speakers/ses/en-us.pth'  # This will be a different voice than the source
if not os.path.exists(target_se_path):
    print(f"Target embedding not found at {target_se_path}")
    # Try to find any examples in resources
    resource_embedding = 'resources/example_reference.pth'
    if os.path.exists(resource_embedding):
        target_se_path = resource_embedding
        print(f"Using example embedding from resources: {target_se_path}")
    else:
        # Use another base speaker as target if available
        import glob
        available_embeddings = glob.glob('checkpoints_v2/base_speakers/ses/*.pth')
        if len(available_embeddings) > 1:
            # Use a different embedding than the source
            for emb in available_embeddings:
                if emb != source_se_path:
                    target_se_path = emb
                    break
            print(f"Using alternative target embedding: {target_se_path}")
        else:
            raise FileNotFoundError("No target embeddings found. Please check the paths.")

# Load the speaker embeddings
source_se = torch.load(source_se_path, map_location=device)
target_se = torch.load(target_se_path, map_location=device)

print(f"Loaded source embedding from: {source_se_path}")
print(f"Loaded target embedding from: {target_se_path}")

print("\nPerforming voice conversion...")
print(f"Output will be saved to {output_path}")

# Run the tone color converter with the correct parameters
encode_message = "custom_voice_conversion"
tone_color_converter.convert(
    audio_src_path=base_audio_path,
    src_se=source_se,
    tgt_se=target_se,
    output_path=output_path,
    tau=0.3,  # This controls the voice conversion strength
    message=encode_message
)

print(f"Conversion complete! Output saved to: {output_path}")
