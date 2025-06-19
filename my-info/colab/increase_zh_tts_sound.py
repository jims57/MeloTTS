# prompt: increase the sound (volume) of this mp3(/content/1748953736971-melo-1.mp3), cause its sound is too low

import subprocess
import pandas as pd
import os
from IPython.display import Audio, display

# Increase the volume of the MP3 file
input_mp3 = "/content/1748953736971-melo-1.mp3"
output_mp3_louder = "/content/1748953736971-melo-1_louder.mp3"
volume_increase_db = 13  # Adjust this value for the desired volume increase (in decibels)

# Delete the old converted MP3 file if it exists
if os.path.exists(output_mp3_louder):
    os.remove(output_mp3_louder)
    print(f"Deleted existing file: {output_mp3_louder}")

try:
    command = [
        "ffmpeg",
        "-i", input_mp3,
        "-af", f"volume={volume_increase_db}dB",
        "-c:a", "libmp3lame", # Specify the audio codec if needed, libmp3lame is common
        "-q:a", "0",          # Maintain high quality (0-9, 0 is best)
        output_mp3_louder
    ]
    subprocess.run(command, check=True)
    print(f"Successfully increased volume of '{input_mp3}' and saved to '{output_mp3_louder}'")

    # Play the original MP3 file for comparison
    print("\n🎵 Original MP3:")
    display(Audio(input_mp3))
    
    # Play the louder MP3 file directly in Colab
    print(f"\n🔊 Louder MP3 (+{volume_increase_db}dB):")
    display(Audio(output_mp3_louder))

except FileNotFoundError:
    print("Error: ffmpeg not found. Please make sure ffmpeg is installed.")
except subprocess.CalledProcessError as e:
    print(f"Error during ffmpeg volume adjustment: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
