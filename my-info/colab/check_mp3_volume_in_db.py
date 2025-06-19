# prompt: tell me this mp3(/content/1749534927040-melo-2.mp3) Db in volume

import subprocess
import re

input_mp3 = "/content/1749534927040-melo-2.mp3"

try:
    command = [
        "ffmpeg",
        "-i", input_mp3,
        "-af", "volumedetect",
        "-f", "null",
        "/dev/null"
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    
    # Parse the stderr output for volume information
    stderr_output = result.stderr
    
    # Look for mean volume in the output
    mean_volume_match = re.search(r'mean_volume: ([-\d.]+) dB', stderr_output)
    max_volume_match = re.search(r'max_volume: ([-\d.]+) dB', stderr_output)
    
    if mean_volume_match:
        mean_volume_db = float(mean_volume_match.group(1))
        print(f"The mean volume of '{input_mp3}' is {mean_volume_db:.2f} dB.")
    
    if max_volume_match:
        max_volume_db = float(max_volume_match.group(1))
        print(f"The max volume of '{input_mp3}' is {max_volume_db:.2f} dB.")
    
    if not mean_volume_match and not max_volume_match:
        print(f"Could not determine the volume for '{input_mp3}'.")
        print("Raw output:")
        print(stderr_output)

except FileNotFoundError:
    print("Error: ffmpeg not found. Please make sure ffmpeg is installed.")
except subprocess.CalledProcessError as e:
    print(f"Error during ffmpeg analysis: {e.stderr}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
