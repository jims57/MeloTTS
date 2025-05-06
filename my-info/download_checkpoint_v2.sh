#!/bin/bash

# URL of the checkpoint zip file
CHECKPOINT_URL="https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
# Output filename
ZIP_FILE="checkpoints_v2_0417.zip"

echo "Downloading OpenVoice checkpoints v2..."
curl -L $CHECKPOINT_URL -o $ZIP_FILE

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
    
    echo "Extracting files..."
    unzip -o $ZIP_FILE
    
    # Check if extraction was successful
    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully."
        echo "Cleaning up..."
        rm $ZIP_FILE
        echo "Done! Checkpoints are ready to use."
    else
        echo "Error: Failed to extract the zip file."
        exit 1
    fi
else
    echo "Error: Failed to download the checkpoint file."
    exit 1
fi