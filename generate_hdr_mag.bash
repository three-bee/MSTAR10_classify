#!/bin/bash
# Generates *.hdr and *.mag header files of raw MSTAR images in RAW_IMAGE_FOLDER_PATH, via the script MSTAR2RAW_SCRIPT_PATH.

# Usage:
# bash auto_dataset.bash RAW_IMAGE_FOLDER_PATH MSTAR2RAW_SCRIPT_PATH
cd "$1"
for file in "$1"/*; do
    ext="${file##*.}"
    if [[ $ext == JPG ]]; then
        continue
    elif [[ $ext == jpeg ]]; then
        continue
    elif [[ $ext == HTM ]]; then
        continue
    else
        "$2" "$file" 1
        echo "$file"
    fi
done