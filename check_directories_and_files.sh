#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <slrm_file>"
  exit 1
fi

slrm_file="$1"

if [ ! -f "$slrm_file" ]; then
  echo "The specified .slrm file does not exist."
  exit 1
fi

# Extract directories and files using grep
directories_and_files=($(grep -oE '/\S+' "$slrm_file"))

# Loop through the extracted items and check if they exist
for item in "${directories_and_files[@]}"; do
  if [ -e "$item" ]; then
    echo "Exists: $item"
  else
    echo "Does not exist: $item"
  fi
done

