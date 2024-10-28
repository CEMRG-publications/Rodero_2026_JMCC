#!/bin/bash

# Check if required arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <sims_folder> <first_simulation> <last_simulation>"
  exit 1
fi

# Assign arguments to variables
sims_folder=$1
first_simulation=$2
last_simulation=$3

# Ensure provided simulations range is valid
if ! [[ $first_simulation =~ ^[0-9]+$ ]] || ! [[ $last_simulation =~ ^[0-9]+$ ]]; then
  echo "Error: Simulation range must be positive integers."
  exit 1
fi

# Loop through the simulation range
for ((i=first_simulation; i<=last_simulation; i++)); do

  # Define unloading and cycle folder paths
  unloading_folder="$sims_folder/unloading_$i"
  cycle_folder="$sims_folder/cycle_$i"



# We need:
# - unloading.log
# - temp/
# - final_data/XX.vol.dat
# - reference.bpts

  # Files and folders to delete in unloading folder
  unload_files_to_delete=(
    "$unloading_folder/unloaded_mesh"
    "$unloading_folder/final_data/x.dynpt"
    "$unloading_folder/final_data/vm.igb"
  )

  # Remove specified files and directories in unloading folder
  for file in "${unload_files_to_delete[@]}"; do
    if [ -e "$file" ]; then
      rm -rf "$file"
      echo "Deleted $file"
    else
      echo "File $file does not exist, skipping."
    fi
  done

  # We need:
# - cav.XX.csv
# - x.dynpt
# - vm_act_seq.dat

  # Files to delete in cycle folder
  cycle_files_to_delete=(
    "$cycle_folder/*.Ca_i.bin"
    "$cycle_folder/*.roe"
    "$cycle_folder/ABRT*"
  )

  # Remove specified files in cycle folder
  for pattern in "${cycle_files_to_delete[@]}"; do
    if ls $pattern 1> /dev/null 2>&1; then
      rm -f $pattern
      echo "Deleted files matching pattern $pattern"
    else
      echo "No files found for pattern $pattern, skipping."
    fi
  done

done

echo "Cleanup completed for simulations $first_simulation to $last_simulation."
