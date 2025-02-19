# Paths


folder_experiment_name=HCM/1/scenarios/53_more_samples
base_folder_local=/media/croderog/SeagateExpansionDrive/$folder_experiment_name

# Parameters
hpc="archer2"
first_simulation=800
last_simulation=1499

if [ "$hpc" == "imperial" ]; then
    username="croderog"
    hpc_address="login.hpc.imperial.ac.uk"
    hpc_root_path="/rds/general/user/croderog/home"
elif [ "$hpc" == "archer2" ]; then
    username="jsolisle"
    hpc_address="login.archer2.ac.uk"
    hpc_root_path="/scratch-nvme/e348/e348/"$username
fi

mkdir -p $base_folder_local/files_to_transfer/slrm
mkdir -p $base_folder_local/files_to_transfer/states

##### SLRM files

for ((i=$first_simulation; i<=$last_simulation; i++)); do
    cp $base_folder_local/slrm/unloading_$i.slrm $base_folder_local/files_to_transfer/slrm/.
    cp $base_folder_local/slrm/cycle_$i.slrm $base_folder_local/files_to_transfer/slrm/.
done

#### SV files

source_folder=$base_folder_local/SS
destination_folder=$base_folder_local/files_to_transfer/states

# Iterate over subdirectories inside source_folder
for category_folder in "$source_folder"/*/; do
    category_name=$(basename "$category_folder")
    
    # Exclude "param" folder
    if [ "$category_name" != "param" ]; then
        # Create destination category folder
        mkdir -p "$destination_folder/$category_name"
        
        # Iterate over sub-subdirectories inside source_folder (0 to 99)
        for ((i=$first_simulation; i<=$last_simulation; i++)); do
            subfolder="$category_folder/$i"
            folder_name=$(basename "$subfolder")
            
            # Check if .sv files exist in the source folder
            sv_files=("$subfolder"/*.sv)
            if [ -n "$(shopt -s nullglob; echo $sv_files)" ]; then
                # Create destination folder for states
                mkdir -p "$destination_folder/$category_name/$folder_name"
                
                # Copy .sv files to destination folder
                cp "$subfolder"/*.sv "$destination_folder/$category_name/$folder_name/"
            fi
        done
    fi
done

# Create a tar archive with the slrm and states folders
tar -czf "$base_folder_local/files_to_transfer.tar.gz" -C "$base_folder_local" files_to_transfer

# Clean
rm -rf $base_folder_local/files_to_transfer/

# Transfer the archive using scp
scp -r $base_folder_local/files_to_transfer.tar.gz $username@$hpc_address:$hpc_root_path/.

# Clean
rm $base_folder_local/files_to_transfer.tar.gz 
