# Set your cluster usernames before running
archer2_user="<archer2_username>"
imperial_user="<imperial_username>"

archer2_root_path="/work/e348/e348/$archer2_user"
imperial_root_path="/rds/general/user/$imperial_user/home"
local_root_path="${DATA_ROOT:?Set DATA_ROOT to your local data directory}"
folder_experiment_name="rodero_healthy/h11/scenarios/5"
first_simulation=50
last_simulation=99
mesh_name="myocardium_AV_FEC_BB_lvrv"

total_simulations=$((last_simulation - first_simulation + 1))
progress=0

ssh -t $archer2_user@login.archer2.ac.uk /bin/bash -s "$archer2_root_path" "$folder_experiment_name" "$total_simulations" "$first_simulation" "$last_simulation" "$mesh_name" << 'EOF'
	archer2_root_path="$1"
	folder_experiment_name="$2"
	total_simulations="$3"
	first_simulation="$4"
	last_simulation="$5"
	mesh_name="$6"
	
	mkdir -p "$archer2_root_path/$folder_experiment_name/unloaded_to_transfer"

	start_time=$(date +%s)

	for ((i=first_simulation; i<=last_simulation; i++)); do
		cp "$archer2_root_path/$folder_experiment_name/simulations/unloaded/${mesh_name}_unloaded_$i.bpts" "$archer2_root_path/$folder_experiment_name/unloaded_to_transfer/" &&
		((progress++))
		if ((progress > 0)); then
			percentage=$((progress * 100 / total_simulations))
			current_time=$(date +%s)
			elapsed_time=$((current_time - start_time))
			eta=$((elapsed_time * (total_simulations - progress) / progress))
			echo -ne "Preparing to scp unloaded files... $percentage% ETA: $eta seconds \r"
		fi
	done

	# Tar and gzip the folder
	cd "$archer2_root_path/$folder_experiment_name"

	echo -e "\nCompressing folder (might take a while)..."
	tar -czf unloaded_to_transfer.tar.gz unloaded_to_transfer
	rm -r unloaded_to_transfer

	echo -e "\nCopy and compression completed."
EOF

echo "Syncing the unloaded folder to the local machine..."

scp $archer2_user@login.archer2.ac.uk:$archer2_root_path/$folder_experiment_name/unloaded_to_transfer.tar.gz $local_root_path/$folder_experiment_name/.

echo "Syncing the unloaded folder to the imperial hpc..."

scp $local_root_path/$folder_experiment_name/unloaded_to_transfer.tar.gz $imperial_user@login.hpc.imperial.ac.uk:$imperial_root_path/.

echo "Cleaning..."

ssh -t $archer2_user@login.archer2.ac.uk << EOF
	rm -rf $archer2_root_path/$folder_experiment_name/unloaded_to_transfer.tar.gz
EOF

rm -rf $local_root_path/$folder_experiment_name/unloaded_to_transfer.tar.gz

echo "Finished."