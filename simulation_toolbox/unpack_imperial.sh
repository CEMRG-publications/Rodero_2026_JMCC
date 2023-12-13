imperial_root_path="/rds/general/user/croderog/home"
mesh_folder="rodero_healthy/h11"
folder_experiment_name=$mesh_folder"/scenarios/5"

ssh -t croderog@login.hpc.imperial.ac.uk /bin/bash -s "$imperial_root_path" "$folder_experiment_name" "$mesh_folder" << 'EOF'

	imperial_root_path="$1"
	folder_experiment_name="$2"
	mesh_folder="$3"
	
	mkdir -p "$imperial_root_path/$folder_experiment_name"

	###### SLRM AND STATES
	
	tar -xzf $imperial_root_path/files_to_transfer.tar.gz 

	cp -r $imperial_root_path/files_to_transfer/slrm $imperial_root_path/$folder_experiment_name/.

	cp -r $imperial_root_path/files_to_transfer/states $imperial_root_path/$folder_experiment_name/.

	###### Unloaded mesh 

	tar -xzf $imperial_root_path/unloaded_to_transfer.tar.gz

	mkdir -p "$imperial_root_path/meshes/$mesh_folder"

	cp -r $imperial_root_path/unloaded_to_transfer/ $imperial_root_path/meshes/$mesh_folder/unloaded

EOF