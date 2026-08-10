import argparse
import os
import numpy as np
import json
import tqdm

from common.mesh_io import read_tets


def electrophysiology_all_cycle_output(datafolder,
								   output_folder,
								   elem_file,
								   tags,
								   basename="cycle_",
								   output_file="Y_EP.txt"):

	mask = np.loadtxt(datafolder+"/output_mask.txt",dtype=int)
	idx_ok = np.where(mask>=0)[0]

	print('Reading mesh elem file...')
	elem = read_tets(elem_file)
	print('Done.')

	if "ventricles" in tags:
		check =  any(t in tags for t in ["lv","rv"])
		if check:
			raise Exception('If you want to set up the "ventricles" label, you should not set the "lv" and the "rv".')
		else:
			ventricle_tags = tags["ventricles"]

	elif (all(t in tags for t in ["lv","rv"])):
		ventricle_tags = tags["lv"]+tags["rv"]

	else:
		raise Exception("You haven't set the tags for neither the ventricles nor the lv and the rv.")

	if "fast_endo" in tags:
		check =  any(t in tags for t in ["fast_endo_lv","fast_endo_rv","fast_endo_sv"])
		if check:
			raise Exception('If you want to set up the "fast_endo" label, you should not set the "fast_endo_lv", "fast_endo_rv" and the "fast_endo_sv".')
		else:
			fec_tags = tags["fast_endo"]

	elif all(t in tags for t in ["fast_endo_lv","fast_endo_rv","fast_endo_sv"]):
		fec_tags = tags["fast_endo_lv"]+tags["fast_endo_rv"]+tags["fast_endo_sv"]

	else:
		raise Exception("You haven't set the tags for neither the fec nor the lv,rv and sv fec.")

	ventricle_tags += fec_tags
	print('Ventricles tags: ')
	for t in ventricle_tags:
		print(str(t))

	atria_tags = tags["atria"]+tags["bachmann_bundle"]
	print('Atria tags: ')
	for t in atria_tags:
		print(str(t))

	V_EIDX = np.where(np.isin(elem[:,-1],ventricle_tags)==1)[0]
	A_EIDX = np.where(np.isin(elem[:,-1],atria_tags)==1)[0]

	V_VTX = np.unique(elem[V_EIDX,0:4].flatten())
	A_VTX = np.unique(elem[A_EIDX,0:4].flatten())

	output = np.zeros((idx_ok.shape[0],2))

	t = tqdm.trange(len(range(idx_ok.shape[0])), desc='Bar desc', leave=True,colour='#FDFD96')
	for i in t:
		t.set_description('Simulation '+basename+str(idx_ok[i])+'...')

		folder = output_folder+'/'+basename+str(idx_ok[i])

		AT=np.loadtxt(folder+"/vm_act_seq.dat",dtype=float)
		if (np.min(AT[V_VTX]<0)):
			raise Exception("The ventricles contain a negative activation time.")
		if (np.min(AT[A_VTX]<0)):
			raise Exception("The atria contain a negative activation time.")
			
		output[i,0] = np.max(AT[A_VTX])-np.min(AT[A_VTX])
		output[i,1] = np.max(AT[V_VTX])-np.min(AT[V_VTX])

	np.savetxt(output_file,output,fmt="%g")



def main(args):

    basefolder        = args.basefolder
    simulations_folder = f"{basefolder}/simulations"
    data_folder        = f"{basefolder}/data"
    output_folder      = f"{basefolder}/output"
    elem_file = args.elem_file

    os.makedirs(output_folder, exist_ok=True)
    
    with open(f"{basefolder}/json_files/tags.json","r") as f:
        tags = json.load(f)


    electrophysiology_all_cycle_output(datafolder = output_folder,
								   output_folder = simulations_folder,
								   elem_file     = elem_file,
								   tags          = tags,
								   basename      = "cycle_",
								   output_file   = f"{data_folder}/Y_EP_only.txt")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate a pdf and a txt file showing the simulations that worked. It also plots the simulations that crashed and the ones who didn't in the parameter space and plots all the pv loops for the ones that worked.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default=os.path.join(os.environ.get("DATA_ROOT", ""), "simulations"),
                        help='Path to the folder where the simulations, data, and figure folders are.')
    parser.add_argument('--elem_file', type=str, help="Path to the elem file of the mesh to compute the activation times.", required=True)

    args = parser.parse_args()

    main(args)
