import os 
import argparse

import numpy as np
import matplotlib.pyplot as plt

from Historia.shared.design_utils import read_labels

def file_exists(full_file_path):
    if not os.path.isfile(full_file_path):
        raise Exception(f"You need to have the file {os.path.abspath(os.path.normpath(full_file_path))}")


def check_fourchamber_unloading_(simulation_folder,
								 chambers):

	converged_1 = False
	with open(simulation_folder+"unloading.log", 'r') as file:
		content = file.read()
		if "--- CONVERGED ---" in content:
			converged_1 = True

	converged_2 = False
	if not os.path.exists(simulation_folder+"temp/"):
		converged_2 = True

	converged = False
	if converged_1 and converged_2:
		converged = True

	output = [converged]
	volumes = []
	for ch in chambers:
		if converged and os.path.exists(simulation_folder+"final_data/"+ch+".vol.dat"):
			vol = np.loadtxt(simulation_folder+"final_data/"+ch+".vol.dat",dtype=float,usecols=[1])
			output.append(vol[0])
		else:
			output.append(-1)

	return output

def check_fourchamber_unloading(basefolder,
								chambers,
								start_sample=0,
								last_sample=1,
								output_file='unloaded_volumes.txt'):

	vol_unloaded = np.zeros((last_sample-start_sample+1,len(chambers)),dtype=float)
	count = 0

	for i in range(start_sample,last_sample+1):	

		output = check_fourchamber_unloading_(basefolder+'/unloading_'+str(i)+'/',
											  chambers)

		if output[0]:
			print('unloading_'+str(i)+' successful...')

			for j in range(len(chambers)):
				vol_unloaded[count,j] = output[1+j]

		else:
			print('unloading_'+str(i)+' crashed...')
			
			for j in range(len(chambers)):
				vol_unloaded[count,j] = -1

		count += 1

	np.savetxt(output_file,vol_unloaded,fmt="%g")

def main(args):

    basefolder = args.path2simulations
    path2figure = args.path2figures
    first_simulation = args.first_simulation
    last_simulation = args.last_simulation

    file_exists(os.path.join(basefolder,"../data/X_mechanics.txt"))
    file_exists(os.path.join(basefolder,"../data/xlabels_mechanics.txt"))

    os.system("mkdir -p "+ os.path.join(basefolder,"unloaded/"))
    os.system("mkdir -p "+ os.path.join(basefolder,"../figures/"))

    check_fourchamber_unloading(basefolder,
							['lv_endo','rv_endo','la_endo','ra_endo'],
							start_sample=first_simulation,
							last_sample=last_simulation,
							output_file=os.path.join(basefolder,'unloaded_volumes.txt'))    
    
    unloaded_volumes = np.loadtxt(os.path.join(basefolder,"unloaded_volumes.txt"),dtype=float)

    X = np.loadtxt(os.path.join(basefolder,"../data/X_mechanics.txt"),dtype=float)
    mask = np.sum(unloaded_volumes,axis=1)

    xlabels = read_labels(os.path.join(basefolder,"../data/xlabels_mechanics.txt"))

    idx_ok = np.where(mask!=0)[0]
    idx_notok = np.where(mask==0)[0]

    if len(X.shape) > 1:
        in_dim = X.shape[1]
		
        out_dim = in_dim
        _, axes = plt.subplots(
            nrows=out_dim,
            ncols=in_dim,
            sharex="col",
            sharey="row",
            figsize=(10,10),
        )
        for i, axis in enumerate(axes.flatten()):
            axis.scatter(X[idx_ok, i % in_dim], X[idx_ok, i // in_dim], c='green', s=1)
            axis.scatter(X[idx_notok, i % in_dim], X[idx_notok, i // in_dim], c='red', s=1)
            inf = min(X[:, i % in_dim])
            sup = max(X[:, i % in_dim])
            mean = 0.5 * (inf + sup)
            delta = sup - mean
            if i // in_dim == out_dim - 1:
                axis.set_xlabel(xlabels[i % in_dim],rotation=90)
                axis.set_xticks([])
                axis.set_xlim(left=inf - 0.3 * delta, right=sup + 0.3 * delta)
            if i % in_dim == 0:
                axis.set_yticks([])
                axis.set_ylabel(xlabels[i // in_dim])
        plt.savefig(os.path.join(path2figure, "unloaded_scatter.png"), bbox_inches="tight", dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to check which unloading simulations worked.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--path2simulations', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/h01/new_unloading/unloading_simulations")
    parser.add_argument('--path2figures', type=str, required=True)
    parser.add_argument('--first_simulation', type=int, required=False, default=0)
    parser.add_argument('--last_simulation', type=int, required=False, default=99)

    args = parser.parse_args()

    main(args)
