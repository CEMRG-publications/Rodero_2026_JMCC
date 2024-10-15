import os
import numpy as np

import argparse

from gpytGPE.utils.design import read_labels

basefolder = "./"

# DATA_folder = os.path.join(basefolder,"data")
# SAMPLING_folder = os.path.join(basefolder,"sampling")
# LHD_folder  = os.path.join(SAMPLING_folder,"LHD")
# NIMP_folder = os.path.join(SAMPLING_folder,"history_matching")

os.system("clear")

def main(args):

	basefolder = args.basefolder
	
	DATA_folder = os.path.join(basefolder,"data")

	N = 800

	# scaling the CV to patient-specific ranges
	xlabels = read_labels(os.path.join(DATA_folder,"xlabels.txt"))

	X = np.loadtxt(os.path.join(DATA_folder,"X_with_normalised_values.txt"),dtype=float)

	I_EP_file = os.path.join(DATA_folder,"I_EP_patient.txt")
	if not os.path.exists(I_EP_file):
		raise Exception(I_EP_file+" not found. Please provide the EP ranges for your patient in this file")
		
	I_EP = np.loadtxt(I_EP_file,dtype=float)

	idx_cv_v = None
	idx_cv_a = None
	for i,xl in enumerate(xlabels):
		if xl=="CV_ventricles":
			idx_cv_v = i
		if xl=="CV_atria":
			idx_cv_a = i
	CV_ventricles_norm = X[:,idx_cv_v]
	CV_atria_norm 	   = X[:,idx_cv_a]
	CV_ventricles = CV_ventricles_norm*(I_EP[0,1]-I_EP[0,0])+I_EP[0,0]
	CV_atria      = CV_atria_norm*(I_EP[1,1]-I_EP[1,0])+I_EP[1,0]
	print("-----------------------------------------------------------------------------")
	print("New CV parameter bounds : ")
	print("CV_ventricles : "+str(np.min(CV_ventricles))+" , "+str(np.max(CV_ventricles)))
	print("CV_atria  	 : "+str(np.min(CV_atria))+" , "+str(np.max(CV_atria)))
	print("-----------------------------------------------------------------------------")
	X[:,idx_cv_v] = CV_ventricles
	X[:,idx_cv_a] = CV_atria


	X_EP = np.loadtxt(os.path.join(DATA_folder,"X_EP_normalised.txt"),dtype=float)
	X_EP[:,0] = CV_ventricles
	X_EP[:,1] = CV_atria


	np.savetxt(os.path.join(DATA_folder,"X_EP.txt"),X_EP,fmt="%g")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    # parser.add_argument('--HPC_tags_file', type=str, required=True,
    #                     help='Full path to the .json file with tags on the HPC')

    # parser.add_argument('--HPC_setup_file', type=str, required=True,
    #                     help='.json file with the setup for the simulations on the HPC')  

    # parser.add_argument('--HPC_path_to_unloading_library', type=str, required=True,
    #                     help='Path to the unloading library on the HPC')  

    # parser.add_argument('--HPC_env_folder', type=str, required=True,
    #                     help='Path to the pyunload virtual environment on the HPC')  

    # parser.add_argument('--HPC_statefolder', type=str, required=True,
    #                     help='Where the states for the cell simulations will be on the HPC')  
	
    parser.add_argument('--basefolder', type=str)

    args = parser.parse_args()

    main(args)