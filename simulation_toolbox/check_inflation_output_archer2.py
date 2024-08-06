import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt

from SIMULATION_library import fourchamber_output, mesh_utils


kPa_to_mmHg = 7.50062
mmHg_to_kPa = 1./kPa_to_mmHg

def file_exists(full_file_path):
    if not os.path.isfile(full_file_path):
        raise Exception("You need to have the file " + full_file_path)

def passive_output_per_chamber(output_folder,
				   elem_file,
				   tags,
				   start_sample=0,
				   last_sample=0,
				   basename="",
				   tend=50,
				   output_file="Y.txt",
				   mask_output_file="output_mask.txt"):

    print('Reading mesh elem file...')
    elem = mesh_utils.read_tets(elem_file)
    print('Done.')
      
    LV_EIDX = np.where(np.isin(elem[:,-1],tags["lv"]+tags["fast_endo_lv"])==1)[0]
    RV_EIDX = np.where(np.isin(elem[:,-1],tags["rv"]+tags["fast_endo_rv"])==1)[0]
    LA_EIDX = np.where(np.isin(elem[:,-1],tags["la"])==1)[0]
    RA_EIDX = np.where(np.isin(elem[:,-1],tags["ra"])==1)[0]

    V_EIDX = np.where(np.isin(elem[:,-1],tags["ventricles"]+tags["fast_endo"])==1)[0]
    A_EIDX = np.where(np.isin(elem[:,-1],tags["atria"]+tags["bachmann_bundle"])==1)[0]

    LV_VTX = np.unique(elem[LV_EIDX,0:4].flatten())
    RV_VTX = np.unique(elem[RV_EIDX,0:4].flatten())
    LA_VTX = np.unique(elem[LA_EIDX,0:4].flatten())
    RA_VTX = np.unique(elem[RA_EIDX,0:4].flatten())

    V_VTX = np.unique(elem[V_EIDX,0:4].flatten())
    A_VTX = np.unique(elem[A_EIDX,0:4].flatten())

    chambers = ['lv','rv','la','ra']

    Y = np.zeros((last_sample-start_sample+1,len(chambers)+6),dtype=float)
    mask_output = np.zeros((last_sample-start_sample+1,),dtype=float)

    for i in range(start_sample,last_sample+1):
        
        for j,c in enumerate(chambers):
            if os.path.isfile(f"{output_folder}/{basename}{i}/{c}_endo.vol.dat"):
                volume = np.loadtxt(output_folder+'/'+basename+str(i)+'/'+c+'_endo.vol.dat',dtype=float,usecols=[1])
                
                if volume.shape[0]==tend+1:
                    mask_output[i] = 1
                    Y[i,j] = np.max(volume)
        if mask_output[i]==1:

            if not os.path.isfile(f"{output_folder}/{basename}{i}/fiberProjectedStrain.txt"):
                last_frame = 5
                print(f"Extracting strains for simulation #{i}, assuming the last frame in {last_frame}")
                os.system(f"igbextract -o ascii_1pLn --f0={last_frame} --f1={last_frame} -O {output_folder}/{basename}{i}/fiberProjectedStrain.txt {output_folder}/{basename}{i}/fiberProjectedStrain.igb")

            # load strains
            print('Computing mean strains '+basename+str(i)+'...')
            strains = np.loadtxt(output_folder+'/'+'/'+basename+str(i)+'/fiberProjectedStrain.txt',dtype=float,skiprows=1)  

            # strains per chamber
            Y[i,j+1] = np.mean(strains[LV_VTX])  
            Y[i,j+2] = np.mean(strains[RV_VTX])  
            Y[i,j+3] = np.mean(strains[LA_VTX])  
            Y[i,j+4] = np.mean(strains[RA_VTX])  

            # ventricular strains
            strains_v = np.mean(strains[V_VTX])
            Y[i,j+5] = strains_v    

            # atrial strains
            strains_a = np.mean(strains[A_VTX])
            Y[i,j+6] = strains_a
        else:
            print('Simulation '+basename+str(i)+' failed.')
    
    np.savetxt(output_file,Y,fmt="%g")
    np.savetxt(mask_output_file,mask_output,fmt="%d")

def plot_passive_output_avoid_crashes(output_folder,
						start_sample=0,
						last_sample=0,
						basename="",
						figname=None,
                        mask_output_file="output_mask.txt"):
     
    mask = np.loadtxt(mask_output_file,dtype=int)

    chambers = ['lv','rv','la','ra']
	
    ax = plt.figure(figsize=(10,10), constrained_layout=True).subplots(2, 2)
    ax = ax.flatten()
    for i in range(start_sample,last_sample+1):
        if mask[i]:
            for j,c in enumerate(chambers):
                volume = np.loadtxt(output_folder+'/'+basename+str(i)+'/'+c+'_endo.vol.dat',dtype=float,usecols=[1])
                pressure = np.loadtxt(output_folder+'/'+basename+str(i)+'/'+c+'_endo.nbc_p.dat',dtype=float,usecols=[1])*kPa_to_mmHg
                
                ax[j].plot(volume,pressure,color='#3489eb')
                ax[j].set_xlabel(c+' volume [mL]')
                ax[j].set_ylabel(c+' pressure [mmHg]')

    if figname is not None:
        plt.savefig(figname,dpi=300)
    else:
        plt.show()

def main(args):

    basefolder         = args.basefolder
    simulations_folder      = f"{basefolder}/simulations"
    output_folder      = f"{basefolder}/output"
    figures_path       = f"{basefolder}/figures"
    elem_file          = args.elem_file
    first_simulation   = args.first_simulation
    last_simulation    = args.last_simulation

    with open(f"{basefolder}/json_files/tags.json","r") as f:
        tags = json.load(f)

    os.makedirs(output_folder,exist_ok=True)
    os.makedirs(figures_path,exist_ok=True)


    passive_output_per_chamber(output_folder = simulations_folder,
				   elem_file = elem_file,
				   tags = tags,
				   start_sample=first_simulation,
				   last_sample=last_simulation,
				   basename="inflation_",
				   tend=50,
				   output_file=f"{output_folder}/Y.txt",
				   mask_output_file=f"{output_folder}/output_mask.txt")
    
    plot_passive_output_avoid_crashes(output_folder=simulations_folder,
                    start_sample=first_simulation,
                    last_sample=last_simulation,
                    basename="inflation_",
                    figname=f"{figures_path}/PV_traces.png",
                    mask_output_file=f"{output_folder}/output_mask.txt")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate a pdf and a txt file showing the simulations that worked. It also plots the simulations that crashed and the ones who didn't in the parameter space and plots all the pv loops for the ones that worked.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/h01/new_unloading/unloading_simulations",
                        help='Path to the folder where the simulations, data, and figure folders are.')
    parser.add_argument('--elem_file', type=str, help="Path to the elem file of the mesh without the bachmann bundle.", required=True)
    parser.add_argument('--first_simulation', type=int)
    parser.add_argument('--last_simulation', type=int)

    args = parser.parse_args()

    main(args)
