import os 
import argparse

import numpy as np
import matplotlib.pyplot as plt
import re
import tqdm
import seaborn as sns
import pandas as pd

from Historia.shared.design_utils import read_labels

def file_exists(full_file_path):
    if not os.path.isfile(full_file_path):
        raise Exception(f"You need to have the file {os.path.abspath(os.path.normpath(full_file_path))}")


def check_fourchamber_unloading_(simulation_folder,
                                 chambers,t):

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
        if converged:
            if os.path.exists(simulation_folder+"final_data/"+ch+".vol.dat"):
                vol = np.loadtxt(simulation_folder+"final_data/"+ch+".vol.dat",dtype=float,usecols=[1])

                vol_final = vol[0]
            else: 
                t.set_description(f"Reading {simulation_folder}final_data/{ch}.vol.dat failed. Reading unloading.log instead...")
                
                with open(f'{simulation_folder}/unloading.log', 'r') as f:
                    contents = f.read()
                
                matches = contents.split('--- CONVERGED ---')[0].split('\n')[-2].split(4*' ')[-2].split('/')
                
                vol = [float(match) for match in matches]
                print(vol)

                chamber_map = {'lv_endo': 0, 'rv_endo': 1, 'la_endo': 2, 'ra_endo': 3}
                vol_final = vol[chamber_map[ch]]

                print(vol_final)
            output.append(vol_final)
        else:
            output.append(-1)

    return output

def check_fourchamber_unloading(basefolder,
                                chambers,
                                start_sample,
                                last_sample,
                                output_file='unloaded_volumes.txt'):

    # vol_unloaded = np.zeros((last_sample-start_sample+1,len(chambers)),dtype=float)
    # count = 0
    n_sim_work = 0
    n_sim_not_work = 0

    if start_sample is not None:
        vol_unloaded = -1*np.ones((last_sample+1,len(chambers)),dtype=float)

        success_bar_format = "{l_bar}\033[92m{bar}\033[0m{r_bar}"  # Green bar
        failure_bar_format = "{l_bar}\033[91m{bar}\033[0m{r_bar}"  # Red bar

        t = tqdm.tqdm(range(start_sample,last_sample+1), desc='Bar desc', leave=True)
        for i in t:   
            worked = True 

            with open(f"{basefolder}/unloading_{i}.out", "r") as file:
                last_lines = file.readlines()[-3:]  # Read last 3 lines

            if any("TIME LIMIT" in line for line in last_lines):
                print(f"Unloading {i} reached wall time")


            if os.path.exists(f"{basefolder}/unloading_{i}/unloading.log"):
                output = check_fourchamber_unloading_(basefolder+'/unloading_'+str(i)+'/',
                                                    chambers,t)

                if output[0]:
                    n_sim_work+=1
                    t.bar_format = success_bar_format
                    t.set_description('unloading_'+str(i)+' successful...')

                    for j in range(len(chambers)):
                        # vol_unloaded[count,j] = output[1+j]

                        vol_unloaded[i,j] = output[1+j]
                else:
                    worked = False
            else:
                worked = False

            if not worked:
                n_sim_not_work+=1
                t.bar_format = failure_bar_format
                t.set_description('unloading_'+str(i)+' crashed...')
                
                for j in range(len(chambers)):
                    # vol_unloaded[count,j] = -1
                    vol_unloaded[i,j] = -1

    else:
        vol_unloaded = -1*np.ones((2,len(chambers)),dtype=float)

        output = check_fourchamber_unloading_(f"{basefolder}/unloading_default/",chambers,t)
        if output[0]:
            n_sim_work+=1
            print('unloading_default successful...')

            for j in range(len(chambers)):
                # vol_unloaded[count,j] = output[1+j]

                vol_unloaded[0,j] = output[1+j]

        else:
            n_sim_not_work+=1
            print('unloading_default crashed...')
            
            for j in range(len(chambers)):
                # vol_unloaded[count,j] = -1
                vol_unloaded[0,j] = -1


    np.savetxt(output_file,vol_unloaded,fmt="%g")

    return (n_sim_work,n_sim_not_work)

def main(args):

    basefolder = args.path2simulations
    path2figure = args.path2figures
    first_simulation = args.first_simulation
    last_simulation = args.last_simulation
    default  = args.default

    if not default:
        file_exists(os.path.join(basefolder,"../data/X_mechanics.txt"))
        file_exists(os.path.join(basefolder,"../data/xlabels_mechanics.txt"))

    os.system("mkdir -p "+ os.path.join(basefolder,"unloaded/"))
    os.system("mkdir -p "+ os.path.join(basefolder,"../figures/"))

    n_sim_work,n_sim_not_work = check_fourchamber_unloading(basefolder,
                            ['lv_endo','rv_endo','la_endo','ra_endo'],
                            start_sample=first_simulation,
                            last_sample=last_simulation,
                            output_file=os.path.join(basefolder,'unloaded_volumes.txt'))    
    
    if not default:
        unloaded_volumes_all_samples = np.loadtxt(os.path.join(basefolder,"unloaded_volumes.txt"),dtype=float)

        unloaded_volumes = unloaded_volumes_all_samples[first_simulation:(last_simulation+1),:]

        X_all_samples = np.loadtxt(os.path.join(basefolder,"../data/X_mechanics.txt"),dtype=float)

        X = X_all_samples[first_simulation:(last_simulation+1),:]

        def is_list_of_lists(lst):
            return all(isinstance(i, list) for i in lst)

        if unloaded_volumes.ndim > 1:
            mask = np.all(unloaded_volumes != -1, axis=1)
        else:
            mask = np.all(unloaded_volumes != -1)

        xlabels = read_labels(os.path.join(basefolder,"../data/xlabels_mechanics.txt"))

        idx_ok = np.where(mask!=0)[0]
        idx_notok = np.where(mask==0)[0]

        if len(X.shape) > 1:
            
            # Prepare the data in a DataFrame for Seaborn
            data = pd.DataFrame(X, columns=xlabels)

            # Assign 'status' column based on the index arrays
            data['status'] = np.where(np.isin(np.arange(X.shape[0]), idx_ok), 'OK', 'Not OK')

            g = sns.pairplot(data, hue ='status', diag_kind='hist', diag_kws={'color':'red', 'bins':30, 'alpha': 0.5}, palette={'OK': 'green', 'Not OK': 'red'}, corner=True)

            sns.move_legend(g, "lower center",bbox_to_anchor=(0.5, -0.035),  title='Group', frameon=False,)

            

            plt.savefig(os.path.join(path2figure, "unloaded_scatter.png"), bbox_inches="tight", dpi=300)

    
    print(f"A total of {n_sim_work} unloadings work, and {n_sim_not_work} did not work.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to check which unloading simulations worked.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--path2simulations', type=str, required=True)
    parser.add_argument('--path2figures', type=str, required=True)
    parser.add_argument('--first_simulation', type=int, required=False)
    parser.add_argument('--last_simulation', type=int, required=False)
    parser.add_argument('--default', action='store_true')
    args = parser.parse_args()

    main(args)
