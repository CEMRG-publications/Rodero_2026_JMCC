import os
import sys

import numpy as np
import json
import copy
import argparse

from SIMULATION_library import simulator_utils
from SIMULATION_library.fch_setup import simulation
from SIMULATION_library.cell_sims_utils import *

from GSA_library import ionic_output
from GSA_library.plotting import plot_Land_output

def check_files(datafolder,fields):

    for f in fields:
        if os.path.exists(datafolder+"xlabels_"+f+".txt") and os.path.exists(datafolder+"X_"+f+".txt"):
            print('Found files for field '+f+'.')
        else:
            raise Exception("Cannot find files for field "+f+".")

def main(args):

    os.system("mkdir -p "+args.paramfolder)
    os.system("mkdir -p "+args.slrmfolder)

    fields = args.fields

    X_tmp = np.loadtxt(args.datafolder+"/X_"+fields[0]+".txt")
    N = X_tmp.shape[0]

    idx_1 = int(args.idx1) if args.idx1 is not None else 0
    idx_2 = int(args.idx2) if args.idx2 is not None else N-1

    sim_setup = simulation()

    if args.setup_file is None:
        setup_file = './'+args.platform+'_setup_'+args.user+'.json'
    else:
        setup_file = args.setup_file
    if not os.path.exists(setup_file):
        raise Exception("You need to have a file called "+setup_file)

    sim_setup.load(setup_file)

    original_meshname = sim_setup.meshname

    # ------------------------------
    # create json files and simulation scripts
    # ------------------------------

    simulator_utils.X_to_json(fields,
    							  args.datafolder,	
    							  args.paramfolder,
    							  default_json=args.defaultfile)

    
    where_to_save_param_string = args.paramfolder_cell
    for i in range(idx_1,idx_2+1):

        sim_setup.meshname = "/unloaded/"+original_meshname+"_unloaded_"+str(i)
        sim_setup.testname = "cycle_"+str(i)
        sim_setup.walltime = '24:00:00'
        sim_setup.cycle_peri_on = True

        sim_setup.torord_init_file = f"{args.HPC_statefolder}/ToRORd_dynCl/{i}/{i}_ToRORd_dynCl_LandHumanStress.sv"
        sim_setup.torord_rv_init_file = f"{args.HPC_statefolder}/ToRORd_dynCl/{i}/{i}_ToRORd_dynCl_LandHumanStress.sv"
        sim_setup.courtemanche_init_file = f"{args.HPC_statefolder}/converted_COURTEMANCHE/{i}/{i}_converted_COURTEMANCHE_LandHumanStress.sv"
       
        simulator_utils.write_simulation_script(args.paramfolder+'/'+str(i)+'.json',
        										args.clinical_data,
        										args.tags,
        										args.slrmfolder+"/cycle_"+str(i)+".slrm",
        										sim_setup,
    											postprocessing=False,
                                                save_param_string=where_to_save_param_string+str(i)+"_")
        
        
    # os.system(f"rm -rf {where_to_save_param_string}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--datafolder', type=str, required=True,
                        help='Provide folder where you have all X_*.txt and xlabels_*.txt')

    parser.add_argument('--fields', nargs='+', required=True,
                        help='Provide the list of fields you need to modify.')

    parser.add_argument('--platform', type=str, required=True,
                        help='HPC platform (tom2 or archer2)')

    parser.add_argument('--paramfolder', type=str, required=True,
                        help='Where to save the json parameter files')

    parser.add_argument('--paramfolder_ToRORd', type=str, required=False,
                        help='Folder containing the parameter strings for ToRORd_dynCl and Land')
    parser.add_argument('--paramfolder_cell', type=str, required=True,
                        help='Folder that will contain the parameter strings for the cell simulations')

    parser.add_argument('--paramfolder_ToRORd_rv', type=str, required=False, default=None,
                        help='Folder containing the parameter strings for ToRORd_dynCl and Land for the RV')

    parser.add_argument('--slrmfolder', type=str, required=True,
                        help='Where to save the slrm files')  

    parser.add_argument('--HPC_statefolder', type=str, required=True,
                        help='To set the states folder in the scripts')  

    parser.add_argument('--defaultfile', type=str, required=False, default="./default/default.json",
                        help='The json default file to modify')  

    parser.add_argument('--idx1', type=str, required=False, default=None,
                        help='First index to generate the files for')  

    parser.add_argument('--idx2', type=str, required=False, default=None,
                        help='Last index to generate the files for')  

    parser.add_argument('--clinical_data', type=str, required=False,
                        default="/data/Dropbox/Sensitivity/patient_data/case19/clinical_data.json",
                        help='Json file with the clinical data.')  

    parser.add_argument('--tags', type=str, required=False,
                        default="./data/tags_lvrv.json",
                        help='Json file with the tags.')  

    parser.add_argument('--user', type=str, required=False,
                        default="mas",
                        help='Username on the HPC')  
    
    parser.add_argument('--setup_file', type=str, required=False,
                        default=None,
                        help='Full path of the settings file')  

    args = parser.parse_args()

    main(args)
