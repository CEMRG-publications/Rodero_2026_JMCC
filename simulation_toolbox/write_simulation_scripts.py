import os

import numpy as np
import json
import argparse

from SIMULATION_library import simulator_utils
from SIMULATION_library.fch_setup import simulation
from SIMULATION_library.cell_sims_utils import generate_bench_script

from common.cell_io import bin_to_dat_folder, plot_Land_output

def main(args):

    print("Generating cycle scripts from "+args.paramfolder+"...")

    if not os.path.exists(args.slrmfolder):
      os.system("mkdir -p "+args.slrmfolder)

    idx_1 = int(args.idx1)
    idx_2 = int(args.idx2)

    sim_setup = simulation()

    if not os.path.exists(args.setup_file):
        raise Exception("Cannot find setup file "+args.setup_file)
    sim_setup.load(args.setup_file)

    original_meshname = sim_setup.meshname

    where_to_save_param_string = args.cell_sims_folder+"/param/"
    os.system("rm -r "+where_to_save_param_string)

    print("Saving the parameter files for the cell simulations in "+where_to_save_param_string+"...")

    for i in range(idx_1,idx_2+1):

        sim_setup.meshname = "/unloaded/"+original_meshname+"_unloaded_"+str(i)
        sim_setup.testname = "cycle_"+str(i)

        sim_setup.torord_init_file = f"{args.HPC_statefolder}/ToRORd_dynCl/{i}/{i}_ToRORd_dynCl_LandHumanStress.sv"
        sim_setup.torord_rv_init_file = f"{args.HPC_statefolder}/ToRORd_dynCl_rv/{i}/{i}_ToRORd_dynCl_rv_LandHumanStress.sv"
        sim_setup.courtemanche_init_file = f"{args.HPC_statefolder}/JB_COURTEMANCHE/{i}/{i}_JB_COURTEMANCHE_LandHumanStress.sv" 

        simulator_utils.write_simulation_script(args.paramfolder+'/'+str(i)+'.json',
                                                args.clinical_data,
                                                args.tags_file,
                                                args.slrmfolder+"/cycle_"+str(i)+".slrm",
                                                sim_setup,
                                                postprocessing=False,
                                                adapt_tubeArea=True, ### Remember to set this to False after HCM!!
                                                save_param_string=where_to_save_param_string+str(i)+"_")

    # --------------------------------------------------------------------------------------------------------
    cell_sims_basefolder = args.cell_sims_folder+"/"
    X = np.loadtxt(os.path.join(args.datafolder,"X.txt"))
    N = X.shape[0]

    with open(args.clinical_data,"r") as f:
      clinical_data = json.load(f)

    # choice = input("Do you want to run the cell simulations - takes a long time and the states should already be there? [y/n] : ")
    choice = 'n'
    while choice not in ['y','n']:
      choice = input("You need to pick y or n : ")
    if choice == 'y':
      print("Running cell simulations...")
      BCL = int(clinical_data["general"]["BCL"])
      NBEATS = 500
      NPROC = 23 # max number of cores - 1      

      generate_bench_script(N,                                # how many simulations to run (one per param.json)
                            BCL,                              # BCL
                            NBEATS,                           # NBEATS
                            cell_sims_basefolder,             # basefolder containing the param folder
                            NPROC,                            # number of CPUs to use for parallel runs
                            0.0,                              # strain
                            "LV",                             # chamber = LV, RV or atria
                            sim_setup.contraction_model,      # contraction model
                            suffix="")
      os.system("bash "+cell_sims_basefolder+"run_ToRORd_dynCl.sh")     

      bin_to_dat_folder(cell_sims_basefolder+"ToRORd_dynCl/",
                                0,
                                N-1,
                                BCL,
                                NBEATS,
                                ["ToRORd_dynCl.Vm.bin","ToRORd_dynCl.Ca_i.bin","ToRORd_dynCl.Tension.bin"],
                                ["Vm.dat","Ca_i.dat","Tension.dat"],
                                cleanup=True)   

      plot_Land_output(cell_sims_basefolder+"ToRORd_dynCl/",
                       N,
                       figname=cell_sims_basefolder+"/ToRORd_output.png",
                       isometric=True)  

      generate_bench_script(N,                                # how many simulations to run (one per param.json)
                            BCL,                              # BCL
                            NBEATS,                           # NBEATS
                            cell_sims_basefolder,             # basefolder containing the param folder
                            NPROC,                            # number of CPUs to use for parallel runs
                            0.0,                              # strain
                            "RV",                             # chamber = LV, RV or atria
                            sim_setup.contraction_model,      # contraction model
                            suffix="_rv")
      os.system("bash "+cell_sims_basefolder+"run_ToRORd_dynCl_rv.sh")      

      bin_to_dat_folder(cell_sims_basefolder+"ToRORd_dynCl_rv/",
                                0,
                                N-1,
                                BCL,
                                NBEATS,
                                ["ToRORd_dynCl.Vm.bin","ToRORd_dynCl.Ca_i.bin","ToRORd_dynCl.Tension.bin"],
                                ["Vm.dat","Ca_i.dat","Tension.dat"],
                                cleanup=True) 
          
      plot_Land_output(cell_sims_basefolder+"ToRORd_dynCl_rv/",
                       N,
                       figname=cell_sims_basefolder+"/ToRORd_dynCl_Land_rv_output.png",
                       isometric=True)  

      generate_bench_script(N,                                # how many simulations to run (one per param.json)
                            BCL,                              # BCL
                            NBEATS,                           # NBEATS
                            cell_sims_basefolder,             # basefolder containing the param folder
                            NPROC,                            # number of CPUs to use for parallel runs
                            0.0,                              # strain
                            "atria",                          # chamber = LV, RV or atria
                            sim_setup.contraction_model,      # contraction model
                            suffix="")
      os.system("bash "+cell_sims_basefolder+"run_JB_COURTEMANCHE.sh")  

      bin_to_dat_folder(cell_sims_basefolder+"JB_COURTEMANCHE/",
                                0,
                                N-1,
                                BCL,
                                NBEATS,
                                ["JB_COURTEMANCHE.Vm.bin","JB_COURTEMANCHE.Ca_i.bin","JB_COURTEMANCHE.Tension.bin"],
                                ["Vm.dat","Ca_i.dat","Tension.dat"],
                                cleanup=True) 
          
      plot_Land_output(cell_sims_basefolder+"JB_COURTEMANCHE/",
                       N,
                       figname=cell_sims_basefolder+"/JB_COURTEMANCHE_output.png",
                       isometric=True)
    else:
      print("Skipping cell simulations")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--datafolder', type=str, required=True,
                        help='Provide folder where you have all X_*.txt and xlabels_*.txt')

    parser.add_argument('--setup_file', type=str, required=True,
                        default=None,
                        help='.json file with the setup for the simulations')  

    parser.add_argument('--paramfolder', type=str, required=True,
                        help='Where to save the json parameter files')

    parser.add_argument('--slrmfolder', type=str, required=True,
                        help='Where to save the slrm files')  

    parser.add_argument('--HPC_statefolder', type=str, required=True,
                        help='Where the states for the cell simulations will be on the HPC')  

    parser.add_argument('--idx1', type=str, required=True, default=None,
                        help='First index to generate the files for')  

    parser.add_argument('--idx2', type=str, required=True, default=None,
                        help='Last index to generate the files for')  

    parser.add_argument('--clinical_data', type=str, required=True,
                        default=None,
                        help='Json file with the clinical data.')  

    parser.add_argument('--tags_file', type=str, required=True,
                        help='Json file with the tags.')  

    parser.add_argument('--cell_sims_folder', type=str, required=False,
                        default="./SS/",
                        help='Where to save the state files for the cell models')  

    args = parser.parse_args()

    main(args)