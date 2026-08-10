import os

import json
import argparse

from common.carp_setup import hpc_headers

def main(args):

    print("Generating unloading scripts from "+args.paramfolder+"...")

    if not os.path.exists(args.slrmfolder):
      os.system("mkdir -p "+args.slrmfolder)

    if not args.default:
      idx_1 = int(args.idx1)
      idx_2 = int(args.idx2)

    input_params = ["a_ventricles",
                    "bf_ventricles",
                    "bfs_ventricles",
                    "bt_ventricles",
                    "a_atria",
                    "bf_atria",
                    "bfs_atria",
                    "bt_atria",
                    "k_peri",
                    "EDP_lv",
                    "EDP_rv",
                    "a_lvrv",
                    "bf_lv_scaling",
                    "bf_rv_scaling",
                    "bf_aa_scaling"]

    settings_file = args.setup_file
    with open(settings_file,"r") as f:
        settings = json.load(f)

    ncores = int(settings["nproc"])
    settings["walltime"] = "02:00:00"

    path2unloading = args.HPC_path_to_unloading_library
    path2carputils = args.HPC_path_to_carputils

    carp_config_file    = args.HPC_path_to_CARP_config
    archer2_config_file = None

    env_folder = args.HPC_env_folder

    env_variabiles = hpc_headers.write_env_variables(path2unloading,
                                                     path2carputils,
                                                     carp_config_file,
                                                     ncores,
                                                     env_folder,
                                                     archer2_config_file)

    print("Saving slrm files to "+args.slrmfolder+"...")

    if args.default:
       with open(os.path.join(args.paramfolder,"default.json"),"r") as f:
            parameters = json.load(f)

       output_basename = "unloading_default"
       header = hpc_headers.write_archer2_header(output_basename,
                                                    output_basename+".out",
                                                    settings["walltime"],
                                                    ncores) 
       runcommand = ["python",args.python_script_path_archer2,"--platform","desktop"]
       runcommand += ["--overwrite-behaviour","overwrite"]
       runcommand += ["--np",str(ncores)]
       runcommand += ["--testname",output_basename]
       runcommand += ["--tags_setup_file",args.HPC_tags_file]
       runcommand += ["--general_setup_file",args.HPC_setup_file]      

       for ip in input_params:
            runcommand += ["--"+ip,str(parameters["mechanics"][ip])]        

       runcommand = ' '.join(runcommand) 

          # -------------------------------------
          # write slrm file 
       slrm_script = os.path.join(args.slrmfolder,output_basename+".slrm")
       f = open(slrm_script,"w")       

       f.write(header)
       f.write(env_variabiles)
       f.write(runcommand)       

       f.close()

    else:
       
      for i in range(idx_1,idx_2+1):
          with open(os.path.join(args.paramfolder,str(i)+".json"),"r") as f:
            parameters = json.load(f)

          output_basename = "unloading_"+str(i)

          header = hpc_headers.write_archer2_header(output_basename,
                                                    output_basename+".out",
                                                    settings["walltime"],
                                                    ncores) 

          runcommand = ["python",args.python_script_path_archer2,"--platform","desktop"]
          runcommand += ["--overwrite-behaviour","overwrite"]
          runcommand += ["--np",str(ncores)]
          runcommand += ["--testname",output_basename]        

          runcommand += ["--tags_setup_file",args.HPC_tags_file]
          runcommand += ["--general_setup_file",args.HPC_setup_file]      

          for ip in input_params:
            runcommand += ["--"+ip,str(parameters["mechanics"][ip])]        

          runcommand = ' '.join(runcommand) 

          # -------------------------------------
          # write slrm file 
          slrm_script = os.path.join(args.slrmfolder,output_basename+".slrm")
          f = open(slrm_script,"w")       

          f.write(header)
          f.write(env_variabiles)
          f.write(runcommand)       

          f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--setup_file', type=str, required=True,
                        help='.json file with the setup for the simulations')  

    parser.add_argument('--paramfolder', type=str, required=True,
                        help='Where to save the json parameter files')

    parser.add_argument('--slrmfolder', type=str, required=True,
                        help='Where to save the slrm files')  

    parser.add_argument('--idx1', type=str, required=False, 
                        help='First index to generate the files for')  

    parser.add_argument('--idx2', type=str, required=False,
                        help='Last index to generate the files for')  

    parser.add_argument('--HPC_tags_file', type=str, required=True,
                        help='Json file with the tags.')  

    parser.add_argument('--HPC_setup_file', type=str, required=True,
                        help='Json file with the tags.')  

    parser.add_argument('--HPC_path_to_unloading_library', type=str, required=True,
                        help='Path to the unloading library on the HPC')  

    parser.add_argument('--HPC_path_to_carputils', type=str, required=False,
                        default="/work/e348/e348/shared/carputils/",
                        help='Path to the carputils on the HPC')  

    parser.add_argument('--HPC_path_to_CARP_config', type=str, required=False,
                        default="/work/e348/e348/shared/software/carpentry-system-petsc/carp.conf",
                        help='Path to the carp.conf file on the HPC')  

    parser.add_argument('--HPC_env_folder', type=str, required=True,
                        help='Path to the pyunload virtual environment on the HPC')  
    parser.add_argument('--python_script_path_archer2', type=str, required=True)
    parser.add_argument('--default', action='store_true')

    args = parser.parse_args()

    main(args)