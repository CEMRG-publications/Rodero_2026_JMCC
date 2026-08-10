import os

import numpy as np
import argparse

from common.carp_setup import simulator_utils

def check_files(datafolder,fields):

    for f in fields:
        if os.path.exists(os.path.join(datafolder,"xlabels_"+f+".txt")) and os.path.exists(os.path.join(datafolder,"X_"+f+".txt")):
            print('Found files for field '+f+'.')
        else:
            raise Exception("Cannot find files for field "+f+".")

def main(args):

    print("Generating .json files from "+args.datafolder+"...")

    if not os.path.exists(args.paramfolder):
      os.system("mkdir -p "+args.paramfolder)

    fields = args.fields
    check_files(args.datafolder,fields)

    X_tmp = np.loadtxt(args.datafolder+"/X_"+fields[0]+".txt")
    N = X_tmp.shape[0]

    # ------------------------------
    # create json files and simulation scripts
    # ------------------------------
    simulator_utils.X_to_json(fields,
                              args.datafolder,	
   							              args.paramfolder,
   							              default_json=args.defaultfile,
                              adapt_beta=True)

    print("Done. Saved parameter files in "+args.paramfolder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--datafolder', type=str, required=True,
                        help='Provide folder where you have all X_*.txt and xlabels_*.txt')

    parser.add_argument('--fields', nargs='+', required=True,
                        help='Provide the list of fields you need to modify.')

    parser.add_argument('--paramfolder', type=str, required=True,
                        help='Where to save the json parameter files')

    parser.add_argument('--defaultfile', type=str, required=True, default=None,
                        help='The json default file to modify')  

    args = parser.parse_args()

    main(args)