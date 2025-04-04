import os

import numpy as np
import json
import argparse

from SIMULATION_library import simulator_utils
from SIMULATION_library.fch_setup import simulation
from Historia.shared.design_utils import read_labels

def check_files(datafolder,fields):

    for f in fields:
        if os.path.exists(datafolder+"xlabels_"+f+".txt") and os.path.exists(datafolder+"X_"+f+".txt"):
            print('Found files for field '+f+'.')
        else:
            raise Exception("Cannot find files for field "+f+".")
        
def X_to_json_modified(labels_fields,
			  datafolder,
			  outputfolder,
			  default_json):

	print('generating json file...')

	os.system('mkdir '+outputfolder)

	
	N = None
	while N is None:
		for lab in labels_fields:
			print(lab)
			X_tmp = np.loadtxt(datafolder+'/X_'+lab+'.txt')
			labels = read_labels(datafolder+'/xlabels_'+lab+'.txt')	
			# print(lab)
			# print(X_tmp.shape)
			if len(X_tmp.shape)>1 or len(labels) == 1:
				N = X_tmp.shape[0]
			elif (len(X_tmp.shape) == 1) and len(labels)>1:
				N = 1

	# generate this dictionary to avoid reading X_*.txt at every iteration
	dct_datasets = {}
	for k,lab in enumerate(labels_fields):
		print(lab)
		dct_datasets[lab] = {}

		labels = read_labels(datafolder+'/xlabels_'+lab+'.txt')	
		dct_datasets[lab]["labels"] = labels

		X = np.loadtxt(datafolder+'/X_'+lab+'.txt')
		dct_datasets[lab]["X"] = X

	for i in range(N):
		# if you want to combine the new parameters
		# with the default json file, then the dictionary
		# is initialised to the default json you give.
		# Otherwise it's empty

		f_input = open(default_json,"r")
		param_dictionary = json.load(f_input)
		f_input.close()

		for k,lab in enumerate(labels_fields):

			labels = dct_datasets[lab]["labels"]
			X = dct_datasets[lab]["X"]

			if default_json is not None:
				subdict = param_dictionary[lab]
			else:
				subdict = {}
			
			if (len(X.shape) == 1) and N==1:
				X = X.reshape(1,X.shape[0])
			elif (len(X.shape) == 1) and N>1:
				X = X.reshape(N,1)

			if (len(labels)!=X.shape[1]):
				raise ValueError('xlabels_'+lab+'.txt'+' and X_'+lab+'.txt do not match')	
			if (X.shape[0]!=N):
				raise ValueError('X_'+lab+'.txt'+' and X_'+labels_fields[0]+'.txt do not match')		
			for j in range(len(labels)):
				subdict[labels[j]] = X[i,j]	


			param_dictionary[lab] = subdict

		with open(outputfolder+'/'+str(i)+'.json', 'w') as f:
		    json.dump(param_dictionary, f, indent=4)



def main(args):

    os.system("mkdir -p "+args.paramfolder)
    os.system("mkdir -p "+args.slrmfolder)

    sim_setup = simulation()

    if args.setup_file is None:
        setup_file = './'+args.platform+'_setup_'+args.user+'.json'
    else:
        setup_file = args.setup_file
    if not os.path.exists(setup_file):
        raise Exception("You need to have a file called "+setup_file)

    sim_setup.load(setup_file)

    original_meshname = sim_setup.meshname

    if args.default:
        # Create a single script using only the default JSON file
        sim_setup.meshname = original_meshname
        sim_setup.testname = "inflation_default"
        sim_setup.walltime = '1:00:00'
        sim_setup.cycle_peri_on = True

        # Write the trace file
        trace_file_path = os.path.join(args.datafolder, "inflation_trace.trc")
        with open(trace_file_path, "w") as trace_file:
            trace_file.write("3\n")
            trace_file.write("0.0      0.0\n")
            trace_file.write("25.0     0.5\n")
            trace_file.write("50.0     1.0\n")
        print(f"Trace file created at: {trace_file_path}")
        sim_setup.trace_file = f"{sim_setup.simulation_folder}/../data/inflation_trace"

        simulator_utils.write_passive_inflation_script(
            json_file=args.defaultfile,
            json_clinical_file=args.clinical_data,
            json_tags_file=args.tags,
            simulation_script=args.slrmfolder + "/inflation_default.slrm",
            setup=sim_setup,
            postprocessing=False,
            get_fibre_strains=True,
            mechDT=args.mechDT
        )
    else:
        # Standard behavior
        fields = args.fields

        X_tmp = np.loadtxt(args.datafolder+"/X_"+fields[0]+".txt")
        N = X_tmp.shape[0]

        idx_1 = int(args.idx1) if args.idx1 is not None else 0
        idx_2 = int(args.idx2) if args.idx2 is not None else N-1

        # Create JSON files and simulation scripts
        X_to_json_modified(fields,
                           args.datafolder,
                           args.paramfolder,
                           default_json=args.defaultfile)

        for i in range(idx_1, idx_2+1):
            sim_setup.meshname = original_meshname
            sim_setup.testname = "inflation_" + str(i)
            sim_setup.walltime = '1:00:00'
            sim_setup.cycle_peri_on = True

            simulator_utils.write_passive_inflation_script(
                json_file=args.paramfolder + '/' + str(i) + '.json',
                json_clinical_file=args.clinical_data,
                json_tags_file=args.tags,
                simulation_script=args.slrmfolder + "/inflation_" + str(i) + ".slrm",
                setup=sim_setup,
                postprocessing=False,
                get_fibre_strains=True,
                mechDT=args.mechDT
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--datafolder', type=str, required=True,
                        help='Provide folder where you have all X_*.txt and xlabels_*.txt')

    parser.add_argument('--fields', nargs='+', required=False,
                        help='Provide the list of fields you need to modify.')

    parser.add_argument('--platform', type=str, required=True,
                        help='HPC platform (tom2 or archer2)')

    parser.add_argument('--paramfolder', type=str, required=True,
                        help='Where to save the json parameter files')

    parser.add_argument('--slrmfolder', type=str, required=True,
                        help='Where to save the slrm files')  

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

    parser.add_argument('--mechDT', type=str, required=False,
                        default=1.0)  

    parser.add_argument('--default', action='store_true',
                        help='If set, creates a script with only the default values from the JSON file.')

    args = parser.parse_args()

    # Validate arguments based on the presence of --default
    if not args.default:
        if not args.datafolder or not args.fields:
            parser.error("--fields is required unless --default is set.")

    main(args)
