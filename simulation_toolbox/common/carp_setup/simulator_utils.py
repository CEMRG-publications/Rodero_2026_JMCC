"""
Writers that turn a parameter set into a CARPentry submission script.

Vendored verbatim from SIMULATION_library.simulator_utils by M. Strocchi, which is not publicly available,
so that this repository does not depend on any private code. Only the internal
imports have been repointed at this package; the source is otherwise unchanged,
including its tab indentation, so that the files it generates stay byte-identical
to the ones used for the published simulations.
"""

import os
import copy

import re
import json
import math

import numpy as np

from gpytGPE.utils.design import read_labels

from .fch_electrophysiology import fcclock,stimulus,ep_region
from .fch_cell import ionic
from .fch_mechanics import mregion,spring,mech_nbc
from .fch_cvs import cavity

# compute_wallarea_tube is vendored here from DATA_library.circadapt_parameters
# (M. Strocchi), which is also not publicly available. It is the wall area of an
# annular tube of the given rest area and thickness.


def compute_wallarea_tube(Aref, thickness):
    # Aref [mm2]    : area of the tube at rest
    # thickness [mm]: thickness of the tube

    Awall = math.pi * thickness * (thickness + 2 * math.sqrt(Aref / math.pi))

    return Awall


import warnings

kPa_to_mmHg = 1/0.133322387415
mmHg_to_kPa = 0.133322387415

def construct_param_string(dct,
						   header='',
						   tail='',
						   param_labels=None):
	# provide param_labels is the dictionary in
	# the keys are different from the parameter
	# names you want in the string
	if param_labels is None:
		param_labels = list(dct.keys())
	param_string = [param_labels[i]+'='+str(dct[par]) for i,par in enumerate(param_labels)]
	param_string = ','.join(param_string)
	param_string = header+param_string+tail

	return param_string

def scale_param_string(param_string,
						 param_name,
						 scale,
						 param_strin_new):

	fID = open(param_string, "r")
	param = fID.read()
	fID.close()		

	old_value = re.search(param_name+"=(.*?),",param).group(1)
	newvalue = scale*float(old_value)

	new_param = param.replace(old_value, str(newvalue), 1)

	fID = open(param_strin_new, 'w')
	fID.write(new_param)
	fID.close()

def split_X(X_file,idx_list,X_output_file_list):

	X = np.loadtxt(X_file,dtype=float)

	for i,idx_v in enumerate(idx_list):
		X_split = np.zeros((X.shape[0],len(idx_v)),dtype=float) 
		X_split = X[:,idx_v]
		np.savetxt(X_output_file_list[i],X_split,fmt='%g')

def X_to_json(labels_fields,
			  datafolder,
			  outputfolder,
			  adapt_beta=False,
			  default_json=None):

	print('generating json file...')

	os.system('mkdir '+outputfolder)

	if adapt_beta:
		if ("ToRORd_land" not in labels_fields) and ("COURTEMANCHE_land" not in labels_fields):
			warnings.warn("You are asking to adapt beta_1 but you are not changing the ToRORd-Land or the Courtemanche-Land parameters. This flag will be ignored.")

		ca50_idx_v = None
		if "ToRORd_land" in labels_fields:
			torordl_labels = read_labels(datafolder+'/xlabels_ToRORd_land.txt')	
			for k,l in enumerate(torordl_labels):
				if l == 'ca50':
					ca50_idx_v = k	

			if ca50_idx_v is None:
				warnings.warn("You are asking to adapt beta_1 but ca50 for the ventricles is not changing. Beta_1 left to its default value.")
		
		ca50_idx_a = None
		if "COURTEMANCHE_land" in labels_fields:
			courtl_labels = read_labels(datafolder+'/xlabels_COURTEMANCHE_land.txt')	
			for k,l in enumerate(courtl_labels):
				if l == 'ca50':
					ca50_idx_a = k	

			if ca50_idx_a is None:
				warnings.warn("You are asking to adapt beta_1 but ca50 for the atria is not changing. Beta_1 left to its default value.")	

			beta_1_default_v = -2.4
			ca50_default_v = 0.805	

			beta_1_default_a = -2.4
			ca50_default_a = 0.86

	N = None
	while N is None:
		for lab in labels_fields:
			X_tmp = np.loadtxt(datafolder+'/X_'+lab+'.txt')
			labels = read_labels(datafolder+'/xlabels_'+lab+'.txt')	
			if len(X_tmp.shape)>1 :
				N = X_tmp.shape[0]
			elif (len(X_tmp.shape) == 1) and len(labels)>1:
				N = 1
			elif (len(X_tmp.shape) == 1) and len(labels)==1:
				N = X_tmp.shape[0]

	# generate this dictionary to avoid reading X_*.txt at every iteration
	dct_datasets = {}
	for k,lab in enumerate(labels_fields):
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

		if default_json is not None:
			f_input = open(default_json,"r")
			param_dictionary = json.load(f_input)
			f_input.close()
		else:
			param_dictionary = {}

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
			size_X = X.shape

			if (len(labels)!=X.shape[1]):
				raise ValueError('xlabels_'+lab+'.txt'+' and X_'+lab+'.txt do not match')	
			if (X.shape[0]!=N):
				raise ValueError('X_'+lab+'.txt'+' and X_'+labels_fields[0]+'.txt do not match')		
			for j in range(len(labels)):
				subdict[labels[j]] = X[i,j]	

			if lab=="ToRORd_land" and adapt_beta and (ca50_idx_v is not None):
				subdict["beta_1"] = beta_1_default_v/ca50_default_v*X[i,ca50_idx_v]

			if lab=="COURTEMANCHE_land" and adapt_beta and (ca50_idx_a is not None):
				subdict["beta_1"] = beta_1_default_a/ca50_default_a*X[i,ca50_idx_a]

			param_dictionary[lab] = subdict

		with open(outputfolder+'/'+str(i)+'.json', 'w') as f:
		    json.dump(param_dictionary, f, indent=4)

def write_script_header(f,
						setup):
	if setup.platform=='archer2':
		f.write('#!/bin/bash --login\n')
		f.write('#\n')
		f.write('#SBATCH --job-name='+setup.testname+'\n')
		f.write('#SBATCH --nodes='+str(math.ceil(setup.nproc/128))+'\n')
		f.write('#SBATCH --tasks-per-node=128\n')
		f.write('#SBATCH --cpus-per-task=1\n')
		f.write('#SBATCH --output='+setup.testname+'.out\n')
		f.write('#SBATCH --mail-type=ALL\n')
		f.write('#SBATCH --mail-user=None\n')
		f.write('#SBATCH --time='+setup.walltime+'\n')
		f.write('#SBATCH --account=e348\n')
		f.write('#SBATCH --partition=standard\n')
		f.write('#SBATCH --qos=standard\n')
		f.write('\n')

		f.write('module load epcc-job-env\n')
		f.write('\n')

		f.write('export OMP_NUM_THREADS=1\n')		
		f.write('\n')	

		f.write('srun --distribution=block:block --hint=nomultithread  '+' '+setup.carpentryfolder+'/carp.pt \\\n')
	elif setup.platform=='tom2':

		f.write('#!/bin/bash \n')
		f.write('#SBATCH -p compute\n')
		f.write('#SBATCH -J '+setup.testname+'\n')
		f.write('#SBATCH --output='+setup.testname+'.out\n')
		f.write('#SBATCH -t 0-'+setup.walltime+'\n')
		f.write('#SBATCH --nodes='+str(math.ceil(setup.nproc/64))+'\n')
		f.write('#SBATCH --ntasks-per-node=64\n')
		f.write('\n')
		
		f.write('NPROC='+str(setup.nproc)+'\n')

		f.write('\n')
		f.write('source ${HOME}/.bashrc\n')
		f.write('\n')
		f.write('export OMP_NUM_THREADS=1\n')
		f.write('\n')

		f.write('mpiexec -np ${NPROC} '+setup.carpentryfolder+'/carp.pt \\\n')
	elif setup.platform=='imperial':
		f.write('#!/bin/bash \n')
		f.write(f'#PBS -l select={math.ceil(setup.nproc/128)}:ncpus=128:mem=96gb\n')
		f.write(f'#PBS -l walltime={setup.walltime}\n')
		f.write(f'#PBS -N {setup.testname}\n')

		f.write('module load tools/dev\n')
		f.write('module load PETSc/3.17.4-foss-2022a\n')
		f.write('module load sundials/2.7.0\n')
		f.write('module load Python/3.10.4-GCCcore-11.3.0\n')

		f.write('export FI_MLX_IFACE=eth0\n')
		f.write('export OMP_NUM_THREADS=1\n')

		f.write(f'export CARP_HOME={setup.carpentryfolder}\n')
		f.write(f'export CARPENTRY_LICENSE=$CARP_HOME/license.bin\n')

		f.write(f'export SIM_NAME={setup.simulation_folder}/{setup.testname}\n')

		f.write(f'mkdir -p $SIM_NAME\n')

		f.write(f'mpirun {setup.carpentryfolder}/carp.pt \\\n')

	else:
		raise ValueError('I do not recognise the platform you selected. Choose between [archer2,tom2,imperial]')


def write_passive_inflation_script(json_file,
								   json_clinical_file,
								   json_tags_file,
								   simulation_script,
								   setup,
								   postprocessing=False,
								   get_fibre_strains=False,
								   mechDT=1.0):

	f_input = open(json_file,"r")
	json_data = json.load(f_input)
	f_input.close()

	f_clinical_input = open(json_clinical_file,"r")
	json_clinical_data = json.load(f_clinical_input)
	f_clinical_input.close()

	f_tags_input = open(json_tags_file,"r")
	json_tags = json.load(f_tags_input)
	f_tags_input.close()
	
	f = open(simulation_script,"w")
	write_script_header(f,setup)

	simulation_options(f,setup.folderopts)

	f.write('  -simID '+setup.simulation_folder+'/'+setup.testname+' \\\n')
	f.write('  -meshname '+setup.meshdir+'/'+setup.meshname+' \\\n')
	if setup.fibresname is not None:
		# you can give a separate .lon file. If you do, the 
		# MESHNAME.lon file is completely ignored
		f.write('	-orthoname '+setup.meshdir+'/'+setup.fibresname+' \\\n') 

	simulation_electrophysiology_solver_setup(f)
	simulation_mechanics_solver_setup(f,setup,None,mode='unloading',mechDT=mechDT)

	f.write('  -num_stim 0 \\\n')
	f.write('  -num_imp_regions 1 \\\n')
	f.write('  -tend 50.0 \\\n')

	tag_list = []
	for t in json_tags:
		tag_list+=json_tags[t]

	passive = ionic(0,'passive','PASSIVE',tag_list,param=None)
	passive.visualise()
	passive.write2file(f)

	simulation_mechanics_regions(f,json_tags,json_data,
				                 material_law=setup.material_law)

	simulation_springs(f,json_data,setup,'inflation')
	simulation_Neumann(f,json_data,json_clinical_data,
					   setup,mode='inflation')
	simulation_surfVols(f,setup)

	simulation_output_options(f,setup,json_clinical_data,json_data,postprocessing,
							  tend=50.0,mode='unloading',
							  get_fibre_strains=get_fibre_strains)

	f.close()

def write_unloading_script(json_file,
						   json_clinical_file,
						   json_tags_file,
						   simulation_script,
						   setup,
						   postprocessing=False):

	f_input = open(json_file,"r")
	json_data = json.load(f_input)
	f_input.close()

	f_clinical_input = open(json_clinical_file,"r")
	json_clinical_data = json.load(f_clinical_input)
	f_clinical_input.close()

	f_tags_input = open(json_tags_file,"r")
	json_tags = json.load(f_tags_input)
	f_tags_input.close()
	
	f = open(simulation_script,"w")
	write_script_header(f,setup)

	simulation_options(f,setup.folderopts)

	f.write('  -simID '+setup.simulation_folder+'/'+setup.testname+' \\\n')
	f.write('  -meshname '+setup.meshdir+'/'+setup.meshname+' \\\n')
	if setup.fibresname is not None:
		# you can give a separate .lon file. If you do, the 
		# MESHNAME.lon file is completely ignored
		f.write('	-orthoname '+setup.meshdir+'/'+setup.fibresname+' \\\n') 

	simulation_electrophysiology_solver_setup(f)
	simulation_mechanics_solver_setup(f,setup,None,mode='unloading')

	f.write('  -num_stim 0 \\\n')
	f.write('  -num_imp_regions 1 \\\n')

	tag_list = []
	for t in json_tags:
		tag_list+=json_tags[t]

	passive = ionic(0,'passive','PASSIVE',tag_list,param=None)
	passive.visualise()
	passive.write2file(f)

	if not setup.unload_atria:
		if ("bf_aa_scaling" in json_data["mechanics"]) and (json_data["mechanics"]["bf_aa_scaling"]!=1):
			raise Exception("You should not set a scaling != 1 for the atria for the unloading if you are not unloading them.")

	if ("bf_lv_scaling" in json_data["mechanics"]) or ("bf_rv_scaling" in json_data["mechanics"]) or ("a_lvrv" in json_data["mechanics"]):
		if setup.material_law!="Guccione":
			raise Exception("The scaling factor for bf is setup only for the Guccione law. Please remove bf_lv_scaling from your json file.")

	simulation_mechanics_regions(f,json_tags,json_data,
				                 material_law=setup.material_law,
				                 scale_bf=True)

	simulation_springs(f,json_data,setup,'unloading')
	simulation_Neumann(f,json_data,json_clinical_data,
					   setup,mode='unloading')
	simulation_surfVols(f,setup)

	f.write('  -experiment 5 \\\n')
	f.write('  -loadStepping '+str(setup.loadStepping)+' \\\n')
	f.write('  -unload_err 1 \\\n')
	f.write('  -unload_conv 0 \\\n')
	f.write('  -unload_tol 1e-03 \\\n')
	f.write('  -unload_maxit 10 \\\n')
	f.write('  -unload_avgwin 1 \\\n')
	f.write('  -unload_stagtol 1.1 \\\n')

	simulation_output_options(f,setup,json_clinical_data,json_data,postprocessing,mode='unloading')

	f.close()

def write_simulation_script(json_file,
							json_clinical_file,
							json_tags_file,
							simulation_script,
							setup,
							postprocessing=False,
							adapt_tubeArea=False,
							save_param_string=None):
	
	f_input = open(json_file,"r")
	json_data = json.load(f_input)
	f_input.close()

	f_clinical_input = open(json_clinical_file,"r")
	json_clinical_data = json.load(f_clinical_input)
	f_clinical_input.close()
	
	f_tags_input = open(json_tags_file,"r")
	json_tags = json.load(f_tags_input)
	f_tags_input.close()

	f = open(simulation_script,"w")
	write_script_header(f,setup)

	simulation_options(f,setup.folderopts)

	f.write('  -simID '+setup.simulation_folder+'/'+setup.testname+' \\\n')
	f.write('  -meshname '+setup.meshdir+'/'+setup.meshname+' \\\n')
	if setup.fibresname is not None:
		f.write('	-orthoname '+setup.meshdir+'/'+setup.fibresname+' \\\n')

	simulation_electrophysiology_solver_setup(f)
	simulation_mechanics_solver_setup(f,setup,
									  NBEATS=setup.nbeats,
									  mode='cycle')

	simulation_electrophysiology_setup(f,setup,json_data,json_clinical_data)
	simulation_stimuli_setup(f,json_clinical_data,json_data,setup)
	simulation_electrophysiology_regions_setup(f,json_tags,json_data)
	simulation_ionic_setup(f,json_tags,json_data,setup,save_param_string=save_param_string)

	simulation_mechanics_regions(f,json_tags,json_data,
				                 material_law=setup.material_law)

	simulation_springs(f,json_data,setup,'cycle')
	simulation_Neumann(f,json_data,json_clinical_data,
					   setup,mode='cycle')
	simulation_surfVols(f,setup)
	simulation_circadapt_setup(f,json_clinical_data,json_data,setup)

	simulation_circadapt_parameters(f,setup,json_data,json_clinical_data,adapt_tubeArea=adapt_tubeArea)

	f.write('  -loadStepping '+str(setup.loadStepping)+' \\\n')

	simulation_output_options(f,setup,json_clinical_data,json_data,
							  postprocessing,
							  NBEATS=setup.nbeats,
							  start_state=setup.restart_state,
							  mode='cycle')

	f.close()

def write_free_contraction_script(json_file,
								  json_clinical_file,
								  json_tags_file,
								  simulation_script,
								  setup,
								  postprocessing=False,
								  adapt_tubeArea=False,
								  save_param_string=None,
								  mechDT=1.0):
	
	f_input = open(json_file,"r")
	json_data = json.load(f_input)
	f_input.close()

	f_clinical_input = open(json_clinical_file,"r")
	json_clinical_data = json.load(f_clinical_input)
	f_clinical_input.close()
	
	f_tags_input = open(json_tags_file,"r")
	json_tags = json.load(f_tags_input)
	f_tags_input.close()

	f = open(simulation_script,"w")
	write_script_header(f,setup)

	simulation_options(f,setup.folderopts)

	f.write('  -simID '+setup.simulation_folder+'/'+setup.testname+' \\\n')
	f.write('  -meshname '+setup.meshdir+'/'+setup.meshname+' \\\n')
	if setup.fibresname is not None:
		f.write('	-orthoname '+setup.meshdir+'/'+setup.fibresname+' \\\n')

	simulation_electrophysiology_solver_setup(f)
	simulation_mechanics_solver_setup(f,setup,
									  NBEATS=setup.nbeats,
									  mode='cycle',
									  mechDT=mechDT)

	simulation_electrophysiology_setup(f,setup,json_data,json_clinical_data)
	simulation_stimuli_setup(f,json_clinical_data,json_data,setup)
	simulation_electrophysiology_regions_setup(f,json_tags,json_data)
	simulation_ionic_setup(f,json_tags,json_data,setup,save_param_string=save_param_string)

	simulation_mechanics_regions(f,json_tags,json_data,
				                 material_law=setup.material_law)

	simulation_springs(f,json_data,setup,'cycle')
	simulation_Neumann(f,json_data,json_clinical_data,
					   setup,mode='cycle')
	simulation_surfVols(f,setup)

	simulation_output_options(f,setup,json_clinical_data,json_data,
							  postprocessing,
							  NBEATS=setup.nbeats,
							  start_state=setup.restart_state,
							  mode='inflation')

	f.close()

def simulation_options(f,
					   FOLDEROPTS):
	f.write('  -ellip_use_pt 0 \\\n')
	f.write('  -parab_use_pt 0 \\\n')
	f.write('  -purk_use_pt 0 \\\n')
	f.write('  -mech_use_pt 0 \\\n')
	f.write('  -parab_options_file '+FOLDEROPTS+'/ilu_cg_opts \\\n')
	f.write('  -ellip_options_file '+FOLDEROPTS+'/amg_cg_opts \\\n')
	f.write('  -mechanics_options_file '+FOLDEROPTS+'/gamg_gmres_opts_agg \\\n')
	f.write('  -mech_finite_element 0 \\\n')

def simulation_electrophysiology_solver_setup(f):
	f.write('  -parab_solve 1 \\\n')
	f.write('  -pstrat_backpermute 1 \\\n')
	f.write('  -localize_pts 1 \\\n')
	f.write('  -cg_tol_ellip 1e-06 \\\n')
	f.write('  -cg_norm_ellip 0 \\\n')
	f.write('  -cg_tol_parab 1e-06 \\\n')
	f.write('  -cg_norm_parab 0 \\\n')
	f.write('  -pstrat 2 \\\n')
	f.write('  -pstrat_i 2 \\\n')
	f.write('  -dt 20.0 \\\n')

def simulation_mechanics_solver_setup(f,
									  setup,
									  NBEATS,
									  mode='cycle',
									  mechDT=1.0):
	maxit_newton = 2
	if not setup.fast_newton:
		maxit_newton = 10

	# f.write('  -load_stiffening 1 \\\n') # new carp version has this as part of mechanic_nbc and it is set to 1 by default in any case
	f.write('  -mech_configuration 1 \\\n')
	f.write('  -mech_tangent_comp 1 \\\n')
	f.write('  -krylov_tol_mech 1e-04 \\\n')
	f.write('  -krylov_norm_mech 0 \\\n')
	f.write('  -krylov_maxit_mech 1000 \\\n')
	#f.write('  -newton_atol_mech 1e-04 \\\n')
	f.write('  -newton_tol_mech 1e-04 \\\n')
	# f.write('  -newton_adaptive_tol_mech 2 \\\n')    # switch to 1 possibly?
	f.write('  -newton_adaptive_tol_mech 0 \\\n')    
	f.write('  -newton_tol_cvsys 1e-04 \\\n')
	f.write('  -newton_line_search 0 \\\n')
	f.write('  -newton_maxit_mech '+str(maxit_newton)+' \\\n')

	if mode=='cycle':
		if NBEATS is not None:
			f.write('  -newton_num_init_beats[0] '+str(setup.initial_nbeats)+' \\\n')
			f.write('  -newton_num_init_beats[1] '+str(max(NBEATS-setup.full_nbeats,0))+' \\\n')
			f.write('  -IO_num_init_beats '+str(max(NBEATS-1,0))+' \\\n')
		else:
			raise Exception('You need to provide the number of beats for cycle simulations.')

	f.write('  -newton_forcing_term 2 \\\n')
	f.write('  -mapping_mode 1 \\\n')
	f.write('  -redist_extracted 0 \\\n')
	f.write('  -mass_lumping 1 \\\n')
	f.write('  -mech_activate_inertia 1 \\\n')
	f.write('  -mech_rho_inf 0.0 \\\n')
	if hasattr(setup,"mech_stiffness_damping"):
		f.write('  -mech_stiffness_damping '+str(setup.mech_stiffness_damping)+' \\\n')
	else:
		f.write('  -mech_stiffness_damping 0.1 \\\n')
	f.write('  -mech_mass_damping 0.1 \\\n')
	f.write('  -mech_vol_split_aniso 0 \\\n')
	f.write('  -mechDT '+str(mechDT)+' \\\n')

def simulation_electrophysiology_setup(f,
									   setup,
                    				   json_data,
                    				   json_clinical_data):

	f.write('  -diffusionOn 0 \\\n')
	f.write('  -bidomain 0 \\\n')

	if hasattr(setup, 'startsim'):
		startsim = setup.startsim
		if startsim not in ["diastasis","diastole"]:
			raise Exception("Please choose startsim to be diastasis or diastole")
	else:
		startsim = "diastole"

	# everything is initialised @ ED @ -100 AVD
	BCL = json_clinical_data["general"]["BCL"]

	if "AV_delay" in json_data["EP"]:
		AV_delay = json_data["EP"]["AV_delay"]
	else:
		if setup.AV_delay is not None:
			AV_delay = setup.AV_delay
		else:
			raise Exception("You need to setup the AV delay in either the setup file or in the parameter file")

	print('#######################################')
	print('    visualising setup for simulation   ')
	print('#######################################')

	if hasattr(setup, 'ERP'):
		if setup.ERP is None:
			ERP = BCL*0.8
		else:
			ERP = setup.ERP
	else:
		ERP = BCL*0.8

	if "VV_delay" in json_data["EP"]:
		VV_delay = json_data["EP"]["VV_delay"]
	elif hasattr(setup, 'VV_delay'):
		if setup.VV_delay is None:
			VV_delay = 0.0
		else:
			VV_delay = setup.VV_delay
	else:
		VV_delay = 0.0

	if startsim=='diastole':
		start = -AV_delay
	elif startsim=='diastasis':
		start = 0.0

	clock = fcclock(BCL,
					start,
					AV_delay,
					ERP=ERP,
					VV_delay=VV_delay)
	clock.visualise()
	clock.write2file(f)

def simulation_stimuli_setup(f,
                             json_clinical_data,
                             json_data,
                             setup):

	BCL = json_clinical_data["general"]["BCL"]

	if "AV_delay" in json_data["EP"]:
		AV_delay = json_data["EP"]["AV_delay"]
	else:
		if setup.AV_delay is not None:
			AV_delay = setup.AV_delay
		else:
			raise Exception("You need to setup the AV delay in either the setup file or in the parameter file")

	# everything is initialised @ ED @ -AVD
	# f.write('  -tend '+str(int(setup.nbeats*BCL-AV_delay))+' \\\n')

	# everything is initialised @ ED but to save the correct number of time steps,
	# you need to run BCL*Nbeats
	f.write('  -tend '+str(int(setup.nbeats*BCL))+' \\\n')

	f.write('  -num_trgs '+str(len(setup.stimuli))+' \\\n')

	for i,s in enumerate(setup.stimuli):
		f.write('  -trigger['+str(i)+'].src 0 \\\n')
		f.write('  -trigger['+str(i)+'].clock_ID '+str(setup.clock_ID[i])+' \\\n')

	f.write('  -num_stim '+str(len(setup.stimuli)+1)+' \\\n')
	for i,s in enumerate(setup.stimuli):
		stim = stimulus(i,s,BCL,setup.nbeats,i,0,
					  	vtx_file=setup.meshdir+'/'+s)
		stim.visualise()
		stim.write2file(f)

	propagation = stimulus(len(setup.stimuli),'propagation',BCL,setup.nbeats,1,8)
	propagation.visualise()
	propagation.write2file(f)

def simulation_electrophysiology_regions_setup(f,
											   json_tags,
				                               json_data):
	ANI_RATIO_ventricles=json_data["EP"]["ani_ratio_ventricles"]
	ANI_RATIO_atria=json_data["EP"]["ani_ratio_atria"]
	K_FEC=json_data["EP"]["k_FEC"]
	K_BB=json_data["EP"]["k_BB"]

	CV_ventricles = json_data["EP"]["CV_ventricles"]
	CV_atria = json_data["EP"]["CV_atria"]

	f.write('  -num_gregions 5 \\\n')
	f.write('  -num_ekregions 5 \\\n')

	if "ventricles" in json_tags:
		check =  any(t in json_tags for t in ["lv","rv"])
		if check:
			raise Exception('If you want to set up the "ventricles" label, you should not set the "lv" and the "rv".')
		else:
			ventricle_tags = json_tags["ventricles"]

	elif (all(t in json_tags for t in ["lv","rv"])):
		ventricle_tags = json_tags["lv"]+json_tags["rv"]

	else:
		raise Exception("You haven't set the tags for neither the ventricles nor the lv and the rv.")

	ventricles = ep_region('ventricles',0,ventricle_tags,
					[CV_ventricles,
					 CV_ventricles*ANI_RATIO_ventricles,
					 CV_ventricles*ANI_RATIO_ventricles])
	ventricles.visualise()
	ventricles.write2file(f)

	if "fast_endo" in json_tags:
		check =  any(t in json_tags for t in ["fast_endo_lv","fast_endo_rv","fast_endo_sv"])
		if check:
			raise Exception('If you want to set up the "fast_endo" label, you should not set the "fast_endo_lv", "fast_endo_rv" and the "fast_endo_sv".')
		else:
			fec_tags = json_tags["fast_endo"]

	elif all(t in json_tags for t in ["fast_endo_lv","fast_endo_rv","fast_endo_sv"]):
		fec_tags = json_tags["fast_endo_lv"]+json_tags["fast_endo_rv"]+json_tags["fast_endo_sv"]

	else:
		raise Exception("You haven't set the tags for neither the fec nor the lv,rv and sv fec.")

	fec = ep_region('FEC_ventricles',1,fec_tags,
					[CV_ventricles*K_FEC,
					 CV_ventricles*ANI_RATIO_ventricles*K_FEC,
					 CV_ventricles*ANI_RATIO_ventricles*K_FEC])
	fec.visualise()
	fec.write2file(f)

	atria = ep_region('atria',2,json_tags["atria"],
  					  [CV_atria,
  					   CV_atria*ANI_RATIO_atria,
  					   CV_atria*ANI_RATIO_atria])
	atria.visualise()
	atria.write2file(f)

	bb = ep_region('Bachmann_bundle',3,json_tags["bachmann_bundle"],
				   [CV_atria*K_BB,
				    CV_atria*ANI_RATIO_atria*K_BB,
				    CV_atria*ANI_RATIO_atria*K_BB])
	bb.visualise()
	bb.write2file(f)

	passive_IDs_list = json_tags["aorta"]+json_tags["pulmonary_artery"]+json_tags["vein_rings"]+json_tags["valve_planes"]+json_tags["AV_plane"]
	passive = ep_region('passive',4,passive_IDs_list,
						[1.0,1.0,1.0],ignore=True)
	passive.visualise()
	passive.write2file(f)

def simulation_ionic_setup(f,
						   json_tags,
				           json_data,
				           setup,
				           save_param_string=None):	

	if save_param_string is not None:
		idx = save_param_string.rfind("/")
		save_param_folder = save_param_string[:idx+1]
		os.system("mkdir -p "+save_param_folder)

	im_param_ventricles = construct_param_string(json_data["ToRORd"],
												 header='flags=ENDO,')

	if save_param_string is not None:
		text_file = open(save_param_string+'param_ToRORd_dynCl.txt', 'w')
		n = text_file.write(im_param_ventricles)
		text_file.close()

	imp_idx_atria = 1

	if all(t in json_tags for t in ["ventricles","fast_endo"]):
		if any(t in json_tags for t in ["lv","rv","fast_endo_lv","fast_endo_rv","fast_endo_sv"]):
			raise Exception('If you want to define the ventricles with the same properties, do not define lv and rv in the tag file.')
		else:
			f.write('  -num_imp_regions 3 \\\n')
			
			IDs_ventricles_list = json_tags["ventricles"]+json_tags["fast_endo"]	

			# previously read the parameter settings from a string
			# fID_t = open(setup.torord_param_string, "r")
			# im_param_ventricles = fID_t.read()
			# fID_t.close()	

			if setup.contraction_model=='Land':

				# previously read the parameter settings from a string
				# fID_t = open(setup.torord_land_param_string, "r")
				# im_param_ventricles_mech = fID_t.read()
				# fID_t.close()

				im_param_ventricles_mech = construct_param_string(json_data["ToRORd_land"])
				if save_param_string is not None:
					text_file = open(save_param_string+'param_ToRORd_dynCl_Land.txt', 'w')
					n = text_file.write(im_param_ventricles_mech)
					text_file.close()

				ventricles = ionic(0,'ventricles','ToRORd_dynCl',IDs_ventricles_list,param=im_param_ventricles,
								   plugin='LandHumanStress',plug_param=im_param_ventricles_mech,
								   im_sv_init=setup.torord_init_file)

			elif setup.contraction_model=='tanh':

				# previously read the parameter settings from a string

				# fID_t = open(setup.torord_tanh_param_string, "r")
				# im_param_ventricles_mech = fID_t.read()
				# fID_t.close()

				im_param_ventricles_mech = construct_param_string(json_data["ToRORd_tanh"])
				if save_param_string is not None:
					text_file = open(save_param_string+'param_ToRORd_dynCl_tanh.txt', 'w')
					n = text_file.write(im_param_ventricles_mech)
					text_file.close()

				ventricles = ionic(0,'ventricles','ToRORd_dynCl',IDs_ventricles_list,param=im_param_ventricles,
								   plugin='TanhStress',plug_param=im_param_ventricles_mech,
								   im_sv_init=setup.torord_init_file)

			else:
				raise Exception('Contraction model not recognised. Pick between Land and tanh')
			ventricles.visualise()
			ventricles.write2file(f)				

	elif all(t in json_tags for t in ["lv","rv","fast_endo_lv","fast_endo_rv","fast_endo_sv"]):

		f.write('  -num_imp_regions 4 \\\n')

		# previously read the parameter settings from a string
		# fID_t = open(setup.torord_param_string, "r")
		# im_param_ventricles = fID_t.read()
		# fID_t.close()		
			
		IDs_lv_list = json_tags["lv"]+json_tags["fast_endo_lv"]+json_tags["fast_endo_sv"]		

		if setup.contraction_model=='Land':

			# previously read the parameter settings from a string
			# fID_t = open(setup.torord_land_param_string, "r")
			# im_param_lv_mech = fID_t.read()
			# fID_t.close()

			im_param_lv_mech = construct_param_string(json_data["ToRORd_land"])
			if save_param_string is not None:
				text_file = open(save_param_string+'param_ToRORd_dynCl_Land.txt', 'w')
				n = text_file.write(im_param_lv_mech)
				text_file.close()

			lv = ionic(0,'lv','ToRORd_dynCl',IDs_lv_list,param=im_param_ventricles,
					   plugin='LandHumanStress',plug_param=im_param_lv_mech,
					   im_sv_init=setup.torord_init_file)

		elif setup.contraction_model=='tanh':

			# previously read the parameter settings from a string
			# fID_t = open(setup.torord_tanh_param_string, "r")
			# im_param_lv_mech = fID_t.read()
			# fID_t.close()

			im_param_lv_mech = construct_param_string(json_data["ToRORd_tanh"])
			if save_param_string is not None:
				text_file = open(save_param_string+'param_ToRORd_dynCl_tanh.txt', 'w')
				n = text_file.write(im_param_lv_mech)
				text_file.close()

			lv = ionic(0,'lv','ToRORd_dynCl',IDs_lv_list,param=im_param_ventricles,
					   plugin='TanhStress',plug_param=im_param_lv_mech,
					   im_sv_init=setup.torord_init_file)

		else:
			raise Exception('Contraction model not recognised. Pick between Land and tanh')

		lv.visualise()
		lv.write2file(f)
		
		IDs_rv_list = json_tags["rv"]+json_tags["fast_endo_rv"]	
		
		if setup.contraction_model=='Land':

			# previously read the parameter settings from a string
			# fID_t = open(setup.torord_land_rv_param_string, "r")
			# im_param_rv_mech = fID_t.read()
			# fID_t.close()

			json_data_rv = copy.deepcopy(json_data["ToRORd_land"])

			if json_data["mechanics"]["Tref_lvrv"] != 1.0:
				json_data_rv["Tref"] = json_data["ToRORd_land"]["Tref"]*json_data["mechanics"]["Tref_lvrv"]

			im_param_rv_mech = construct_param_string(json_data_rv)
			if save_param_string is not None:
				text_file = open(save_param_string+'param_ToRORd_dynCl_Land_rv.txt', 'w')
				n = text_file.write(im_param_rv_mech)
				text_file.close()

			rv = ionic(1,'rv','ToRORd_dynCl',IDs_rv_list,param=im_param_ventricles,
					   plugin='LandHumanStress',plug_param=im_param_rv_mech,
					   im_sv_init=setup.torord_rv_init_file)

		elif setup.contraction_model=='tanh':

			# previously read the parameter settings from a string
			# fID_t = open(setup.torord_tanh_rv_param_string, "r")
			# im_param_rv_mech = fID_t.read()
			# fID_t.close()

			json_data_rv = copy.deepcopy(json_data["ToRORd_tanh"])

			if json_data["mechanics"]["Tref_lvrv"] != 1.0:
				json_data_rv["Tpeak"] = json_data["ToRORd_tanh"]["Tpeak"]*json_data["mechanics"]["Tref_lvrv"]

			im_param_rv_mech = construct_param_string(json_data_rv)
			if save_param_string is not None:
				text_file = open(save_param_string+'param_ToRORd_dynCl_tanh_rv.txt', 'w')
				n = text_file.write(im_param_rv_mech)
				text_file.close()

			rv = ionic(1,'rv','ToRORd_dynCl',IDs_rv_list,param=im_param_ventricles,
					   plugin='TanhStress',plug_param=im_param_rv_mech,
					   im_sv_init=setup.torord_rv_init_file)
		else:
			raise Exception('Contraction model not recognised. Pick between Land and tanh')
		rv.visualise()
		rv.write2file(f)		

		imp_idx_atria = 2

	im_param_atria = construct_param_string(json_data["COURTEMANCHE"])
	if save_param_string is not None:
		text_file = open(save_param_string+'param_JB_COURTEMANCHE.txt', 'w')
		n = text_file.write(im_param_atria)
		text_file.close()

	# previously read the parameter settings from a string
	# fID_t = open(setup.courtemanche_param_string, "r")
	# im_param_atria = fID_t.read()
	# fID_t.close()

	IDs_atria_list = json_tags["atria"]+json_tags["bachmann_bundle"]

	if setup.contraction_model=='Land':

		# previously read the parameter settings from a string
		# fID_t = open(setup.courtemanche_land_param_string, "r")
		# im_param_atria_mech = fID_t.read()
		# fID_t.close()

		im_param_atria_mech = construct_param_string(json_data["COURTEMANCHE_land"])
		if save_param_string is not None:
			text_file = open(save_param_string+'param_JB_COURTEMANCHE_Land.txt', 'w')
			n = text_file.write(im_param_atria_mech)
			text_file.close()

		atria = ionic(imp_idx_atria,'atria','JB_COURTEMANCHE',IDs_atria_list,param=im_param_atria,
					  plugin='LandHumanStress',plug_param=im_param_atria_mech,
					  im_sv_init=setup.courtemanche_init_file)

	elif setup.contraction_model=='tanh':

		# previously read the parameter settings from a string
		# fID_t = open(setup.courtemanche_tanh_param_string, "r")
		# im_param_atria_mech = fID_t.read()
		# fID_t.close()

		im_param_atria_mech = construct_param_string(json_data["COURTEMANCHE_tanh"])
		if save_param_string is not None:
			text_file = open(save_param_string+'param_JB_COURTEMANCHE_tanh.txt', 'w')
			n = text_file.write(im_param_atria_mech)
			text_file.close()

		atria = ionic(imp_idx_atria,'atria','JB_COURTEMANCHE',IDs_atria_list,param=im_param_atria,
					  plugin='TanhStress',plug_param=im_param_atria_mech,
					  im_sv_init=setup.courtemanche_init_file)

	else:
		raise Exception('Contraction model not recognised. Pick between Land and tanh')

	atria.visualise()
	atria.write2file(f)

	passive_IDs_list = json_tags["aorta"]+json_tags["pulmonary_artery"]+json_tags["vein_rings"]+json_tags["valve_planes"]+json_tags["AV_plane"]
	
	passive = ionic(imp_idx_atria+1,'passive','PASSIVE',passive_IDs_list,param=None)
	passive.visualise()
	passive.write2file(f)

	f.write('  -mech_use_actStress 1 \\\n')
	f.write('  -active_stress_mode 5 \\\n') # dispersion
	f.write('  -veldep 1 \\\n')
	f.write('  -mech_lambda_upd 2 \\\n')
	f.write('  -mech_deform_elec 0 \\\n')

def simulation_mechanics_regions(f,
							     json_tags,
				                 json_data,
				                 material_law='Usyk',
				                 scale_bf=False):

	dispersion = False
	dct_param_ventricles = {}
	dct_param_atria = {}
	if material_law=='Holzapfel':
		param_labels = ['a','a_f','a_s','a_n','a_fs','a_fn','a_ns',
						'b','b_f','b_s','b_n','b_fs','b_fn','b_ns']
		for p in param_labels:
			dct_param_ventricles[p] = json_data['mechanics'][p.replace('_','')+'_ventricles']
			dct_param_atria[p] = json_data['mechanics'][p.replace('_','')+'_atria']

		dispersion=True
		mat_type = 17
		
	elif material_law=='Guccione':
		param_labels = ['a','b_f','b_fs','b_t']

		for p in param_labels:
			dct_param_ventricles[p] = json_data['mechanics'][p.replace('_','')+'_ventricles']
			dct_param_atria[p] = json_data['mechanics'][p.replace('_','')+'_atria']

		mat_type = 9

	elif material_law=='Usyk':
		param_labels = ['a','b_ff','b_ss','b_nn','b_fs','b_fn','b_ns']

		for p in param_labels:
			dct_param_ventricles[p] = json_data['mechanics'][p.replace('_','')+'_ventricles']
			dct_param_atria[p] = json_data['mechanics'][p.replace('_','')+'_atria']

		mat_type = 14

	else:
		raise Exception('Material law not recognised. Choose between Holzapfel, Guccione, Usyk')

	if dispersion:
		tail=',delta_f=0,delta_s=0,delta_n=0,h_k=0'
	else:
		tail=''

	mreg_idx_atria = 1
	if all(t in json_tags for t in ["ventricles","fast_endo"]):

		if "bf_rv_scaling" in json_data["mechanics"]:
			raise Exception("You are not separating tags for LV and RV so the RV will be assigned with the LV parameters. Remove bf_rv_scaling from your parameters.")

		if any(t in json_tags for t in ["lv","rv","fast_endo_lv","fast_endo_rv","fast_endo_sv"]):
			raise Exception('If you want to define the ventricles with the same properties, do not define lv and rv in the tag file.')
		else:
			f.write('  -num_mregions 6 \\\n')

			if scale_bf and ("bf_lv_scaling" in json_data["mechanics"]):
				dct_param_ventricles['b_f'] = dct_param_ventricles['b_f']*json_data["mechanics"]["bf_lv_scaling"]

			param_string_ventricles = construct_param_string(dct_param_ventricles,
															 header='kappa=1000.0,',
															 param_labels=param_labels,
															 tail=tail)

			IDs_ventricles_list = json_tags["ventricles"]+json_tags["fast_endo"]+json_tags["AV_plane"]
			mregion_ventricles = mregion('ventricles',0,IDs_ventricles_list,mat_type,param_string_ventricles)
			mregion_ventricles.visualise()
			mregion_ventricles.write2file(f)

	elif all(t in json_tags for t in ["lv","rv","fast_endo_lv","fast_endo_rv","fast_endo_sv"]):

		mreg_idx_atria = 2

		if "a_lvrv" in json_data['mechanics']:
			a_lvrv = json_data['mechanics']['a_lvrv']
		else:
			a_lvrv = 1.0

		f.write('  -num_mregions 7 \\\n')

		rv_dct_param_ventricles = copy.deepcopy(dct_param_ventricles)
		if scale_bf and ("bf_lv_scaling" in json_data["mechanics"]):
			dct_param_ventricles['b_f'] = dct_param_ventricles['b_f']*json_data["mechanics"]["bf_lv_scaling"]

		param_string_ventricles = construct_param_string(dct_param_ventricles,
															 header='kappa=1000.0,',
															 param_labels=param_labels,
															 tail=tail)

		IDs_lv_list = json_tags["lv"]+json_tags["fast_endo_lv"]+json_tags["fast_endo_sv"]+json_tags["AV_plane"]		

		mregion_lv = mregion('lv',0,IDs_lv_list,mat_type,param_string_ventricles)
		mregion_lv.visualise()
		mregion_lv.write2file(f)

		IDs_rv_list = json_tags["rv"]+json_tags["fast_endo_rv"]
		rv_dct_param_ventricles["a"] = a_lvrv*dct_param_ventricles["a"]

		if scale_bf and ("bf_rv_scaling" in json_data["mechanics"]):
			rv_dct_param_ventricles["b_f"] = rv_dct_param_ventricles["b_f"]*json_data["mechanics"]["bf_rv_scaling"]

		param_string_rv = construct_param_string(rv_dct_param_ventricles,
														 header='kappa=1000.0,',
														 param_labels=param_labels,
														 tail=tail) 

		mregion_rv = mregion('rv',1,IDs_rv_list,mat_type,param_string_rv)
		mregion_rv.visualise()
		mregion_rv.write2file(f)	

		mreg_idx_atria = 2

	if scale_bf and ("bf_aa_scaling" in json_data["mechanics"]):
		dct_param_atria['b_f'] = dct_param_atria['b_f']*json_data["mechanics"]["bf_aa_scaling"]

	param_string_atria = construct_param_string(dct_param_atria,
												header='kappa=1000.0,',
												param_labels=param_labels,
												tail=tail)

	IDs_atria_list = json_tags["atria"]+json_tags["bachmann_bundle"]
	mregion_atria = mregion('atria',mreg_idx_atria,IDs_atria_list,mat_type,param_string_atria)
	mregion_atria.visualise()
	mregion_atria.write2file(f)

	mregion_valves = mregion('valve_planes',mreg_idx_atria+1,json_tags["valve_planes"],2,'kappa=1000.0,c=1000.0')
	mregion_valves.visualise()
	mregion_valves.write2file(f)

	mregion_aorta = mregion('aorta',mreg_idx_atria+2,json_tags["aorta"],2,'kappa=1000.0,c=26.66')
	mregion_aorta.visualise()
	mregion_aorta.write2file(f)

	mregion_pa = mregion('pulmonary_artery',mreg_idx_atria+3,json_tags["pulmonary_artery"],2,'kappa=1000.0,c=3.7')
	mregion_pa.visualise()
	mregion_pa.write2file(f)

	mregion_veins = mregion('vein_rings',mreg_idx_atria+4,json_tags["vein_rings"],2,'kappa=1000.0,c=7.45')
	mregion_veins.visualise()
	mregion_veins.write2file(f)

def simulation_springs(f,
				       json_data,
				       setup,
				       mode):
	
	if setup.cycle_peri_on or mode=='unloading' or mode=='inflation':
		f.write('  -nspring_update 1 \\\n') # spring reference updated after loading - but what about nspring_config?
		f.write('  -num_mechanic_ed 1 \\\n')
		f.write('  -num_mechanic_bs '+str(1+len(setup.springs))+' \\\n')
	else:
		f.write('  -num_mechanic_bs '+str(len(setup.springs))+' \\\n')		

	if mode=='cycle' or mode=='inflation':
		if setup.cycle_peri_on:
			k_peri = json_data["mechanics"]["k_peri"]
		else:
			k_peri = 0.0
	elif mode=='unloading':
		if setup.load_peri_on:
			k_peri = json_data["mechanics"]["k_peri"]
		else:
			k_peri = 0.0

	if setup.cycle_peri_on or mode=='unloading' or mode=='inflation':

		if setup.peri_scale is None:
			raise Exception("If you want the pericardium, you need to provide a pericardium scale")

		i = 1
		peri = spring('pericardium',0,k_peri,ncomp=1,elem_file=setup.meshdir+'/'+setup.peri_scale)
		peri.visualise()
		peri.write2file(f)
	else:
		i = 0

	for j,bc in enumerate(setup.springs):
		sbc = spring(bc,i,setup.k_springs[j])
		sbc.visualise()
		sbc.write2file(f)
		i += 1

def simulation_Neumann(f,
					   json_data,
					   json_clinical_data,
					   setup,
				       mode='cycle'):

	trace_file = None
	if mode=='inflation':
		# p_shift_diastasis = json_data["mechanics"]["p_shift_diastasis"]
		# EDP = json_clinical_data["pressure"]["EDP"]

		# lvrv_ratio = setup.EDP_lvrv_ratio

		if "p_shift_diastasis" in json_data["mechanics"]:
			raise Exception("Updated: p_shift_diastasis was removed and EDP_lv was included as a parameter")

		pLV = json_data["mechanics"]["EDP_lv"]*mmHg_to_kPa #mmHg
		pRV = json_data["mechanics"]["EDP_rv"]*mmHg_to_kPa
		# pLV = 9.0*mmHg_to_kPa # Alboni et al 1995 PACE
		# pRV = 5.6*mmHg_to_kPa # Alboni et al 1995 PACE
		pLA = pLV
		pRA = pRV

		if setup.trace_file is None:
			raise Exception("You need to provide a trace file to inflate")
		else:
			trace_file = setup.trace_file
	elif mode=='unloading':

		if "p_shift_diastasis" in json_data["mechanics"]:
			raise Exception("Updated: p_shift_diastasis was removed and EDP_lv was included as a parameter")

		# lvrv_ratio = setup.EDP_lvrv_ratio # Alboni et al 1995 PACE

		pLV = json_data["mechanics"]["EDP_lv"]*mmHg_to_kPa
		# pLV = 9.0*mmHg_to_kPa # Alboni et al 1995 PACE
		pRV = json_data["mechanics"]["EDP_rv"]*mmHg_to_kPa    # Alboni et al 1995 PACE

		if not setup.unload_atria:
			pLA = 0.0
			pRA = 0.0
		else:
			pLA = pLV
			pRA = pRV	

	elif mode=='cycle':
		pLV = 0.0
		pRV = 0.0
		pLA = 0.0
		pRA = 0.0
	else:
		raise Exception('Mode not recognised: please choose between cycle, inflation or unloading')
	
	if setup.cycle_peri_on or mode=='unloading' or mode=='inflation':
		f.write('  -num_mechanic_nbc '+str(5+len(setup.springs))+' \\\n')
	else:
		f.write('  -num_mechanic_nbc '+str(4+len(setup.springs))+' \\\n')

	lvendo = mech_nbc('lv_endo',0,setup.meshdir+'/'+setup.lvendo_name,pressure=pLV,spring=False,trace_file=trace_file)
	lvendo.visualise()
	lvendo.write2file(f)

	rvendo = mech_nbc('rv_endo',1,setup.meshdir+'/'+setup.rvendo_name,pressure=pRV,spring=False,trace_file=trace_file)
	rvendo.visualise()
	rvendo.write2file(f)

	laendo = mech_nbc('la_endo',2,setup.meshdir+'/'+setup.laendo_name,pressure=pLA,spring=False,trace_file=trace_file)
	laendo.visualise()
	laendo.write2file(f)

	raendo = mech_nbc('ra_endo',3,setup.meshdir+'/'+setup.raendo_name,pressure=pRA,spring=False,trace_file=trace_file)
	raendo.visualise()
	raendo.write2file(f)

	if setup.load_peri_on:
		peri_nspring_config = 1
	else:
		peri_nspring_config = 2

	if setup.cycle_peri_on or mode=='unloading' or mode=='inflation':
		peri = mech_nbc('pericardium',4,setup.meshdir+'/'+setup.epi_name,spring=True,spring_idx=-1,nspring_idx=0,
				   		nspring_config=peri_nspring_config)
		peri.visualise()
		peri.write2file(f)
		iS = 1
		iBC = 5

	else:
		iS = 0
		iBC = 4
	
	for bc in setup.springs:
		sbc = mech_nbc(bc,iBC,setup.meshdir+'/'+bc,spring=True,spring_idx=iS,nspring_idx=-1)
		sbc.visualise()
		sbc.write2file(f)

		iBC += 1
		iS += 1

def define_surf_volume(f,
					   sID,
					   name,
					   surf_file,
					   grid=8):
	f.write('  -surfVols['+str(sID)+'].name '+name+' \\\n')
	f.write('  -surfVols['+str(sID)+'].surf_file '+surf_file+' \\\n')
	f.write('  -surfVols['+str(sID)+'].grid '+str(grid)+' \\\n')	

def simulation_surfVols(f,
					    setup):
	f.write('  -numSurfVols 4 \\\n')

	define_surf_volume(f,0,'lv_endo',setup.meshdir+'/'+setup.lvendo_name,grid=8)
	define_surf_volume(f,1,'rv_endo',setup.meshdir+'/'+setup.rvendo_name,grid=8)
	define_surf_volume(f,2,'la_endo',setup.meshdir+'/'+setup.laendo_name,grid=8)
	define_surf_volume(f,3,'ra_endo',setup.meshdir+'/'+setup.raendo_name,grid=8)

	f.write('  -volumeTracking 1 \\\n')
	f.write('  -numElemVols 1 \\\n')
	f.write('  -elemVols[0].name tissue \\\n')
	f.write('  -elemVols[0].grid 8 \\\n')
	f.write('  -elemVols[0].numtags 0 \\\n')

def simulation_circadapt_setup(f,
							   json_clinical_data,
						 	   json_data,
						 	   setup):
	f.write('  -CVS_mode 1 \\\n')
	f.write('  -num_cavities 4 \\\n')

	if "p_shift_diastasis" in json_data["mechanics"]:
		raise Exception("Updated: p_shift_diastasis was removed and EDP_lv was included as a parameter")


	pLV0 = json_data["mechanics"]["EDP_lv"]
	pRV0 = json_data["mechanics"]["EDP_rv"]

	if not setup.unload_atria:
		pLA0 = 0.0001
		pRA0 = 0.0001
	else:
		pLA0 = pLV0
		pRA0 = pRV0

	pAo0 = 80.0
	pPa0 = 15.0
	pPVe0 = 4.0
	pVe0 = 4.0

	lv = cavity('lv',0,0,0,0,pLV0,pLA0,pAo0)
	lv.visualise()
	lv.write2file(f)

	rv = cavity('rv',1,1,1,1,pRV0,pRA0,pPa0)
	rv.visualise()
	rv.write2file(f)

	la = cavity('la',2,2,2,2,pLA0,pPVe0,pLV0)
	la.visualise()
	la.write2file(f)

	ra = cavity('ra',3,3,3,3,pRA0,pVe0,pRV0)
	ra.visualise()
	ra.write2file(f)

def simulation_circadapt_parameters(f,
									setup,
									json_data,
									json_clinical_data,
									adapt_tubeArea=False):
	if "AV_delay" in json_data["EP"]:
		AV_delay = json_data["EP"]["AV_delay"]
	else:
		if setup.AV_delay is not None:
			AV_delay = setup.AV_delay
		else:
			raise Exception("You need to setup the AV delay in either the setup file or in the parameter file")

	f.write('  -cvs_q_rest '+str(json_clinical_data["general"]["qrest"])+' \\\n')
	f.write('  -cvs_cycle_length '+str(json_clinical_data["general"]["BCL"]/1000.)+' \\\n')

	valves = ["mitral","tricuspid","aortic","pulmonary"]
	for valve_name in valves:
		# valve areas
		if valve_name+"_area" not in json_data["circadapt"]:
			f.write('  -cvs_'+valve_name+'_vlv_area '+str(json_clinical_data["valve_areas"][valve_name])+' \\\n')
		else:
			f.write('  -cvs_'+valve_name+'_vlv_area '+str(json_data["circadapt"][valve_name+"_area"])+' \\\n')

	if "sysOrifice_area" not in json_data["circadapt"]:
		syso_area = json_clinical_data["valve_areas"]["sysOrifice"]
	else:
		syso_area = json_data["circadapt"]["sysOrifice_area"]
	f.write('  -cvs_syo_area '+str(syso_area)+' \\\n')

	if "pulmOrifice_area" not in json_data["circadapt"]:
		pulmo_area = json_clinical_data["valve_areas"]["pulmOrifice"]
	else:
		pulmo_area = json_data["circadapt"]["pulmOrifice_area"]
	f.write('  -cvs_puo_area '+str(pulmo_area)+' \\\n')	

	# peripheral circulation
	f.write('  -cvs_systemic_pressure_drop '+str(json_data["circadapt"]["DPsys"])+' \\\n')
	f.write('  -cvs_pulmonary_pressure_drop '+str(json_data["circadapt"]["DPpulm"])+' \\\n')
	f.write('  -cvs_systemic_resistance_factor '+str(json_data["circadapt"]["Rsys"])+' \\\n')
	f.write('  -cvs_pulmonary_resistance_factor '+str(json_data["circadapt"]["Rpulm"])+' \\\n')
	
	# tube parameters
	if not adapt_tubeArea:
		ao_wall_area = json_clinical_data["tubes"]["Ao_wallarea"]
		ap_wall_area = json_clinical_data["tubes"]["Ap_wallarea"]
		vc_wall_area = json_clinical_data["tubes"]["Vc_wallarea"]
		vp_wall_area = json_clinical_data["tubes"]["Pv_wallarea"]
	else:
		if "aortic_area" not in json_data["circadapt"]:
			raise Exception('If you want to recompute the valve areas, you need to provide a new one that is not from clinical data.')
		ao_wall_area = compute_wallarea_tube(json_data["circadapt"]["aortic_area"],2.0)

		if "pulmonary_area" not in json_data["circadapt"]:
			raise Exception('If you want to recompute the valve areas, you need to provide a new one that is not from clinical data.')
		ap_wall_area = compute_wallarea_tube(json_data["circadapt"]["pulmonary_area"],2.0)

		if "sysOrifice_area" not in json_data["circadapt"]:
			raise Exception('If you want to recompute the valve areas, you need to provide a new one that is not from clinical data.')
		vc_wall_area = compute_wallarea_tube(json_data["circadapt"]["sysOrifice_area"],1.0)

		if "pulmOrifice_area" not in json_data["circadapt"]:
			raise Exception('If you want to recompute the valve areas, you need to provide a new one that is not from clinical data.')
		vp_wall_area = compute_wallarea_tube(json_data["circadapt"]["pulmOrifice_area"],1.0)

	f.write('  -cvs_ao_wall_area '+str(ao_wall_area)+' \\\n')	
	f.write('  -cvs_ao_length '+str(json_data["circadapt"]["Aol"])+' \\\n')

	if "Aop0" in json_data["circadapt"]:
		raise Exception("You are trying to set -cvs_ao_p0ref_factor. This is not supported in newer CARP versions. Remove AoP0, Pap0, Vep0 and PVep0 from your json fils.")

	# f.write('  -cvs_ao_p0ref_factor '+str(json_data["circadapt"]["Aop0"])+' \\\n')

	if "AoPref" in json_data["circadapt"]:
		f.write('  -cvs_ao_ref_pressure '+str(json_data["circadapt"]["AoPref"])+' \\\n')

	f.write('  -cvs_arterial_stiffness_exp '+str(json_data["circadapt"]["kArt"])+' \\\n')

	f.write('  -cvs_ap_wall_area '+str(ap_wall_area)+' \\\n')
	f.write('  -cvs_ap_length '+str(json_data["circadapt"]["Pal"])+' \\\n')

	if "Pap0" in json_data["circadapt"]:
		raise Exception("You are trying to set -cvs_ap_p0ref_factor. This is not supported in newer CARP versions")

	if "PaPref" in json_data["circadapt"]:
		f.write('  -cvs_ap_ref_pressure '+str(json_data["circadapt"]["PaPref"])+' \\\n')
		
	# f.write('  -cvs_ap_p0ref_factor '+str(json_data["circadapt"]["Pap0"])+' \\\n')
	f.write('  -cvs_pulmart_stiffness_exp '+str(json_data["circadapt"]["kPArt"])+' \\\n')

	f.write('  -cvs_vc_wall_area '+str(vc_wall_area)+' \\\n')
	f.write('  -cvs_vc_length '+str(json_data["circadapt"]["Vel"])+' \\\n')

	if "Vep0" in json_data["circadapt"]:
		raise Exception("You are trying to set -cvs_vc_p0ref_factor. This is not supported in newer CARP versions")

	if "VePref" in json_data["circadapt"]:
		f.write('  -cvs_vc_ref_pressure '+str(json_data["circadapt"]["VePref"])+' \\\n')

	# f.write('  -cvs_vc_p0ref_factor '+str(json_data["circadapt"]["Vep0"])+' \\\n')
	f.write('  -cvs_venous_stiffness_exp '+str(json_data["circadapt"]["kVe"])+' \\\n')

	f.write('  -cvs_vp_wall_area '+str(vp_wall_area)+' \\\n')
	f.write('  -cvs_vp_length '+str(json_data["circadapt"]["PVel"])+' \\\n')

	if "PVep0" in json_data["circadapt"]:
		raise Exception("You are trying to set -cvs_vp_p0ref_factor. This is not supported in newer CARP versions")

	if "PVePref" in json_data["circadapt"]:
		f.write('  -cvs_vp_ref_pressure '+str(json_data["circadapt"]["PVePref"])+' \\\n')

	# f.write('  -cvs_vp_p0ref_factor '+str(json_data["circadapt"]["PVep0"])+' \\\n')
	f.write('  -cvs_pulmven_stiffness_exp '+str(json_data["circadapt"]["kPVe"])+' \\\n')

	# chamber wall volumes
	f.write('  -cvs_ra_wall_vol '+str(json_clinical_data["chambers"]["RA_wallVolume"])+' \\\n')
	f.write('  -cvs_la_wall_vol '+str(json_clinical_data["chambers"]["LA_wallVolume"])+' \\\n')
	f.write('  -cvs_rv_wall_vol '+str(json_clinical_data["chambers"]["RV_wallVolume"])+' \\\n')
	f.write('  -cvs_sv_wall_vol '+str(json_clinical_data["chambers"]["Sept_wallVolume"])+' \\\n')
	f.write('  -cvs_lv_wall_vol '+str(json_clinical_data["chambers"]["LV_wallVolume"])+' \\\n')

	# chamber wall areas
	# f.write('  -cvs_ra_surf_area '+str(json_clinical_data["chambers"]["RA_wallArea"])+' \\\n')
	# f.write('  -cvs_la_surf_area '+str(json_clinical_data["chambers"]["LA_wallArea"])+' \\\n')
	# f.write('  -cvs_rv_surf_area '+str(json_clinical_data["chambers"]["RV_wallArea"])+' \\\n')
	# f.write('  -cvs_sv_surf_area '+str(json_clinical_data["chambers"]["Sept_wallArea"])+' \\\n')
	# f.write('  -cvs_lv_surf_area '+str(json_clinical_data["chambers"]["LV_wallArea"])+' \\\n')

	# chamber volumes - probably not needed
	# f.write('  -cvs_ra_vol '+str(json_clinical_data["chambers"]["RA_volume"])+' \\\n')
	# f.write('  -cvs_la_vol '+str(json_clinical_data["chambers"]["LA_volume"])+' \\\n')
	# f.write('  -cvs_rv_vol '+str(json_clinical_data["chambers"]["RV_volume"])+' \\\n')
	# f.write('  -cvs_lv_vol '+str(json_clinical_data["chambers"]["LV_volume"])+' \\\n')

	# timings 
	f.write('  -cvs_cycle_length '+str(json_clinical_data["general"]["BCL"]/1000.)+' \\\n')
	# f.write('  -cvs_t_ra_clock 0.0 \\\n')
	# f.write('  -cvs_t_aa_delay 0.0 \\\n')
	# f.write('  -cvs_t_av_delay '+str(AV_delay/1000.)+' \\\n')
	# f.write('  -cvs_t_vv_delay 0.0 \\\n')

	# circ adapt settings
	f.write('  -cvs_tube_model 4 \\\n')         # circadapt tube
	f.write('  -cvs_valve_model 4 \\\n')        # circadapt valve
	f.write('  -cvs_mass_conservation 1 \\\n')
	f.write('  -cvs_CircAdapt_version 1 \\\n')  # 2015
	f.write('  -cvs_ode_solver 0 \\\n') 	    # forward-euler
	f.write('  -cvs_irregular_heart_rate 0 \\\n') 	  
	f.write('  -cvs_allow_negative_p 1 \\\n') 	  
	f.write('  -cvs_exercise 0 \\\n') 	  

def simulation_output_options(f,
							  setup,
							  json_clinical_data,
							  json_data,
							  postprocessing,
							  tend=0.0,
							  NBEATS=None,
							  mode='cycle',
							  start_state=None,
							  get_fibre_strains=False):
	f.write('  -timedt 10.0 \\\n')
	f.write('  -gridout_i 0 \\\n')
	f.write('  -gzip_data 0 \\\n')

	if mode=='cycle':

		if "AV_delay" in json_data["EP"]:
			AV_delay = json_data["EP"]["AV_delay"]
		else:
			if setup.AV_delay is not None:
				AV_delay = setup.AV_delay
			else:
				raise Exception("You need to setup the AV delay in either the setup file or in the parameter file")

		# f.write('  -chkpt_intv '+str(json_clinical_data["general"]["BCL"])+' \\\n')
		f.write('  -num_tsav 2 \\\n')
		f.write('  -tsav[0] 0.0 \\\n')
		f.write('  -tsav[1] '+str(int(NBEATS*json_clinical_data["general"]["BCL"]-AV_delay))+' \\\n')

		if start_state is not None:
			f.write('  -start_statef '+start_state+' \\\n')

	f.write('  -mech_output 1 \\\n')	
	if not postprocessing:
		if not get_fibre_strains:
			f.write('  -strain_value 0 \\\n')
		else:
			f.write('  -strain_value 1 \\\n')
		f.write('  -spacedt 10.0 \\\n')
	else:
		f.write('  -strain_value 1 \\\n')   # fibre strains on nodes
		f.write('  -post_processing_opts 64 \\\n')
		f.write('  -experiment 4 \\\n')
		f.write('  -spacedt 1.0 \\\n')
	f.write('  -stretch_value 0 \\\n')
	f.write('  -work_value 0 \\\n')
	f.write('  -spring_value 0 \\\n')
	f.write('  -pressure_value 0 \\\n')
	f.write('  -stress_value 0')

