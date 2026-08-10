"""
Slurm job headers and environment setup for ARCHER2.

Vendored verbatim from UNLOADING_library.headers.hpc_headers by M. Strocchi, which is not publicly available,
so that this repository does not depend on any private code. Only the internal
imports have been repointed at this package; the source is otherwise unchanged,
including its tab indentation, so that the files it generates stay byte-identical
to the ones used for the published simulations.
"""

import os
import sys

def write_archer2_header(jobname,
						 out_filename,
						 walltime,
						 ncores):
	
	if (ncores % 128!=0):
		raise Exception("Archer2 nodes have 128 cores each. Please use a number of cores that is a multiple of ")

	nnodes = int(ncores/128)

	header = ["#!/bin/bash --login"]
	header += ["#"]
	header += ["#SBATCH --job-name="+jobname]
	header += ["#SBATCH --output="+out_filename]
	header += ["#SBATCH --time="+walltime]
	header += ["#SBATCH --nodes="+str(nnodes)]
	header += ["#SBATCH --tasks-per-node=128"]
	header += ["#SBATCH --cpus-per-task=1"]
	header += ["#SBATCH --mail-type=ALL"]
	header += ["#SBATCH --mail-user=None"]
	header += ["#SBATCH --account=e348"]
	header += ["#SBATCH --partition=standard"]
	header += ["#SBATCH --qos=standard"]
	header += [""]
	header += [""]

	header = "\n".join(header)

	return header

def write_env_variables(path2unloading,
						path2carputils,
						carp_config_file,
						ncores,
						env_folder,
						archer2_config_file=None):

	env_variabiles = ["export PYTHONPATH=$PYTHONPATH:"+path2unloading]
	env_variabiles += ["export PYTHONPATH=$PYTHONPATH:"+path2carputils]
	env_variabiles += [""]

	env_variabiles += ["export CARPPLATFORM=CARPENTRY"]
	env_variabiles += ["export CARPFLV=CRAY"]
	env_variabiles += ["export SHARED=/work/e348/shared/"]
	env_variabiles += [""]

	env_variabiles += ["NPROC="+str(ncores)]
	env_variabiles += [""]

	env_variabiles += ["export OMP_NUM_THREADS=1"]
	env_variabiles += [""]

	env_variabiles += ["source "+env_folder+"bin/activate"]

	if archer2_config_file is not None:
		env_variabiles += ["source "+archer2_config_file]
	env_variabiles += ["source "+carp_config_file]

	env_variabiles += [""]
	env_variabiles += [""]

	env_variabiles = "\n".join(env_variabiles)

	return env_variabiles