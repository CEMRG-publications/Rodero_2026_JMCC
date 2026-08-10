"""
Generation of the single-cell (bench) simulation scripts.

Vendored verbatim from SIMULATION_library.cell_sims_utils by M. Strocchi, which is not publicly available,
so that this repository does not depend on any private code. Only the internal
imports have been repointed at this package; the source is otherwise unchanged,
including its tab indentation, so that the files it generates stay byte-identical
to the ones used for the published simulations.
"""

import os
import sys

def generate_bench_script(N,
						  BCL,
						  NBEATS,
						  basefolder,
						  NPROC,
						  strain,
						  chamber,
						  contraction_model,
						  suffix):

	if basefolder[-1]=="/":
		basefolder = basefolder[:-1]
	
	if chamber=="LV":
		ionic_model = "ToRORd_dynCl"
	elif chamber=="RV":
		ionic_model = "ToRORd_dynCl"
	elif chamber=="atria":
		ionic_model = "JB_COURTEMANCHE"
	else:
		raise Exception("Unsupported chamber - pick LV, RV or atria")

	f = open(basefolder+"/run_"+ionic_model+suffix+".sh","w")

	f.write("#!/bin/bash\n")

	os.system("mkdir "+basefolder+"/"+ionic_model+suffix+"/")

	f.write("\n")
	
	DURATION  = NBEATS*BCL
	N_end1    = (N//NPROC)*NPROC-1
	N_start2  = (N//NPROC)*NPROC	
	N_end2    = N-1

	cmd=["cmd="+'"'+"bench.pt"]
	cmd += ["--numstim",str(NBEATS)+"\n"]
	cmd += ["--bcl",str(BCL)+"\n"]
	cmd += ["--past-stim",str(BCL)+"\n"]
	cmd += ["--imp",ionic_model+"\n"]
	cmd += ["--dt","0.02"+"\n"]
	cmd += ["--stim-curr","60.0"+"\n"]
	cmd += ["--dt-out","1.0"+"\n"]

	if contraction_model=="Land":
		cmd += ["--plug-in","LandHumanStress"+"\n"]
		cmd += ["--save-ini-file","${output}/${PBS_ARRAY_INDEX}_"+ionic_model+suffix+"_LandHumanStress.sv"+"\n"]
	elif contraction_model=="tanh":
		cmd += ["--plug-in","TanhStress"+"\n"]
		cmd += ["--save-ini-file","${output}/${PBS_ARRAY_INDEX}_"+ionic_model+suffix+"_TanhStress"+suffix+".sv"+"\n"]

	cmd += ["--strain",str(strain)+"\n"]
	cmd += ["--strain-rate","0.0"+"\n"]
	cmd += ["--strain-time","0.0"+"\n"]
	cmd += ["--strain-dur",str(DURATION)+"\n"]
	cmd += ["--imp-par","${pp_ionic}"+"\n"]
	cmd += ["--plug-par","${pp_land}"+"\n"]
	cmd += ["--save-ini-time ",str(DURATION)+"\n"]

	cmd += ["--fout=${output}/"+ionic_model+"\n"]
	cmd += ["--plug-sv-dump=Tension"+"\n"]
	cmd += ["--imp-sv-dump=Ca_i"+"\n"]

	cmd_str = " ".join(cmd)
	cmd_final = cmd_str+" --bin\n --no-trace\n >/dev/null 2>&1 &"+'"\n'

	# --------------------------------------------------
	# first loop 
	if N_end1>=0:
		f.write("NPROC="+str(NPROC)+"\n")
		f.write("i=0\n")

		f.write("\n")
		f.write("\n")

		f.write("for PBS_ARRAY_INDEX in $(seq 0 "+str(N_end1)+" )\n")	

		f.write("do\n")
		f.write("\n")	

		f.write("pp_ionic=$(cat "+basefolder+"/param/${PBS_ARRAY_INDEX}_param_"+ionic_model+".txt)\n")
		f.write("pp_land=$(cat "+basefolder+"/param/${PBS_ARRAY_INDEX}_param_"+ionic_model+"_"+contraction_model+suffix+".txt)\n")	

		f.write("\n")	

		f.write("output="+basefolder+"/"+ionic_model+suffix+"/$PBS_ARRAY_INDEX\n")
		f.write("mkdir ${output}\n")	

		f.write("\n")
		f.write(cmd_final)
		f.write("\n")
		f.write("eval $cmd\n")
		f.write("\n")

		f.write("((i++))\n")
		f.write("[[ $((i%NPROC)) -eq 0 ]] && wait\n")
		f.write("done\n")

		f.write("\n")
		f.write("\n")
		f.write("\n")
		f.write("\n")

	f.write("NPROC="+str(N-N_start2)+"\n")
	f.write("i=0\n")
	f.write("\n")
	f.write("\n")

	f.write("for PBS_ARRAY_INDEX in $(seq "+str(N_start2)+" "+str(N_end2)+" )\n")	

	f.write("do\n")
	f.write("\n")	

	f.write("pp_ionic=$(cat "+basefolder+"/param/${PBS_ARRAY_INDEX}_param_"+ionic_model+".txt)\n")
	f.write("pp_land=$(cat "+basefolder+"/param/${PBS_ARRAY_INDEX}_param_"+ionic_model+"_"+contraction_model+suffix+".txt)\n")	

	f.write("\n")	

	f.write("output="+basefolder+"/"+ionic_model+suffix+"/$PBS_ARRAY_INDEX\n")
	f.write("mkdir ${output}\n")	

	f.write("\n")
	f.write(cmd_final)
	f.write("eval $cmd\n")
	f.write("\n")
	f.write("\n")
	f.write("((i++))\n")
	f.write("[[ $((i%NPROC)) -eq 0 ]] && wait\n")
	f.write("done\n")	

def move_sv_file(run_folder,
				 N,
				 destination_folder):

	os.system("mkdir -p "+destination_folder)

	for i in range(N):
		folder = run_folder+"/"+str(i)+"/"

		os.system("cp "+folder+"*.sv "+destination_folder)





