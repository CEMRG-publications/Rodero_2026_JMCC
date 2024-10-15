
import argparse
import json
import numpy as np
import os

def json_to_init(stimuli, tag_file, json_param_file, init_file_name):

    # Read tags
    f_input = open(tag_file,"r")
    tags = json.load(f_input)
    f_input.close()

    # Read CVs
    f_input = open(json_param_file,"r")
    params = json.load(f_input)
    f_input.close()

    tags_ventricles_names = ["LV", "RV"]
    CV_ventricle_name = "CV_ventricles"
    if not CV_ventricle_name in params.keys():
        CV_ventricle_name = "CV_f_v"
    k_ventricles_name = "k_ventricles"
    if not k_ventricles_name in params.keys():
        k_ventricles_name = "ani_ratio_v"

    tags_FEC_names = ["FEC_LV", "FEC_RV", "FEC_SV"]
    k_FEC_name = "k_FEC"

    tags_atria_names = ["LA", "RA"]
    CV_atria_name = "CV_atria"
    if not CV_atria_name in params.keys():
        CV_atria_name = "CV_f_a"
    k_atria_name = "k_atria"
    if not k_atria_name in params.keys():
        k_atria_name = "ani_ratio_a"

    tags_bachmann_names = ["BB"]
    k_BB_name = "k_BB"

    vtx = []
    nVtx = 0

    for vtxFile in stimuli:
        temp = np.loadtxt(vtxFile, dtype=int, skiprows=2, ndmin=1)
        vtx.append(temp)
        nVtx += temp.shape[0]

    # write .init file
    f = open(init_file_name,'w')

    # header
    f.write('vf:0 vs:0 vn:0 vPS:0\n') # Default properties for tags not specified
    f.write('retro_delay:0 antero_delay:0\n') # If there's no 1D purkinje system, it's ignored.
    # number of stimuli and regions
    f.write('%d %d\n' % (int(nVtx), int(len(tags_ventricles_names)) + len(tags_FEC_names) + len(tags_atria_names) + len(tags_bachmann_names)))
    # stimulus
    for i in range(len(vtx)):
        if len(vtx[i]) == 1:
            f.write('%d %f\n' % (vtx[i],0))
        else:
            for n in vtx[i]:
                f.write('%d %f\n' % (int(n),0))
                
    return_tags_str = ''
    # ek regions
    for i,tag_name in enumerate(tags_ventricles_names):
        f.write('%d %f %f %f\n' % (int(tags[tag_name]), 
                                   float(params["EP"][CV_ventricle_name]), 
                                   float(params["EP"][CV_ventricle_name])*float(params["EP"][k_ventricles_name]), 
                                   float(params["EP"][CV_ventricle_name])*float(params["EP"][k_ventricles_name])))
        return_tags_str += ',' + str(int(tags[tag_name]))

    for i,tag_name in enumerate(tags_FEC_names):
        f.write('%d %f %f %f\n' % (int(tags[tag_name]), 
                                   float(params["EP"][CV_ventricle_name])*float(params["EP"][k_FEC_name]), 
                                   float(params["EP"][CV_ventricle_name])*float(params["EP"][k_FEC_name]), 
                                   float(params["EP"][CV_ventricle_name])*float(params["EP"][k_FEC_name])))
        return_tags_str += ',' + str(int(tags[tag_name]))

    for i,tag_name in enumerate(tags_atria_names):
        f.write('%d %f %f %f\n' % (int(tags[tag_name]), 
                                   float(params["EP"][CV_atria_name]), 
                                   float(params["EP"][CV_atria_name])*float(params["EP"][k_atria_name]), 
                                   float(params["EP"][CV_atria_name])*float(params["EP"][k_atria_name])))
        return_tags_str += ',' + str(int(tags[tag_name]))

    for i,tag_name in enumerate(tags_bachmann_names):
        f.write('%d %f %f %f\n' % (int(tags[tag_name]), 
                                   float(params["EP"][CV_atria_name])*float(params["EP"][k_BB_name]), 
                                   float(params["EP"][CV_atria_name])*float(params["EP"][k_BB_name]), 
                                   float(params["EP"][CV_atria_name])*float(params["EP"][k_BB_name])))
        return_tags_str += ',' + str(int(tags[tag_name]))

    f.close()
    
    return return_tags_str[1:]

def electrophysiology_output(basefolder,
							 elem_file,
							 tags,
	   						 start_sample=0,
	   						 last_sample=1,
	   						 output_file='Y.txt'):

	print('Reading mesh elem file...')
	elem = np.loadtxt(elem_file,dtype=int,usecols=[1,2,3,4,5],skiprows=1)
	print('Done.')

	V_EIDX = np.where(np.isin(elem[:,-1],tags["ventricles"]+tags["fast_endo"])==1)[0]
	A_EIDX = np.where(np.isin(elem[:,-1],tags["atria"]+tags["bachmann_bundle"])==1)[0]

	V_VTX = np.unique(elem[V_EIDX,0:4].flatten())
	A_VTX = np.unique(elem[A_EIDX,0:4].flatten())

	output = np.zeros((last_sample-start_sample+1,2))

	count = 0
	for i in range(start_sample,last_sample+1):
		print('Computing output for '+str(i)+'.dat...')
		AT=np.loadtxt(os.path.join(basefolder,str(i)+".dat"),dtype=float)
		if (np.min(AT[V_VTX]<0)):
			raise Exception("The ventricles contain a negative activation time.")
		if (np.min(AT[A_VTX]<0)):
			raise Exception("The atria contain a negative activation time.")
			
		output[count,0] = np.max(AT[A_VTX])-np.min(AT[A_VTX])
        
		output[count,1] = np.max(AT[V_VTX])-np.min(AT[V_VTX])
		count += 1

	np.savetxt(output_file,output,fmt="%g")
def main(args):

    heart_folder = args.heart_folder
    scenario = args.scenario
    Nsim = args.Nsim

    stimuli = [f'{heart_folder}/sims_folder/fascicles_lv.vtx',
                    f'{heart_folder}/sims_folder/fascicles_rv.vtx',
                    f'{heart_folder}/sims_folder/SAN.vtx']

    json_param_path        = f'{heart_folder}/scenarios/{scenario}/json_files/'
    tag_file        = f'{json_param_path}/tags_EP.json'
    init_file_path  = f'{heart_folder}/scenarios/{scenario}/data/init_files'

    os.system("mkdir -p " + init_file_path)

    for sim_num in range(Nsim):
        tags_activated = json_to_init(stimuli=stimuli,
                    tag_file=tag_file,
                    json_param_file=os.path.join(json_param_path,str(sim_num) + '.json'),
                    init_file_name=os.path.join(init_file_path,str(sim_num) + '.init')
                    )
    

    sims_folder = f'{heart_folder}/scenarios/{scenario}/simulations'

    meshname = f'{heart_folder}/sims_folder/myocardium_AV_FEC_BB_lvrv'


    cmd = ['ekbatch',meshname]
    init_cmd = ','.join([os.path.join(init_file_path,str(sim_num)) for sim_num in range(Nsim)])

    os.system(' '.join(cmd+[init_cmd] + [tags_activated]))

    os.makedirs(sims_folder,exist_ok=True)
    for sim_num in range(Nsim):
        os.system('mv ' + os.path.join(init_file_path,str(sim_num) + '.dat ') + sims_folder)

    basefolder = sims_folder
    elem_file = f"{meshname}.elem"

    f_input = open(tag_file,"r")
    tags = json.load(f_input)
    f_input.close()


    tags_modified = tags.copy()
    tags_modified["ventricles"] = [tags_modified["LV"], tags_modified["RV"]]
    tags_modified["fast_endo"] = [tags_modified["FEC_RV"], tags_modified["FEC_SV"]]
    tags_modified["atria"] = [tags_modified["LA"], tags_modified["RA"]]
    tags_modified["bachmann_bundle"] = [tags_modified["BB"]]

    output_path = f'{heart_folder}/scenarios/{scenario}/output'

    os.makedirs(output_path,exist_ok=True)

    electrophysiology_output(basefolder=basefolder,
                                elem_file=elem_file,
                                tags=tags_modified,
                                start_sample=0,
                                last_sample=Nsim-1,
                                output_file=os.path.join(output_path,'Y.txt'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=""
                                     )
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--heart_folder',     
                        type=str, 
                        required=True, 
                        help="")
    parser.add_argument('--scenario',     
                    type=str, 
                    required=True, 
                    help="")
    parser.add_argument('--Nsim',     
                    type=str, 
                    required=False, 
                    default=120)
    args = parser.parse_args()

    main(args)
    
