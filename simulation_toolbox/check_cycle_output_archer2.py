import argparse
import os
import numpy as np
from pandas import read_csv
import json
import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor


from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

from common import fourchamber_output
from common.mesh_io import read_tets

def plot_statistics_file(basefolder):
    # Load data from Y.txt
    data = np.loadtxt(f'{basefolder}/data/Y.txt')

    # Load labels from ylabels.txt
    with open(f'{basefolder}/data/ylabels.txt', 'r') as file:
        labels = file.read().splitlines()

    # Calculate mean and standard deviation for each variable
    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    # Save results to a file
    with open(f'{basefolder}/output/output_statistics.csv', 'w') as file:
        file.write("Variable\tMean\tStd\tMin\tMax\n")
        for label, mean, std, min, max in zip(labels, mean_values, std_values, min_values, max_values):
            file.write(f"{label}\t{mean:.2f}\t{std:.2f}\t{min:.2f}\t{max:.2f}\n")

def artery_cycle_output_free_output_mask_name(pressure):
    
    diast_pressure = pressure.min()
    syst_pressure = pressure.max()
    pulse_pressure = syst_pressure - diast_pressure
    mean_pressure = (2*diast_pressure + syst_pressure)/3.0

    return([diast_pressure, syst_pressure, pulse_pressure, mean_pressure])


def VV_output_free_IV_thr(time,volume,pressure,IV_thr):

	time_pmax = time[0]+np.where(pressure==np.max(pressure))[0][0]
	
	dv = np.gradient(volume)

	ind_IVC_ = np.intersect1d(np.where(np.abs(dv)<=IV_thr)[0],np.where(time<=time_pmax)[0])
	
	jump = np.where(np.gradient(ind_IVC_)>1)[0]	

	if len(jump) == 0:
		ind_IVC = ind_IVC_
	else:
		ind_IVC = ind_IVC_[jump[-1]:-1]	

	ind_IVR_ = np.intersect1d(np.where(np.abs(dv)<=IV_thr)[0],np.where(time>=time_pmax)[0])
	jump = np.where(np.gradient(ind_IVR_)>1)[0]	

	if len(jump) == 0:
		ind_IVR = ind_IVR_
	else:
		ind_IVR = ind_IVR_[0:jump[0]]	

	ind_ED = ind_IVC[0]	

	EDV = volume[ind_ED]
	EDP = pressure[ind_ED]	

	# max P
	pMax = np.max(pressure)
	pMax_idx = np.where(pressure==pMax)[0][0]	

	ESV = np.min(volume)	

	dpdt_idx = np.where(pressure>EDP)[0]
	ind_IVR = np.intersect1d(dpdt_idx,ind_IVR)
	if not len(ind_IVR):
		ind_IVR = np.intersect1d(dpdt_idx,np.where(time>=time_pmax)[0])	

	ind_IVC = np.intersect1d(dpdt_idx,ind_IVC)

	if ind_IVC.size:	
		dpdt_ = np.diff(pressure)/np.diff(time)*1000.0
		dpdt = np.zeros((pressure.shape[0],),dtype=float)
		dpdt[0] = dpdt_[0]
		dpdt[1:] = dpdt_
		# dpdt = np.gradient(pressure)*1000.0	

		# to detect oscillations: during IVC the derivative should always be positive
		# if it's not it means that there are oscillations (normally due to the valves)
		# so if we find anywehere the dpdt is negative, we discard whatever happens after
		# because that derivative will be wrong
		wrong_IVC = np.where(dpdt[ind_IVC]<=-50.0)[0]
		if len(wrong_IVC)>0:
			print('Found oscillations during IVC... Removing indices after oscillation...')
			ind_IVC = ind_IVC[:wrong_IVC[0]]
		dpdtMax = np.max(dpdt[:pMax_idx])	

		# to detect oscillations: during IVR the derivative should always be negative
		# if it's not it means that there are oscillations (normally due to the valves)
		# so if we find anywehere the dpdt is positive, we discard whatever happens after
		# because that derivative will be wrong
		wrong_IVR = np.where(dpdt[ind_IVR]>=50.0)[0]
		if len(wrong_IVR)>0:
			print('Found oscillations during IVR... Removing indices after oscillation...')
			ind_IVR = ind_IVR[:wrong_IVR[0]]
		if ind_IVR.size:
			dpdtMin = np.min(dpdt[pMax_idx:])		
			output = [EDV,EDP,ESV,pMax,dpdtMax,dpdtMin]
		else:
			output = [0,0,0,0,0,0]

	else:
		output = [0,0,0,0,0,0]

	return output


def cycle_output_free_output_mask_name(datafolder,
                 output_folder,
                 BCL,
                 NBEATS,
                 AVD,
                 IV_thr,
                 first_simulation = 0,
                 basename="cycle_",
                 output_file="Y.txt",
                 visualise=True,
                 output_mask="output_mask.txt"
                 ):

    print('WARNING: remember that the AVD is needed to compute the last cycle, as the simulation starts @ -AVD ms')

    print('Computing only output for successful simulations...')

    mask = np.loadtxt(f"{datafolder}/{output_mask}",dtype=int)
    idx_ok = np.where(mask==1)[0]

    output = []

    t = tqdm.trange(len(range(idx_ok.shape[0])), desc='Bar desc', leave=True,colour='#B3EBF2')
    for i in t:
        if first_simulation is None: # Default simulation
            first_simulation = 0
            sim_number = 'default'
        else:
            sim_number = first_simulation + idx_ok[i]
        t.set_description('Simulation '+basename+str(sim_number)+'...')
        folder = output_folder+'/'+basename+str(sim_number)

        lv = read_csv(folder+'/cav.LV.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')
        rv = read_csv(folder+'/cav.RV.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')
        la = read_csv(folder+'/cav.LA.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')
        ra = read_csv(folder+'/cav.RA.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')

        ao = read_csv(f"{folder}/tube.AO.csv", delimiter=",", skipinitialspace=True,
                           header=0, comment='#')
        
        pa = read_csv(f"{folder}/tube.AP.csv", delimiter=",", skipinitialspace=True,
                           header=0, comment='#')

        time = np.array(lv['Time'])
        
        start = int((NBEATS-1)*BCL-AVD[idx_ok[i]])
        # start = time[-1]-BCL
        end = start+BCL

        last_beat = np.intersect1d(np.where(np.array(lv['Time'])>=start)[0],
                                   np.where(np.array(lv['Time'])<=end)[0])
        time = np.array(lv['Time'][last_beat])

        volume_lv = np.array(lv['Volume'][last_beat])
        pressure_lv = np.array(lv['Pressure'][last_beat])

        volume_rv = np.array(rv['Volume'][last_beat])
        pressure_rv = np.array(rv['Pressure'][last_beat])

        volume_la = np.array(la['Volume'][last_beat])
        pressure_la = np.array(la['Pressure'][last_beat])

        volume_ra = np.array(ra['Volume'][last_beat])
        pressure_ra = np.array(ra['Pressure'][last_beat])

        pressure_ao = np.array(ao['Pressure'][last_beat])
        pressure_pa = np.array(pa['Pressure'][last_beat])

        labels_computed = []

        lvoutput = VV_output_free_IV_thr(time=time, volume=volume_lv,pressure=pressure_lv, IV_thr=IV_thr)
        LVSV = lvoutput[0] - lvoutput[2]
        LVEF = 100*(LVSV / lvoutput[0])
        labels_computed.extend(["LVedv","LVedp","LVesv","LVpMax","LVdpdtMax","LVdpdtMin","LVSV","LVEF"])
        lvoutput.extend([LVSV,LVEF])

        rvoutput = VV_output_free_IV_thr(time=time,volume=volume_rv,pressure=pressure_rv, IV_thr=IV_thr)
        labels_computed.extend(["RVedv","RVedp","RVesv","RVpMax","RVdpdtMax","RVdpdtMin","RVSV","RVEF"])
        RVSV = rvoutput[0] - rvoutput[2]
        RVEF = 100*(RVSV / rvoutput[0])
        rvoutput.extend([RVSV,RVEF])

        laoutput = fourchamber_output.AA_output(time,volume_la,pressure_la)
        labels_computed.extend(["LAedv","LAesv","LAvMax","LApMax"])

        raoutput = fourchamber_output.AA_output(time,volume_ra,pressure_ra)
        labels_computed.extend(["RAedv","RAesv","RAvMax","RApMax"])

        laoutput_ej = fourchamber_output.AA_output_ej(time,volume_la)
        labels_computed.extend(["LAsvA","LAinflV","LAsvV"])
        raoutput_ej = fourchamber_output.AA_output_ej(time,volume_ra)
        labels_computed.extend(["RAsvA","RAinflV","RAsvV"])


        aooutput = artery_cycle_output_free_output_mask_name(pressure_ao)
        labels_computed.extend(["diastAP","systAP","pulseAP","mAP"])

        paoutput = artery_cycle_output_free_output_mask_name(pressure_pa)
        labels_computed.extend(["diastPAP","systPAP","pulsePAP","mPAP"])

        if visualise:
            fourchamber_output.check_ventricle_output(time,volume_lv,pressure_lv,lvoutput)
            fourchamber_output.check_ventricle_output(time,volume_rv,pressure_rv,rvoutput)
            fourchamber_output.check_atria_output(time,volume_la,pressure_la,laoutput)
            fourchamber_output.check_atria_output(time,volume_ra,pressure_ra,raoutput)


        concatenated = np.concatenate((lvoutput, rvoutput, laoutput, raoutput, laoutput_ej, raoutput_ej, aooutput, paoutput), axis=0)
        output.append(concatenated)
        
    output = np.array(output, dtype=object)

    np.savetxt(output_file, output, fmt='%.2f')

    return(labels_computed)

def timings_output(datafolder,
                 output_folder,
                 BCL,
                 NBEATS,
                 AVD,
                 IV_thr,
                 first_simulation = 0,
                 basename="cycle_",
                 output_file="Y_timings.txt",
                 output_mask="output_mask.txt"
                 ):

    print('Computing only output for successful simulations...')

    mask = np.loadtxt(f"{datafolder}/{output_mask}",dtype=int)
    idx_ok = np.where(mask==1)[0]

    output = []

    t = tqdm.trange(len(range(idx_ok.shape[0])), desc='Bar desc', leave=True,colour='#779ECB')
    for i in t:
        if first_simulation is None: # Default simulation
            first_simulation = 0
            sim_number = 'default'
        else:
            sim_number = first_simulation + idx_ok[i]
        t.set_description('Simulation '+basename+str(sim_number)+'...')
        folder = output_folder+'/'+basename+str(sim_number)

        lv = read_csv(folder+'/cav.LV.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')
        rv = read_csv(folder+'/cav.RV.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')
        
        time = np.array(lv['Time'])
        
        start = int((NBEATS-1)*BCL-AVD[idx_ok[i]])
        # start = time[-1]-BCL
        end = start+BCL

        last_beat = np.intersect1d(np.where(np.array(lv['Time'])>=start)[0],
                                   np.where(np.array(lv['Time'])<=end)[0])
        time = np.array(lv['Time'][last_beat])

        volume_lv = np.array(lv['Volume'][last_beat])
        pressure_lv = np.array(lv['Pressure'][last_beat])

        time_pmax_lv = time[0]+np.where(pressure_lv==np.max(pressure_lv))[0][0]
	
        dv_lv = np.gradient(volume_lv)


        ind_IVC_lv_ = np.intersect1d(np.where(np.abs(dv_lv)<=IV_thr)[0],np.where(time<=time_pmax_lv)[0])

        jump_ivc_lv = np.where(np.gradient(ind_IVC_lv_)>1)[0]	

        if len(jump_ivc_lv) == 0:
            ind_IVC_lv = ind_IVC_lv_
        else:
            ind_IVC_lv = ind_IVC_lv_[jump_ivc_lv[-1]:-1]	

        ind_IVR_lv_ = np.intersect1d(np.where(np.abs(dv_lv)<=IV_thr)[0],np.where(time>time_pmax_lv)[0])

        jump_ivr_lv = np.where(np.gradient(ind_IVR_lv_)>1)[0]	

        if len(jump_ivr_lv) == 0:
            ind_IVR_lv = ind_IVR_lv_
        else:
            ind_IVR_lv = ind_IVR_lv_[0:jump_ivr_lv[0]]

        labels_computed = []
        timings_output_lv = []

        labels_computed.extend(["LVivc","LVeje","LVivr","LVfil"])

        timings_output_lv.extend([time[ind_IVC_lv[0]],
                               time[ind_IVC_lv[-1]],
                               time[ind_IVR_lv[0]],
                               time[ind_IVR_lv[-1]]])
        
        volume_rv = np.array(rv['Volume'][last_beat])
        pressure_rv = np.array(rv['Pressure'][last_beat])

        time_pmax_rv = time[0]+np.where(pressure_rv==np.max(pressure_rv))[0][0]
	
        dv_rv = np.gradient(volume_rv)

        ind_IVC_rv_ = np.intersect1d(np.where(np.abs(dv_rv)<=IV_thr)[0],np.where(time<=time_pmax_rv)[0])

        jump_ivc_rv = np.where(np.gradient(ind_IVC_rv_)>1)[0]	

        if len(jump_ivc_rv) == 0:
            ind_IVC_rv = ind_IVC_rv_
        else:
            ind_IVC_rv = ind_IVC_rv_[jump_ivc_rv[-1]:-1]	

        ind_IVR_rv_ = np.intersect1d(np.where(np.abs(dv_rv)<=IV_thr)[0],np.where(time>time_pmax_rv)[0])

        jump_ivr_rv = np.where(np.gradient(ind_IVR_rv_)>1)[0]	

        if len(jump_ivr_rv) == 0:
            ind_IVR_rv = ind_IVR_rv_
        else:
            ind_IVR_rv = ind_IVR_rv_[0:jump_ivr_rv[0]]


        timings_output_rv = []

        labels_computed.extend(["RVivc","RVeje","RVivr","RVfil"])
        timings_output_rv.extend([time[ind_IVC_rv[0]],
                               time[ind_IVC_rv[-1]],
                               time[ind_IVR_rv[0]],
                               time[ind_IVR_rv[-1]]])



        concatenated = np.concatenate((timings_output_lv,timings_output_rv), axis=0)

        output.append(concatenated)
        
    output = np.array(output, dtype=object)

    np.savetxt(output_file, output, fmt='%.2f')

    return(labels_computed)


# Top-level function for multiprocessing
def process_simulation(i, idx_ok, output_folder, basename, first_simulation, A_VTX, V_VTX):
    sim_num = 'default' if first_simulation is None else first_simulation + idx_ok[i]
    folder = os.path.join(output_folder, f"{basename}{sim_num}")
    filepath = os.path.join(folder, "vm_act_seq.dat")

    try:
        AT = np.loadtxt(filepath, dtype=float)
    except Exception as e:
        return (i, None, None, f"Read error: {e}")

    AT_A = AT[A_VTX]
    AT_V = AT[V_VTX]

    if np.any(AT_V < 0):
        return (i, None, None, "Negative ventricular AT")
    if np.any(AT_A < 0):
        return (i, None, None, "Negative atrial AT")

    A_TAT = AT_A.max() - AT_A.min()
    V_TAT = AT_V.max() - AT_V.min()
    return (i, A_TAT, V_TAT, None)


def electrophysiology_cycle_output_output_mask_free(datafolder,
                                                    output_folder,
                                                    elem_file,
                                                    tags,
                                                    first_simulation=0,
                                                    basename="cycle_",
                                                    output_file="Y_EP.txt",
                                                    output_mask="output_mask.txt"):
    print('Computing only output for successful simulations...')

    mask = np.loadtxt(os.path.join(datafolder, output_mask), dtype=int)
    idx_ok = np.where(mask == 1)[0]

    print('Reading mesh elem file...')
    elem = read_tets(elem_file)
    print('Done.')

    if "ventricles" in tags:
        if any(t in tags for t in ["lv", "rv"]):
            raise Exception('Cannot combine "ventricles" with "lv"/"rv" tags.')
        ventricle_tags = tags["ventricles"]
    elif all(t in tags for t in ["lv", "rv"]):
        ventricle_tags = tags["lv"] + tags["rv"]
    else:
        raise Exception("Ventricle tags not specified correctly.")

    if "fast_endo" in tags:
        if any(t in tags for t in ["fast_endo_lv", "fast_endo_rv", "fast_endo_sv"]):
            raise Exception('Cannot combine "fast_endo" with its sub-tags.')
        fec_tags = tags["fast_endo"]
    elif all(t in tags for t in ["fast_endo_lv", "fast_endo_rv", "fast_endo_sv"]):
        fec_tags = tags["fast_endo_lv"] + tags["fast_endo_rv"] + tags["fast_endo_sv"]
    else:
        raise Exception("Fast endocardial conduction (FEC) tags not set correctly.")

    ventricle_tags += fec_tags
    atria_tags = tags["atria"] + tags["bachmann_bundle"]

    V_EIDX = np.where(np.isin(elem[:, -1], ventricle_tags))[0]
    A_EIDX = np.where(np.isin(elem[:, -1], atria_tags))[0]

    V_VTX = np.unique(elem[V_EIDX, :4].flatten())
    A_VTX = np.unique(elem[A_EIDX, :4].flatten())

    output = np.zeros((idx_ok.shape[0], 2))

    print(f"Processing {idx_ok.shape[0]} simulations in parallel...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_simulation, i, idx_ok, output_folder, basename, first_simulation, A_VTX, V_VTX)
            for i in range(idx_ok.shape[0])
        ]

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Simulations", colour='#FDFD96'):
            i, a_tat, v_tat, error = future.result()
            if error:
                print(f"[Warning] Sim {idx_ok[i]} skipped due to error: {error}")
                continue
            output[i, 0] = a_tat
            output[i, 1] = v_tat

    np.savetxt(output_file, output, fmt="%g")
    print(f"Output saved to {output_file}")

    return ["A_TAT", "V_TAT"]


def electrophysiology_cycle_output_output_mask_free_no_parallel(datafolder,
                                   output_folder,
                                   elem_file,
                                   tags,
                                   first_simulation=0,
                                   basename="cycle_",
                                   output_file="Y_EP.txt",
                                   output_mask="output_mask.txt"):

    print('Computing only output for successful simulations...')

    mask = np.loadtxt(f"{datafolder}/{output_mask}",dtype=int)
    idx_ok = np.where(mask==1)[0]

    print('Reading mesh elem file...')
    elem = read_tets(elem_file)
    print('Done.')

    if "ventricles" in tags:
        check =  any(t in tags for t in ["lv","rv"])
        if check:
            raise Exception('If you want to set up the "ventricles" label, you should not set the "lv" and the "rv".')
        else:
            ventricle_tags = tags["ventricles"]

    elif (all(t in tags for t in ["lv","rv"])):
        ventricle_tags = tags["lv"]+tags["rv"]

    else:
        raise Exception("You haven't set the tags for neither the ventricles nor the lv and the rv.")

    if "fast_endo" in tags:
        check =  any(t in tags for t in ["fast_endo_lv","fast_endo_rv","fast_endo_sv"])
        if check:
            raise Exception('If you want to set up the "fast_endo" label, you should not set the "fast_endo_lv", "fast_endo_rv" and the "fast_endo_sv".')
        else:
            fec_tags = tags["fast_endo"]

    elif all(t in tags for t in ["fast_endo_lv","fast_endo_rv","fast_endo_sv"]):
        fec_tags = tags["fast_endo_lv"]+tags["fast_endo_rv"]+tags["fast_endo_sv"]

    else:
        raise Exception("You haven't set the tags for neither the fec nor the lv,rv and sv fec.")

    ventricle_tags += fec_tags
    # print('Ventricles tags: ')
    # for t in ventricle_tags:
    #     print(str(t))

    atria_tags = tags["atria"]+tags["bachmann_bundle"]
    # print('Atria tags: ')
    # for t in atria_tags:
    #     print(str(t))

    V_EIDX = np.where(np.isin(elem[:,-1],ventricle_tags)==1)[0]
    A_EIDX = np.where(np.isin(elem[:,-1],atria_tags)==1)[0]

    V_VTX = np.unique(elem[V_EIDX,0:4].flatten())
    A_VTX = np.unique(elem[A_EIDX,0:4].flatten())

    output = np.zeros((idx_ok.shape[0],2))
    t = tqdm.trange(len(range(idx_ok.shape[0])), desc='Bar desc', leave=True,colour='#FDFD96')
    for i in t:
        # Determine simulation number
        sim_num = 'default' if first_simulation is None else first_simulation + idx_ok[i]

        # Update progress bar description
        t.set_description(f'Simulation {basename}{sim_num}...')

        # Load activation times
        folder = f"{output_folder}/{basename}{sim_num}"
        AT = np.loadtxt(f"{folder}/vm_act_seq.dat", dtype=float)

        # Extract ventricular and atrial activation times
        AT_A = AT[A_VTX]
        AT_V = AT[V_VTX]

        # Check for negative activation times
        if np.any(AT_V < 0):
            raise Exception("The ventricles contain a negative activation time.")
        if np.any(AT_A < 0):
            raise Exception("The atria contain a negative activation time.")

        # Calculate and store activation ranges
        output[i, 0] = AT_A.max() - AT_A.min()
        output[i, 1] = AT_V.max() - AT_V.min()

    np.savetxt(output_file,output,fmt="%g")

    return(["A_TAT","V_TAT"])


def     cycle_simulation_summary(output_folder,
                             BCL,
                             AVD,
                             NBEATS,
                             start_sample,
                             last_sample,
                             IV_thr,
                             basename="cycle",
                             maskoutput="output_mask.txt",
                             output_file="simulation_summary.pdf",
                             unloaded_volumes=None,
                             include_last_AVD=False,
                             sims_folder=None
                             ):
    
    if start_sample is not None:
        output = np.zeros((last_sample-start_sample+1,),dtype=int)
    else: # Default simulation
        output = np.zeros((1,),dtype=int)

    if unloaded_volumes is not None:
        unloaded = np.loadtxt(unloaded_volumes,dtype=float)
        unloaded_failed = np.where(unloaded[:,0]==-1)[0]
    else:
        unloaded_failed = []

    document = SimpleDocTemplate(output_file, pagesize=A4, title='Simulation Summary')
    tab = []
    items = []
    header = ['sim','success','error']
    tab.append(header)

    count_PD = 0
    count_OK = 0
    count_notOK = 0
    count_NA = 0
    count_warning = 0

    if start_sample is not None:
        t = tqdm.trange(len(range(0,last_sample-start_sample+1)), desc='Bar desc', leave=True,colour='#C3B1E1')
        for sim_index in t:
            sim_number = start_sample + sim_index
            folder = output_folder+'/'+basename+str(sim_number)

            if os.path.exists(folder) and os.path.isfile(f"{folder}/cav.LV.csv"):
                t.set_description(f"Reading {folder}/cav.LV.csv...")
                
                lv = read_csv(folder+'/cav.LV.csv', delimiter=",", skipinitialspace=True,
                                header=0, comment='#')
                volume = np.array(lv['Volume'])
                time = np.array(lv['Time'])

                if include_last_AVD:
                    check_tend = BCL*NBEATS
                else:
                    check_tend = BCL*NBEATS - AVD[sim_index]

                init_last_beat = BCL*(NBEATS-1) - AVD[sim_index]
                end_last_beat = BCL*NBEATS - AVD[sim_index]                

                last_beat = np.intersect1d(np.where(time>=init_last_beat)[0],
                                        np.where(time<=end_last_beat)[0])


                if last_beat.shape[0]>0:
                    volume_last_beat = volume[last_beat]
                    SV = np.max(volume_last_beat)-np.min(volume_last_beat)
                    pressure_lv = np.array(lv['Pressure'][last_beat])
                    
                    # Checking that there's an IVR phase:
                    # dv = np.gradient(volume_last_beat)
                    # time_pmax = time[0]+np.where(pressure_lv==np.max(pressure_lv))[0][0]
                    # ind_IVC = np.intersect1d(np.where(np.abs(dv)<=0.01)[0],np.where(time<=time_pmax-10.)[0])
                    # ind_IVR = np.intersect1d(np.where(np.abs(dv)<=0.01)[0],np.where(time>=time_pmax+10.)[0])
                    
                    # ind_ED = ind_IVC[0]
                    # EDP = pressure_lv[ind_ED]
                    # dpdt_idx = np.where(pressure_lv>EDP)[0]
                    # ind_IVR = np.intersect1d(dpdt_idx,ind_IVR)

                    # Checking RV IVR phase:
                    rv = read_csv(folder+'/cav.RV.csv', delimiter=",", skipinitialspace=True,
                                header=0, comment='#')
                    volume_rv = np.array(rv['Volume'][last_beat])
                    pressure_rv = np.array(rv['Pressure'][last_beat])

                    # dv_rv = np.gradient(volume_rv[last_beat])
                    # time_pmax_rv = time[0]+np.where(pressure_rv==np.max(pressure_rv))[0][0]
                    # ind_IVC_rv = np.intersect1d(np.where(np.abs(dv_rv)<=0.01)[0],np.where(time<=time_pmax_rv-10.)[0])
                    # ind_IVR_rv = np.intersect1d(np.where(np.abs(dv_rv)<=0.01)[0],np.where(time>=time_pmax_rv+10.)[0])
                    # ind_ED_rv = ind_IVC_rv[0]
                    # EDP_rv = pressure_rv[ind_ED_rv]
                    # dpdt_idx_rv = np.where(pressure_rv>EDP_rv)[0]
                    # ind_IVR_rv = np.intersect1d(dpdt_idx_rv,ind_IVR_rv)

                    

                if len(lv)>0 and max(time) == int(check_tend) and len(last_beat) > 0:

                    LV_suitable = check_suitable_VV_output(time = time[last_beat],
                                                           volume = volume_last_beat,pressure = pressure_lv,
                                                           t=t,
                                                           IV_thr=IV_thr)
                    RV_suitable = check_suitable_VV_output(time=time[last_beat],
                                                           volume=volume_rv,
                                                           pressure=pressure_rv,
                                                           t=t,
                                                           IV_thr=IV_thr)

                    if LV_suitable and RV_suitable and (SV>5.0):


                        output[sim_index] = 1
                        tab.append(list([basename+str(sim_number),'Y',]))
                        count_OK += 1
                    else:
                        # print(f"Sim index: {sim_index}, Before {output[:(sim_index+1)]}")
                        output[sim_index] = 0
                        tab.append(list([basename+str(sim_number),'NA',]))
                        count_NA += 1
                        # print(f"Usable simulations: {np.count_nonzero(output==1)}, countok: {count_OK}, count na: {np.count_nonzero(output==-5)}")

                else:
                    output[sim_index] = 0
                    error_file = os.path.join(sims_folder, f"{basename}{sim_number}.out")
                    error_message = "New error"
                    warning_flag = False
                    with open(error_file,'r') as file:
                        for line in file:
                            # Check if the line contains the specified substring
                            if "mechanic solver diverged" in line:
                                error_message = line.strip()
                            if "MPICH" in line:
                                error_message = "MPICH error"
                                warning_flag = True
                            if "CANCELLED" in line:
                                error_message += " / CANCELLED"
                                warning_flag = True
                                break


                    if warning_flag or error_message == "New error":
                        count_warning += 1
                        

                    tab.append(list([basename+str(sim_number),'N',error_message]))
                    count_notOK += 1
            else:
                if sim_number in unloaded_failed:
                    output[sim_index] = 0
                    tab.append(list([basename+str(sim_number),'N',"Unloading failed"]))
                    count_notOK += 1
                else:
                    output[sim_index] = 0
                    tab.append(list([basename+str(sim_number),'PD',]))
                    count_PD += 1

    else: ### Default simulation
        folder = f"{output_folder}/{basename}default"

        if os.path.exists(folder) and os.path.isfile(f"{folder}/cav.LV.csv"):
               
            print(f"Reading {folder}/cav.LV.csv...")

            lv = read_csv(folder+'/cav.LV.csv', delimiter=",", skipinitialspace=True,
                               header=0, comment='#')
            volume = np.array(lv['Volume'])
            time = np.array(lv['Time'])

            if include_last_AVD:
                check_tend = BCL*NBEATS
            else:
                check_tend = BCL*NBEATS - AVD

            init_last_beat = BCL*(NBEATS-1) - AVD
            end_last_beat = BCL*NBEATS - AVD               

            last_beat = np.intersect1d(np.where(time>=init_last_beat)[0],
                                       np.where(time<=end_last_beat)[0])


            if last_beat.shape[0]>0:
                volume_last_beat = volume[last_beat]
                SV = np.max(volume_last_beat)-np.min(volume_last_beat)
                pressure_lv = np.array(lv['Pressure'][last_beat])
            # print(f"len(lv) is {len(lv)} and should be > 0 ")
            # print(f"max(time) is {max(time)} and should be equal to int(check_tend) which is {int(check_tend)}")
            # print(f"SV is {SV} and should be > 5")
                  
                # Checking that there's an IVR phase:
                dv = np.gradient(volume_last_beat)
                time_pmax = time[0]+np.where(pressure_lv==np.max(pressure_lv))[0][0]
                ind_IVC = np.intersect1d(np.where(np.abs(dv)<=IV_thr)[0],np.where(time<=time_pmax)[0])
                ind_IVR = np.intersect1d(np.where(np.abs(dv)<=IV_thr)[0],np.where(time>=time_pmax)[0])
                ind_ED = ind_IVC[0]
                EDP = pressure_lv[ind_ED]
                dpdt_idx = np.where(pressure_lv>EDP)[0]
                ind_IVR = np.intersect1d(dpdt_idx,ind_IVR)

                # Checking RV IVR phase:
                rv = read_csv(folder+'/cav.RV.csv', delimiter=",", skipinitialspace=True,
                               header=0, comment='#')
                volume_rv = np.array(rv['Volume'][last_beat])
                pressure_rv = np.array(rv['Pressure'][last_beat])

                # dv_rv = np.gradient(volume_rv[last_beat])
                # time_pmax_rv = time[0]+np.where(pressure_rv==np.max(pressure_rv))[0][0]
                # ind_IVC_rv = np.intersect1d(np.where(np.abs(dv_rv)<=0.01)[0],np.where(time<=time_pmax_rv-10.)[0])
                # ind_IVR_rv = np.intersect1d(np.where(np.abs(dv_rv)<=0.01)[0],np.where(time>=time_pmax_rv+10.)[0])
                # ind_ED_rv = ind_IVC_rv[0]
                # EDP_rv = pressure_rv[ind_ED_rv]
                # dpdt_idx_rv = np.where(pressure_rv>EDP_rv)[0]
                # ind_IVR_rv = np.intersect1d(dpdt_idx_rv,ind_IVR_rv)

                


            if len(lv)>0 and max(time) == int(check_tend):

                LV_suitable = check_suitable_VV_output(time[last_beat],volume_last_beat,pressure_lv,t)
                RV_suitable = check_suitable_VV_output(time[last_beat],volume_rv,pressure_rv,t)

                if LV_suitable and RV_suitable and (SV>5.0):

                    output[0] = 1
                    tab.append(list([basename+'default','Y',]))
                    count_OK += 1
                else:
                    output[0] = 0
                    tab.append(list([basename+'default','NA',]))
                    count_NA += 1
            else:
                output[0] = 0
                error_file = os.path.join(sims_folder, f"{basename}{sim_number}.out")
                error_message = "New error"

                warning_flag = False
                with open(error_file,'r') as file:
                    for line in file:
                        # Check if the line contains the specified substring
                        if "mechanic solver diverged" in line:
                            error_message = line.strip()
                        if "MPICH" in line:
                            error_message = "MPICH error"
                            warning_flag = True
                        if "CANCELLED" in line:
                            error_message += " / CANCELLED"
                            warning_flag = True


                if warning_flag or error_message == "New error":
                    count_warning += 1
                tab.append(list([basename+'default','N',error_message]))
                count_notOK += 1
        else:
            if sim_number in unloaded_failed:
                output[0] = 0
                tab.append(list([basename+'default','N',"Unloading failed"]))
                count_notOK += 1
            else:
                output[0] = 0
                tab.append(list([basename+'default','PD',]))
                count_PD += 1

    success_rate = np.round(100*count_OK/(count_OK+count_notOK+count_NA),2)
    tab.append(list(['','OK = '+str(count_OK),]))
    tab.append(list(['','NOT ANALYSABLE = '+str(count_NA),]))
    tab.append(list(['','CRASHED = '+str(count_notOK),]))
    tab.append(list(['','PD = '+str(count_PD),]))
    tab.append(list(['','WARNING = '+str(count_warning),]))
    tab.append(list(['',f'SUCCESS RATE = {success_rate}%',]))

    table = Table(tab)

    table.setStyle(TableStyle([('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
    ('BOX', (0,0), (-1,-1), 0.25, colors.black)
    ]))

    for ii in range(1,len(tab)):
        for jj in range(1,len(tab[ii])):
            if tab[ii][jj]=='Y':
                table.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.lightgreen)]))
            if tab[ii][jj]=='NA':
                table.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.lightgrey)]))
            elif tab[ii][jj]=='N':
                table.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.Color(1, 0.41176, 0.38039) )]))
            elif tab[ii][jj]=='PD':
                table.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.lightyellow)]))
            elif 'CANCELLED' in tab[ii][jj] or 'New error' in tab[ii][jj] or 'MPICH' in tab[ii][jj]:
                table.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.orange)]))
                count_warning+=1
            elif tab[ii][jj]=='Unloading failed':
                table.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.Color(1, 0.41176, 0.38039, alpha=0.5) )]))
            

    items.append(table)
    document.build(items)

    # print(f"Usable simulations: {np.count_nonzero(output == 1)}, countok: {count_OK}, count na: {count_NA}")

    np.savetxt(maskoutput,output,fmt='%s')

def file_exists(full_file_path):
    if not os.path.isfile(full_file_path):
        raise Exception(f"You need to have the file {os.path.abspath(os.path.normpath(full_file_path))}")

def check_suitable_VV_output(time,volume,pressure,t,IV_thr):

    time_pmax = time[0]+np.where(pressure==np.max(pressure))[0][0]
    
    dv = np.gradient(volume)

    ind_IVC_ = np.intersect1d(np.where(np.abs(dv)<=IV_thr)[0],np.where(time<=time_pmax)[0])
    jump = np.where(np.gradient(ind_IVC_)>1)[0]

    if len(jump) == 0:
        ind_IVC = ind_IVC_
    else:
        ind_IVC = ind_IVC_[jump[-1]:-1]

    if len(ind_IVC) <= 5:
        return False

    ind_IVR_ = np.intersect1d(np.where(np.abs(dv)<=IV_thr)[0],np.where(time>=time_pmax)[0])

    if(len(ind_IVR_)) <= 5:
        return False

    jump = np.where(np.gradient(ind_IVR_)>1)[0]

    if len(jump) == 0:
        ind_IVR = ind_IVR_
    else:
        ind_IVR = ind_IVR_[0:jump[0]]

    if(len(ind_IVR)) <= 5:
        return False

    ind_ED = ind_IVC[0]

    EDP = pressure[ind_ED]



    dpdt_idx = np.where(pressure>EDP)[0]
    ind_IVR = np.intersect1d(dpdt_idx,ind_IVR)
    if not len(ind_IVR):
        ind_IVR = np.intersect1d(dpdt_idx,np.where(time>=time_pmax)[0])

    ind_IVC = np.intersect1d(dpdt_idx,ind_IVC)
    
    dpdt_ = np.diff(pressure)/np.diff(time)*1000.0
    dpdt = np.zeros((pressure.shape[0],),dtype=float)
    dpdt[0] = dpdt_[0]
    dpdt[1:] = dpdt_
    # dpdt = np.gradient(pressure)*1000.0

    # to detect oscillations: during IVC the derivative should always be positive
    # if it's not it means that there are oscillations (normally due to the valves)
    # so if we find anywehere the dpdt is negative, we discard whatever happens after
    # because that derivative will be wrong
    wrong_IVC = np.where(dpdt[ind_IVC]<=-50.0)[0]
    if len(wrong_IVC)>0:
        t.set_description('Found oscillations during IVC... Removing indices after oscillation...')
        ind_IVC = ind_IVC[:wrong_IVC[0]]

    # to detect oscillations: during IVR the derivative should always be negative
    # if it's not it means that there are oscillations (normally due to the valves)
    # so if we find anywehere the dpdt is positive, we discard whatever happens after
    # because that derivative will be wrong
    wrong_IVR = np.where(dpdt[ind_IVR]>=50.0)[0]
    if len(wrong_IVR)>0:
        t.set_description('Found oscillations during IVR... Removing indices after oscillation...') # print('Found oscillations during IVR... Removing indices after oscillation...')
        ind_IVR = ind_IVR[:wrong_IVR[0]]
    
    if (len(dpdt[ind_IVR]) > 0) and (len(dpdt[ind_IVC]) > 0):
        return True 
    else:
        return False



def main(args):

    basefolder         = args.basefolder
    simulations_folder = f"{basefolder}/simulations"
    unloaded_volumes   = f"{simulations_folder}/unloaded_volumes.txt"
    data_folder        = f"{basefolder}/data"
    output_folder      = f"{basefolder}/output"
    figures_path       = f"{basefolder}/figures"
    elem_file_local          = args.elem_file
    n_beat             = args.n_beat
    first_simulation   = args.first_simulation
    last_simulation    = args.last_simulation
    default            = args.default

    IV_thr = 0.1 # Max difference in volume for the isovolumic phases

    # file_exists(f'{basefolder}/data/ylabels.txt')
    # file_exists(f'{elem_file}')

    if not default:
        X   = np.loadtxt(f"{basefolder}/data/X.txt")
        with open(f'{basefolder}/data/xlabels.txt', 'r') as file:
            xlabels = file.read().splitlines()
        
        AVD_initial = X[:, xlabels.index('AV_delay')]
        AVD = AVD_initial[first_simulation:(last_simulation+1)]
        print(f'AVD: {AVD}')

        with open(f"{basefolder}/json_files/clinical_data.json", "r") as clinical_data:
            clinical_json = json.load(clinical_data)
        BCL = clinical_json["general"]["BCL"]
    
    else:
        with open(f"{basefolder}/json_files/default.json",'r') as default_file:
            default_json = json.load(default_file)
        AVD = default_json["EP"]["AV_delay"]

        with open(f"{basefolder}/json_files/clinical_data_GENERIC.json", "r") as clinical_data:
            clinical_json = json.load(clinical_data)
        BCL = clinical_json["general"]["BCL"]
    print(f"BCL: {BCL}")
    elem_file = os.path.abspath(os.path.normpath(elem_file_local))
    
    if not os.path.isfile(elem_file):
          
          basename = ''.join(elem_file.split('.')[:-1])
          file_exists(f"{basename}.belem")
          file_exists(f"{basename}.bpts")
          
          os.system(f"meshtool convert -imsh={basename} -omsh={basename} -ifmt=carp_bin -ofmt=carp_txt")

          clean_ascii = True
    else:
          clean_ascii = False

    

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)


    cycle_simulation_summary(output_folder    = simulations_folder,
                                                BCL              = BCL,
                                                AVD              = AVD,
                                                NBEATS           = n_beat,
                                                unloaded_volumes = unloaded_volumes,
                                                start_sample     = first_simulation,
                                                last_sample      = last_simulation,
                                                basename         = "cycle_",
                                                maskoutput       = f"{output_folder}/output_mask_beat_{n_beat}.txt",
                                                output_file      = f"{output_folder}/simulation_summary_beat_{n_beat}.pdf",
                                                include_last_AVD=True,
                                                sims_folder=simulations_folder,
                                                IV_thr=IV_thr
                                            )
    
    labels_computed = cycle_output_free_output_mask_name(datafolder    = output_folder,
                                    output_folder = simulations_folder,
                                    BCL           = BCL,
                                    AVD           = AVD,
                                    NBEATS        = n_beat,
                                    first_simulation=first_simulation,
                                    basename      = "cycle_",
                                    output_file   = f"{data_folder}/Y_mechanics_beat_{n_beat}.txt",
                                    visualise     = False,
                                    output_mask=f"output_mask_beat_{n_beat}.txt",
                                    IV_thr=IV_thr)
    
    labels_timings = timings_output(datafolder    = output_folder,
                                    output_folder = simulations_folder,
                                    BCL           = BCL,
                                    AVD           = AVD,
                                    NBEATS        = n_beat,
                                    first_simulation=first_simulation,
                                    basename      = "cycle_",
                                    output_file   = f"{data_folder}/Y_timings_beat_{n_beat}.txt",
                                    output_mask=f"output_mask_beat_{n_beat}.txt",
                                    IV_thr=IV_thr)

    with open(f"{basefolder}/json_files/tags_lvrv_fch.json","r") as f:
        tags = json.load(f)

    EP_labels = electrophysiology_cycle_output_output_mask_free(datafolder = output_folder,
                                   output_folder = simulations_folder,
                                   elem_file     = elem_file,
                                   tags          = tags,
                                   first_simulation=first_simulation,
                                   basename      = "cycle_",
                                   output_file   = f"{data_folder}/Y_EP_beat_{n_beat}.txt",
                                   output_mask=f"output_mask_beat_{n_beat}.txt")


    

    Y_array = []

    for field in ['mechanics','timings','EP']:
        Y_ = np.loadtxt(f"{data_folder}/Y_{field}_beat_{n_beat}.txt", dtype=float)
        Y_array.append(Y_)
    if isinstance(Y_array[0][0],np.ndarray): # Checking that its dimension is > 1
        Y = np.concatenate(Y_array, axis=1)
    else: # Default simulation or 1 simulation
        Y = np.concatenate(Y_array, axis=0)

    np.savetxt(f"{data_folder}/Y.txt",Y,fmt="%g")

    final_labels = np.concatenate((labels_computed, labels_timings, EP_labels))

    with open(f"{data_folder}/ylabels.txt", "w") as file:
        for label in final_labels:
            file.write(label + "\n")

    if not default:
        plot_statistics_file(basefolder=args.basefolder)
    

    if clean_ascii:
          os.system(f"rm {basename}.elem {basename}.pts {basename}.lon")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate a pdf and a txt file showing the simulations that worked. It also plots the simulations that crashed and the ones who didn't in the parameter space and plots all the pv loops for the ones that worked.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default=os.path.join(os.environ.get("DATA_ROOT", ""), "simulations"),
                        help='Path to the folder where the simulations, data, and figure folders are.')
    parser.add_argument('--elem_file', type=str, help="Path to the elem file of the mesh to compute the activation times.", required=True)
    parser.add_argument('--n_beat', type=int, required=False, help="Heartbeat number to compute the output.", default=5)
    parser.add_argument('--first_simulation', type=int)
    parser.add_argument('--last_simulation', type=int)
    parser.add_argument('--default', action='store_true')

    args = parser.parse_args()

    main(args)
