import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import json

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

from SIMULATION_library import fourchamber_output, mesh_utils

def plot_crashed_simulations(X, xlabels, output_mask, figure_savepath, first_simulation, last_simulation):

    num_points, num_variables = X.shape

    # Calculate the percentage of 0's in the output mask
    percentage_zeros = np.mean(output_mask == 0) * 100

    # Create pair-plots
    fig, axes = plt.subplots(
        nrows=num_variables, ncols=num_variables, figsize=(10, 10), sharex="col", sharey="row"
    )

    # Iterate through each pair of variables and create scatter plots
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            x_var = X[first_simulation:(last_simulation+1), j]
            y_var = X[first_simulation:(last_simulation+1), i]

            # Plot the data points with different colors based on the output_mask values
            ax.scatter(
                x_var[output_mask == 0],
                y_var[output_mask == 0],
                c="red",
                marker="x",
                label="Crashed",
                s=20
            )
            ax.scatter(
                x_var[output_mask == 1],
                y_var[output_mask == 1],
                c="green",
                marker="o",
                label="Converged",
                s=20
            )

            if i == num_variables - 1:
                ax.set_xlabel(xlabels[j])
            if j == 0:
                ax.set_ylabel(xlabels[i])

    # Add a common title for all the subplots
    plt.suptitle(f"Crashed simulations: {percentage_zeros:.2f}%", fontsize=16)

    # Move the legend to avoid overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add a common legend for all the subplots
    axes[0, -1].legend(loc=[1,0])

    os.makedirs(figure_savepath, exist_ok=True)

    plt.savefig(os.path.join(figure_savepath,"success_crashed.png"), dpi=300)

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

    # Calculate ejection fractions for all cases
    LVEF_all = 100 * (data[:, labels.index('LVedv')] - data[:, labels.index('LVesv')]) / data[:, labels.index('LVedv')]
    RVEF_all = 100 * (data[:, labels.index('RVedv')] - data[:, labels.index('RVesv')]) / data[:, labels.index('RVedv')]
    LAEF_all = 100 * (data[:, labels.index('LAedv')] - data[:, labels.index('LAesv')]) / data[:, labels.index('LAedv')]
    RAEF_all = 100 * (data[:, labels.index('RAedv')] - data[:, labels.index('RAesv')]) / data[:, labels.index('RAedv')]

    # Calculate mean and standard deviation of ejection fractions
    mean_LVEF = np.mean(LVEF_all)
    std_LVEF = np.std(LVEF_all)
    min_LVEF = np.min(LVEF_all)
    max_LVEF = np.max(LVEF_all)
    mean_RVEF = np.mean(RVEF_all)
    std_RVEF = np.std(RVEF_all)
    min_RVEF = np.min(RVEF_all)
    max_RVEF = np.max(RVEF_all)
    mean_LAEF = np.mean(LAEF_all)
    std_LAEF = np.std(LAEF_all)
    min_LAEF = np.min(LAEF_all)
    max_LAEF = np.max(LAEF_all)
    mean_RAEF = np.mean(RAEF_all)
    std_RAEF = np.std(RAEF_all)
    min_RAEF = np.min(RAEF_all)
    max_RAEF = np.max(RAEF_all)

    # Save results to a file
    with open(f'{basefolder}/output/output_statistics.csv', 'w') as file:
        file.write("Variable\tMean\tStd\tMin\tMax\n")
        for label, mean, std, min, max in zip(labels, mean_values, std_values, min_values, max_values):
            file.write(f"{label}\t{mean:.2f}\t{std:.2f}\t{min:.2f}\t{max:.2f}\n")

        # Add ejection fractions to the file
        file.write(f"LVEF\t{mean_LVEF:.2f}\t{std_LVEF:.2f}\t{min_LVEF:.2f}\t{max_LVEF:.2f}\n")
        file.write(f"RVEF\t{mean_RVEF:.2f}\t{std_RVEF:.2f}\t{min_RVEF:.2f}\t{max_RVEF:.2f}\n")
        file.write(f"LAEF\t{mean_LAEF:.2f}\t{std_LAEF:.2f}\t{min_LAEF:.2f}\t{max_LAEF:.2f}\n")
        file.write(f"RAEF\t{mean_RAEF:.2f}\t{std_RAEF:.2f}\t{min_RAEF:.2f}\t{max_RAEF:.2f}\n")

def print_PV_loops_all_cycles(path_to_simulation, BCL, case_number):
    
    chambers = ['LV', 'RV', 'LA', 'RA']
    colours = ['red', 'blue', '#F6BE00', 'green']
    
    ax = plt.figure(figsize=(10,10), constrained_layout=True).subplots(2, 2)
    ax = ax.flatten()
    n_cycles = 5

    for j, chamber_name in enumerate(chambers):
        
        for n in range(n_cycles):

            chamber_structure = read_csv(os.path.join(path_to_simulation,'cav.'+chamber_name+'.csv'), delimiter=",",
                          skipinitialspace=True, header=0, comment='#')
            time = np.array(chamber_structure['Time'])

    #         start = time[-1]-BCL
            start = time[0] + n*BCL
            end = start + BCL

            plot_time = np.where((time>=start) & (time<end))[0]

            volume = np.array(chamber_structure['Volume'][plot_time])
            pressure = np.array(chamber_structure['Pressure'][plot_time])
            ax[j].plot(volume,pressure,color=colours[j],linewidth=3.0, alpha = 0.1+n*((1-0.1)/n_cycles))
            ax[j].set_xlabel(chamber_name+' volume [mL]')
            ax[j].set_ylabel(chamber_name+' pressure [mmHg]')
        EF = round(100*(np.max(volume)-np.min(volume))/np.max(volume),2)
        ax[j].text(0.95, 0.95, 'EF: ' + str(EF) + "%", horizontalalignment='right', verticalalignment='top',
                transform=ax[j].transAxes)
    plt.suptitle(f"Simulation #{case_number}", fontsize=20, weight='bold')
    plt.savefig(f"{path_to_simulation}/../../figures/{case_number}_pv_loops_all_cycles.png",dpi=300)
    plt.close('all')
    
def cycle_output_free_output_mask_name(datafolder,
                 output_folder,
                 BCL,
                 NBEATS,
                 AVD,
                 basename="cycle_",
                 output_file="Y.txt",
                 visualise=True,
                 output_mask="output_mask.txt"):

    print('WARNING: remember that the AVD is needed to compute the last cycle, as the simulation starts @ -AVD ms')

    print('Computing only output for successful simulations...')

    mask = np.loadtxt(f"{datafolder}/{output_mask}",dtype=int)
    idx_ok = np.where(mask==1)[0]

    output = np.zeros((idx_ok.shape[0],26))

    for i in range(idx_ok.shape[0]):
        print('Simulation '+basename+str(idx_ok[i])+'...')
        folder = output_folder+'/'+basename+str(idx_ok[i])

        lv = read_csv(folder+'/cav.LV.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')
        rv = read_csv(folder+'/cav.RV.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')
        la = read_csv(folder+'/cav.LA.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')
        ra = read_csv(folder+'/cav.RA.csv', delimiter=",", skipinitialspace=True,
                           header=0, comment='#')

        time = np.array(lv['Time'])
        start = int((NBEATS-1)*BCL-AVD[i])
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

        lvoutput = fourchamber_output.VV_output(time,volume_lv,pressure_lv,BCL)
        rvoutput = fourchamber_output.VV_output(time,volume_rv,pressure_rv,BCL)
        laoutput = fourchamber_output.AA_output(time,volume_la,pressure_la)
        raoutput = fourchamber_output.AA_output(time,volume_ra,pressure_ra)

        laoutput_ej = fourchamber_output.AA_output_ej(time,volume_la)
        raoutput_ej = fourchamber_output.AA_output_ej(time,volume_ra)

        if visualise:
            fourchamber_output.check_ventricle_output(time,volume_lv,pressure_lv,lvoutput)
            fourchamber_output.check_ventricle_output(time,volume_rv,pressure_rv,rvoutput)
            fourchamber_output.check_atria_output(time,volume_la,pressure_la,laoutput)
            fourchamber_output.check_atria_output(time,volume_ra,pressure_rv,raoutput)

        output[i,:] = np.concatenate((lvoutput,rvoutput,
                                      laoutput,raoutput,
                                      laoutput_ej,raoutput_ej),axis=0)

    np.savetxt(output_file, output, fmt='%.2f')

def electrophysiology_cycle_output_output_mask_free(datafolder,
                                   output_folder,
                                   elem_file,
                                   tags,
                                   basename="cycle_",
                                   output_file="Y_EP.txt",
                                   output_mask="output_mask.txt"):

    print('Computing only output for successful simulations...')

    mask = np.loadtxt(f"{datafolder}/{output_mask}",dtype=int)
    idx_ok = np.where(mask==1)[0]

    print('Reading mesh elem file...')
    elem = mesh_utils.read_tets(elem_file)
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

    for i in range(idx_ok.shape[0]):
        print('Simulation '+basename+str(idx_ok[i])+'...')

        folder = output_folder+'/'+basename+str(idx_ok[i])

        AT=np.loadtxt(folder+"/vm_act_seq.dat",dtype=float)
        if (np.min(AT[V_VTX]<0)):
            raise Exception("The ventricles contain a negative activation time.")
        if (np.min(AT[A_VTX]<0)):
            raise Exception("The atria contain a negative activation time.")
            
        output[i,0] = np.max(AT[A_VTX])-np.min(AT[A_VTX])
        output[i,1] = np.max(AT[V_VTX])-np.min(AT[V_VTX])

    np.savetxt(output_file,output,fmt="%g")


def cycle_simulation_summary(output_folder,
							 BCL,
							 AVD,
							 NBEATS,
							 start_sample=0,
							 last_sample=0,
							 basename="cycle",
							 maskoutput="output_mask.txt",
							 output_file="simulation_summary.pdf",
							 unloaded_volumes=None,
							 include_last_AVD=False):
	
	output = np.zeros((last_sample-start_sample+1,),dtype=int)

	if unloaded_volumes is not None:
		unloaded = np.loadtxt(unloaded_volumes,dtype=float)
		unloaded_failed = np.where(unloaded[:,0]==-1)[0]
	else:
		unloaded_failed = []

	document = SimpleDocTemplate(output_file, pagesize=A4, title='Simulation Summary')
	tab = []
	items = []
	header = ['sim','success']
	tab.append(header)

	count_PD = 0
	count_OK = 0
	count_notOK = 0
	for i in range(start_sample,last_sample+1):
		folder = output_folder+'/'+basename+str(i)

		if os.path.exists(folder) and os.path.isfile(f"{folder}/cav.LV.csv"):
               
			print(f"Reading {folder}/cav.LV.csv...")

			lv = read_csv(folder+'/cav.LV.csv', delimiter=",", skipinitialspace=True,
							   header=0, comment='#')
			volume = np.array(lv['Volume'])
			time = np.array(lv['Time'])

			if include_last_AVD:
				check_tend = BCL*NBEATS
			else:
				check_tend = BCL*NBEATS - AVD[i]

			init_last_beat = BCL*(NBEATS-1) - AVD[i]
			end_last_beat = BCL*NBEATS - AVD[i]				

			last_beat = np.intersect1d(np.where(time>=init_last_beat)[0],
									   np.where(time<=end_last_beat)[0])

			if last_beat.shape[0]>0:
				volume_last_beat = volume[last_beat]
				SV = np.max(volume_last_beat)-np.min(volume_last_beat)
			# print(f"len(lv) is {len(lv)} and should be > 0 ")
			# print(f"max(time) is {max(time)} and should be equal to int(check_tend) which is {int(check_tend)}")
			# print(f"SV is {SV} and should be > 5")
			if len(lv)>0 and max(time) == int(check_tend) and (SV>5.0):
				output[i] = 1
				tab.append(list([basename+str(i),'Y']))
				count_OK += 1
			else:
				output[i] = 0
				tab.append(list([basename+str(i),'N']))
				count_notOK += 1
		else:
			if i in unloaded_failed:
				output[i] = 0
				tab.append(list([basename+str(i),'N']))
				count_notOK += 1
			else:
				output[i] = -1
				tab.append(list([basename+str(i),'PD']))
				count_PD += 1

	tab.append(list(['','OK = '+str(count_OK)]))
	tab.append(list(['','CRASHED = '+str(count_notOK)]))
	tab.append(list(['','PD = '+str(count_PD)]))
	table = Table(tab)

	table.setStyle(TableStyle([('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
	('BOX', (0,0), (-1,-1), 0.25, colors.black)
	]))

	for ii in range(1,len(tab)):
		for jj in range(1,len(tab[ii])):
			if tab[ii][jj]=='Y':
				table.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.lightgreen)]))
			elif tab[ii][jj]=='N':
				table.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.fidred)]))
			elif tab[ii][jj]=='PD':
				table.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.lightyellow)]))

	items.append(table)
	document.build(items)

	np.savetxt(maskoutput,output,fmt='%s')

def file_exists(full_file_path):
    if not os.path.isfile(full_file_path):
        raise Exception("You need to have the file " + full_file_path)

def main(args):

    basefolder         = args.basefolder
    simulations_folder = f"{basefolder}/simulations"
    unloaded_volumes   = f"{simulations_folder}/unloaded_volumes.txt"
    data_folder        = f"{basefolder}/data"
    output_folder      = f"{basefolder}/output"
    figures_path       = f"{basefolder}/figures"
    BCL                = args.BCL
    elem_file          = args.elem_file
    n_beat             = args.n_beat
    first_simulation   = args.first_simulation
    last_simulation    = args.last_simulation

    file_exists(f'{basefolder}/data/ylabels.txt')
    file_exists(f'{elem_file}')

    

    os.makedirs(output_folder, exist_ok=True)

    cycle_simulation_summary(output_folder    = simulations_folder,
                                                BCL              = BCL,
                                                AVD              = 200*[0],
                                                NBEATS           = n_beat,
                                                unloaded_volumes = unloaded_volumes,
                                                start_sample     = first_simulation,
                                                last_sample      = last_simulation,
                                                basename         = "cycle_",
                                                maskoutput       = f"{output_folder}/output_mask_beat_{n_beat}.txt",
                                                output_file      = f"{output_folder}/simulation_summary_beat_{n_beat}.pdf"
                                            )
    
    cycle_output_free_output_mask_name(datafolder    = output_folder,
                                    output_folder = simulations_folder,
                                    BCL           = BCL,
                                    AVD           = 100*[100],
                                    NBEATS        = n_beat,
                                    basename      = "cycle_",
                                    output_file   = f"{data_folder}/Y_mechanics_beat_{n_beat}.txt",
                                    visualise     = False,
                                    output_mask=f"output_mask_beat_{n_beat}.txt")
    
    output_mask = np.loadtxt(f"{output_folder}/output_mask_beat_{n_beat}.txt")
    with open(f"{basefolder}/json_files/tags.json","r") as f:
        tags = json.load(f)

    electrophysiology_cycle_output_output_mask_free(datafolder = output_folder,
                                   output_folder = simulations_folder,
                                   elem_file     = elem_file,
                                   tags          = tags,
                                   basename      = "cycle_",
                                   output_file   = f"{data_folder}/Y_EP_beat_{n_beat}.txt",
                                   output_mask=f"output_mask_beat_{n_beat}.txt")


    fourchamber_output.plot_pvloops_all(datafolder    = output_folder,
                                        output_folder = simulations_folder,
                                        BCL           = BCL,
                                        basename      = "cycle_",
                                        mask_file     = f"{output_folder}/output_mask_beat_{n_beat}.txt",
                                        NBEATS        = n_beat,
                                        figname       = f"{figures_path}/all_pv_loops_beat_{n_beat}.png")
    
    Y_array = []

    for field in ['mechanics','EP']:
        Y_ = np.loadtxt(f"{data_folder}/Y_{field}_beat_{n_beat}.txt", dtype=float)
        Y_array.append(Y_)

    Y = np.concatenate(Y_array, axis=1)

    np.savetxt(f"{data_folder}/Y.txt",Y,fmt="%g")
    
    plot_statistics_file(basefolder=args.basefolder)

    for index, value in enumerate(output_mask):
        # If the value is 1, plot the pv loop
        if value == 1:
               print(f"Plotting PV loops of simulation #{index}...")
               path2simulation = f"{simulations_folder}/cycle_{index}"
               
               print_PV_loops_all_cycles(path_to_simulation = path2simulation,BCL = BCL, case_number=index)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate a pdf and a txt file showing the simulations that worked. It also plots the simulations that crashed and the ones who didn't in the parameter space and plots all the pv loops for the ones that worked.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/h01/new_unloading/unloading_simulations",
                        help='Path to the folder where the simulations, data, and figure folders are.')
    parser.add_argument('--BCL', type=int, required=False, default=1000)
    parser.add_argument('--elem_file', type=str, help="Path to the elem file of the mesh to compute the activation times.", required=True)
    parser.add_argument('--n_beat', type=int, required=False, help="Heartbeat number to compute the output.", default=5)
    parser.add_argument('--first_simulation', type=int)
    parser.add_argument('--last_simulation', type=int)

    args = parser.parse_args()

    main(args)
