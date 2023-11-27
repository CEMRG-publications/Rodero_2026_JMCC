from pandas import read_csv
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def print_PV_loops(path_to_simulation, figtitle):
    
    chambers = ['LV', 'RV', 'LA', 'RA']
    colours = ['red', 'blue', '#F6BE00', 'green']
    BCL = 1000
    
    ax = plt.figure(figsize=(10,10), constrained_layout=True).subplots(2, 2)
    ax = ax.flatten()

    for j, chamber_name in enumerate(chambers):

        chamber_structure = read_csv(os.path.join(path_to_simulation,'cav.'+chamber_name+'.csv'), delimiter=",",
                      skipinitialspace=True, header=0, comment='#')
        time = np.array(chamber_structure['Time'])

        start = time[-1]-BCL
#         start = time[0]

        plot_time = np.where(time>=start)[0]

        volume = np.array(chamber_structure['Volume'][plot_time])
        pressure = np.array(chamber_structure['Pressure'][plot_time])
        alphas = np.linspace(0.1, 1, len(volume))
        ax[j].plot(volume,pressure,color=colours[j],linewidth=3.0)
        ax[j].set_xlabel(chamber_name+' volume [mL]')
        ax[j].set_ylabel(chamber_name+' pressure [mmHg]')
        EF = round(100*(np.max(volume)-np.min(volume))/np.max(volume),2)
        ax[j].text(0.95, 0.95, 'EF: ' + str(EF) + "%", horizontalalignment='right', verticalalignment='top',
                   transform=ax[j].transAxes)
        
    plt.suptitle(figtitle, fontsize=20, weight='bold')
#     plt.show()
    plt.savefig(os.path.join(path_to_simulation,'pv_loops.png'),dpi=300)
    
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

def main(args):
      
    basefolder = args.basefolder
    BCL = args.BCL

    path2output = f"{basefolder}/output"
    
    output_mask = np.loadtxt(f"{path2output}/output_mask.txt")
    
    for index, value in enumerate(output_mask):
        # If the value is 1, plot the pv loop
        if value == 1:
               print(f"Plotting PV loops of simulation #{index}...")
               path2simulation = f"{basefolder}/simulations/cycle_{index}"
               
               print_PV_loops_all_cycles(path_to_simulation = path2simulation,BCL = BCL, case_number=index)

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    

    parser.add_argument('--basefolder', type=str, required=True)
    parser.add_argument('--BCL', type=float, required=True)

    args = parser.parse_args()

    main(args)
