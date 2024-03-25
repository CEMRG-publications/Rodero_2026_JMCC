import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_total_crashed(X, xlabels, output_mask, figure_savepath, first_simulation, last_simulation):

    output_mask_old = output_mask
    output_mask = []
    output_mask = output_mask_old[first_simulation:(last_simulation+1)]

    num_samples, num_variables = X.shape

    if num_samples > len(output_mask):
        X = X[:len(output_mask),:]
    # Calculate the percentage of 0's in the output mask
    crashed_simulations = np.count_nonzero(output_mask!=1)
    percentage_zeros = 100*crashed_simulations/len(output_mask)


    # Create pair-plots
    _, axes = plt.subplots(
        nrows=num_variables, ncols=num_variables, figsize=(13, 10), sharex="col", sharey="row"
    )

    # Iterate through each pair of variables and create scatter plots
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            x_var = X[first_simulation:(last_simulation+1), j]
            y_var = X[first_simulation:(last_simulation+1), i]

            # Plot the data points with different colors based on the output_mask values
            ax.scatter(
                x_var[output_mask != 1],
                y_var[output_mask != 1],
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
    plt.suptitle(f"Total crashed simulations: {crashed_simulations}/{len(output_mask)} ({percentage_zeros:.2f}%)", fontsize=16)

    # Move the legend to avoid overlapping
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    # Add a common legend for all the subplots
    axes[0, -1].legend(loc=[1,0])

    plt.savefig(os.path.join(figure_savepath,"success_crashed_total.png"), dpi=300, bbox_inches='tight')


def plot_crashed_cycles(X, xlabels, output_mask, figure_savepath):

    num_samples, num_variables = X.shape

    if num_samples > len(output_mask):
        X = X[:len(output_mask),:]

    # Calculate the percentage of 0's in the output mask
    crashed_simulations = np.count_nonzero(output_mask==0)
    percentage_zeros = 100*crashed_simulations/np.count_nonzero(output_mask!=-1)


    # Create pair-plots
    _, axes = plt.subplots(
        nrows=num_variables, ncols=num_variables, figsize=(13, 10), sharex="col", sharey="row"
    )

    # Iterate through each pair of variables and create scatter plots
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            x_var = X[:, j]
            y_var = X[:, i]

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
    plt.suptitle(f"Crashed cycle simulations: {crashed_simulations}/{np.count_nonzero(output_mask!=-1)} ({percentage_zeros:.2f}%)", fontsize=16)

    # Move the legend to avoid overlapping
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    # Add a common legend for all the subplots
    axes[0, -1].legend(loc=[1,0])

    plt.savefig(os.path.join(figure_savepath,"success_crashed_cycle.png"), dpi=300, bbox_inches='tight')

def plot_crashed_cycles_only_by_phase(X, xlabels, output_mask, figure_savepath, sims_folder, chamber, bcl):

    files_per_last_timestep = {
        "First beat" : np.array([0, 0, 0, 0, 0]),
        "Second beat": np.array([0, 0, 0, 0, 0]),
        "Third beat" : np.array([0, 0, 0, 0, 0]),
        "Fourth beat": np.array([0, 0, 0, 0, 0]),
        "Fifth beat" : np.array([0, 0, 0, 0, 0])
    }

    beats_indices = {
        0: "First beat",
        1: "Second beat",
        2: "Third beat",
        3: "Fourth beat",
        4: "Fifth beat"
    }
    phases_indices = {
        "IVC" : 0,
        "ejec": 1,
        "IVR" : 2,
        "fill": 3,
        "load": 4
    }

    colors_by_phase = {
        "IVC" : '#d55e00',
        "ejec": '#cc79a7',
        "IVR" : '#0072b2',
        "fill": '#f0e442',
        "load": '#000000'
    }

    marker_by_phase = {
        "IVC" : '.',
        "ejec": 'p',
        "IVR" : 'x',
        "fill": 'd',
        "load": 's'
    }

    color_array = []
    marker_array = []

    array_phase = []

    for simnum in range(len(output_mask)):
        if output_mask[simnum] == 0 and os.path.isfile(f"{sims_folder}/cycle_{simnum}/cav.{chamber}.csv"):
            csv_file = f"{sims_folder}/cycle_{simnum}/cav.{chamber}.csv"

        # Read the CSV file and extract information
            df = pd.read_csv(csv_file, skiprows=[1])
            header_without_spaces = [x.strip(' ') for x in df.columns.values.tolist()]
            df.columns = header_without_spaces
            last_state = df['State'].iloc[-1].strip(' ')
            last_time = df['Time'].iloc[-1]

            array_phase.append(phases_indices[last_state])

            heartbeat_number = beats_indices[min(int(last_time/bcl),4)]
            cycle_phase_index = phases_indices[last_state]

            color_array.append(colors_by_phase[last_state])
            marker_array.append(marker_by_phase[last_state])


            files_per_last_timestep[heartbeat_number][cycle_phase_index] += 1
        else:
            array_phase.append(-1)

    
    array_phase = np.array(array_phase)


    num_samples, num_variables = X.shape

    if num_samples > len(output_mask):
        X = X[:len(output_mask),:]


    # Create pair-plots
    _, axes = plt.subplots(
        nrows=num_variables, ncols=num_variables, figsize=(13, 10), sharex="col", sharey="row"
    )

    # Iterate through each pair of variables and create scatter plots
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            x_var = X[:, j]
            y_var = X[:, i]

            # Plot the data points with different colors based on the output_mask values
            ax.scatter(
                x_var[array_phase == 0],
                y_var[array_phase == 0],
                c=colors_by_phase["IVC"],
                marker=marker_by_phase["IVC"],
                label="IVC",
                s=20
            )
            ax.scatter(
                x_var[array_phase == 1],
                y_var[array_phase == 1],
                c=colors_by_phase["ejec"],
                marker=marker_by_phase["ejec"],
                label="ejec",
                s=20
            )
            ax.scatter(
                x_var[array_phase == 2],
                y_var[array_phase == 2],
                c=colors_by_phase["IVR"],
                marker=marker_by_phase["IVR"],
                label="IVR",
                s=20
            )
            ax.scatter(
                x_var[array_phase == 3],
                y_var[array_phase == 3],
                c=colors_by_phase["fill"],
                marker=marker_by_phase["fill"],
                label="fill",
                s=20
            )

            ax.scatter(
                x_var[array_phase == 4],
                y_var[array_phase == 4],
                c=colors_by_phase["load"],
                marker=marker_by_phase["load"],
                label="load",
                s=20
            )

            if i == num_variables - 1:
                ax.set_xlabel(xlabels[j])
            if j == 0:
                ax.set_ylabel(xlabels[i])

    # Add a common title for all the subplots
    plt.suptitle(f"Cardiac phase where the simulation crashed in the {chamber}", fontsize=16)

    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    # Add a common legend for all the subplots
    axes[0, -1].legend(loc=[1,0])

    plt.savefig(os.path.join(figure_savepath,f"cardiac_phase_crashed_{chamber}.png"), dpi=300, bbox_inches='tight')

def plot_crashed_cycles_only_by_time(X, xlabels, output_mask, figure_savepath, sims_folder, bcl):

    colors_by_time = {
        0 : '#d55e00',
        1: '#cc79a7',
        2 : '#0072b2',
        3: '#f0e442',
        4: '#009e73',
        5: '#6a0dad'
    }

    marker_by_time = {
        0: '.',
        1: 'p',
        2: 'x',
        3: 'd',
        4: '<',
        5: '^' 
    }

    color_array = []
    marker_array = []

    array_time = []

    for simnum in range(len(output_mask)):
        if output_mask[simnum] == 0 and os.path.isfile(f"{sims_folder}/cycle_{simnum}/cav.LV.csv"):
            csv_file = f"{sims_folder}/cycle_{simnum}/cav.LV.csv"
            
            df = pd.read_csv(csv_file, skiprows=[1])
            header_without_spaces = [x.strip(' ') for x in df.columns.values.tolist()]
            df.columns = header_without_spaces

            last_time = df['Time'].iloc[-1]

            heartbeat_number = int(last_time/bcl)
            array_time.append(heartbeat_number)

            color_array.append(colors_by_time[heartbeat_number])
            marker_array.append(marker_by_time[heartbeat_number])

        else:
            array_time.append(-1)

    
    array_time = np.array(array_time)

    num_samples, num_variables = X.shape

    if num_samples > len(output_mask):
        X = X[:len(output_mask),:]

    # Create pair-plots
    _, axes = plt.subplots(
        nrows=num_variables, ncols=num_variables, figsize=(13, 10), sharex="col", sharey="row"
    )

    # Iterate through each pair of variables and create scatter plots
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            x_var = X[:, j]
            y_var = X[:, i]

            # Plot the data points with different colors based on the output_mask values
            ax.scatter(
                x_var[array_time == 0],
                y_var[array_time == 0],
                c=colors_by_time[0],
                marker=marker_by_time[0],
                label="Beat 1",
                s=20
            )
            ax.scatter(
                x_var[array_time == 1],
                y_var[array_time == 1],
                c=colors_by_time[1],
                marker=marker_by_time[1],
                label="Beat 2",
                s=20
            )
            ax.scatter(
                x_var[array_time == 2],
                y_var[array_time == 2],
                c=colors_by_time[2],
                marker=marker_by_time[2],
                label="Beat 3",
                s=20
            )
            ax.scatter(
                x_var[array_time == 3],
                y_var[array_time == 3],
                c=colors_by_time[3],
                marker=marker_by_time[3],
                label="Beat 4",
                s=20
            )
            ax.scatter(
                x_var[array_time == 4],
                y_var[array_time == 4],
                c=colors_by_time[4],
                marker=marker_by_time[4],
                label="Beat 5",
                s=20
            )

            ax.scatter(
                x_var[array_time == 5],
                y_var[array_time == 5],
                c=colors_by_time[5],
                marker=marker_by_time[5],
                label="Beat 5 (no ejection)",
                s=20
            )

            if i == num_variables - 1:
                ax.set_xlabel(xlabels[j])
            if j == 0:
                ax.set_ylabel(xlabels[i])

    # Add a common title for all the subplots
    plt.suptitle(f"Heart beat when the simulation crashed", fontsize=16)

    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    # Add a common legend for all the subplots
    axes[0, -1].legend(loc=[1,0])

    plt.savefig(os.path.join(figure_savepath,f"hearbeat_crashed.png"), dpi=300, bbox_inches='tight')


def main(args):

    data_folder        = f"{args.basefolder}/data"
    output_folder      = f"{args.basefolder}/output"
    figures_path       = f"{args.basefolder}/figures"
    sims_folder        = f"{args.basefolder}/simulations"
    first_simulation   = args.first_simulation
    last_simulation    = args.last_simulation
    BCL                = args.BCL
    n_beat             = args.n_beat

    os.makedirs(output_folder, exist_ok=True)

    X       = np.loadtxt(f"{data_folder}/X.txt")
    xlabels = np.loadtxt(f"{data_folder}/xlabels.txt", dtype=str)
    
    output_mask_cycle = np.loadtxt(f"{output_folder}/output_mask_beat_{n_beat}.txt")

    plot_crashed_cycles(X               = X,
                        xlabels         = xlabels,
                        output_mask     = output_mask_cycle,
                        figure_savepath = figures_path) 
    
    plot_total_crashed(X               = X,
                       xlabels         = xlabels,
                       output_mask     = output_mask_cycle,
                       figure_savepath = figures_path,
                       first_simulation= first_simulation,
                       last_simulation = last_simulation) 

    plot_crashed_cycles_only_by_phase(X               = X, 
                                      xlabels         = xlabels, 
                                      output_mask     = output_mask_cycle, 
                                      figure_savepath = figures_path,
                                      sims_folder     = sims_folder,
                                      chamber         = "LV",
                                      bcl             = BCL)
    
    plot_crashed_cycles_only_by_phase(X               = X, 
                                      xlabels         = xlabels, 
                                      output_mask     = output_mask_cycle, 
                                      figure_savepath = figures_path,
                                      sims_folder     = sims_folder,
                                      chamber         = "RV",
                                      bcl             = BCL)
    
    plot_crashed_cycles_only_by_phase(X               = X, 
                                      xlabels         = xlabels, 
                                      output_mask     = output_mask_cycle, 
                                      figure_savepath = figures_path,
                                      sims_folder     = sims_folder,
                                      chamber         = "LA",
                                      bcl             = BCL)
    
    plot_crashed_cycles_only_by_phase(X               = X, 
                                      xlabels         = xlabels, 
                                      output_mask     = output_mask_cycle, 
                                      figure_savepath = figures_path,
                                      sims_folder     = sims_folder,
                                      chamber         = "RA",
                                      bcl             = BCL)
    
    plot_crashed_cycles_only_by_time(X               = X,
                                     xlabels         = xlabels, 
                                     output_mask     = output_mask_cycle, 
                                     figure_savepath = figures_path,
                                     sims_folder     = sims_folder,
                                     bcl             = BCL)
    

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    

    parser.add_argument('--basefolder', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/h01/new_unloading/unloading_simulations")
    parser.add_argument('--first_simulation', type=int, required=True, default=0)
    parser.add_argument('--last_simulation', type=int, required=True, default=99)
    parser.add_argument('--BCL', type=int, required=True)
    parser.add_argument('--n_beat', type=int, required=False, help="Heartbeat number to compute the output.", default=5)

    args = parser.parse_args()

    main(args)
