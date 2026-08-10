import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json 
import tqdm

def plot_total_crashed(X, xlabels, output_mask, figure_savepath, first_simulation, last_simulation,only_prints=False,print_separately=False,print_pair_plots=True):

    output_mask_old = output_mask
    output_mask = []
    output_mask = output_mask_old[first_simulation:(last_simulation+1)]

    num_samples, num_variables = X.shape

    if num_samples > len(output_mask):
        X = X[:len(output_mask),:]
    # Calculate the percentage of 0's in the output mask
    crashed_simulations = np.count_nonzero(output_mask!=1)
    percentage_zeros = 100*crashed_simulations/len(output_mask)

    if not only_prints:

        if print_pair_plots:
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
        
        if print_separately:
            os.makedirs(f"{figure_savepath}/total_crashed", exist_ok=True)
            t = tqdm.trange(num_variables, desc='Bar desc', leave=True,colour='#8B0000')
            print_message_outer = "Plotting total crashes depending on value of " 
            biggest_label = max([len(lab) for lab in xlabels])
            total_message_length_outer = len(print_message_outer) + biggest_label + 3
            for i in t:
                t.set_description(f"{print_message_outer}{xlabels[i]}...".ljust(total_message_length_outer))
                t2 = tqdm.trange(i, desc='Bar desc', leave=True,colour='#FF5733')
                print_message = "..against " 
                total_message_length = len(print_message) + biggest_label
                for j in t2:
                    t2.set_description(f"{print_message}{xlabels[j]}".ljust(total_message_length))
                    plt.figure(figsize=(6, 5))
                    
                    x_var = X[first_simulation:(last_simulation + 1), j]
                    y_var = X[first_simulation:(last_simulation + 1), i]
                    
                    # Plot the data points with different colors based on the output_mask values
                    plt.scatter(
                        x_var[output_mask[first_simulation:(last_simulation + 1)] != 1],
                        y_var[output_mask[first_simulation:(last_simulation + 1)] != 1],
                        c="red",
                        marker="x",
                        label="Crashed",
                        s=20
                    )
                    plt.scatter(
                        x_var[output_mask[first_simulation:(last_simulation + 1)] == 1],
                        y_var[output_mask[first_simulation:(last_simulation + 1)] == 1],
                        c="green",
                        marker="o",
                        label="Converged",
                        s=20
                    )

                    # Set labels
                    plt.xlabel(xlabels[j])
                    plt.ylabel(xlabels[i])

                    # Add a title
                    plt.title(f"Total crashed simulations: {crashed_simulations}/{len(output_mask)} ({percentage_zeros:.2f}%)", fontsize=16)

                    # Add legend
                    plt.legend()

                    # Save the figure for this pair
                    filename = f"crashed_total_{xlabels[j]}_vs_{xlabels[i]}.png".replace(" ", "_")
                    
                    plt.savefig(f"{figure_savepath}/total_crashed/{filename}", dpi=300, bbox_inches='tight')
                    plt.close()

    print(f"Total crashed simulations: {crashed_simulations}/{len(output_mask)} ({percentage_zeros:.2f}%)")


def plot_crashed_cycles(X, xlabels, output_mask, figure_savepath, only_prints=False,print_pair_plots=True,print_separately=False):

    num_samples, num_variables = X.shape

    if num_samples > len(output_mask):
        X = X[:len(output_mask),:]

    # Calculate the percentage of 0's in the output mask
    crashed_simulations = np.count_nonzero(output_mask==0)
    percentage_zeros = 100*crashed_simulations/np.count_nonzero(output_mask!=-1)

    if not only_prints:

        if print_pair_plots:
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
                
        if print_separately:
            os.makedirs(f"{figure_savepath}/crashed_cycles",exist_ok=True)
            t = tqdm.trange(num_variables, desc='Bar desc', leave=True,colour='#8B0000')
            print_message_outer = "Plotting crashes depending on value of " 
            biggest_label = max([len(lab) for lab in xlabels])
            total_message_length_outer = len(print_message_outer) + biggest_label + 3
            for i in t:
                t.set_description(f"{print_message_outer}{xlabels[i]}...".ljust(total_message_length_outer))
                t2 = tqdm.trange(i, desc='Bar desc', leave=True,colour='#FF5733')
                print_message = "..against " 
                total_message_length = len(print_message) + biggest_label
                for j in t2:
                    t2.set_description(f"{print_message}{xlabels[j]}".ljust(total_message_length))
                    plt.figure(figsize=(6, 5))
                    
                    x_var = X[:, j]
                    y_var = X[:, i]
                    
                    # Plot the data points with different colors based on the output_mask values
                    plt.scatter(
                        x_var[output_mask == 0],
                        y_var[output_mask == 0],
                        c="red",
                        marker="x",
                        label="Crashed",
                        s=20
                    )
                    plt.scatter(
                        x_var[output_mask == 1],
                        y_var[output_mask == 1],
                        c="green",
                        marker="o",
                        label="Converged",
                        s=20
                    )

                    # Set labels
                    plt.xlabel(xlabels[j])
                    plt.ylabel(xlabels[i])

                    # Add a title
                    plt.title(f"Crashed cycle simulations: {crashed_simulations}/{np.count_nonzero(output_mask!=-1)} ({percentage_zeros:.2f}%)", fontsize=16)

                    # Add legend
                    plt.legend()

                    # Save the figure for this pair
                    filename = f"success_crashed_cycle_{xlabels[j]}_vs_{xlabels[i]}.png".replace(" ", "_")
                    
                    plt.savefig(f"{figure_savepath}/crashed_cycles/{filename}", dpi=300, bbox_inches='tight')
                    plt.close()

def plot_crashed_cycles_only_by_phase(X, xlabels, output_mask, figure_savepath, sims_folder, chamber, bcl, only_prints=False,print_pair_plots=True,print_separately=False):

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

    if not only_prints:
        if print_pair_plots:
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

        if print_separately:
            os.makedirs(f"{figure_savepath}/crashed_cycles_only_by_phase_{chamber}", exist_ok=True)
            t = tqdm.trange(num_variables, desc='Bar desc', leave=True,colour='#8B0000')
            print_message_outer = "Plotting crashes by phase depending on value of " 
            biggest_label = max([len(lab) for lab in xlabels])
            total_message_length_outer = len(print_message_outer) + biggest_label + 13
            for i in t:
                t.set_description(f"{print_message_outer}{xlabels[i]} in the {chamber}...".ljust(total_message_length_outer))
                t2 = tqdm.trange(i, desc='Bar desc', leave=True,colour='#FF5733')
                print_message = "..against " 
                total_message_length = len(print_message) + biggest_label
                for j in t2:
                    t2.set_description(f"{print_message}{xlabels[j]}".ljust(total_message_length))                    
                    plt.figure(figsize=(6, 5))
                    
                    x_var = X[:, j]
                    y_var = X[:, i]
                    
                    # Plot the data points with different colors based on the array_phase values
                    plt.scatter(
                        x_var[array_phase == 0],
                        y_var[array_phase == 0],
                        c=colors_by_phase["IVC"],
                        marker=marker_by_phase["IVC"],
                        label="IVC",
                        s=20
                    )
                    plt.scatter(
                        x_var[array_phase == 1],
                        y_var[array_phase == 1],
                        c=colors_by_phase["ejec"],
                        marker=marker_by_phase["ejec"],
                        label="ejec",
                        s=20
                    )
                    plt.scatter(
                        x_var[array_phase == 2],
                        y_var[array_phase == 2],
                        c=colors_by_phase["IVR"],
                        marker=marker_by_phase["IVR"],
                        label="IVR",
                        s=20
                    )
                    plt.scatter(
                        x_var[array_phase == 3],
                        y_var[array_phase == 3],
                        c=colors_by_phase["fill"],
                        marker=marker_by_phase["fill"],
                        label="fill",
                        s=20
                    )
                    plt.scatter(
                        x_var[array_phase == 4],
                        y_var[array_phase == 4],
                        c=colors_by_phase["load"],
                        marker=marker_by_phase["load"],
                        label="load",
                        s=20
                    )

                    # Set labels
                    plt.xlabel(xlabels[j])
                    plt.ylabel(xlabels[i])

                    # Add a title
                    plt.title(f"Cardiac phase: {chamber}", fontsize=16)

                    # Add legend
                    plt.legend()

                    # Save the figure for this pair
                    filename = f"cardiac_phase_{xlabels[j]}_vs_{xlabels[i]}.png".replace(" ", "_")
                    
                    plt.savefig(f"{figure_savepath}/crashed_cycles_only_by_phase_{chamber}/{filename}", dpi=300, bbox_inches='tight')
                    plt.close()

    for key, value in phases_indices.items():
        print(f"In the {chamber}, {np.count_nonzero([array_phase==value])} crashed during {key}")

def plot_crashed_cycles_only_by_time(X, xlabels, output_mask, figure_savepath, sims_folder, bcl,only_prints=False,print_pair_plots=True,print_separately=False):

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

    if not only_prints:

        if print_pair_plots:
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
        if print_separately:
            os.makedirs(f"{figure_savepath}/crashed_cycles_only_by_time", exist_ok=True)
            t = tqdm.trange(num_variables, desc='Bar desc', leave=True,colour='#8B0000')
            print_message_outer = "Plotting crashes by time depending on value of " 
            biggest_label = max([len(lab) for lab in xlabels])
            total_message_length_outer = len(print_message_outer) + biggest_label + 3
            for i in t:
                t.set_description(f"{print_message_outer}{xlabels[i]}...".ljust(total_message_length_outer))
                t2 = tqdm.trange(i, desc='Bar desc', leave=True,colour='#FF5733')
                print_message = "..against " 
                total_message_length = len(print_message) + biggest_label
                for j in t2:
                    t2.set_description(f"{print_message}{xlabels[j]}".ljust(total_message_length))
                    plt.figure(figsize=(6, 5))
                    
                    x_var = X[:, j]
                    y_var = X[:, i]
                    
                    # Plot the data points with different colors based on the array_time values
                    for beat_t in range(6):
                        plt.scatter(
                            x_var[array_time == beat_t],
                            y_var[array_time == beat_t],
                            c=colors_by_time[beat_t],
                            marker=marker_by_time[beat_t],
                            label=f"Beat {beat_t+1}" if beat_t < 5 else "Beat 5 (no ejection)",
                            s=20
                        )

                    # Set labels
                    plt.xlabel(xlabels[j])
                    plt.ylabel(xlabels[i])

                    # Add a title
                    plt.title(f"Heart beat crash analysis", fontsize=16)

                    # Add legend
                    plt.legend()

                    # Save the figure for this pair
                    filename = f"heartbeat_crashed_{xlabels[j]}_vs_{xlabels[i]}.png".replace(" ", "_")
                    
                    plt.savefig(f"{figure_savepath}/crashed_cycles_only_by_time/{filename}", dpi=300, bbox_inches='tight')
                    plt.close()

    print(f"{np.count_nonzero([array_time==0])} crashed during beat 1")
    print(f"{np.count_nonzero([array_time==1])} crashed during beat 2")
    print(f"{np.count_nonzero([array_time==2])} crashed during beat 3")
    print(f"{np.count_nonzero([array_time==3])} crashed during beat 4")
    print(f"{np.count_nonzero([array_time==5])} crashed during beat 5 before ejection")
    print(f"{np.count_nonzero([array_time==4])} crashed during beat 5 after ejection")


def main(args):

    data_folder        = f"{args.basefolder}/data"
    output_folder      = f"{args.basefolder}/output"
    figures_path       = f"{args.basefolder}/figures"
    sims_folder        = f"{args.basefolder}/simulations"
    first_simulation   = args.first_simulation
    last_simulation    = args.last_simulation
    n_beat             = args.n_beat
    only_prints        = args.only_prints
    print_pair_plots   = args.print_pair_plots
    print_separately   = args.print_separately

    os.makedirs(output_folder, exist_ok=True)

    X       = np.loadtxt(f"{data_folder}/X.txt")
    xlabels = np.loadtxt(f"{data_folder}/xlabels.txt", dtype=str)
    
    output_mask_cycle = np.loadtxt(f"{output_folder}/output_mask_beat_{n_beat}.txt")

    with open(f"{args.basefolder}/json_files/clinical_data.json", "r") as clinical_data:
            clinical_json = json.load(clinical_data)
    
    BCL = clinical_json["general"]["BCL"]

    
    print(f"Plotting crashed cycles...")
    plot_crashed_cycles(X               = X,
                        xlabels         = xlabels,
                        output_mask     = output_mask_cycle,
                        figure_savepath = figures_path,
                        only_prints = only_prints,
                        print_separately = print_separately,
                        print_pair_plots = print_pair_plots) 
    print(f"Plotting total crashed...")
    plot_total_crashed(X               = X,
                       xlabels         = xlabels,
                       output_mask     = output_mask_cycle,
                       figure_savepath = figures_path,
                       first_simulation= first_simulation,
                       last_simulation = last_simulation,
                       only_prints=only_prints,
                        print_separately = print_separately,
                        print_pair_plots = print_pair_plots) 
    print(f"PLotting crashed cycles only by phase in LV...")
    plot_crashed_cycles_only_by_phase(X               = X, 
                                      xlabels         = xlabels, 
                                      output_mask     = output_mask_cycle, 
                                      figure_savepath = figures_path,
                                      sims_folder     = sims_folder,
                                      chamber         = "LV",
                                      bcl             = BCL,
                                      only_prints=only_prints,
                        print_separately = print_separately,
                        print_pair_plots = print_pair_plots)
    print(f"PLotting crashed cycles only by phase in RV...")

    plot_crashed_cycles_only_by_phase(X               = X, 
                                      xlabels         = xlabels, 
                                      output_mask     = output_mask_cycle, 
                                      figure_savepath = figures_path,
                                      sims_folder     = sims_folder,
                                      chamber         = "RV",
                                      bcl             = BCL,
                                      only_prints=only_prints,
                        print_separately = print_separately,
                        print_pair_plots = print_pair_plots)
    print(f"PLotting crashed cycles only by phase in LA...")

    plot_crashed_cycles_only_by_phase(X               = X, 
                                      xlabels         = xlabels, 
                                      output_mask     = output_mask_cycle, 
                                      figure_savepath = figures_path,
                                      sims_folder     = sims_folder,
                                      chamber         = "LA",
                                      bcl             = BCL,
                                      only_prints=only_prints,
                        print_separately = print_separately,
                        print_pair_plots = print_pair_plots)
    print(f"PLotting crashed cycles only by phase in RA...")

    plot_crashed_cycles_only_by_phase(X               = X, 
                                      xlabels         = xlabels, 
                                      output_mask     = output_mask_cycle, 
                                      figure_savepath = figures_path,
                                      sims_folder     = sims_folder,
                                      chamber         = "RA",
                                      bcl             = BCL,
                                      only_prints=only_prints,
                        print_separately = print_separately,
                        print_pair_plots = print_pair_plots)
                        
    print(f"PLotting crashed cycles only by time...")

    plot_crashed_cycles_only_by_time(X               = X,
                                     xlabels         = xlabels, 
                                     output_mask     = output_mask_cycle, 
                                     figure_savepath = figures_path,
                                     sims_folder     = sims_folder,
                                     bcl             = BCL,
                                      only_prints=only_prints,
                        print_separately = print_separately,
                        print_pair_plots = print_pair_plots)
    

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    

    parser.add_argument('--basefolder', type=str, required=True,
                        default=os.path.join(os.environ.get("DATA_ROOT", ""), "simulations"))
    parser.add_argument('--first_simulation', type=int, required=True, default=0)
    parser.add_argument('--last_simulation', type=int, required=True, default=99)
    parser.add_argument('--n_beat', type=int, required=False, help="Heartbeat number to compute the output.", default=5)
    parser.add_argument('--only_prints',action='store_true')
    parser.add_argument('--print_separately',action='store_true')
    parser.add_argument('--print_pair_plots',action='store_true')
    args = parser.parse_args()

    main(args)
