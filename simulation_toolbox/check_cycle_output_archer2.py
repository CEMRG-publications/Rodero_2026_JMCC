import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from SIMULATION_library import fourchamber_output

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

    # Calculate ejection fractions for all cases
    LVEF_all = 100 * (data[:, labels.index('LVedv')] - data[:, labels.index('LVesv')]) / data[:, labels.index('LVedv')]
    RVEF_all = 100 * (data[:, labels.index('RVedv')] - data[:, labels.index('RVesv')]) / data[:, labels.index('RVedv')]
    LAEF_all = 100 * (data[:, labels.index('LAedv')] - data[:, labels.index('LAesv')]) / data[:, labels.index('LAedv')]
    RAEF_all = 100 * (data[:, labels.index('RAedv')] - data[:, labels.index('RAesv')]) / data[:, labels.index('RAedv')]

    # Calculate mean and standard deviation of ejection fractions
    mean_LVEF = np.mean(LVEF_all)
    std_LVEF = np.std(LVEF_all)
    mean_RVEF = np.mean(RVEF_all)
    std_RVEF = np.std(RVEF_all)
    mean_LAEF = np.mean(LAEF_all)
    std_LAEF = np.std(LAEF_all)
    mean_RAEF = np.mean(RAEF_all)
    std_RAEF = np.std(RAEF_all)

    # Save results to a file
    with open(f'{basefolder}/data/output_statistics.txt', 'w') as file:
        file.write("Variable\tMean\tStd\n")
        for label, mean, std in zip(labels, mean_values, std_values):
            file.write(f"{label}\t{mean:.2f}\t{std:.2f}\n")

        # Add ejection fractions to the file
        file.write("\nEjection Fractions\n")
        file.write(f"LVEF\t{mean_LVEF:.2f}\t{std_LVEF:.2f}\n")
        file.write(f"RVEF\t{mean_RVEF:.2f}\t{std_RVEF:.2f}\n")
        file.write(f"LAEF\t{mean_LAEF:.2f}\t{std_LAEF:.2f}\n")
        file.write(f"RAEF\t{mean_RAEF:.2f}\t{std_RAEF:.2f}\n")

def main(args):

    simulations_folder = f"{args.basefolder}/simulations"
    unloaded_volumes   = f"{simulations_folder}/unloaded_volumes.txt"
    data_folder        = f"{args.basefolder}/data"
    output_folder      = f"{args.basefolder}/output"
    figures_path       = f"{args.basefolder}/figures"
    BCL = args.BCL
    first_simulation = args.first_simulation
    last_simulation = args.last_simulation

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.isfile(f"{data_folder}/X_mechanics.txt"):
        X       = np.loadtxt(f"{data_folder}/X.txt")
    else:
        X       = np.loadtxt(f"{data_folder}/X_mechanics.txt")

    if os.path.isfile(f"{data_folder}/xlabels.txt"):
        xlabels = np.loadtxt(f"{data_folder}/xlabels.txt", dtype=str)
    else:
        xlabels = np.loadtxt(f"{data_folder}/xlabels_mechanics.txt", dtype=str)

    fourchamber_output.cycle_simulation_summary(output_folder    = simulations_folder,
                                                BCL              = BCL,
                                                AVD              = 200*[0],
                                                NBEATS           = 5,
                                                unloaded_volumes = unloaded_volumes,
                                                start_sample     = args.first_simulation,
                                                last_sample      = args.last_simulation,
                                                basename         = "cycle_",
                                                maskoutput       = f"{output_folder}/output_mask.txt",
                                                output_file      = f"{output_folder}/simulation_summary.pdf"
                                            )
    
    fourchamber_output.cycle_output(datafolder    = output_folder,
                                    output_folder = simulations_folder,
                                    BCL           = BCL,
                                    AVD           = 100*[100],
                                    NBEATS        = 5,
                                    basename      = "cycle_",
                                    output_file   = f"{data_folder}/Y.txt",
                                    visualise     = False)
    
    output_mask = np.loadtxt(f"{output_folder}/output_mask.txt")

    
        
    plot_crashed_simulations(X               = X,
                             xlabels         = xlabels,
                             output_mask     = output_mask,
                             figure_savepath = figures_path,
                             first_simulation =  first_simulation,
                             last_simulation = last_simulation) 
    
    fourchamber_output.plot_pvloops_all(datafolder    = output_folder,
                                        output_folder = simulations_folder,
                                        BCL           = BCL,
                                        basename      = "cycle_",
                                        mask_file     = f"{output_folder}/output_mask.txt",
                                        NBEATS        = 5,
                                        figname       = f"{figures_path}/all_pv_loops.png")
    
    plot_statistics_file(basefolder=args.basefolder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate a pdf and a txt file showing the simulations that worked. It also plots the simulations that crashed and the ones who didn't in the parameter space and plots all the pv loops for the ones that worked.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/h01/new_unloading/unloading_simulations",
                        help='Path to the folder where the simulations, data, and figure folders are.')
    parser.add_argument('--first_simulation', type=int, required=True, default=0)
    parser.add_argument('--last_simulation', type=int, required=True, default=99)
    parser.add_argument('--BCL', type=int, required=False, default=1000)

    args = parser.parse_args()

    main(args)
