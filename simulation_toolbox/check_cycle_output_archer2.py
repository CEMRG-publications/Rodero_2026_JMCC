import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from SIMULATION_library import fourchamber_output

def plot_crashed_simulations(X, xlabels, output_mask, figure_savepath):

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
    plt.suptitle(f"Crashed simulations: {percentage_zeros:.2f}%", fontsize=16)

    # Move the legend to avoid overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add a common legend for all the subplots
    axes[0, -1].legend(loc=[1,0])

    plt.savefig(os.path.join(figure_savepath,"success_crashed.png"), dpi=300)

def main(args):

    simulations_folder = f"{args.basefolder}/simulations"
    unloaded_volumes   = f"{simulations_folder}/unloaded_volumes.txt"
    data_folder        = f"{args.basefolder}/data"
    output_folder      = f"{args.basefolder}/output"
    figures_path       = f"{args.basefolder}/figures"
    BCL = args.BCL

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
                             figure_savepath = figures_path) 
    
    fourchamber_output.plot_pvloops_all(datafolder    = output_folder,
                                        output_folder = simulations_folder,
                                        BCL           = BCL,
                                        basename      = "cycle_",
                                        mask_file     = f"{output_folder}/output_mask.txt",
                                        NBEATS        = 5,
                                        figname       = f"{figures_path}/all_pv_loops.png")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate a pdf and a txt file showing the simulations that worked. It also plots the simulations that crashed and the ones who didn't in the parameter space and plots all the pv loops for the ones that worked.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/h01/new_unloading/unloading_simulations")
    parser.add_argument('--first_simulation', type=int, required=True, default=0)
    parser.add_argument('--last_simulation', type=int, required=True, default=99)
    parser.add_argument('--BCL', type=int, required=True, default=1000)

    args = parser.parse_args()

    main(args)
