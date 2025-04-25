import os
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np

def read_labels(file_path):
    return np.loadtxt(file_path, dtype=str)

def plot_GSA_radar_chart(scenarios, xlabels, ylabels, savepath, fontsize, figname_preffix, legend=[], colors=[], threshold=0):
    for output_idx in range(len(ylabels)):
        ylabel_name = ylabels[output_idx]
        figname = f"{figname_preffix}_{ylabel_name}"

        theta = xlabels
        r = np.zeros((len(scenarios), len(theta)))

        for i, scenario in enumerate(scenarios):
            # Read Si_total.csv and convert to numpy array of floats
            with open(f"{scenario}/output/Si_total.csv", 'r') as f:
                csv_reader = csv.reader(f)
                ST_all = np.array([list(map(float, row)) for row in csv_reader])

            order_effects = ST_all[:, output_idx]
            order_effects = order_effects / np.sum(order_effects)

            r[i] = order_effects

        # Filter theta and r based on the threshold
        filtered_indices = [i for i in range(len(theta)) if np.any(r[:, i] >= threshold)]
        filtered_theta = [theta[i] for i in filtered_indices]
        filtered_r = r[:, filtered_indices]

        # Recompute theta radians to be equispaced
        theta_radians = np.linspace(0, 2 * np.pi, len(filtered_theta), endpoint=False).tolist()
        theta_radians += theta_radians[:1]

        filtered_r = np.concatenate((filtered_r, filtered_r[:, :1]), axis=1)

        if len(colors) == 0:
            colors = ['#e662ad', '#f7c26f', '#6ecb74', '#173263']

        # Create polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        if len(legend) == 0:
            legend = [f"Heart #{i+1}" for i in range(len(scenarios))]

        for i, scenario in enumerate(scenarios):
            ax.plot(theta_radians, filtered_r[i], marker='.', label=legend[i], color=colors[i])

        # Set theta ticks and labels
        ax.set_xticks(theta_radians[:-1])
        ax.set_xticklabels(filtered_theta, fontsize=fontsize)
        ax.xaxis.set_tick_params(which="major", pad=10)

        ax.set_rmax(1)

        title = f"{ylabels[output_idx]}"
        if title.startswith("$"):
            title = f"$\mathbf{{{title[1:-1]}}}$"

        plt.title(title, fontsize=fontsize+5, fontweight="bold", pad=30)
        plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        plt.tight_layout()

        os.makedirs(savepath, exist_ok=True)
        save_file_path = f"{savepath}/{figname}.png"
        plt.savefig(save_file_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to: {save_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot GSA Radar Chart")
    parser.add_argument('--scenarios', nargs='+', required=True, help='Paths to the scenario folders')
    parser.add_argument('--xlabels', required=True, help='Path to the xlabels file')
    parser.add_argument('--ylabels', required=True, help='Path to the ylabels file')
    parser.add_argument('--savepath', required=True, help='Path to save the figures')
    parser.add_argument('--fontsize', type=int, default=14, help='Font size for the plot')
    parser.add_argument('--figname_preffix', required=True, help='Prefix for the figure names')
    parser.add_argument('--legend', nargs='*', default=[], help='Legend labels')
    parser.add_argument('--colors', nargs='*', default=[], help='Colors for the plot')
    parser.add_argument('--threshold', type=float, default=0, help='Threshold for displaying xlabels')

    args = parser.parse_args()

    xlabels = read_labels(args.xlabels)
    ylabels = read_labels(args.ylabels)

    plot_GSA_radar_chart(
        scenarios=args.scenarios,
        xlabels=xlabels,
        ylabels=ylabels,
        savepath=args.savepath,
        fontsize=args.fontsize,
        figname_preffix=args.figname_preffix,
        legend=args.legend,
        colors=args.colors,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()