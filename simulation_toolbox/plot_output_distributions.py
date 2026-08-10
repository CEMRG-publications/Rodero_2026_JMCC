import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

def plot_biomarker_variation(basefolder, high_range_threshold=400):
    # Load data
    Y = np.loadtxt(f"{basefolder}/data/Y.txt", dtype=float)
    ylabels = np.loadtxt(f"{basefolder}/data/ylabels.txt", dtype=str)

    # Calculate the variation for each biomarker
    variations = (np.max(Y, axis=0) - np.min(Y, axis=0)) / np.mean(Y, axis=0)
    abs_variations = np.abs(variations)
    variances = np.var(Y, axis=0)

    # Sort variations, variances, and ylabels
    sorted_indices = np.argsort(abs_variations)
    variations = variations[sorted_indices]
    abs_variations = abs_variations[sorted_indices]
    variances = variances[sorted_indices]
    ylabels = ylabels[sorted_indices]

    # Split indices based on high_range_threshold
    high_range_indices = np.where(np.max(Y, axis=0) - np.min(Y, axis=0) > high_range_threshold)[0]
    low_range_indices = np.where(np.max(Y, axis=0) - np.min(Y, axis=0) <= high_range_threshold)[0]

    def create_plot():
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(3, 2, figure=fig)

        # Plot absolute variations for low range
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(ylabels[low_range_indices], abs_variations[low_range_indices], 'o', color='#8B0000')
        ax1.axhline(y=0.2, color='gray', linestyle='--')
        ax1.set_xlabel('Biomarkers')
        ax1.set_ylabel('|(max - min) / mean|')
        ax1.set_title('Absolute Biomarker Variation (Low Range)')
        ax1.set_xticks(range(len(low_range_indices)))
        ax1.set_xticklabels(ylabels[low_range_indices], rotation=90)

        # Plot variances for low range
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(ylabels[low_range_indices], variances[low_range_indices], 'o', color='#8B0000')
        ax2.axhline(y=0.2, color='gray', linestyle='--')
        ax2.set_xlabel('Biomarkers')
        ax2.set_ylabel('Variance')
        ax2.set_title('Biomarker Variance (Low Range)')
        ax2.set_xticks(range(len(low_range_indices)))
        ax2.set_xticklabels(ylabels[low_range_indices], rotation=90)

        # Plot violin plots for all biomarkers
        ax3 = fig.add_subplot(gs[1, :])
        for i, ylabel in enumerate(ylabels):
            y_train = np.loadtxt(f"{basefolder}/output/emulators/{ylabel}/y_train.txt", dtype=float)
            y_test = np.loadtxt(f"{basefolder}/output/emulators/{ylabel}/y_test.txt", dtype=float)
            parts = ax3.violinplot([y_train - np.mean(y_train), y_test - np.mean(y_test)], positions=[i - 0.2, i + 0.2], showmeans=True, showmedians=False, showextrema=False)
            ax3.scatter(np.random.normal(i - 0.2, 0.02, size=y_train.shape), y_train - np.mean(y_train), color='#1f77b4', s=10, alpha=0.5)
            ax3.scatter(np.random.normal(i + 0.2, 0.02, size=y_test.shape), y_test - np.mean(y_test), color='#ff7f0e', s=10, alpha=0.5)
            for j, pc in enumerate(parts['bodies']):
                if j % 2 == 0:  # Left violin plot
                    pc.set_facecolor('#1f77b4')
                else:  # Right violin plot
                    pc.set_facecolor('#ff7f0e')
                pc.set_edgecolor('black')
                pc.set_alpha(0.5)
            parts['cmeans'].set_color('black')

        ax3.set_xlabel('Biomarkers')
        ax3.set_ylabel('Values')
        ax3.set_title('Biomarker Values Distribution')
        ax3.set_xticks(range(len(ylabels)))
        ax3.set_xticklabels(ylabels, rotation=90)

        # Add legend
        training_patch = plt.Line2D([0], [0], color='#1f77b4', lw=4, label='Training Set')
        test_patch = plt.Line2D([0], [0], color='#ff7f0e', lw=4, label='Test Set')
        ax3.legend(handles=[training_patch, test_patch], loc='upper right')

        # Plot violin plots for high range biomarkers
        ax4 = fig.add_subplot(gs[2, :])
        high_range_violin_indices = []
        for i, ylabel in enumerate(ylabels):
            y_train = np.loadtxt(f"{basefolder}/output/emulators/{ylabel}/y_train.txt", dtype=float)
            if np.max(y_train) - np.min(y_train) > high_range_threshold:
                high_range_violin_indices.append(i)

        for i in range(len(high_range_violin_indices)):
            ylabel = ylabels[high_range_violin_indices[i]]
            y_train = np.loadtxt(f"{basefolder}/output/emulators/{ylabel}/y_train.txt", dtype=float)
            y_test = np.loadtxt(f"{basefolder}/output/emulators/{ylabel}/y_test.txt", dtype=float)
            parts = ax4.violinplot([y_train - np.mean(y_train), y_test - np.mean(y_test)], positions=[i - 0.2, i + 0.2], showmeans=True, showmedians=False, showextrema=False)
            ax4.scatter(np.random.normal(i - 0.2, 0.02, size=y_train.shape), y_train - np.mean(y_train), color='#1f77b4', s=10, alpha=0.5)
            ax4.scatter(np.random.normal(i + 0.2, 0.02, size=y_test.shape), y_test - np.mean(y_test), color='#ff7f0e', s=10, alpha=0.5)
            for j, pc in enumerate(parts['bodies']):
                if j % 2 == 0:  # Left violin plot
                    pc.set_facecolor('#1f77b4')
                else:  # Right violin plot
                    pc.set_facecolor('#ff7f0e')
                pc.set_edgecolor('black')
                pc.set_alpha(0.5)
            parts['cmeans'].set_color('black')

        ax4.set_xlabel('Biomarkers')
        ax4.set_ylabel('Values')
        ax4.set_title('Biomarker Values Distribution (High Range)')
        ax4.set_xticks(range(len(high_range_violin_indices)))
        ax4.set_xticklabels([ylabels[i] for i in high_range_violin_indices], rotation=90)

        # Add legend
        ax4.legend(handles=[training_patch, test_patch], loc='upper right')

        plt.tight_layout()

        # Save the plot
        os.makedirs(f"{basefolder}/figures/biomarkers", exist_ok=True)
        plt.savefig(f"{basefolder}/figures/biomarkers/biomarker_variation.png", dpi=300)
        plt.close()

    # Create the combined plot
    create_plot()

def main(args):
    basefolder = args.basefolder

    # Call the new function to plot biomarker variation
    plot_biomarker_variation(basefolder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default=os.path.join(os.environ.get("DATA_ROOT", ""), "simulations"),
                        help='Path to the folder where the simulations, data, and figure folders are.')

    args = parser.parse_args()

    main(args)
