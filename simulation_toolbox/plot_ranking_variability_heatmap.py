import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import pandas as pd

from common.utils import generate_gsa_ranking_files, read_xlabels_dict


# ------------------------
# Helper functions
# ------------------------
def create_blue_colormap(n_ranks):
    """Create a discrete colormap from white to blue for variability."""
    colors = ['white', '#0000ff']  # white to blue
    return LinearSegmentedColormap.from_list('white_blue', colors, N=n_ranks+1)


import numpy as np
import pandas as pd
import os

def save_summary_statistics(variability_data, param_labels, output_labels, savepath, figname_prefix):
    """
    Save summary statistics for variability data.
    Creates two files: one for per-parameter stats and one for per-output stats.
    Also includes overall averages across all valid numbers.
    """
    
    # --- Per-parameter statistics ---
    param_stats = []
    for i, param in enumerate(param_labels):
        param_data = variability_data[i, :]
        valid_data = param_data[~np.isnan(param_data)]
        if len(valid_data) > 0:
            mean_var = np.mean(valid_data)
            median_var = np.median(valid_data)
            max_var = np.max(valid_data)
            param_stats.append({
                'Parameter': param,
                'Mean_Variability': mean_var,
                'Median_Variability': median_var,
                'Max_Variability': max_var,
                'N_Valid_Outputs': len(valid_data)
            })
    
    # Sort by mean variability (highest first)
    param_stats = sorted(param_stats, key=lambda x: x['Mean_Variability'], reverse=True)
    
    # --- Overall statistics across all valid values ---
    all_valid_values = variability_data[~np.isnan(variability_data)]
    overall_mean = np.mean(all_valid_values)
    overall_median = np.median(all_valid_values)
    overall_max = np.max(all_valid_values)
    
    # Add overall average row
    param_stats.append({
        'Parameter': 'Overall_Average',
        'Mean_Variability': overall_mean,
        'Median_Variability': overall_median,
        'Max_Variability': overall_max,
        'N_Valid_Outputs': len(all_valid_values)
    })
    
    # Save per-parameter statistics
    param_df = pd.DataFrame(param_stats)
    param_file = os.path.join(savepath, f"{figname_prefix}_parameter_variability_summary.csv")
    param_df.to_csv(param_file, index=False, float_format='%.3f')
    print(f"Parameter variability summary saved to: {param_file}")
    
    
    # --- Per-output statistics ---
    output_stats = []
    for j, output in enumerate(output_labels):
        output_data = variability_data[:, j]
        valid_data = output_data[~np.isnan(output_data)]
        if len(valid_data) > 0:
            mean_var = np.mean(valid_data)
            median_var = np.median(valid_data)
            max_var = np.max(valid_data)
            output_stats.append({
                'Output': output,
                'Mean_Variability': mean_var,
                'Median_Variability': median_var,
                'Max_Variability': max_var,
                'N_Valid_Parameters': len(valid_data)
            })
    
    # Sort by mean variability (highest first)
    output_stats = sorted(output_stats, key=lambda x: x['Mean_Variability'], reverse=True)
    
    # Add overall average row (same overall stats)
    output_stats.append({
        'Output': 'Overall_Average',
        'Mean_Variability': overall_mean,
        'Median_Variability': overall_median,
        'Max_Variability': overall_max,
        'N_Valid_Parameters': len(all_valid_values)
    })
    
    # Save per-output statistics
    output_df = pd.DataFrame(output_stats)
    output_file = os.path.join(savepath, f"{figname_prefix}_output_variability_summary.csv")
    output_df.to_csv(output_file, index=False, float_format='%.3f')
    print(f"Output variability summary saved to: {output_file}")
    
    return param_df, output_df


def plot_ranking_variability_heatmap(
    scenarios,
    savepath,
    xlabels_dict,
    ylabels_raw_all,
    ylabels_latex_all,
    features_idx_list,
    fontsize=10,
    gsa_mode="Si_total",
    mode="max",
    figname="ranking_variability_heatmap.png",
    title=None,
    figname_prefix=""
):
    """
    Creates a heatmap of ranking variability across multiple scenarios.
    Value = max rank - min rank for each parameter/output feature.
    Zeros are included in statistics but rows/columns with only zeros/NaNs
    are removed only for the plot.
    """

    # Collect all parameters from the overall ranking of the first scenario
    overall_rank_file = os.path.join(scenarios[0], "output", f"Rank_{gsa_mode}_{mode}.txt")
    if not os.path.exists(overall_rank_file):
        print(f"Warning: Overall rank file not found: {overall_rank_file}")
        return

    all_params = []
    with open(overall_rank_file, "r") as f:
        for line in f.readlines():
            param, _ = line.strip().split("\t")
            all_params.append(param)

    # Initialize variability matrix
    variability_data = np.zeros((len(all_params), len(features_idx_list)))

    for j, feature_idx in enumerate(features_idx_list):
        ylabel_raw = ylabels_raw_all[feature_idx]

        # Collect ranks across scenarios
        ranks_across_scenarios = {param: [] for param in all_params}
        relevant_mask = {param: False for param in all_params}

        for scenario in scenarios:
            rank_file_path = os.path.join(scenario, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")

            if os.path.exists(rank_file_path):
                param_ranks = {}
                param_effects = {}

                with open(rank_file_path, "r") as f:
                    for i, line in enumerate(f.readlines()):
                        param, value = line.strip().split("\t")
                        param_ranks[param] = i + 1
                        param_effects[param] = float(value)

                for param in all_params:
                    if param in param_ranks:
                        effect = param_effects[param]
                        if effect >= 0.05:  # relevant
                            relevant_mask[param] = True
                            ranks_across_scenarios[param].append(param_ranks[param])
            else:
                print(f"Warning: Rank file not found: {rank_file_path}")

        # Compute variability for each parameter
        for i, param in enumerate(all_params):
            if relevant_mask[param]:
                ranks = ranks_across_scenarios[param]
                if len(ranks) > 0:
                    variability_data[i, j] = max(ranks) - min(ranks)
                else:
                    variability_data[i, j] = np.nan
            else:
                variability_data[i, j] = np.nan

    # Remove parameters irrelevant in ALL outputs (all NaNs)
    relevant_params_mask = ~np.all(np.isnan(variability_data), axis=1)
    variability_data = variability_data[relevant_params_mask, :]
    relevant_params = [all_params[i] for i, mask in enumerate(relevant_params_mask) if mask]

    if len(relevant_params) == 0:
        print("Warning: No relevant parameters found across scenarios")
        return

    # Sort params alphabetically (latex if available)
    sort_idx = np.argsort([xlabels_dict.get(p, {}).get("latex", p) for p in relevant_params])
    relevant_params = [relevant_params[i] for i in sort_idx]
    variability_data = variability_data[sort_idx, :]

    # Create labels
    param_labels = [xlabels_dict.get(param, {}).get("latex", param) for param in relevant_params]
    output_labels = [ylabels_latex_all[idx] for idx in features_idx_list]

    # --- Save summary statistics (includes zeros) ---
    param_df, output_df = save_summary_statistics(
        variability_data, param_labels, output_labels, savepath, figname_prefix
    )

    # --- Filter only for plotting ---
    def has_nonzero_valid(arr):
        return np.any(~np.isnan(arr) & (arr != 0))

    row_mask = np.apply_along_axis(has_nonzero_valid, 1, variability_data)
    col_mask = np.apply_along_axis(has_nonzero_valid, 0, variability_data)

    plot_data = variability_data[row_mask][:, col_mask]
    plot_param_labels = [param_labels[i] for i, keep in enumerate(row_mask) if keep]
    plot_output_labels = [output_labels[i] for i, keep in enumerate(col_mask) if keep]

    if plot_data.size == 0:
        print("Warning: No non-zero relevant data found for plotting.")
        return

    # --- Continue with plotting ---
    max_var = int(np.nanmax(plot_data))
    n_outputs = len(plot_output_labels)
    n_params = len(plot_param_labels)
    fig_width = max(16, n_outputs * 1.5)
    fig_height = max(12, n_params * 0.8)
    _, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = create_blue_colormap(max_var)
    mask_low_importance = np.isnan(plot_data)

    hm = sns.heatmap(
        plot_data,
        xticklabels=plot_output_labels,
        yticklabels=plot_param_labels,
        cmap=cmap,
        vmin=0,
        vmax=max_var,
        cbar_kws={
            'label': 'Ranking position',
            'shrink': 1.0
        },
        ax=ax,
        mask=mask_low_importance,
        linewidths=1,
        linecolor='black',
        square=True,
        cbar=True
    )

    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(size=0, labelsize=fontsize)
    cbar.set_label('Ranking Variability', fontsize=fontsize)

    # Grey out irrelevant cells
    if np.any(mask_low_importance):
        for i in range(plot_data.shape[0]):
            for j in range(plot_data.shape[1]):
                if mask_low_importance[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                               facecolor='lightgray', edgecolor='black', linewidth=1))

    # Add text annotations
    n_rows, n_cols = plot_data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = plot_data[i, j]
            if not np.isnan(val):
                txt = ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{int(round(val))}",
                    ha="center",
                    va="center",
                    fontsize=fontsize - 2,
                    color="black",
                )
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=6, foreground="white"),
                    path_effects.Normal()
                ])

    # Labels
    ax.set_xlabel('Output Features', fontsize=fontsize)
    if title is None:
        title = f"Ranking Variability Across {len(scenarios)} Scenarios"
    ax.set_title(title, fontsize=fontsize + 2, fontweight='bold', pad=20)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize - 2)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize - 2)
    plt.tight_layout()

    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Ranking variability heatmap saved to: {output_path}")


def generate_gsa_ranking_variability_heatmap(
    scenarios,
    xlabels_file,
    ylabels_file,
    savepath,
    ylabels_dict,
    xlabels_dict,
    fontsize=12,
    figname_prefix="",
):
    """Main driver for variability heatmap."""
    features_idx_list, ylabels_raw_all, ylabels_latex_all = generate_gsa_ranking_files(
        xlabels_file=xlabels_file,
        ylabels_file=ylabels_file,
        ylabels_dict=ylabels_dict,
        scenarios=scenarios
    )
    
    xlabels = np.loadtxt(xlabels_file, dtype=str)
    _, xlabels_dict_all = read_xlabels_dict(xlabels_dict, xlabels)
    
    plot_ranking_variability_heatmap(
        scenarios=scenarios,
        savepath=savepath,
        xlabels_dict=xlabels_dict_all,
        ylabels_raw_all=ylabels_raw_all,
        ylabels_latex_all=ylabels_latex_all,
        features_idx_list=features_idx_list,
        fontsize=fontsize,
        figname=f"{figname_prefix}_variability_heatmap.png",
        title=f"Sensitivity Analysis Variability Across Anatomies",
        figname_prefix=figname_prefix
    )


def main():
    parser = argparse.ArgumentParser(description="Plot GSA Ranking Variability Heatmaps")
    parser.add_argument('--scenarios', nargs='+', required=True,
                        help='Paths to the scenario folders')
    parser.add_argument('--xlabels', required=True,
                        help='Path to the xlabels file')
    parser.add_argument('--ylabels', required=True,
                        help='Path to the ylabels file')
    parser.add_argument('--savepath', required=True,
                        help='Path to save the figures')
    parser.add_argument('--fontsize', type=int, default=12,
                        help='Font size for the plot')
    parser.add_argument('--figname_prefix', required=True,
                        help='Prefix for the figure names')
    parser.add_argument('--ylabels_dict', type=str, required=True,
                        help='Path to the ylabels dictionary file')
    parser.add_argument('--xlabels_dict', type=str, required=True,
                        help='Path to the xlabels dictionary file')
    
    args = parser.parse_args()
    
    generate_gsa_ranking_variability_heatmap(
        scenarios=args.scenarios,
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        savepath=args.savepath,
        fontsize=args.fontsize,
        figname_prefix=args.figname_prefix,
        ylabels_dict=args.ylabels_dict,
        xlabels_dict=args.xlabels_dict
    )


if __name__ == "__main__":
    main()