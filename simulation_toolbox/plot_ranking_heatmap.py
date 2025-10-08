import argparse
from collections import defaultdict
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
def create_custom_colormap(n_ranks):
    """Create a discrete red to white colormap for rankings (darker = better rank)."""
    colors = ['white','#ff0000'] 
    return LinearSegmentedColormap.from_list('red_white', colors, N=n_ranks)


def save_summary_statistics(heatmap_data, param_labels, output_labels, savepath, figname_prefix):
    """
    Save summary statistics for ranking data.
    Creates two files: one for per-parameter stats and one for per-output stats.
    Also includes overall averages across all valid numbers.
    """
    
    # --- Per-parameter statistics ---
    param_stats = []
    for i, param in enumerate(param_labels):
        param_data = heatmap_data[i, :]
        valid_data = param_data[~np.isnan(param_data)]
        if len(valid_data) > 0:
            mean_var = np.mean(valid_data)
            median_var = np.median(valid_data)
            max_var = np.max(valid_data)
            param_stats.append({
                'Parameter': param,
                'Mean_ranking': mean_var,
                'Median_ranking': median_var,
                'Max_ranking': max_var,
                'N_Valid_Outputs': len(valid_data)
            })
    
    # Sort by mean ranking (highest first)
    param_stats = sorted(param_stats, key=lambda x: x['Mean_ranking'], reverse=False)
    
    # --- Overall statistics across all valid values ---
    all_valid_values = heatmap_data[~np.isnan(heatmap_data)]
    overall_mean = np.mean(all_valid_values)
    overall_median = np.median(all_valid_values)
    overall_max = np.max(all_valid_values)
    
    # Add overall average row
    param_stats.append({
        'Parameter': 'Overall_Average',
        'Mean_ranking': overall_mean,
        'Median_ranking': overall_median,
        'Max_ranking': overall_max,
        'N_Valid_Outputs': len(all_valid_values)
    })
    
    # Save per-parameter statistics
    param_df = pd.DataFrame(param_stats)
    param_file = os.path.join(savepath, f"{figname_prefix}_parameter_ranking_summary.csv")
    param_df.to_csv(param_file, index=False, float_format='%.3f')
    print(f"Parameter ranking summary saved to: {param_file}")
    
    
    # --- Per-output statistics ---
    output_stats = []
    for j, output in enumerate(output_labels):
        output_data = heatmap_data[:, j]
        valid_data = output_data[~np.isnan(output_data)]
        if len(valid_data) > 0:
            mean_var = np.mean(valid_data)
            median_var = np.median(valid_data)
            max_var = np.max(valid_data)
            output_stats.append({
                'Output': output,
                'Mean_ranking': mean_var,
                'Median_ranking': median_var,
                'Max_ranking': max_var,
                'N_Valid_Parameters': len(valid_data)
            })
    
    # Sort by mean ranking (highest first)
    output_stats = sorted(output_stats, key=lambda x: x['Mean_ranking'], reverse=False)
    
    # Add overall average row (same overall stats)
    output_stats.append({
        'Output': 'Overall_Average',
        'Mean_ranking': overall_mean,
        'Median_ranking': overall_median,
        'Max_ranking': overall_max,
        'N_Valid_Parameters': len(all_valid_values)
    })
    
    # Save per-output statistics
    output_df = pd.DataFrame(output_stats)
    output_file = os.path.join(savepath, f"{figname_prefix}_output_ranking_summary.csv")
    output_df.to_csv(output_file, index=False, float_format='%.3f')
    print(f"Output ranking summary saved to: {output_file}")
    
    return param_df, output_df



def plot_ranking_heatmap_all_outputs(
    scenario,
    savepath,
    xlabels_dict,
    ylabels_dict,
    ylabels_raw_all,
    ylabels_latex_all,
    features_idx_list,
    fontsize=10,
    gsa_mode="Si_total",
    mode="max",
    figname="ranking_heatmap_all.png",
    title=None
):
    """
    Creates a comprehensive heatmap with all output features as columns.
    """
    
    scenario_name = scenario.rstrip("/").split("/")[-3]
    
    # Read overall ranking file to get parameter list
    overall_rank_file = os.path.join(scenario, "output", f"Rank_{gsa_mode}_{mode}.txt")
    
    if not os.path.exists(overall_rank_file):
        print(f"Warning: Overall rank file not found: {overall_rank_file}")
        return
    
    # Get all parameters from overall ranking
    all_params = []
    with open(overall_rank_file, "r") as f:
        for line in f.readlines():
            param, _ = line.strip().split("\t")
            all_params.append(param)
    
    # Initialize data matrix
    heatmap_data = np.zeros((len(all_params), len(features_idx_list)))
    
    # Fill data matrix
    for j, feature_idx in enumerate(features_idx_list):
        ylabel_raw = ylabels_raw_all[feature_idx]
        rank_file_path = os.path.join(scenario, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
        
        if os.path.exists(rank_file_path):
            param_ranks = {}
            param_effects = {}
            
            with open(rank_file_path, "r") as f:
                for i, line in enumerate(f.readlines()):
                    param, value = line.strip().split("\t")
                    param_ranks[param] = i + 1
                    param_effects[param] = float(value)
            
            # Fill column j
            for i, param in enumerate(all_params):
                if param in param_ranks:
                    effect = param_effects[param]
                    if effect < 0.05:  # Low importance
                        heatmap_data[i, j] = np.nan
                    else:
                        # Use actual ranking position (1 = best rank)
                        rank = param_ranks[param]
                        heatmap_data[i, j] = rank
                else:
                    heatmap_data[i, j] = np.nan
        else:
            print(f"Warning: Rank file not found: {rank_file_path}")
            heatmap_data[:, j] = np.nan
    
    # Remove parameters that are always not relevant (whole row is NaN/grey)
    relevant_params_mask = ~np.all(np.isnan(heatmap_data), axis=1)
    heatmap_data = heatmap_data[relevant_params_mask, :]
    relevant_params = [all_params[i] for i, mask in enumerate(relevant_params_mask) if mask]
    
    if len(relevant_params) == 0:
        print(f"Warning: No relevant parameters found for scenario {scenario_name}")
        return

    sort_idx = np.argsort([xlabels_dict.get(p, {}).get("latex", p) for p in relevant_params])
    relevant_params = [relevant_params[i] for i in sort_idx]
    heatmap_data = heatmap_data[sort_idx, :]
    
    # Create labels
    param_labels = [xlabels_dict.get(param, {}).get("latex", param) for param in relevant_params]
    output_labels = [ylabels_latex_all[idx] for idx in features_idx_list]
    
    # Get the maximum rank for discrete colormap
    max_rank = int(np.nanmax(heatmap_data))
    
    # Create figure with square aspect ratio - make it much larger
    n_outputs = len(features_idx_list)
    n_params = len(relevant_params)
    
    # Calculate figure size based on number of elements, with minimum size
    fig_width = max(16, n_outputs * 1.5)
    fig_height = max(12, n_params * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create discrete colormap
    cmap = create_custom_colormap(max_rank)
    cmap = cmap.reversed()  # Reverse for correct ranking (higher = better rank)
    
    # Create mask for low importance values
    mask_low_importance = np.isnan(heatmap_data)
    
    # Plot heatmap with square tiles and edges
    hm = sns.heatmap(
        heatmap_data,
        xticklabels=output_labels,
        yticklabels=param_labels,
        cmap=cmap,
        vmin=1,
        vmax=max_rank,
        cbar_kws={
            'label': 'Ranking position',
            'shrink': 1.0  # Make colorbar same height as heatmap
        },
        ax=ax,
        mask=mask_low_importance,
        linewidths=1,
        linecolor='black',
        square=True,
        cbar=True
    )
    
    # Customize colorbar - remove ticks but keep labels, invert order, and set fontsize
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([])  # Remove tick marks
    
    # Set tick labels at the center of each color segment, inverted order (1 at top)
    tick_positions = np.arange(1, max_rank + 1)
    cbar.set_ticks(tick_positions)
    # Reverse the tick labels so rank 1 appears at top
    # cbar.set_ticklabels(reversed(tick_positions))
    cbar.ax.tick_params(size=0, labelsize=fontsize)  # Remove tick marks, set fontsize
    
    # Set colorbar label fontsize
    cbar.set_label('Ranking position', fontsize=fontsize)
    
    # Color low importance parameters grey
    if np.any(mask_low_importance):
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                if mask_low_importance[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, 
                                             facecolor='lightgray', edgecolor='black', linewidth=1))
    

        # Add rank numbers as white text with a thin black outline
    n_rows, n_cols = heatmap_data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                txt = ax.text(
                    j + 0.5,              # heatmap cells are unit squares starting at integer coords
                    i + 0.5,
                    f"{int(round(val))}", # show rank as an integer
                    ha="center",
                    va="center",
                    fontsize=fontsize-2,
                    color="white",
                )
                # thin black outline
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=1, foreground="black"),
                    path_effects.Normal()
                ])


    # Formatting
    ax.set_xlabel('Output Features', fontsize=fontsize)
    # ax.set_ylabel('Parameters', fontsize=fontsize)
    
    if title is None:
        title = f"Sensitivity analysis for scenario {scenario_name}"
    
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize-2)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize-2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Comprehensive ranking heatmap saved to: {output_path}")


    # Save summary statistics for this scenario
    save_summary_statistics(heatmap_data = heatmap_data, 
    param_labels = param_labels, 
    output_labels= output_labels, 
    savepath = savepath, 
    figname_prefix = f"baseline_scenario_{figname[-5]}")


def generate_gsa_ranking_heatmaps(
    scenarios,
    scenarios_names,
    xlabels_file,
    ylabels_file,
    savepath,
    ylabels_dict,
    xlabels_dict,
    fontsize=12,
    figname_prefix="",
):
    """
    Main function to generate ranking heatmaps for all scenarios.
    Only generates comprehensive heatmaps with all outputs.
    """
    
    # Generate ranking files and get feature information
    features_idx_list, ylabels_raw_all, ylabels_latex_all = generate_gsa_ranking_files(
        xlabels_file=xlabels_file,
        ylabels_file=ylabels_file,
        ylabels_dict=ylabels_dict,
        scenarios=scenarios
    )
    
    # Read xlabels dictionary
    xlabels = np.loadtxt(xlabels_file, dtype=str)
    _, xlabels_dict_all = read_xlabels_dict(xlabels_dict, xlabels)
    
    # Generate comprehensive heatmaps for each scenario
    for i,scenario in enumerate(scenarios):
        scenario_name = scenarios_names[i]
        
        # Generate only comprehensive heatmap with all outputs
        plot_ranking_heatmap_all_outputs(
            scenario=scenario,
            savepath=savepath,
            xlabels_dict=xlabels_dict_all,
            ylabels_dict=ylabels_dict,
            ylabels_raw_all=ylabels_raw_all,
            ylabels_latex_all=ylabels_latex_all,
            features_idx_list=features_idx_list,
            fontsize=fontsize,
            figname=f"{figname_prefix}_heatmap_{i}.png",
            title=f"Sensitivity analysis for {scenario_name}"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot GSA Ranking Heatmaps")
    parser.add_argument('--scenarios', nargs='+', required=True, 
                       help='Paths to the scenario folders')
    parser.add_argument('--scenarios_names', nargs='+', required=True, 
                       help='Names of the scenarios for labeling')
    # xlabels and ylabels files
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

    

    
    generate_gsa_ranking_heatmaps(
        scenarios=args.scenarios,
        scenarios_names=args.scenarios_names,
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