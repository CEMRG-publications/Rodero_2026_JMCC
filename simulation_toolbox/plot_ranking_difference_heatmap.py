import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from common.utils import generate_gsa_ranking_files, read_xlabels_dict


# ------------------------
# Helper functions
# ------------------------

def create_blue_red_colormap(min_diff, max_diff):
    """Create a discrete blue-white-red colormap with integer bins centered at 0."""
    
    # Define bin edges (one wider so each integer gets a cell)
    bounds = np.arange(min_diff - 0.5, max_diff + 1.5, 1)  
    
    n_bins = len(bounds) - 1
    
    # Base gradient (blue → white → red)
    base_cmap = LinearSegmentedColormap.from_list(
        "blue_white_red", ["#0000ff", "white", "#ff0000"], N=n_bins
    )
    
    # Discretize into integer steps
    colors = [base_cmap(i/(n_bins-1)) for i in range(n_bins)]
    cmap = ListedColormap(colors, name="blue_white_red_discrete")
    
    # BoundaryNorm aligns integers with tick positions
    norm = BoundaryNorm(boundaries=bounds, ncolors=n_bins)
    
    # Valid tick positions are integers only
    ticks = np.arange(min_diff, max_diff + 1, 1)
    
    return cmap, norm, ticks


def plot_ranking_difference_heatmap(
    baseline_scenario,
    modified_scenario,
    savepath,
    xlabels_dict,
    ylabels_raw_all,
    ylabels_latex_all,
    features_idx_list,
    fontsize=10,
    gsa_mode="Si_total",
    mode="max",
    figname="ranking_difference_heatmap.png",
    title=None,
    filter_by_baseline=False
):
    """
    Creates a heatmap of ranking difference across multiple scenarios.
    Value = modified - baseline for each parameter/output feature.
    Adds circles with diameters proportional to modified scenario ranking position.
    
    Parameters:
    -----------
    filter_by_baseline : bool
        If True, only include parameters that are relevant in the baseline scenario
        If False, include all parameters that are relevant in the modified scenario
    """
    
    # Collect all parameters from the overall ranking of the first scenario
    overall_rank_file = os.path.join(baseline_scenario, "output", f"Rank_{gsa_mode}_{mode}.txt")
    if not os.path.exists(overall_rank_file):
        print(f"ERROR: Overall rank file not found: {overall_rank_file}")
        return
    
    # print(f"Reading overall rank file: {overall_rank_file}")
    all_params = []
    with open(overall_rank_file, "r") as f:
        for line in f.readlines():
            param, _ = line.strip().split("\t")
            all_params.append(param)
    
    # Initialize difference matrix and modified rankings matrix
    difference_data = np.zeros((len(all_params), len(features_idx_list)))
    modified_rankings_data = np.full((len(all_params), len(features_idx_list)), np.nan)
    
    # Track relevance for both scenarios
    baseline_relevance = {param: [False] * len(features_idx_list) for param in all_params}
    modified_relevance = {param: [False] * len(features_idx_list) for param in all_params}
    
    # Track ranking changes for analysis
    ranking_changes = []  # Store (param, output, diff_value, baseline_rank, modified_rank)
    
    for j, feature_idx in enumerate(features_idx_list):
        ylabel_raw = ylabels_raw_all[feature_idx]
        ylabel_latex = ylabels_latex_all[feature_idx]
        
        # Process modified scenario
        modified_ranks = {}
        modified_scenario_path = os.path.join(modified_scenario, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
        
        if os.path.exists(modified_scenario_path):
            # print(f"Reading modified rank file: {modified_scenario_path}")
            with open(modified_scenario_path, "r") as f:
                for i, line in enumerate(f.readlines()):
                    param, value = line.strip().split("\t")
                    modified_ranks[param] = i + 1
                    effect = float(value)
                    if effect >= 0.05:  # relevant
                        modified_relevance[param][j] = True
        else:
            print(f"ERROR: Modified rank file not found: {modified_scenario_path}")

        # Process baseline scenario
        baseline_ranks = {}
        baseline_scenario_path = os.path.join(baseline_scenario, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
        
        if os.path.exists(baseline_scenario_path):
            # print(f"Reading baseline rank file: {baseline_scenario_path}")
            with open(baseline_scenario_path, "r") as f:
                for i, line in enumerate(f.readlines()):
                    param, value = line.strip().split("\t")
                    baseline_ranks[param] = i + 1
                    effect = float(value)
                    if effect >= 0.05:  # relevant
                        baseline_relevance[param][j] = True
        else:
            print(f"ERROR: Baseline rank file not found: {baseline_scenario_path}")
        
        # Compute differences and store modified rankings for each parameter
        for i, param in enumerate(all_params):
            if filter_by_baseline:
                # Only compute difference if parameter is relevant in baseline
                if baseline_relevance[param][j] and param in modified_ranks and param in baseline_ranks:
                    diff_value = - modified_ranks[param] + baseline_ranks[param]
                    difference_data[i, j] = diff_value
                    modified_rankings_data[i, j] = modified_ranks[param]
                    ranking_changes.append((param, ylabel_latex, diff_value, baseline_ranks[param], modified_ranks[param]))
                else:
                    difference_data[i, j] = np.nan
            else:
                # Compute difference if parameter is relevant in modified scenario
                if modified_relevance[param][j] and param in modified_ranks and param in baseline_ranks:
                    diff_value = - modified_ranks[param] + baseline_ranks[param]
                    difference_data[i, j] = diff_value
                    modified_rankings_data[i, j] = modified_ranks[param]
                    ranking_changes.append((param, ylabel_latex, diff_value, baseline_ranks[param], modified_ranks[param]))
                else:
                    difference_data[i, j] = np.nan
    
    # Analyze ranking changes
    if ranking_changes:
        # Find maximum increase (most positive difference)
        max_increase = max(ranking_changes, key=lambda x: x[2])
        max_increase_value = max_increase[2]
        max_increases = [(param, output, diff, baseline_r, modified_r) for param, output, diff, baseline_r, modified_r in ranking_changes if diff == max_increase_value]
        
        # Find maximum decrease (most negative difference)
        min_decrease = min(ranking_changes, key=lambda x: x[2])
        max_decrease_value = abs(min_decrease[2])
        max_decreases = [(param, output, diff, baseline_r, modified_r) for param, output, diff, baseline_r, modified_r in ranking_changes if diff == min_decrease[2]]
        
        # Find second largest decrease
        unique_decreases = sorted(list(set([diff for _, _, diff, _, _ in ranking_changes if diff < 0])))
        second_decrease_value = None
        second_decreases = []
        if len(unique_decreases) >= 2:
            second_decrease_value = abs(unique_decreases[1])  # Second most negative (second largest decrease)
            second_decreases = [(param, output, diff, baseline_r, modified_r) for param, output, diff, baseline_r, modified_r in ranking_changes if diff == unique_decreases[1]]
        
        # Find third largest decrease
        third_decrease_value = None
        third_decreases = []
        if len(unique_decreases) >= 3:
            third_decrease_value = abs(unique_decreases[2])  # Third most negative (third largest decrease)
            third_decreases = [(param, output, diff, baseline_r, modified_r) for param, output, diff, baseline_r, modified_r in ranking_changes if diff == unique_decreases[2]] 


        print(f"\nRANKING CHANGE ANALYSIS:")
        print(f"Maximum ranking increase: +{max_increase_value} positions")
        for param, output, diff, baseline_r, modified_r in max_increases:
            param_display = xlabels_dict.get(param, {}).get("latex", param)
            print(f"  Parameter: {param_display} | Output: {output} | {baseline_r} → {modified_r}")
        
        print(f"Maximum ranking decrease: -{max_decrease_value} positions")
        for param, output, diff, baseline_r, modified_r in max_decreases:
            param_display = xlabels_dict.get(param, {}).get("latex", param)
            print(f"  Parameter: {param_display} | Output: {output} | {baseline_r} → {modified_r}")
        
        if second_decrease_value is not None:
            print(f"Second largest ranking decrease: -{second_decrease_value} positions")
            for param, output, diff, baseline_r, modified_r in second_decreases:
                param_display = xlabels_dict.get(param, {}).get("latex", param)
                print(f"  Parameter: {param_display} | Output: {output} | {baseline_r} → {modified_r}")
        
        if third_decrease_value is not None:
            print(f"Third largest ranking decrease: -{third_decrease_value} positions")
            for param, output, diff, baseline_r, modified_r in third_decreases:
                param_display = xlabels_dict.get(param, {}).get("latex", param)
                print(f"  Parameter: {param_display} | Output: {output} | {baseline_r} → {modified_r}")
        
        print("\n\n\n\n")  # 5 empty lines at the end
    else:
        print("\nNo ranking changes found.")
        print("\n\n")  # 3 empty lines at the end
    
    # Remove parameters irrelevant in ALL outputs (based on filtering criterion)
    if filter_by_baseline:
        relevant_params_mask = np.any([any(baseline_relevance[param]) for param in all_params])
        relevant_params_mask = [any(baseline_relevance[param]) for param in all_params]
    else:
        relevant_params_mask = [any(modified_relevance[param]) for param in all_params]
    
    difference_data = difference_data[relevant_params_mask, :]
    modified_rankings_data = modified_rankings_data[relevant_params_mask, :]
    relevant_params = [all_params[i] for i, mask in enumerate(relevant_params_mask) if mask]
    
    if len(relevant_params) == 0:
        print("ERROR: No relevant parameters found across scenarios")
        return
    
    # Sort params alphabetically (latex if available)
    sort_idx = np.argsort([xlabels_dict.get(p, {}).get("latex", p) for p in relevant_params])
    relevant_params = [relevant_params[i] for i in sort_idx]
    difference_data = difference_data[sort_idx, :]
    modified_rankings_data = modified_rankings_data[sort_idx, :]
    
    # Create labels
    param_labels = [xlabels_dict.get(param, {}).get("latex", param) for param in relevant_params]
    output_labels = [ylabels_latex_all[idx] for idx in features_idx_list]
    
    # Get max difference for valid (non-NaN) values and make symmetric
    valid_data = difference_data[~np.isnan(difference_data)]
    if len(valid_data) > 0:
        max_abs_diff = int(np.max(np.abs(valid_data)))
        max_diff = max_abs_diff
        min_diff = -max_abs_diff
    else:
        max_diff = 0
        min_diff = 0
    
    # Create figure
    n_outputs = len(features_idx_list)
    n_params = len(relevant_params)
    fig_width = max(16, n_outputs * 1.5)
    fig_height = max(12, n_params * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create mask for parameters that are irrelevant (NaN values)
    mask_irrelevant = np.isnan(difference_data)
    
    # Create symmetric colormap
    cmap, norm, ticks = create_blue_red_colormap(min_diff=min(min_diff, -6), max_diff=max(max_diff, 6))

    # Create white background heatmap (no colors in tiles)
    hm = sns.heatmap(
        np.ones_like(difference_data),  # All white background
        xticklabels=output_labels,
        yticklabels=param_labels,
        cmap='Greys',
        vmin=0, vmax=1,
        cbar=False,  # No colorbar for background
        ax=ax,
        linewidths=1,
        linecolor='black',
        square=True
    )
    
    # Grey out irrelevant parameters
    for i in range(difference_data.shape[0]):
        for j in range(difference_data.shape[1]):
            if mask_irrelevant[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                        facecolor='lightgray', edgecolor='black', linewidth=1))
            else:
                # White background for relevant parameters
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                        facecolor='white', edgecolor='black', linewidth=1))
    
    # Add colored rectangles with ranking numbers for non-negligible tiles
    for i in range(difference_data.shape[0]):
        for j in range(difference_data.shape[1]):
            if not mask_irrelevant[i, j]:  # Only for non-negligible parameters
                ranking = modified_rankings_data[i, j]
                diff_value = difference_data[i, j]
                if not np.isnan(ranking) and not np.isnan(diff_value):
                    # Get color from colormap based on difference value
                    color = cmap(norm(diff_value))
                    
                    # Create colored rectangle for the entire cell
                    rect = plt.Rectangle((j, i), 1, 1, fill=True,
                                       facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    
                    # Add ranking number in the center of the cell with white text and black outline
                    center_x = j + 0.5
                    center_y = i + 0.5

                    # Outline text (white, thicker)
                    outline = ax.text(center_x, center_y, str(int(ranking)),
                                    ha='center', va='center',
                                    fontsize=fontsize, color='white',
                                    path_effects=[path_effects.Stroke(linewidth=6, foreground='white'),
                                                    path_effects.Normal()])

                    # Foreground text (black, sits on top, unchanged thickness)
                    txt = ax.text(center_x, center_y, str(int(ranking)),
                                ha='center', va='center',
                                fontsize=fontsize, color='black')

                    
                    
    
    # Add colorbar manually
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=1.0, ticks=ticks)
    cbar.ax.tick_params(size=0, labelsize=fontsize)
    cbar.set_label('Ranking difference compared to baseline', fontsize=fontsize)
    
    # Labels
    ax.set_xlabel('Output Features', fontsize=fontsize)
    if title is None:
        if filter_by_baseline:
            title = f"Ranking Difference (Baseline-Relevant Parameters Only)"
        else:
            title = f"Ranking Difference"
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=20)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize-2)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize-2)
    plt.tight_layout()
    
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    # print(f"Writing heatmap to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()


def generate_gsa_ranking_difference_heatmap(
    baseline_scenario,
    modified_scenario,
    xlabels_file,
    ylabels_file,
    savepath,
    ylabels_dict,
    xlabels_dict,
    fontsize=12,
    figname_prefix="",
    title=None
):
    """Main driver for difference heatmap - generates both versions."""
    
    if not os.path.isfile(f"{modified_scenario}/output/Si_total.csv") and os.path.isfile(f"{modified_scenario}/Si_total.csv"):
        # print(f"Copying Si_total.csv to output directory: {modified_scenario}/Si_total.csv → {modified_scenario}/output/Si_total.csv")
        os.system(f"cp {modified_scenario}/Si_total.csv {modified_scenario}/output/Si_total.csv")

    # print(f"Reading xlabels file: {xlabels_file}")
    # print(f"Reading ylabels file: {ylabels_file}")
    features_idx_list, ylabels_raw_all, ylabels_latex_all = generate_gsa_ranking_files(
        xlabels_file=xlabels_file,
        ylabels_file=ylabels_file,
        ylabels_dict=ylabels_dict,
        scenarios=[baseline_scenario, modified_scenario]
    )
    
    xlabels = np.loadtxt(xlabels_file, dtype=str)
    # print(f"Reading xlabels dictionary: {xlabels_dict}")
    _, xlabels_dict_all = read_xlabels_dict(xlabels_dict, xlabels)
    
    # Generate first heatmap: All parameters relevant in modified scenario
    # print("\nGenerating heatmap for all modified-relevant parameters...")
    plot_ranking_difference_heatmap(
        baseline_scenario=baseline_scenario,
        modified_scenario=modified_scenario,
        savepath=savepath,
        xlabels_dict=xlabels_dict_all,
        ylabels_raw_all=ylabels_raw_all,
        ylabels_latex_all=ylabels_latex_all,
        features_idx_list=features_idx_list,
        fontsize=fontsize,
        figname=f"{figname_prefix}_difference_heatmap_all_modified.png",
        title=f"{title}" if title else None,
        filter_by_baseline=False
    )

def main():
    parser = argparse.ArgumentParser(description="Plot GSA Ranking Difference Heatmaps")
    parser.add_argument('--baseline_scenario', required=True,
                        help='Path to the baseline scenario directory')
    parser.add_argument('--modified_scenario', required=True,
                        help='Path to the modified scenario directory')
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
    parser.add_argument('--title', type=str, default="Sensitivity Analysis Difference Compared to Baseline",
                        help='Title for the heatmap')
    
    args = parser.parse_args()
    
    generate_gsa_ranking_difference_heatmap(
        baseline_scenario=args.baseline_scenario,
        modified_scenario=args.modified_scenario,
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        savepath=args.savepath,
        fontsize=args.fontsize,
        figname_prefix=args.figname_prefix,
        ylabels_dict=args.ylabels_dict,
        xlabels_dict=args.xlabels_dict,
        title=args.title
    )


if __name__ == "__main__":
    main()