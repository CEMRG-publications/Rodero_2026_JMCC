import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from common.utils import generate_gsa_ranking_files, read_xlabels_dict


# ------------------------
# Helper functions
# ------------------------
def create_blue_colormap(n_ranks):
    """Create a discrete colormap from white to blue for variability."""
    colors = ['white', '#0000ff']  # white to blue
    return LinearSegmentedColormap.from_list('white_blue', colors, N=n_ranks+1)


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
    title=None
):
    """
    Creates a heatmap of ranking variability across multiple scenarios.
    Value = max rank - min rank for each parameter/output feature.
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
    
    # Remove parameters irrelevant in ALL outputs
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
    
    # Get max variability
    max_var = int(np.nanmax(variability_data))
    
    # Create figure
    n_outputs = len(features_idx_list)
    n_params = len(relevant_params)
    fig_width = max(16, n_outputs * 1.5)
    fig_height = max(12, n_params * 0.8)
    _, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    cmap = create_blue_colormap(max_var)
    
    mask_low_importance = np.isnan(variability_data)
    
    hm = sns.heatmap(
        variability_data,
        xticklabels=output_labels,
        yticklabels=param_labels,
        cmap=cmap,
        vmin=0,
        vmax=max_var,
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

    cbar = hm.collections[0].colorbar
    # Keep ticks, but hide the tick *marks*
    cbar.ax.tick_params(size=0, labelsize=fontsize)  
    cbar.set_label('Ranking Variability', fontsize=fontsize)


    
    # Grey out irrelevant
    if np.any(mask_low_importance):
        for i in range(variability_data.shape[0]):
            for j in range(variability_data.shape[1]):
                if mask_low_importance[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                            facecolor='lightgray', edgecolor='black', linewidth=1))
    
    # Labels
    ax.set_xlabel('Output Features', fontsize=fontsize)
    if title is None:
        title = f"Ranking Variability Across {len(scenarios)} Scenarios"
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=20)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize-2)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize-2)
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
        title=f"Sensitivity Analysis Variability Across Anatomies"
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
