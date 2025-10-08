import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.patches import Rectangle

from common.utils import generate_gsa_ranking_files, read_xlabels_dict


def compute_median_ranks(scenarios, features_idx_list, ylabels_raw_all, 
                         gsa_mode="Si_total", mode="max"):
    """
    Compute the median rank for each parameter across all outputs and scenarios.
    
    Returns:
        dict: {param_name: median_rank}
    """
    # Get all parameters from first scenario
    overall_rank_file = os.path.join(scenarios[0], "output", f"Rank_{gsa_mode}_{mode}.txt")
    if not os.path.exists(overall_rank_file):
        print(f"Warning: Overall rank file not found: {overall_rank_file}")
        return {}
    
    all_params = []
    with open(overall_rank_file, "r") as f:
        for line in f.readlines():
            param, _ = line.strip().split("\t")
            all_params.append(param)
    
    # Collect all ranks for each parameter
    param_ranks = {param: [] for param in all_params}
    
    for scenario in scenarios:
        for feature_idx in features_idx_list:
            ylabel_raw = ylabels_raw_all[feature_idx]
            rank_file_path = os.path.join(scenario, "output", 
                                         f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
            
            if os.path.exists(rank_file_path):
                with open(rank_file_path, "r") as f:
                    for i, line in enumerate(f.readlines()):
                        param, value = line.strip().split("\t")
                        effect = float(value)
                        if effect >= 0.05:  # Only consider relevant parameters
                            param_ranks[param].append(i + 1)
    
    # Compute median rank for each parameter
    median_ranks = {}
    for param, ranks in param_ranks.items():
        if len(ranks) > 0:
            median_ranks[param] = np.median(ranks)
        else:
            median_ranks[param] = np.inf  # Parameters never relevant
    
    return median_ranks


def plot_output_comparison(
    scenarios,
    scenarios_names,
    savepath,
    xlabels_dict,
    ylabels_raw_all,
    ylabels_latex_all,
    features_idx_list,
    fontsize=10,
    gsa_mode="Si_total",
    mode="max",
    figname_prefix="output_comparison",
    n_top_params=15
):
    """
    Create comparison plots for each output showing sensitivity values across scenarios.
    Parameters are ordered by their median rank across all outputs/scenarios.
    """
    
    # Compute median ranks
    median_ranks = compute_median_ranks(scenarios, features_idx_list, ylabels_raw_all,
                                       gsa_mode, mode)
    
    # Sort parameters by median rank (best to worst)
    sorted_params = sorted(median_ranks.items(), key=lambda x: x[1])
    top_params = [param for param, rank in sorted_params[:n_top_params] if rank != np.inf]
    
    if len(top_params) == 0:
        print("Warning: No relevant parameters found")
        return
    
    print(f"Top {len(top_params)} parameters by median rank:")
    for i, param in enumerate(top_params):
        print(f"  {i+1}. {param} (median rank: {median_ranks[param]:.1f})")
    
    # Create parameter labels
    param_labels = [xlabels_dict.get(param, {}).get("latex", param) 
                   for param in top_params]
    
    # Create a plot for each output feature
    n_scenarios = len(scenarios)
    colors = plt.cm.Set2(np.linspace(0, 1, n_scenarios))
    
    for feature_idx in features_idx_list:
        ylabel_raw = ylabels_raw_all[feature_idx]
        ylabel_latex = ylabels_latex_all[feature_idx]
        
        # Collect sensitivity values for this output across scenarios
        sensitivity_data = np.zeros((len(top_params), n_scenarios))
        
        for j, scenario in enumerate(scenarios):
            rank_file_path = os.path.join(scenario, "output", 
                                         f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
            
            if os.path.exists(rank_file_path):
                param_effects = {}
                with open(rank_file_path, "r") as f:
                    for line in f.readlines():
                        param, value = line.strip().split("\t")
                        param_effects[param] = float(value)
                
                for i, param in enumerate(top_params):
                    if param in param_effects:
                        sensitivity_data[i, j] = param_effects[param]
                    else:
                        sensitivity_data[i, j] = np.nan
            else:
                sensitivity_data[:, j] = np.nan
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot bars for each scenario
        x = np.arange(len(top_params))
        width = 0.8 / n_scenarios
        
        for j in range(n_scenarios):
            offset = (j - n_scenarios/2 + 0.5) * width
            values = sensitivity_data[:, j]
            
            # Replace zeros and very small values with NaN for log scale
            values_plot = values.copy()
            values_plot[values_plot < 1e-10] = np.nan
            
            ax.bar(x + offset, values_plot, width, 
                  label=scenarios_names[j], color=colors[j], alpha=0.8)
        
        # Set log scale for y-axis
        ax.set_yscale('log')
        
        # Add horizontal line at 0.05 threshold
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, 
                  label='Relevance threshold (0.05)', zorder=0)
        
        # Formatting
        ax.set_xlabel('Parameters (ordered by median rank)', fontsize=fontsize)
        ax.set_ylabel(f'Sensitivity Index ({gsa_mode})', fontsize=fontsize)
        ax.set_title(f'Sensitivity Comparison: {ylabel_latex}', 
                    fontsize=fontsize+2, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, rotation=45, ha='right', fontsize=fontsize-2)
        ax.tick_params(axis='y', labelsize=fontsize-2)
        
        # Set y-axis limits
        min_val = np.nanmin(sensitivity_data[sensitivity_data > 0])
        max_val = np.nanmax(sensitivity_data)
        ax.set_ylim([max(min_val * 0.5, 1e-4), max_val * 2])
        
        ax.legend(fontsize=fontsize-2, loc='upper right')
        ax.grid(True, which='both', alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(savepath, exist_ok=True)
        output_path = os.path.join(savepath, 
                                  f"{figname_prefix}_{ylabel_raw}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Output comparison saved to: {output_path}")
    
    # Save parameter ranking summary
    ranking_df = pd.DataFrame([
        {
            'Parameter': param,
            'Parameter_Label': xlabels_dict.get(param, {}).get("latex", param),
            'Median_Rank': median_ranks[param],
            'Rank_Order': i + 1
        }
        for i, param in enumerate(top_params)
    ])
    
    summary_file = os.path.join(savepath, f"{figname_prefix}_parameter_ranking.csv")
    ranking_df.to_csv(summary_file, index=False, float_format='%.2f')
    print(f"Parameter ranking summary saved to: {summary_file}")


def generate_gsa_output_comparison(
    scenarios,
    scenarios_names,
    xlabels_file,
    ylabels_file,
    savepath,
    ylabels_dict,
    xlabels_dict,
    fontsize=12,
    figname_prefix="output_comparison",
    n_top_params=15
):
    """Main driver for output comparison plots."""
    
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
    
    # Generate comparison plots
    plot_output_comparison(
        scenarios=scenarios,
        scenarios_names=scenarios_names,
        savepath=savepath,
        xlabels_dict=xlabels_dict_all,
        ylabels_raw_all=ylabels_raw_all,
        ylabels_latex_all=ylabels_latex_all,
        features_idx_list=features_idx_list,
        fontsize=fontsize,
        figname_prefix=figname_prefix,
        n_top_params=n_top_params
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot GSA Output Comparison Across Scenarios"
    )
    parser.add_argument('--scenarios', nargs='+', required=True,
                        help='Paths to the scenario folders')
    parser.add_argument('--scenarios_names', nargs='+', required=True,
                        help='Names of the scenarios for labeling')
    parser.add_argument('--xlabels', required=True,
                        help='Path to the xlabels file')
    parser.add_argument('--ylabels', required=True,
                        help='Path to the ylabels file')
    parser.add_argument('--savepath', required=True,
                        help='Path to save the figures')
    parser.add_argument('--fontsize', type=int, default=12,
                        help='Font size for the plot')
    parser.add_argument('--figname_prefix', default='output_comparison',
                        help='Prefix for the figure names')
    parser.add_argument('--ylabels_dict', type=str, required=True,
                        help='Path to the ylabels dictionary file')
    parser.add_argument('--xlabels_dict', type=str, required=True,
                        help='Path to the xlabels dictionary file')
    parser.add_argument('--n_top_params', type=int, default=10,
                        help='Number of top parameters to show')
    
    args = parser.parse_args()
    
    generate_gsa_output_comparison(
        scenarios=args.scenarios,
        scenarios_names=args.scenarios_names,
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        savepath=args.savepath,
        fontsize=args.fontsize,
        figname_prefix=args.figname_prefix,
        ylabels_dict=args.ylabels_dict,
        xlabels_dict=args.xlabels_dict,
        n_top_params=args.n_top_params
    )


if __name__ == "__main__":
    main()