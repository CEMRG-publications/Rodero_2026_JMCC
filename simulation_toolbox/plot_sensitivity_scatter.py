import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def load_xlabels_dict(xlabels_dict_file):
    """Load the xlabels dictionary with parameter metadata and functional groups."""
    with open(xlabels_dict_file, 'r') as f:
        xlabels_dict = json.load(f)
    return xlabels_dict


def load_ylabels_dict(ylabels_dict_file):
    """Load the ylabels dictionary with output metadata."""
    with open(ylabels_dict_file, 'r') as f:
        ylabels_dict = json.load(f)
    return ylabels_dict


def load_annotations(annotations_file):
    """Load annotations file with parameter-output pairs to highlight.
    Format: {output_name: [param1, param2, ...]}
    """
    if annotations_file and os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            return json.load(f)
    return {}


def load_group_colors(group_colors_file):
    """
    Load group color and marker mapping from JSON file.
    Format: {group_name: {"color": hex_color, "marker": matplotlib_marker}}
    Returns a dictionary mapping groups to dicts with keys 'color' and 'marker'.
    If file doesn't exist, returns empty dict.
    """
    if group_colors_file and os.path.exists(group_colors_file):
        with open(group_colors_file, 'r') as f:
            return json.load(f)
    return {}



def get_parameter_group(param, xlabels_dict):
    """Get the functional group for a parameter, default to 'other' if not found."""
    if param in xlabels_dict:
        group = xlabels_dict[param].get('group', None)
        if group is not None:
            return group
        else:
            raise Exception(f"Parameter {param} does not have a group in xlabels dictionary.")
    else:
        raise Exception(f"Parameter {param} not found in xlabels dictionary.")



def extract_sensitivity_data(scenarios, ylabels_raw_all, gsa_mode="Si_total", mode="max"):
    """
    Extract sensitivity data for all parameters and outputs across anatomies (scenarios).
    Returns a nested dictionary: output_name -> parameter_name -> list of sensitivities across anatomies.
    """
    output_data = {}
    
    for ylabel_raw in ylabels_raw_all:
        output_data[ylabel_raw] = defaultdict(list)
        
        for scenario_idx, scenario in enumerate(scenarios):
            rank_file = os.path.join(scenario, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
            
            if not os.path.exists(rank_file):
                print(f"Warning: File not found: {rank_file}")
                continue
            
            # Read the ranking file
            with open(rank_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        param = parts[0]
                        sensitivity = float(parts[1])
                        output_data[ylabel_raw][param].append(sensitivity)
    
    return output_data


def aggregate_sensitivity_across_anatomies(output_data):
    """
    For each parameter and output, compute variability across anatomies.
    Returns a dictionary: output_name -> DataFrame with columns:
    parameter, mean_sensitivity (across anatomies), std_sensitivity, cv_sensitivity, relevant (bool)
    """
    aggregated_by_output = {}
    
    for output_name, param_data in output_data.items():
        agg_list = []
        
        for param, sensitivities in param_data.items():
            if len(sensitivities) > 0:
                mean_sens = np.mean(sensitivities)
                std_sens = np.std(sensitivities)
                cv_sens = std_sens / mean_sens if mean_sens != 0 else 0
                # Relevant if mean sensitivity >= 0.05
                relevant = mean_sens >= 0.05
                
                agg_list.append({
                    'parameter': param,
                    'mean_sensitivity': mean_sens,
                    'std_sensitivity': std_sens,
                    'cv_sensitivity': cv_sens,
                    'n_anatomies': len(sensitivities),
                    'relevant': relevant
                })
        
        aggregated_by_output[output_name] = pd.DataFrame(agg_list)
    
    return aggregated_by_output


def create_multipanel_scatter(
    scenarios,
    outputs,
    xlabels_dict,
    ylabels_dict,
    savepath,
    fontsize=12,
    figname="importance_vs_variability.png",
    gsa_mode="Si_total",
    mode="max",
    ylabels_raw_all=None,
    annotations_file=None,
    group_colors_file=None,
    supertitle=None
):
    """
    Create a multipanel scatter plot with one panel per output.
    x-axis: parameter's sensitivity in that output (using mean across anatomies)
    y-axis: parameter's variability across anatomies (std dev)
    color: functional group
    Each point = one parameter
    """
    
    # Extract sensitivity data for all outputs across anatomies
    if ylabels_raw_all is None:
        ylabels_raw_all = outputs
    
    output_data = extract_sensitivity_data(scenarios, ylabels_raw_all, gsa_mode, mode)
    
    # Aggregate data across anatomies
    aggregated_by_output = aggregate_sensitivity_across_anatomies(output_data)

    # Determine global axis limits
    all_means = []
    all_stds = []
    for df in aggregated_by_output.values():
        if len(df) > 0:
            all_means.extend(df['mean_sensitivity'].values)
            all_stds.extend(df['std_sensitivity'].values)
    
    if all_means and all_stds:
        x_min, x_max = 0, max(all_means) * 1.15  # add small margin
        y_min, y_max = 0, max(all_stds) * 1.15
    else:
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1

    
    # Load annotations
    annotations = load_annotations(annotations_file)
    
    # Load group colors
    group_colors_custom = load_group_colors(group_colors_file)
    
    # Get all unique parameters and their functional groups
    all_params = set()
    for df in aggregated_by_output.values():
        all_params.update(df['parameter'].values)
    
    param_groups = {param: get_parameter_group(param, xlabels_dict) 
                   for param in all_params}
    
    # Add group information to each output's DataFrame
    for output_name in aggregated_by_output:
        aggregated_by_output[output_name]['group'] = aggregated_by_output[output_name]['parameter'].map(param_groups)
    
    # Create multipanel figure
    n_outputs = len(outputs)
    n_cols = min(3, n_outputs)
    n_rows = int(np.ceil(n_outputs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    # Get unique groups and create colormap
    groups = sorted(set(param_groups.values()))
    
    # Use custom colors and markers if provided, otherwise use defaults
    color_map = {}
    marker_map = {}
    if group_colors_custom:
        for group in groups:
            color_map[group] = group_colors_custom.get(group, {}).get('color', '#808080')
            marker_map[group] = group_colors_custom.get(group, {}).get('marker', 'o')
    else:
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        default_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        for i, group in enumerate(groups):
            color_map[group] = colors[i]
            marker_map[group] = default_markers[i % len(default_markers)]

    
    # Plot each output
    for idx, output in enumerate(outputs):
        ax = axes[idx]

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Determine subplot position
        row_idx = idx // n_cols
        col_idx = idx % n_cols

        # Show x-axis labels/ticks only for the bottom row
        if row_idx != n_rows - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            # Add labels and formatting
            ax.set_xlabel('Mean Sensitivity across anatomies', fontsize=fontsize-1)
        
        
        # Show y-axis labels/ticks only for the first column
        if col_idx != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Std. Dev. across anatomies', fontsize=fontsize-1)


        
        if output not in aggregated_by_output:
            ax.text(0.5, 0.5, f"No data for {output}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue
        
        output_df = aggregated_by_output[output]
        
        if len(output_df) == 0:
            ax.text(0.5, 0.5, f"No data for {output}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue
        
        # Separate relevant and irrelevant parameters
        relevant_df = output_df[output_df['relevant']]
        irrelevant_df = output_df[~output_df['relevant']]
        
        # Plot irrelevant parameters (grey and transparent)
        if len(irrelevant_df) > 0:
            for group in groups:
                group_data = irrelevant_df[irrelevant_df['group'] == group]
                if len(group_data) > 0:
                    ax.scatter(
                        group_data['mean_sensitivity'],
                        group_data['std_sensitivity'],
                        color='grey',
                        s=100,
                        alpha=0.2,
                        edgecolors='grey',
                        linewidth=0.5
                    )
        
        # Plot relevant parameters (colored by group)
        for group in groups:
            group_data = relevant_df[relevant_df['group'] == group]
            if len(group_data) > 0:
                ax.scatter(
                    group_data['mean_sensitivity'],
                    group_data['std_sensitivity'],
                    label=group,
                    color=color_map[group],
                    marker=marker_map[group],
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )

        
        # Add vertical line at x=0.05 threshold
        ax.axvline(x=0.05, color='red', linestyle='-', alpha=0.3, linewidth=2)
        
        
        
        # Use latex format from ylabels_dict if available
        output_title = output
        if output in ylabels_dict:
            output_title = ylabels_dict[output].get('latex', output)
        
        ax.set_title(output_title, fontsize=fontsize, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=fontsize-2)
        
        # Add annotations for specified parameters
        if output in annotations:
            annotated_params = annotations[output]
            # Create text boxes in a designated area (top-right corner of plot)
            text_y_start = y_max * 0.95
            text_line_height = (y_max - y_min) * 0.08
            
            for i, param in enumerate(annotated_params):
                param_data = output_df[output_df['parameter'] == param]
                if len(param_data) > 0:
                    x = param_data['mean_sensitivity'].values[0]
                    y = param_data['std_sensitivity'].values[0]
                    # Get parameter label from xlabels_dict if available
                    param_label = xlabels_dict.get(param, {}).get('latex', param)
                    
                    # Text box position in top-right corner
                    text_x = x_max * 0.98
                    text_y = text_y_start - (i * text_line_height)
                    
                    # Draw arrow from point to text box
                    ax.annotate('', xy=(text_x * 0.85, text_y), xytext=(x, y),
                               arrowprops=dict(arrowstyle='->', lw=1, color='black', alpha=0.5))
                    
                    # Add text in the designated area
                    ax.text(text_x, text_y, param_label, fontsize=fontsize-5,
                           ha='right', va='center',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.6, edgecolor='black', linewidth=0.5))
    
    # Add legend to the figure
    if len(groups) > 0:
        handles = [
            plt.scatter([], [], color=color_map[g], marker=marker_map[g], s=100, alpha=0.7, 
                        edgecolors='black', linewidth=0.5, label=g) 
            for g in groups
        ]

        fig.legend(handles, groups, loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  ncol=min(len(groups), 3), fontsize=fontsize-1, frameon=True)
    
    # Hide empty subplots
    for idx in range(n_outputs, len(axes)):
        axes[idx].set_visible(False)
    
    # Add supertitle if provided
    if supertitle:
        fig.suptitle(supertitle, fontsize=fontsize+7, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Multipanel scatter plot saved to: {output_path}")
    
    # Save summary statistics
    save_scatter_summary(aggregated_by_output, savepath, figname)


def save_scatter_summary(aggregated_by_output, savepath, figname):
    """Save summary statistics for each output."""
    for output_name, df in aggregated_by_output.items():
        summary_file = os.path.join(savepath, f"{output_name}_scatter_summary.csv")
        df.to_csv(summary_file, index=False, float_format='%.4f')
        print(f"Summary statistics saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate importance vs. variability scatter plots for GSA results"
    )
    parser.add_argument('--scenarios', nargs='+', required=True,
                       help='Paths to the scenario folders (each is an anatomy)')
    parser.add_argument('--outputs', nargs='+', required=True,
                       help='Output names to plot (without file extension)')
    parser.add_argument('--xlabels_dict', required=True,
                       help='Path to the xlabels dictionary JSON file')
    parser.add_argument('--ylabels_dict', required=True,
                       help='Path to the ylabels dictionary JSON file')
    parser.add_argument('--savepath', required=True,
                       help='Path to save the figures')
    parser.add_argument('--fontsize', type=int, default=12,
                       help='Font size for the plot')
    parser.add_argument('--figname', default='importance_vs_variability.png',
                       help='Output figure name')
    parser.add_argument('--annotations', default=None,
                       help='Path to JSON file with annotations (format: {output_name: [param1, param2, ...]})')
    parser.add_argument('--group_colors', default=None,
                       help='Path to JSON file with group colors (format: {group_name: hex_color})')
    parser.add_argument('--supertitle', default=None,
                       help='Super title for the entire figure')
    parser.add_argument('--gsa_mode', default='Si_total',
                       help='GSA mode (e.g., Si_total, Si)')
    parser.add_argument('--mode', default='max',
                       help='Mode for ranking (e.g., max, min)')
    
    args = parser.parse_args()
    
    # Load dictionaries
    xlabels_dict = load_xlabels_dict(args.xlabels_dict)
    ylabels_dict = load_ylabels_dict(args.ylabels_dict)
    
    # Create scatter plots
    create_multipanel_scatter(
        scenarios=args.scenarios,
        outputs=args.outputs,
        xlabels_dict=xlabels_dict,
        ylabels_dict=ylabels_dict,
        savepath=args.savepath,
        fontsize=args.fontsize,
        figname=args.figname,
        gsa_mode=args.gsa_mode,
        mode=args.mode,
        ylabels_raw_all=args.outputs,
        annotations_file=args.annotations,
        group_colors_file=args.group_colors,
        supertitle=args.supertitle,
    )


if __name__ == "__main__":
    main()