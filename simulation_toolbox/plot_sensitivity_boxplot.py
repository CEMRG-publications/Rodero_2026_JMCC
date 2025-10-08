import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def aggregate_sensitivity_for_boxplot(output_data, xlabels_dict):
    """
    For each output, prepare data for boxplot visualization.
    Returns a dictionary: output_name -> DataFrame with columns:
    parameter, sensitivities (list), mean_sensitivity, group
    """
    aggregated_by_output = {}
    
    for output_name, param_data in output_data.items():
        agg_list = []
        
        for param, sensitivities in param_data.items():
            if len(sensitivities) > 0:
                mean_sens = np.mean(sensitivities)
                # Only include parameters with at least one point above threshold
                if max(sensitivities) >= 0.05:
                    try:
                        group = get_parameter_group(param, xlabels_dict)
                        agg_list.append({
                            'parameter': param,
                            'sensitivities': sensitivities,
                            'mean_sensitivity': mean_sens,
                            'group': group
                        })
                    except Exception as e:
                        print(f"Warning: {e}")
        
        aggregated_by_output[output_name] = agg_list
    
    return aggregated_by_output


def get_anatomy_markers(n_anatomies):
    """Generate a list of distinct markers for each anatomy."""
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd']
    if n_anatomies <= len(markers):
        return markers[:n_anatomies]
    else:
        # If more anatomies than markers, cycle through
        return [markers[i % len(markers)] for i in range(n_anatomies)]


def create_multipanel_boxplot(
    scenarios,
    outputs,
    xlabels_dict,
    ylabels_dict,
    savepath,
    fontsize=12,
    figname="sensitivity_boxplots.png",
    gsa_mode="Si_total",
    mode="max",
    ylabels_raw_all=None,
    annotations_file=None,
    group_colors_file=None,
    supertitle=None,
    anatomy_names=None
):
    """
    Create a multipanel boxplot where each panel shows boxplots for one output.
    x-axis: parameters (categorical, sorted by mean sensitivity in descending order)
    y-axis: sensitivity (log scale)
    color: functional group
    Each boxplot = one parameter with points from different anatomies
    """
    
    # Extract sensitivity data for all outputs across anatomies
    if ylabels_raw_all is None:
        ylabels_raw_all = outputs
    
    output_data = extract_sensitivity_data(scenarios, ylabels_raw_all, gsa_mode, mode)
    
    # Aggregate data for boxplot
    aggregated_by_output = aggregate_sensitivity_for_boxplot(output_data, xlabels_dict)

    # Determine global y-limit across all outputs
    global_ymax = 0
    for output_data_list in aggregated_by_output.values():
        for item in output_data_list:
            if len(item['sensitivities']) > 0:
                local_max = max(item['sensitivities'])
                if local_max > global_ymax:
                    global_ymax = local_max

    # Optionally set a lower bound for visibility on log scale
    global_ymin = 0.04  # or whatever fits your data
    print(f"Global y-limit across all outputs: ymin={global_ymin}, ymax={global_ymax:.3f}")

    
    # Load annotations
    annotations = load_annotations(annotations_file)
    
    # Load group colors
    group_colors_custom = load_group_colors(group_colors_file)
    
    # Get anatomy markers and names
    n_anatomies = len(scenarios)
    anatomy_markers = get_anatomy_markers(n_anatomies)
    if anatomy_names is None:
        anatomy_names = [f"Anatomy {i+1}" for i in range(n_anatomies)]
    
    # Create multipanel figure
    n_outputs = len(outputs)
    n_cols = min(3, n_outputs)
    n_rows = int(np.ceil(n_outputs / n_cols))
    
    # Fixed width for consistent x-axis space, variable height for clarity
    fig_width = 8 * n_cols
    fig_height = 7 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    axes = np.atleast_1d(axes).flatten()
    
    # Collect all unique groups for legend
    all_groups = set()
    for output_data_list in aggregated_by_output.values():
        for item in output_data_list:
            all_groups.add(item['group'])
    all_groups = sorted(all_groups)
    
    # Create color map for groups
    color_map = {}
    if group_colors_custom:
        for group in all_groups:
            color_map[group] = group_colors_custom.get(group, {}).get('color', '#808080')
    else:
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_groups)))
        for i, group in enumerate(all_groups):
            color_map[group] = colors[i]
    
    # Plot each output
    for idx, output in enumerate(outputs):
        ax = axes[idx]
        
        # Determine subplot position
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        
        ax.set_xticklabels([])
        
        # Show y-axis labels/ticks only for the first column
        if col_idx != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Log-Sensitivity', fontsize=fontsize-1)
        
        if output not in aggregated_by_output:
            ax.text(0.5, 0.5, f"No data for {output}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue
        
        output_data_list = aggregated_by_output[output]
        
        if len(output_data_list) == 0:
            ax.text(0.5, 0.5, f"No data above threshold for {output}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue
        
        # Sort by mean sensitivity in descending order
        output_data_list_sorted = sorted(output_data_list, 
                                        key=lambda x: x['mean_sensitivity'], 
                                        reverse=True)
        
        # Prepare data for boxplot
        param_labels = []
        boxplot_data = []
        param_groups = []
        
        for item in output_data_list_sorted:
            param_labels.append(item['parameter'])
            boxplot_data.append(item['sensitivities'])
            param_groups.append(item['group'])

        # Overlay individual points (anatomies) with different markers
        for i, sensitivities in enumerate(boxplot_data):
            y = sensitivities
            x = np.random.normal(i+1, 0.04, size=len(y))  # Add jitter
            for j, sens in enumerate(y):
                marker = anatomy_markers[j % n_anatomies]
                ax.scatter(x[j], sens, alpha=0.6, s=60, color='black', 
                          marker=marker, zorder=3, edgecolor='black', linewidth=0.5)
        
        # Plot boxplots
        bp = ax.boxplot(boxplot_data, labels=param_labels, patch_artist=True,
                       showfliers=False, widths=0.6)
        
        # Color boxplots by group
        for patch, group in zip(bp['boxes'], param_groups):
            patch.set_facecolor(color_map[group])
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        
        # Style whiskers, caps, and medians
        for whisker in bp['whiskers']:
            whisker.set(linewidth=1, color='black', alpha=0.7)
        for cap in bp['caps']:
            cap.set(linewidth=1, color='black', alpha=0.7)
        # for median in bp['medians']:
        #     median.set(linewidth=2, color='darkred')
        
        
        
        # Add vertical line at y=0.05 threshold
        ax.axhline(y=0.05, color='red', linestyle='-', alpha=0.3, linewidth=2)
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        # Apply consistent y-limits across all subplots
        ax.set_ylim(global_ymin, global_ymax)

        
        # Use latex format from ylabels_dict if available
        output_title = output
        if output in ylabels_dict:
            output_title = ylabels_dict[output].get('latex', output)
        
        ax.set_title(output_title, fontsize=fontsize, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.tick_params(labelsize=fontsize-2)
        
        # Use LaTeX labels from xlabels_dict if available
        latex_labels = []
        for param in param_labels:
            latex_label = xlabels_dict.get(param, {}).get('latex', param)
            latex_labels.append(latex_label)

        # Apply LaTeX labels to x-axis
        ax.set_xticks(range(1, len(param_labels) + 1))
        ax.set_xticklabels(latex_labels, rotation=45, ha='right', fontsize=fontsize-3)

        
        # Add annotations for specified parameters
        if output in annotations:
            annotated_params = annotations[output]
            
            for param in annotated_params:
                param_idx = None
                for i, item in enumerate(output_data_list_sorted):
                    if item['parameter'] == param:
                        param_idx = i
                        break
                
                if param_idx is not None:
                    # Get the mean sensitivity for positioning
                    mean_sens = output_data_list_sorted[param_idx]['mean_sensitivity']
                    x_pos = param_idx + 1
                    
                    # Get parameter label from xlabels_dict if available
                    param_label = xlabels_dict.get(param, {}).get('latex', param)
                    
                    # Draw arrow and annotation box
                    ax.annotate(param_label, xy=(x_pos, mean_sens),
                               xytext=(x_pos + 0.3, mean_sens * 2),
                               fontsize=fontsize-5,
                               ha='left',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                                        alpha=0.6, edgecolor='black', linewidth=0.5),
                               arrowprops=dict(arrowstyle='->', lw=1, color='black', alpha=0.5))
    
    # --- Separate legends: Output type (colors) and Heart (markers) ---

    # Create custom legend for output types (color-coded boxplots)
    group_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[g], alpha=0.5,
                      edgecolor='black', linewidth=1, label=g)
        for g in all_groups
    ]

    # Create custom legend for hearts (marker-coded anatomies)
    anatomy_handles = [
        plt.Line2D([0], [0], marker=anatomy_markers[i], color='w',
                   markerfacecolor='black', markeredgecolor='black',
                   markersize=8, label=anatomy_names[i], linewidth=0)
        for i in range(n_anatomies)
    ]

    # First legend — Output type (box colors)
    legend1 = fig.legend(
        handles=group_handles,
        title="Output type",
        loc='upper center',
        bbox_to_anchor=(0.77, 0.35),
        ncol=1,
        fontsize=fontsize-1,
        frameon=False,
        title_fontsize=fontsize
    )
    legend1.get_title().set_fontweight('bold')


    # Second legend — Heart (markers)
    legend2 = fig.legend(
        handles=anatomy_handles,
        title="Heart",
        loc='upper center',
        bbox_to_anchor=(0.92, 0.35),
        ncol=1,
        fontsize=fontsize-1,
        frameon=False,
        title_fontsize=fontsize
    )
    legend2.get_title().set_fontweight('bold')

    # Add both legends to figure manually
    fig.add_artist(legend1)
    fig.add_artist(legend2)

    
    # Hide empty subplots
    for idx in range(n_outputs, len(axes)):
        axes[idx].set_visible(False)
    
    # Add supertitle if provided
    if supertitle:
        fig.suptitle(supertitle, fontsize=fontsize+9, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    
    # Save figure
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Multipanel boxplot saved to: {output_path}")
    
    # Save summary statistics
    save_boxplot_summary(aggregated_by_output, savepath, figname)


def save_boxplot_summary(aggregated_by_output, savepath, figname):
    """Save summary statistics for each output."""
    for output_name, data_list in aggregated_by_output.items():
        if len(data_list) > 0:
            summary_data = []
            for item in data_list:
                sensitivities = item['sensitivities']
                summary_data.append({
                    'parameter': item['parameter'],
                    'group': item['group'],
                    'mean': np.mean(sensitivities),
                    'std': np.std(sensitivities),
                    'min': np.min(sensitivities),
                    'max': np.max(sensitivities),
                    'range':  np.max(sensitivities)-np.min(sensitivities),
                    'median': np.median(sensitivities),
                    'n_anatomies': len(sensitivities)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(savepath, f"{output_name}_boxplot_summary.csv")
            summary_df.to_csv(summary_file, index=False, float_format='%.4f')
            print(f"Summary statistics saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity boxplots for GSA results across anatomies"
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
    parser.add_argument('--figname', default='sensitivity_boxplots.png',
                       help='Output figure name')
    parser.add_argument('--annotations', default=None,
                       help='Path to JSON file with annotations (format: {output_name: [param1, param2, ...]})')
    parser.add_argument('--group_colors', default=None,
                       help='Path to JSON file with group colors (format: {group_name: {"color": hex_color}})')
    parser.add_argument('--supertitle', default=None,
                       help='Super title for the entire figure')
    parser.add_argument('--gsa_mode', default='Si_total',
                       help='GSA mode (e.g., Si_total, Si)')
    parser.add_argument('--mode', default='max',
                       help='Mode for ranking (e.g., max, min)')
    parser.add_argument('--anatomy_names', nargs='+', default=None,
                       help='Names for each anatomy (should match number of scenarios)')
    
    args = parser.parse_args()
    
    # Load dictionaries
    xlabels_dict = load_xlabels_dict(args.xlabels_dict)
    ylabels_dict = load_ylabels_dict(args.ylabels_dict)
    
    # Create boxplots
    create_multipanel_boxplot(
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
        anatomy_names=args.anatomy_names,
    )


if __name__ == "__main__":
    main()