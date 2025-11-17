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
    Load group color mapping from JSON file.
    Format: {group_name: {"color": hex_color}}
    """
    if group_colors_file and os.path.exists(group_colors_file):
        with open(group_colors_file, 'r') as f:
            return json.load(f)
    return {}


def load_exclusions(exclusions_file):
    """
    Load exclusions file mapping modified scenarios to parameters to exclude.
    Format: {modified_scenario_name: [param1, param2, ...]}
    If file doesn't exist, returns empty dict.
    """
    if exclusions_file and os.path.exists(exclusions_file):
        with open(exclusions_file, 'r') as f:
            return json.load(f)
    return {}


def get_parameter_group(param, xlabels_dict):
    """Get the functional group for a parameter."""
    if param in xlabels_dict:
        group = xlabels_dict[param].get('group', None)
        if group is not None:
            return group
        else:
            raise Exception(f"Parameter {param} does not have a group in xlabels dictionary.")
    else:
        raise Exception(f"Parameter {param} not found in xlabels dictionary.")


def extract_sensitivity_data_by_anatomy(anatomy_data, ylabels_raw_all, 
                                        modified_names, exclusions,
                                        gsa_mode="Si_total", mode="max"):
    """
    Extract sensitivity data for all parameters and outputs for one anatomy.
    anatomy_data: dict with keys 'baseline' and 'modified' (list of paths)
    modified_names: list of modified scenario names (for exclusion lookup)
    exclusions: dict mapping modified scenario names to lists of parameters to exclude
    Returns: dict with structure:
        output_name -> parameter_name -> {'baseline': float, 'modified': [floats]}
    """
    output_data = {}
    
    for ylabel_raw in ylabels_raw_all:
        output_data[ylabel_raw] = defaultdict(lambda: {'baseline': None, 'modified': []})
        
        # Load baseline
        baseline_rank_file = os.path.join(anatomy_data['baseline'], "output", 
                                         f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
        if not os.path.isfile(baseline_rank_file):
            baseline_rank_file = os.path.join(anatomy_data['baseline'],  
                                         f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
        with open(baseline_rank_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    param = parts[0]
                    sensitivity = float(parts[1])
                    output_data[ylabel_raw][param]['baseline'] = sensitivity
        
        # Load modified scenarios
        for i, modified_scenario in enumerate(anatomy_data['modified']):
            modified_name = modified_names[i]
            excluded_params = exclusions.get(modified_name, [])
            
            modified_rank_file = os.path.join(modified_scenario, "output",
                                            f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
            if not os.path.isfile(modified_rank_file):
                modified_rank_file = os.path.join(modified_scenario,
                                            f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
            with open(modified_rank_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        param = parts[0]
                        # Skip if parameter is in exclusion list for this modified scenario
                        if param in excluded_params:
                            # print(f"{param} excluded")
                            continue
                        sensitivity = float(parts[1])
                        output_data[ylabel_raw][param]['modified'].append(sensitivity)
    
    return output_data


def extract_all_anatomies_data(anatomies_dict, ylabels_raw_all, modified_names, 
                               exclusions, gsa_mode="Si_total", mode="max"):
    """
    Extract data for all anatomies.
    Returns: dict with structure:
        anatomy_name -> output_name -> parameter_name -> {'baseline': float, 'modified': [floats]}
    """
    all_data = {}
    for anatomy_name, anatomy_data in anatomies_dict.items():
        all_data[anatomy_name] = extract_sensitivity_data_by_anatomy(
            anatomy_data, ylabels_raw_all, modified_names, exclusions, gsa_mode, mode
        )
    return all_data


def aggregate_for_comparison_boxplot(all_anatomies_data, xlabels_dict, threshold=0.05):
    """
    Prepare data for boxplot visualization comparing baseline vs modified scenarios.
    Returns nested dict: output_name -> parameter_name -> list of anatomy_data
    where each anatomy_data contains: baseline value and modified values
    """
    aggregated_by_output = defaultdict(lambda: defaultdict(list))
    
    # Collect all unique parameters
    all_params = set()
    for anatomy_data in all_anatomies_data.values():
        for output_data in anatomy_data.values():
            all_params.update(output_data.keys())
    
    # For each output, collect data across all anatomies
    for anatomy_name, anatomy_data in all_anatomies_data.items():
        for output_name, param_data in anatomy_data.items():
            for param in all_params:
                if param in param_data:
                    baseline_val = param_data[param]['baseline']
                    modified_vals = param_data[param]['modified']
                    
                    # Only include if we have data
                    if baseline_val is not None and len(modified_vals) > 0:
                        # Check threshold: include if baseline OR any modified exceeds threshold
                        max_modified = max(modified_vals)
                        if baseline_val >= threshold or max_modified >= threshold:
                            try:
                                group = get_parameter_group(param, xlabels_dict)
                                aggregated_by_output[output_name][param].append({
                                    'anatomy': anatomy_name,
                                    'baseline': baseline_val,
                                    'modified': modified_vals,
                                    'group': group
                                })
                            except Exception as e:
                                print(f"Warning: {e}")
    
    return aggregated_by_output


def create_comparison_boxplot(
    anatomies_dict,
    outputs,
    xlabels_dict,
    ylabels_dict,
    savepath,
    fontsize=12,
    figname="sensitivity_comparison_boxplots.png",
    gsa_mode="Si_total",
    mode="max",
    ylabels_raw_all=None,
    annotations_file=None,
    group_colors_file=None,
    supertitle=None,
    threshold=0.05,
    exclusions_file=None
):
    """
    Create multipanel boxplot comparing baseline vs modified scenarios.
    Each panel = one output
    For each parameter: 5 boxplots (one per anatomy)
    Each boxplot contains: baseline (red diamond) + modified scenarios (black circles)
    """
    
    if ylabels_raw_all is None:
        ylabels_raw_all = outputs
    
    # Load exclusions
    exclusions = load_exclusions(exclusions_file)
    if exclusions:
        print(f"Loaded exclusions for {len(exclusions)} modified scenarios")
        for mod_name, params in exclusions.items():
            print(f"  {mod_name}: excluding {len(params)} parameters")
    
    # Get modified scenario names from first anatomy
    first_anatomy = list(anatomies_dict.values())[0]
    modified_names = [os.path.basename(path) for path in first_anatomy['modified']]
    
    # Extract all data
    print("Extracting sensitivity data from all anatomies...")
    all_anatomies_data = extract_all_anatomies_data(anatomies_dict, ylabels_raw_all, 
                                                    modified_names, exclusions, 
                                                    gsa_mode, mode)
    
    # Aggregate data for plotting
    print("Aggregating data for comparison...")
    aggregated_by_output = aggregate_for_comparison_boxplot(all_anatomies_data, xlabels_dict, threshold)
    
    # Load annotations and group colors
    annotations = load_annotations(annotations_file)
    group_colors_custom = load_group_colors(group_colors_file)
    
    # Determine global y-limits
    global_ymax = 0
    for output_name, param_data in aggregated_by_output.items():
        for param, anatomy_list in param_data.items():
            for anatomy_entry in anatomy_list:
                local_max = max([anatomy_entry['baseline']] + anatomy_entry['modified'])
                if local_max > global_ymax:
                    global_ymax = local_max
    
    global_ymin = 0.04
    print(f"Global y-limit: ymin={global_ymin}, ymax={global_ymax:.3f}")
    
    # Get anatomy names
    anatomy_names = list(anatomies_dict.keys())
    n_anatomies = len(anatomy_names)
    
    # Create figure
    n_outputs = len(outputs)
    n_cols = min(3, n_outputs)
    n_rows = int(np.ceil(n_outputs / n_cols))
    
    fig_width = 8 * n_cols
    fig_height = 7 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).flatten()
    
    # Collect all unique groups for legend
    all_groups = set()
    for param_data in aggregated_by_output.values():
        for anatomy_list in param_data.values():
            for entry in anatomy_list:
                all_groups.add(entry['group'])
    all_groups = sorted(all_groups)
    
    # Create color map
    color_map = {}
    if group_colors_custom:
        for group in all_groups:
            color_map[group] = group_colors_custom.get(group, {}).get('color', '#808080')
    else:
        # colors = plt.cm.Set2(np.linspace(0, 1, len(all_groups)))
        for i, group in enumerate(all_groups):
            color_map[group] = '#808080'
    
    # Plot each output
    for idx, output in enumerate(outputs):
        ax = axes[idx]
        
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        
        # Y-axis label only for first column
        if col_idx != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Log-Sensitivity', fontsize=fontsize-1)
        
        if output not in aggregated_by_output:
            ax.text(0.5, 0.5, f"No data for {output}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue
        
        param_data = aggregated_by_output[output]
        
        if len(param_data) == 0:
            ax.text(0.5, 0.5, f"No data above threshold for {output}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue
        
        # Sort parameters by mean baseline sensitivity across anatomies
        param_mean_baseline = {}
        for param, anatomy_list in param_data.items():
            baseline_vals = [entry['baseline'] for entry in anatomy_list]
            param_mean_baseline[param] = np.mean(baseline_vals)
        
        sorted_params = sorted(param_mean_baseline.keys(), 
                              key=lambda p: param_mean_baseline[p], 
                              reverse=True)
        
        # Prepare boxplot data
        # For each parameter, we'll have n_anatomies boxplots side-by-side
        all_boxplot_data = []
        all_positions = []
        all_colors = []
        all_baseline_positions = []
        all_baseline_values = []
        all_anatomy_labels = []  # Track which anatomy each boxplot belongs to
        param_groups = []
        
        x_position = 1  # Start at 1 for better spacing
        param_positions = {}  # Store center position for each parameter
        param_x_starts = {}  # Store start position for each parameter
        
        for param_idx, param in enumerate(sorted_params):
            anatomy_list = param_data[param]
            
            # Get group for this parameter (should be same across all anatomies)
            param_group = anatomy_list[0]['group']
            
            # Store start position
            param_x_starts[param] = x_position
            
            # Count how many anatomies have data for this parameter
            n_anatomies_with_data = 0
            for anatomy_name in anatomy_names:
                for entry in anatomy_list:
                    if entry['anatomy'] == anatomy_name:
                        n_anatomies_with_data += 1
                        break
            
            # Calculate center position for this parameter
            param_center = x_position + (n_anatomies_with_data - 1) / 2.0
            param_positions[param] = param_center
            
            for anatomy_idx, anatomy_name in enumerate(anatomy_names):
                # Find data for this anatomy
                anatomy_entry = None
                for entry in anatomy_list:
                    if entry['anatomy'] == anatomy_name:
                        anatomy_entry = entry
                        break
                
                # if anatomy_entry is None:
                #     # No data for this anatomy, skip
                #     continue
                if anatomy_entry is not None:
                    # Boxplot data: modified scenarios
                    modified_vals = anatomy_entry['modified']
                    baseline_val = anatomy_entry['baseline']
                    
                    all_boxplot_data.append(modified_vals)
                    all_positions.append(x_position)
                    all_colors.append(color_map[param_group])
                    all_anatomy_labels.append(anatomy_name)
                    
                    # Store baseline position and value for plotting as diamond
                    all_baseline_positions.append(x_position)
                    all_baseline_values.append(baseline_val)
                
                x_position += 1
            
            # Add spacing between parameters
            # x_position += 1
            param_groups.append(param_group)
        
        # Plot boxplots for modified scenarios
        if len(all_boxplot_data) > 0:
            # Overlay individual points (modified scenarios) as black circles
            for i, (modified_vals, pos) in enumerate(zip(all_boxplot_data, all_positions)):
                y = modified_vals
                x = np.random.normal(pos, 0.05, size=len(y))
                ax.scatter(x, y, alpha=0.6, s=30, color='black', 
                          marker='o', zorder=3, edgecolor='black', linewidth=0.5)
            
            # Plot baseline as red diamonds
            ax.scatter(all_baseline_positions, all_baseline_values, 
                      color='red', marker='D', s=30, zorder=3, 
                      edgecolor='darkred', linewidth=0.5, label='Baseline', alpha=0.6)

            bp = ax.boxplot(all_boxplot_data, positions=all_positions, 
                           patch_artist=True, showfliers=False, widths=0.7)
            
            # Color boxplots
            for patch, color in zip(bp['boxes'], all_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)
            
            # Style whiskers, caps, medians
            for whisker in bp['whiskers']:
                whisker.set(linewidth=1, color='black', alpha=0.7)
            for cap in bp['caps']:
                cap.set(linewidth=1, color='black', alpha=0.7)
            for median in bp['medians']:
                median.set(linewidth=2, color='#AA151B')
            

            
            # Add vertical separators between parameters
            for param in sorted_params:
                if param in param_x_starts:
                    # Draw a light vertical line before each parameter group
                    ax.axvline(x=param_x_starts[param] - 0.5, color='gray', 
                              linestyle=':', alpha=0.3, linewidth=1)
        
        # Add threshold line
        ax.axhline(y=threshold, color='#AA151B', linestyle='-', alpha=0.3, linewidth=2)
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        ax.set_ylim(global_ymin, global_ymax)
        
        # Set title
        output_title = output
        if output in ylabels_dict:
            output_title = ylabels_dict[output].get('latex', output)
        ax.set_title(output_title, fontsize=fontsize, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.tick_params(labelsize=fontsize-2)
        
        # X-axis labels: parameter names at center of each parameter group
        xtick_positions = []
        xtick_labels = []
        for param in sorted_params:
            if param in param_positions:
                xtick_positions.append(param_positions[param])
                latex_label = xlabels_dict.get(param, {}).get('latex', param)
                xtick_labels.append(latex_label)
        
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=fontsize-3)
        
        # Add annotations
        if output in annotations:
            annotated_params = annotations[output]
            for param in annotated_params:
                if param in param_positions:
                    param_center = param_positions[param]
                    # Get mean baseline value for annotation position
                    anatomy_list = param_data[param]
                    mean_baseline = np.mean([e['baseline'] for e in anatomy_list])
                    
                    param_label = xlabels_dict.get(param, {}).get('latex', param)
                    ax.annotate(param_label, xy=(param_center, mean_baseline),
                               xytext=(param_center + 0.5, mean_baseline * 2),
                               fontsize=fontsize-5, ha='left',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                                       alpha=0.6, edgecolor='black', linewidth=0.5),
                               arrowprops=dict(arrowstyle='->', lw=1, color='black', alpha=0.5))
    
    # Create legends
    # Legend 1: Parameter groups (colors)
    group_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[g], alpha=0.5,
                     edgecolor='black', linewidth=1, label=g)
        for g in all_groups
    ]
    
    # Legend 2: Baseline vs Modified
    scenario_handles = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                  markeredgecolor='darkred', markersize=10, label='Baseline', linewidth=0),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                  markeredgecolor='black', markersize=8, label='Modified', linewidth=0)
    ]
    
    
    legend2 = fig.legend(handles=scenario_handles, title="Scenario type",
                        loc='upper center', bbox_to_anchor=(0.77, 0.3),
                        ncol=1, fontsize=fontsize+2, frameon=False,
                        title_fontsize=fontsize+3)
    legend2.get_title().set_fontweight('bold')
    
    
    fig.add_artist(legend2)
    
    # Hide empty subplots
    for idx in range(n_outputs, len(axes)):
        axes[idx].set_visible(False)
    
    # Add supertitle
    if supertitle:
        fig.suptitle(supertitle, fontsize=fontsize+12, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    
    # Save figure
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Comparison boxplot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity comparison boxplots (baseline vs modified scenarios)"
    )
    
    # Anatomy arguments (dynamically add based on number of anatomies)
    parser.add_argument('--n_anatomies', type=int, required=True,
                       help='Number of anatomies')
    parser.add_argument('--anatomy_names', nargs='+', required=True,
                       help='Names for each anatomy')
    
    # Will use --anatomy1_baseline, --anatomy1_modified, etc.
    parser.add_argument('--outputs', nargs='+', required=True,
                       help='Output names to plot')
    parser.add_argument('--xlabels_dict', required=True,
                       help='Path to xlabels dictionary JSON')
    parser.add_argument('--ylabels_dict', required=True,
                       help='Path to ylabels dictionary JSON')
    parser.add_argument('--savepath', required=True,
                       help='Path to save figures')
    parser.add_argument('--fontsize', type=int, default=12,
                       help='Font size')
    parser.add_argument('--figname', default='sensitivity_comparison_boxplots.png',
                       help='Output figure name')
    parser.add_argument('--annotations', default=None,
                       help='Path to annotations JSON')
    parser.add_argument('--group_colors', default=None,
                       help='Path to group colors JSON')
    parser.add_argument('--supertitle', default=None,
                       help='Super title for figure')
    parser.add_argument('--gsa_mode', default='Si_total',
                       help='GSA mode')
    parser.add_argument('--mode', default='max',
                       help='Ranking mode')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Sensitivity threshold for inclusion')
    parser.add_argument('--exclusions', default=None,
                       help='Path to JSON file with exclusions (format: {modified_scenario: [param1, param2, ...]})')
    
    # Parse known args first to get n_anatomies
    args, remaining = parser.parse_known_args()
    args, remaining = parser.parse_known_args()
    
    # Add anatomy-specific arguments
    for i in range(1, args.n_anatomies + 1):
        parser.add_argument(f'--anatomy{i}_baseline', required=True,
                           help=f'Baseline scenario path for anatomy {i}')
        parser.add_argument(f'--anatomy{i}_modified', nargs='+', required=True,
                           help=f'Modified scenario paths for anatomy {i}')
    
    # Parse all arguments
    args = parser.parse_args()
    
    # Build anatomies dictionary
    anatomies_dict = {}
    for i, anatomy_name in enumerate(args.anatomy_names, 1):
        baseline_path = getattr(args, f'anatomy{i}_baseline')
        modified_paths = getattr(args, f'anatomy{i}_modified')
        anatomies_dict[anatomy_name] = {
            'baseline': baseline_path,
            'modified': modified_paths
        }
    
    # Load dictionaries
    xlabels_dict = load_xlabels_dict(args.xlabels_dict)
    ylabels_dict = load_ylabels_dict(args.ylabels_dict)
    
    # Create boxplots
    create_comparison_boxplot(
        anatomies_dict=anatomies_dict,
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
        threshold=args.threshold,
        exclusions_file=args.exclusions
    )


if __name__ == "__main__":
    main()