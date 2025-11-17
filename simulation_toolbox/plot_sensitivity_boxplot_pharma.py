import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
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


def load_scenario_colors(scenario_colors_file):
    """
    Load scenario color mapping from JSON file.
    Format: {scenario_name: {"color": hex_color, "label": display_label}}
    The baseline scenario color is always red and doesn't need to be specified.
    """
    if scenario_colors_file and os.path.exists(scenario_colors_file):
        with open(scenario_colors_file, 'r') as f:
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
        output_name -> parameter_name -> {'baseline': float, 'modified': {scenario_name: float}}
    """
    output_data = {}
    
    for ylabel_raw in ylabels_raw_all:
        output_data[ylabel_raw] = defaultdict(lambda: {'baseline': None, 'modified': {}})
        
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
                            continue
                        sensitivity = float(parts[1])
                        output_data[ylabel_raw][param]['modified'][modified_name] = sensitivity
    
    return output_data


def extract_all_anatomies_data(anatomies_dict, ylabels_raw_all, modified_names, 
                               exclusions, gsa_mode="Si_total", mode="max"):
    """
    Extract data for all anatomies.
    Returns: dict with structure:
        anatomy_name -> output_name -> parameter_name -> {'baseline': float, 'modified': {scenario_name: float}}
    """
    all_data = {}
    for anatomy_name, anatomy_data in anatomies_dict.items():
        all_data[anatomy_name] = extract_sensitivity_data_by_anatomy(
            anatomy_data, ylabels_raw_all, modified_names, exclusions, gsa_mode, mode
        )
    return all_data


def aggregate_for_scenario_boxplot(all_anatomies_data, xlabels_dict, threshold=0.05):
    """
    Prepare data for box visualization with scenarios.
    Returns nested dict: output_name -> parameter_name -> scenario_name -> list of (anatomy_name, value) tuples
    Also returns baseline data: output_name -> parameter_name -> list of (anatomy_name, value) tuples
    """
    aggregated_by_output = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    baseline_by_output = defaultdict(lambda: defaultdict(list))
    
    # Collect all unique parameters and scenarios
    all_params = set()
    all_scenarios = set()
    for anatomy_data in all_anatomies_data.values():
        for output_data in anatomy_data.values():
            all_params.update(output_data.keys())
            for param_data in output_data.values():
                all_scenarios.update(param_data['modified'].keys())
    
    all_scenarios = sorted(all_scenarios)
    
    # For each output and parameter, collect data across anatomies
    for anatomy_name, anatomy_data in all_anatomies_data.items():
        for output_name, param_data in anatomy_data.items():
            for param in all_params:
                if param in param_data:
                    baseline_val = param_data[param]['baseline']
                    modified_dict = param_data[param]['modified']
                    
                    # Store baseline value for this anatomy with anatomy name
                    if baseline_val is not None:
                        baseline_by_output[output_name][param].append((anatomy_name, baseline_val))
                    
                    # Store modified values for each scenario with anatomy name
                    for scenario_name in all_scenarios:
                        if scenario_name in modified_dict:
                            scenario_val = modified_dict[scenario_name]
                            aggregated_by_output[output_name][param][scenario_name].append((anatomy_name, scenario_val))
    
    # Filter by threshold: include parameter if baseline OR any scenario exceeds threshold
    filtered_aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    filtered_baseline = defaultdict(lambda: defaultdict(list))
    
    for output_name in aggregated_by_output:
        for param in aggregated_by_output[output_name]:
            baseline_data = baseline_by_output[output_name][param]
            scenario_data = aggregated_by_output[output_name][param]
            
            if len(baseline_data) == 0:
                continue
            
            # Check if any value exceeds threshold
            max_baseline = max([val for _, val in baseline_data])
            max_scenarios = []
            for scenario_name in scenario_data:
                if len(scenario_data[scenario_name]) > 0:
                    max_scenarios.append(max([val for _, val in scenario_data[scenario_name]]))
            
            max_scenario_val = max(max_scenarios) if max_scenarios else 0
            
            if max_baseline >= threshold or max_scenario_val >= threshold:
                try:
                    group = get_parameter_group(param, xlabels_dict)
                    filtered_baseline[output_name][param] = baseline_data
                    filtered_aggregated[output_name][param] = scenario_data
                except Exception as e:
                    print(f"Warning: {e}")
    
    return filtered_aggregated, filtered_baseline, all_scenarios


def create_scenario_boxplot(
    anatomies_dict,
    outputs,
    xlabels_dict,
    ylabels_dict,
    savepath,
    fontsize=12,
    figname="sensitivity_scenario_boxes.png",
    gsa_mode="Si_total",
    mode="max",
    ylabels_raw_all=None,
    annotations_file=None,
    group_colors_file=None,
    scenario_colors_file=None,
    supertitle=None,
    threshold=0.05,
    exclusions_file=None,
    box_width=0.7
):
    """
    Create multipanel plot with scenarios as boxes/patches (instead of boxplots).
    Each panel = one output
    For each parameter: N+1 boxes (baseline + N modified scenarios)
    Each box represents the range (min to max) across anatomies
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
    print("Aggregating data for scenario comparison...")
    aggregated_by_output, baseline_by_output, all_scenarios = aggregate_for_scenario_boxplot(
        all_anatomies_data, xlabels_dict, threshold)
    
    # Load annotations and group colors
    annotations = load_annotations(annotations_file)
    group_colors_custom = load_group_colors(group_colors_file)
    scenario_colors_custom = load_scenario_colors(scenario_colors_file)
    
    # Determine global y-limits
    global_ymax = 0
    for output_name in aggregated_by_output:
        for param in aggregated_by_output[output_name]:
            # Check baseline
            baseline_data = baseline_by_output[output_name][param]
            if baseline_data:
                local_max = max([val for _, val in baseline_data])
                if local_max > global_ymax:
                    global_ymax = local_max
            # Check scenarios
            for scenario_name in aggregated_by_output[output_name][param]:
                scenario_data = aggregated_by_output[output_name][param][scenario_name]
                if scenario_data:
                    local_max = max([val for _, val in scenario_data])
                    if local_max > global_ymax:
                        global_ymax = local_max
    
    global_ymin = 0.04
    print(f"Global y-limit: ymin={global_ymin}, ymax={global_ymax:.3f}")
    
    # Get anatomy names
    anatomy_names = list(anatomies_dict.keys())
    n_anatomies = len(anatomy_names)
    
    # Define markers for each anatomy
    available_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    anatomy_markers = {}
    for i, anatomy_name in enumerate(anatomy_names):
        anatomy_markers[anatomy_name] = available_markers[i % len(available_markers)]
    
    # Create figure
    n_outputs = len(outputs)
    n_cols = min(3, n_outputs)
    n_rows = int(np.ceil(n_outputs / n_cols))
    
    fig_width = 24
    fig_height = 7 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).flatten()
    
    # Collect all unique groups for legend
    all_groups = set()
    for output_name in aggregated_by_output:
        for param in aggregated_by_output[output_name]:
            try:
                group = get_parameter_group(param, xlabels_dict)
                all_groups.add(group)
            except:
                pass
    all_groups = sorted(all_groups)
    
    # Create scenario color map
    scenario_color_map = {}
    scenario_labels = {}
    for scenario_name in all_scenarios:
        if scenario_name in scenario_colors_custom:
            scenario_color_map[scenario_name] = scenario_colors_custom[scenario_name].get('color', '#808080')
            scenario_labels[scenario_name] = scenario_colors_custom[scenario_name].get('label', scenario_name)
        else:
            scenario_color_map[scenario_name] = '#808080'
            scenario_labels[scenario_name] = scenario_name
    
    # Plot each output
    for idx, output in enumerate(outputs):
        ax = axes[idx]
        
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        
        # Y-axis label only for first column
        if col_idx != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Log-Sensitivity')
        
        if output not in aggregated_by_output:
            ax.text(0.5, 0.5, f"No data for {output}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue
        
        param_data = aggregated_by_output[output]
        param_baseline_data = baseline_by_output[output]
        
        if len(param_data) == 0:
            ax.text(0.5, 0.5, f"No data above threshold for {output}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue
        
        # Sort parameters by mean baseline sensitivity
        param_mean_baseline = {}
        for param in param_data.keys():
            baseline_data = param_baseline_data[param]
            param_mean_baseline[param] = np.mean([val for _, val in baseline_data]) if baseline_data else 0
        
        sorted_params = sorted(param_mean_baseline.keys(), 
                              key=lambda p: param_mean_baseline[p], 
                              reverse=True)
        
        # Prepare box data
        all_positions = []
        all_colors = []
        all_scenario_labels = []
        all_mins = []
        all_maxs = []
        all_means = []
        all_anatomy_data = []  # Store individual anatomy data
        
        x_position = 1
        param_positions = {}
        param_x_starts = {}
        
        n_scenarios_per_param = len(all_scenarios) + 1  # +1 for baseline
        
        for param_idx, param in enumerate(sorted_params):
            scenario_data = param_data[param]
            baseline_data = param_baseline_data[param]
            
            # Get group for this parameter
            param_group = get_parameter_group(param, xlabels_dict)
            
            # Store start position
            param_x_starts[param] = x_position
            
            # Calculate center position for this parameter
            param_center = x_position + (n_scenarios_per_param - 1) / 2.0
            param_positions[param] = param_center
            
            # Add baseline box
            if baseline_data:
                values = [val for _, val in baseline_data]
                all_positions.append(x_position)
                all_colors.append('#AA151B')
                all_scenario_labels.append('Baseline')
                all_mins.append(min(values))
                all_maxs.append(max(values))
                all_means.append(np.mean(values))
                all_anatomy_data.append(baseline_data)
                x_position += 1
            
            # Add scenario boxes
            for scenario_name in all_scenarios:
                if scenario_name in scenario_data and len(scenario_data[scenario_name]) > 0:
                    scenario_vals_with_anatomy = scenario_data[scenario_name]
                    values = [val for _, val in scenario_vals_with_anatomy]
                    all_positions.append(x_position)
                    all_colors.append(scenario_color_map.get(scenario_name, '#808080'))
                    all_scenario_labels.append(scenario_name)
                    all_mins.append(min(values))
                    all_maxs.append(max(values))
                    all_means.append(np.mean(values))
                    all_anatomy_data.append(scenario_vals_with_anatomy)
                x_position += 1
        
        # Plot boxes/patches
        if len(all_positions) > 0:
            for i, (pos, color, min_val, max_val, mean_val, anatomy_data, scenario) in enumerate(
                zip(all_positions, all_colors, all_mins, all_maxs, all_means, all_anatomy_data, all_scenario_labels)):
                
                # Draw box from min to max
                height = max_val - min_val
                rect = Rectangle((pos - box_width/2, min_val), box_width, height,
                                facecolor=color, alpha=1, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                
                # Overlay individual points (anatomies) with different markers
                for anatomy_name, val in anatomy_data:
                    marker = anatomy_markers[anatomy_name]
                    x = np.random.normal(pos, 0.05)
                    if scenario == 'Baseline':
                        ax.scatter(x, val, alpha=0.6, s=50, color='black', 
                                  marker=marker, zorder=4, edgecolor='black', linewidth=0.5)
                    else:
                        ax.scatter(x, val, alpha=0.6, s=50, color='black', 
                                  marker=marker, zorder=4, edgecolor='black', linewidth=0.5)
            
            # Add vertical separators between parameters
            for param in sorted_params:
                if param in param_x_starts:
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
        ax.set_title(output_title, fontsize=18, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.tick_params(labelsize=16)
        
        # X-axis labels: parameter names at center of each parameter group
        xtick_positions = []
        xtick_labels = []
        for param in sorted_params:
            if param in param_positions:
                xtick_positions.append(param_positions[param])
                latex_label = xlabels_dict.get(param, {}).get('latex', param)
                xtick_labels.append(latex_label)
        
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=15)
        
        # Add annotations
        if output in annotations:
            annotated_params = annotations[output]
            for param in annotated_params:
                if param in param_positions and param in param_baseline_data:
                    param_center = param_positions[param]
                    mean_baseline = np.mean([val for _, val in param_baseline_data[param]])
                    
                    param_label = xlabels_dict.get(param, {}).get('latex', param)
                    ax.annotate(param_label, xy=(param_center, mean_baseline),
                               xytext=(param_center + 0.5, mean_baseline * 2),
                               fontsize=fontsize-5, ha='left',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                                       alpha=0.6, edgecolor='black', linewidth=0.5),
                               arrowprops=dict(arrowstyle='->', lw=1, color='black', alpha=0.5))
    
    # Create legends
    # Legend 1: Scenario colors
    scenario_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#AA151B', alpha=1,
                     edgecolor='black', linewidth=1, label='Baseline')
    ]
    
    for scenario_name in all_scenarios:
        color = scenario_color_map.get(scenario_name, '#808080')
        label = scenario_labels.get(scenario_name, scenario_name)
        scenario_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=1,
                         edgecolor='black', linewidth=1, label=label)
        )
    
    legend1 = fig.legend(handles=scenario_handles, title="Treatment",
                        loc='upper center', bbox_to_anchor=(0.77, 0.35),
                        ncol=1, fontsize=20, frameon=False,
                        title_fontsize=21)
    legend1.get_title().set_fontweight('bold')
    
    # Legend 2: Anatomy markers
    anatomy_handles = []
    for anatomy_name in anatomy_names:
        marker = anatomy_markers[anatomy_name]
        anatomy_handles.append(
            plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='black',
                      markeredgecolor='black', markersize=10, label=anatomy_name, linewidth=0)
        )
    
    legend2 = fig.legend(handles=anatomy_handles, title="Heart",
                        loc='upper center', bbox_to_anchor=(0.92, 0.35),
                        ncol=1, fontsize=20, frameon=False,
                        title_fontsize=21)
    legend2.get_title().set_fontweight('bold')
    
    fig.add_artist(legend1)
    fig.add_artist(legend2)
    
    # Hide empty subplots
    for idx in range(n_outputs, len(axes)):
        axes[idx].set_visible(False)
    
    # Add supertitle
    if supertitle:
        fig.suptitle(supertitle, fontsize=30, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    
    # Save figure
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Scenario boxes plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity plots with scenarios as boxes/patches (baseline + modified)"
    )
    
    # Anatomy arguments
    parser.add_argument('--n_anatomies', type=int, required=True,
                       help='Number of anatomies')
    parser.add_argument('--anatomy_names', nargs='+', required=True,
                       help='Names for each anatomy')
    
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
    parser.add_argument('--figname', default='sensitivity_scenario_boxes.png',
                       help='Output figure name')
    parser.add_argument('--annotations', default=None,
                       help='Path to annotations JSON')
    parser.add_argument('--group_colors', default=None,
                       help='Path to group colors JSON')
    parser.add_argument('--scenario_colors', default=None,
                       help='Path to scenario colors JSON (format: {scenario_name: {"color": hex, "label": str}})')
    parser.add_argument('--supertitle', default=None,
                       help='Super title for figure')
    parser.add_argument('--gsa_mode', default='Si_total',
                       help='GSA mode')
    parser.add_argument('--mode', default='max',
                       help='Ranking mode')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Sensitivity threshold for inclusion')
    parser.add_argument('--exclusions', default=None,
                       help='Path to JSON file with exclusions')
    parser.add_argument('--box_width', type=float, default=0.7,
                       help='Width of boxes')
    
    # Parse known args first to get n_anatomies
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
    
    # Create box plots
    create_scenario_boxplot(
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
        scenario_colors_file=args.scenario_colors,
        supertitle=args.supertitle,
        threshold=args.threshold,
        exclusions_file=args.exclusions,
        box_width=args.box_width
    )


if __name__ == "__main__":
    main()