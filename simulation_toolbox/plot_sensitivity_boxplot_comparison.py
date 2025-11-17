import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches


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


def get_all_relevant_params(all_anatomies_data, outputs, threshold=0.05):
    """
    Get the union of all parameters that are relevant for any output across all anatomies.
    A parameter is relevant if baseline OR any modified scenario exceeds threshold.
    """
    all_params = set()
    
    for anatomy_name, anatomy_data in all_anatomies_data.items():
        for output_name in outputs:
            if output_name not in anatomy_data:
                continue
            param_data = anatomy_data[output_name]
            
            for param, values in param_data.items():
                baseline_val = values['baseline']
                modified_vals = values['modified']
                
                if baseline_val is not None and len(modified_vals) > 0:
                    max_modified = max(modified_vals) if len(modified_vals) > 0 else 0
                    if baseline_val >= threshold or max_modified >= threshold:
                        all_params.add(param)
    
    return sorted(all_params)

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

    Position logic changed to guarantee symmetric padding:
    - gap_between_params is interpreted as the total empty horizontal spacing between
      two consecutive parameter groups (in the same coordinate units as slot_width).
    - The visible padding between the last marker of one group and the separator,
      and between the separator and the first marker of the next group, will both be
      gap_between_params / 2.0.

    All other behavior mirrors the original function (data extraction, aggregation,
    plotting of boxpoints and baseline markers, legends, etc.).
    """
    from collections import defaultdict

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

    # Aggregate data per output -> param -> list of anatomy entries
    print("Aggregating data for comparison...")
    aggregated_by_output = defaultdict(lambda: defaultdict(list))

    # Collect all unique parameters across anatomies (raw union)
    all_params = set()
    for anatomy_data in all_anatomies_data.values():
        for output_data in anatomy_data.values():
            all_params.update(output_data.keys())

    # Build aggregated_by_output (same inclusion/threshold logic as before)
    for anatomy_name, anatomy_data in all_anatomies_data.items():
        for output_name, param_data in anatomy_data.items():
            for param in all_params:
                if param in param_data:
                    baseline_val = param_data[param]['baseline']
                    modified_vals = param_data[param]['modified']
                    if baseline_val is not None and len(modified_vals) > 0:
                        max_modified = max(modified_vals) if len(modified_vals) > 0 else 0
                        if baseline_val >= threshold or max_modified >= threshold:
                            try:
                                group = get_parameter_group(param, xlabels_dict)
                            except Exception as e:
                                print(f"Warning: {e}")
                                continue
                            aggregated_by_output[output_name][param].append({
                                'anatomy': anatomy_name,
                                'baseline': baseline_val,
                                'modified': modified_vals,
                                'group': group
                            })

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

    # Anatomy names and styles
    anatomy_names = list(anatomies_dict.keys())
    n_anatomies = len(anatomy_names)
    available_markers = ['o', 's', '^', 'D', 'v']
    available_colors = ['#AA151B', '#AA151B', '#AA151B', '#AA151B', '#AA151B']

    anatomy_styles = {}
    for i, anatomy_name in enumerate(anatomy_names):
        anatomy_styles[anatomy_name] = {
            'marker': available_markers[i % len(available_markers)],
            'color': available_colors[i % len(available_colors)]
        }

    # Collect all groups present (for color map)
    all_groups = set()
    for output_data in aggregated_by_output.values():
        for anatomy_list in output_data.values():
            for entry in anatomy_list:
                all_groups.add(entry['group'])
    all_groups = sorted(all_groups)

    # Create color map for groups
    color_map = {}
    if group_colors_custom:
        for group in all_groups:
            color_map[group] = group_colors_custom.get(group, {}).get('color', '#808080')
    else:
        for i, group in enumerate(all_groups):
            color_map[group] = '#808080'

    # --- GLOBAL PARAM ORDER (same across panels) ---
    # Compute mean baseline sensitivity across all anatomies/outputs for ordering
    param_mean_sensitivity = {}
    for param in all_params:
        sensitivities = []
        for anatomy_name, anatomy_data in all_anatomies_data.items():
            for output_name in outputs:
                if output_name in anatomy_data and param in anatomy_data[output_name]:
                    baseline_val = anatomy_data[output_name][param]['baseline']
                    if baseline_val is not None:
                        sensitivities.append(baseline_val)
        param_mean_sensitivity[param] = np.mean(sensitivities) if sensitivities else 0.0

    global_sorted_params = sorted(list(all_params), key=lambda p: param_mean_sensitivity[p], reverse=True)
    # ---------------------------------------------------------------------------

    # Determine relevant params = those that are in aggregated_by_output for at least one output
    relevant_params = [p for p in global_sorted_params if any(p in aggregated_by_output[out] for out in outputs)]

    # ---- IMPORTANT CHANGE: Only display (reserve slots for) relevant_params ----
    display_params = relevant_params  # use this for slotting, sizing, xticks
    # ---------------------------------------------------------------------------

    # Layout parameters for slotting
    slots_per_param = n_anatomies             # reserve exactly one slot per anatomy for each displayed parameter
    gap_between_params = 1.0                  # total empty spacing between groups (tweakable)
    slot_width = 1.0                          # distance between adjacent anatomy slots
    base_start = 1.5                          # left offset for the first group's first slot

    # Compute param start positions and centers globally based on display_params
    param_start_pos = {}   # param -> start slot (float)
    param_center_pos = {}  # param -> center slot (float)
    for idx, param in enumerate(display_params):
        # use ((slots_per_param - 1) * slot_width + gap_between_params) as the increment
        # so that the distance between last marker of group i and first marker of group i+1 == gap_between_params
        start = base_start + idx * ((slots_per_param - 1) * slot_width + gap_between_params)
        center = start + ((slots_per_param - 1) * slot_width) / 2.0
        param_start_pos[param] = start
        param_center_pos[param] = center

    # Global x extent based on display_params (end of last group's last slot, plus half-gap padding)
    if len(display_params) > 0:
        last_start = param_start_pos[display_params[-1]]
        global_max_pos = last_start + ((slots_per_param - 1) * slot_width) + gap_between_params / 2.0
    else:
        global_max_pos = 0

    # --- Figure sizing: base width on number of displayed parameters (not all global params) ---
    n_outputs = len(outputs)
    if len(display_params) > 0:
        fig_width = 24
    else:
        fig_width = 10
    fig_height = 3 * n_outputs

    fig, axes = plt.subplots(n_outputs, 1, figsize=(fig_width, fig_height))
    if n_outputs == 1:
        axes = [axes]

    # Plot each output
    for output_idx, output in enumerate(outputs):
        ax = axes[output_idx]

        if output not in aggregated_by_output or len(aggregated_by_output[output]) == 0:
            ax.text(0.5, 0.5, f"No data above threshold for {output}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue

        param_data = aggregated_by_output[output]

        # Prepare lists for boxplot positions/data
        all_boxplot_data = []
        all_positions = []
        all_colors = []
        all_baseline_positions = []
        all_baseline_values = []
        all_baseline_anatomy_labels = []

        # For annotation and local plotting we use param_center_pos (only for displayed params)
        param_local_centers = {}

        # Build positions by iterating display_params and reserving slots per anatomy
        for param in display_params:
            start = param_start_pos[param]
            param_local_centers[param] = param_center_pos[param]

            # For each anatomy (fixed order), compute slot and add data if present
            for a_idx, anatomy_name in enumerate(anatomy_names):
                slot_pos = start + a_idx * slot_width  # one slot per anatomy in order

                # find entry for this anatomy and param (if any)
                entry = None
                if param in param_data:
                    for e in param_data[param]:
                        if e['anatomy'] == anatomy_name:
                            entry = e
                            break

                if entry is not None:
                    modified_vals = entry['modified']
                    baseline_val = entry['baseline']

                    # Add box data
                    all_boxplot_data.append(modified_vals)
                    all_positions.append(slot_pos)
                    all_colors.append(color_map[entry['group']])

                    # Record baseline marker
                    all_baseline_positions.append(slot_pos)
                    all_baseline_values.append(baseline_val)
                    all_baseline_anatomy_labels.append(anatomy_name)
                else:
                    # slot reserved but empty for this anatomy/param
                    pass

        # If no boxplot data for this output (shouldn't happen given earlier check), skip
        if len(all_boxplot_data) == 0:
            ax.text(0.5, 0.5, f"No data above threshold for {output}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue

        # Plot modified scenario points as small dots (overlaid) + boxplots
        for vals, pos in zip(all_boxplot_data, all_positions):
            if len(vals) == 0:
                continue
            x_jitter = np.random.normal(pos, 0.08, size=len(vals))
            ax.scatter(x_jitter, vals, alpha=0.6, s=25, marker='o', zorder=3,
                       edgecolor='black', linewidth=0.3, color='black')

        # Plot baseline markers using anatomy-specific markers/colors
        for pos, val, anat in zip(all_baseline_positions, all_baseline_values, all_baseline_anatomy_labels):
            marker = anatomy_styles[anat]['marker']
            color = anatomy_styles[anat]['color']
            ax.scatter(pos, val, color=color, marker=marker, s=150, zorder=4,
                       alpha=0.85, edgecolor='black', linewidth=1.0)

        # Add vertical separators between parameter blocks.
        # Draw the separator halfway into the gap_before each group so padding is symmetric.
        for param in display_params:
            start = param_start_pos[param]
            sep_x = start - (gap_between_params / 2.0)
            ax.axvline(x=sep_x, color='gray', linestyle=':', alpha=0.25, linewidth=1)

        # Add threshold line
        ax.axhline(y=threshold, color='#AA151B', linestyle='-', alpha=0.3, linewidth=2)

        # Set y-axis to log scale
        ax.set_yscale('log')
        ax.set_ylim(global_ymin, global_ymax)

        # Set title and grid
        output_title = output
        if output in ylabels_dict:
            output_title = ylabels_dict[output].get('latex', output)
        ax.set_title(output_title, fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.tick_params(labelsize=16)

        # X-axis ticks/labels: use centers for display_params; show labels only on bottom panel
        xtick_positions = [param_center_pos[p] for p in display_params]
        xtick_labels = [xlabels_dict.get(p, {}).get('latex', p) for p in display_params]

        if output_idx == n_outputs - 1:
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=15)
        else:
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels([])

        # Set x limits to cover full reserved range (including half-gap padding at right)
        if global_max_pos > 0:
            ax.set_xlim(0.5, global_max_pos + 0.5)
        else:
            if len(all_positions) > 0:
                ax.set_xlim(0.5, max(all_positions) + 0.5)

        # Add annotations if requested (use param_local_centers)
        if output in annotations:
            annotated_params = annotations[output]
            for param in annotated_params:
                if param in param_local_centers:
                    param_center = param_local_centers[param]
                    if param in param_data:
                        anatomy_list = param_data[param]
                        mean_baseline = np.mean([e['baseline'] for e in anatomy_list])
                        param_label = xlabels_dict.get(param, {}).get('latex', param)
                        ax.annotate(param_label, xy=(param_center, mean_baseline),
                                   xytext=(param_center + 0.5, mean_baseline * 2),
                                   fontsize=fontsize - 4, ha='left',
                                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                                             alpha=0.6, edgecolor='black', linewidth=0.5),
                                   arrowprops=dict(arrowstyle='->', lw=1, color='black', alpha=0.5))


    # --- ANATOMY LEGEND (markers only, black color) ---
    anatomy_handles = []
    for anatomy_name in anatomy_names:
        marker = anatomy_styles[anatomy_name]['marker']
        handle = plt.Line2D([0], [0], marker=marker, color='w',
                            markerfacecolor='black',  # force black in legend
                            markeredgecolor='black',
                            markersize=12, label=anatomy_name, linewidth=0)
        anatomy_handles.append(handle)

    if anatomy_handles:
        legend_anat = fig.legend(handles=anatomy_handles, title="Anatomies",
                                loc='upper right', bbox_to_anchor=(0.98, 0.98),
                                ncol=1, fontsize=20, frameon=True,
                                title_fontsize=21)
        legend_anat.get_title().set_fontweight('bold')

    # --- SCENARIO TYPE LEGEND ---
    scenario_handles = [
        mpatches.Patch(color='#AA151B', label='Baseline'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                markersize=10, label='Functional remodeling')
    ]

    legend_scenario = fig.legend(handles=scenario_handles, title="Scenario type",
                                loc='upper right', bbox_to_anchor=(0.98, 0.86),
                                ncol=1, fontsize=20, frameon=True,
                                title_fontsize=21)
    legend_scenario.get_title().set_fontweight('bold')


    # Add supertitle
    if supertitle:
        fig.suptitle(supertitle, fontsize=30, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

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