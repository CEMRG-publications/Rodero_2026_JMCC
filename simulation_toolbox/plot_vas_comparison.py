import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re


def load_xlabels_dict(xlabels_dict_file):
    """Load the xlabels dictionary with parameter metadata."""
    with open(xlabels_dict_file, 'r') as f:
        return json.load(f)


def load_ylabels_dict(ylabels_dict_file):
    """Load the ylabels dictionary with output metadata."""
    with open(ylabels_dict_file, 'r') as f:
        return json.load(f)


def load_exclusions(exclusions_file):
    """Load exclusions file mapping modified scenarios to parameters to exclude."""
    if exclusions_file and os.path.exists(exclusions_file):
        with open(exclusions_file, 'r') as f:
            return json.load(f)
    return {}


def extract_sensitivity_data(scenario_path, ylabels_raw_all, gsa_mode="Si_total", mode="max"):
    """Extract sensitivity data from a single scenario."""
    output_data = {}
    for ylabel_raw in ylabels_raw_all:
        output_data[ylabel_raw] = {}
        rank_file = os.path.join(scenario_path, f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
        if not os.path.exists(rank_file):
            rank_file = os.path.join(scenario_path, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")

        if os.path.exists(rank_file):
            with open(rank_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        param = parts[0]
                        sensitivity = float(parts[1])
                        output_data[ylabel_raw][param] = sensitivity
    return output_data


def compute_vas_metrics(anatomies_dict, ylabels_raw_all, exclusions, 
                        gsa_mode="Si_total", mode="max", threshold=0.05):
    """
    Compute VAS (Variability in Anatomical Sensitivity) metrics.
    
    For each output-parameter pair:
    1. Compute baseline range across anatomies
    2. For each modified scenario, compute the range across anatomies
    3. VAS = (modified_range - baseline_range) / baseline_range * 100
    4. For each output-scenario pair, take max(VAS) across all parameters
    
    Returns:
        vas_by_output: dict with structure output -> scenario -> max_vas
        detailed_vas: dict with structure output -> scenario -> parameter -> vas
    """
    anatomy_names = list(anatomies_dict.keys())
    
    # Get modified scenario names
    first_anatomy = list(anatomies_dict.values())[0]
    modified_names = [os.path.basename(path) for path in first_anatomy['modified']]
    n_scenarios = len(modified_names)
    
    # Extract all baseline data
    baseline_data = {}
    for anatomy_name, anatomy_info in anatomies_dict.items():
        baseline_data[anatomy_name] = extract_sensitivity_data(
            anatomy_info['baseline'], ylabels_raw_all, gsa_mode, mode
        )
    
    # Extract all modified scenario data
    modified_data = {scenario_name: {} for scenario_name in modified_names}
    for anatomy_name, anatomy_info in anatomies_dict.items():
        for i, modified_path in enumerate(anatomy_info['modified']):
            scenario_name = modified_names[i]
            modified_data[scenario_name][anatomy_name] = extract_sensitivity_data(
                modified_path, ylabels_raw_all, gsa_mode, mode
            )
    
    # Compute VAS for each output-parameter-scenario combination
    detailed_vas = defaultdict(lambda: defaultdict(dict))
    vas_by_output = defaultdict(dict)
    
    for output_name in ylabels_raw_all:
        # Collect all parameters for this output
        all_params = set()
        for anatomy_data in baseline_data.values():
            if output_name in anatomy_data:
                all_params.update(anatomy_data[output_name].keys())
        
        for scenario_name in modified_names:
            excluded_params = exclusions.get(scenario_name, [])
            max_vas = -np.inf
            max_vas_param = None
            
            for param in all_params:
                # Skip excluded parameters
                if param in excluded_params:
                    continue
                
                # Collect baseline values across anatomies
                baseline_values = []
                for anatomy_name in anatomy_names:
                    if output_name in baseline_data[anatomy_name]:
                        if param in baseline_data[anatomy_name][output_name]:
                            baseline_values.append(baseline_data[anatomy_name][output_name][param])
                
                # Collect modified values across anatomies
                modified_values = []
                for anatomy_name in anatomy_names:
                    if anatomy_name in modified_data[scenario_name]:
                        if output_name in modified_data[scenario_name][anatomy_name]:
                            if param in modified_data[scenario_name][anatomy_name][output_name]:
                                modified_values.append(modified_data[scenario_name][anatomy_name][output_name][param])
                
                # Only compute VAS if we have enough data
                if len(baseline_values) >= 2 and len(modified_values) >= 2:
                    # Check if any value exceeds threshold
                    if max(baseline_values) < threshold and max(modified_values) < threshold:
                        continue
                    
                    baseline_range = max(baseline_values) - min(baseline_values)
                    modified_range = max(modified_values) - min(modified_values)

                    
                    # Compute VAS (percentage change in range)
                    if baseline_range > 0:
                        vas = ((modified_range - baseline_range) / baseline_range) * 100
                        detailed_vas[output_name][scenario_name][param] = vas

                        
                        # Track maximum VAS for this output-scenario pair
                        if abs(vas) > abs(max_vas) or max_vas==-np.inf:
                            max_vas = vas
                            max_vas_param = param
            
            # Store the maximum VAS for this output-scenario pair
            if max_vas != -np.inf:
                vas_by_output[output_name][scenario_name] = {
                    'vas': max_vas,
                    'param': max_vas_param
                }
    
    return vas_by_output, detailed_vas, modified_names


def plot_vas_comparison(vas_by_output, modified_names, outputs, ylabels_dict, 
                       xlabels_dict, savepath, fontsize=12, 
                       figname="vas_comparison.png", supertitle=None):
    """
    Create multipanel figure showing VAS for each output.
    
    Each panel shows horizontal bars for each modified scenario,
    centered at x=0, extending left (negative VAS) or right (positive VAS).
    """
    n_outputs = len(outputs)
    n_cols = min(3, n_outputs)
    n_rows = int(np.ceil(n_outputs / n_cols))
    
    fig_width = 10 * n_cols
    fig_height = 8 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).flatten()
    
    # Determine global x-limits
    all_vas_values = []
    for output_data in vas_by_output.values():
        for scenario_data in output_data.values():
            all_vas_values.append(scenario_data['vas'])
    
    if len(all_vas_values) > 0:
        max_abs_vas = max(abs(v) for v in all_vas_values)
        next_hundred = 100*(1+int(max_abs_vas/100))+100
        xlim = (-next_hundred, next_hundred)
    else:
        xlim = (-100, 100)
    
    
    # Plot each output
    for idx, output in enumerate(outputs):
        ax = axes[idx]
        
        if output not in vas_by_output or len(vas_by_output[output]) == 0:
            print(f"{output} not found in {vas_by_output}")
            ax.text(0.5, 0.5, f"No data for {output}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=fontsize)
            ax.set_visible(True)
            continue
        
        output_data = vas_by_output[output]
        
        # Prepare data for plotting
        y_positions = np.arange(len(modified_names))
        vas_values = []
        colors = []
        
        for scenario_name in modified_names:
            if scenario_name in output_data:
                vas = output_data[scenario_name]['vas']
                vas_values.append(vas)
                # Color based on sign
                colors.append('#F1BF00' if vas > 0 else '#AA151B')
            else:
                vas_values.append(0)
                colors.append('#cccccc')
        
        # Create horizontal bar chart
        bars = ax.barh(y_positions, vas_values, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=0.5)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linewidth=2, linestyle='-', zorder=0)
        
        # Set x-limits (same for all panels)
        ax.set_xlim(xlim)
        
        if idx % n_cols == 0:
            # Y-axis: scenario names
            ax.set_yticks(y_positions)

            # Extract the variable name (anything between GSA_ and _lower)
            vars = [re.search(r"GSA_(.+?)_lower", s).group(1) for s in modified_names]

            # Extract the number after lower_
            lower_vals = [re.search(r"lower_([^_]+)", s).group(1) for s in modified_names]

            latex_friendly_vars = [xlabels_dict[var]['latex'] for var in vars]
            prefixes = ["↑ " if lv == "50.0" else "↓ " for lv in lower_vals]

            final_ylabel = [prefixes[i] + latex_friendly_vars[i] for i in range(len(prefixes))]


            ax.set_yticklabels(final_ylabel, fontsize=fontsize-2)
        

        # # X-axis label
        # ax.set_xlabel('VAS (%)', fontsize=fontsize)
        ax.set_xticks([])
        
        # Title
        output_title = ylabels_dict.get(output, {}).get('latex', output)
        ax.set_title(output_title, fontsize=fontsize+2, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.tick_params(labelsize=fontsize-2)
        
        # Add value labels on bars
        for i, (bar, vas) in enumerate(zip(bars, vas_values)):
            if vas != 0:
                # Position text inside or outside bar depending on length
                x_pos = vas + (3 if vas > 0 else -3)
                ha = 'left' if vas > 0 else 'right'
                ax.text(x_pos, i, f'{vas:.0f}%', va='center', ha=ha, 
                       fontsize=fontsize-3)
    
    # Hide empty subplots
    for idx in range(n_outputs, len(axes)):
        axes[idx].set_visible(False)
    
    # Add supertitle
    if supertitle:
        fig.suptitle(supertitle, fontsize=fontsize+8, fontweight='bold', y=0.995)
    
    # Add legend
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor='#d62728', alpha=0.7, edgecolor='black', label='Increased variability'),
    #     Patch(facecolor='#1f77b4', alpha=0.7, edgecolor='black', label='Decreased variability')
    # ]
    # fig.legend(handles=legend_elements, loc='upper right', 
    #           bbox_to_anchor=(0.98, 0.98), fontsize=fontsize, 
    #           frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"VAS comparison plot saved to: {output_path}")


def save_vas_summary(vas_by_output, detailed_vas, modified_names, ylabels_dict, 
                     xlabels_dict, savepath):
    """Save VAS results to text files."""
    os.makedirs(savepath, exist_ok=True)
    
    # Save summary
    summary_file = os.path.join(savepath, "vas_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("VAS (Variability in Anatomical Sensitivity) Summary\n")
        f.write("="*80 + "\n\n")
        
        for output in vas_by_output.keys():
            output_label = ylabels_dict.get(output, {}).get('latex', output)
            f.write(f"\nOutput: {output_label}\n")
            f.write("-"*80 + "\n")
            
            for scenario in modified_names:
                if scenario in vas_by_output[output]:
                    vas = vas_by_output[output][scenario]['vas']
                    param = vas_by_output[output][scenario]['param']
                    param_label = xlabels_dict.get(param, {}).get('latex', param)
                    f.write(f"  {scenario}: {vas:+.2f}% (max at parameter: {param_label})\n")
    
    print(f"VAS summary saved to: {summary_file}")
    
    # Save detailed VAS
    detailed_file = os.path.join(savepath, "vas_detailed.txt")
    with open(detailed_file, 'w') as f:
        f.write("Output\tScenario\tParameter\tVAS(%)\n")
        for output in detailed_vas.keys():
            output_label = ylabels_dict.get(output, {}).get('latex', output)
            for scenario in detailed_vas[output].keys():
                for param, vas in detailed_vas[output][scenario].items():
                    param_label = xlabels_dict.get(param, {}).get('latex', param)
                    f.write(f"{output_label}\t{scenario}\t{param_label}\t{vas:.2f}\n")
    
    print(f"Detailed VAS saved to: {detailed_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VAS (Variability in Anatomical Sensitivity) comparison plots"
    )
    
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
    parser.add_argument('--exclusions', default=None,
                       help='Path to exclusions JSON')
    parser.add_argument('--savepath', required=True,
                       help='Path to save figures')
    parser.add_argument('--fontsize', type=int, default=12,
                       help='Font size')
    parser.add_argument('--figname', default='vas_comparison.png',
                       help='Output figure name')
    parser.add_argument('--supertitle', default=None,
                       help='Super title for figure')
    parser.add_argument('--gsa_mode', default='Si_total',
                       help='GSA mode')
    parser.add_argument('--mode', default='max',
                       help='Ranking mode')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Sensitivity threshold for inclusion')
    
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
    exclusions = load_exclusions(args.exclusions)
    
    # Compute VAS metrics
    vas_by_output, detailed_vas, modified_names = compute_vas_metrics(
        anatomies_dict, args.outputs, exclusions, 
        args.gsa_mode, args.mode, args.threshold
    )
    
    # Plot VAS comparison
    plot_vas_comparison(
        vas_by_output, modified_names, args.outputs, ylabels_dict,
        xlabels_dict, args.savepath, args.fontsize, 
        args.figname, args.supertitle
    )
    
    # Save summary
    save_vas_summary(vas_by_output, detailed_vas, modified_names, 
                    ylabels_dict, xlabels_dict, args.savepath)
    
    print("\nVAS analysis complete!")


if __name__ == "__main__":
    main()