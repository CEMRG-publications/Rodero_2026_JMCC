import argparse
import json
import os
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


def analyze_functional_vs_anatomical(all_anatomies_data, xlabels_dict, threshold=0.05):
    """
    For each parameter-output pair:
    1. Compute the range of baseline sensitivities across anatomies
    2. Count how many modified scenario values fall outside this range
    3. Return detailed statistics
    """
    
    # Collect all unique parameters
    all_params = set()
    for anatomy_data in all_anatomies_data.values():
        for output_data in anatomy_data.values():
            all_params.update(output_data.keys())
    
    # Statistics storage
    results = {
        'by_param_output': {},  # param -> output -> stats
        'by_output': defaultdict(lambda: {'total_cases': 0, 'outside_range': 0, 'params_affected': set()}),
        'by_modified_scenario': defaultdict(lambda: {'total_cases': 0, 'outside_range': 0}),
        'by_param': defaultdict(lambda: {'total_cases': 0, 'outside_range': 0, 'outputs_affected': set()}),
        'global': {'total_cases': 0, 'outside_range': 0, 'params_with_outside': set()}
    }
    
    # Determine maximum number of modified scenarios across all data
    n_modified_scenarios = 0
    for anatomy_data in all_anatomies_data.values():
        for output_data in anatomy_data.values():
            for param_data in output_data.values():
                if param_data['modified']:
                    n_modified_scenarios = max(n_modified_scenarios, len(param_data['modified']))
    
    anatomy_names = list(all_anatomies_data.keys())
    
    # Iterate through each output
    for output_name in set().union(*[set(anatomy_data.keys()) for anatomy_data in all_anatomies_data.values()]):
        
        # Iterate through each parameter
        for param in all_params:
            # Collect baseline values across all anatomies for this param-output pair
            baseline_values = []
            modified_values_by_scenario = [[] for _ in range(n_modified_scenarios)]
            
            for anatomy_name in anatomy_names:
                if output_name in all_anatomies_data[anatomy_name]:
                    if param in all_anatomies_data[anatomy_name][output_name]:
                        data = all_anatomies_data[anatomy_name][output_name][param]
                        
                        if data['baseline'] is not None:
                            baseline_values.append(data['baseline'])
                        
                        # Collect modified values by scenario index
                        for i, mod_val in enumerate(data['modified']):
                            if i < len(modified_values_by_scenario):
                                modified_values_by_scenario[i].append(mod_val)
            
            # Only analyze if we have baseline data and modified data
            if len(baseline_values) > 0 and any(len(mv) > 0 for mv in modified_values_by_scenario):
                # Check threshold: include if baseline OR any modified exceeds threshold
                max_baseline = max(baseline_values)
                max_modified_overall = max([max(mv) for mv in modified_values_by_scenario if len(mv) > 0])
                
                if max_baseline < threshold and max_modified_overall < threshold:
                    continue  # Skip if below threshold
                
                try:
                    group = get_parameter_group(param, xlabels_dict)
                except Exception as e:
                    print(f"Warning: {e}")
                    continue
                
                # Compute baseline range
                baseline_min = min(baseline_values)
                baseline_max = max(baseline_values)
                
                # Count how many modified values fall outside this range
                outside_count = 0
                total_modified_count = 0
                
                for scenario_idx, mod_vals in enumerate(modified_values_by_scenario):
                    for mod_val in mod_vals:
                        total_modified_count += 1
                        results['global']['total_cases'] += 1
                        results['by_output'][output_name]['total_cases'] += 1
                        results['by_param'][param]['total_cases'] += 1
                        results['by_modified_scenario'][scenario_idx]['total_cases'] += 1
                        
                        if mod_val < baseline_min or mod_val > baseline_max:
                            outside_count += 1
                            results['global']['outside_range'] += 1
                            results['by_output'][output_name]['outside_range'] += 1
                            results['by_param'][param]['outside_range'] += 1
                            results['by_modified_scenario'][scenario_idx]['outside_range'] += 1
                            
                            # Track which params/outputs are affected
                            results['global']['params_with_outside'].add(param)
                            results['by_output'][output_name]['params_affected'].add(param)
                            results['by_param'][param]['outputs_affected'].add(output_name)
                
                # Store detailed results for this param-output pair
                if param not in results['by_param_output']:
                    results['by_param_output'][param] = {}
                
                results['by_param_output'][param][output_name] = {
                    'baseline_range': (baseline_min, baseline_max),
                    'baseline_values': baseline_values,
                    'total_modified': total_modified_count,
                    'outside_range': outside_count,
                    'percentage': 100 * outside_count / total_modified_count if total_modified_count > 0 else 0,
                    'group': group
                }
    
    return results, n_modified_scenarios


def print_results(results, n_modified_scenarios, anatomies_dict, xlabels_dict, ylabels_dict, savepath=None):
    """Print formatted results of the analysis and optionally save to files."""
    
    # Prepare output strings
    output_lines = []
    
    output_lines.append("="*80)
    output_lines.append("FUNCTIONAL vs ANATOMICAL VARIABILITY ANALYSIS")
    output_lines.append("="*80)
    
    # Get modified scenario names
    first_anatomy = list(anatomies_dict.values())[0]
    modified_names = [os.path.basename(path) for path in first_anatomy['modified']]
    
    # Global summary
    output_lines.append("")
    output_lines.append("-"*80)
    output_lines.append("GLOBAL SUMMARY")
    output_lines.append("-"*80)
    total_cases = results['global']['total_cases']
    outside_cases = results['global']['outside_range']
    percentage = 100 * outside_cases / total_cases if total_cases > 0 else 0
    
    output_lines.append(f"Total modified scenario measurements: {total_cases}")
    output_lines.append(f"Cases outside baseline range: {outside_cases} ({percentage:.2f}%)")
    output_lines.append(f"Number of parameters with at least one case outside range: {len(results['global']['params_with_outside'])}")
    
    # By output
    output_lines.append("")
    output_lines.append("-"*80)
    output_lines.append("BY OUTPUT")
    output_lines.append("-"*80)
    
    # Sort outputs by percentage outside range
    output_stats = []
    for output_name, stats in results['by_output'].items():
        pct = 100 * stats['outside_range'] / stats['total_cases'] if stats['total_cases'] > 0 else 0
        output_stats.append((output_name, stats, pct))
    output_stats.sort(key=lambda x: x[2], reverse=True)
    
    for output_name, stats, pct in output_stats:
        output_label = ylabels_dict.get(output_name, {}).get('latex', output_name)
        output_lines.append(f"\n{output_label}:")
        output_lines.append(f"  Cases outside range: {stats['outside_range']} / {stats['total_cases']} ({pct:.2f}%)")
        output_lines.append(f"  Parameters affected: {len(stats['params_affected'])}")
    
    # By modified scenario
    output_lines.append("")
    output_lines.append("-"*80)
    output_lines.append("BY MODIFIED SCENARIO")
    output_lines.append("-"*80)
    
    # Sort scenarios by percentage outside range
    scenario_stats = []
    for scenario_idx, stats in results['by_modified_scenario'].items():
        pct = 100 * stats['outside_range'] / stats['total_cases'] if stats['total_cases'] > 0 else 0
        scenario_name = modified_names[scenario_idx] if scenario_idx < len(modified_names) else f"Scenario {scenario_idx}"
        scenario_stats.append((scenario_name, stats, pct))
    scenario_stats.sort(key=lambda x: x[2], reverse=True)
    
    for scenario_name, stats, pct in scenario_stats:
        output_lines.append(f"\n{scenario_name}:")
        output_lines.append(f"  Cases outside range: {stats['outside_range']} / {stats['total_cases']} ({pct:.2f}%)")
    
    # By parameter
    output_lines.append("")
    output_lines.append("-"*80)
    output_lines.append("BY PARAMETER (Top 20)")
    output_lines.append("-"*80)
    
    # Sort parameters by percentage outside range
    param_stats = []
    for param, stats in results['by_param'].items():
        pct = 100 * stats['outside_range'] / stats['total_cases'] if stats['total_cases'] > 0 else 0
        param_stats.append((param, stats, pct))
    param_stats.sort(key=lambda x: x[2], reverse=True)
    
    for param, stats, pct in param_stats[:20]:
        param_label = xlabels_dict.get(param, {}).get('latex', param)
        output_lines.append(f"\n{param_label}:")
        output_lines.append(f"  Cases outside range: {stats['outside_range']} / {stats['total_cases']} ({pct:.2f}%)")
        output_lines.append(f"  Outputs affected: {len(stats['outputs_affected'])}")
    
    # Detailed parameter-output breakdown (top cases)
    output_lines.append("")
    output_lines.append("-"*80)
    output_lines.append("PARAMETER-OUTPUT PAIRS WITH HIGHEST VARIABILITY (Top 20)")
    output_lines.append("-"*80)
    
    all_pairs = []
    for param, output_dict in results['by_param_output'].items():
        for output_name, stats in output_dict.items():
            all_pairs.append((param, output_name, stats))
    
    # Sort by percentage
    all_pairs.sort(key=lambda x: x[2]['percentage'], reverse=True)
    
    for param, output_name, stats in all_pairs[:20]:
        param_label = xlabels_dict.get(param, {}).get('latex', param)
        output_label = ylabels_dict.get(output_name, {}).get('latex', output_name)
        
        output_lines.append(f"\n{param_label} → {output_label}:")
        output_lines.append(f"  Baseline range: [{stats['baseline_range'][0]:.6f}, {stats['baseline_range'][1]:.6f}]")
        output_lines.append(f"  Cases outside range: {stats['outside_range']} / {stats['total_modified']} ({stats['percentage']:.2f}%)")
        output_lines.append(f"  Parameter group: {stats['group']}")
    
    output_lines.append("")
    output_lines.append("="*80)
    
    # Print to console
    # for line in output_lines:
    #     print(line)
    
    # Save to files if savepath is provided
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        
        # Save complete report
        report_file = os.path.join(savepath, "functional_vs_anatomical_complete_report.txt")
        with open(report_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nComplete report saved to: {report_file}")
        
        # Save by-output summary as CSV-like format
        output_summary_file = os.path.join(savepath, "by_output_summary.txt")
        with open(output_summary_file, 'w') as f:
            f.write("Output\tCases_Outside\tTotal_Cases\tPercentage\tParams_Affected\n")
            for output_name, stats, pct in output_stats:
                output_label = ylabels_dict.get(output_name, {}).get('latex', output_name)
                f.write(f"{output_label}\t{stats['outside_range']}\t{stats['total_cases']}\t{pct:.2f}\t{len(stats['params_affected'])}\n")
        print(f"Output summary saved to: {output_summary_file}")
        
        # Save by-scenario summary
        scenario_summary_file = os.path.join(savepath, "by_scenario_summary.txt")
        with open(scenario_summary_file, 'w') as f:
            f.write("Scenario\tCases_Outside\tTotal_Cases\tPercentage\n")
            for scenario_name, stats, pct in scenario_stats:
                f.write(f"{scenario_name}\t{stats['outside_range']}\t{stats['total_cases']}\t{pct:.2f}\n")
        print(f"Scenario summary saved to: {scenario_summary_file}")
        
        # Save by-parameter summary (all parameters, not just top 20)
        param_summary_file = os.path.join(savepath, "by_parameter_summary.txt")
        with open(param_summary_file, 'w') as f:
            f.write("Parameter\tCases_Outside\tTotal_Cases\tPercentage\tOutputs_Affected\n")
            for param, stats, pct in param_stats:
                param_label = xlabels_dict.get(param, {}).get('latex', param)
                f.write(f"{param_label}\t{stats['outside_range']}\t{stats['total_cases']}\t{pct:.2f}\t{len(stats['outputs_affected'])}\n")
        print(f"Parameter summary saved to: {param_summary_file}")
        
        # Save detailed parameter-output pairs (all pairs, not just top 20)
        detailed_file = os.path.join(savepath, "parameter_output_detailed.txt")
        with open(detailed_file, 'w') as f:
            f.write("Parameter\tOutput\tBaseline_Min\tBaseline_Max\tCases_Outside\tTotal_Cases\tPercentage\tGroup\n")
            for param, output_name, stats in all_pairs:
                param_label = xlabels_dict.get(param, {}).get('latex', param)
                output_label = ylabels_dict.get(output_name, {}).get('latex', output_name)
                f.write(f"{param_label}\t{output_label}\t{stats['baseline_range'][0]:.6f}\t{stats['baseline_range'][1]:.6f}\t")
                f.write(f"{stats['outside_range']}\t{stats['total_modified']}\t{stats['percentage']:.2f}\t{stats['group']}\n")
        print(f"Detailed parameter-output pairs saved to: {detailed_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze functional vs anatomical variability in sensitivity analysis"
    )
    
    # Anatomy arguments
    parser.add_argument('--n_anatomies', type=int, required=True,
                       help='Number of anatomies')
    parser.add_argument('--anatomy_names', nargs='+', required=True,
                       help='Names for each anatomy')
    parser.add_argument('--outputs', nargs='+', required=True,
                       help='Output names to analyze')
    parser.add_argument('--xlabels_dict', required=True,
                       help='Path to xlabels dictionary JSON')
    parser.add_argument('--ylabels_dict', required=True,
                       help='Path to ylabels dictionary JSON')
    parser.add_argument('--gsa_mode', default='Si_total',
                       help='GSA mode')
    parser.add_argument('--mode', default='max',
                       help='Ranking mode')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Sensitivity threshold for inclusion')
    parser.add_argument('--exclusions', default=None,
                       help='Path to JSON file with exclusions')
    parser.add_argument('--savepath', required=True,
                       help='Path to save result files')
    
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
    
    # Load exclusions
    exclusions = load_exclusions(args.exclusions)
    if exclusions:
        print(f"Loaded exclusions for {len(exclusions)} modified scenarios")
    
    # Get modified scenario names
    first_anatomy = list(anatomies_dict.values())[0]
    modified_names = [os.path.basename(path) for path in first_anatomy['modified']]
    
    # Extract all data
    print("Extracting sensitivity data from all anatomies...")
    all_anatomies_data = extract_all_anatomies_data(
        anatomies_dict, args.outputs, modified_names, exclusions, 
        args.gsa_mode, args.mode
    )
    
    # Perform analysis
    print("Analyzing functional vs anatomical variability...")
    results, n_modified_scenarios = analyze_functional_vs_anatomical(
        all_anatomies_data, xlabels_dict, args.threshold
    )
    
    # Print results
    print_results(results, n_modified_scenarios, anatomies_dict, xlabels_dict, ylabels_dict, args.savepath)


if __name__ == "__main__":
    main()