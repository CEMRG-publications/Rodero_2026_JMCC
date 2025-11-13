import argparse
import json
import os
import numpy as np
import pandas as pd
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


def extract_baseline_sensitivity_data(scenarios, ylabels_raw_all, anatomy_names, gsa_mode="Si_total", mode="max", threshold=0.05):
    """
    Extract baseline sensitivity data for all parameters and outputs across anatomies.
    Returns nested dictionary: parameter -> output -> {values: [], anatomy_names: []}
    Only includes parameter-output pairs where at least one anatomy exceeds threshold.
    """
    # First pass: collect all data
    temp_data = {}  # output -> parameter -> list of (anatomy_idx, sensitivity)
    
    for ylabel_raw in ylabels_raw_all:
        temp_data[ylabel_raw] = defaultdict(list)
        
        for scenario_idx, scenario in enumerate(scenarios):
            rank_file = os.path.join(scenario, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
            
            if not os.path.exists(rank_file):
                print(f"Warning: File not found: {rank_file}")
                continue
            
            with open(rank_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        param = parts[0]
                        sensitivity = float(parts[1])
                        temp_data[ylabel_raw][param].append((scenario_idx, sensitivity))
    
    # Reorganize: parameter -> output -> {values, anatomy_names}
    param_data = defaultdict(lambda: defaultdict(lambda: {'values': [], 'anatomy_names': []}))
    
    for output_name, params in temp_data.items():
        for param, data_list in params.items():
            # Check threshold
            values = [v for _, v in data_list]
            if len(values) > 0 and max(values) >= threshold:
                # Sort by anatomy index to maintain order
                data_list.sort(key=lambda x: x[0])
                
                for anat_idx, sensitivity in data_list:
                    param_data[param][output_name]['values'].append(sensitivity)
                    param_data[param][output_name]['anatomy_names'].append(anatomy_names[anat_idx])
    
    return param_data


def compute_statistics(values):
    """Compute statistics for a list of values."""
    if len(values) == 0:
        return None
    
    return {
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'range': np.max(values) - np.min(values)
    }


def format_range_string(stats):
    """Format range as [min, max]."""
    return f"[{stats['min']:.4f}, {stats['max']:.4f}]"


def format_mean_std_string(stats):
    """Format as mean±std."""
    return f"{stats['mean']:.4f}±{stats['std']:.4f}"


def print_and_save_parameter_summaries(param_data, xlabels_dict, ylabels_dict, savepath, output_groups=None):
    """
    Print human-readable summaries for each parameter.
    Format: "Parameter name range for Output is [min, max] or mean±std"
    """
    os.makedirs(savepath, exist_ok=True)
    
    output_lines = []
    csv_data = []
    
    output_lines.append("="*120)
    output_lines.append("BASELINE SENSITIVITY SUMMARY: PARAMETER-OUTPUT RANGES")
    output_lines.append("="*120)
    output_lines.append("")
    
    # Sort parameters by total number of outputs affected
    param_output_counts = [(param, len(outputs)) for param, outputs in param_data.items()]
    param_output_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Print each parameter
    for param, n_outputs in param_output_counts:
        param_label = xlabels_dict.get(param, {}).get('latex', param)
        
        try:
            group = get_parameter_group(param, xlabels_dict)
        except Exception as e:
            print(f"Warning: {e}")
            continue
        
        output_lines.append("-"*120)
        output_lines.append(f"{param_label} ({param}) - Group: {group}")
        output_lines.append("-"*120)
        
        # Get all outputs for this parameter, sorted by mean sensitivity
        outputs_list = []
        for output_name, data in param_data[param].items():
            stats = compute_statistics(data['values'])
            if stats:
                outputs_list.append((output_name, stats, data))
        
        outputs_list.sort(key=lambda x: x[1]['mean'], reverse=True)
        
        # Print individual output ranges
        for output_name, stats, data in outputs_list:
            output_label = ylabels_dict.get(output_name, {}).get('latex', output_name)
            range_str = format_range_string(stats)
            mean_std_str = format_mean_std_string(stats)
            
            line = f"  • {param_label} range for {output_label} is {range_str} or {mean_std_str}"
            output_lines.append(line)
            
            # Add detailed anatomy values
            anatomy_details = ", ".join([f"{aname}: {val:.4f}" for aname, val in 
                                        zip(data['anatomy_names'], data['values'])])
            output_lines.append(f"    Anatomies: {anatomy_details}")
            
            # Store for CSV
            csv_data.append({
                'Parameter': param,
                'Parameter_Label': param_label,
                'Group': group,
                'Output': output_name,
                'Output_Label': output_label,
                'Range_Min': stats['min'],
                'Range_Max': stats['max'],
                'Range': stats['range'],
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Median': stats['median'],
                'N_Anatomies': len(data['values'])
            })
        
        output_lines.append("")
    
    # Print grouped summaries if output_groups provided
    if output_groups:
        output_lines.append("")
        output_lines.append("="*120)
        output_lines.append("BASELINE SENSITIVITY SUMMARY: PARAMETER RANGES ACROSS OUTPUT GROUPS")
        output_lines.append("="*120)
        output_lines.append("")
        
        grouped_csv_data = []
        
        for param, outputs_dict in param_data.items():
            param_label = xlabels_dict.get(param, {}).get('latex', param)
            
            try:
                group = get_parameter_group(param, xlabels_dict)
            except Exception as e:
                continue
            
            # For each output group
            for group_name, group_outputs in output_groups.items():
                # Collect values for outputs in this group
                all_values = []
                included_outputs = []
                
                for output_name in group_outputs:
                    if output_name in outputs_dict:
                        all_values.extend(outputs_dict[output_name]['values'])
                        output_label = ylabels_dict.get(output_name, {}).get('latex', output_name)
                        included_outputs.append(output_label)
                
                if len(all_values) > 0:
                    stats = compute_statistics(all_values)
                    range_str = format_range_string(stats)
                    mean_std_str = format_mean_std_string(stats)
                    
                    if param not in [line for line in output_lines if line.startswith(f"  {param_label}")]:
                        output_lines.append(f"{param_label} ({param}) - Group: {group}")
                    
                    outputs_str = ", ".join(included_outputs)
                    line = f"  • {param_label} sensitivities for {group_name} outputs ({outputs_str}) is {range_str} or {mean_std_str}"
                    output_lines.append(line)
                    
                    grouped_csv_data.append({
                        'Parameter': param,
                        'Parameter_Label': param_label,
                        'Parameter_Group': group,
                        'Output_Group': group_name,
                        'Outputs_Included': outputs_str,
                        'N_Outputs': len(included_outputs),
                        'Range_Min': stats['min'],
                        'Range_Max': stats['max'],
                        'Range': stats['range'],
                        'Mean': stats['mean'],
                        'Std': stats['std'],
                        'Median': stats['median'],
                        'N_Values': len(all_values)
                    })
        
        output_lines.append("")
        
        # Save grouped CSV
        if grouped_csv_data:
            grouped_df = pd.DataFrame(grouped_csv_data)
            grouped_df = grouped_df.sort_values(['Parameter', 'Output_Group'])
            grouped_file = os.path.join(savepath, "baseline_parameter_grouped_ranges.csv")
            grouped_df.to_csv(grouped_file, index=False, float_format='%.6f')
            print(f"Grouped ranges saved to: {grouped_file}")
    
    # Save text output
    text_file = os.path.join(savepath, "baseline_parameter_output_ranges.txt")
    with open(text_file, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"Text summary saved to: {text_file}")
    
    # Save CSV
    csv_df = pd.DataFrame(csv_data)
    csv_df = csv_df.sort_values(['Parameter', 'Mean'], ascending=[True, False])
    csv_file = os.path.join(savepath, "baseline_parameter_output_ranges.csv")
    csv_df.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"CSV data saved to: {csv_file}")
    
    # Print to console
    for line in output_lines:
        print(line)


def print_output_summaries(param_data, xlabels_dict, ylabels_dict, savepath):
    """
    Print summaries organized by output (instead of by parameter).
    Format: "For Output: Parameter1 range is [min, max] or mean±std, Parameter2 range is..."
    """
    
    # Reorganize: output -> parameter -> data
    output_data = defaultdict(dict)
    for param, outputs_dict in param_data.items():
        for output_name, data in outputs_dict.items():
            output_data[output_name][param] = data
    
    output_lines = []
    output_lines.append("")
    output_lines.append("="*120)
    output_lines.append("BASELINE SENSITIVITY SUMMARY: OUTPUT-PARAMETER RANGES")
    output_lines.append("="*120)
    output_lines.append("")
    
    csv_data = []
    
    # Sort outputs by number of parameters
    output_param_counts = [(output, len(params)) for output, params in output_data.items()]
    output_param_counts.sort(key=lambda x: x[1], reverse=True)
    
    for output_name, n_params in output_param_counts:
        output_label = ylabels_dict.get(output_name, {}).get('latex', output_name)
        
        output_lines.append("-"*120)
        output_lines.append(f"{output_label} ({output_name})")
        output_lines.append("-"*120)
        
        # Get all parameters for this output, sorted by mean sensitivity
        params_list = []
        for param, data in output_data[output_name].items():
            stats = compute_statistics(data['values'])
            if stats:
                params_list.append((param, stats, data))
        
        params_list.sort(key=lambda x: x[1]['mean'], reverse=True)
        
        # Print each parameter
        for param, stats, data in params_list:
            param_label = xlabels_dict.get(param, {}).get('latex', param)
            range_str = format_range_string(stats)
            mean_std_str = format_mean_std_string(stats)
            
            line = f"  • {param_label} range is {range_str} or {mean_std_str}"
            output_lines.append(line)
            
            csv_data.append({
                'Output': output_name,
                'Output_Label': output_label,
                'Parameter': param,
                'Parameter_Label': param_label,
                'Range_Min': stats['min'],
                'Range_Max': stats['max'],
                'Range': stats['range'],
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Median': stats['median']
            })
        
        output_lines.append("")
    
    # Save text output
    text_file = os.path.join(savepath, "baseline_output_parameter_ranges.txt")
    with open(text_file, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"Output-organized text summary saved to: {text_file}")
    
    # Save CSV
    csv_df = pd.DataFrame(csv_data)
    csv_file = os.path.join(savepath, "baseline_output_parameter_ranges.csv")
    csv_df.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"Output-organized CSV saved to: {csv_file}")
    
    # Print to console
    for line in output_lines:
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze baseline sensitivity ranges across anatomies with human-readable format"
    )
    parser.add_argument('--scenarios', nargs='+', required=True,
                       help='Paths to the baseline scenario folders (each is an anatomy)')
    parser.add_argument('--outputs', nargs='+', required=True,
                       help='Output names to analyze')
    parser.add_argument('--xlabels_dict', required=True,
                       help='Path to the xlabels dictionary JSON file')
    parser.add_argument('--ylabels_dict', required=True,
                       help='Path to the ylabels dictionary JSON file')
    parser.add_argument('--savepath', required=True,
                       help='Path to save the results')
    parser.add_argument('--gsa_mode', default='Si_total',
                       help='GSA mode (e.g., Si_total, Si)')
    parser.add_argument('--mode', default='max',
                       help='Mode for ranking (e.g., max, min)')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Threshold for including parameter-output pairs')
    parser.add_argument('--output_groups', default=None,
                       help='Path to JSON file defining output groups (format: {group_name: [output1, output2, ...]})')
    parser.add_argument('--anatomy_names', nargs='+', default=None,
                       help='Names for each anatomy (should match number of scenarios)')
    
    args = parser.parse_args()
    
    # Load dictionaries
    xlabels_dict = load_xlabels_dict(args.xlabels_dict)
    ylabels_dict = load_ylabels_dict(args.ylabels_dict)
    
    # Load output groups if provided
    output_groups = None
    if args.output_groups and os.path.exists(args.output_groups):
        with open(args.output_groups, 'r') as f:
            output_groups = json.load(f)
    
    # Set anatomy names
    if args.anatomy_names is None:
        anatomy_names = [f"Anatomy {i+1}" for i in range(len(args.scenarios))]
    else:
        anatomy_names = args.anatomy_names
    
    # Extract baseline sensitivity data
    print(f"Extracting baseline sensitivity data from {len(args.scenarios)} anatomies...")
    param_data = extract_baseline_sensitivity_data(
        scenarios=args.scenarios,
        ylabels_raw_all=args.outputs,
        anatomy_names=anatomy_names,
        gsa_mode=args.gsa_mode,
        mode=args.mode,
        threshold=args.threshold
    )
    
    # Print summaries organized by parameter
    print("\nGenerating parameter-organized summaries...")
    print_and_save_parameter_summaries(param_data, xlabels_dict, ylabels_dict, 
                                      args.savepath, output_groups)
    
    # Print summaries organized by output
    print("\nGenerating output-organized summaries...")
    print_output_summaries(param_data, xlabels_dict, ylabels_dict, args.savepath)
    
    print("\n" + "="*120)
    print("BASELINE SENSITIVITY ANALYSIS COMPLETE")
    print(f"Results saved to: {args.savepath}")
    print("="*120)


if __name__ == "__main__":
    main()
