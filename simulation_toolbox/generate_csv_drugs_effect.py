import argparse
import pandas as pd
import numpy as np
import os
import json
from common.utils import generate_gsa_ranking_files, read_xlabels_dict


def get_best_ranking_for_parameter(scenario_path, parameter, ylabels_raw_all, gsa_mode="Si_total", mode="max"):
    """
    Find the highest ranking position (lowest rank number) for a parameter across all outputs.
    
    Returns:
        best_rank (int): Best ranking position (1 is best)
        best_output (str): Output feature name where best ranking was achieved
        effect_value (float): GSA effect value at best ranking
    """
    best_rank = float('inf')
    best_output = None
    best_effect = 0.0
    
    for ylabel_raw in ylabels_raw_all:
        rank_file = os.path.join(scenario_path, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
        
        if os.path.exists(rank_file):
            with open(rank_file, "r") as f:
                for i, line in enumerate(f.readlines()):
                    param, effect_str = line.strip().split("\t")
                    effect = float(effect_str)
                    
                    if param == parameter and effect >= 0.05:  # Only consider relevant parameters
                        rank = i + 1
                        if rank < best_rank:
                            best_rank = rank
                            best_output = ylabel_raw
                            best_effect = effect
    
    if best_rank == float('inf'):
        return None, None, None
    
    return int(best_rank), best_output, best_effect


def get_variability_at_baseline(baseline_scenarios, parameter, output, gsa_mode="Si_total", mode="max"):
    """
    Calculate ranking variability for a parameter at a specific output across baseline scenarios.
    
    Returns:
        variability (int): max_rank - min_rank across baseline scenarios
    """
    ranks = []
    
    for scenario in baseline_scenarios:
        rank_file = os.path.join(scenario, "output", f"Rank_{gsa_mode}_{mode}_{output}.txt")
        
        if os.path.exists(rank_file):
            with open(rank_file, "r") as f:
                for i, line in enumerate(f.readlines()):
                    param, effect_str = line.strip().split("\t")
                    effect = float(effect_str)
                    
                    if param == parameter and effect >= 0.05:
                        ranks.append(i + 1)
                        break
    
    if len(ranks) < 2:
        return None  # Need at least 2 scenarios for variability
    
    return max(ranks) - min(ranks)


def auto_detect_modified_scenarios(baseline_scenario):
    """
    Automatically detect modified scenarios from GSA_* folders in baseline/output directory.
    Returns list of modified scenario paths with metadata.
    """
    output_dir = os.path.join(baseline_scenario, "output")
    modified_scenarios = []
    
    if not os.path.exists(output_dir):
        print(f"Warning: Output directory not found: {output_dir}")
        return []
    
    # Look for GSA_* directories
    for item in os.listdir(output_dir):
        full_path = os.path.join(output_dir, item)
        if os.path.isdir(full_path) and item.startswith("GSA_"):
            # Extract parameter name and change type
            if "lower_50.0_upper_100.0" in item:
                param = item.replace("GSA_", "").replace("_lower_50.0_upper_100.0", "")
                change_type = "increased"
            elif "lower_0.0_upper_50.0" in item:
                param = item.replace("GSA_", "").replace("_lower_0.0_upper_50.0", "")
                change_type = "decreased"
            else:
                param = item.replace("GSA_", "")
                change_type = "unknown"
            
            modified_scenarios.append({
                'path': full_path,
                'name': item,
                'parameter': param,
                'change_type': change_type
            })
    
    print(f"Auto-detected {len(modified_scenarios)} modified scenarios")
    return modified_scenarios


def get_default_parameters():
    """
    Return the default set of parameters based on common GSA analysis parameters.
    These are typical parameters that are often modified in cardiac modeling studies.
    """
    return [
        "g_K1_scale",
        "g_Kr_scale", 
        "g_Ks_scale",
        "g_to_scale",
        "g_CaL_scale",
        "g_Na_scale",
        "g_NaL_scale",
        "g_bNa_scale",
        "g_NaCa_scale",
        "g_NaK_scale",
        "g_pCa_scale",
        "g_pK_scale",
        "g_bCa_scale",
        "Jup_scale",
        "Jleak_scale",
        "Jrel_scale"
    ]


def create_parameter_ranking_table(
    baseline_scenario,
    parameters=None,
    xlabels_file=None,
    ylabels_file=None,
    ylabels_dict=None,
    xlabels_dict=None,
    savepath=None,
    figname_prefix="parameter_ranking_comparison",
    gsa_mode="Si_total",
    mode="max",
    modified_scenarios=None
):
    """
    Create a CSV table comparing parameter rankings across modified scenarios.
    """
    
    # Auto-detect modified scenarios if not provided
    if modified_scenarios is None:
        detected_scenarios = auto_detect_modified_scenarios(baseline_scenario)
        modified_scenarios = [s['path'] for s in detected_scenarios]
        scenario_metadata = {s['path']: s for s in detected_scenarios}
    else:
        scenario_metadata = {}
    
    if not modified_scenarios:
        print("No modified scenarios found!")
        return None
    
    # Set default paths based on bash script structure
    baseline_parent = os.path.dirname(os.path.dirname(os.path.dirname(baseline_scenario)))  # Go up three levels like in bash script
    gsa_analysis_dir = os.path.join(baseline_parent, "GSA_analysis", "cycle")
    
    if xlabels_file is None:
        xlabels_file = os.path.join(gsa_analysis_dir, "xlabels.txt")
    if ylabels_file is None:
        ylabels_file = os.path.join(gsa_analysis_dir, "ylabels.txt")
    if ylabels_dict is None:
        ylabels_dict = os.path.join(gsa_analysis_dir, "ylabels_filtered.json")
    if xlabels_dict is None:
        xlabels_dict = os.path.join(gsa_analysis_dir, "xlabels_to_plot.json")
    if savepath is None:
        savepath = os.path.join(baseline_parent, "GSA_analysis")
    
    # Generate ranking files for all scenarios
    all_scenarios = modified_scenarios + [baseline_scenario]
    features_idx_list, ylabels_raw_all, ylabels_latex_all = generate_gsa_ranking_files(
        xlabels_file=xlabels_file,
        ylabels_file=ylabels_file,
        ylabels_dict=ylabels_dict,
        scenarios=all_scenarios
    )
    
    # Read xlabels dictionary
    xlabels = np.loadtxt(xlabels_file, dtype=str)
    _, xlabels_dict_all = read_xlabels_dict(xlabels_dict, xlabels)
    
    # Auto-detect parameters if not provided
    if parameters is None:
        # First try to detect from modified scenarios
        detected_params = []
        for scenario_path in modified_scenarios:
            if scenario_path in scenario_metadata:
                param = scenario_metadata[scenario_path]['parameter']
                if param not in detected_params:
                    detected_params.append(param)
        
        # If no parameters detected from scenarios, use defaults
        if detected_params:
            parameters = detected_params
            print(f"Auto-detected parameters from scenarios: {parameters}")
        else:
            parameters = get_default_parameters()
            print(f"Using default parameter set: {parameters}")
            print("Note: Only parameters with modified scenarios will show data")
    
    # Read ylabels dictionary for output name mapping
    with open(ylabels_dict, 'r') as f:
        ylabels_dict_data = json.load(f)
    
    # Use baseline scenario for comparison
    baseline_scenarios = [baseline_scenario]
    
    # Initialize results list
    results = []
    
    for scenario in modified_scenarios:
        # Get scenario name with metadata if available
        if scenario in scenario_metadata:
            meta = scenario_metadata[scenario]
            scenario_name = f"{meta['parameter']}_{meta['change_type']}"
        else:
            scenario_name = os.path.basename(scenario.rstrip('/'))
            
        row_data = {'Scenario': scenario_name}
        
        for param in parameters:
            if param not in xlabels:
                print(f"Warning: Parameter '{param}' not found in xlabels")
                continue
            
            # Get parameter's LaTeX name for display
            param_latex = xlabels_dict_all.get(param, {}).get("latex", param)
            
            # Get best ranking for modified scenario
            mod_rank, mod_output, mod_effect = get_best_ranking_for_parameter(
                scenario, param, ylabels_raw_all, gsa_mode, mode
            )
            
            # Get best ranking for baseline scenario
            base_rank, base_output, base_effect = get_best_ranking_for_parameter(
                baseline_scenario, param, ylabels_raw_all, gsa_mode, mode
            )
            
            # Get output names with LaTeX formatting
            mod_output_latex = None
            base_output_latex = None
            
            if mod_output:
                output_idx = ylabels_raw_all.index(mod_output)
                mod_output_latex = ylabels_latex_all[output_idx]
            
            if base_output:
                output_idx = ylabels_raw_all.index(base_output)
                base_output_latex = ylabels_latex_all[output_idx]
            
            # Calculate variability at baseline (only if multiple baseline scenarios provided)
            baseline_variability = None
            if base_output and len(baseline_scenarios) > 1:
                baseline_variability = get_variability_at_baseline(
                    baseline_scenarios, param, base_output, gsa_mode, mode
                )
            elif len(baseline_scenarios) == 1:
                baseline_variability = 0  # No variability with single scenario
            
            # Add columns for this parameter
            param_prefix = f"{param_latex}"
            
            row_data[f"{param_prefix}_Modified_Rank"] = mod_rank if mod_rank else "N/A"
            row_data[f"{param_prefix}_Modified_Output"] = mod_output_latex if mod_output_latex else "N/A"
            row_data[f"{param_prefix}_Modified_Effect"] = f"{mod_effect:.3f}" if mod_effect else "N/A"
            
            row_data[f"{param_prefix}_Baseline_Rank"] = base_rank if base_rank else "N/A"
            row_data[f"{param_prefix}_Baseline_Output"] = base_output_latex if base_output_latex else "N/A"
            row_data[f"{param_prefix}_Baseline_Effect"] = f"{base_effect:.3f}" if base_effect else "N/A"
            
            row_data[f"{param_prefix}_Baseline_Variability"] = baseline_variability if baseline_variability else "N/A"
            
            # Calculate rank difference (baseline - modified, positive means improved in modified)
            if mod_rank and base_rank:
                rank_diff = base_rank - mod_rank
                row_data[f"{param_prefix}_Rank_Change"] = rank_diff
            else:
                row_data[f"{param_prefix}_Rank_Change"] = "N/A"
        
        results.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(savepath, exist_ok=True)
    output_file = os.path.join(savepath, f"{figname_prefix}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Parameter ranking comparison table saved to: {output_file}")
    print(f"Table contains {len(results)} scenarios and {len(parameters)} parameters")
    
    # Print summary statistics
    print("\nSummary:")
    for param in parameters:
        param_latex = xlabels_dict_all.get(param, {}).get("latex", param)
        rank_changes = df[f"{param_latex}_Rank_Change"].replace("N/A", np.nan).astype(float)
        valid_changes = rank_changes.dropna()
        
        if len(valid_changes) > 0:
            print(f"  {param_latex}:")
            print(f"    Mean rank change: {valid_changes.mean():.2f}")
            print(f"    Scenarios with improvement: {(valid_changes > 0).sum()}/{len(valid_changes)}")
            print(f"    Best improvement: {valid_changes.max():.0f}")
            print(f"    Worst change: {valid_changes.min():.0f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Create Parameter Ranking Comparison Table")
    parser.add_argument('--baseline_scenario', required=True,
                        help='Path to baseline scenario directory (will auto-detect modified scenarios in output/)')
    parser.add_argument('--parameters', nargs='+', 
                        help='Parameter names to analyze (default: common cardiac model parameters, or auto-detect from modified scenarios)')
    parser.add_argument('--xlabels', 
                        help='Path to the xlabels file (default: ../../GSA_analysis/cycle/xlabels.txt)')
    parser.add_argument('--ylabels', 
                        help='Path to the ylabels file (default: ../../GSA_analysis/cycle/ylabels.txt)')
    parser.add_argument('--savepath', 
                        help='Path to save the CSV table (default: ../../GSA_analysis/)')
    parser.add_argument('--figname_prefix', default='parameter_ranking_comparison',
                        help='Prefix for the output filename')
    parser.add_argument('--ylabels_dict', 
                        help='Path to the ylabels dictionary file (default: ../../GSA_analysis/cycle/ylabels_filtered.json)')
    parser.add_argument('--xlabels_dict', 
                        help='Path to the xlabels dictionary file (default: ../../GSA_analysis/cycle/xlabels_to_plot.json)')
    parser.add_argument('--gsa_mode', default='Si_total',
                        help='GSA mode (default: Si_total)')
    parser.add_argument('--mode', default='max',
                        help='Ranking mode (default: max)')
    # Optional: allow manual specification of modified scenarios
    parser.add_argument('--modified_scenarios', nargs='+',
                        help='Paths to modified scenario directories (optional, will auto-detect if not provided)')
    
    args = parser.parse_args()
    
    create_parameter_ranking_table(
        baseline_scenario=args.baseline_scenario,
        parameters=args.parameters,
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        ylabels_dict=args.ylabels_dict,
        xlabels_dict=args.xlabels_dict,
        savepath=args.savepath,
        figname_prefix=args.figname_prefix,
        gsa_mode=args.gsa_mode,
        mode=args.mode,
        modified_scenarios=args.modified_scenarios
    )


if __name__ == "__main__":
    main()