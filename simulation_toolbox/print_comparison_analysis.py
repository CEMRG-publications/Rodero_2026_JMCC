import argparse
import json
import os
import numpy as np
import pandas as pd

def load_xlabels_dict(xlabels_dict_file):
    with open(xlabels_dict_file, 'r') as f:
        return json.load(f)

def load_ylabels_dict(ylabels_dict_file):
    with open(ylabels_dict_file, 'r') as f:
        return json.load(f)

def load_exclusions(exclusions_file):
    if exclusions_file and os.path.exists(exclusions_file):
        with open(exclusions_file, 'r') as f:
            return json.load(f)
    return {}

def extract_sensitivity_data(scenario_path, ylabels_raw_all, gsa_mode="Si_total", mode="max"):
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

def compute_differences(baseline_data, modified_data, excluded_params=None):
    if excluded_params is None:
        excluded_params = []
    differences = {'absolute': {}, 'relative': {}, 'raw': {}, 'percentage': {}}
    for output_name in baseline_data.keys():
        differences['absolute'][output_name] = {}
        differences['relative'][output_name] = {}
        differences['raw'][output_name] = {}
        differences['percentage'][output_name] = {}
        baseline_params = baseline_data[output_name]
        modified_params = modified_data.get(output_name, {})
        for param in baseline_params.keys():
            if param in excluded_params:
                continue
            if param in modified_params:
                baseline_val = baseline_params[param]
                modified_val = modified_params[param]
                differences['absolute'][output_name][param] = abs(modified_val - baseline_val)
                differences['relative'][output_name][param] = abs((modified_val - baseline_val) / baseline_val) if baseline_val != 0 else None
                differences['raw'][output_name][param] = modified_val - baseline_val
                # Percentage difference from baseline
                if baseline_val != 0:
                    differences['percentage'][output_name][param] = ((modified_val - baseline_val) / baseline_val) * 100
                else:
                    differences['percentage'][output_name][param] = None
    return differences

def analyze_scenario_differences(anatomies_dict, ylabels_raw_all, xlabels_dict, ylabels_dict, exclusions, gsa_mode="Si_total", mode="max"):
    first_anatomy = list(anatomies_dict.values())[0]
    modified_names = [os.path.basename(path) for path in first_anatomy['modified']]
    anatomy_names = list(anatomies_dict.keys())

    all_differences = {}
    for anatomy_name, anatomy_data in anatomies_dict.items():
        baseline_data = extract_sensitivity_data(anatomy_data['baseline'], ylabels_raw_all, gsa_mode, mode)
        all_differences[anatomy_name] = {}
        for i, modified_path in enumerate(anatomy_data['modified']):
            modified_name = modified_names[i]
            modified_data = extract_sensitivity_data(modified_path, ylabels_raw_all, gsa_mode, mode)
            excluded_params = exclusions.get(modified_name, [])
            differences = compute_differences(baseline_data, modified_data, excluded_params)
            all_differences[anatomy_name][modified_name] = differences
    return all_differences, modified_names, anatomy_names

def compute_statistics(values):
    filtered = [v for v in values if v is not None]
    if len(filtered) == 0:
        return None
    return {
        'mean': np.mean(filtered),
        'std': np.std(filtered),
        'min': np.min(filtered),
        'max': np.max(filtered),
        'median': np.median(filtered),
        'count': len(filtered),
        'range': np.max(filtered) - np.min(filtered)
    }

def compute_range_statistics(all_differences, modified_names, anatomy_names, ylabels_raw_all):
    anatomy_ranges, output_ranges, parameter_ranges, scenario_ranges = {}, {}, {}, {}
    individual_ranges = []

    # Overall ranges by anatomy
    for anat in anatomy_names:
        vals = []
        pct_vals = []
        for mod in modified_names:
            absolute_diffs = all_differences[anat][mod]['absolute']
            percentage_diffs = all_differences[anat][mod]['percentage']
            
            for out_name, param_dict in absolute_diffs.items():
                for param, value in param_dict.items():
                    if value is not None:
                        vals.append(value)
                    pct_val = percentage_diffs.get(out_name, {}).get(param)
                    if pct_val is not None:
                        pct_vals.append(pct_val)
        if vals:
            anatomy_ranges[anat] = {
                'mean': np.mean(vals), 
                'std': np.std(vals), 
                'range': np.max(vals)-np.min(vals),
                'pct_mean': np.mean(pct_vals) if pct_vals else None,
                'pct_std': np.std(pct_vals) if pct_vals else None
            }

    # Range by output
    for out in ylabels_raw_all:
        vals = [v for anat in anatomy_names for mod in modified_names 
                for p, v in all_differences[anat][mod]['absolute'].get(out, {}).items() if v is not None]
        pct_vals = [v for anat in anatomy_names for mod in modified_names 
                    for p, v in all_differences[anat][mod]['percentage'].get(out, {}).items() if v is not None]
        if vals:
            output_ranges[out] = {
                'mean': np.mean(vals), 
                'std': np.std(vals), 
                'range': np.max(vals)-np.min(vals),
                'pct_mean': np.mean(pct_vals) if pct_vals else None,
                'pct_std': np.std(pct_vals) if pct_vals else None
            }

    # Range by parameter
    all_params = set()
    for anat in anatomy_names:
        for mod in modified_names:
            for out in all_differences[anat][mod]['absolute'].keys():
                all_params.update(all_differences[anat][mod]['absolute'][out].keys())
    for param in all_params:
        vals = [all_differences[anat][mod]['absolute'][out][param]
                for anat in anatomy_names for mod in modified_names for out in all_differences[anat][mod]['absolute'].keys()
                if param in all_differences[anat][mod]['absolute'][out] and all_differences[anat][mod]['absolute'][out][param] is not None]
        pct_vals = [all_differences[anat][mod]['percentage'][out][param]
                    for anat in anatomy_names for mod in modified_names for out in all_differences[anat][mod]['percentage'].keys()
                    if param in all_differences[anat][mod]['percentage'][out] and all_differences[anat][mod]['percentage'][out][param] is not None]
        if vals:
            parameter_ranges[param] = {
                'mean': np.mean(vals), 
                'std': np.std(vals), 
                'range': np.max(vals)-np.min(vals),
                'pct_mean': np.mean(pct_vals) if pct_vals else None,
                'pct_std': np.std(pct_vals) if pct_vals else None
            }

    # Range by modified scenario
    for mod in modified_names:
        vals = [v for anat in anatomy_names for out in all_differences[anat][mod]['absolute'].values() for v in out.values() if v is not None]
        pct_vals = [v for anat in anatomy_names for out in all_differences[anat][mod]['percentage'].values() for v in out.values() if v is not None]
        if vals:
            scenario_ranges[mod] = {
                'mean': np.mean(vals), 
                'std': np.std(vals), 
                'range': np.max(vals)-np.min(vals),
                'pct_mean': np.mean(pct_vals) if pct_vals else None,
                'pct_std': np.std(pct_vals) if pct_vals else None
            }

    # Top 3 individual combinations
    for anat in anatomy_names:
        for out in ylabels_raw_all:
            all_params_for_out = set()
            for mod in modified_names:
                all_params_for_out.update(all_differences[anat][mod]['absolute'].get(out, {}).keys())
            for param in all_params_for_out:
                vals = [np.abs(all_differences[anat][mod]['absolute'][out][param])
                        for mod in modified_names
                        if param in all_differences[anat][mod]['absolute'].get(out, {}) and all_differences[anat][mod]['absolute'][out][param] is not None]
                pct_vals = [np.abs(all_differences[anat][mod]['percentage'][out][param])
                            for mod in modified_names
                            if param in all_differences[anat][mod]['percentage'].get(out, {}) and all_differences[anat][mod]['percentage'][out][param] is not None]
                if len(vals) > 1:
                    individual_ranges.append({
                        'anatomy': anat, 
                        'output': out, 
                        'parameter': param,
                        'range': np.max(vals)-np.min(vals),
                        'mean': np.mean(vals), 
                        'std': np.std(vals),
                        'min': np.min(vals),
                        'max': np.max(vals),
                        'pct_range': np.max(pct_vals)-np.min(pct_vals) if pct_vals else None,
                        'pct_mean': np.mean(pct_vals) if pct_vals else None,
                        'pct_std': np.std(pct_vals) if pct_vals else None,
                        'pct_min': np.min(pct_vals) if pct_vals else None,
                        'pct_max': np.max(pct_vals) if pct_vals else None
                    })
    return anatomy_ranges, output_ranges, parameter_ranges, scenario_ranges, individual_ranges

def print_and_save_top_ranges(anatomy_ranges, output_ranges, parameter_ranges, scenario_ranges, individual_ranges,
                              all_differences, modified_names, anatomy_names, xlabels_dict, ylabels_dict, savepath):

    os.makedirs(savepath, exist_ok=True)
    results_text = []

    # Compute overall statistics across all data
    all_vals = [np.abs(v) for anat in anatomy_names for mod in modified_names for out in all_differences[anat][mod]['absolute'].values() for v in out.values() if v is not None]
    all_pct_vals = [np.abs(v) for anat in anatomy_names for mod in modified_names for out in all_differences[anat][mod]['percentage'].values() for v in out.values() if v is not None]
    overall_mean, overall_std, overall_min, overall_max = np.mean(all_vals), np.std(all_vals), np.min(all_vals), np.max(all_vals)
    overall_pct_mean, overall_pct_std, overall_pct_min, overall_pct_max = np.mean(all_pct_vals), np.std(all_pct_vals), np.min(all_pct_vals), np.max(all_pct_vals)
    msg = f"Overall, modifying the scenarios resulted in a difference of {overall_mean:.2e}±{overall_std:.2e} ([{overall_min:.2f}, {overall_max:.2f}]) in the sensitivity when compared to baseline ({overall_pct_mean:.2f}%±{overall_pct_std:.2f}%, range [{overall_pct_min:.2f}%, {overall_pct_max:.2f}%])"
    print("\n" + "="*120)
    print(msg)
    results_text.append(msg)



    print("\nTOP 3 INDIVIDUAL COMBINATIONS (ANATOMY, OUTPUT, PARAMETER, SCENARIO)")
    results_text.append("\nTOP 3 INDIVIDUAL COMBINATIONS (ANATOMY, OUTPUT, PARAMETER, SCENARIO)")

    sorted_individuals = sorted(individual_ranges, key=lambda x: x['range'], reverse=True)[:3]

    for i, combo in enumerate(sorted_individuals, 1):
        out_label = ylabels_dict.get(combo['output'], {}).get('latex', combo['output'])
        param_label = xlabels_dict.get(combo['parameter'], {}).get('latex', combo['parameter'])
        
        # Determine which scenario has the maximum difference for this combo
        max_diff = -np.inf
        max_scenario = None
        for mod in modified_names:
            val = all_differences[combo['anatomy']][mod]['absolute'].get(combo['output'], {}).get(combo['parameter'], None)
            if val is not None and val > max_diff:
                max_diff = val
                max_diff_pct = all_differences[combo['anatomy']][mod]['percentage'].get(combo['output'], {}).get(combo['parameter'], None)
                
                max_scenario = mod
        
        pct_str = f" ({combo['pct_mean']:.2f}%±{combo['pct_std']:.2f}%, [{combo['min']:.2f}%, {combo['max']:.2f}%])" if combo['pct_mean'] is not None else ""
        msg = (f"#{i}: Anatomy='{combo['anatomy']}', Output='{out_label}', Parameter='{param_label}', "
            f"Scenario='{max_scenario}' with max difference={max_diff:.2e} ({max_diff_pct:.2f}%) | ")
        print(msg)
        results_text.append(msg)


    # Top 3 Anatomies
    print("\nTOP 3 ANATOMIES WITH HIGHEST VARIABILITY")
    results_text.append("\nTOP 3 ANATOMIES WITH HIGHEST VARIABILITY")
    sorted_anatomies = sorted(anatomy_ranges.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
    for i, (anat, stats) in enumerate(sorted_anatomies, 1):
        pct_str = f" ({stats['pct_mean']:.2f}%±{stats['pct_std']:.2f}%)" if stats['pct_mean'] is not None else ""
        msg = f"#{i}: The anatomy '{anat}' has the highest variability between baseline and scenarios, with a mean of {stats['mean']:.2e}±{stats['std']:.2e}{pct_str}"
        print(msg)
        results_text.append(msg)

    # Top 3 Outputs
    print("\nTOP 3 OUTPUTS WITH HIGHEST VARIABILITY")
    results_text.append("\nTOP 3 OUTPUTS WITH HIGHEST VARIABILITY")
    sorted_outputs = sorted(output_ranges.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
    for i, (out, stats) in enumerate(sorted_outputs, 1):
        label = ylabels_dict.get(out, {}).get('latex', out)
        pct_str = f" ({stats['pct_mean']:.2f}%±{stats['pct_std']:.2f}%)" if stats['pct_mean'] is not None else ""
        msg = f"#{i}: The output '{label}' has the highest variability between baseline and scenarios, with a range of {stats['mean']:.6f}±{stats['std']:.6f}{pct_str}"
        print(msg)
        results_text.append(msg)

    # Top 3 Parameters
    print("\nTOP 3 PARAMETERS WITH HIGHEST VARIABILITY")
    results_text.append("\nTOP 3 PARAMETERS WITH HIGHEST VARIABILITY")
    sorted_params = sorted(parameter_ranges.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
    for i, (param, stats) in enumerate(sorted_params, 1):
        label = xlabels_dict.get(param, {}).get('latex', param)
        pct_str = f" ({stats['pct_mean']:.2f}%±{stats['pct_std']:.2f}%)" if stats['pct_mean'] is not None else ""
        msg = f"#{i}: The parameter '{label}' has the highest variability between baseline and scenarios, with a range of {stats['mean']:.6f}±{stats['std']:.6f}{pct_str}"
        print(msg)
        results_text.append(msg)

    # Top 3 Scenarios
    print("\nTOP 3 MODIFIED SCENARIOS WITH HIGHEST VARIABILITY")
    results_text.append("\nTOP 3 MODIFIED SCENARIOS WITH HIGHEST VARIABILITY")
    sorted_scenarios = sorted(scenario_ranges.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
    for i, (mod, stats) in enumerate(sorted_scenarios, 1):
        pct_str = f" ({stats['pct_mean']:.2f}%±{stats['pct_std']:.2f}%)" if stats['pct_mean'] is not None else ""
        msg = f"#{i}: The modified scenario '{mod}' has the highest variability between baseline and scenarios, with a range of {stats['mean']:.6f}±{stats['std']:.6f}{pct_str}"
        print(msg)
        results_text.append(msg)

    # Save summary
    summary_file = os.path.join(savepath, "top3_variability_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("\n".join(results_text))
    print(f"\nTop 3 variability summary saved to: {summary_file}")

    # Save CSV files
    save_detailed_range_csv(anatomy_ranges, output_ranges, parameter_ranges, scenario_ranges, individual_ranges, xlabels_dict, ylabels_dict, savepath)

def save_detailed_range_csv(anatomy_ranges, output_ranges, parameter_ranges, scenario_ranges, individual_ranges, xlabels_dict, ylabels_dict, savepath):
    # Anatomy
    df = pd.DataFrame([{'anatomy': k, **v} for k,v in anatomy_ranges.items()])
    df.to_csv(os.path.join(savepath, "ranges_by_anatomy.csv"), index=False)
    # Output
    df = pd.DataFrame([{'output': ylabels_dict.get(k, {}).get('latex', k), 'output_raw': k, **v} for k,v in output_ranges.items()])
    df.to_csv(os.path.join(savepath, "ranges_by_output.csv"), index=False)
    # Parameter
    df = pd.DataFrame([{'parameter': xlabels_dict.get(k, {}).get('latex', k), 'parameter_raw': k, **v} for k,v in parameter_ranges.items()])
    df.to_csv(os.path.join(savepath, "ranges_by_parameter.csv"), index=False)
    # Scenario
    df = pd.DataFrame([{ 'modified_scenario': k, **v} for k,v in scenario_ranges.items()])
    df.to_csv(os.path.join(savepath, "ranges_by_scenario.csv"), index=False)
    # Individual
    df = pd.DataFrame([{
        'anatomy': c['anatomy'],
        'output': ylabels_dict.get(c['output'], {}).get('latex', c['output']),
        'output_raw': c['output'],
        'parameter': xlabels_dict.get(c['parameter'], {}).get('latex', c['parameter']),
        'parameter_raw': c['parameter'],
        'range': c['range'], 
        'mean': c['mean'], 
        'std': c['std'],
        'pct_range': c.get('pct_range'),
        'pct_mean': c.get('pct_mean'),
        'pct_std': c.get('pct_std')
    } for c in individual_ranges])
    df.to_csv(os.path.join(savepath, "ranges_by_individual_combination.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_anatomies', type=int, required=True)
    parser.add_argument('--anatomy_names', nargs='+', required=True)
    parser.add_argument('--xlabels_dict', required=True)
    parser.add_argument('--ylabels_dict', required=True)
    parser.add_argument('--exclusions', default=None)
    parser.add_argument('--savepath', required=True)
    parser.add_argument('--outputs', nargs='+', required=True)
    parser.add_argument('--anatomy1_baseline', required=True)
    parser.add_argument('--anatomy2_baseline', required=True)
    parser.add_argument('--anatomy3_baseline', required=True)
    parser.add_argument('--anatomy4_baseline', required=True)
    parser.add_argument('--anatomy5_baseline', required=True)
    parser.add_argument('--anatomy1_modified', nargs='+', required=True)
    parser.add_argument('--anatomy2_modified', nargs='+', required=True)
    parser.add_argument('--anatomy3_modified', nargs='+', required=True)
    parser.add_argument('--anatomy4_modified', nargs='+', required=True)
    parser.add_argument('--anatomy5_modified', nargs='+', required=True)
    args = parser.parse_args()

    xlabels_dict = load_xlabels_dict(args.xlabels_dict)
    ylabels_dict = load_ylabels_dict(args.ylabels_dict)
    exclusions = load_exclusions(args.exclusions)

    anatomies_dict = {
        args.anatomy_names[0]: {'baseline': args.anatomy1_baseline, 'modified': args.anatomy1_modified},
        args.anatomy_names[1]: {'baseline': args.anatomy2_baseline, 'modified': args.anatomy2_modified},
        args.anatomy_names[2]: {'baseline': args.anatomy3_baseline, 'modified': args.anatomy3_modified},
        args.anatomy_names[3]: {'baseline': args.anatomy4_baseline, 'modified': args.anatomy4_modified},
        args.anatomy_names[4]: {'baseline': args.anatomy5_baseline, 'modified': args.anatomy5_modified}
    }

    all_differences, modified_names, anatomy_names = analyze_scenario_differences(
        anatomies_dict, args.outputs, xlabels_dict, ylabels_dict, exclusions
    )

    anatomy_ranges, output_ranges, parameter_ranges, scenario_ranges, individual_ranges = compute_range_statistics(
        all_differences, modified_names, anatomy_names, args.outputs
    )

    print_and_save_top_ranges(anatomy_ranges, output_ranges, parameter_ranges, scenario_ranges, individual_ranges,
                              all_differences, modified_names, anatomy_names, xlabels_dict, ylabels_dict, args.savepath)

if __name__ == "__main__":
    main()