import argparse
import numpy as np
import os
import pandas as pd

from common.utils import generate_gsa_ranking_files, read_xlabels_dict


def compute_sensitivity_ranges(
    scenarios,
    xlabels_dict,
    ylabels_raw_all,
    ylabels_latex_all,
    features_idx_list,
    gsa_mode="Si_total",
    mode="max"
):
    """
    Compute sensitivity ranges (min, max, mean, median) across scenarios.
    For each parameter-output pair, collects values from all scenarios.
    """
    
    # Collect all parameters from the overall ranking of the first scenario
    overall_rank_file = os.path.join(scenarios[0], "output", f"Rank_{gsa_mode}_{mode}.txt")
    if not os.path.exists(overall_rank_file):
        print(f"Warning: Overall rank file not found: {overall_rank_file}")
        return None, None, None

    all_params = []
    with open(overall_rank_file, "r") as f:
        for line in f.readlines():
            param, _ = line.strip().split("\t")
            all_params.append(param)

    # Initialize: dict to store sensitivity values across scenarios for each param-output pair
    sensitivity_ranges = {}  # (param_idx, output_idx) -> list of values

    for j, feature_idx in enumerate(features_idx_list):
        ylabel_raw = ylabels_raw_all[feature_idx]

        for scenario in scenarios:
            rank_file_path = os.path.join(scenario, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")

            if os.path.exists(rank_file_path):
                with open(rank_file_path, "r") as f:
                    for i, line in enumerate(f.readlines()):
                        param, value = line.strip().split("\t")
                        if param in all_params:
                            param_idx = all_params.index(param)
                            value_float = float(value)
                            
                            key = (param_idx, j)
                            if key not in sensitivity_ranges:
                                sensitivity_ranges[key] = []
                            sensitivity_ranges[key].append(value_float)

    return sensitivity_ranges, all_params, ylabels_latex_all


def compute_stats(values):
    """Compute statistics for a list of values."""
    if len(values) == 0:
        return None
    return {
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'count': len(values)
    }


def print_sensitivity_range_summary(
    sensitivity_ranges,
    all_params,
    ylabels_latex_all,
    features_idx_list,
    xlabels_dict,
    threshold=0.05,
    range_threshold=0.1
):
    """
    Print summary of sensitivity ranges across scenarios.
    Also computes the percentage of parameter-output pairs with range <= range_threshold.
    """
    
    print("\n" + "="*100)
    print("SENSITIVITY RANGES ACROSS SCENARIOS (No Threshold)")
    print("="*100)
    
    stats_list = []
    for (param_idx, output_idx), values in sensitivity_ranges.items():
        stats = compute_stats(values)
        if stats is None:
            continue
        param_label = xlabels_dict.get(all_params[param_idx], {}).get("latex", all_params[param_idx])
        output_label = ylabels_latex_all[features_idx_list[output_idx]]
        
        stats_list.append({
            'Parameter': param_label,
            'Output': output_label,
            'Min': stats['min'],
            'Max': stats['max'],
            'Mean': stats['mean'],
            'Median': stats['median'],
            'Std': stats['std'],
            'Count': stats['count'],
            'Range': stats['max'] - stats['min']
        })
    
    stats_df = pd.DataFrame(stats_list)
    
    # Overall statistics without threshold
    all_range_values = []
    for values in sensitivity_ranges.values():
        all_range_values.extend(values)
    
    overall_stats = compute_stats(all_range_values)
    
    print(f"\n{'Parameter':<25} {'Output':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print("-" * 100)
    for _, row in stats_df.iterrows():
        print(f"{row['Parameter']:<25} {row['Output']:<25} {row['Min']:>10.4f} {row['Max']:>10.4f} {row['Mean']:>10.4f} {row['Median']:>10.4f} {row['Std']:>10.4f}")
    
    # Percentage of cases with range <= range_threshold
    total_cases = len(stats_df)
    small_range_cases = len(stats_df[stats_df["Range"] <= range_threshold])
    perc_small_range = (small_range_cases / total_cases * 100) if total_cases > 0 else 0.0

    print("\n" + "-"*100)
    print("OVERALL STATISTICS (No Threshold)")
    print("-"*100)
    print(f"Overall Min: {overall_stats['min']:.4f}")
    print(f"Overall Max: {overall_stats['max']:.4f}")
    print(f"Overall Mean: {overall_stats['mean']:.4f}")
    print(f"Overall Median: {overall_stats['median']:.4f}")
    print(f"Overall Std Dev: {overall_stats['std']:.4f}")
    print(f"Total Data Points: {len(all_range_values)}")
    print(f"Percentage of (Param, Output) pairs with Range ≤ {range_threshold}: {perc_small_range:.2f}%")

    # With threshold: only include if ALL scenarios >= threshold
    print("\n\n" + "="*100)
    print(f"SENSITIVITY RANGES ACROSS SCENARIOS (All Scenarios >= {threshold})")
    print("="*100)
    
    stats_list_thresh = []
    for (param_idx, output_idx), values in sensitivity_ranges.items():
        if all(v >= threshold for v in values):
            stats = compute_stats(values)
            if stats is None:
                continue
            param_label = xlabels_dict.get(all_params[param_idx], {}).get("latex", all_params[param_idx])
            output_label = ylabels_latex_all[features_idx_list[output_idx]]
            
            stats_list_thresh.append({
                'Parameter': param_label,
                'Output': output_label,
                'Min': stats['min'],
                'Max': stats['max'],
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Std': stats['std'],
                'Count': stats['count'],
                'Range': stats['max'] - stats['min']
            })
    
    if len(stats_list_thresh) > 0:
        stats_df_thresh = pd.DataFrame(stats_list_thresh)
        
        print(f"\n{'Parameter':<25} {'Output':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10} {'Std':>10}")
        print("-" * 100)
        for _, row in stats_df_thresh.iterrows():
            print(f"{row['Parameter']:<25} {row['Output']:<25} {row['Min']:>10.4f} {row['Max']:>10.4f} {row['Mean']:>10.4f} {row['Median']:>10.4f} {row['Std']:>10.4f}")
        
        # Overall statistics with threshold
        thresh_values = []
        for values in sensitivity_ranges.values():
            if all(v >= threshold for v in values):
                thresh_values.extend(values)
        
        overall_stats_thresh = compute_stats(thresh_values)

        total_cases_thresh = len(stats_df_thresh)
        small_range_cases_thresh = len(stats_df_thresh[stats_df_thresh["Range"] <= range_threshold])
        perc_small_range_thresh = (small_range_cases_thresh / total_cases_thresh * 100) if total_cases_thresh > 0 else 0.0
        
        print("\n" + "-"*100)
        print(f"OVERALL STATISTICS (All Scenarios >= {threshold})")
        print("-"*100)
        print(f"Overall Min: {overall_stats_thresh['min']:.4f}")
        print(f"Overall Max: {overall_stats_thresh['max']:.4f}")
        print(f"Overall Mean: {overall_stats_thresh['mean']:.4f}")
        print(f"Overall Median: {overall_stats_thresh['median']:.4f}")
        print(f"Overall Std Dev: {overall_stats_thresh['std']:.4f}")
        print(f"Total Data Points: {len(thresh_values)}")
        print(f"Percentage of (Param, Output) pairs with Range ≤ {range_threshold}: {perc_small_range_thresh:.2f}%")
    else:
        print(f"\nNo parameter-output pairs found with all scenarios >= {threshold}")


def generate_gsa_sensitivity_range_summary(
    scenarios,
    xlabels_file,
    ylabels_file,
    ylabels_dict,
    xlabels_dict,
    threshold=0.05,
    range_threshold=0.1
):
    """Main driver for sensitivity range summary."""
    features_idx_list, ylabels_raw_all, ylabels_latex_all = generate_gsa_ranking_files(
        xlabels_file=xlabels_file,
        ylabels_file=ylabels_file,
        ylabels_dict=ylabels_dict,
        scenarios=scenarios
    )
    
    xlabels = np.loadtxt(xlabels_file, dtype=str)
    _, xlabels_dict_all = read_xlabels_dict(xlabels_dict, xlabels)
    
    sensitivity_ranges, all_params, ylabels_latex = compute_sensitivity_ranges(
        scenarios=scenarios,
        xlabels_dict=xlabels_dict_all,
        ylabels_raw_all=ylabels_raw_all,
        ylabels_latex_all=ylabels_latex_all,
        features_idx_list=features_idx_list,
    )
    
    if sensitivity_ranges is not None:
        print_sensitivity_range_summary(
            sensitivity_ranges=sensitivity_ranges,
            all_params=all_params,
            ylabels_latex_all=ylabels_latex_all,
            features_idx_list=features_idx_list,
            xlabels_dict=xlabels_dict_all,
            threshold=threshold,
            range_threshold=range_threshold
        )


def main():
    parser = argparse.ArgumentParser(description="Script to print a summary of sensitivity ranges across baseline scenarios.")
    parser.add_argument('--scenarios', nargs='+', required=True,
                        help='Paths to the scenario folders')
    parser.add_argument('--xlabels', required=True,
                        help='Path to the xlabels file')
    parser.add_argument('--ylabels', required=True,
                        help='Path to the ylabels file')
    parser.add_argument('--ylabels_dict', type=str, required=True,
                        help='Path to the ylabels dictionary file')
    parser.add_argument('--xlabels_dict', type=str, required=True,
                        help='Path to the xlabels dictionary file')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Threshold for all scenarios (default: 0.05)')
    parser.add_argument('--range_threshold', type=float, default=0.1,
                        help='Maximum range value to consider for percentage calculation (default: 0.1)')
    
    args = parser.parse_args()
    
    generate_gsa_sensitivity_range_summary(
        scenarios=args.scenarios,
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        ylabels_dict=args.ylabels_dict,
        xlabels_dict=args.xlabels_dict,
        threshold=args.threshold,
        range_threshold=args.range_threshold
    )


if __name__ == "__main__":
    main()
