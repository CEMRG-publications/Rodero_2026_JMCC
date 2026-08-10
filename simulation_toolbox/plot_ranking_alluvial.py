#!/usr/bin/env python3
"""
Two-column GSA alluvial plotting script.

- Always draws exactly two vertical lanes: Baseline (left) and Modified (right).
- Each feature gets a stacked block of flows (threads) between the same two lanes.
- A parameter that affects many features will appear as separate flows in each feature block,
  but will have the same color everywhere.
- Horizontal separators between feature blocks and an optional legend can be toggled via CLI flags.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from common.utils import generate_gsa_ranking_files, read_xlabels_dict

# ------------------------------
# Utilities
# ------------------------------
def get_ranking_data(scenario_path, ylabel_raw, gsa_mode="Si_total", mode="max"):
    """
    Read rank file and return (param -> rank), (param -> effect).
    Accepts flexible whitespace/tab separators. If file missing returns empty dicts.
    """
    rank_file_path = os.path.join(scenario_path, "output", f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
    if not os.path.exists(rank_file_path):
        # fallback: maybe rank file is directly in scenario root (rare)
        alt_path = os.path.join(scenario_path, f"Rank_{gsa_mode}_{mode}_{ylabel_raw}.txt")
        if os.path.exists(alt_path):
            rank_file_path = alt_path
        else:
            print(f"Warning: Rank file not found: {rank_file_path}")
            return {}, {}

    param_ranks = {}
    param_effects = {}
    with open(rank_file_path, "r") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            # split by whitespace/tabs. Expect last token to be numeric effect
            parts = s.split()
            if len(parts) < 2:
                continue
            param = parts[0]
            # try last token as float (handles e.g. "param\t0.123")
            val_token = parts[-1]
            try:
                val = float(val_token.replace(",", "."))
            except Exception:
                # if not parseable, skip row
                continue
            param_ranks[param] = i + 1
            param_effects[param] = val
    return param_ranks, param_effects


def draw_flow(ax, x1, y1, x2, y2, color, alpha=0.75, width_scale=1.0):
    """
    Draw a smooth cubic-bezier-like curve between two points.
    width_scale affects line width (useful for emphasizing stronger effects).
    """
    mid_x = (x1 + x2) / 2.0
    # control points (kept near the endpoints' y to avoid large vertical swings)
    ctrl_y1 = y1
    ctrl_y2 = y2

    t = np.linspace(0.0, 1.0, 200)
    x_curve = (1-t)**3 * x1 + 3*(1-t)**2 * t * mid_x + 3*(1-t) * t**2 * mid_x + t**3 * x2
    y_curve = (1-t)**3 * y1 + 3*(1-t)**2 * t * ctrl_y1 + 3*(1-t) * t**2 * ctrl_y2 + t**3 * y2

    lw = max(0.8, 1.5 * width_scale)
    ax.plot(x_curve, y_curve, color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')


# ------------------------------
# Plotting: Single-feature (unchanged style)
# ------------------------------
def plot_single_feature_alluvial(baseline_scenario, modified_scenario, feature_idx,
                                ylabels_raw_all, ylabels_latex_all, xlabels_dict,
                                savepath, figname_prefix, modified_par, fontsize=10,
                                relevance_threshold=0.05, top_n=15):
    ylabel_raw = ylabels_raw_all[feature_idx]
    ylabel_latex = ylabels_latex_all[feature_idx]

    baseline_ranks, baseline_effects = get_ranking_data(baseline_scenario, ylabel_raw)
    modified_ranks, modified_effects = get_ranking_data(modified_scenario, ylabel_raw)

    if not baseline_ranks and not modified_ranks:
        print(f"Warning: Could not load ranking data for feature {ylabel_raw}")
        return

    # Relevant parameters (use absolute effect to capture magnitude)
    relevant = set(p for p, v in baseline_effects.items() if abs(v) >= relevance_threshold) | \
               set(p for p, v in modified_effects.items() if abs(v) >= relevance_threshold)

    if not relevant:
        print(f"Warning: No relevant parameters for feature {ylabel_raw}")
        return

    # pick top_n by baseline rank (fallback to modified)
    candidates = sorted(list(relevant), key=lambda p: baseline_ranks.get(p, 999))
    if len(candidates) == 0:
        candidates = sorted(list(relevant), key=lambda p: modified_ranks.get(p, 999))
    chosen = candidates[:top_n]

    max_rank = max(len(baseline_ranks), len(modified_ranks), top_n)
    fig, ax = plt.subplots(figsize=(12, max(6, len(chosen)*0.4 + 2)))

    # consistent coloring per parameter
    cmap = plt.cm.get_cmap("tab20", max(2, len(chosen)))
    param_colors = {p: cmap(i) for i, p in enumerate(chosen)}

    x_baseline = 0.2
    x_modified = 0.8

    for param in chosen:
        br = baseline_ranks.get(param, max_rank)
        mr = modified_ranks.get(param, max_rank)
        br_c = min(br, max_rank)
        mr_c = min(mr, max_rank)
        yb = max_rank - br_c + 1
        ym = max_rank - mr_c + 1
        color = param_colors[param]
        eff_b = baseline_effects.get(param, 0.0)
        eff_m = modified_effects.get(param, 0.0)
        width_scale = 1.0 + max(abs(eff_b), abs(eff_m)) * 4.0

        draw_flow(ax, x_baseline, yb, x_modified, ym, color, alpha=0.8, width_scale=width_scale)
        ax.scatter([x_baseline], [yb], c=[color], s=80, zorder=10, edgecolor='black', linewidth=0.8)
        ax.scatter([x_modified], [ym], c=[color], s=80, zorder=10, edgecolor='black', linewidth=0.8)

        label = xlabels_dict.get(param, {}).get("latex", param)
        ax.text(x_baseline - 0.05, yb, f"{br}. {label}", ha='right', va='center', fontsize=fontsize-1)
        ax.text(x_modified + 0.05, ym, f"{mr}. {label}", ha='left', va='center', fontsize=fontsize-1)

    ax.axvline(x=x_baseline, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=x_modified, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_title(f'Parameter Ranking for {ylabel_latex} when modifying {modified_par}', fontsize=fontsize+2, weight='bold', pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ['top', 'right', 'bottom', 'left']:
        ax.spines[sp].set_visible(False)

    ylim = ax.get_ylim()
    plt.text(0.2, ylim[0]-0.25, 'Baseline', ha='center', va='bottom', fontsize=fontsize+1, weight='bold')
    plt.text(0.8, ylim[0]-0.25, 'Modified', ha='center', va='bottom', fontsize=fontsize+1, weight='bold')

    plt.tight_layout()
    os.makedirs(savepath, exist_ok=True)
    figname = f"{figname_prefix}_alluvial_{ylabel_raw}.png"
    outpath = os.path.join(savepath, figname)
    plt.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved: {outpath}")



# ------------------------------
# Driver
# ------------------------------
def generate_gsa_alluvial_plots(baseline_scenario, modified_scenario, xlabels_file, ylabels_file,
                               savepath, ylabels_dict_path, xlabels_dict_path, modified_par, fontsize=12,
                               figname_prefix="", relevance_threshold=0.05, top_n=15,
                               separators=True, show_legend=False):
    # ensure any Si_total.csv is available under 'output' in modified scenario
    mod_output_si = os.path.join(modified_scenario, "output", "Si_total.csv")
    mod_si = os.path.join(modified_scenario, "Si_total.csv")
    if not os.path.isfile(mod_output_si) and os.path.isfile(mod_si):
        os.makedirs(os.path.dirname(mod_output_si), exist_ok=True)
        os.system(f"cp {mod_si} {mod_output_si}")

    # This function should create the Rank_* files as side-effect and return feature indices and labels
    features_idx_list, ylabels_raw_all, ylabels_latex_all = generate_gsa_ranking_files(
        xlabels_file=xlabels_file,
        ylabels_file=ylabels_file,
        ylabels_dict=ylabels_dict_path,
        scenarios=[baseline_scenario, modified_scenario]
    )

    # read xlabels dictionary (expected to return _, xlabels_dict_all)
    xlabels_arr = np.loadtxt(xlabels_file, dtype=str)
    _, xlabels_dict_all = read_xlabels_dict(xlabels_dict_path, xlabels_arr)



    # individual feature plots (keeps previous single-feature behavior)
    for feature_idx in features_idx_list:
        plot_single_feature_alluvial(
            baseline_scenario=baseline_scenario,
            modified_scenario=modified_scenario,
            feature_idx=feature_idx,
            ylabels_raw_all=ylabels_raw_all,
            ylabels_latex_all=ylabels_latex_all,
            xlabels_dict=xlabels_dict_all,
            savepath=savepath,
            figname_prefix=figname_prefix,
            fontsize=fontsize,
            relevance_threshold=relevance_threshold,
            top_n=top_n,
            modified_par = modified_par
        )


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate GSA two-column alluvial plots")
    parser.add_argument('--baseline_scenario', required=True, help='Path to baseline scenario directory')
    parser.add_argument('--modified_scenario', required=True, help='Path to modified scenario directory')
    parser.add_argument('--xlabels', required=True, help='Path to xlabels file')
    parser.add_argument('--ylabels', required=True, help='Path to ylabels file')
    parser.add_argument('--savepath', required=True, help='Directory to save figures')
    parser.add_argument('--fontsize', type=int, default=12, help='Font size for labels/titles')
    parser.add_argument('--figname_prefix', required=True, help='Prefix for figure filenames')
    parser.add_argument('--ylabels_dict', type=str, required=True, help='Path to ylabels dictionary file')
    parser.add_argument('--xlabels_dict', type=str, required=True, help='Path to xlabels dictionary file')
    parser.add_argument('--relevance_threshold', type=float, default=0.05,
                        help='Minimum absolute effect size to consider parameter relevant (uses abs(effect))')
    parser.add_argument('--top_n', type=int, default=15, help='Maximum number of parameters to show per feature block')
    parser.add_argument('--no-separators', action='store_true', help='Disable horizontal separators between feature blocks')
    parser.add_argument('--legend', action='store_true', help='Include a legend mapping colors to parameter names')
    parser.add_argument('--modified_par', type=str, required=True, help='Name of the modified parameter (for title)')   
    args = parser.parse_args()

    generate_gsa_alluvial_plots(
        baseline_scenario=args.baseline_scenario,
        modified_scenario=args.modified_scenario,
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        savepath=args.savepath,
        ylabels_dict_path=args.ylabels_dict,
        xlabels_dict_path=args.xlabels_dict,
        fontsize=args.fontsize,
        figname_prefix=args.figname_prefix,
        relevance_threshold=args.relevance_threshold,
        top_n=args.top_n,
        separators=(not args.no_separators),
        show_legend=args.legend,
        modified_par=args.modified_par
    )

if __name__ == "__main__":
    main()
