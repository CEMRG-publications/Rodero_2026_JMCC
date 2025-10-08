import argparse
from collections import defaultdict
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os

from common.utils import generate_gsa_ranking_files, read_xlabels_dict

# ------------------------
# Helper functions
# ------------------------
def get_color_for_param(param, xlabels_dict):
    for k, v in xlabels_dict.items():
        if v.get("latex") == param:
            return v.get("color", "#8F8F8F")
    return "#8F8F8F"

def get_color_with_alpha(base_color, effect):
    alpha = 0.3 if effect < 0.05 else 1.0
    return to_rgba(base_color, alpha)

# Helper function: determine color
def get_color(effect, rank_range, very_light_grey, high_variability, grey, rank_range_threshold):
    if effect < 0.05:
        return very_light_grey, 'low_importance'
    elif rank_range >= rank_range_threshold:
        return high_variability, 'high_variability'
    else:
        return grey, 'high_importance'

# ------------------------
    # Helper: check if param appears in top_n of any scenario
# ------------------------
def appears_in_top_n_any_scenario(param, all_scenarios, param_ranks, top_n, all_params):
    for s in all_scenarios:
        if param_ranks[param].get(s, len(all_params)+1) <= top_n:
            return True
    return False


def plot_bump_chart_from_rankings_color_specific(
    scenarios,
    savepath,
    legend,
    xlabels_dict,
    fontsize=12,
    gsa_mode="Si_total",
    mode="max",
    figname="bump_chart.png",
    rank_file=None,
    title=None,
    top_n=40
):
    """
    Generates a bump chart showing parameter rankings with parameter-specific colors.

    - Top_n parameters in scenario 1 have continuous lines and points.
    - Parameters outside top_n in scenario 1 but in top_n later appear with unique markers.
    - Colors are assigned per parameter using xlabels_dict[param]["color"].
    - Low-importance parameters (effect < 0.05) are shown with transparent colors.
    - Parameters outside top_n are shown in a legend if they appear in top_n of any scenario.
    """


    param_ranks = defaultdict(dict)
    param_effects = defaultdict(dict)
    all_params = set()
    param_names_per_scenario = []

    if rank_file is None:
        rank_file = f"Rank_{gsa_mode}_{mode}.txt"

    # ------------------------
    # Load rankings and effects
    # ------------------------
    for scenario in scenarios:
        rank_file_full_path = os.path.join(scenario, "output", rank_file)
        scenario_name = scenario.rstrip("/").split("/")[-3]

        with open(rank_file_full_path, "r") as f:
            params_in_this_scenario = []
            for i, line in enumerate(f.readlines()):
                param, value = line.strip().split("\t")
                param_ranks[param][scenario_name] = i + 1
                param_effects[param][scenario_name] = float(value)
                all_params.add(param)
                params_in_this_scenario.append(param)
            param_names_per_scenario.append(params_in_this_scenario)

    # ------------------------
    # Consistency check
    # ------------------------
    first_param_set = set(param_names_per_scenario[0])
    first_rank_file = os.path.join(scenarios[0], "output", rank_file)
    for idx, param_list in enumerate(param_names_per_scenario[1:], start=1):
        this_param_set = set(param_list)
        this_rank_file = os.path.join(scenarios[idx], "output", rank_file)
        missing = first_param_set - this_param_set
        extra = this_param_set - first_param_set
        if missing or extra:
            print(f"\nParameter names mismatch detected!")
            print(f"First scenario rank file: {first_rank_file}")
            print(f"Current scenario rank file: {this_rank_file}")
            if missing:
                print(f"Parameters missing in scenario {idx+1}: {sorted(missing)}")
            if extra:
                print(f"Extra parameters in scenario {idx+1}: {sorted(extra)}")
            raise ValueError("Parameter names mismatch between scenarios.")

    if len(all_params) != len(param_names_per_scenario[0]):
        raise ValueError("Possible repeated parameter names in the xlabels file.")

    all_scenarios = [s.rstrip("/").split("/")[-3] for s in scenarios]
    scenario1, last_scenario = all_scenarios[0], all_scenarios[-1]

    # ------------------------
    # Identify top_n parameters for first and last scenario
    # ------------------------
    top_n_params = sorted(first_param_set, key=lambda p: param_ranks[p][scenario1])[:top_n]
    top_n_last_params = sorted(first_param_set, key=lambda p: param_ranks[p][last_scenario])[:top_n]
    outside_top_n_params = sorted(all_params - set(top_n_params))

    # ------------------------
    # Assign markers to outside params
    # ------------------------
    special_markers = ['s', 'D', '^', 'v', 'P', '*', 'X', 'h', '<', '>']
    param_to_marker = {
        param: special_markers[i % len(special_markers)]
        for i, param in enumerate(outside_top_n_params)
    }

    # ------------------------
    # Initialize plot
    # ------------------------
    fig_width = 10
    fig_height = max(6, len(top_n_params) * 0.25)
    _, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)

    outside_params_plotted = set()



    # ------------------------
    # Plot top_n parameters
    # ------------------------
    for param in top_n_params:
        ranks = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        ranks_clipped = [r if r <= top_n else np.nan for r in ranks]
        base_color = get_color_for_param(param, xlabels_dict)

        for i in range(len(all_scenarios)-1):
            r1, r2 = ranks_clipped[i], ranks_clipped[i+1]
            if np.isnan(r1) or np.isnan(r2):
                continue
            effect = max(effects[i], effects[i+1])
            color = get_color_with_alpha(base_color, effect)
            ax.plot([i, i+1], [r1, r2], color=color, linewidth=2, zorder=1)

        for ix, (rank, effect) in enumerate(zip(ranks_clipped, effects)):
            if np.isnan(rank):
                continue
            color = get_color_with_alpha(base_color, effect)
            ax.scatter(ix, rank, s=200, marker='.', color=color, zorder=3)

    # ------------------------
    # Plot outside_top_n parameters
    # ------------------------
    for param in outside_top_n_params:
        ranks_all = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects_all = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        marker = param_to_marker[param]
        base_color = get_color_for_param(param, xlabels_dict)
        prev_rank, prev_ix = None, None
        appeared_in_top_n = False

        for ix, (rank, effect) in enumerate(zip(ranks_all, effects_all)):
            color = get_color_with_alpha(base_color, effect)
            if rank <= top_n:
                appeared_in_top_n = True
                ax.scatter(ix, rank, s=100, marker=marker,
                           facecolors='white', edgecolors=color,
                           linewidths=1.5, zorder=3)

                if prev_rank is not None and prev_rank <= top_n:
                    ax.plot([prev_ix, ix], [prev_rank, rank],
                            color=color, linewidth=1.5, zorder=2)

            prev_rank, prev_ix = rank, ix

        if appeared_in_top_n:
            outside_params_plotted.add(param)

    # ------------------------
    # Label parameters
    # ------------------------
    for param in top_n_params:
        rank1 = param_ranks[param][scenario1]
        label = xlabels_dict.get(param, {}).get("latex", param)
        ax.text(-0.3, rank1, label, fontsize=fontsize-2, va='center', ha='right')

    for param in top_n_last_params:
        rank_last = param_ranks[param][last_scenario]
        label = xlabels_dict.get(param, {}).get("latex", param)
        ax.text(len(all_scenarios)-0.7, rank_last, label, fontsize=fontsize-2, va='center', ha='left')

    # ------------------------
    # Axis and title formatting
    # ------------------------
    ax.invert_yaxis()
    ax.set_ylim(top_n+0.5, 0.5)
    ax.set_xticks(np.arange(len(all_scenarios)))
    ax.set_xticklabels(legend, fontsize=fontsize, rotation=30)
    ax.axes.yaxis.set_visible(False)
    ax.spines[:].set_visible(False)

    if title is None:
        title = "Bump Chart of Parameter Rankings in Different Meshes"
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold')

    # ------------------------
    # Legend for parameter categories
    # ------------------------
    legend_mapping = {
        "Rsys": "Left side parameter",
        "Rpulm": "Right side parameter",
        "a_ventricles": "Either side parameter"
    }

    legend_handles = []
    for param, label in legend_mapping.items():
        color = xlabels_dict.get(param, {}).get("color", "#8F8F8F")
        handle = Line2D([0], [0], color=color, lw=3, label=label)
        legend_handles.append(handle)

    ax.legend(
        handles=legend_handles,
        loc='upper center',
        fontsize=fontsize-1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.4),
        ncol=3
    )

    # ------------------------
    # Legend for outside top_n parameters
    # ------------------------
    outside_legend_params = (outside_params_plotted - set(top_n_last_params))
    if outside_legend_params:
        special_legend_handles = []
        for param in sorted(outside_legend_params):
            marker = param_to_marker[param]
            base_color = get_color_for_param(param, xlabels_dict)
            effects = [param_effects[param].get(s, 0.0) for s in all_scenarios]
            avg_effect = np.mean(effects)
            color = get_color_with_alpha(base_color, avg_effect)
            label = xlabels_dict.get(param, {}).get("latex", param)

            handle = Line2D(
                [0], [0],
                marker=marker,
                color='black',
                label=label,
                markerfacecolor='white',
                markeredgecolor=color,
                markersize=10,
                linestyle='None',
                markeredgewidth=1.5
            )
            special_legend_handles.append(handle)

        ax.figure.legend(
            handles=special_legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.2),
            fontsize=fontsize-2,
            frameon=False,
            ncol=min(len(special_legend_handles), 5)
        )

    # ------------------------
    # Save
    # ------------------------
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Bump chart saved to: {output_path}")


def plot_bump_chart_from_rankings(
    scenarios,
    savepath,
    legend,
    fontsize=12,
    gsa_mode="Si_total",
    mode="max",
    figname="bump_chart.png",
    rank_file=None,
    title=None,
    top_n=40,
    xlabels_to_plot=None,
    rank_range_threshold=100
):
    """
    Generates a bump chart showing parameter rankings with simplified coloring.
    Coloring logic:
        - Low importance: effect < 0.05
        - High importance & stable: effect ≥ 0.05 and small variability
        - High importance & highly variable: effect ≥ 0.05 and rank_range ≥ rank_range_threshold
    """


    if xlabels_to_plot is None:
        xlabels_to_plot = {}

    param_ranks = defaultdict(dict)
    param_effects = defaultdict(dict)
    all_params = set()
    param_names_per_scenario = []

    if rank_file is None:
        rank_file = f"Rank_{gsa_mode}_{mode}.txt"

    # ------------------------
    # Load rankings and effects
    # ------------------------
    for scenario in scenarios:
        rank_file_full_path = os.path.join(scenario, "output", rank_file)
        scenario_name = scenario.rstrip("/").split("/")[-3]

        with open(rank_file_full_path, "r") as f:
            params_in_this_scenario = []
            for i, line in enumerate(f.readlines()):
                param, value = line.strip().split("\t")
                param_ranks[param][scenario_name] = i + 1  # 1-based rank
                param_effects[param][scenario_name] = float(value)
                all_params.add(param)
                params_in_this_scenario.append(param)
            param_names_per_scenario.append(params_in_this_scenario)

    # ------------------------
    # Consistency check
    # ------------------------
    first_param_set = set(param_names_per_scenario[0])
    first_rank_file = os.path.join(scenarios[0], "output", rank_file)
    for idx, param_list in enumerate(param_names_per_scenario[1:], start=1):
        this_param_set = set(param_list)
        this_rank_file = os.path.join(scenarios[idx], "output", rank_file)
        missing = first_param_set - this_param_set
        extra = this_param_set - first_param_set
        if missing or extra:
            print(f"\nParameter names mismatch detected!")
            print(f"First scenario rank file: {first_rank_file}")
            print(f"Current scenario rank file: {this_rank_file}")
            if missing:
                print(f"Parameters missing in scenario {idx+1}: {sorted(missing)}")
            if extra:
                print(f"Extra parameters in scenario {idx+1}: {sorted(extra)}")
            raise ValueError("Parameter names mismatch between scenarios.")

    if len(all_params) != len(param_names_per_scenario[0]):
        raise ValueError("Possible repeated parameter names in the xlabels file.")

    all_scenarios = [s.rstrip("/").split("/")[-3] for s in scenarios]
    scenario1, last_scenario = all_scenarios[0], all_scenarios[-1]

    # ------------------------
    # Identify top_n parameters for first and last scenario
    # ------------------------
    top_n_params = sorted(first_param_set, key=lambda p: param_ranks[p][scenario1])[:top_n]
    top_n_last_params = sorted(first_param_set, key=lambda p: param_ranks[p][last_scenario])[:top_n]
    outside_top_n_params = sorted(all_params - set(top_n_params))

    # ------------------------
    # Assign markers to outside params
    # ------------------------
    special_markers = ['s', 'D', '^', 'v', 'P', '*', 'X', 'h', '<', '>']
    param_to_marker = {
        param: special_markers[i % len(special_markers)]
        for i, param in enumerate(outside_top_n_params)
    }

    # ------------------------
    # Initialize plot
    # ------------------------
    _, ax = plt.subplots(figsize=(16, max(6, len(top_n_params) * 0.25)), constrained_layout=True)
    very_light_grey = "#8888887A"
    grey = "#888888"
    high_variability = "#c08978"
    colors_used_top_n, outside_params_plotted = set(), set()

    

    # ------------------------
    # Track high variability parameters across all parameters (not just top_n)
    # ------------------------
    high_var_params_all = set()
    for param in all_params:
        ranks = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        rank_range = max(ranks) - min(ranks)
        if max(effects) >= 0.05 and rank_range >= rank_range_threshold:
            high_var_params_all.add(param)

    # ------------------------
    # Plot top_n parameters (from first scenario)
    # ------------------------
    for param in top_n_params:
        ranks = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        ranks_clipped = [r if r <= top_n else np.nan for r in ranks]
        rank_range = max(ranks) - min(ranks)

        # Lines
        for i in range(len(all_scenarios)-1):
            r1, r2 = ranks_clipped[i], ranks_clipped[i+1]
            if np.isnan(r1) or np.isnan(r2):
                continue
            color, color_label = get_color(max(effects[i], effects[i+1]), 
                                           rank_range,
                                           very_light_grey, high_variability, grey,
                                           rank_range_threshold)
            colors_used_top_n.add(color_label)  # Track colors used for legend
            ax.plot([i, i+1], [r1, r2], color=color, linewidth=2, zorder=1)

        # Points
        for ix, (rank, effect) in enumerate(zip(ranks_clipped, effects)):
            if np.isnan(rank):
                continue
            color, color_label = get_color(effect, rank_range, very_light_grey, high_variability, grey, rank_range_threshold)
            colors_used_top_n.add(color_label)  # Track colors used for legend
            ax.scatter(ix, rank, s=200, marker='.', color=color, zorder=3)

    

    # ------------------------
    # Plot outside_top_n parameters and track those for legend
    # ------------------------
    high_var_params_for_legend = set()

    for param in outside_top_n_params:
        ranks_all = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects_all = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        rank_range = max(ranks_all) - min(ranks_all)
        marker = param_to_marker[param]
        color, _ = get_color(max(effects_all), rank_range, very_light_grey, high_variability, grey, rank_range_threshold)

        prev_ix, prev_rank = None, None
        for ix, rank in enumerate(ranks_all):
            if rank <= top_n:
                ax.scatter(ix, rank, s=100, marker=marker, facecolors='white',
                           edgecolors=color, linewidths=1.5, zorder=3)
                if prev_ix is not None:
                    ax.plot([prev_ix, ix], [prev_rank, rank], color=color,
                            linewidth=1.5, zorder=1, linestyle='-')
                prev_ix, prev_rank = ix, rank
                outside_params_plotted.add(param)
            else:
                prev_ix, prev_rank = None, None

        # Add to legend if highly variable AND appears in top_n of any scenario
        if param in high_var_params_all and appears_in_top_n_any_scenario(param, all_scenarios, param_ranks, top_n, all_params):
            high_var_params_for_legend.add(param)

    # ------------------------
    # Label parameters on left (first scenario) and right (last scenario)
    # ------------------------
    for param in top_n_params:
        rank1 = param_ranks[param][scenario1]
        label = xlabels_to_plot.get(param, {}).get("latex", param)
        ax.text(-0.3, rank1, label, fontsize=fontsize-2, va='center', ha='right')

    for param in top_n_last_params:
        rank_last = param_ranks[param][last_scenario]
        label = xlabels_to_plot.get(param, {}).get("latex", param)
        ax.text(len(all_scenarios)-0.7, rank_last, label, fontsize=fontsize-2, va='center', ha='left')

    # ------------------------
    # Axis & title formatting
    # ------------------------
    ax.invert_yaxis()
    ax.set_ylim(top_n+0.5, 0.5)
    ax.set_xticks(np.arange(len(all_scenarios)))
    ax.set_xticklabels(legend, fontsize=fontsize, rotation=30)
    ax.axes.yaxis.set_visible(False)
    ax.spines[:].set_visible(False)
    ax.set_title(title or "Bump Chart of Parameter Rankings in Different Meshes",
                 fontsize=fontsize+2, fontweight='bold')

    # ------------------------
    # Legends
    # ------------------------
    legend_elements = []
    if 'low_importance' in colors_used_top_n:
        legend_elements.append(Line2D([0], [0], color=very_light_grey, lw=2, label='Low importance'))
    if 'high_importance' in colors_used_top_n and 'low_importance' in colors_used_top_n:
        legend_elements.append(Line2D([0], [0], color=grey, lw=2, label='High importance'))
    # Show high variability legend if any high variability parameter exists in the plot
    if high_var_params_all and 'low_importance' in colors_used_top_n:
        legend_elements.append(Line2D([0], [0], color=high_variability, lw=2,
                                      label='High importance & highly variable'))
    if high_var_params_all and 'low_importance' not in colors_used_top_n:
        legend_elements.append(Line2D([0], [0], color=high_variability, lw=2,
                                      label='Highly variable'))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.4), fontsize=fontsize-2, frameon=False, ncol=3)

    # Combine outside params that were plotted + high variability params to show in legend,
    # excluding those already labeled on right side in last scenario top_n
    outside_legend_params = (outside_params_plotted | high_var_params_for_legend) - set(top_n_last_params)
    if outside_legend_params:
        special_legend_handles = [
            Line2D([0], [0], marker=param_to_marker[param], color='black',
                   label=xlabels_to_plot.get(param, {}).get("latex", param),
                   markerfacecolor='white', markersize=10, linestyle='None', markeredgewidth=1.5)
            for param in sorted(outside_legend_params)
        ]
        ax.figure.legend(handles=special_legend_handles, loc='lower center',
                         bbox_to_anchor=(0.5, -0.2), fontsize=fontsize-2,
                         frameon=False, ncol=min(len(special_legend_handles), 5))

    # ------------------------
    # Save
    # ------------------------
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Bump chart saved to: {output_path}")


def generate_gsa_bump_chart(
    scenarios,
    xlabels_file,
    ylabels_file,
    savepath,
    ylabels_dict,
    xlabels_dict,
    fontsize=14,
    figname_preffix="",
    legend=None,
    top_n=10,
    rank_range_threshold=100
):
    
    features_idx_list, ylabels_raw_all, ylabels_latex_all = generate_gsa_ranking_files(
        xlabels_file=xlabels_file,
        ylabels_file=ylabels_file,
        ylabels_dict=ylabels_dict,
        scenarios=scenarios
    )

    xlabels = np.loadtxt(xlabels_file, dtype=str)
    _, xlabels_dict_all = read_xlabels_dict(xlabels_dict, xlabels)


    # plot_bump_chart_from_rankings_color_specific(
    #     scenarios=scenarios,
    #     savepath=savepath,
    #     legend=legend,
    #     xlabels_dict=xlabels_dict_all,
    #     fontsize=fontsize,
    #     gsa_mode="Si_total",
    #     mode="max",
    #     figname=f"{figname_preffix}_bump_chart_colored.png",
    #     rank_file=None,  # Use default rank file
    #     title="Bump chart of parameter rankings in different patients for all functional outputs",
    #     top_n=top_n
    #     )

    
    plot_bump_chart_from_rankings(
        scenarios=scenarios,
        savepath=savepath,
        fontsize=fontsize,
        gsa_mode="Si_total",
        mode="max",
        figname=f"{figname_preffix}_bump_chart.png",
        title="Bump chart of parameter rankings in different patients for all functional outputs",
        legend=legend,
        xlabels_to_plot=xlabels_dict_all,
        top_n=top_n,
        rank_range_threshold=rank_range_threshold
        )
    


    for feature_idx in features_idx_list:

        # plot_bump_chart_from_rankings_color_specific(
        #     scenarios=scenarios,
        #     savepath=savepath,
        #     legend=legend,
        #     xlabels_dict=xlabels_dict_all,
        #     fontsize=fontsize,
        #     gsa_mode="Si_total",
        #     mode="max",
        #     figname=f"{figname_preffix}_bump_chart_{ylabels_raw_all[feature_idx]}_colored.png",
        #     rank_file=f"Rank_Si_total_max_{ylabels_raw_all[feature_idx]}.txt",
        #     title=f"Bump chart of parameter rankings in different patients for {ylabels_latex_all[feature_idx]}",
        #     top_n=top_n
        # )

            
        plot_bump_chart_from_rankings(
            scenarios=scenarios,
            savepath=savepath,
            fontsize=fontsize,
            gsa_mode="Si_total",
            mode="max",
            figname=f"{figname_preffix}_bump_chart_{ylabels_raw_all[feature_idx]}.png",
            rank_file=f"Rank_Si_total_max_{ylabels_raw_all[feature_idx]}.txt",
            title=f"Bump chart of parameter rankings in different patients for {ylabels_latex_all[feature_idx]}",
            legend=legend,
            xlabels_to_plot=xlabels_dict_all,
            top_n=top_n,
            rank_range_threshold=rank_range_threshold
        )


def main():
    parser = argparse.ArgumentParser(description="Plot GSA Visualizations")
    parser.add_argument('--scenarios', nargs='+', required=True, help='Paths to the scenario folders')
    parser.add_argument('--xlabels', required=True, help='Path to the xlabels file')
    parser.add_argument('--ylabels', required=True, help='Path to the ylabels file')
    parser.add_argument('--savepath', required=True, help='Path to save the figures')
    parser.add_argument('--fontsize', type=int, default=14, help='Font size for the plot')
    parser.add_argument('--figname_preffix', required=True, help='Prefix for the figure names')
    parser.add_argument('--legend', nargs='*', default=[], help='Legend labels')
    parser.add_argument('--ylabels_dict', type=str, required=True, help='Path to the ylabels dictionary file (optional)')
    parser.add_argument('--xlabels_dict', type=str, required=True, help='Path to the xlabels dictionary file (optional)')
    parser.add_argument('--top_n', type=int, default=100, help='Number of top parameters to display in the bump chart')
    parser.add_argument('--rank_range_threshold', type=int, default=100, help="Threshold for what it is considered varible.")
    args = parser.parse_args()


    generate_gsa_bump_chart(
        scenarios=args.scenarios,
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        savepath=args.savepath,
        fontsize=args.fontsize,
        figname_preffix=args.figname_preffix,
        legend=args.legend,
        ylabels_dict=args.ylabels_dict,
        xlabels_dict=args.xlabels_dict,
        top_n=args.top_n,
        rank_range_threshold=args.rank_range_threshold
    )

if __name__ == "__main__":
    main()