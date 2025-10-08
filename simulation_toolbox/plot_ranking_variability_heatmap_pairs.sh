#!/bin/bash
set -euo pipefail  # safer bash settings: exit on errors, unset vars, and pipe failures

# Define all 5 scenario paths
scenarios=(
    "/media/croderog/Bob/HCM/1/scenarios/53_more_samples/"
    "/media/croderog/SeagateExpansionDrive/HCM/2/scenarios/47_more_samples/"
    "/data/HCM/3/scenarios/48_more_samples/"
    "/media/croderog/SeagateExpansionDrive/HCM/4/scenarios/49_more_samples/"
    "/media/croderog/Bob/HCM/5/scenarios/50_more_samples/"
)

# Common arguments
xlabels="/media/croderog/Bob/HCM/GSA_analysis/cycle/xlabels.txt"
ylabels="/media/croderog/Bob/HCM/GSA_analysis/cycle/ylabels.txt"
savepath="/media/croderog/Bob/HCM/GSA_analysis/baseline_comparison"
figname_prefix="cycle"
ylabels_dict="/media/croderog/Bob/HCM/GSA_analysis/cycle/ylabels_filtered.json"
xlabels_dict="/media/croderog/Bob/HCM/GSA_analysis/cycle/xlabels_to_plot.json"
fontsize=25

# Always quote variable expansions in mkdir
mkdir -p "$savepath"

# Loop over all unique pairs of scenarios
for ((i = 0; i < ${#scenarios[@]}; i++)); do
    for ((j = i + 1; j < ${#scenarios[@]}; j++)); do
        s1="${scenarios[i]}"
        s2="${scenarios[j]}"
        echo "Running pair: $s1 and $s2"

        # Correct arithmetic expansion for readability
        pairname="${figname_prefix}_pair_$((i + 1))_$((j + 1))"

        # Properly quote all variable expansions (especially paths with spaces)
        python3 plot_ranking_variability_heatmap.py \
            --scenarios "$s1" "$s2" \
            --xlabels "$xlabels" \
            --ylabels "$ylabels" \
            --savepath "$savepath" \
            --figname_prefix "$pairname" \
            --ylabels_dict "$ylabels_dict" \
            --xlabels_dict "$xlabels_dict" \
            --fontsize "$fontsize"
    done
done
