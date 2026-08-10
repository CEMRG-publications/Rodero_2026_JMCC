#!/bin/bash

# ---------------------------------------------------------------------------
# Data and results locations.
#
#   DATA_ROOT     directory holding the simulation data, so that the scenarios
#                 live at $DATA_ROOT/HCM/<case>/scenarios/<scenario>/
#   RESULTS_ROOT  where the figures and tables are written
#                 (defaults to $DATA_ROOT/results)
#
# Set them in the environment before running, for example:
#   export DATA_ROOT=/path/to/data
# ---------------------------------------------------------------------------
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing HCM/ (see README)}"
RESULTS_ROOT="${RESULTS_ROOT:-$DATA_ROOT/results}"


# Base paths
BASELINE="${DATA_ROOT}/HCM/1/scenarios/53_more_samples/"
OUTPUT_DIR="${DATA_ROOT}/HCM/1/scenarios/53_more_samples/output"
XLABELS="${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels.txt"
YLABELS="${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels.txt"
SAVE="${RESULTS_ROOT}/"
YLABELS_DICT="${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json"
XLABELS_DICT="${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json"

# Create subfolder for modified interval results
SAVE_MODIFIED="${SAVE}/modified_intervals"
mkdir -p "$SAVE_MODIFIED"

# Loop through all GSA_* folders
for dir in "$OUTPUT_DIR"/GSA_*; do
    [ -d "$dir" ] || continue  # skip if not a directory
    dirname=$(basename "$dir")

    # Extract parameter name (everything between GSA_ and _lower)
    param=$(echo "$dirname" | sed -E 's/^GSA_(.*)_lower.*/\1/')

    # Determine increased or decreased
    if [[ "$dirname" == *"lower_50.0_upper_100.0"* ]]; then
        change="increased"
    elif [[ "$dirname" == *"lower_0.0_upper_50.0"* ]]; then
        change="decreased"
    else
        change="unknown"
    fi

    # Set figure prefix
    figname_prefix="${param}_${change}"

    # Look up LaTeX label from JSON
    latex_label=$(jq -r --arg p "$param" '.[$p].latex // empty' "$XLABELS_DICT")

    # If not found, fallback to param itself
    if [[ -z "$latex_label" || "$latex_label" == "null" ]]; then
        latex_label="$param"
    fi

    # Capitalize first letter of change
    change_cap=$(echo "$change" | sed 's/.*/\u&/')

    # Final title
    title="Parameter ranking with ${change_cap} ${latex_label}"

    # Run the command
    echo "Running: $param"_"$change"
    python3 plot_ranking_difference_heatmap.py \
      --baseline "$BASELINE" \
      --modified "$dir" \
      --xlabels "$XLABELS" \
      --ylabels "$YLABELS" \
      --savepath "$SAVE_MODIFIED" \
      --fontsize 25 \
      --figname_prefix "$figname_prefix" \
      --ylabels_dict "$YLABELS_DICT" \
      --xlabels_dict "$XLABELS_DICT" \
      --title "$title"
done
