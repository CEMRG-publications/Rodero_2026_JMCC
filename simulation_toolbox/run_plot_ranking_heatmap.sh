#!/bin/bash
set -e  # Exit if any command fails

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


echo "=== Starting ranking heatmap generation ==="

# -----------------------------------
# Define baseline directories
# -----------------------------------
anatomy_base_dirs=(
  "${DATA_ROOT}/HCM/1/scenarios/53_more_samples/"
  "${DATA_ROOT}/HCM/2/scenarios/47_more_samples/"
  "${DATA_ROOT}/HCM/3/scenarios/48_more_samples/"
  "${DATA_ROOT}/HCM/4/scenarios/49_more_samples/"
  "${DATA_ROOT}/HCM/5/scenarios/50_more_samples/"
)

anatomy_names=("Mid-to-apical LVH" "LVOTO" "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern")

# -----------------------------------
# Define parameters for modified scenarios (Advanced HCM)
# -----------------------------------
PARAMS_advanced_HCM=(
  "CV_atria 0.0 50.0" 
  "aortic_area 0.0 50.0"
  "sysOrifice_area 0.0 50.0"
  "CV_ventricles 50.0 100.0"
  "Rsys 50.0 100.0"
  "Rpulm 50.0 100.0"
  "PaPref 50.0 100.0"
  "kArt 50.0 100.0"
  "ca50_A 50.0 100.0"
  "a_ventricles 50.0 100.0"
  "k_peri 50.0 100.0"
  "AoPref 50.0 100.0"
 "dr_V 50.0 100.0"
 "mu_V 50.0 100.0"
  "EDP_rv 50.0 100.0"
  "g_CaL 50.0 100.0"
 "perm50_A 50.0 100.0"
  "a_atria 50.0 100.0"
)

PARAMS_pharma=(
  "dr_V 0.0 50.0"
  "mu_V 0.0 50.0"
)
# -----------------------------------
# Function: check directories exist
# -----------------------------------
check_dir() {
  if [ ! -d "$1" ]; then
    echo "❌ ERROR: Directory does not exist: $1"
    exit 1
  fi
}

# -----------------------------------
# 1️⃣ Call for baseline scenarios only
# -----------------------------------
baseline_scenarios=()
baseline_names=()

for i in "${!anatomy_base_dirs[@]}"; do
  dir="${anatomy_base_dirs[i]}"
  check_dir "$dir"
  baseline_scenarios+=("$dir")
  baseline_names+=("${anatomy_names[i]} baseline")
done

# echo "=== Generating ranking heatmap for baseline scenarios ==="
python3 plot_ranking_heatmap.py \
  --scenarios "${baseline_scenarios[@]}" \
  --xlabels ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels.txt \
  --ylabels ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels.txt \
  --savepath ${RESULTS_ROOT}/heatmaps/baseline \
  --figname_prefix baseline \
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json \
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json \
  --fontsize 25 \
  --scenarios_names "${baseline_names[@]}"


echo "✅ Baseline ranking heatmap generated."

# -----------------------------------
# 2️⃣ Call for Advanced HCM (modified) scenarios
# -----------------------------------
advanced_scenarios=()
advanced_names=()

for i in "${!anatomy_base_dirs[@]}"; do
  base="${anatomy_base_dirs[i]}"
  anat_name="${anatomy_names[i]}"

  for param in "${PARAMS_advanced_HCM[@]}"; do
    set -- $param
    pname=$1
    lower=$2
    upper=$3

    mod_dir="${base}output/GSA_${pname}_lower_${lower}_upper_${upper}"
    check_dir "$mod_dir"
    advanced_scenarios+=("$mod_dir")
    advanced_names+=("${anat_name} with ${pname} (${lower}-${upper})")
  done
done

echo "=== Generating ranking heatmap for Advanced HCM scenarios ==="
python3 plot_ranking_heatmap.py \
  --scenarios "${advanced_scenarios[@]}" \
  --xlabels ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels.txt \
  --ylabels ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels.txt \
  --savepath ${RESULTS_ROOT}/heatmaps/functional_remodeling \
  --figname_prefix functional_remodeled \
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json \
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json \
  --fontsize 25 \
  --scenarios_names "${advanced_names[@]}" \
#   --skip_plots_and_summaries

echo "✅ Advanced HCM ranking heatmap generated."


# -----------------------------------
# 3 Call for pharma scenarios
# -----------------------------------
pharma_scenarios=()
pharma_names=()

for i in "${!anatomy_base_dirs[@]}"; do
  base="${anatomy_base_dirs[i]}"
  anat_name="${anatomy_names[i]}"

  for param in "${PARAMS_pharma[@]}"; do
    set -- $param
    pname=$1
    lower=$2
    upper=$3

    mod_dir="${base}output/GSA_${pname}_lower_${lower}_upper_${upper}"
    check_dir "$mod_dir"
    pharma_scenarios+=("$mod_dir")
    pharma_names+=("${anat_name} with ${pname} (${lower}-${upper})")
  done
done

echo "=== Generating ranking heatmap for pharma HCM scenarios ==="
python3 plot_ranking_heatmap.py \
  --scenarios "${pharma_scenarios[@]}" \
  --xlabels ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels.txt \
  --ylabels ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels.txt \
  --savepath ${RESULTS_ROOT}/heatmaps/pharma \
  --figname_prefix pharma \
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json \
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json \
  --fontsize 25 \
  --scenarios_names "${pharma_names[@]}" \
  --skip_plots_and_summaries

echo "✅ Pharma HCM ranking heatmap generated."
