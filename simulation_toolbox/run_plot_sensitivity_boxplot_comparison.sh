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


echo "=== Starting sensitivity boxplot comparison for all chambers ==="

# -----------------------------------
# Define parameter ranges for advanced HCM
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

# -----------------------------------
# Baseline directories
# -----------------------------------
anatomy1_base=${DATA_ROOT}/HCM/1/scenarios/53_more_samples/
anatomy2_base=${DATA_ROOT}/HCM/2/scenarios/47_more_samples/
anatomy3_base=${DATA_ROOT}/HCM/3/scenarios/48_more_samples/
anatomy4_base=${DATA_ROOT}/HCM/4/scenarios/49_more_samples/
anatomy5_base=${DATA_ROOT}/HCM/5/scenarios/50_more_samples/

# -----------------------------------
# Build modified scenario paths for each anatomy
# -----------------------------------
anatomy1_mod=()
anatomy2_mod=()
anatomy3_mod=()
anatomy4_mod=()
anatomy5_mod=()

for param in "${PARAMS_advanced_HCM[@]}"; do
  set -- $param
  pname=$1
  lower=$2
  upper=$3

  anatomy1_mod+=(${DATA_ROOT}/HCM/1/scenarios/53_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
  anatomy2_mod+=(${DATA_ROOT}/HCM/2/scenarios/47_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
  anatomy3_mod+=(${DATA_ROOT}/HCM/3/scenarios/48_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
  anatomy4_mod+=(${DATA_ROOT}/HCM/4/scenarios/49_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
  anatomy5_mod+=(${DATA_ROOT}/HCM/5/scenarios/50_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
done

# -----------------------------------
# Check directories
# -----------------------------------
check_dir() {
  if [ ! -d "$1" ]; then
    echo "❌ ERROR: Directory does not exist: $1"
    exit 1
  fi
}

echo "🔍 Checking baseline directories..."
for base in "$anatomy1_base" "$anatomy2_base" "$anatomy3_base" "$anatomy4_base" "$anatomy5_base"; do
  check_dir "$base"
done

echo "🔍 Checking modified scenario directories..."
for dir in "${anatomy1_mod[@]}" "${anatomy2_mod[@]}" "${anatomy3_mod[@]}" "${anatomy4_mod[@]}" "${anatomy5_mod[@]}"; do
  check_dir "$dir"
done

echo "✅ All required directories found."

# -----------------------------------
# Define anatomy names
# -----------------------------------
ANATOMY_NAMES=("Mid-to-apical LVH" "LVOTO" "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern")

# -----------------------------------
# Define all chamber configurations
# -----------------------------------
declare -A CONFIGS

CONFIGS["LV"]="LVedv LVedp LVesv LVpMax LVSV LVEF LVdpdtMax V_TAT|LV_outputs|Global sensitivity analysis in LV outputs on functional remodeling scenarios"
CONFIGS["RV"]="RVedv RVedp RVesv RVpMax RVSV RVEF RVdpdtMax V_TAT|RV_outputs|Global sensitivity analysis in RV outputs on functional remodeling scenarios"
CONFIGS["LA"]="LAedv LAvMax LApMax LAinflV A_TAT|LA_outputs|Global sensitivity analysis in LA outputs on functional remodeling scenarios"
CONFIGS["RA"]="RAedv RAvMax RApMax RAinflV A_TAT|RA_outputs|Global sensitivity analysis in RA outputs on functional remodeling scenarios"
CONFIGS["ART"]="diastAP systAP pulseAP mAP diastPAP systPAP pulsePAP mPAP|artery_outputs|Global sensitivity analysis in arterial outputs on functional remodeling scenarios"

# -----------------------------------
# Common arguments
# -----------------------------------
COMMON_ARGS=(
  --n_anatomies 5
  --anatomy_names "${ANATOMY_NAMES[@]}"
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json
  --exclusions ${DATA_ROOT}/HCM/GSA_analysis/cycle/parameters_exclusions.json
  --savepath ${RESULTS_ROOT}/boxplots_functional_remodeling
  --fontsize 22
)

# -----------------------------------
# Loop over all chamber configs
# -----------------------------------
for chamber in LV RV LA RA ART; do
  IFS="|" read -r outputs figname supertitle <<< "${CONFIGS[$chamber]}"

  echo ""
  echo "=== 🧠 Running sensitivity comparison for ${chamber} ==="

  cmd=(python3 plot_sensitivity_boxplot_comparison.py
    "${COMMON_ARGS[@]}"
    --outputs $outputs
    --figname sensitivity_boxplot_${figname}.png
    --supertitle "$supertitle"
    --anatomy1_baseline "$anatomy1_base"
    --anatomy2_baseline "$anatomy2_base"
    --anatomy3_baseline "$anatomy3_base"
    --anatomy4_baseline "$anatomy4_base"
    --anatomy5_baseline "$anatomy5_base"
    --anatomy1_modified "${anatomy1_mod[@]}"
    --anatomy2_modified "${anatomy2_mod[@]}"
    --anatomy3_modified "${anatomy3_mod[@]}"
    --anatomy4_modified "${anatomy4_mod[@]}"
    --anatomy5_modified "${anatomy5_mod[@]}"
  )

  echo "Executing: ${cmd[*]}"
  "${cmd[@]}"
  echo "✅ ${chamber} sensitivity comparison done."
done

echo ""
echo "🎉 All chamber sensitivity comparisons completed successfully."