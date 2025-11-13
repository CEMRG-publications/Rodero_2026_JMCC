#!/bin/bash
set -e  # Exit if any command fails

echo "=== Starting sensitivity boxplot comparison for all chambers (single combined call) ==="

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
anatomy1_base=/media/croderog/Bob/HCM/1/scenarios/53_more_samples/
anatomy2_base=/media/croderog/SeagateExpansionDrive/HCM/2/scenarios/47_more_samples/
anatomy3_base=/data/HCM/3/scenarios/48_more_samples/
anatomy4_base=/media/croderog/SeagateExpansionDrive/HCM/4/scenarios/49_more_samples/
anatomy5_base=/media/croderog/Bob/HCM/5/scenarios/50_more_samples/

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

  anatomy1_mod+=(/media/croderog/Bob/HCM/1/scenarios/53_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
  anatomy2_mod+=(/media/croderog/SeagateExpansionDrive/HCM/2/scenarios/47_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
  anatomy3_mod+=(/data/HCM/3/scenarios/48_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
  anatomy4_mod+=(/media/croderog/SeagateExpansionDrive/HCM/4/scenarios/49_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
  anatomy5_mod+=(/media/croderog/Bob/HCM/5/scenarios/50_more_samples/output/GSA_${pname}_lower_${lower}_upper_${upper})
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
# Define all outputs (combined from all chambers)
# -----------------------------------
ALL_OUTPUTS=(
  LVedv LVedp LVesv LVpMax LVSV LVEF LVdpdtMax V_TAT
  RVedv RVedp RVesv RVpMax RVSV RVEF RVdpdtMax
  LAedv LAvMax LApMax LAinflV
  RAedv RAvMax RApMax RAinflV
  diastAP systAP pulseAP mAP diastPAP systPAP pulsePAP mPAP
)

# -----------------------------------
# Define save path for combined output
# -----------------------------------
SAVE_PATH=/media/croderog/Bob/HCM/GSA_analysis/boxplots_advanced_disease/all_chambers
mkdir -p "$SAVE_PATH"

# -----------------------------------
# Run combined comparison
# -----------------------------------
echo ""
echo "=== 🧠 Running single combined sensitivity comparison for all chambers ==="

cmd=(python3 print_comparison_analysis.py
  --n_anatomies 5
  --anatomy_names "Mid-to-apical LVH" LVOTO "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern"
  --xlabels_dict /media/croderog/Bob/HCM/GSA_analysis/cycle/xlabels_to_plot.json
  --ylabels_dict /media/croderog/Bob/HCM/GSA_analysis/cycle/ylabels_filtered.json
  --exclusions /media/croderog/Bob/HCM/GSA_analysis/cycle/parameters_exclusions.json
  --savepath "$SAVE_PATH"
  --outputs "${ALL_OUTPUTS[@]}"
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

# echo "Executing: ${cmd[*]}"
"${cmd[@]}"

echo ""
echo "✅ Combined sensitivity comparison for all chambers completed."
echo "📁 Results saved in: ${SAVE_PATH}"
