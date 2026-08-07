#!/bin/bash
set -e  # Exit if any command fails

echo "=== Starting VAS (Variability in Anatomical Sensitivity) comparison plots ==="

# -----------------------------------
# Define parameter ranges for pharma
# -----------------------------------
PARAMS_pharma=(
  "wfrac_V 0.0 50.0"
  "mu_V 0.0 50.0"
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

for param in "${PARAMS_pharma[@]}"; do
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
# Define all chamber configurations
# -----------------------------------
declare -A CONFIGS

CONFIGS["LV"]="LVedv LVedp LVesv LVpMax LVSV LVEF LVdpdtMax V_TAT"
CONFIGS["RV"]="RVedv RVedp RVesv RVpMax RVSV RVEF RVdpdtMax V_TAT"
CONFIGS["LA"]="LAedv LAvMax LApMax LAinflV A_TAT"
CONFIGS["RA"]="RAedv RAvMax RApMax RAinflV A_TAT"
CONFIGS["ART"]="diastAP systAP pulseAP mAP diastPAP systPAP pulsePAP mPAP"
CONFIGS["ALL"]="LVedv LVedp LVesv LVpMax LVSV LVEF LVdpdtMax V_TAT \
RVedv RVedp RVesv RVpMax RVSV RVEF RVdpdtMax \
LAedv LAvMax LApMax LAinflV A_TAT \
RAedv RAvMax RApMax RAinflV \
diastAP systAP pulseAP mAP diastPAP systPAP pulsePAP mPAP"

# -----------------------------------
# Common arguments
# -----------------------------------
COMMON_ARGS=(
  --n_anatomies 5
  --anatomy_names "Mid-to-apical LVH" LVOTO "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern"
  --xlabels_dict /media/croderog/Bob/HCM/GSA_analysis/cycle/xlabels_to_plot.json
  --ylabels_dict /media/croderog/Bob/HCM/GSA_analysis/cycle/ylabels_filtered.json
  --exclusions /media/croderog/Bob/HCM/GSA_analysis/cycle/parameters_exclusions.json
  --fontsize 14
  --threshold 0.05
)

# -----------------------------------
# Loop over all chamber configs
# -----------------------------------
for chamber in "${!CONFIGS[@]}"; do
  IFS="|" read -r outputs <<< "${CONFIGS[$chamber]}"

  echo ""
  echo "=== 📊 Generating VAS comparison plot for ${chamber} ==="

  cmd=(python3 plot_vas_comparison.py
    "${COMMON_ARGS[@]}"
    --outputs $outputs
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
    --savepath /media/croderog/Bob/HCM/GSA_analysis/VAS_plots_pharma/${chamber}
    --figname vas_comparison_${chamber}.png
    --supertitle "Variability in anatomical sensitivity for the ${chamber} outputs"
  )

  "${cmd[@]}"
  echo "✅ ${chamber} VAS plot completed."
done

echo ""
echo "🎉 All VAS comparison plots generated successfully!"
echo ""
echo "📁 Results saved in: /media/croderog/Bob/HCM/GSA_analysis/VAS_plots/"