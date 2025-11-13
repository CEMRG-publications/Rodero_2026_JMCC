#!/bin/bash
set -e  # Exit if any command fails

echo "=== Starting baseline sensitivity summary analysis ==="

# -----------------------------------
# Baseline directories
# -----------------------------------
anatomy1_base=/media/croderog/Bob/HCM/1/scenarios/53_more_samples/
anatomy2_base=/media/croderog/SeagateExpansionDrive/HCM/2/scenarios/47_more_samples/
anatomy3_base=/data/HCM/3/scenarios/48_more_samples/
anatomy4_base=/media/croderog/SeagateExpansionDrive/HCM/4/scenarios/49_more_samples/
anatomy5_base=/media/croderog/Bob/HCM/5/scenarios/50_more_samples/

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

echo "✅ All required directories found."

# -----------------------------------
# Define all output groups
# -----------------------------------

# All LV outputs
LV_OUTPUTS=(LVedv LVedp LVesv LVpMax LVSV LVEF LVdpdtMax V_TAT)

# All RV outputs
RV_OUTPUTS=(RVedv RVedp RVesv RVpMax RVSV RVEF RVdpdtMax V_TAT)

# All LA outputs
LA_OUTPUTS=(LAedv LAvMax LApMax LAinflV A_TAT)

# All RA outputs
RA_OUTPUTS=(RAedv RAvMax RApMax RAinflV A_TAT)

# All arterial outputs
ART_OUTPUTS=(diastAP systAP pulseAP mAP diastPAP systPAP pulsePAP mPAP)

# All outputs combined
ALL_OUTPUTS=(
  LVedv LVedp LVesv LVpMax LVSV LVEF LVdpdtMax V_TAT
  RVedv RVedp RVesv RVpMax RVSV RVEF RVdpdtMax
  LAedv LAvMax LApMax LAinflV A_TAT
  RAedv RAvMax RApMax RAinflV
  diastAP systAP pulseAP mAP diastPAP systPAP pulsePAP mPAP
)

# -----------------------------------
# Common arguments
# -----------------------------------
SCENARIOS=(
  "$anatomy1_base"
  "$anatomy2_base"
  "$anatomy3_base"
  "$anatomy4_base"
  "$anatomy5_base"
)

ANATOMY_NAMES=("Mid-to-apical LVH" "LVOTO" "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern")

COMMON_ARGS=(
  --xlabels_dict /media/croderog/Bob/HCM/GSA_analysis/cycle/xlabels_to_plot.json
  --ylabels_dict /media/croderog/Bob/HCM/GSA_analysis/cycle/ylabels_filtered.json
  --threshold 0.05
  --anatomy_names "${ANATOMY_NAMES[@]}"
)

# -----------------------------------
# Create output groups JSON file for grouped analyses
# -----------------------------------
OUTPUT_GROUPS_FILE=/tmp/output_groups.json

# -----------------------------------
# Analysis 1: LV outputs only
# -----------------------------------
echo ""
echo "=== 🫀 Running baseline sensitivity summary for LV OUTPUTS ==="

cat > "$OUTPUT_GROUPS_FILE" << EOF
{
  "LV_volume": ["LVedv", "LVesv", "LVSV"],
  "LV_pressure": ["LVedp", "LVpMax"],
  "LV_function": ["LVEF", "LVdpdtMax"],
  "LV_all": ["LVedv", "LVedp", "LVesv", "LVpMax", "LVSV", "LVEF", "LVdpdtMax", "V_TAT"]
}
EOF

python3 plot_baseline_sensitivity_summary.py \
  --scenarios "${SCENARIOS[@]}" \
  --outputs "${LV_OUTPUTS[@]}" \
  --savepath /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/LV_outputs \
  --output_groups "$OUTPUT_GROUPS_FILE" \
  "${COMMON_ARGS[@]}"

echo "✅ LV outputs analysis complete."

# -----------------------------------
# Analysis 2: RV outputs only
# -----------------------------------
echo ""
echo "=== 🫀 Running baseline sensitivity summary for RV OUTPUTS ==="

cat > "$OUTPUT_GROUPS_FILE" << EOF
{
  "RV_volume": ["RVedv", "RVesv", "RVSV"],
  "RV_pressure": ["RVedp", "RVpMax"],
  "RV_function": ["RVEF", "RVdpdtMax"],
  "RV_all": ["RVedv", "RVedp", "RVesv", "RVpMax", "RVSV", "RVEF", "RVdpdtMax", "V_TAT"]
}
EOF

python3 plot_baseline_sensitivity_summary.py \
  --scenarios "${SCENARIOS[@]}" \
  --outputs "${RV_OUTPUTS[@]}" \
  --savepath /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/RV_outputs \
  --output_groups "$OUTPUT_GROUPS_FILE" \
  "${COMMON_ARGS[@]}"

echo "✅ RV outputs analysis complete."

# -----------------------------------
# Analysis 3: LA outputs only
# -----------------------------------
echo ""
echo "=== 🫀 Running baseline sensitivity summary for LA OUTPUTS ==="

cat > "$OUTPUT_GROUPS_FILE" << EOF
{
  "LA_volume": ["LAedv", "LAinflV"],
  "LA_velocity": ["LAvMax"],
  "LA_pressure": ["LApMax"],
  "LA_all": ["LAedv", "LAvMax", "LApMax", "LAinflV", "A_TAT"]
}
EOF

python3 plot_baseline_sensitivity_summary.py \
  --scenarios "${SCENARIOS[@]}" \
  --outputs "${LA_OUTPUTS[@]}" \
  --savepath /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/LA_outputs \
  --output_groups "$OUTPUT_GROUPS_FILE" \
  "${COMMON_ARGS[@]}"

echo "✅ LA outputs analysis complete."

# -----------------------------------
# Analysis 4: RA outputs only
# -----------------------------------
echo ""
echo "=== 🫀 Running baseline sensitivity summary for RA OUTPUTS ==="

cat > "$OUTPUT_GROUPS_FILE" << EOF
{
  "RA_volume": ["RAedv", "RAinflV"],
  "RA_velocity": ["RAvMax"],
  "RA_pressure": ["RApMax"],
  "RA_all": ["RAedv", "RAvMax", "RApMax", "RAinflV", "A_TAT"]
}
EOF

python3 plot_baseline_sensitivity_summary.py \
  --scenarios "${SCENARIOS[@]}" \
  --outputs "${RA_OUTPUTS[@]}" \
  --savepath /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/RA_outputs \
  --output_groups "$OUTPUT_GROUPS_FILE" \
  "${COMMON_ARGS[@]}"

echo "✅ RA outputs analysis complete."

# -----------------------------------
# Analysis 5: Arterial outputs only
# -----------------------------------
echo ""
echo "=== 🫀 Running baseline sensitivity summary for ARTERIAL OUTPUTS ==="

cat > "$OUTPUT_GROUPS_FILE" << EOF
{
  "systemic_arterial": ["diastAP", "systAP", "pulseAP", "mAP"],
  "pulmonary_arterial": ["diastPAP", "systPAP", "pulsePAP", "mPAP"],
  "all_arterial": ["diastAP", "systAP", "pulseAP", "mAP", "diastPAP", "systPAP", "pulsePAP", "mPAP"]
}
EOF

python3 plot_baseline_sensitivity_summary.py \
  --scenarios "${SCENARIOS[@]}" \
  --outputs "${ART_OUTPUTS[@]}" \
  --savepath /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/arterial_outputs \
  --output_groups "$OUTPUT_GROUPS_FILE" \
  "${COMMON_ARGS[@]}"

echo "✅ Arterial outputs analysis complete."

# -----------------------------------
# Analysis 6: All ventricular outputs (LV + RV)
# -----------------------------------
echo ""
echo "=== 🫀 Running baseline sensitivity summary for VENTRICULAR OUTPUTS ==="

VENTRICULAR_OUTPUTS=(
  LVedv LVedp LVesv LVpMax LVSV LVEF LVdpdtMax V_TAT
  RVedv RVedp RVesv RVpMax RVSV RVEF RVdpdtMax
)

cat > "$OUTPUT_GROUPS_FILE" << EOF
{
  "ventricular_volume": ["LVedv", "LVesv", "LVSV", "RVedv", "RVesv", "RVSV"],
  "ventricular_pressure": ["LVedp", "LVpMax", "RVedp", "RVpMax"],
  "ventricular_function": ["LVEF", "LVdpdtMax", "RVEF", "RVdpdtMax"],
  "all_ventricular": ["LVedv", "LVedp", "LVesv", "LVpMax", "LVSV", "LVEF", "LVdpdtMax", "RVedv", "RVedp", "RVesv", "RVpMax", "RVSV", "RVEF", "RVdpdtMax", "V_TAT"]
}
EOF

python3 plot_baseline_sensitivity_summary.py \
  --scenarios "${SCENARIOS[@]}" \
  --outputs "${VENTRICULAR_OUTPUTS[@]}" \
  --savepath /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/ventricular_outputs \
  --output_groups "$OUTPUT_GROUPS_FILE" \
  "${COMMON_ARGS[@]}"

echo "✅ Ventricular outputs analysis complete."

# -----------------------------------
# Analysis 7: All atrial outputs (LA + RA)
# -----------------------------------
echo ""
echo "=== 🫀 Running baseline sensitivity summary for ATRIAL OUTPUTS ==="

ATRIAL_OUTPUTS=(
  LAedv LAvMax LApMax LAinflV A_TAT
  RAedv RAvMax RApMax RAinflV
)

cat > "$OUTPUT_GROUPS_FILE" << EOF
{
  "atrial_volume": ["LAedv", "LAinflV", "RAedv", "RAinflV"],
  "atrial_velocity": ["LAvMax", "RAvMax"],
  "atrial_pressure": ["LApMax", "RApMax"],
  "all_atrial": ["LAedv", "LAvMax", "LApMax", "LAinflV", "RAedv", "RAvMax", "RApMax", "RAinflV", "A_TAT"]
}
EOF

python3 plot_baseline_sensitivity_summary.py \
  --scenarios "${SCENARIOS[@]}" \
  --outputs "${ATRIAL_OUTPUTS[@]}" \
  --savepath /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/atrial_outputs \
  --output_groups "$OUTPUT_GROUPS_FILE" \
  "${COMMON_ARGS[@]}"

echo "✅ Atrial outputs analysis complete."

# -----------------------------------
# Analysis 8: All outputs combined
# -----------------------------------
echo ""
echo "=== 🫀 Running baseline sensitivity summary for ALL OUTPUTS ==="

cat > "$OUTPUT_GROUPS_FILE" << EOF
{
  "LV": ["LVedv", "LVedp", "LVesv", "LVpMax", "LVSV", "LVEF", "LVdpdtMax"],
  "RV": ["RVedv", "RVedp", "RVesv", "RVpMax", "RVSV", "RVEF", "RVdpdtMax"],
  "LA": ["LAedv", "LAvMax", "LApMax", "LAinflV"],
  "RA": ["RAedv", "RAvMax", "RApMax", "RAinflV"],
  "Arterial": ["diastAP", "systAP", "pulseAP", "mAP", "diastPAP", "systPAP", "pulsePAP", "mPAP"],
  "Ventricular": ["LVedv", "LVedp", "LVesv", "LVpMax", "LVSV", "LVEF", "LVdpdtMax", "RVedv", "RVedp", "RVesv", "RVpMax", "RVSV", "RVEF", "RVdpdtMax"],
  "Atrial": ["LAedv", "LAvMax", "LApMax", "LAinflV", "RAedv", "RAvMax", "RApMax", "RAinflV"]
}
EOF

python3 plot_baseline_sensitivity_summary.py \
  --scenarios "${SCENARIOS[@]}" \
  --outputs "${ALL_OUTPUTS[@]}" \
  --savepath /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/all_outputs \
  --output_groups "$OUTPUT_GROUPS_FILE" \
  "${COMMON_ARGS[@]}"

echo "✅ All outputs analysis complete."

# -----------------------------------
# Analysis 9: All ventricular hemodynamics outputs (LV + RV)
# -----------------------------------
echo ""
echo "=== 🫀 Running baseline sensitivity summary for VENTRICULAR HEMODYNAMICS OUTPUTS ==="

VENTRICULAR_HEMODYNAMICS_OUTPUTS=(
  LVedv LVedp LVesv LVpMax LVSV LVEF 
  RVedv RVedp RVesv RVpMax RVSV RVEF 
)

cat > "$OUTPUT_GROUPS_FILE" << EOF
{
  "ventricular_volume": ["LVedv", "LVesv", "LVSV", "RVedv", "RVesv", "RVSV"],
  "ventricular_pressure": ["LVedp", "LVpMax", "RVedp", "RVpMax"],
  "ventricular_function": ["LVEF", "LVdpdtMax", "RVEF", "RVdpdtMax"],
  "ventricular_hemodynamics":  [ "LVdpdtMax", "RVdpdtMax", "LVedv", "LVedp", "LVesv", "LVpMax", "LVSV", "LVEF", "RVedv", "RVedp", "RVesv", "RVpMax", "RVSV", "RVEF"],
  "all_ventricular": ["LVedv", "LVedp", "LVesv", "LVpMax", "LVSV", "LVEF", "LVdpdtMax", "RVedv", "RVedp", "RVesv", "RVpMax", "RVSV", "RVEF", "RVdpdtMax", "V_TAT"]
}
EOF

python3 plot_baseline_sensitivity_summary.py \
  --scenarios "${SCENARIOS[@]}" \
  --outputs "${VENTRICULAR_HEMODYNAMICS_OUTPUTS[@]}" \
  --savepath /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/ventricular_hemodynamics_outputs \
  --output_groups "$OUTPUT_GROUPS_FILE" \
  "${COMMON_ARGS[@]}"

echo "✅ Ventricular outputs analysis complete."

# -----------------------------------
# Cleanup temporary files
# -----------------------------------
rm -f "$OUTPUT_GROUPS_FILE"

echo ""
echo "🎉 All baseline sensitivity summary analyses completed successfully!"
echo "📊 Results saved in: /media/croderog/Bob/HCM/GSA_analysis/baseline_summary/"
