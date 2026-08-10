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


python3 plot_sensitivity_boxplot.py \
  --scenarios ${DATA_ROOT}/HCM/1/scenarios/53_more_samples/ \
  ${DATA_ROOT}/HCM/2/scenarios/47_more_samples/ \
  ${DATA_ROOT}/HCM/3/scenarios/48_more_samples/ \
  ${DATA_ROOT}/HCM/4/scenarios/49_more_samples/ \
  ${DATA_ROOT}/HCM/5/scenarios/50_more_samples/ \
  --outputs LVedv LVedp LVesv LVpMax LVSV LVEF LVdpdtMax V_TAT \
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json \
  --savepath ${RESULTS_ROOT}/boxplot \
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json \
  --supertitle "Global sensitivity analysis in LV outputs" \
  --fontsize 18 \
  --anatomy_names "Mid-to-apical LVH" LVOTO "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern" \
  --group_colors ${DATA_ROOT}/HCM/GSA_analysis/cycle/group_colors.json \
  --figname sensitivity_boxplot_LV_outputs.png \

python3 plot_sensitivity_boxplot.py \
  --scenarios ${DATA_ROOT}/HCM/1/scenarios/53_more_samples/ \
  ${DATA_ROOT}/HCM/2/scenarios/47_more_samples/ \
  ${DATA_ROOT}/HCM/3/scenarios/48_more_samples/ \
  ${DATA_ROOT}/HCM/4/scenarios/49_more_samples/ \
  ${DATA_ROOT}/HCM/5/scenarios/50_more_samples/ \
  --outputs RVedv RVedp RVesv RVpMax RVEF RVSV RVdpdtMax V_TAT\
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json \
  --savepath ${RESULTS_ROOT}/boxplot \
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json \
  --supertitle "Global sensitivity analysis in RV outputs" \
  --fontsize 18 \
  --anatomy_names "Mid-to-apical LVH" LVOTO "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern" \
  --group_colors ${DATA_ROOT}/HCM/GSA_analysis/cycle/group_colors.json \
  --figname sensitivity_boxplot_RV_outputs.png \


  python3 plot_sensitivity_boxplot.py \
  --scenarios ${DATA_ROOT}/HCM/1/scenarios/53_more_samples/ \
  ${DATA_ROOT}/HCM/2/scenarios/47_more_samples/ \
  ${DATA_ROOT}/HCM/3/scenarios/48_more_samples/ \
  ${DATA_ROOT}/HCM/4/scenarios/49_more_samples/ \
  ${DATA_ROOT}/HCM/5/scenarios/50_more_samples/ \
  --outputs LAedv LAvMax LApMax LAinflV A_TAT\
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json \
  --savepath ${RESULTS_ROOT}/boxplot \
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json \
  --supertitle "Global sensitivity analysis in LA outputs" \
  --fontsize 18 \
  --anatomy_names "Mid-to-apical LVH" LVOTO "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern" \
  --group_colors ${DATA_ROOT}/HCM/GSA_analysis/cycle/group_colors.json \
  --figname sensitivity_boxplot_LA_outputs.png \

    python3 plot_sensitivity_boxplot.py \
  --scenarios ${DATA_ROOT}/HCM/1/scenarios/53_more_samples/ \
  ${DATA_ROOT}/HCM/2/scenarios/47_more_samples/ \
  ${DATA_ROOT}/HCM/3/scenarios/48_more_samples/ \
  ${DATA_ROOT}/HCM/4/scenarios/49_more_samples/ \
  ${DATA_ROOT}/HCM/5/scenarios/50_more_samples/ \
  --outputs RAedv RAvMax RApMax RAinflV A_TAT\
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json \
  --savepath ${RESULTS_ROOT}/boxplot \
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json \
  --supertitle "Global sensitivity analysis in RA outputs" \
  --fontsize 18 \
  --anatomy_names "Mid-to-apical LVH" LVOTO "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern" \
  --group_colors ${DATA_ROOT}/HCM/GSA_analysis/cycle/group_colors.json \
  --figname sensitivity_boxplot_RA_outputs.png \


      python3 plot_sensitivity_boxplot.py \
  --scenarios ${DATA_ROOT}/HCM/1/scenarios/53_more_samples/ \
  ${DATA_ROOT}/HCM/2/scenarios/47_more_samples/ \
  ${DATA_ROOT}/HCM/3/scenarios/48_more_samples/ \
  ${DATA_ROOT}/HCM/4/scenarios/49_more_samples/ \
  ${DATA_ROOT}/HCM/5/scenarios/50_more_samples/ \
  --outputs diastAP systAP pulseAP mAP diastPAP systPAP pulsePAP mPAP\
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json \
  --savepath ${RESULTS_ROOT}/boxplot \
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json \
  --supertitle "Global sensitivity analysis in arterial outputs" \
  --fontsize 18 \
  --anatomy_names "Mid-to-apical LVH" LVOTO "Isolated basal LVH" "Milder asymmetric LVH" "Undifferentiated pattern" \
  --group_colors ${DATA_ROOT}/HCM/GSA_analysis/cycle/group_colors.json \
  --figname sensitivity_boxplot_artery_outputs.png \