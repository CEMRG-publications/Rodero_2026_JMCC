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


python3 plot_sensitivity_scatter.py \
  --scenarios ${DATA_ROOT}/HCM/1/scenarios/53_more_samples/ \
  ${DATA_ROOT}/HCM/2/scenarios/47_more_samples/ \
  ${DATA_ROOT}/HCM/3/scenarios/48_more_samples/ \
  ${DATA_ROOT}/HCM/4/scenarios/49_more_samples/ \
  ${DATA_ROOT}/HCM/5/scenarios/50_more_samples/ \
  --outputs LVedv LVedp LVesv LVpMax LVdpdtMax LVSV LVEF V_TAT \
  --xlabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/xlabels_to_plot.json \
  --savepath ${RESULTS_ROOT}/scatter \
  --ylabels_dict ${DATA_ROOT}/HCM/GSA_analysis/cycle/ylabels_filtered.json \
  --annotations ${DATA_ROOT}/HCM/GSA_analysis/cycle/annotations.json \
  --supertitle "Global sensitivity analysis in LV outputs" \
  --fontsize 16
