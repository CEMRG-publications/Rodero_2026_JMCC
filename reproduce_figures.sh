#!/bin/bash
#
# Reproduce the drug-scenario analysis of Rodero et al. (2026) from a clean clone.
#
# With no arguments it runs against the small example dataset shipped in
# example_data/, which is enough to check that the installation works and to see
# the analysis end to end. Point DATA_ROOT at a full download from Zenodo (see
# the README) to reproduce the paper figures over the complete output set.
#
#   ./reproduce_figures.sh                      # example data -> ./results
#   DATA_ROOT=/path/to/zenodo ./reproduce_figures.sh
#   RESULTS_ROOT=/somewhere ./reproduce_figures.sh
#
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_ROOT="${DATA_ROOT:-$REPO_DIR/example_data}"
export RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
export MPLBACKEND="${MPLBACKEND:-Agg}"   # render without a display

PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo " Reproducing the drug-scenario analysis"
echo "============================================================"
echo " Repository  : $REPO_DIR"
echo " DATA_ROOT   : $DATA_ROOT"
echo " RESULTS_ROOT: $RESULTS_ROOT"
echo " Python      : $($PYTHON --version 2>&1)"
echo

if [ ! -d "$DATA_ROOT/HCM" ]; then
    echo "ERROR: $DATA_ROOT/HCM does not exist."
    echo "Set DATA_ROOT to a directory containing HCM/<case>/scenarios/<scenario>/."
    exit 1
fi

# Check the dependencies before doing any work, so a missing package is reported
# once rather than as a traceback halfway through.
$PYTHON - <<'EOF'
import importlib.util, sys
missing = [m for m in ("numpy", "pandas", "matplotlib", "scipy", "seaborn")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("ERROR: missing packages: " + ", ".join(missing) + "\n"
             "Install them with:  poetry install")
EOF

mkdir -p "$RESULTS_ROOT"
cd "$REPO_DIR/simulation_toolbox"

run_step () {
    echo "------------------------------------------------------------"
    echo " $1"
    echo "------------------------------------------------------------"
    shift
    if bash "$@" > "$RESULTS_ROOT/$(basename "$1" .sh).log" 2>&1; then
        echo "   done"
    else
        echo "   FAILED, see $RESULTS_ROOT/$(basename "$1" .sh).log"
        return 1
    fi
}

run_step "Variability in anatomical sensitivity (VAS), drug scenarios" \
         run_plot_VAS_comparison_pharma.sh
run_step "Sensitivity boxplots, drug scenarios" \
         run_plot_sensitivity_boxplot_pharma.sh
run_step "Printed comparison statistics, drug scenarios" \
         run_print_comparison_analysis_pharma.sh
run_step "Functional versus anatomical contribution, drug scenarios" \
         run_print_comparison_functional_vs_anatomical_pharma.sh

echo
echo "============================================================"
n_fig=$(find "$RESULTS_ROOT" -name "*.png" | wc -l)
n_tab=$(find "$RESULTS_ROOT" -name "*.csv" -o -name "*.txt" | wc -l)
echo " Finished: $n_fig figures and $n_tab tables in"
echo "   $RESULTS_ROOT"
echo "============================================================"
