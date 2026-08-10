#!/usr/bin/env python3
"""
Generate the parameter ranking files that the figure scripts read.

The global sensitivity analysis writes one Si_total.csv per scenario, a matrix of
total-order Sobol indices with one row per parameter and one column per output.
Most figure scripts do not read that matrix directly: they read the derived
Rank_Si_total_max_<output>.txt files, which rank the parameters for each output.
This script produces those files, so that a freshly cloned repository can be
taken from the shipped Si_total.csv matrices to the figures in one step.

Run it once per set of scenarios before the figure scripts, for example:

    export DATA_ROOT=../example_data
    python generate_ranking_files.py \\
        --xlabels  $DATA_ROOT/HCM/1/scenarios/53_more_samples/data/xlabels.txt \\
        --ylabels  $DATA_ROOT/HCM/1/scenarios/53_more_samples/data/ylabels.txt \\
        --ylabels_dict $DATA_ROOT/HCM/GSA_analysis/cycle/ylabels_filtered.json \\
        --scenarios $DATA_ROOT/HCM/1/scenarios/53_more_samples/output

Each scenario argument is a directory holding Si_total.csv (or a directory whose
output/ subfolder holds it). The ranking files are written next to it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.utils import generate_gsa_ranking_files  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Generate Rank_Si_total_max_*.txt files from Si_total.csv matrices.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--xlabels', required=True,
                        help='Path to xlabels.txt (one parameter name per line).')
    parser.add_argument('--ylabels', required=True,
                        help='Path to ylabels.txt (one output name per line).')
    parser.add_argument('--ylabels_dict', required=True,
                        help='JSON dictionary selecting which outputs to rank.')
    parser.add_argument('--scenarios', nargs='+', required=True,
                        help='One or more scenario folders containing Si_total.csv.')

    args = parser.parse_args()

    # Check the inputs up front so a missing file is reported before any work
    for path in [args.xlabels, args.ylabels, args.ylabels_dict]:
        if not os.path.isfile(path):
            raise SystemExit(f"Cannot find {path}")
    missing = [s for s in args.scenarios
               if not (os.path.isfile(os.path.join(s, "Si_total.csv"))
                       or os.path.isfile(os.path.join(s, "output", "Si_total.csv")))]
    if missing:
        raise SystemExit("No Si_total.csv in:\n  " + "\n  ".join(missing))

    print("=" * 70)
    print(f"Generating ranking files for {len(args.scenarios)} scenario(s)")
    print("=" * 70)

    features_idx, ylabels_raw, _ = generate_gsa_ranking_files(
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        ylabels_dict=args.ylabels_dict,
        scenarios=args.scenarios,
    )

    print("=" * 70)
    print(f"Ranked {len(features_idx)} outputs of the {len(ylabels_raw)} available.")
    for scenario in args.scenarios:
        target = scenario if os.path.isfile(os.path.join(scenario, "Si_total.csv")) \
            else os.path.join(scenario, "output")
        n = len([f for f in os.listdir(target) if f.startswith("Rank_Si_total_max")])
        print(f"  {target}: {n} ranking files")
    print("Done.")


if __name__ == "__main__":
    main()
