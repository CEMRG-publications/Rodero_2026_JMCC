"""
Batch-plot PV loops for *every working cycle* of each HCM case.

For each case it reads the scenario's output_mask_beat_5.txt (0-indexed; 1 = the
simulation reached beat 5), then renders one PV-loop figure per working cycle
into a per-case subfolder:

    <output_root>/
        HCM1/ cycle_3.png  cycle_6.png  ...   (~500 plots)
        HCM2/ ...
        ...

The per-figure plotting (4-chamber 2x2 grid, beats coloured light->dark green,
single shared legend) is delegated to plot_case in plot_HCM_pv_loops.py.

The simulation data is spread across external drives, so each cycle is searched
for across all candidate drives within the mask's scenario folder.
"""
import os
import sys
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_HCM_pv_loops import plot_case, CHAMBERS  # noqa: E402

DRIVES = [
    "/media/croderog/SeagateExpansionDrive",
    "/media/croderog/Seagate Expansion Drive",
    "/media/croderog/Elements",
    "/media/croderog/Elements_1",
    "/media/croderog/Bob",
    "/data",
]

# label -> (patient folder, scenario number, phenotype name)
CASES = {
    "HCM1": (1, 53, "Mid-to-apical LVH"),
    "HCM2": (2, 47, "LVOTO"),
    "HCM3": (3, 48, "Isolated basal LVH"),
    "HCM4": (4, 49, "Milder asymmetric LVH"),
    "HCM5": (5, 50, "Undifferentiated pattern"),
}


def first_existing(rel_candidates):
    """Return the first (drive, scenario, path) that exists, else None."""
    for scen, rel in rel_candidates:
        for d in DRIVES:
            p = os.path.join(d, rel)
            if os.path.exists(p):
                return d, scen, p
    return None


def find_mask(folder, scen):
    """Locate the output_mask, preferring the _more_samples variant."""
    cands = []
    for s in (f"{scen}_more_samples", f"{scen}"):
        cands.append((s, f"HCM/{folder}/scenarios/{s}/output/output_mask_beat_5.txt"))
    return first_existing(cands)


def find_bcl(folder, scen):
    cands = []
    for s in (f"{scen}_more_samples", f"{scen}"):
        cands.append((s, f"HCM/{folder}/scenarios/{s}/json_files/clinical_data.json"))
    hit = first_existing(cands)
    if hit is None:
        return None
    with open(hit[2]) as f:
        return json.load(f).get("general", {}).get("BCL")


def find_cycle_dir(folder, scen_name, idx):
    """Find cycle_<idx> within scen_name across drives, with all 4 cav files."""
    for d in DRIVES:
        cdir = os.path.join(d, f"HCM/{folder}/scenarios/{scen_name}/simulations/cycle_{idx}")
        if all(os.path.isfile(os.path.join(cdir, f"cav.{c}.csv")) for c in CHAMBERS):
            return cdir
    return None


def process_case(label, output_root, dpi, limit, force):
    folder, scen, phenotype = CASES[label]
    mask_hit = find_mask(folder, scen)
    if mask_hit is None:
        print(f"[{label}] no output_mask found, skipping.")
        return
    _, scen_name, mask_path = mask_hit
    bcl = find_bcl(folder, scen)
    if bcl is None:
        print(f"[{label}] no BCL (clinical_data.json) found, skipping.")
        return

    with open(mask_path) as f:
        mask = [l.strip() for l in f if l.strip() != ""]
    working = [i for i, v in enumerate(mask) if v == "1"]

    out_dir = os.path.join(output_root, label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{label}] {phenotype} | scenario {scen_name} | BCL {bcl} ms | "
          f"{len(working)} working cycles in mask -> {out_dir}")

    plotted = skipped_missing = skipped_existing = errored = 0
    for n, idx in enumerate(working):
        if limit and plotted >= limit:
            break
        out_png = os.path.join(out_dir, f"pv_loops_cycle_{idx}.png")
        if not force and os.path.isfile(out_png):
            skipped_existing += 1
            continue
        cdir = find_cycle_dir(folder, scen_name, idx)
        if cdir is None:
            skipped_missing += 1
            continue
        try:
            plot_case(label=f"cycle_{idx}", sim_dir=cdir, BCL=float(bcl),
                      n_beats=5, t_start=0.0, output_dir=out_dir, overlay=False,
                      title=f"{phenotype} (cycle {idx})", dpi=dpi)
        except Exception as e:
            plt.close('all')  # don't leak the half-built figure
            errored += 1
            print(f"[{label}]   SKIP cycle_{idx}: {type(e).__name__}: {e}")
            continue
        plotted += 1
        if plotted % 50 == 0:
            print(f"[{label}]   ...{plotted} plotted")

    print(f"[{label}] DONE: {plotted} plotted, {skipped_existing} already existed, "
          f"{skipped_missing} not found locally, {errored} errored (bad/corrupt files).")


def main():
    ap = argparse.ArgumentParser(description="Batch PV-loop plots for all working HCM cycles.")
    ap.add_argument('--output_root', type=str,
                    default="/media/croderog/Bob/HCM/figures/all_working_cycles",
                    help="Root folder; one subfolder per case is created inside.")
    ap.add_argument('--cases', nargs='+', default=list(CASES.keys()),
                    help="Subset of cases to process (default: all).")
    ap.add_argument('--dpi', type=int, default=150,
                    help="Output resolution (default 150; lower keeps the batch fast/small).")
    ap.add_argument('--limit', type=int, default=None,
                    help="Max plots per case (for testing).")
    ap.add_argument('--force', action='store_true',
                    help="Re-plot even if the output PNG already exists "
                         "(default: skip existing, so the run is resumable).")
    args = ap.parse_args()

    for label in args.cases:
        process_case(label, args.output_root, args.dpi, args.limit, args.force)


if __name__ == '__main__':
    main()
