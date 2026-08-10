"""
Controlled single-cell (openCARP `bench`) dose-response validation of the
myosin-inhibitor drug models.

Holding a representative ventricular myocyte fixed (ionic params + all other
Land contraction params), we sweep ONE contraction parameter across its sampled
range and measure the resulting steady-state active-tension twitch. This
isolates each drug's mechanistic knob (no other parameters varying):

  - Mavacamten -> dr_V  (Land `dr`,  ratio of strongly bound actin-myosin units)
  - Aficamten  -> mu_V  (Land `mu`,  weak-to-strong crossbridge transition rate)
  - Tref_V     (Land `Tref`, reference tension) as positive control / re-map candidate

For each swept value we run `bench.pt` (ToRORd_dynCl + LandHumanStress, same
protocol as the cohort: 500 beats at the patient BCL, isometric at slack length)
and report peak/developed/diastolic tension and relaxation kinetics. Output: a
dose-response figure (tension vs parameter) + transient overlays + a CSV.

The point: show whether REDUCING each parameter reproduces the drugs' clinical
negative inotropy at the cell level (it does for Tref and weakly for mu_V; it
does NOT for dr_V).
"""

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = os.environ.get("BENCH", "bench.pt")

# Scenario providing the representative cell and the parameter ranges (HCM1)
SCENARIO = os.path.join(os.environ.get("DATA_ROOT", ""), "HCM/1/scenarios/53_more_samples")
BCL = 1017.0

# swept parameters: name -> (X.txt column, Land-file key, colour)
# Titles use the manuscript's parameter wording (not code names); SYMBOL gives the
# short math symbol used on the x-axes and in the transient legends.
SWEEPS: "OrderedDict[str, Tuple[int, str, str]]" = OrderedDict([
    ("Ratio of strongly bound\nactin-myosin units", (7, "dr", "#999999")),
    ("Weak-to-strong crossbridge\ntransition rate (aficamten)", (12, "mu", "#F1BF00")),
    ("Ratio of weakly bound\nactin-myosin units (mavacamten)", (8, "wfrac", "#CE6A0E")),
])
SYMBOL = {"dr": r"$r_s$", "mu": r"$\mu$", "wfrac": r"$w_{frac}$"}

# which half of each parameter's range represents the drug (reduces contractility);
# dr is None: reducing it RAISES tension (the wrong direction), so no drug interval is shaded.
DRUG_HALF = {"mu": "lower", "wfrac": "lower", "dr": None}


# --------------------------------------------------------------------------- #
# param-file helpers
# --------------------------------------------------------------------------- #

def parse_par(s: str) -> "OrderedDict[str, str]":
    d = OrderedDict()
    for tok in s.strip().split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def serialise_par(d: "OrderedDict[str, str]") -> str:
    return ",".join(f"{k}={v}" for k, v in d.items())


def pick_representative_cell(X: np.ndarray, ss_dir: str) -> int:
    """Index of the cell whose ventricular Land params (cols 3..12) are closest
    to the cohort median, among cells that actually have output."""
    cols = list(range(3, 13))
    med = np.median(X[:, cols], axis=0)
    std = X[:, cols].std(axis=0)
    have = sorted(int(d) for d in os.listdir(ss_dir)
                  if os.path.isfile(f"{ss_dir}/{d}/ToRORd_dynCl.Tension.bin"))
    best, bestd = None, np.inf
    for i in have:
        if i >= X.shape[0]:
            continue
        dist = np.sum(((X[i, cols] - med) / std) ** 2)
        if dist < bestd:
            best, bestd = i, dist
    return best


# --------------------------------------------------------------------------- #
# run bench
# --------------------------------------------------------------------------- #

def run_one(args) -> Tuple[str, float, str]:
    """Run one bench sweep point. Returns (out_dir, value, key)."""
    out_dir, pp_ion, pp_land, n_beats, bcl = args
    os.makedirs(out_dir, exist_ok=True)
    dur = n_beats * bcl
    cmd = [
        BENCH, "--numstim", str(n_beats), "--bcl", str(bcl), "--past-stim", str(bcl),
        "--imp", "ToRORd_dynCl", "--dt", "0.02", "--stim-curr", "60.0", "--dt-out", "1.0",
        "--plug-in", "LandHumanStress",
        "--strain", "0.0", "--strain-rate", "0.0", "--strain-time", "0.0",
        "--strain-dur", str(dur),
        "--imp-par", pp_ion, "--plug-par", pp_land,
        f"--fout={out_dir}/ToRORd_dynCl",
        "--plug-sv-dump=Tension", "--imp-sv-dump=Ca_i", "--bin", "--no-trace",
    ]
    with open(f"{out_dir}/bench.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    return out_dir


def last_beat(path: str, n_beat: int) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    a = np.fromfile(path, dtype="<f8")
    if a.shape[0] < 2 * n_beat or not np.all(np.isfinite(a[-n_beat:])):
        return None
    return a[-n_beat:]


def metrics(T: np.ndarray) -> Dict[str, float]:
    peak = float(T.max()); ipeak = int(T.argmax()); dia = float(T.min())
    dev = peak - dia
    rt50 = rt90 = np.nan
    if dev > 0:
        after = T[ipeak:]
        b50 = np.where(after <= dia + 0.5 * dev)[0]
        b90 = np.where(after <= dia + 0.1 * dev)[0]
        rt50 = float(b50[0]) if b50.size else np.nan
        rt90 = float(b90[0]) if b90.size else np.nan
    return {"peak_kPa": peak, "diastolic_kPa": dia, "developed_kPa": dev,
            "time_to_peak_ms": float(ipeak), "RT50_ms": rt50, "RT90_ms": rt90}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_beats", type=int, default=500)
    p.add_argument("--n_points", type=int, default=15)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--workdir", default="/tmp/drug_cell_sweep")
    p.add_argument("--replot", action="store_true",
                   help="skip the bench sims and re-plot from an existing (kept) workdir")
    p.add_argument("--output_dir", default=os.path.join(os.environ.get("RESULTS_ROOT", "results"), "drug_validation"))
    p.add_argument("--keep_workdir", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    n_beat = int(round(BCL))
    X = np.loadtxt(f"{SCENARIO}/data/X.txt")
    ss = f"{SCENARIO}/SS/ToRORd_dynCl"
    base_idx = pick_representative_cell(X, ss)
    print(f"Representative baseline cell: sample #{base_idx}", flush=True)

    pp_ion = open(f"{SCENARIO}/SS/param/{base_idx}_param_ToRORd_dynCl.txt").read().strip()
    base_land = parse_par(open(f"{SCENARIO}/SS/param/{base_idx}_param_ToRORd_dynCl_Land.txt").read())
    print(f"  baseline Land: dr={base_land['dr']} mu={base_land['mu']} Tref={base_land['Tref']}", flush=True)

    if not args.replot and os.path.isdir(args.workdir):
        shutil.rmtree(args.workdir)

    # build all run specs
    jobs = []
    plan = {}  # sweep_name -> list of (value, out_dir)
    for name, (col, key, _color) in SWEEPS.items():
        lo, hi = X[:, col].min(), X[:, col].max()
        vals = np.linspace(lo, hi, args.n_points)
        plan[name] = []
        for j, v in enumerate(vals):
            land = OrderedDict(base_land)
            land[key] = f"{v:.6g}"
            out = f"{args.workdir}/{key}/{j:02d}"
            jobs.append((out, pp_ion, serialise_par(land), args.n_beats, BCL))
            plan[name].append((float(v), out))
        print(f"  {name}: sweep [{lo:.4g}, {hi:.4g}] x{args.n_points} (baseline {key}={base_land[key]})",
              flush=True)

    if args.replot:
        print(f"\n--replot: skipping sims, re-plotting from {args.workdir}", flush=True)
    else:
        print(f"\nRunning {len(jobs)} bench sims ({args.n_beats} beats) on {args.workers} workers...",
              flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, j) for j in jobs]
            for _ in as_completed(futs):
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} done", flush=True)

    # collect metrics + traces
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    traces = {}  # name -> list of (value, T_beat)
    for name, (col, key, _color) in SWEEPS.items():
        traces[name] = []
        for v, out in plan[name]:
            T = last_beat(f"{out}/ToRORd_dynCl.Tension.bin", n_beat)
            if T is None:
                print(f"  WARN no output: {name} value={v:.4g}")
                continue
            m = metrics(T)
            m.update({"sweep": name, "param_key": key, "value": v})
            rows.append(m)
            traces[name].append((v, T))
    df = pd.DataFrame(rows)
    if df.empty:
        print(
            "\nERROR: no bench output found in workdir — all simulations missing.\n"
            "  If you used --replot, the workdir may have been cleared (e.g. /tmp is\n"
            "  wiped on reboot). Re-run without --replot to regenerate the simulations,\n"
            f"  or use --workdir pointing to a kept directory:\n"
            f"    python validate_drug_cell_sweep.py --keep_workdir --workdir <path> ...",
            flush=True,
        )
        return
    csv = f"{args.output_dir}/drug_cell_sweep_metrics.csv"
    df.to_csv(csv, index=False)
    print(f"\nWrote {csv}", flush=True)

    # ---- figure: row1 dose-response (peak+developed), row2 transient overlays ----
    n = len(SWEEPS)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9))
    for c, (name, (col, key, color)) in enumerate(SWEEPS.items()):
        sub = df[df["sweep"] == name].sort_values("value")
        lo, hi = X[:, col].min(), X[:, col].max()
        mid = 0.5 * (lo + hi)
        ax = axes[0, c]
        ax.plot(sub["value"], sub["peak_kPa"], "o-", color=color, label="peak")
        ax.plot(sub["value"], sub["developed_kPa"], "s--", color=color, alpha=0.6, label="developed")
        dh = DRUG_HALF.get(key)
        if dh == "lower":
            ax.axvspan(lo, mid, color=color, alpha=0.10, label="drug interval (lower half)")
        elif dh == "upper":
            ax.axvspan(mid, hi, color=color, alpha=0.10, label="drug interval (upper half)")
        ax.axvline(mid, color="grey", ls=":", lw=1.2, label="range midpoint")
        ax.axvline(float(base_land[key]), color="k", ls="-", lw=1.8, label="baseline (medoid cell)")
        ax.set_xlabel(SYMBOL[key], fontsize=12); ax.set_ylabel("Tension (kPa)")
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        # transient overlays at min / baseline / max
        ax2 = axes[1, c]
        t = np.arange(n_beat)
        trs = traces[name]
        if trs:
            vals = [v for v, _ in trs]
            style = {"min": dict(color=color, alpha=1.0),
                     "baseline": dict(color="k", alpha=0.9),
                     "max": dict(color=color, alpha=0.30)}
            for target, lab in [(min(vals), "min"), (float(base_land[key]), "baseline"), (max(vals), "max")]:
                k = int(np.argmin([abs(v - target) for v in vals]))
                v, T = trs[k]
                ax2.plot(t, T, lw=1.8, label=f"{lab} ({SYMBOL[key]}={v:.3g})", **style[lab])
        ax2.set_xlabel("Time within last beat (ms)"); ax2.set_ylabel("Active tension (kPa)")
        ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle("Single-cell dose-response of the contraction model to the drug parameters\n"
                 f"(representative ventricular myocyte #{base_idx}, {args.n_beats} beats, isometric)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = f"{args.output_dir}/drug_cell_sweep.png"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}", flush=True)

    # concise console summary of direction
    print("\n=== direction of effect (peak tension at min vs max of each sweep) ===")
    for name in SWEEPS:
        sub = df[df["sweep"] == name].sort_values("value")
        if len(sub) >= 2:
            pmin, pmax = sub.iloc[0]["peak_kPa"], sub.iloc[-1]["peak_kPa"]
            arrow = "REDUCES" if pmin < pmax else "INCREASES"
            print(f"  {name}: peak at low end={pmin:.1f} kPa, high end={pmax:.1f} kPa "
                  f"-> reducing the parameter {arrow} tension (Δ={pmin-pmax:+.1f})")

    if not args.keep_workdir:
        shutil.rmtree(args.workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
