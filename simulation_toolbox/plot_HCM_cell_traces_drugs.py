"""
Plot single-cell (openCARP `bench`) ventricular traces for the two myosin-inhibitor
drug scenarios, highlighting the simulations that fall in the BOTTOM HALF of the
affected parameter's sampled interval, and report summary metrics (peak tension,
etc.) for both ALL simulations and the bottom-half subset.

Drug -> affected ventricular parameter (column in data/X.txt, see GSA xlabels.txt):
  - Mavacamten -> dr_V   (col 7,  "ratio of strongly bound actin-myosin units")
  - Aficamten  -> mu_V   (col 12, "weak-to-strong crossbridge transition rate")

"Bottom half" = parameter value below the MIDPOINT of its interval, where the
interval is the [min, max] of that parameter across the case's data/X.txt
(two equal parts -> split at (min+max)/2).

Cell-sim traces live in <scenario>/SS/ToRORd_dynCl/<i>/ToRORd_dynCl.<signal>.bin
(little-endian float64, 1 ms sampling). Sample index i maps 1:1 to data/X.txt row i
(verified against SS/param/<i>_param_ToRORd_dynCl_Land.txt). Only the last beat
(steady state) is read, via numpy.memmap, to keep I/O small.

Produces, per drug:
  - one figure per HCM case (active tension + Ca_i over the last beat),
  - one pooled figure across all cases (time normalised to % of the cycle),
  - a metrics CSV (mean +/- std and [min,max] interval) for ALL vs BOTTOM-HALF.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Static configuration
# --------------------------------------------------------------------------- #

# Per-case cell-sim source directories (union across drives), the scenario folder
# that holds data/X.txt + json_files/clinical_data.json, and the phenotype name.
CASES: Dict[int, Dict] = {
    1: {
        "phenotype": "Mid-to-apical LVH",
        "scenario": "/media/croderog/Bob/HCM/1/scenarios/53_more_samples",
        "ss_dirs": [
            "/media/croderog/Bob/HCM/1/scenarios/53_more_samples/SS/ToRORd_dynCl",
            "/media/croderog/SeagateExpansionDrive/HCM/1/scenarios/53_more_samples/SS/ToRORd_dynCl",
        ],
    },
    2: {
        # base scenario 47 holds cell sims 0-799; 47_more_samples holds 800+
        "phenotype": "LVOTO",
        "scenario": "/data/HCM/2/scenarios/47_more_samples",
        "ss_dirs": [
            "/media/croderog/SeagateExpansionDrive/HCM/2/scenarios/47/SS/ToRORd_dynCl",
            "/data/HCM/2/scenarios/47_more_samples/SS/ToRORd_dynCl",
        ],
    },
    3: {
        "phenotype": "Isolated basal LVH",
        "scenario": "/data/HCM/3/scenarios/48_more_samples",
        "ss_dirs": [
            "/media/croderog/Elements/HCM/3/scenarios/48/SS/ToRORd_dynCl",
            "/data/HCM/3/scenarios/48_more_samples/SS/ToRORd_dynCl",
        ],
    },
    4: {
        "phenotype": "Milder asymmetric LVH",
        "scenario": "/media/croderog/Elements/HCM/4/scenarios/49_more_samples",
        "ss_dirs": [
            "/media/croderog/Elements/HCM/4/scenarios/49/SS/ToRORd_dynCl",
            "/media/croderog/Elements/HCM/4/scenarios/49_more_samples/SS/ToRORd_dynCl",
        ],
    },
    5: {
        "phenotype": "Undifferentiated pattern",
        "scenario": "/data/HCM/5/scenarios/50_more_samples",
        "ss_dirs": ["/data/HCM/5/scenarios/50_more_samples/SS/ToRORd_dynCl"],
    },
}

DRUGS: Dict[str, Dict] = {
    "Mavacamten": {"col": 7, "param": "dr_V", "color": "#CE6A0E"},
    "Aficamten": {"col": 12, "param": "mu_V", "color": "#F1BF00"},
}

BASELINE_COLOR = "#b0b0b0"
ION = "ToRORd_dynCl"  # LV ventricular ionic model


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def get_bcl(scenario: str) -> float:
    """Beating cycle length (ms) from clinical_data.json."""
    with open(os.path.join(scenario, "json_files", "clinical_data.json")) as f:
        return float(json.load(f)["general"]["BCL"])


def read_last_beat(bin_path: str, n_beat: int) -> Optional[np.ndarray]:
    """Return the last `n_beat` samples of a little-endian float64 trace, or None
    if the file is missing / too short / non-finite. Uses memmap so only the tail
    is touched on disk."""
    try:
        mm = np.memmap(bin_path, dtype="<f8", mode="r")
    except (FileNotFoundError, ValueError):
        return None
    if mm.shape[0] < 2 * n_beat:
        return None
    beat = np.array(mm[-n_beat:])  # copy out of the memmap
    del mm
    if not np.all(np.isfinite(beat)):
        return None
    return beat


def collect_sim_indices(ss_dirs: List[str]) -> Dict[int, str]:
    """Map sample index -> directory holding its Tension.bin (first drive wins)."""
    found: Dict[int, str] = {}
    for ss in ss_dirs:
        if not os.path.isdir(ss):
            continue
        for name in os.listdir(ss):
            sim_dir = os.path.join(ss, name)
            if not os.path.isdir(sim_dir):
                continue
            try:
                idx = int(name)
            except ValueError:
                continue
            if idx in found:
                continue
            if os.path.isfile(os.path.join(sim_dir, f"{ION}.Tension.bin")):
                found[idx] = sim_dir
    return found


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def tension_metrics(T: np.ndarray) -> Dict[str, float]:
    """Scalar metrics on one steady-state beat of active tension (kPa, 1 ms grid)."""
    peak = float(T.max())
    ipeak = int(T.argmax())
    dia = float(T.min())
    dev = peak - dia
    # relaxation times measured from the peak
    rt50 = rt90 = np.nan
    if dev > 0:
        after = T[ipeak:]
        below50 = np.where(after <= dia + 0.5 * dev)[0]
        below90 = np.where(after <= dia + 0.1 * dev)[0]
        if below50.size:
            rt50 = float(below50[0])
        if below90.size:
            rt90 = float(below90[0])
    return {
        "peak_tension_kPa": peak,
        "diastolic_tension_kPa": dia,
        "developed_tension_kPa": dev,
        "time_to_peak_ms": float(ipeak),
        "RT50_ms": rt50,
        "RT90_ms": rt90,
    }


def calcium_metrics(C: np.ndarray) -> Dict[str, float]:
    peak = float(C.max())
    dia = float(C.min())
    return {
        "peak_Cai_uM": peak,
        "diastolic_Cai_uM": dia,
        "CaT_amplitude_uM": peak - dia,
    }


# --------------------------------------------------------------------------- #
# Per-case loading
# --------------------------------------------------------------------------- #

def load_case(case: int, with_ca: bool, max_sims: Optional[int]) -> Dict:
    cfg = CASES[case]
    scenario = cfg["scenario"]
    bcl = get_bcl(scenario)
    n_beat = int(round(bcl))
    X = np.loadtxt(os.path.join(scenario, "data", "X.txt"))

    sims = collect_sim_indices(cfg["ss_dirs"])
    idxs = sorted(sims)
    if max_sims is not None:
        idxs = idxs[:max_sims]

    records = []
    tensions: List[np.ndarray] = []
    calciums: List[np.ndarray] = []
    for k, idx in enumerate(idxs):
        if idx >= X.shape[0]:
            continue
        sim_dir = sims[idx]
        T = read_last_beat(os.path.join(sim_dir, f"{ION}.Tension.bin"), n_beat)
        if T is None:
            continue
        C = None
        if with_ca:
            C = read_last_beat(os.path.join(sim_dir, f"{ION}.Ca_i.bin"), n_beat)
            if C is None:
                continue
        rec = {"idx": idx, "dr_V": float(X[idx, 7]), "mu_V": float(X[idx, 12])}
        rec.update(tension_metrics(T))
        if C is not None:
            rec.update(calcium_metrics(C))
        records.append(rec)
        tensions.append(T)
        calciums.append(C if C is not None else np.empty(0))
        if (k + 1) % 500 == 0:
            print(f"  HCM{case}: read {k + 1}/{len(idxs)} sims", flush=True)

    df = pd.DataFrame.from_records(records)
    print(f"  HCM{case}: usable cell sims = {len(df)} (BCL={bcl:g} ms)", flush=True)

    # interval midpoints from the FULL design (all X.txt rows -> robust),
    # keyed by parameter name (dr_V, mu_V)
    midpoints = {d["param"]: 0.5 * (X[:, d["col"]].min() + X[:, d["col"]].max())
                 for d in DRUGS.values()}
    return {
        "case": case,
        "phenotype": cfg["phenotype"],
        "bcl": bcl,
        "n_beat": n_beat,
        "df": df,
        "tensions": tensions,
        "calciums": calciums,
        "midpoints": midpoints,
    }


# --------------------------------------------------------------------------- #
# Metrics table
# --------------------------------------------------------------------------- #

METRIC_COLS = [
    "peak_tension_kPa", "diastolic_tension_kPa", "developed_tension_kPa",
    "time_to_peak_ms", "RT50_ms", "RT90_ms",
    "peak_Cai_uM", "diastolic_Cai_uM", "CaT_amplitude_uM",
]


def summarise(df: pd.DataFrame, label: str) -> List[Dict]:
    rows = []
    for m in METRIC_COLS:
        if m not in df.columns:
            continue
        v = df[m].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({
            "group": label, "n": v.size, "metric": m,
            "mean": v.mean(), "std": v.std(ddof=1) if v.size > 1 else 0.0,
            "min": v.min(), "max": v.max(),
        })
    return rows


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_panel(ax, t, traces, mask_bottom, color, ylabel, title):
    """ALL traces in grey, then the bottom-half subset highlighted in drug colour."""
    for tr in traces:
        if tr.size == 0:
            continue
        ax.plot(t, tr, color=BASELINE_COLOR, lw=0.4, alpha=0.10, rasterized=True)
    for i, tr in enumerate(traces):
        if tr.size == 0 or not mask_bottom[i]:
            continue
        ax.plot(t, tr, color=color, lw=0.5, alpha=0.40, rasterized=True)
    ax.set_xlabel(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, ls="--")


def metrics_textbox(ax, df_all, df_bot, color):
    def line(df, key, unit=""):
        v = df[key].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return "n/a"
        return f"{v.mean():.1f}±{v.std(ddof=1) if v.size>1 else 0:.1f} [{v.min():.1f}, {v.max():.1f}]{unit}"

    txt = (
        f"Peak tension (kPa)\n"
        f"  All (n={len(df_all)}): {line(df_all,'peak_tension_kPa')}\n"
        f"  Bottom (n={len(df_bot)}): {line(df_bot,'peak_tension_kPa')}\n"
        f"Developed tension (kPa)\n"
        f"  All: {line(df_all,'developed_tension_kPa')}\n"
        f"  Bottom: {line(df_bot,'developed_tension_kPa')}"
    )
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.85))


def make_case_figure(case_data, drug, drug_cfg, savepath, dpi):
    df = case_data["df"]
    if df.empty:
        print(f"  skip HCM{case_data['case']} {drug}: no sims")
        return None
    col_name = drug_cfg["param"]
    mid = case_data["midpoints"][col_name]
    mask = (df[col_name].to_numpy() < mid)
    n_beat = case_data["n_beat"]
    t = np.arange(n_beat)

    with_ca = "peak_Cai_uM" in df.columns
    ncols = 2 if with_ca else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)
    axes = axes[0]

    plot_panel(axes[0], t, case_data["tensions"], mask, drug_cfg["color"],
               "Active tension (kPa)", "Time within last beat (ms)")
    metrics_textbox(axes[0], df, df[mask], drug_cfg["color"])
    if with_ca:
        plot_panel(axes[1], t, case_data["calciums"], mask, drug_cfg["color"],
                   "[Ca]$_i$ ($\\mu$M)", "Time within last beat (ms)")

    handles = [
        Line2D([0], [0], color=BASELINE_COLOR, lw=2,
               label=f"All sims (n={len(df)})"),
        Line2D([0], [0], color=drug_cfg["color"], lw=2,
               label=f"{drug}: {col_name} < {mid:.3g} (n={int(mask.sum())})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle(f"{drug} — HCM{case_data['case']}: {case_data['phenotype']}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    os.makedirs(savepath, exist_ok=True)
    out = os.path.join(savepath, f"cell_traces_{drug}_HCM{case_data['case']}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

    rows = summarise(df, "All") + summarise(df[mask], f"BottomHalf({col_name})")
    mdf = pd.DataFrame(rows)
    mdf.insert(0, "case", f"HCM{case_data['case']}")
    mdf.to_csv(os.path.join(savepath, f"metrics_{drug}_HCM{case_data['case']}.csv"),
               index=False)
    return mdf


def make_combined_figure(all_cases, drug, drug_cfg, savepath, dpi, n_grid=200):
    """Pool all cases; normalise each last beat to 0-100% of the cycle."""
    col_name = drug_cfg["param"]
    xg = np.linspace(0, 100, n_grid)
    T_all, T_bot, C_all, C_bot = [], [], [], []
    dfs = []
    for cd in all_cases:
        df = cd["df"]
        if df.empty:
            continue
        mid = cd["midpoints"][col_name]
        mask = (df[col_name].to_numpy() < mid)
        nb = cd["n_beat"]
        src = np.linspace(0, 100, nb)
        for i, tr in enumerate(cd["tensions"]):
            if tr.size == 0:
                continue
            (T_bot if mask[i] else T_all).append(np.interp(xg, src, tr))
        for i, cc in enumerate(cd["calciums"]):
            if cc.size == 0:
                continue
            (C_bot if mask[i] else C_all).append(np.interp(xg, src, cc))
        d = df.copy()
        d["__bottom__"] = mask
        dfs.append(d)
    if not dfs:
        return None
    pooled = pd.concat(dfs, ignore_index=True)
    bot = pooled["__bottom__"].to_numpy()

    with_ca = bool(C_all or C_bot)
    ncols = 2 if with_ca else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)
    axes = axes[0]

    for tr in T_all + T_bot:
        axes[0].plot(xg, tr, color=BASELINE_COLOR, lw=0.3, alpha=0.06, rasterized=True)
    for tr in T_bot:
        axes[0].plot(xg, tr, color=drug_cfg["color"], lw=0.4, alpha=0.30, rasterized=True)
    axes[0].set_xlabel("% of cycle")
    axes[0].set_ylabel("Active tension (kPa)")
    axes[0].grid(True, alpha=0.3, ls="--")
    metrics_textbox(axes[0], pooled, pooled[bot], drug_cfg["color"])

    if with_ca:
        for tr in C_all + C_bot:
            axes[1].plot(xg, tr, color=BASELINE_COLOR, lw=0.3, alpha=0.06, rasterized=True)
        for tr in C_bot:
            axes[1].plot(xg, tr, color=drug_cfg["color"], lw=0.4, alpha=0.30, rasterized=True)
        axes[1].set_xlabel("% of cycle")
        axes[1].set_ylabel("[Ca]$_i$ ($\\mu$M)")
        axes[1].grid(True, alpha=0.3, ls="--")

    handles = [
        Line2D([0], [0], color=BASELINE_COLOR, lw=2, label=f"All sims (n={len(pooled)})"),
        Line2D([0], [0], color=drug_cfg["color"], lw=2,
               label=f"{drug}: {col_name} bottom half (n={int(bot.sum())})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle(f"{drug} — all HCM cases pooled", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(savepath, exist_ok=True)
    out = os.path.join(savepath, f"cell_traces_{drug}_ALLCASES.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

    rows = summarise(pooled, "All") + summarise(pooled[bot], f"BottomHalf({col_name})")
    mdf = pd.DataFrame(rows)
    mdf.insert(0, "case", "ALLCASES")
    mdf.to_csv(os.path.join(savepath, f"metrics_{drug}_ALLCASES.csv"), index=False)
    return mdf


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--drugs", nargs="+", default=list(DRUGS.keys()),
                   choices=list(DRUGS.keys()))
    p.add_argument("--output_dir", default="/media/croderog/Bob/HCM/figures/cell_traces_drugs")
    p.add_argument("--with_ca", action="store_true", default=True,
                   help="also read/plot intracellular calcium (default on)")
    p.add_argument("--no_ca", dest="with_ca", action="store_false")
    p.add_argument("--max_sims", type=int, default=None,
                   help="cap sims per case (for quick tests)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no_combined", action="store_true")
    args = p.parse_args()

    print(f"Loading cell sims for cases {args.cases} ...", flush=True)
    loaded = []
    for c in args.cases:
        print(f"HCM{c}:", flush=True)
        loaded.append(load_case(c, args.with_ca, args.max_sims))

    all_metrics = []
    for drug in args.drugs:
        cfg = DRUGS[drug]
        print(f"\n=== {drug} ({cfg['param']}) ===", flush=True)
        for cd in loaded:
            m = make_case_figure(cd, drug, cfg, args.output_dir, args.dpi)
            if m is not None:
                m.insert(0, "drug", drug)
                all_metrics.append(m)
        if not args.no_combined:
            m = make_combined_figure(loaded, drug, cfg, args.output_dir, args.dpi)
            if m is not None:
                m.insert(0, "drug", drug)
                all_metrics.append(m)

    if all_metrics:
        full = pd.concat(all_metrics, ignore_index=True)
        out = os.path.join(args.output_dir, "metrics_summary_all.csv")
        full.to_csv(out, index=False)
        print(f"\nWrote combined metrics table: {out}")


if __name__ == "__main__":
    main()
