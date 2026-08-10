"""
Statistics of the change in hemodynamic outputs between beat 4 and beat 5.

For each HCM case it recomputes the pipeline's mechanics biomarkers (the 38
outputs in cycle_output) for beat 4 and beat 5 of every *working* cycle (the
output_mask_beat_5 sims, which by definition completed both beats), using the
EXACT pipeline definitions imported from check_cycle_output_archer2 /
common.fourchamber_output and the same AVD-based beat window
(start = (N-1)*BCL - AVD, end = start + BCL).

It then reports, per output, statistics of the beat5-vs-beat4 difference both in
absolute units and as a percentage (base = beat 4), per heart and pooled across
all hearts. Output: CSV tables (detailed long form + a compact mean-|%Δ| pivot)
plus a printed summary.

Note: the timings (LVivc/eje/ivr/fil, ...) and EP outputs (A_TAT, V_TAT) from the
full 48-label set are excluded here -- timings are within-beat phase times that
need separate beat-relative normalisation, and A_TAT/V_TAT are EP activation
times that are beat-invariant. Ask if you want timings added (beat-relative).
"""
import os
import io
import sys
import json
import argparse
import contextlib
import numpy as np
from pandas import read_csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_cycle_output_archer2 import (  # noqa: E402
    VV_output_free_IV_thr,
    artery_cycle_output_free_output_mask_name as artery_output,
)
from common.fourchamber_output import AA_output, AA_output_ej  # noqa: E402

IV_THR = 0.1
EPS = 1e-6  # below this |beat4| we skip the % difference (avoid blow-ups)

# Roots searched for the simulation data. Set DATA_ROOT to one or more
# directories, colon-separated like PATH, each containing HCM/<case>/scenarios/.
DRIVES = [d for d in os.environ.get("DATA_ROOT", "").split(os.pathsep) if d]
# label -> (patient folder, scenario number, phenotype)
CASES = {
    "HCM1": (1, 53, "Mid-to-apical LVH"),
    "HCM2": (2, 47, "LVOTO"),
    "HCM3": (3, 48, "Isolated basal LVH"),
    "HCM4": (4, 49, "Milder asymmetric LVH"),
    "HCM5": (5, 50, "Undifferentiated pattern"),
}

LABELS = [
    "LVedv", "LVedp", "LVesv", "LVpMax", "LVdpdtMax", "LVdpdtMin", "LVSV", "LVEF",  # LVedp at index 1

    "RVedv", "RVedp", "RVesv", "RVpMax", "RVdpdtMax", "RVdpdtMin", "RVSV", "RVEF",
    "LAedv", "LAesv", "LAvMax", "LApMax",
    "RAedv", "RAesv", "RAvMax", "RApMax",
    "LAsvA", "LAinflV", "LAsvV",
    "RAsvA", "RAinflV", "RAsvV",
    "diastAP", "systAP", "pulseAP", "mAP",
    "diastPAP", "systPAP", "pulsePAP", "mPAP",
]
NOUT = len(LABELS)


def find_first(rel_candidates):
    """Return (scen, path) of the first existing candidate, preferring _more_samples."""
    for scen, rel in rel_candidates:
        for d in DRIVES:
            p = os.path.join(d, rel)
            if os.path.exists(p):
                return scen, d, p
    return None


def _vv_block(time, vol, pres):
    """Ventricular 8-vector [edv,edp,esv,pMax,dpdtMax,dpdtMin,SV,EF] or NaNs."""
    out = np.full(8, np.nan)
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # silence "oscillations" prints
            v = VV_output_free_IV_thr(time, vol, pres, IV_THR)
        if v[0] and v[0] > 0:  # EDV>0 means extraction succeeded
            sv = v[0] - v[2]
            out[:6] = v
            out[6] = sv
            out[7] = 100.0 * sv / v[0]
    except Exception:
        pass
    return out


def mechanics_vector(time, vols, press):
    """Compute the 38-output vector for one beat. vols/press: dict per chamber/tube."""
    out = np.full(NOUT, np.nan)
    out[0:8] = _vv_block(time, vols["LV"], press["LV"])
    out[8:16] = _vv_block(time, vols["RV"], press["RV"])
    try:
        out[16:20] = AA_output(time, vols["LA"], press["LA"])
    except Exception:
        pass
    try:
        out[20:24] = AA_output(time, vols["RA"], press["RA"])
    except Exception:
        pass
    try:
        out[24:27] = AA_output_ej(time, vols["LA"])
    except Exception:
        pass
    try:
        out[27:30] = AA_output_ej(time, vols["RA"])
    except Exception:
        pass
    try:
        out[30:34] = artery_output(press["AO"])
    except Exception:
        pass
    try:
        out[34:38] = artery_output(press["AP"])
    except Exception:
        pass
    return out


def beat_indices(time_all, nbeat, bcl, avd):
    start = (nbeat - 1) * bcl - avd
    end = start + bcl
    return np.where((time_all >= start) & (time_all <= end))[0]


def read_cycle(cdir):
    """Read the 6 chamber/tube files; return (time_all, vols, press) or None on error."""
    cav = {}
    for c in ["LV", "RV", "LA", "RA"]:
        cav[c] = read_csv(os.path.join(cdir, f"cav.{c}.csv"), delimiter=",",
                          skipinitialspace=True, header=0, comment="#",
                          usecols=["Time", "Volume", "Pressure"])
    tube = {}
    for tname, key in [("AO", "AO"), ("AP", "AP")]:
        tube[key] = read_csv(os.path.join(cdir, f"tube.{tname}.csv"), delimiter=",",
                             skipinitialspace=True, header=0, comment="#",
                             usecols=["Time", "Pressure"])
    time_all = np.asarray(cav["LV"]["Time"], dtype=float)
    vols = {c: np.asarray(cav[c]["Volume"], dtype=float) for c in cav}
    press = {c: np.asarray(cav[c]["Pressure"], dtype=float) for c in cav}
    press["AO"] = np.asarray(tube["AO"]["Pressure"], dtype=float)
    press["AP"] = np.asarray(tube["AP"]["Pressure"], dtype=float)
    return time_all, vols, press


def find_cycle_dir(folder, scen_name, idx):
    for d in DRIVES:
        cdir = os.path.join(d, f"HCM/{folder}/scenarios/{scen_name}/simulations/cycle_{idx}")
        if all(os.path.isfile(os.path.join(cdir, f"cav.{c}.csv")) for c in ["LV", "RV", "LA", "RA"]) \
           and all(os.path.isfile(os.path.join(cdir, f"tube.{t}.csv")) for t in ["AO", "AP"]):
            return cdir
    return None


LVEDP_IDX = LABELS.index("LVedp")


def process_case(label, limit=None, discard_neg_edp=True):
    """Return (B4, B5) arrays of shape (n_cycles, 38) for this heart.

    If discard_neg_edp, cycles whose beat-5 (last heartbeat) LV EDP is negative
    are dropped as non-physiological.
    """
    folder, scen, phenotype = CASES[label]
    variants = [(f"{scen}_more_samples", None), (f"{scen}", None)]

    mask_hit = find_first([(s, f"HCM/{folder}/scenarios/{s}/output/output_mask_beat_5.txt")
                           for s, _ in variants])
    if mask_hit is None:
        print(f"[{label}] no mask found, skipping."); return None, None
    scen_name, _, mask_path = mask_hit

    x_hit = find_first([(scen_name, f"HCM/{folder}/scenarios/{scen_name}/data/X.txt")])
    xl_hit = find_first([(scen_name, f"HCM/{folder}/scenarios/{scen_name}/data/xlabels.txt")])
    bcl_hit = find_first([(s, f"HCM/{folder}/scenarios/{s}/json_files/clinical_data.json")
                          for s, _ in variants])
    if not (x_hit and xl_hit and bcl_hit):
        print(f"[{label}] missing X.txt/xlabels/clinical_data, skipping."); return None, None

    with open(xl_hit[2]) as f:
        xlabels = f.read().splitlines()
    av_col = xlabels.index("AV_delay")
    avd_all = np.loadtxt(x_hit[2], usecols=av_col)
    bcl = float(json.load(open(bcl_hit[2]))["general"]["BCL"])

    mask = np.loadtxt(mask_path, dtype=int)
    working = np.where(mask == 1)[0]

    print(f"[{label}] {phenotype} | scenario {scen_name} | BCL {bcl} ms | "
          f"{len(working)} working cycles")

    B4, B5 = [], []
    used = errored = missing = neg_edp = 0
    for idx in working:
        if limit and used >= limit:
            break
        cdir = find_cycle_dir(folder, scen_name, idx)
        if cdir is None:
            missing += 1; continue
        try:
            time_all, vols, press = read_cycle(cdir)
            avd = float(avd_all[idx])
            i4 = beat_indices(time_all, 4, bcl, avd)
            i5 = beat_indices(time_all, 5, bcl, avd)
            if len(i4) < 10 or len(i5) < 10:
                errored += 1; continue
            v4 = mechanics_vector(time_all[i4], {c: vols[c][i4] for c in vols},
                                  {c: press[c][i4] for c in press})
            v5 = mechanics_vector(time_all[i5], {c: vols[c][i5] for c in vols},
                                  {c: press[c][i5] for c in press})
            # Discard non-physiological sims: negative LV EDP at the last beat.
            if discard_neg_edp and np.isfinite(v5[LVEDP_IDX]) and v5[LVEDP_IDX] < 0:
                neg_edp += 1; continue
            B4.append(v4); B5.append(v5); used += 1
        except Exception as e:
            errored += 1
            print(f"[{label}]   SKIP cycle_{idx}: {type(e).__name__}: {e}")
        if used and used % 100 == 0:
            print(f"[{label}]   ...{used} cycles processed")

    print(f"[{label}] DONE: {used} used, {neg_edp} discarded (beat-5 LV EDP<0), "
          f"{missing} not found locally, {errored} errored.")
    if not B4:
        return None, None
    return np.array(B4), np.array(B5)


def summarise(B4, B5, heart):
    """Per-output stats of (beat5 - beat4) in absolute units and %."""
    rows = []
    diff = B5 - B4
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * diff / B4
    pct[np.abs(B4) < EPS] = np.nan
    for j, name in enumerate(LABELS):
        valid = np.isfinite(B4[:, j]) & np.isfinite(B5[:, j])
        d = diff[valid, j]
        ad = np.abs(d)
        p = pct[valid, j]
        ap = np.abs(p[np.isfinite(p)])
        rows.append({
            "heart": heart, "output": name, "n_cycles": int(valid.sum()),
            "mean_abs_diff": np.mean(ad) if ad.size else np.nan,
            "median_abs_diff": np.median(ad) if ad.size else np.nan,
            "max_abs_diff": np.max(ad) if ad.size else np.nan,
            "mean_signed_diff": np.mean(d) if d.size else np.nan,
            "mean_abs_pct": np.mean(ap) if ap.size else np.nan,
            "median_abs_pct": np.median(ap) if ap.size else np.nan,
            "p95_abs_pct": np.percentile(ap, 95) if ap.size else np.nan,
            "max_abs_pct": np.max(ap) if ap.size else np.nan,
        })
    return rows


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description="Beat-4 vs beat-5 output-difference statistics.")
    ap.add_argument("--output_dir", default=os.path.join(os.environ.get("RESULTS_ROOT", "results"), "beat4_vs_beat5_stats"))
    ap.add_argument("--cases", nargs="+", default=list(CASES.keys()))
    ap.add_argument("--limit", type=int, default=None, help="Max cycles per heart (testing).")
    ap.add_argument("--keep_neg_edp", action="store_true",
                    help="Keep cycles with negative beat-5 LV EDP (default: discard them).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []
    pooled_B4, pooled_B5 = [], []
    for label in args.cases:
        B4, B5 = process_case(label, limit=args.limit,
                              discard_neg_edp=not args.keep_neg_edp)
        if B4 is None:
            continue
        all_rows.extend(summarise(B4, B5, label))
        pooled_B4.append(B4); pooled_B5.append(B5)

    if pooled_B4:
        all_rows.extend(summarise(np.vstack(pooled_B4), np.vstack(pooled_B5), "ALL"))

    df = pd.DataFrame(all_rows)
    detailed = os.path.join(args.output_dir, "beat4_vs_beat5_stats_detailed.csv")
    df.to_csv(detailed, index=False, float_format="%.4g")

    # Compact pivot: mean |%Δ| per output (rows) x heart (cols), output order preserved.
    pivot = df.pivot(index="output", columns="heart", values="mean_abs_pct").reindex(LABELS)
    cols = [c for c in args.cases + ["ALL"] if c in pivot.columns]
    pivot = pivot[cols]
    pivot_path = os.path.join(args.output_dir, "beat4_vs_beat5_meanabspct_pivot.csv")
    pivot.to_csv(pivot_path, float_format="%.3f")

    print("\n=== mean |%Δ| (beat5 vs beat4) per output ===")
    print(pivot.round(3).to_string())
    print(f"\nSaved:\n  {detailed}\n  {pivot_path}")


if __name__ == "__main__":
    main()
