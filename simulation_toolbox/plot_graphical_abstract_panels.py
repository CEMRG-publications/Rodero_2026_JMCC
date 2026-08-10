"""
Graphical-abstract result panels (plots only; no heart icons / takeaway box).

All panels use the paper's consistent colour palette:
  - parameters are coloured by their group (group_colors.json: Cardiovascular
    #CC6677, Tissue #D3D3D3, Cellular #F1BF00, Active tension #88CCEE, ...);
  - the VAS panel uses the paper's VAS convention (yellow #F1BF00 = anatomy
    matters MORE, dark red #AA151B = less), matching plot_vas_comparison.py.

Panels (each saved as its own transparent PNG to compose elsewhere):

  Panel A  "Afterload dominates"   - ranked horizontal bar of the parameters'
           total Sobol index for one output (default LVEF), mean across the five
           anatomies, with each anatomy overlaid as a dot. LV afterload towers
           over everything and is the #1 driver in all five anatomies.

  Panel B (reshuffle)  - before/after bars for one output (default LVesv):
           baseline vs increased ventricular stiffness, mean across anatomies.
           Shows the importance reshuffle (stiffness's own influence collapses).

  Panel B (VAS)        - the paper's VAS metric for the increased-ventricular-
           stiffness scenario: % change in the between-anatomy spread of
           sensitivity. Positive = the output becomes MORE anatomy-dependent.
           18/28 outputs are positive, i.e. functional remodeling increases
           anatomical dependency (the manuscript title's second claim).
"""

import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# the project's exact VAS definition (Variability in Anatomical Sensitivity)
from plot_vas_comparison import compute_vas_metrics

# --------------------------------------------------------------------------- #
# Data locations. DATA_ROOT is the directory holding HCM/, and RESULTS_ROOT is
# where the figures are written; both are read from the environment (see README).
# --------------------------------------------------------------------------- #
DATA_ROOT = os.environ.get("DATA_ROOT", "")
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(DATA_ROOT, "results"))
CONFIG_DIR = os.path.join(DATA_ROOT, "HCM/GSA_analysis/cycle")

ANATOMIES = [
    ("Mid-to-apical LVH",        os.path.join(DATA_ROOT, "HCM/1/scenarios/53_more_samples")),
    ("LVOTO",                    os.path.join(DATA_ROOT, "HCM/2/scenarios/47_more_samples")),
    ("Isolated basal LVH",       os.path.join(DATA_ROOT, "HCM/3/scenarios/48_more_samples")),
    ("Milder asymmetric LVH",    os.path.join(DATA_ROOT, "HCM/4/scenarios/49_more_samples")),
    ("Undifferentiated pattern", os.path.join(DATA_ROOT, "HCM/5/scenarios/50_more_samples")),
]
STIFF_SCENARIO = "GSA_a_ventricles_lower_50.0_upper_100.0"  # increased ventricular stiffness
XLABELS_JSON = os.path.join(CONFIG_DIR, "xlabels_to_plot.json")
YLABELS_JSON = os.path.join(CONFIG_DIR, "ylabels_filtered.json")
GROUP_COLORS_JSON = os.path.join(CONFIG_DIR, "group_colors.json")
EXCLUSIONS_JSON = os.path.join(CONFIG_DIR, "parameters_exclusions.json")
OUTPUT_DIR = os.path.join(RESULTS_ROOT, "graphical_abstract")

PANEL_A_OUTPUT = "LVEF"
PANEL_B_OUTPUTS = ["LVesv", "LVEF"]   # reshuffle variant

# VAS variant: compute over the full output set (for the honest global count),
# but only plot a curated, clinically readable subset spanning chambers + signs.
# the paper's full 32-output "ALL" set (so the VAS counts match the manuscript: 32/32, 30/32)
VAS_FULL_OUTPUTS = [
    "LVedv", "LVedp", "LVesv", "LVpMax", "LVSV", "LVEF", "LVdpdtMax", "V_TAT",
    "RVedv", "RVedp", "RVesv", "RVpMax", "RVSV", "RVEF", "RVdpdtMax",
    "LAedv", "LAvMax", "LApMax", "LAinflV", "A_TAT",
    "RAedv", "RAvMax", "RApMax", "RAinflV",
    "diastAP", "systAP", "pulseAP", "mAP", "diastPAP", "systPAP", "pulsePAP", "mPAP",
]
VAS_PANEL_OUTPUTS = ["RVedv", "systAP", "mPAP", "RApMax", "LVdpdtMax", "LVEF", "LVSV", "LVedv"]

# pharma panel: the two drug scenarios compared on the same VAS metric
SCENARIO_COLORS_JSON = os.path.join(CONFIG_DIR, "scenario_colors.json")
MAVA_SCENARIO = "GSA_wfrac_V_lower_0.0_upper_50.0"
AFI_SCENARIO = "GSA_mu_V_lower_0.0_upper_50.0"
PHARMA_PANEL_OUTPUTS = ["RVedv", "mPAP", "RApMax", "LVedp", "systAP", "diastAP", "mAP", "LVEF", "LVdpdtMax"]
# signed (glanceable) pharma variant: include the two outputs where aficamten goes negative
# (RVSV, LAedv) so the red "less anatomy-dependent" bars surface the 32/32 vs 30/32 difference.
PHARMA_SIGNED_OUTPUTS = ["RVedv", "mPAP", "RApMax", "LVedp", "systAP", "LVEF", "LVdpdtMax", "RVSV", "LAedv"]

# paper VAS sign convention (see plot_vas_comparison.py)
VAS_POS = "#F1BF00"   # output becomes MORE anatomy-dependent
VAS_NEG = "#AA151B"   # less anatomy-dependent
FALLBACK = "#808080"


def read_rank(path):
    d = {}
    if not os.path.isfile(path):
        return d
    for ln in open(path):
        p = ln.strip().split("\t")
        if len(p) == 2:
            d[p[0]] = float(p[1])
    return d


def load(output, scenario=None):
    """{anatomy_name: {param: Si}} for one output, baseline (scenario=None) or a scenario."""
    res = {}
    for name, base in ANATOMIES:
        sub = "output" if scenario is None else f"output/{scenario}"
        res[name] = read_rank(f"{base}/{sub}/Rank_Si_total_max_{output}.txt")
    return res


def latex_param(code, xmap):
    return xmap.get(code, {}).get("latex", code)


def latex_output(code, ymap):
    return ymap.get(code, {}).get("latex", code)


def colour_by_group(code, xmap, gmap):
    """Paper palette: a parameter's colour is its group's colour (group_colors.json)."""
    grp = xmap.get(code, {}).get("group")
    return gmap.get(grp, {}).get("color", FALLBACK)


# --------------------------------------------------------------------------- #
# Panel A: afterload dominates (single output, mean across anatomies + dots)
# --------------------------------------------------------------------------- #
def panel_a(output, xmap, ymap, gmap, top_n=8):
    base = load(output)
    params = set().union(*[set(d) for d in base.values()])
    mean = {p: np.mean([base[a].get(p, 0.0) for a, _ in ANATOMIES]) for p in params}
    order = sorted(params, key=lambda p: mean[p], reverse=True)[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y = np.arange(len(order))
    ax.barh(y, [mean[p] for p in order],
            color=[colour_by_group(p, xmap, gmap) for p in order],
            edgecolor="white", height=0.72, zorder=2)
    for i, p in enumerate(order):
        xs = [base[a].get(p, 0.0) for a, _ in ANATOMIES]
        ax.scatter(xs, [i] * len(xs), s=14, color="#33373B", alpha=0.65, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([latex_param(p, xmap) for p in order], fontsize=10)
    ax.set_xlabel("Share of variance (total Sobol index)", fontsize=10)
    ax.set_xlim(0, 1.0)
    if "Rsys" in order:
        rsys_mean = mean["Rsys"]
        ax.text(rsys_mean - 0.02, order.index("Rsys"), f"{rsys_mean*100:.0f}%",
                va="center", ha="right", color="white", fontsize=11,
                fontweight="bold", zorder=4)
    ax.set_title(f"What drives {latex_output(output, ymap)}?\nLV afterload is the #1 driver in all five anatomies",
                 fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()
    out = f"{OUTPUT_DIR}/panelA_afterload_{output}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Panel B (reshuffle): before/after, single output, bars coloured by group
# --------------------------------------------------------------------------- #
def panel_b_reshuffle(output, xmap, ymap, gmap, top_n=6):
    base = load(output)
    stiff = load(output, STIFF_SCENARIO)
    params = set().union(*[set(d) for d in list(base.values()) + list(stiff.values())])
    mb = {p: np.mean([base[a].get(p, 0.0) for a, _ in ANATOMIES]) for p in params}
    ms = {p: np.mean([stiff[a].get(p, 0.0) for a, _ in ANATOMIES]) for p in params}
    order = sorted(params, key=lambda p: max(mb[p], ms[p]), reverse=True)[:top_n][::-1]
    cols = [colour_by_group(p, xmap, gmap) for p in order]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y = np.arange(len(order))
    h = 0.38
    ax.barh(y + h / 2, [mb[p] for p in order], height=h, color=cols, alpha=0.40,
            edgecolor="white", zorder=2)
    ax.barh(y - h / 2, [ms[p] for p in order], height=h, color=cols, alpha=1.0,
            edgecolor="white", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([latex_param(p, xmap) for p in order], fontsize=10)
    for tick, p in zip(ax.get_yticklabels(), order):
        if p in ("Rsys", "a_ventricles"):
            tick.set_fontweight("bold")
    ax.set_xlabel("Share of variance (total Sobol index)", fontsize=10)
    ax.set_title(f"Functional remodeling reshuffles what matters for {latex_output(output, ymap)}\n"
                 f"afterload rises, the influence of stiffness itself falls", fontsize=11, loc="left")
    ax.legend(handles=[Patch(facecolor="#777777", alpha=0.40, label="Baseline"),
                       Patch(facecolor="#777777", alpha=1.0, label="Increased ventricular stiffness")],
              fontsize=9, loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()
    out = f"{OUTPUT_DIR}/panelB_reshuffle_{output}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Panel B (VAS): the paper's anatomical-dependency metric for increased stiffness
# --------------------------------------------------------------------------- #
def panel_b_vas(xmap, ymap, gmap):
    exclusions = json.load(open(EXCLUSIONS_JSON))
    anatomies_dict = {n: {"baseline": b, "modified": [f"{b}/output/{STIFF_SCENARIO}"]}
                      for n, b in ANATOMIES}
    vas_by_output, _, names = compute_vas_metrics(
        anatomies_dict, VAS_FULL_OUTPUTS, exclusions, "Si_total", "max", threshold=0.05)
    scen = names[0]

    def vas_of(o):
        return vas_by_output.get(o, {}).get(scen, {}).get("vas")

    n_pos = sum(1 for o in VAS_FULL_OUTPUTS if (vas_of(o) or 0) > 0)
    n_tot = sum(1 for o in VAS_FULL_OUTPUTS if vas_of(o) is not None)

    rows = [(o, vas_of(o)) for o in VAS_PANEL_OUTPUTS if vas_of(o) is not None]
    rows.sort(key=lambda r: r[1])  # ascending -> largest at top in barh
    labels = [latex_output(o, ymap) for o, _ in rows]
    vals = [v for _, v in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=[VAS_POS if v > 0 else VAS_NEG for v in vals],
            edgecolor="black", linewidth=0.5, zorder=2)
    ax.axvline(0, color="black", lw=1.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Change in between-anatomy spread of sensitivity, VAS (%)", fontsize=10)
    mx = max(abs(v) for v in vals) * 1.25
    ax.set_xlim(-mx, mx)
    for yi, v in zip(y, vals):
        ax.text(v + (4 if v > 0 else -4), yi, f"{v:+.0f}%", va="center",
                ha="left" if v > 0 else "right", fontsize=10)
    ax.set_title("Functional remodeling increases anatomical dependency\n"
                 "under increased ventricular stiffness, sensitivity spreads across anatomies",
                 fontsize=11, loc="left")
    ax.legend(handles=[Patch(facecolor=VAS_POS, edgecolor="black", label="more anatomy-dependent"),
                       Patch(facecolor=VAS_NEG, edgecolor="black", label="less anatomy-dependent")],
              fontsize=9, loc="lower right", frameon=False)
    ax.text(0.5, -0.18, f"{n_pos} of {n_tot} outputs become more anatomy-dependent (VAS > 0)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, style="italic", color="#333333")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()
    out = f"{OUTPUT_DIR}/panelB_VAS_stiffness.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  wrote {out}  ({n_pos}/{n_tot} outputs VAS>0)")


# --------------------------------------------------------------------------- #
# Pharma panel: both myosin inhibitors increase anatomical dependency (VAS)
# --------------------------------------------------------------------------- #
def panel_pharma_vas(xmap, ymap):
    scol = json.load(open(SCENARIO_COLORS_JSON))
    c_mava = scol[MAVA_SCENARIO]["color"]
    c_afi = scol[AFI_SCENARIO]["color"]
    exclusions = json.load(open(EXCLUSIONS_JSON))
    anatomies_dict = {n: {"baseline": b,
                          "modified": [f"{b}/output/{MAVA_SCENARIO}", f"{b}/output/{AFI_SCENARIO}"]}
                      for n, b in ANATOMIES}
    vbo, _, names = compute_vas_metrics(
        anatomies_dict, VAS_FULL_OUTPUTS, exclusions, "Si_total", "max", threshold=0.05)

    def v(o, i):
        return vbo.get(o, {}).get(names[i], {}).get("vas")

    n_tot = sum(1 for o in VAS_FULL_OUTPUTS if v(o, 0) is not None)
    mava_pos = sum(1 for o in VAS_FULL_OUTPUTS if (v(o, 0) or 0) > 0)
    afi_pos = sum(1 for o in VAS_FULL_OUTPUTS if (v(o, 1) or 0) > 0)

    rows = [(o, v(o, 0), v(o, 1)) for o in PHARMA_PANEL_OUTPUTS
            if v(o, 0) is not None and v(o, 1) is not None]
    rows.sort(key=lambda r: max(r[1], r[2]))   # largest at top
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for yi, (_, m, a) in zip(y, rows):
        ax.plot([a, m], [yi, yi], color="#BBBBBB", lw=2, zorder=1)
    ax.scatter([r[1] for r in rows], y, s=95, color=c_mava, edgecolor="black",
               lw=0.5, zorder=3, label="Mavacamten")
    ax.scatter([r[2] for r in rows], y, s=95, color=c_afi, edgecolor="black",
               lw=0.5, zorder=3, label="Aficamten")
    ax.axvline(0, color="black", lw=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([latex_output(o, ymap) for o, _, _ in rows], fontsize=11)
    ax.set_xlabel("Change in between-anatomy spread of sensitivity, VAS (%)", fontsize=10)
    # ax.set_title("Both myosin inhibitors increase anatomical dependency\n"
    #              "mavacamten and aficamten produce comparable, modest shifts", fontsize=11, loc="left")
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    # ax.text(0.5, -0.18,
    #         f"VAS increases for {mava_pos}/{n_tot} outputs (mavacamten) and {afi_pos}/{n_tot} (aficamten)",
    #         transform=ax.transAxes, ha="center", va="top", fontsize=9, style="italic", color="#333333")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()
    out = f"{OUTPUT_DIR}/panel_pharma_VAS_drugs.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  wrote {out}  (mava {mava_pos}/{n_tot}, afi {afi_pos}/{n_tot} outputs VAS>0)")


def panel_pharma_vas_signed(xmap, ymap):
    """Glanceable pharma panel: two side-by-side VAS columns (mavacamten | aficamten),
    bars coloured by SIGN (yellow = more anatomy-dependent, red = less) exactly like the
    functional-remodeling VAS panel. Both columns mostly yellow => both drugs increase
    anatomical dependency; aficamten's red bars surface the 30/32 vs 32/32 difference."""
    exclusions = json.load(open(EXCLUSIONS_JSON))
    anatomies_dict = {n: {"baseline": b,
                          "modified": [f"{b}/output/{MAVA_SCENARIO}", f"{b}/output/{AFI_SCENARIO}"]}
                      for n, b in ANATOMIES}
    vbo, _, names = compute_vas_metrics(
        anatomies_dict, VAS_FULL_OUTPUTS, exclusions, "Si_total", "max", threshold=0.05)

    def v(o, i):
        return vbo.get(o, {}).get(names[i], {}).get("vas")

    n_tot = sum(1 for o in VAS_FULL_OUTPUTS if v(o, 0) is not None)
    mava_pos = sum(1 for o in VAS_FULL_OUTPUTS if (v(o, 0) or 0) > 0)
    afi_pos = sum(1 for o in VAS_FULL_OUTPUTS if (v(o, 1) or 0) > 0)

    rows = [(o, v(o, 0), v(o, 1)) for o in PHARMA_SIGNED_OUTPUTS
            if v(o, 0) is not None and v(o, 1) is not None]
    rows.sort(key=lambda r: r[1])  # by mavacamten, ascending -> largest at top
    y = np.arange(len(rows))
    allv = [r[1] for r in rows] + [r[2] for r in rows]
    pad = 0.13 * (max(allv) - min(allv))
    xlim = (min(allv) - pad, max(allv) + 1.6 * pad)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 4.8), sharey=True)
    for ax, idx, title in [(axL, 1, "Mavacamten"), (axR, 2, "Aficamten")]:
        vals = [r[idx] for r in rows]
        ax.barh(y, vals, color=[VAS_POS if x > 0 else VAS_NEG for x in vals],
                edgecolor="black", linewidth=0.5, zorder=2)
        ax.axvline(0, color="black", lw=1.2, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("VAS (%)", fontsize=10)
        for yi, x in zip(y, vals):
            ax.text(x + (0.02 * (xlim[1] - xlim[0])) * (1 if x >= 0 else -1), yi,
                    f"{x:+.0f}%", va="center", ha="left" if x >= 0 else "right", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", alpha=0.25, zorder=0)
    axL.set_yticks(y)
    axL.set_yticklabels([latex_output(o, ymap) for o, _, _ in rows], fontsize=11)

    fig.legend(handles=[Patch(facecolor=VAS_POS, edgecolor="black", label="more anatomy-dependent"),
                        Patch(facecolor=VAS_NEG, edgecolor="black", label="less anatomy-dependent")],
               fontsize=10, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    # fig.suptitle("Both myosin inhibitors increase anatomical dependency (comparable, modest)\n"
    #              f"VAS up in {mava_pos}/{n_tot} outputs (mavacamten) vs {afi_pos}/{n_tot} (aficamten)",
    #              fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = f"{OUTPUT_DIR}/panel_pharma_VAS_signed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  wrote {out}  (mava {mava_pos}/{n_tot}, afi {afi_pos}/{n_tot} VAS>0)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xmap = json.load(open(XLABELS_JSON))
    ymap = json.load(open(YLABELS_JSON))
    gmap = json.load(open(GROUP_COLORS_JSON))
    print("Panel A (afterload dominates):")
    panel_a(PANEL_A_OUTPUT, xmap, ymap, gmap, 5)
    print("Panel B (reshuffle):")
    for o in PANEL_B_OUTPUTS:
        panel_b_reshuffle(o, xmap, ymap, gmap)
    print("Panel B (VAS - anatomical dependency):")
    panel_b_vas(xmap, ymap, gmap)
    print("Pharma panel (mavacamten vs aficamten, VAS):")
    panel_pharma_vas(xmap, ymap)
    panel_pharma_vas_signed(xmap, ymap)
    print(f"\nAll panels in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
