"""
Plot PV loops for a representative simulation of each HCM case.

For every case it reads the four chamber files (cav.LV/RV/LA/RA.csv) from one
*working* simulation directory and draws the full multi-beat run (NOT just the
last heartbeat), splitting the trace into individual beats so each cycle's PV
loop is visible. One figure is produced per case (2x2 chamber panels by default).

Example
-------
python plot_HCM_pv_loops.py \
    --case "HCM1=$DATA_ROOT/HCM/1/scenarios/53/simulations/cycle_123" \
    --case "HCM2=$DATA_ROOT/HCM/2/scenarios/47/simulations/cycle_45" \
    --BCL 800 --n_beats 5 --output_dir ./figures
"""
import os
import argparse
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pandas import read_csv

CHAMBERS = ['LV', 'RV', 'LA', 'RA']


def load_chamber(path_to_simulation: str, chamber: str) -> tuple:
    """Read the full Time/Pressure/Volume trace for one chamber.

    Args:
        path_to_simulation: Directory holding the cav.<chamber>.csv files.
        chamber: Chamber name, e.g. 'LV'.

    Returns:
        (time, pressure, volume) numpy arrays, or (None, None, None) if the
        file is missing.
    """
    chamber_file = os.path.join(path_to_simulation, f'cav.{chamber}.csv')
    if not os.path.isfile(chamber_file):
        return None, None, None

    data = read_csv(chamber_file, delimiter=",", skipinitialspace=True,
                    header=0, comment='#')
    time = np.array(data['Time'])
    pressure = np.array(data['Pressure'])
    volume = np.array(data['Volume'])
    return time, pressure, volume


def split_into_beats(time: np.ndarray, t_start: float, BCL: float,
                     n_beats: Optional[int]) -> list:
    """Return a list of boolean masks, one per beat, of length BCL each.

    Beats are defined as [t_start + k*BCL, t_start + (k+1)*BCL). Samples before
    t_start (e.g. the loading ramp, State=load) are dropped.
    """
    rel = time - t_start
    total = time.max() - t_start
    max_beats = int(np.ceil(total / BCL)) if total > 0 else 0
    if n_beats is not None:
        max_beats = min(max_beats, n_beats)

    masks = []
    for k in range(max_beats):
        masks.append((rel >= k * BCL) & (rel < (k + 1) * BCL))
    return masks


def plot_case(label: str, sim_dir: str, BCL: float, n_beats: Optional[int],
              t_start: float, output_dir: str, overlay: bool,
              title: Optional[str] = None, dpi: int = 300) -> None:
    """Plot all four chambers for one case, one beat per colour, save a PNG."""
    if overlay:
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        axes = {c: ax for c in CHAMBERS}
        chamber_colours = {'LV': 'red', 'RV': 'blue', 'LA': '#F6BE00', 'RA': 'green'}
    else:
        fig, axarr = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
        axes = dict(zip(CHAMBERS, axarr.flatten()))

    n_for_cmap = n_beats if n_beats else 5
    # Beat colours run from light green (first beat) to dark green (last beat).
    green_ramp = LinearSegmentedColormap.from_list(
        "light_dark_green", ["lightgreen", "darkgreen"])
    beat_cmap = green_ramp(np.linspace(0, 1, max(n_for_cmap, 1)))

    for chamber in CHAMBERS:
        time, pressure, volume = load_chamber(sim_dir, chamber)
        if time is None:
            print(f"[{label}] WARNING: missing cav.{chamber}.csv, skipping.")
            continue

        masks = split_into_beats(time, t_start, BCL, n_beats)
        ax = axes[chamber]
        for k, mask in enumerate(masks):
            if not mask.any():
                continue
            if overlay:
                colour = chamber_colours[chamber]
                lbl = chamber if k == len(masks) - 1 else None
            else:
                colour = beat_cmap[k % len(beat_cmap)]
                lbl = f'Beat {k + 1}'
            ax.plot(volume[mask], pressure[mask], color=colour,
                    linewidth=2.5, label=lbl)

        if not overlay:
            ax.set_title(chamber, fontsize=18, fontweight='bold')
            ax.set_xlabel('Volume [mL]', fontsize=14)
            ax.set_ylabel('Pressure [mmHg]', fontsize=14)
            ax.grid(True, alpha=0.3)

    if overlay:
        ax = axes['LV']
        ax.set_xlabel('Volume [mL]', fontsize=20, fontweight='bold')
        ax.set_ylabel('Pressure [mmHg]', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=18)
    else:
        # Single shared legend for the whole figure (the beat colours are
        # common to all four chambers, so there's no need to repeat it).
        n_show = n_beats if n_beats else len(beat_cmap)
        handles = [Line2D([0], [0], color=beat_cmap[k], lw=2.5,
                          label=f'Beat {k + 1}') for k in range(n_show)]
        fig.legend(handles=handles, loc='outside lower center',
                   ncol=n_show, fontsize=13)

    fig.suptitle(title or label, fontsize=22, fontweight='bold')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'pv_loops_{label}.png')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[{label}] saved -> {out_path}")


def parse_case(spec: str) -> tuple:
    """Parse a 'LABEL=/path/to/sim' CLI argument."""
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f"--case must be LABEL=PATH, got: {spec!r}")
    label, path = spec.split('=', 1)
    return label.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Plot PV loops (all beats, all chambers) per HCM case.")
    parser.add_argument('--case', type=parse_case, action='append', required=True,
                        help="Repeatable. Format: LABEL=/path/to/cycle_<N> dir "
                             "containing cav.*.csv files.")
    parser.add_argument('--BCL', type=float, required=True,
                        help="Basic cycle length in ms (length of one heartbeat).")
    parser.add_argument('--n_beats', type=int, default=None,
                        help="Number of beats to plot. Default: all available.")
    parser.add_argument('--t_start', type=float, default=0.0,
                        help="Time (ms) of the first beat; samples before this "
                             "(e.g. loading) are dropped. Default 0.")
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help="Directory for the output PNGs.")
    parser.add_argument('--title', type=str, default=None,
                        help="Figure title (e.g. the phenotype name). Defaults "
                             "to the case label. Applies to every --case in "
                             "this invocation, so call once per case.")
    parser.add_argument('--dpi', type=int, default=300,
                        help="Output resolution. Default 300.")
    parser.add_argument('--overlay', action='store_true',
                        help="Overlay all 4 chambers on one axes instead of a "
                             "2x2 grid (beats are not colour-separated).")
    args = parser.parse_args()

    for label, sim_dir in args.case:
        plot_case(label=label, sim_dir=sim_dir, BCL=args.BCL,
                  n_beats=args.n_beats, t_start=args.t_start,
                  output_dir=args.output_dir, overlay=args.overlay,
                  title=args.title, dpi=args.dpi)


if __name__ == '__main__':
    main()
