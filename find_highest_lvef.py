#!/usr/bin/env python3
"""
Script to find which simulation number had the highest LVEF.
The simulation number is with respect to X.txt (original row index).
"""

import numpy as np
import argparse
import os


def find_highest_lvef(basefolder: str, lvef_label: str = "LVEF") -> dict:
    """
    Find the simulation with the highest LVEF value.

    Parameters
    ----------
    basefolder : str
        Path to the folder containing data/X.txt, data/Y.txt,
        data/ylabels.txt, and output/output_mask_beat_5.txt
    lvef_label : str
        The label to search for in ylabels.txt (default: "LVEF")

    Returns
    -------
    dict
        Dictionary with simulation info including:
        - original_sim_number: 1-based index in X.txt
        - masked_index: 0-based index in the masked data
        - lvef_value: the LVEF value
        - input_parameters: the X values for that simulation
    """
    # Define file paths
    x_path = os.path.join(basefolder, "data", "X.txt")
    y_path = os.path.join(basefolder, "data", "Y.txt")
    ylabels_path = os.path.join(basefolder, "data", "ylabels.txt")
    mask_path = os.path.join(basefolder, "output", "output_mask_beat_5.txt")

    # Load data
    X_all = np.loadtxt(x_path, dtype=float)
    Y_all = np.loadtxt(y_path, dtype=float)
    mask = np.loadtxt(mask_path, dtype=float).astype(bool)

    with open(ylabels_path, "r") as f:
        ylabels = [line.strip() for line in f.readlines() if line.strip()]

    # Find LVEF column index
    lvef_idx = None
    for i, label in enumerate(ylabels):
        if lvef_label in label:
            lvef_idx = i
            print(f"Found '{label}' at column index {i}")
            break

    if lvef_idx is None:
        raise ValueError(f"Could not find '{lvef_label}' in ylabels: {ylabels}")

    # Use Y size as the reference (Y contains only the computed outputs)
    n_y = Y_all.shape[0]
    print(f"Y.txt has {n_y} rows, X.txt has {X_all.shape[0]} rows, mask has {mask.shape[0]} rows")

    # Trim mask and X to match Y size
    mask_trimmed = mask[:n_y]
    X_trimmed = X_all[:n_y]

    # Get the LVEF column
    if Y_all.ndim == 1:
        lvef_values = Y_all
    else:
        lvef_values = Y_all[:, lvef_idx]

    # Create array of original indices (0-based)
    original_indices = np.arange(n_y)

    # Apply mask to get valid data
    lvef_masked = lvef_values[mask_trimmed]
    original_indices_masked = original_indices[mask_trimmed]
    X_masked = X_trimmed[mask_trimmed]

    # Find the index of maximum LVEF in the masked data
    max_masked_idx = np.argmax(lvef_masked)

    # Get the original simulation number (convert to 1-based for user display)
    original_sim_number = original_indices_masked[max_masked_idx] + 1  # 1-based
    max_lvef_value = lvef_masked[max_masked_idx]
    input_params = X_masked[max_masked_idx]

    result = {
        "original_sim_number": original_sim_number,
        "original_index_0based": original_sim_number - 1,
        "masked_index": max_masked_idx,
        "lvef_value": max_lvef_value,
        "input_parameters": input_params,
        "total_simulations": n_y,
        "valid_simulations": mask_trimmed.sum(),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Find the simulation with the highest LVEF value"
    )
    parser.add_argument(
        "basefolder",
        type=str,
        help="Path to the folder containing data/ and output/ subfolders"
    )
    parser.add_argument(
        "--lvef-label",
        type=str,
        default="LVEF",
        help="Label to search for in ylabels.txt (default: LVEF)"
    )

    args = parser.parse_args()

    result = find_highest_lvef(args.basefolder, args.lvef_label)

    print("\n" + "=" * 60)
    print("HIGHEST LVEF SIMULATION")
    print("=" * 60)
    print(f"Simulation number (in X.txt):  {result['original_sim_number']}")
    print(f"  (0-based index:              {result['original_index_0based']})")
    print(f"LVEF value:                    {result['lvef_value']:.4f}")
    print(f"Total simulations:             {result['total_simulations']}")
    print(f"Valid simulations (after mask):{result['valid_simulations']}")
    print("-" * 60)
    print(f"Input parameters (X values):")
    print(f"  {result['input_parameters']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
