import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import argparse

def extract_pressure_volumes(path_to_simulation: str, chamber: str, BCL: int) -> tuple:
    """
    Extracts pressure and volume data for a specific chamber in a simulation.

    Args:
        path_to_simulation (str): Directory with simulation data.
        chamber (str): Chamber name (e.g., 'LV').
        BCL (int): Basic Cycle Length in ms.

    Returns:
        Tuple of pressure and volume arrays from the last heartbeat.
    """
    chamber_file = os.path.join(path_to_simulation, f'cav.{chamber}.csv')
    chamber_data = read_csv(chamber_file, delimiter=",", skipinitialspace=True, header=0, comment='#')

    time = np.array(chamber_data['Time'])
    volume = np.array(chamber_data['Volume'])
    pressure = np.array(chamber_data['Pressure'])

    start_time = time[-1] - BCL
    indices = np.where(time >= start_time)[0]

    return pressure[indices], volume[indices]

def print_PV_loop(chambers: list, 
                  path_to_simulation: str, 
                  BCL: int, 
                  colours: list, 
                  output_path: str) -> None:
    """
    Plots the complete PV loops for all specified chambers on a single plot.

    Args:
        chambers (list): List of chamber names (e.g., ['LV', 'RV']).
        path_to_simulation (str): Path to the directory containing chamber CSVs.
        BCL (int): Basic cycle length (ms).
        colours (list): List of colours for each chamber.
        output_path (str): Full path to save the plot image.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 10), constrained_layout=True)

    for chamber, color in zip(chambers, colours):
        pressure, volume = extract_pressure_volumes(path_to_simulation, chamber, BCL)
        plt.plot(volume, pressure, label=chamber, color=color, linewidth=3.0)

    plt.xlabel('Volume [mL]', fontsize=20, fontweight='bold')
    plt.ylabel('Pressure [mmHg]', fontsize=20, fontweight='bold')
    # plt.title('Pressure-Volume Loops of Heart Chambers', fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot PV loops for heart chambers.")
    parser.add_argument('--path_to_simulation', type=str, default="./sim_data", help="Path to the simulation directory")
    parser.add_argument('--output_path', type=str, default="pv_loops.png", help="Output file path for the plot")
    parser.add_argument('--BCL', type=int, default=833, help="Basic cycle length in ms")
    args = parser.parse_args()

    chambers = ['LV', 'RV', 'LA', 'RA']
    colours = ['red', 'blue', '#F6BE00', 'green']

    print_PV_loop(chambers=chambers,
                  path_to_simulation=args.path_to_simulation,
                  BCL=args.BCL,
                  colours=colours,
                  output_path=args.output_path)

if __name__ == "__main__":
    main()
