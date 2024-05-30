from pandas import read_csv
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
def extract_pressure_volumes(path_to_simulation: str, 
                             chamber: str, 
                             BCL: int) -> tuple:
    """
    Extracts pressure and volume data for a specific chamber in a simulation.

    Args:
        path_to_simulation (str): The path to the directory containing the simulation data.
        chamber (str): The name of the chamber for which to extract pressure and volume data.
        BCL (int): The Basic Cycle Length, which determines the time range to extract.

    Returns:
        pressure (numpy.ndarray): An array of pressure values for the specified chamber.
        volume (numpy.ndarray): An array of volume values for the specified chamber.
    """

    chamber_structure = read_csv(os.path.join(path_to_simulation,f'cav.{chamber}.csv'), delimiter=",",
                    skipinitialspace=True, header=0, comment='#')

    time = np.array(chamber_structure['Time'])
    volume = np.array(chamber_structure['Volume'])
    pressure = np.array(chamber_structure['Pressure'])

    # Calculate the start time based on the Basic Cycle Length
    start = time[-1] - BCL

    # Select the relevant time range for the volume and pressure arrays
    plot_time = np.where(time >= start)[0]
    volume = volume[plot_time]
    pressure = pressure[plot_time]

    return pressure, volume

def main(args):
      
    simsfolder = args.simsfolder
    reference_sim = args.reference_sim
    new_sim = args.new_sim
    BCL = args.BCL
    crashed = args.crashed

    path2output = f"{simsfolder}/figures"
    
    reference_color = 'red'
    new_color       = 'purple'
    

    p, v = extract_pressure_volumes(path_to_simulation=f"{simsfolder}/{reference_sim}",
                                               chamber='LV',
                                               BCL=BCL)
    pressure_dict = {}
    volume_dict   = {}

    pressure_dict['LV'] = p
    volume_dict['LV'] = v


    volume = volume_dict['LV']
    pressure = pressure_dict['LV']

    reference_volume = volume
    reference_pressure = pressure

    if not crashed:
        p, v = extract_pressure_volumes(path_to_simulation=f"{simsfolder}/{new_sim}",
                                                chamber='LV',
                                                BCL=BCL)
        pressure_dict = {}
        volume_dict   = {}

        pressure_dict['LV'] = p
        volume_dict['LV'] = v


        volume = volume_dict['LV']
        pressure = pressure_dict['LV']

        new_volume = volume
        new_pressure = pressure

    ax = plt.figure(figsize=(10,10), constrained_layout=True)


    plt.plot(reference_volume, reference_pressure, '--',color=reference_color, linewidth=3.0,alpha=0.2)
    if not crashed:
        plt.plot(new_volume, new_pressure, color=new_color, linewidth=3.0)
    plt.xlabel('LV volume (mL)', fontsize=24, fontweight='bold')
    plt.ylabel('LV pressure (mmHg)', fontsize=24, fontweight='bold')
    plt.xlim(xmin=50,xmax=200)
    plt.ylim(ymin=0,ymax=150)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    if not crashed:
        plt.legend(['Target simulation', 'Your simulation'],fontsize=22)
    else:
        plt.legend(['Target simulation'],fontsize=22)

    if not crashed:
        if reference_sim != new_sim:
            plt.text(0.5, 0.75, 'Better luck \nnext time! :)', fontsize=40, alpha=0.1, ha='center', va='center', rotation=0, transform=plt.gcf().transFigure)
        else:
            plt.text(0.5, 0.75, 'Success! :D', fontsize=40, alpha=0.1, ha='center', va='center', rotation=0, transform=plt.gcf().transFigure)
    else:
        plt.text(0.5, 0.5, 'Your simulation \ncrashed! :(', fontsize=50, alpha=1, ha='center', va='center', rotation=0, transform=plt.gcf().transFigure, bbox=dict(facecolor='purple', alpha=0.5, boxstyle='round,pad=0.1',edgecolor='none'))

        plt.text(0.75, 0.25, 'and you don\'t know why', fontsize=20, alpha=0.5, ha='center', va='center', rotation=25, transform=plt.gcf().transFigure)

        plt.text(0.25, 0.15, 'time to cry', fontsize=10, alpha=0.5, ha='center', va='center', rotation=0, transform=plt.gcf().transFigure)


    screenshot_name = f"{simsfolder}/../figures/comparison_{reference_sim}_{new_sim}.png"
    plt.savefig(os.path.join(screenshot_name),dpi=300)
    plt.close('all')
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    

    parser.add_argument('--simsfolder', type=str, required=True)
    parser.add_argument('--reference_sim', type=str, required=True)
    parser.add_argument('--new_sim', type=str, required=True)
    parser.add_argument('--BCL', type=float, required=True)
    parser.add_argument('--crashed', action='store_true', default=False, required=False)

    args = parser.parse_args()

    main(args)
