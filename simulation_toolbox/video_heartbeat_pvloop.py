import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from pathlib import Path
import pyvista as pv
import shutil
import subprocess
import tqdm
import os
import vtk

from GSA_library.mesh_utils import read_IGB_file, read_elem
from GSA_library.pyvista_utils import read_pvcc_paraview_file

def read_pts(filename: str) -> np.ndarray:
    """
    Read a file containing points coordinates and return an array of these points.

    Args:
        filename (str): The name of the file to be read.

    Returns:
        numpy.ndarray: An array containing the coordinates of the points read from the file.
    """
    logging.info(f"Reading {filename}...")
    return np.loadtxt(filename, dtype=float, skiprows=1)

def carp_to_pyvista(meshname: str) -> tuple:
    """
    Reads a mesh file in CARP format and converts it into a pyvista format.

    Args:
        meshname (str): The basename of the mesh file to be read.

    Returns:
        tuple: A tuple containing the pyvista mesh object representing the input mesh and an array containing the tags associated with each element in the mesh.
    """
    # Read points coordinates
    pts = read_pts(meshname + '.pts')

    # Read element tags
    elem_tags = read_elem(meshname + '.elem', el_type='Tt', tags=True)
    logging.info("Read.")

    # Extract element indices and tags
    elem = elem_tags[:, :4]
    tags = elem_tags[:, -1]

    # Create pyvista mesh object
    tets = np.column_stack((np.ones((elem.shape[0],), dtype=int) * 4, elem)).flatten()
    cell_type = np.ones((elem.shape[0],), dtype=int) * vtk.VTK_TETRA
    plt_msh = pv.UnstructuredGrid(tets, cell_type, pts)

    return plt_msh, tags

def print_screenshot_regions(plt_msh: pv.UnstructuredGrid, 
                             tags_tmp: np.ndarray, 
                             screenshot_name: str,
                             camera_settings: dict, 
                             fig_w: int = 1200, 
                             fig_h: int = 1200) -> None:
    """
    Takes a pyvista mesh object, a numpy array of tags, a screenshot name, camera settings, and optional figure width and height as inputs.
    Creates a plotter object, adds the mesh to the plotter with specified opacity and color mapping based on the tags, sets the camera position and other settings, takes a screenshot of the plot, and saves it to the specified file.

    Args:
        plt_msh (pyvista.UnstructuredGrid): A pyvista mesh object representing the input mesh.
        tags_tmp (numpy.ndarray): An array containing the tags associated with each element in the mesh.
        screenshot_name (str): The name of the file to save the screenshot.
        camera_settings (dict): A dictionary containing camera settings including position, focal point, up vector, and view angle.
        fig_w (int, optional): The width of the figure in pixels. Default is 1200.
        fig_h (int, optional): The height of the figure in pixels. Default is 1200.

    Returns:
        None. The function saves a screenshot of the plot to the specified file.
    """
    tags_list = np.sort(np.unique(tags_tmp))
    
    cmap = plt.get_cmap("coolwarm").reversed()
    rgba_colors = cmap(np.linspace(0, 1, len(tags_list)))
    colors = [f'#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}' for r, g, b, _ in rgba_colors]

    opacity = np.ones_like(tags_tmp, dtype=float)
    tags = tags_tmp

    plt_msh.cell_data["ID"] = tags
    plt_msh.cell_data["opacity"] = opacity

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'

    _ = plotter.add_mesh(plt_msh,
                        opacity="opacity",
                        scalars="ID",
                        n_colors=len(colors) + 1,
                        cmap=colors)

    plotter.remove_scalar_bar()
    
    plotter.camera.position = camera_settings["position"]
    plotter.camera.focal_point = camera_settings["focal_point"]
    plotter.camera.up = camera_settings["up"]
    plotter.camera.view_angle = camera_settings["view_angle"]

    plotter.screenshot(filename=screenshot_name, transparent_background=None, return_img=True, window_size=[fig_w, fig_h])
    plotter.close()

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

def visualise_motion(displacement_file: str, 
                     meshname: str, 
                     basename_motion: str, 
                     basename_pvloop: str,
                     camera_settings_filename: str, 
                     path_to_simulation: str, 
                     BCL: int, 
                     dt: int = 10,
                     only_LV: bool = False,
                     save_dt: int = 1) -> None:
    """
    Visualizes the motion of a mesh over time by creating a series of screenshots and PV loop plots.

    Args:
        displacement_file: The file containing displacement data.
        meshname: The name of the mesh.
        basename_motion: The base name for the motion screenshots.
        basename_pvloop: The base name for the PV loop plots.
        camera_settings_filename: The file containing camera settings.
        path_to_simulation: The path to the simulation data.
        BCL: The Basic Cycle Length.
        dt: The time step (default is 10).
        save_dt: Saving timestep (default is 1).

    Returns:
        None. The function generates a series of screenshots and PV loop plots.
    """
    _, u = read_IGB_file(displacement_file)
    nt = u.shape[0]

    # Initialize mesh
    pv_msh, tags_temp = carp_to_pyvista(meshname)
    camera_settings = read_pvcc_paraview_file(camera_settings_filename)

    if not only_LV:
        chambers = ['LV', 'RV', 'LA', 'RA']
        colours = ['red', 'blue', '#F6BE00', 'green']
    else:
        chambers = ['LV']
        colours = ['red']

    pressure_dict = {}
    volume_dict = {}

    for chamber in chambers:
        p, v = extract_pressure_volumes(path_to_simulation=path_to_simulation,
                                               chamber=chamber,
                                               BCL=BCL)

        pressure_dict[chamber] = p
        volume_dict[chamber] = v


    for t in tqdm.tqdm(range(0, nt, save_dt)):
        logging.info(f"Processing time step {t}/{nt-1}...")

        pv_msh.points = u[t, :, :]

        screenshot_path = Path(f'{basename_motion}{t}.png')
        pvloop_path = Path(f'{basename_pvloop}{t}.png')
        if not screenshot_path.exists():
            logging.info('Printing screenshot...')
            print_screenshot_regions(plt_msh=pv_msh,
                                     tags_tmp=tags_temp,
                                     screenshot_name=screenshot_path,
                                     camera_settings=camera_settings,
                                     fig_w=1200,
                                     fig_h=1200)
        if not pvloop_path.exists():
            logging.info('Printing PV loop...')
            
            print_PV_loop(chambers=chambers, 
                          screenshot_name=pvloop_path,
                          volume_dict=volume_dict,
                          pressure_dict=pressure_dict,
                          colours=colours,
                          end_time = int(t*dt))

def print_PV_loop(chambers: list, 
                  screenshot_name: str, 
                  volume_dict: dict, 
                  pressure_dict: dict, 
                  colours: list, 
                  end_time: int) -> None:
    """
    Plot the pressure-volume (PV) loop for each chamber in a cardiac simulation.

    Parameters:
    chambers: A list of chamber names.
    screenshot_name: The name of the file to save the PV loop screenshot.
    volume_dict: A dictionary containing volume data for each chamber.
    pressure_dict: A dictionary containing pressure data for each chamber.
    colours: A list of colors for each chamber.
    end_time: The end time for the PV loop plot.

    Returns:
    None. The function generates a PV loop plot and saves it as a screenshot.
    """
    if len(chambers) > 1:
        ax = plt.figure(figsize=(10,10), constrained_layout=True).subplots(2, 2)
        ax = ax.flatten()

    else:
        ax = plt.figure(figsize=(10,10), constrained_layout=True)


    for j, chamber_name in enumerate(chambers):

        volume = volume_dict[chamber_name]
        pressure = pressure_dict[chamber_name]

        volume_to_plot = volume[:end_time]
        pressure_to_plot = pressure[:end_time]

        if len(chambers) > 1:

            ax[j].plot(volume_to_plot,pressure_to_plot,color=colours[j],linewidth=3.0)
            ax[j].set_xlabel(chamber_name+' volume [mL]')
            ax[j].set_ylabel(chamber_name+' pressure [mmHg]')
            ax[j].set_xlim(xmin=min(volume)-0.1*(max(volume)-min(volume)),xmax=max(volume)+0.1*(max(volume)-min(volume)))
            ax[j].set_ylim(ymin=min(pressure)-0.1*(max(pressure)-min(pressure)),ymax=max(pressure)+0.1*(max(pressure)-min(pressure)))
        
        else:
            plt.plot(volume_to_plot, pressure_to_plot, color=colours[j], linewidth=3.0)
            plt.xlabel(chamber_name + ' volume (mL)', fontsize=24, fontweight='bold')
            plt.ylabel(chamber_name + ' pressure (mmHg)', fontsize=24, fontweight='bold')
            plt.xlim(xmin=50,xmax=200)
            plt.ylim(ymin=0,ymax=150)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)


        
    plt.savefig(os.path.join(screenshot_name),dpi=300)
    plt.close('all')


def main(args):
    meshname          = args.meshname
    camera_file       = args.camera_file
    displacement_file = args.disp
    BCL               = args.bcl
    dt                = args.dt
    fps               = args.fps
    only_LV           = args.only_LV
    video_directory   = args.video_directory
    video_name        = args.video_name
    save_dt           = args.save_dt

    # Extract the directory of the displacement file
    disp_directory = os.path.dirname(os.path.abspath(displacement_file))
    output_directory_motion = Path(disp_directory) / 'delete' / 'motion'
    output_directory_motion.mkdir(parents=True, exist_ok=True)

    # Generate the full path for the output_name in the 'motion' subfolder
    basename_motion = os.path.join(output_directory_motion, "cycle_")

        # Create the output directory if it doesn't exist
    output_directory_pvloop = os.path.join(disp_directory, "delete", "PV_loops")
    os.makedirs(output_directory_pvloop, exist_ok=True)

    # Generate the full path for the output_name in the 'motion' subfolder
    basename_pvloop = os.path.join(output_directory_pvloop, "pvloop_")
    
    visualise_motion(displacement_file=displacement_file,
                     meshname=meshname,
                     basename_motion = basename_motion,
                     basename_pvloop = basename_pvloop,
                     camera_settings_filename=camera_file,
                     path_to_simulation=disp_directory,
                     BCL=BCL,
                     dt=dt,
                     only_LV=only_LV,
                     save_dt=save_dt)
    
    if video_directory is None:
        video_directory = disp_directory

    subprocess.run(['ffmpeg', '-y', '-r', str(fps), '-i', f'{output_directory_motion}/cycle_%d.png', '-i', f'{output_directory_pvloop}/pvloop_%d.png', '-filter_complex', '[0:v]scale=-1:3000[0v];[1:v]scale=-1:3000[1v];[0v][1v]hstack=inputs=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{video_directory}/{video_name}.mp4'])

    # shutil.rmtree(os.path.join(disp_directory, "delete"), ignore_errors=True)

    logging.info(f"Your video is in {video_directory}/{video_name}.mp4")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to generate a video of a heart beating and plot the PV loop of the last heartbeat.")
    parser.add_argument('--meshname', type=str, help="Full path of the meshname without the extension.")
    parser.add_argument('--camera_file', type=str, help="Full path of the camera settings file (including .pvcc extension). You can get this file from Paraview.")
    parser.add_argument('--disp', type=str, help="Full path of the x.dynpt displacement file, including the extension.")
    parser.add_argument('--bcl', default=833, type=int, help="Basic cycle length in ms.")
    parser.add_argument('--dt', default=10, type=int, help="spacedt used in the simulation (every how many timestep was displacement saved).")
    parser.add_argument('--fps', default=24, type=int, help="Frames per second for the final video.")
    parser.add_argument('--only_LV', default=False, action='store_true', help="Flag to plot only the PV loop of the left ventricle.")
    parser.add_argument('--video_directory', default=None, type=str, help="Path where the video will be saved. If None, it will be saved in the same directory as the cycle file.")
    parser.add_argument('--video_name', default="motion_pvloop", type=str, help="Name of the video file.")
    parser.add_argument('--save_dt', default=1, type=int, help="Interval to save the screenshots.")

    args = parser.parse_args()
    
    main(args)
