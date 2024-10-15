from GSA_library.mesh_utils import *
from GSA_library.pyvista_utils import read_pvcc_paraview_file, read_mesh, carp_to_pyvista, read_IGB_file
from simulation_toolbox.common.utils import file_exists

import argparse
import tqdm
import json
import pyvista as pv
import os

def print_screenshot(plt_msh,
                     screenshot_name,
                     fig_w                          = 400,
                     fig_h                          = 400,
                     camera_settings                = None,
                     meshcolor                      = 'lightgray',
                     clipped                        = False,
                     clipping_plane_origin          = None,
                     clipping_plane_normal = None  
                     ):

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"
                
    if clipped:
        clip_filter = plt_msh.clip(origin = clipping_plane_origin, normal = clipping_plane_normal)

        _ = plotter.add_mesh(mesh  = clip_filter,
                            color = meshcolor)
    
    else:

        _ = plotter.add_mesh(mesh  = plt_msh,
                            color = meshcolor)
    

    plotter.camera.position    = camera_settings["position"]
    plotter.camera.focal_point = camera_settings["focal_point"]
    plotter.camera.up          = camera_settings["up"]
    plotter.camera.view_angle  = camera_settings["view_angle"]
            
    print("Taking screenshot...")

    plotter.screenshot(filename            = screenshot_name,
                    transparent_background = None,
                    return_img             = True,
                    window_size            = [fig_w,fig_h])
    print("Done.")

    plotter.close()

def visualise_motion(screenshot_basename,
                     pts,
                     elem,
                     u,
                     camera_settings = None,
                     window_size     = 400,
                     meshcolor       = 'lightgray',
                     original        = False,
                     clipped         = False,
                     clipping_plane_origin = None,
                     clipping_plane_normal = None):
    
    nt = u.shape[0]
    np = u.shape[1]

    if np!=pts.shape[0]:
        raise Exception("Mesh and displacement file dimensions do not match.")

    # initialise mesh
    pv_msh = carp_to_pyvista(pts,elem)

    if clipped:
            screenshot_basename += "_clipped"
    


    for t in tqdm.tqdm(range(nt)):
        print(f"Processing time step {t}/{nt-1}...")

        pv_msh.points = u[t,:,:]
        
        if original:

            if not os.path.exists(f"{screenshot_basename}{t}.png"):
                print_screenshot(plt_msh         = pv_msh,
                                 screenshot_name = f"{screenshot_basename}{t}.png",
                                 fig_w           = window_size,
                                 fig_h           = window_size,
                                 camera_settings = camera_settings,
                                 meshcolor       = meshcolor)
        if clipped:

            if not os.path.exists(f"{screenshot_basename}{t}.png"):
                print_screenshot(plt_msh         = pv_msh,
                                 screenshot_name = f"{screenshot_basename}{t}.png",
                                 fig_w           = window_size,
                                 fig_h           = window_size,
                                 camera_settings = camera_settings,
                                 meshcolor       = meshcolor,
                                 clipping_plane_origin = clipping_plane_origin,
                                 clipping_plane_normal = clipping_plane_normal)





def main(args):
    
    path2figure   = args.path2figure
    simfolder     = args.simfolder
    unloaded_mesh = args.unloaded_mesh
    original      = args.original
    clipped       = args.clipped
    clipping_settings_path = args.clipping_settings_path

    
    if not original and not clipped:
        parser.error("You need to specify if you want a clipped or a original geometry.")
    if clipped and clipping_settings_path is None:
        parser.error("To create the clipped screenshot you must specify the clipping settings file path.")
    
    os.makedirs(path2figure, exist_ok=True)

    files_to_check = [f"{unloaded_mesh}.belem",
                      f"{unloaded_mesh}.bpts",
                      f"{simfolder}/x.dynpt",
                      f"{path2figure}/../camera_settings_anterior.pvcc",
                      f"{path2figure}/../camera_settings_posterior.pvcc",
                      clipping_settings_path]

    file_exists(files_to_check=files_to_check)

        
    ## We read the displacement and the mesh only once

    pts,elem = read_mesh(unloaded_mesh)
    _,u      = read_IGB_file(f"{simfolder}/x.dynpt")
    
    camera_settings_path_anterior = f"{path2figure}/../camera_settings_anterior.pvcc"

    f = open(camera_settings_path_anterior)
    camera_settings_anterior = read_pvcc_paraview_file(camera_settings_path_anterior)

    if clipped:
        f = open(clipping_settings_path)
        clipping_settings = json.load(f)
        f.close()

        clipping_plane_origin           = tuple(clipping_settings["origin"])
        clipping_plane_normal_anterior  = tuple(clipping_settings["normal"])
        clipping_plane_normal_posterior = tuple(-x for x in clipping_plane_normal_anterior)

        print(clipping_plane_origin)
        print(clipping_plane_normal_anterior)
    else:
        clipping_plane_origin           = None
        clipping_plane_normal_anterior  = None
        clipping_plane_normal_posterior = None
    
    visualise_motion(screenshot_basename = f"{path2figure}/cycle_anterior",
                    camera_settings      = camera_settings_anterior,
                    window_size          = 1000,
                    pts                  = pts,
                    elem                 = elem,
                    u                    = u,
                    original             = original,
                    clipped              = clipped,
                    clipping_plane_origin = clipping_plane_origin,
                    clipping_plane_normal = clipping_plane_normal_anterior)
    
    camera_settings_path_posterior = f"{path2figure}/../camera_settings_posterior.pvcc"

    f = open(camera_settings_path_posterior)
    camera_settings_posterior = read_pvcc_paraview_file(camera_settings_path_posterior)
    
    visualise_motion(screenshot_basename = f"{path2figure}/cycle_posterior",
                        camera_settings  = camera_settings_posterior,
                        window_size      = 1000,
                        pts              = pts,
                        elem             = elem,
                        u                = u,
                        original         = original,
                        clipped          = clipped,
                    clipping_plane_origin = clipping_plane_origin,
                    clipping_plane_normal = clipping_plane_normal_posterior)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Set of functions to print screenshots of the motion from a mechanics simulation.",
                                     epilog=f"When finished, copy over the images to your local machine and run in the following command in the images folder: ffmpeg -r FRAMERATE -i SCREENSHOT_BASENAME\%d.png -vcodec libx264 -vf scale=WINDOW_SIZE:WINDOW_SIZE VIDEO_NAME.avi"
                                     )
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--simfolder',     
                        type=str, 
                        required=True, 
                        help="Full path to the simulation folder.")
    parser.add_argument('--unloaded_mesh', 
                        type=str, 
                        required=True, 
                        help="Full path and basename of the unloaded mesh.")
    parser.add_argument('--path2figure',   
                        type=str, 
                        required=True, 
                        help="Full path to the folder where the screenshots will be saved. The camera settings are expected to be in the parent folder. If it does not exist it will be created.")
    parser.add_argument('--original',   
                        action='store_true',
                        default=False, 
                        help="If True, it prints the beating heart in its volumetric form, meaning not clipped. You must specify --original or --clipped or both.")
    parser.add_argument('--clipped',   
                        action='store_true',
                        default=False, help="If True, it prints the beating heart in its clipped form. You must provide the directory to the clipping settings file. You must specify --original, --clipped or both.")
    parser.add_argument('--clipping_settings_path',
                        type=str,
                        default=None,
                        help="Directory to the clipping settings file, including file name.")
    args = parser.parse_args()

    main(args)
    
