import os 
import argparse
import numpy as np
import tqdm
import pyvista as pv

from GSA_library.pyvista_utils import read_pvcc_paraview_file, carp_to_pyvista
from GSA_library.mesh_utils import read_elem, read_pts

def file_exists(full_file_path):
    if not os.path.isfile(full_file_path):
        raise Exception("You need to have the file " + full_file_path)

def main(args):

    basefolder       = args.basefolder
    path2figure      = args.path2figure
    initial_mesh_path     = args.initial_mesh_path
    initial_mesh_name     = args.initial_mesh_name
    first_simulation = args.first_simulation
    last_simulation  = args.last_simulation
    only_setup = args.only_setup

    initial_mesh = f"{initial_mesh_path}/{initial_mesh_name}"

    clipping_plane_origin           = (67234.6, 53056.1, 62056.3)
    clipping_plane_normal_anterior  = (0.25083,-0.230159,0.940279)
    clipping_plane_normal_posterior = tuple(-x for x in clipping_plane_normal_anterior)

    unloaded_volumes = np.loadtxt(os.path.join(basefolder,"unloaded_volumes.txt"),dtype=float)


    file_exists(f"{initial_mesh}.belem")
    file_exists(f"{initial_mesh}.blon")

    if not only_setup:
        file_exists(f"{path2figure}/camera_settings_anterior.pvcc")
    
        camera_settings_path_anterior  = f"{path2figure}/camera_settings_anterior.pvcc"
        camera_settings_path_posterior = f"{path2figure}/camera_settings_posterior.pvcc"
    
        f = open(camera_settings_path_anterior)
        camera_settings_anterior = read_pvcc_paraview_file(camera_settings_path_anterior)
  
        f = open(camera_settings_path_posterior)
        camera_settings_posterior = read_pvcc_paraview_file(camera_settings_path_posterior)
  
    ### Read original elem, it's the same for both configurations

    if not os.path.exists(f"{initial_mesh}.elem"):

        cmd = ["meshtool convert",
              f"-imsh={initial_mesh}",
              f"-omsh={initial_mesh}",
               "-ifmt=carp_bin",
               "-ofmt=carp_txt"]
        
        cmd_str = ' '.join(cmd)
        os.system(cmd_str)  
    
    if not only_setup:
        elem_file = read_elem(f"{initial_mesh}.elem")
        print("Read.")


    ### Read original pts

    if not os.path.exists(f"{initial_mesh}.pts"):

        cmd = ["meshtool convert",
              f"-imsh={initial_mesh}",
              f"-omsh={initial_mesh}",
               "-ifmt=carp_bin",
               "-ofmt=carp_txt"]
        
        cmd_str = ' '.join(cmd)
        os.system(cmd_str)  
    
    if not only_setup:
        initial_pts = read_pts(f"{initial_mesh}.pts")
        print("Read.")
        pv_msh_init = carp_to_pyvista(initial_pts,elem_file)



    for i in tqdm.tqdm(range(first_simulation, last_simulation+1)):

        if np.sum(unloaded_volumes[i,:])!=0:

            simulation_folder = os.path.join(basefolder,"unloading_"+str(i))

            unloaded_meshname = os.path.join(basefolder,"unloaded/myocardium_AV_FEC_BB_lvrv_unloaded_"+str(i))
            
            print("Copying files...")
            os.system(f"cp {simulation_folder}/cur_reference.bpts {unloaded_meshname}.bpts")
            os.system(f"cp {initial_mesh}.belem {unloaded_meshname}.belem")
            os.system(f"cp {initial_mesh}.blon {unloaded_meshname}.blon")
            print("Copied")
      

            ### Read unloaded pts

            if not os.path.exists(f"{unloaded_meshname}.pts"):

                cmd = ["meshtool convert",
                      f"-imsh={unloaded_meshname}",
                      f"-omsh={unloaded_meshname}",
                       "-ifmt=carp_bin",
                       "-ofmt=carp_txt"]
                
                cmd_str = ' '.join(cmd)
                os.system(cmd_str)  
            
            if not only_setup:
                unloaded_pts = read_pts(f"{unloaded_meshname}.pts")
                print("Read.")
                
                pv_msh_unloaded = carp_to_pyvista(unloaded_pts,elem_file)
                

                ##### Anterior
                print("Plotting anterior view...")
                plotter = pv.Plotter(off_screen=True)

                plotter.background_color = "white"

                _ = plotter.add_mesh(mesh    = pv_msh_unloaded,
                                    color   = 'red',
                                    opacity = 0.8)
                _ = plotter.add_mesh(mesh    = pv_msh_init,
                                    color   = 'lightgray',
                                    opacity = 0.1)

                plotter.camera.position    = camera_settings_anterior["position"]
                plotter.camera.focal_point = camera_settings_anterior["focal_point"]
                plotter.camera.up          = camera_settings_anterior["up"]
                plotter.camera.view_angle  = camera_settings_anterior["view_angle"]
                
                print("Taking screenshot...")
                plotter.screenshot(filename               = f"{path2figure}/unloaded_{i}_anterior.png",
                                transparent_background = None,
                                return_img             = True,
                                window_size            = [2000,2000]
                                )
                print("Done.")
                
                plotter.close()

                ##### Posterior
                print("Plotting posterior view...")
                plotter = pv.Plotter(off_screen=True)

                plotter.background_color = "white"

                _ = plotter.add_mesh(mesh    = pv_msh_unloaded,
                                    color   = 'red',
                                    opacity = 0.8)
                _ = plotter.add_mesh(mesh    = pv_msh_init,
                                    color   = 'lightgray',
                                    opacity = 0.1)

                plotter.camera.position    = camera_settings_posterior["position"]
                plotter.camera.focal_point = camera_settings_posterior["focal_point"]
                plotter.camera.up          = camera_settings_posterior["up"]
                plotter.camera.view_angle  = camera_settings_posterior["view_angle"]

                print("Taking screenshot...")
                plotter.screenshot(filename               = f"{path2figure}/unloaded_{i}_posterior.png",
                                transparent_background = None,
                                return_img             = True,
                                window_size            = [2000,2000]
                                )
                print("Done.")

                plotter.close()

                ##### Anterior clipped
                print("Plotting anterior view...")

                plotter = pv.Plotter(off_screen=True)

                plotter.background_color = "white"

                clip_filter = pv_msh_unloaded.clip(origin = clipping_plane_origin, 
                                                normal = clipping_plane_normal_anterior)
                
                _ = plotter.add_mesh(mesh    = clip_filter, 
                                    color   = 'red', 
                                    opacity = 1
                                    )
                _ = plotter.add_mesh(mesh    = pv_msh_init,
                                    color   = 'lightgray',
                                    opacity = 0.1
                                    )

                plotter.camera.position    = camera_settings_anterior["position"]
                plotter.camera.focal_point = camera_settings_anterior["focal_point"]
                plotter.camera.up          = camera_settings_anterior["up"]
                plotter.camera.view_angle  = camera_settings_anterior["view_angle"]
                
                print("Taking screenshot...")
                plotter.screenshot(filename               = f"{path2figure}/unloaded_{i}_anterior_clipped.png", 
                                transparent_background = None, 
                                return_img             = True,
                                window_size            = [2000,2000]
                                )
                print("Done.")

                plotter.close()
                
                ##### Posterior clipped
                print("Plotting posterior view clipped...")

                plotter = pv.Plotter(off_screen=True)

                plotter.background_color = "white"

                clip_filter = pv_msh_unloaded.clip(origin = clipping_plane_origin, 
                                                normal = clipping_plane_normal_posterior)
                
                _ = plotter.add_mesh(mesh    = clip_filter, 
                                    color   = 'red', 
                                    opacity = 1
                                    )
                _ = plotter.add_mesh(mesh    = pv_msh_init,
                                    color   = 'lightgray',
                                    opacity = 0.1
                                    )

                plotter.camera.position    = camera_settings_posterior["position"]
                plotter.camera.focal_point = camera_settings_posterior["focal_point"]
                plotter.camera.up          = camera_settings_posterior["up"]
                plotter.camera.view_angle  = camera_settings_posterior["view_angle"]

                print("Taking screenshot...")
                plotter.screenshot(filename               = f"{path2figure}/unloaded_{i}_posterior_clipped.png", 
                                transparent_background = None, 
                                return_img             = True,
                                window_size            = [2000,2000]
                                )
                print("Done.")

                plotter.close()
            
    print("Cleaning...")

    os.system(f"cp -r {basefolder}/unloaded {initial_mesh_path}/.")

    os.system(f"rm {initial_mesh}.pts")
    os.system(f"rm {initial_mesh}.elem")
    os.system(f"rm {initial_mesh}.lon")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate screenshots of the unloading geometries. It also setup the meshes for the cycle simulations.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/h01/new_unloading/unloading_simulations",help="Path to the simulations folder.")
    parser.add_argument('--path2figure', type=str, required=False)
    parser.add_argument('--first_simulation', type=int, required=False, default=-1)
    parser.add_argument('--last_simulation', type=int, required=False, default=-1)
    parser.add_argument('--initial_mesh_path',type=str,required=True)
    parser.add_argument('--initial_mesh_name',type=str,required=True)
    parser.add_argument('--only_setup', action='store_true', help="Use this command if you just want to set up the folder architecture ready but not taking the screenshots.")

    args = parser.parse_args()

    main(args)
