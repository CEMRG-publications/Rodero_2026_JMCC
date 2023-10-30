from GSA_library.mesh_utils import read_IGB_file, carp_to_pyvista
from GSA_library.pyvista_utils import read_pvcc_paraview_file, read_mesh

import argparse
import tqdm
import pyvista as pv
import os

def print_screenshot(plt_msh,
					 screenshot_name,
					 fig_w           = 400,
					 fig_h           = 400,
					 camera_settings = None,
					 meshcolor       = 'lightgray'):

	plotter = pv.Plotter(off_screen=True)
	plotter.background_color = "white"

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

def visualise_motion(displacement_file,
					 meshname,
					 screenshot_basename,
					 camera_settings = None,
					 window_size     = 400,
					 meshcolor       = 'lightgray'):

	pts,elem = read_mesh(meshname)
	_,u      = read_IGB_file(displacement_file)
	
	nt = u.shape[0]
	np = u.shape[1]

	if np!=pts.shape[0]:
		raise Exception("Mesh and displacement file dimensions do not match.")

	# initialise mesh
	pv_msh = carp_to_pyvista(pts,elem)

	for t in tqdm.tqdm(range(nt)):
		print(f"Processing time step {t}/{nt-1}...")

		pv_msh.points = u[t,:,:]

		if not os.path.exists(f"{screenshot_basename}{t}.png"):
			if camera_settings is None:
				print_screenshot(plt_msh         = pv_msh,
								 screenshot_name = f"{screenshot_basename}{t}.png",
								 fig_w           = window_size,
								 fig_h           = window_size)
			else:
				print_screenshot(plt_msh         = pv_msh,
								 screenshot_name = f"{screenshot_basename}{t}.png",
								 fig_w           = window_size,
								 fig_h           = window_size,
								 camera_settings = camera_settings,
								 meshcolor       = meshcolor)

	# cmd = ["ffmpeg -r",str(framerate),"-i",screenshot_basename+"%d.png"]
	# cmd += ["-vcodec","libx264","-vf","scale="+str(window_size)+":"+str(window_size),screenshot_basename+".avi"]
	# cmd_str = " ".join(cmd)
	# os.system(cmd_str)

def main(args):
	
    path2figure   = args.path2figure
    simfolder     = args.simfolder
    unloaded_mesh = args.unloaded_mesh
	
    os.makedirs(path2figure, exist_ok=True)
    
    camera_settings_path_anterior = f"{path2figure}/../camera_settings_anterior.pvcc"

    f = open(camera_settings_path_anterior)
    camera_settings_anterior = read_pvcc_paraview_file(camera_settings_path_anterior)
    
    visualise_motion(displacement_file   = f"{simfolder}/x.dynpt",
					 meshname            = f"{unloaded_mesh}",
					 screenshot_basename = f"{path2figure}/cycle_anterior",
					 camera_settings     = camera_settings_anterior,
					 window_size         = 1000)
	
    camera_settings_path_posterior = f"{path2figure}/../camera_settings_posterior.pvcc"

    f = open(camera_settings_path_posterior)
    camera_settings_posterior = read_pvcc_paraview_file(camera_settings_path_posterior)
    
    visualise_motion(displacement_file   = f"{simfolder}/x.dynpt",
					 meshname            = f"{unloaded_mesh}",
					 screenshot_basename = f"{path2figure}/cycle_posterior",
					 camera_settings     = camera_settings_posterior,
					 window_size         = 1000)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate a video from a cycle simulation.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--simfolder',     type=str, required=True)
    parser.add_argument('--unloaded_mesh', type=str, required=True)
    parser.add_argument('--path2figure',   type=str, required=True)
    args = parser.parse_args()

    main(args)
	
