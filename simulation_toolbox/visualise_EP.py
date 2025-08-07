from common.utils import rotate_mesh, load_json, read_elem, read_pts, pts_elem_to_pyvista, carp_to_pyvista
from common.visualisation import print_screenshot_video, print_activation_times

import argparse
import tqdm
import math
import numpy as np
import os
import tqdm
import pyvista as pv


def make_activation_video(meshname,
						  activation_file,
						  video_folder,
						  camera_file,
					 	  inactive_color="lightgray",
					 	  active_color="firebrick",
						  opacity=1.0):
	
	camera_settings = load_json(camera_file)

	# plt_msh = carp_to_pyvista(meshname)
	plt_msh = pv.read(meshname)

	
	act = np.loadtxt(activation_file,dtype=float)
	
	t0 = 0 
	tend = math.ceil(np.max(act[act < 1e6]))

	act[act < 0] = tend+10


	count = 0
	print(t0)
	print(tend)
	for t in tqdm.tqdm(range(t0,tend+1)):

		binary_vector = (act<=t)

		print_screenshot_video(plt_msh,
					           binary_vector,
					           video_folder+"/act_{:03d}.png".format(count),
					           camera_settings,
					           title="time = "+str(t)+" ms",
					           fig_w=1200,
					           fig_h=1200,
					           inactive_color=inactive_color,
					           active_color=active_color,
							   opacity=opacity)

		count += 1

def visualise_activation(meshname,
						  activation_file,
						  output_folder,
					 	  inactive_color="lightgray",
						  opacity=1.0):

	# camera_settings = load_json(camera_file)

	# plt_msh = carp_to_pyvista(meshname)
	
    # pts  = read_pts(meshname+'.pts')
    # elem = read_elem(meshname+'.elem',el_type='Tt',tags=True)

    # plt_msh = pts_elem_to_pyvista(pts=pts, elem=elem, add_tags=True)


    plt_msh = pv.read(meshname + ".vtk")  # meshname should now be path to the `.vtk` file


    plt_msh = rotate_mesh(plt_msh, target_direction=[0,-1,0])
	
    act = np.loadtxt(activation_file,dtype=float)
	
    t0 = 0 
    tend = math.ceil(np.max(act[act < 1e6]))

    act[act < 0] = tend+10

    binary_vector = (act<=tend)

    os.makedirs(output_folder, exist_ok=True)


    print_activation_times(plt_msh,
                            binary_vector,
                            f"{output_folder}/activation.png",
                            fig_w=1200,
                            fig_h=1200,
                            inactive_color=inactive_color,
                            opacity=opacity,
                            camera_elevation_increment=20)



def main(args):
    
    meshname = args.meshname
    activation_file = args.activation_file
    output_folder = args.output_folder

    visualise_activation(meshname = meshname,
						  activation_file = activation_file,
						  output_folder = output_folder,
					 	  inactive_color="lightgray",
						  opacity=1.0)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=""
                                     )
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--meshname',     
                        type=str, 
                        required=True, 
                        help="")
    parser.add_argument('--activation_file',     
                    type=str, 
                    required=True, 
                    help="")
    parser.add_argument('--output_folder',     
                    type=str, 
                    required=True, 
                    help="")
    args = parser.parse_args()

    main(args)
    
