from GSA_library.mesh_utils import read_IGB_file, carp_to_pyvista
from GSA_library.pyvista_utils import read_pvcc_paraview_file, read_mesh

import argparse
import math
import numpy as np
import tqdm
import pyvista as pv
import os
import vtk

def pts_elem_to_pyvista(pts,elem,add_tags=False,el_type='Tt'):

    tmp_elem = elem

    if el_type == 'Tt':
        final_elem = tmp_elem[:,:4]  
        tets = np.column_stack((np.ones((final_elem.shape[0],),dtype=int)*4,final_elem)).flatten() 
        cell_type = np.ones((final_elem.shape[0],),dtype=int)*vtk.VTK_TETRA
    elif el_type == 'Tr':
        final_elem = tmp_elem[:,:3]
        tets = np.column_stack((np.ones((final_elem.shape[0],),dtype=int)*3,final_elem)).flatten() 
        cell_type = np.ones((final_elem.shape[0],),dtype=int)*vtk.VTK_TRIANGLE

    
    plt_msh = pv.UnstructuredGrid(tets,cell_type,pts)
    if add_tags:
        tags = tmp_elem[:,-1]
        plt_msh.cell_data["ID"] = tags

    return plt_msh

def rotation_matrix(u: np.ndarray, theta: float) -> np.ndarray:
	'''
	Calculate the rotation matrix for a given axis and angle.

	Parameters:
	- u (array-like): 3-element array representing the rotation axis.
	- theta (float): Angle of rotation in radians.

	Returns:
	- array: 3x3 rotation matrix.
	'''
	R = np.zeros((3,3),dtype=float)
	R[0,0] = u[0]**2 +math.cos(theta) * (1 - u[0]**2)
	R[0,1] = (1 -math.cos(theta)) * u[0] * u[1] - u[2] *math.sin(theta)
	R[0,2] = (1 -math.cos(theta)) * u[0] * u[2] + u[1] *math.sin(theta)	

	R[1,0] = (1 -math.cos(theta)) * u[0] * u[1] + u[2] *math.sin(theta)
	R[1,1] = u[1]**2 +math.cos(theta) * (1 - u[1]**2)
	R[1,2] = (1 - math.cos(theta)) * u[1] * u[2] - u[0] *math.sin(theta)	

	R[2,0] = (1 - math.cos(theta)) * u[0] * u[2] - u[1] *math.sin(theta)
	R[2,1] = (1 - math.cos(theta)) * u[1] * u[2] + u[0] *math.sin(theta)
	R[2,2] = u[2]**2 +math.cos(theta) * (1 - u[2]**2)

	return R

def rotate_mesh(plt_msh,
				lv_tag=1,
				mv_tag=7,
				tv_tag=8):

	print("Aligning mesh to centre it in 0,0,0 and to have the posterior-anterior direction as 0,-1,0...")

	pts = plt_msh.points
	elem = plt_msh.cells
	elem = np.reshape(elem,(int(plt_msh.cells.shape[0]/5),5))
	elem = elem[:,1:]

	tags = plt_msh.cell_data["ID"]
	eidx_lv = np.where(tags==lv_tag)[0]
	vtx_lv = np.unique(elem[eidx_lv,:].flatten())
	eidx_mv = np.where(tags==mv_tag)[0]
	vtx_mv = np.unique(elem[eidx_mv,:].flatten())
	eidx_tv = np.where(tags==tv_tag)[0]
	vtx_tv = np.unique(elem[eidx_tv,:].flatten())

	cog_mv = np.mean(pts[vtx_mv,:],axis=0)
	cog_tv = np.mean(pts[vtx_tv,:],axis=0)


	dd = np.linalg.norm(pts[vtx_lv,:]-cog_mv,axis=1)
	idx_apex = vtx_lv[np.argmax(dd)]

	cog = np.mean(np.array([cog_mv,cog_tv,pts[idx_apex,:]]),axis=0)


	pts_transformed = plt_msh.points-cog
	
	v0 = cog_tv-cog_mv
	v0 = v0/np.linalg.norm(v0)
	v1 = pts[idx_apex,:]-cog_mv
	v1 = v1/np.linalg.norm(v1)
	n = np.cross(v0,v1)


	n = n/np.linalg.norm(n)

	#### Rotate so the anterior direction is at the front

	target_direction = np.array([0,-1,0])

	axis_of_rotation = np.cross(n,target_direction)
	axis_of_rotation = axis_of_rotation/np.linalg.norm(axis_of_rotation)

	angle = math.acos(np.dot(n, target_direction))
	R = rotation_matrix(axis_of_rotation,angle)

	for i in range(pts.shape[0]):
		pts_transformed[i,:] = np.dot(R,pts_transformed[i,:])

	# Rotate so the apex is at the bottom
	target_direction_y = np.array([0,0,-1])

	cog_mv = np.mean(pts_transformed[vtx_mv,:],axis=0)
	long_axis = pts_transformed[idx_apex,:]-cog_mv
	long_axis = long_axis/np.linalg.norm(long_axis)


	angle_y = np.arccos(np.clip(np.dot(long_axis, target_direction_y), -1.0, 1.0))

	cross_product = np.cross(long_axis, target_direction_y)

	### To take into acount clockwise and anticlockwise angles
	if np.linalg.norm(cross_product) != 0:
		direction = np.sign(np.dot(cross_product, np.array([0, -1, 0])))
		angle_y *= direction


	print(f"Long axis: {long_axis}\nTarget direction: {target_direction_y}\nAngle: {angle_y}")
	R_y = rotation_matrix(target_direction,angle_y)	

	for i in range(pts.shape[0]):
		pts_transformed[i,:] = np.dot(R_y,pts_transformed[i,:])


	plt_msh.points = pts_transformed

	return plt_msh

def print_screenshot(plt_msh,
					 screenshot_name,
					 azimuth,
					 elevation,
					 roll,
					 fig_w           = 400,
					 fig_h           = 400,
					 meshcolor       = 'lightgray'):

	plotter = pv.Plotter(off_screen=True)
	plotter.background_color = "white"

	_ = plotter.add_mesh(mesh  = plt_msh,
					     color = meshcolor)
	plotter.camera.azimuth += azimuth
	plotter.camera.elevation += elevation
	plotter.camera.roll += roll

            
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
					 azimuth,
					 elevation,
					 roll,
					 window_size     = 400,
					 create_video       = False):

	if not os.path.exists(f"{meshname}.elem") or not os.path.exists(f"{meshname}.pts"):

		cmd = ["meshtool convert",
                f"-imsh={meshname}",
                f"-omsh={meshname}",
                "-ifmt=carp_bin",
                "-ofmt=carp_txt"]
            
		cmd_str = ' '.join(cmd)
		os.system(cmd_str) 

	pts,elem = read_mesh(meshname)

	pv_msh_init_original = pts_elem_to_pyvista(pts=pts, elem=elem, add_tags=True)
	pv_msh = rotate_mesh(pv_msh_init_original)


	_,u      = read_IGB_file(displacement_file)
	
	nt = u.shape[0]
	np = u.shape[1]

	if np!=pts.shape[0]:
		raise Exception("Mesh and displacement file dimensions do not match.")


	for t in tqdm.tqdm(range(nt)):
		print(f"Processing time step {t}/{nt-1}...")

		pv_msh.points = u[t,:,:]

		if not os.path.exists(f"{screenshot_basename}{t}.png"):
			print_screenshot(plt_msh         = pv_msh,
								 screenshot_name = f"{screenshot_basename}{str(t).zfill(3)}.png",
								 fig_w           = window_size,
								 fig_h           = window_size,
								 azimuth = azimuth,
								 elevation=elevation,
								 roll=roll)

	if create_video:
		cmd = ["ffmpeg -r",str(12),"-i",screenshot_basename+"%d.png"]
		cmd += ["-vcodec","libx264","-vf","scale="+str(window_size)+":"+str(window_size),screenshot_basename+".avi"]
		cmd_str = " ".join(cmd)
		os.system(cmd_str)

def file_exists(full_file_path):
    if not os.path.isfile(full_file_path):
        raise Exception("You need to have the file " + full_file_path)

def main(args):
	
    path2figure   = args.path2figure
    simfolder     = args.simfolder
    unloaded_mesh = args.unloaded_mesh
    create_video = args.create_video
    azimuth = args.azimuth
    elevation = args.elevation
    roll = args.roll
    output_basename = args.output_basename
	
    os.makedirs(path2figure, exist_ok=True)

    file_exists(f"{simfolder}/x.dynpt")
    
    visualise_motion(displacement_file   = f"{simfolder}/x.dynpt",
					 meshname            = f"{unloaded_mesh}",
					 screenshot_basename = f"{path2figure}/{output_basename}",
					 window_size         = 1000,
					 create_video = create_video,
					 azimuth = azimuth,
					 elevation = elevation,
					 roll = roll)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate a video from a cycle simulation.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--simfolder',     type=str, required=True)
    parser.add_argument('--unloaded_mesh', type=str, required=True)
    parser.add_argument('--path2figure',   type=str, required=True)
    parser.add_argument('--azimuth',  default=180, type=float)
    parser.add_argument('--elevation', default=0,  type=float)
    parser.add_argument('--roll', default=0,  type=float)
    parser.add_argument('--output_basename', default='cycle_anterior', type=str)
    parser.add_argument('--create_video', action='store_true')
    args = parser.parse_args()

    main(args)
	
