import os 
import argparse
import math
import numpy as np
import tqdm
import pyvista as pv

from common.mesh_io import read_elem, read_pts

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

def file_exists(full_file_path):
    if not os.path.isfile(full_file_path):
        raise Exception("You need to have the file " + full_file_path)

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

def main(args):

    basefolder       = args.basefolder
    path2figure      = args.path2figure
    initial_mesh_path     = args.initial_mesh_path
    initial_mesh_name     = args.initial_mesh_name
    first_simulation = args.first_simulation
    last_simulation  = args.last_simulation
    only_setup = args.only_setup

    initial_mesh = f"{initial_mesh_path}/{initial_mesh_name}"

    clipping_plane_origin           = (0, 0, 0)
    clipping_plane_normal_anterior  = (0,-1,0)
    clipping_plane_normal_posterior = tuple(-x for x in clipping_plane_normal_anterior)

    unloaded_volumes = np.loadtxt(os.path.join(basefolder,"unloaded_volumes.txt"),dtype=float)

    file_exists(f"{initial_mesh}.belem")
    file_exists(f"{initial_mesh}.blon")

  
    ### Read original elem, it's the same for both configurations
    if not only_setup:

        if not os.path.exists(f"{initial_mesh}.elem"):

            cmd = ["meshtool convert",
                f"-imsh={initial_mesh}",
                f"-omsh={initial_mesh}",
                "-ifmt=carp_bin",
                "-ofmt=carp_txt"]
            
            cmd_str = ' '.join(cmd)
            os.system(cmd_str)  
    
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
    
        initial_pts = read_pts(f"{initial_mesh}.pts")
        print("Read.")

        print("Loading vtk package...")

        import vtk

        pv_msh_init_original = pts_elem_to_pyvista(pts=initial_pts, elem=elem_file, add_tags=True)
        pv_msh_init = rotate_mesh(pv_msh_init_original)


    t = tqdm.tqdm(range(last_simulation - first_simulation + 1),colour='#C3B1E1')
    for unloading_row_num in t:
        if np.ndim(unloaded_volumes) > 1:
            unloaded_check = np.sum(unloaded_volumes[unloading_row_num,:])
        else:
            unloaded_check = np.sum(unloaded_volumes[unloading_row_num])
        if unloaded_check!=0:

            i = first_simulation + unloading_row_num

            simulation_folder = os.path.join(basefolder,"unloading_"+str(i))

            unloaded_meshname = os.path.join(basefolder,"unloaded/myocardium_AV_FEC_BB_lvrv_unloaded_"+str(i))
            
            t.set_description(f"Copying mesh {i}...")
            if os.path.isfile(f"{simulation_folder}/reference.bpts"):
                os.system(f"cp {simulation_folder}/reference.bpts {unloaded_meshname}.bpts")
                os.system(f"cp {initial_mesh}.belem {unloaded_meshname}.belem")
                os.system(f"cp {initial_mesh}.blon {unloaded_meshname}.blon")
            elif os.path.isfile(f"{simulation_folder}/reference.pts"):
                os.system(f"cp {simulation_folder}/reference.pts {unloaded_meshname}.pts")
                os.system(f"cp {initial_mesh}.elem {unloaded_meshname}.elem")
                os.system(f"cp {initial_mesh}.lon {unloaded_meshname}.lon")
                cmd = ["meshtool convert",
                        f"-imsh={unloaded_meshname}",
                        f"-omsh={unloaded_meshname}",
                        "-ifmt=carp_txt",
                        "-ofmt=carp_bin"]
                cmd_str = ' '.join(cmd)
                os.system(cmd_str) 

                os.system(f"rm {unloaded_meshname}.lon {unloaded_meshname}.pts {unloaded_meshname}.elem 2>/dev/null")
            
      
            if not only_setup:
                ### Read unloaded pts

                if not os.path.exists(f"{unloaded_meshname}.pts"):

                    cmd = ["meshtool convert",
                        f"-imsh={unloaded_meshname}",
                        f"-omsh={unloaded_meshname}",
                        "-ifmt=carp_bin",
                        "-ofmt=carp_txt"]
                    
                    cmd_str = ' '.join(cmd)
                    os.system(cmd_str)  
            
            
                unloaded_pts = read_pts(f"{unloaded_meshname}.pts")
                print("Read.")
                
                pv_msh_unloaded_original_position = pts_elem_to_pyvista(pts=unloaded_pts, elem=elem_file, add_tags=True)
                
                pv_msh_unloaded = rotate_mesh(pv_msh_unloaded_original_position)

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
                plotter.camera.azimuth += 180


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

                plotter.camera.azimuth += 180

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


                print("Taking screenshot...")
                plotter.screenshot(filename               = f"{path2figure}/unloaded_{i}_posterior_clipped.png", 
                                transparent_background = None, 
                                return_img             = True,
                                window_size            = [2000,2000]
                                )
                print("Done.")

                plotter.close()
            
    print("Cleaning...")

    os.makedirs(f"{initial_mesh_path}/unloaded", exist_ok=True)
    os.system(f"cp {basefolder}/unloaded/*.belem {initial_mesh_path}/unloaded/.")
    os.system(f"cp {basefolder}/unloaded/*.bpts {initial_mesh_path}/unloaded/.")
    os.system(f"cp {basefolder}/unloaded/*.blon {initial_mesh_path}/unloaded/.")

    if os.path.isfile(f"{initial_mesh}.pts"):
        os.system(f"rm {initial_mesh}.pts")
    if os.path.isfile(f"{initial_mesh}.elem"):
        os.system(f"rm {initial_mesh}.elem")
    if os.path.isfile(f"{initial_mesh}.lon"):
        os.system(f"rm {initial_mesh}.lon")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate screenshots of the unloading geometries. It also setup the meshes for the cycle simulations.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default=os.path.join(os.environ.get("DATA_ROOT", ""), "simulations"),help="Path to the simulations folder.")
    parser.add_argument('--path2figure', type=str, required=False)
    parser.add_argument('--first_simulation', type=int, required=False, default=-1)
    parser.add_argument('--last_simulation', type=int, required=False, default=-1)
    parser.add_argument('--initial_mesh_path',type=str,required=True)
    parser.add_argument('--initial_mesh_name',type=str,required=True)
    parser.add_argument('--only_setup', action='store_true', help="Use this command if you just want to set up the folder architecture ready but not taking the screenshots.")

    args = parser.parse_args()

    main(args)
