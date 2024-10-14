import copy
import json
import math
import numpy as np
import os
import pyvista as pv
import vtk
import tqdm

def file_exists(files_to_check):
	
	if isinstance(files_to_check,list):
		for filepath in files_to_check:
			file_exists(files_to_check=filepath)
	
	elif isinstance(files_to_check, str):
		if not os.path.isfile(files_to_check):
			raise Exception(f"{files_to_check} not found.")
	else:
		raise Exception(f"Only strings can be checked. {files_to_check} is not a string.")
	

def read_elem(filename,el_type='Tt',tags=True):
	print('Reading '+filename+'...')

	if el_type=='Tt':
		if tags:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3,4,5))
		else:
			filtered_lines = []
			with open(filename, 'r') as infile:
				first_line = True
				for i in tqdm.tqdm(range(infile)):
					line = infile[i]
					if first_line:
						first_line = False
						continue
					else:
					# Split the line into columns
						columns = line.split()
						# Check if the number of columns is 6
						if len(columns) == 6:
							filtered_lines.append(columns[1:5])
						else:
							break
    
			# Convert the filtered lines to a numpy array
			# Skipping the first row (header) and using specific columns
			data = np.array(filtered_lines, dtype=int)
			return data
			# return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3,4))
	elif el_type=='Tr':
		if tags:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3,4))
		else:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3))
	elif el_type=='Ln':
		if tags:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3))
		else:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2))
	else:
		raise Exception('element type not recognised. Accepted: Tt, Tr, Ln')

def read_pts(filename):
    print(f'Reading: {filename}')
    return np.loadtxt(filename, dtype=float, skiprows=1)

def carp_to_pyvista(meshname):

	pts = np.loadtxt(meshname+'.pts', dtype=float, skiprows=1)
	elem = read_elem(meshname+'.elem',el_type='Tt',tags=False)

	tets = np.column_stack((np.ones((elem.shape[0],),dtype=int)*4,elem)).flatten()
	cell_type = np.ones((elem.shape[0],),dtype=int)*vtk.VTK_TETRA	

	plt_msh = pv.UnstructuredGrid(tets,cell_type,pts)

	return plt_msh

def numpy_hook(dct):
	for key, value in dct.items():
		if isinstance(value, list):
			value = np.array(value)
			dct[key] = value
	return dct

def load_json(filename):
	print('Reading '+filename+'...')

	dct = {}
	with open(filename, "r") as f:
		dct = json.load(f, object_hook=numpy_hook)
	return dct

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


def rotate_mesh(plt_msh,
				lv_tag=1,
				mv_tag=7,
				tv_tag=8,
				fibres=None,
				target_direction = [0,-1,0]):

	print(f"Aligning mesh to centre it in 0,0,0 and to have the posterior-anterior direction as {target_direction}...")

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

	target_direction = np.array(target_direction)

	axis_of_rotation = np.cross(n,target_direction)
	axis_of_rotation = axis_of_rotation/np.linalg.norm(axis_of_rotation)

	angle = math.acos(np.dot(n, target_direction))
	R = rotation_matrix(axis_of_rotation,angle)

	for i in range(pts.shape[0]):
		pts_transformed[i,:] = np.dot(R,pts_transformed[i,:])

	if fibres is not None:
		fibres_transformed = copy.deepcopy(fibres)
		for i in range(fibres.shape[0]):
			fibres_transformed[i,:] = np.dot(R,fibres[i,:])

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


	R_y = rotation_matrix(target_direction,angle_y)	

	for i in range(pts.shape[0]):
		pts_transformed[i,:] = np.dot(R_y,pts_transformed[i,:])

	if fibres is not None:
		for i in range(fibres.shape[0]):
			fibres_transformed[i,:] = np.dot(R_y,fibres_transformed[i,:])

	plt_msh.points = pts_transformed

	if fibres is not None:
		return plt_msh,fibres_transformed
	else:
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
