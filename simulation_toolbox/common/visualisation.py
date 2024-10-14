import numpy as np
import pyvista as pv


def print_screenshot_video(plt_msh,
						   binary_vector,
						   screenshot_name,
						   camera_settings=None,
						   title=None,
						   fig_w=1200,
						   fig_h=1200,
						   inactive_color="gray",
						   active_color="darkred",
						   view="anterior",
						   opacity=1.0):

	plotter = pv.Plotter(off_screen=True)
	plotter.background_color = 'white'

	plt_msh.point_data["at"] = binary_vector

	msh = plotter.add_mesh(plt_msh,opacity=opacity,
						   scalars="at",
						   cmap=[inactive_color,active_color],
						   clim=np.array([0.,1.]))

	plotter.remove_scalar_bar()

	if camera_settings is not None:
		plotter.camera.azimuth = camera_settings[view]["azimuth"]
		plotter.camera.elevation = camera_settings[view]["elevation"]
	if title is not None:
		plotter.add_title(title,
						font_size=12,
						font="arial",
						color="black")
	print("Printing...")
	plotter.screenshot(filename=screenshot_name, 
					   transparent_background=None, 
					   return_img=True,
					   window_size=[fig_w,fig_h])
	print("Printed")
	plotter.close()



def print_activation_times(plt_msh,
                           act_times,            # Array of activation times for each point
                           screenshot_name,
                           title=None,
                           fig_w=1200,
                           fig_h=1200,
                           inactive_color="gray",
                           blue_gradient="Blues",    # Colormap for tags 1 and 2
                           orange_gradient="Oranges", # Colormap for tags 3 and 4
                           opacity=1.0,
                           camera_roll_increment=0,
                           camera_azimuth_increment=0,
                           camera_elevation_increment=0):
    
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'

    # Extract tag IDs from the mesh cell data
    tag_array = plt_msh.cell_data["ID"]
    plt_msh.point_data["Activation"] = act_times  # Assign activation times to point data for gradient coloring

    # Create masks for the different tag groups
    inactive_mask = tag_array == 0
    blue_mask = np.isin(tag_array, [1, 2])
    orange_mask = np.isin(tag_array, [3, 4])

    # Plot inactive cells with solid inactive color
    plotter.add_mesh(plt_msh.extract_cells(inactive_mask), 
                     color=inactive_color, 
                     show_edges=False, 
                     opacity=opacity)

    # Plot tags 1 and 2 with blue gradient using activation times
    plotter.add_mesh(plt_msh.extract_cells(blue_mask), 
                     scalars="Activation", 
                     cmap=blue_gradient, 
                     clim=[np.min(act_times), np.max(act_times)], 
                     show_edges=False, 
                     opacity=opacity)

    # Plot tags 3 and 4 with orange gradient using activation times
    plotter.add_mesh(plt_msh.extract_cells(orange_mask), 
                     scalars="Activation", 
                     cmap=orange_gradient, 
                     clim=[np.min(act_times), np.max(act_times)], 
                     show_edges=False, 
                     opacity=opacity)

    # Center and camera adjustments
    submesh_cells = np.unique(plt_msh.cells, axis=0)
    submesh_points = plt_msh.points[submesh_cells, :]
    center = np.mean(submesh_points, axis=0)

    plotter.view_xz()
    plotter.camera.focal_point = center
    plotter.camera.roll += camera_roll_increment
    plotter.camera.azimuth += camera_azimuth_increment
    plotter.camera.elevation += camera_elevation_increment

    if title is not None:
        plotter.add_title(title, font_size=12, font="arial", color="black")

    print("Printing...")
    plotter.screenshot(filename=screenshot_name, 
                       transparent_background=None, 
                       return_img=True, 
                       window_size=[fig_w, fig_h])
    print("Printed")
    plotter.close()