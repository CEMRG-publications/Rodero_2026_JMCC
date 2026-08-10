#!/usr/bin/env python3
"""
Script to create a video visualizing the motion of a sliced mesh with displacement-based coloring.
The color scale ranges from black (no motion) to bright red (maximum displacement).
"""

import os
import sys
import argparse
import numpy as np
import pyvista as pv
import math
import matplotlib.pyplot as plt
import tempfile
import subprocess
import tqdm

# Import functions from your existing libraries
from common.mesh_io import read_mesh, carp_to_pyvista, read_IGB_file
from common.utils import file_exists

def rotation_matrix(u: np.ndarray, theta: float) -> np.ndarray:
    """Calculates a 3x3 rotation matrix for a given axis and angle."""
    u = u / np.linalg.norm(u)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    I = np.eye(3)
    ux = np.array([[0, -u[2], u[1]],
                   [u[2], 0, -u[0]],
                   [-u[1], u[0], 0]])
    R = cos_t * I + sin_t * ux + (1 - cos_t) * np.outer(u, u)
    return R

def align_mesh_points(points, cells, tags, lv_tag=1, mv_tag=7, tv_tag=8):
    """
    Align mesh points using tag-based landmark alignment.
    Returns aligned points.
    """
    def get_tag_points(tag):
        cell_ids = np.where(tags == tag)[0]
        vertex_ids = np.unique(cells[cell_ids].flatten())
        return vertex_ids

    vtx_lv = get_tag_points(lv_tag)
    vtx_mv = get_tag_points(mv_tag)
    vtx_tv = get_tag_points(tv_tag)

    cog_mv = np.mean(points[vtx_mv], axis=0)
    cog_tv = np.mean(points[vtx_tv], axis=0)

    # Apex estimation: farthest LV point from MV
    dd = np.linalg.norm(points[vtx_lv] - cog_mv, axis=1)
    idx_apex = vtx_lv[np.argmax(dd)]

    # Translation to origin
    cog = np.mean([cog_mv, cog_tv, points[idx_apex]], axis=0)
    points = points - cog

    # Rotate to align MV-TV axis with Y
    v0 = cog_tv - cog_mv
    v0 /= np.linalg.norm(v0)
    v1 = points[idx_apex] - cog_mv
    v1 /= np.linalg.norm(v1)
    n = np.cross(v0, v1)
    n /= np.linalg.norm(n)

    target_direction = np.array([0, -1, 0])
    rot_axis = np.cross(n, target_direction)
    rot_axis /= np.linalg.norm(rot_axis)
    angle = math.acos(np.clip(np.dot(n, target_direction), -1.0, 1.0))
    R = rotation_matrix(rot_axis, angle)
    points = points @ R.T

    # Rotate apex to Z- axis (down)
    cog_mv = np.mean(points[vtx_mv], axis=0)
    long_axis = points[idx_apex] - cog_mv
    long_axis /= np.linalg.norm(long_axis)

    target_y = np.array([0, 0, -1])
    angle_y = np.arccos(np.clip(np.dot(long_axis, target_y), -1.0, 1.0))
    cross_y = np.cross(long_axis, target_y)
    if np.linalg.norm(cross_y) > 1e-6:
        sign = np.sign(np.dot(cross_y, np.array([0, -1, 0])))
        angle_y *= sign
    R_y = rotation_matrix(target_direction, angle_y)
    points = points @ R_y.T

    return points

def create_displacement_colormap():
    """Create a custom colormap from black to bright red."""
    colors = ['#000000', '#330000', '#660000', '#990000', '#CC0000', '#FF0000', '#FF3333', '#FF6666']
    n_bins = 256
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('black_to_red', colors, N=n_bins)
    return cmap

def calculate_displacement_magnitude(original_points, deformed_points):
    """Calculate the magnitude of displacement for each point."""
    displacement_vectors = deformed_points - original_points
    displacement_magnitude = np.linalg.norm(displacement_vectors, axis=1)
    return displacement_magnitude

def create_sliced_motion_frame(pv_mesh, 
                              original_points,
                              deformed_points,
                              slice_origin=None,
                              slice_normal=(0, 1, 0),
                              fig_w=800,
                              fig_h=800,
                              camera_settings=None,
                              colormap=None,
                              global_max_disp=None):
    """
    Create a single frame showing the sliced mesh with displacement coloring.
    """
    # Update mesh points to deformed configuration
    pv_mesh.points = deformed_points
    
    # Calculate displacement magnitude for each point
    displacement_mag = calculate_displacement_magnitude(original_points, deformed_points)
    
    # Add displacement magnitude as point data
    pv_mesh.point_data['displacement_magnitude'] = displacement_mag
    
    # Default slice origin to mesh center if not provided
    if slice_origin is None:
        slice_origin = pv_mesh.center
    
    # Create slice
    sliced_mesh = pv_mesh.slice(normal=slice_normal, origin=slice_origin)
    
    # Set up plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'
    
    # Add sliced mesh with displacement coloring
    if colormap is None:
        colormap = create_displacement_colormap()
    
    # Use consistent color scaling across all frames
    disp_min = 0.0
    disp_max = global_max_disp if global_max_disp is not None else np.max(displacement_mag)
    
    plotter.add_mesh(sliced_mesh,
                     scalars='displacement_magnitude',
                     cmap=colormap,
                     clim=[disp_min, disp_max],
                     show_edges=False,
                     opacity=1.0)
    
    # Remove scalar bar (color bar) for cleaner visualization
    plotter.remove_scalar_bar()
    
    # Set camera if provided
    if camera_settings is not None:
        plotter.camera.position = camera_settings["position"]
        plotter.camera.focal_point = camera_settings["focal_point"]
        plotter.camera.up = camera_settings["up"]
        plotter.camera.view_angle = camera_settings["view_angle"]
    else:
        # Default camera view
        plotter.view_xz()
        plotter.camera.focal_point = sliced_mesh.center
        plotter.camera.zoom(1.2)
    
    # Take screenshot
    screenshot = plotter.screenshot(transparent_background=None,
                                   return_img=True,
                                   window_size=[fig_w, fig_h])
    plotter.close()
    
    return screenshot

def create_sliced_motion_video(mesh_file,
                              displacement_file,
                              output_video,
                              slice_origin=None,
                              slice_normal=(0, 1, 0),
                              fig_w=800,
                              fig_h=800,
                              fps=24,
                              camera_settings=None,
                              temp_dir=None):
    """
    Create an MP4 video showing sliced mesh motion with displacement coloring.
    """
    print("Loading mesh and displacement data...")
    
    # Read original mesh (before alignment)
    pts_orig, elem = read_mesh(mesh_file)
    
    # Read displacement data
    _, u = read_IGB_file(displacement_file)
    nt = u.shape[0]  # number of time steps
    
    print(f"Loaded mesh with {pts_orig.shape[0]} points and {nt} time steps")
    
    # Create PyVista mesh for tag extraction
    temp_mesh = carp_to_pyvista(pts_orig, elem)
    
    # Convert to VTK format temporarily to get tags
    temp_vtk_file = f"{mesh_file}_temp.vtk"
    os.system(f"meshtool convert -imsh {mesh_file} -omsh {temp_vtk_file} -ifmt=carp_bin -ofmt=vtk")
    
    # Read the VTK mesh to get element tags
    vtk_mesh = pv.read(temp_vtk_file)
    
    # Clean up temporary VTK file
    if os.path.exists(temp_vtk_file):
        os.remove(temp_vtk_file)
    
    # Extract element tags and connectivity
    if "elemTag" not in vtk_mesh.cell_data:
        raise KeyError("Mesh must contain 'elemTag' in cell_data.")
    
    tags = vtk_mesh.cell_data["elemTag"]
    cells = vtk_mesh.cells.reshape(-1, 5)[:, 1:]  # Tetrahedra: 4 points per cell
    
    # Align the original mesh points
    print("Aligning mesh...")
    pts_aligned = align_mesh_points(pts_orig.copy(), cells, tags, lv_tag=1, mv_tag=7, tv_tag=8)
    
    # Apply the same alignment transformation to all displacement time steps
    print("Aligning displacement data...")
    # Calculate the transformation that was applied to the original points
    translation = np.mean([pts_orig, pts_aligned], axis=0)  # This is approximate - we need the actual transformation
    
    # Since we don't have the exact transformation matrices, we need to apply the alignment to each timestep
    u_aligned = np.zeros_like(u)
    for t in range(nt):
        u_aligned[t, :, :] = align_mesh_points(u[t, :, :].copy(), cells, tags, lv_tag=1, mv_tag=7, tv_tag=8)
    
    # Create PyVista mesh with aligned points
    pv_mesh = carp_to_pyvista(pts_aligned, elem)
    
    # Add element tags to the PyVista mesh
    pv_mesh.cell_data["elemTag"] = tags
    
    # Create custom colormap
    colormap = create_displacement_colormap()
    
    # Calculate global displacement range for consistent coloring
    print("Calculating global displacement range...")
    all_displacements = []
    for t in range(nt):
        disp_mag = calculate_displacement_magnitude(pts_aligned, u_aligned[t, :, :])
        all_displacements.extend(disp_mag)
    
    global_max_disp = np.max(all_displacements)
    print(f"Global maximum displacement: {global_max_disp:.4f}")
    
    # Create temporary directory for frames
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Generating {nt} frames...")
    
    # Generate frames
    frame_files = []
    for t in tqdm.tqdm(range(nt)):
        frame_filename = os.path.join(temp_dir, f"frame_{t:06d}.png")
        frame_files.append(frame_filename)
        
        # Create frame
        screenshot = create_sliced_motion_frame(
            pv_mesh=pv_mesh.copy(),  # Use copy to avoid modifying original
            original_points=pts_aligned,
            deformed_points=u_aligned[t, :, :],
            slice_origin=slice_origin,
            slice_normal=slice_normal,
            fig_w=fig_w,
            fig_h=fig_h,
            camera_settings=camera_settings,
            colormap=colormap,
            global_max_disp=global_max_disp
        )
        
        # Save frame
        fig, ax = plt.subplots(figsize=(fig_w/100, fig_h/100), dpi=100)
        ax.imshow(screenshot)
        ax.axis('off')
        plt.savefig(frame_filename, bbox_inches='tight', dpi=100, pad_inches=0)
        plt.close(fig)
    
    print("Creating video with ffmpeg...")
    
    # Create video using ffmpeg
    frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-r', str(fps),  # input frame rate
        '-i', frame_pattern,  # input pattern
        '-c:v', 'libx264',  # video codec
        '-pix_fmt', 'yuv420p',  # pixel format for compatibility
        '-vf', f'scale={fig_w}:{fig_h}',  # scale filter
        output_video
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"Video saved successfully: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    for frame_file in frame_files:
        if os.path.exists(frame_file):
            os.remove(frame_file)
    
    if temp_dir != tempfile.gettempdir():
        try:
            os.rmdir(temp_dir)
        except:
            pass  # Directory might not be empty
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Create a video visualizing sliced mesh motion with displacement-based coloring.",
        epilog="The output video shows mesh motion with colors from black (no displacement) to bright red (maximum displacement)."
    )
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    
    parser.add_argument('--mesh_file', 
                        type=str, 
                        required=True, 
                        help="Path and basename of the mesh file (without extension, expects .bpts and .belem files)")
    
    parser.add_argument('--displacement_file', 
                        type=str, 
                        required=True, 
                        help="Path to the displacement file (.dynpt)")
    
    parser.add_argument('--output_video', 
                        type=str, 
                        required=True, 
                        help="Output video file path (.mp4)")
    
    parser.add_argument('--slice_origin', 
                        type=float, 
                        nargs=3, 
                        default=None,
                        help="Origin point for slicing plane (x y z). If not provided, uses mesh center.")
    
    parser.add_argument('--slice_normal', 
                        type=float, 
                        nargs=3, 
                        default=[0, 1, 0],
                        help="Normal vector for slicing plane (x y z)")
    
    parser.add_argument('--fig_width', 
                        type=int, 
                        default=800,
                        help="Video width in pixels")
    
    parser.add_argument('--fig_height', 
                        type=int, 
                        default=800,
                        help="Video height in pixels")
    
    parser.add_argument('--fps', 
                        type=int, 
                        default=24,
                        help="Frames per second for output video")
    
    parser.add_argument('--temp_dir', 
                        type=str, 
                        default=None,
                        help="Temporary directory for frame images (default: system temp)")
    
    args = parser.parse_args()
    
    # Check required files exist
    files_to_check = [
        f"{args.mesh_file}.bpts",
        f"{args.mesh_file}.belem",
        args.displacement_file
    ]
    
    try:
        file_exists(files_to_check=files_to_check)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create video
    success = create_sliced_motion_video(
        mesh_file=args.mesh_file,
        displacement_file=args.displacement_file,
        output_video=args.output_video,
        slice_origin=args.slice_origin,
        slice_normal=args.slice_normal,
        fig_w=args.fig_width,
        fig_h=args.fig_height,
        fps=args.fps,
        temp_dir=args.temp_dir
    )
    
    if success:
        print("Video creation completed successfully!")
    else:
        print("Video creation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()