"""Core mesh-to-image operations used in multi-view preprocessing.

The input mesh must already be aligned and centered. Batch processing, mesh
alignment, patient records and file-system orchestration are not included.
"""

import copy

import numpy as np
import open3d as o3d


DEFAULT_RESOLUTION = (1500, 1080)
CAMERA_ZOOM = 0.55
BACKGROUND_COLOR = np.array([0.0, 0.0, 0.0])
LABEL_MAX = 48


def create_gray_mesh_from_labels(combined_mesh, combined_labels):
    """Map vertex labels to grayscale values using vectorized operations."""
    if combined_labels is None:
        return None
    
    gray_mesh = copy.deepcopy(combined_mesh)
    
    # Vectorized grayscale mapping.
    # Normalize labels 0-48 to the range 0-1.
    gray_values = np.clip(combined_labels.astype(np.float64) / LABEL_MAX, 0, 1)
    gray_colors = np.column_stack([gray_values, gray_values, gray_values])
    
    gray_mesh.vertex_colors = o3d.utility.Vector3dVector(gray_colors)
    return gray_mesh


def setup_camera_and_render(mesh, x_angle_degrees, y_angle_degrees, 
                            resolution=DEFAULT_RESOLUTION, for_grayscale=True):
    """Configure the camera and render a mesh at the requested X/Y angles."""
    # Create the visualizer.
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=resolution[0], height=resolution[1], visible=False)

    # Apply X-axis rotation to the mesh for augmentation.
    rotated_mesh = copy.deepcopy(mesh)
    if x_angle_degrees != 0:
        x_rotation = rotated_mesh.get_rotation_matrix_from_xyz((np.radians(x_angle_degrees), 0, 0))
        center = rotated_mesh.get_center()
        rotated_mesh.translate(-center)
        rotated_mesh.rotate(x_rotation, center=(0, 0, 0))
        rotated_mesh.translate(center)

    vis.add_geometry(rotated_mesh)

    # Get rendering options.
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.mesh_show_wireframe = False
    render_option.background_color = BACKGROUND_COLOR

    # Disable lighting for grayscale rendering.
    if for_grayscale:
        render_option.light_on = False

    # Set camera parameters.
    view_control = vis.get_view_control()
    y_angle_rad = np.radians(y_angle_degrees)

    view_control.reset_camera_local_rotate()
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 1, 0])
    view_control.set_front([
        -np.sin(y_angle_rad),
        0,
        np.cos(y_angle_rad)
    ])
    view_control.set_zoom(CAMERA_ZOOM)

    # Render and capture the image.
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    image_np = np.asarray(image)
    return (image_np * 255).astype(np.uint8)
