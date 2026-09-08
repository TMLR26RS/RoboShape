import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def look_at_matrix(eye, target, up):
    z_axis = normalize(eye - target) # Forward (towards viewer)
    x_axis = normalize(np.cross(up, z_axis)) # Right
    y_axis = np.cross(z_axis, x_axis) # Up
    
    #rotation
    R = np.eye(4)
    R[0, :3] = x_axis
    R[1, :3] = y_axis
    R[2, :3] = z_axis
    
    # translation 
    T = np.eye(4)
    T[:3, 3] = -eye
    
    # View matrix is R * T
    return R @ T

def project_points(points, eye, target, up, fov_deg=60, aspect_ratio=1.0, near=0.1, far=100.0):
   
    view_mat = look_at_matrix(eye, target, up)
    
    # Homogeneous coordinates
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Transform to Camera Space
    points_cam = (view_mat @ points_h.T).T
    
    # Perspective Projection Matrix (simplified for standard CV pinhole)
    # f = 1 / tan(fov/2)
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    
   
 
    valid_mask = points_cam[:, 2] < -near
    
    points_cam = points_cam[valid_mask]
    indices = np.where(valid_mask)[0]
    
    if len(points_cam) == 0:
        return np.array([]), np.array([])

    z = -points_cam[:, 2] # depth
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    
    u = f * (x / z) / aspect_ratio
    v = f * (y / z)
    
    return np.stack([u, v], axis=1), indices

def render_in_room(ply_path, output_image, eye=None, target=None):
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} not found.")
        return

    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)

    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    
    print(f"Scene Bounds: Min {min_bound}, Max {max_bound}")
    print(f"Scene Center: {center}")

    
    
    if eye is None:
        
        size = max_bound - min_bound
        eye_x = min_bound[0] + size[0] * 0.1
        eye_y = min_bound[1] + size[1] * 0.1
        eye_z = min_bound[2] + 1.5 # 1.5 units up (assuming meters)
        
        # If units are not meters, we might need to adjust. Scannet is usually meters.
        eye = np.array([eye_x, eye_y, eye_z])
    
    if target is None:
        target = center 
        
        target[2] = eye[2] 

    up = np.array([0, 0, 1]) # Z is up
    
    print(f"Camera Eye: {eye}")
    print(f"Camera Target: {target}")
    
    # Project points
    proj_points, valid_indices = project_points(points, eye, target, up, fov_deg=80)
    
    if len(proj_points) == 0:
        print("Warning: No points visible from this camera angle!")
        return

    valid_colors = colors[valid_indices]
    
    
    dists = np.linalg.norm(points[valid_indices] - eye, axis=1)
    sort_order = np.argsort(dists)[::-1] # Farthest first
    
    proj_points_sorted = proj_points[sort_order]
    colors_sorted = valid_colors[sort_order]
    
    plt.figure(figsize=(12, 9))
    plt.scatter(proj_points_sorted[:, 0], proj_points_sorted[:, 1], c=colors_sorted, s=1.0, edgecolors='none', alpha=0.9)
    plt.xlim(-1, 1) # NDC range
    plt.ylim(-1, 1)
    
    plt.gca().set_aspect('equal')
    
    plt.axis('off')
    
    plt.title("In-Room Perspective View")
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved in-room visualization to {output_image}")

def generate_trajectory(center, radius, height, steps=20):
    trajectory = []
    
    angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = height
        trajectory.append(np.array([x, y, z]))
    return trajectory

def generate_walking_trajectory(start, end, height, steps=20):
    trajectory = []
    
    frequency = 3.0 * (2 * np.pi) 
    amplitude = 0.05 
    
    for t in np.linspace(0, 1, steps):
        pos = start + (end - start) * t
        # Add sine wave to Z
        z_offset = np.sin(t * frequency) * amplitude
        pos[2] = height + z_offset
        trajectory.append(pos)
    return trajectory

def render_gif(ply_path, output_gif, eye=None, steps=30):
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} not found.")
        return

    print(f"Loading {ply_path} for GIF generation...")
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.05) # Coarser for speed
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    
    
    size = max_bound - min_bound
    
    start_point = min_bound + size * 0.1
    end_point = max_bound - size * 0.1
    
    if eye is not None:
        start_point[:2] = eye[:2]
        
        vec_to_center = center[:2] - eye[:2]
        end_point[:2] = center[:2] + vec_to_center # Continue through center
        height = eye[2]
    else:
        height = min_bound[2] + 1.6 # 1.6m eye level
        
    print(f"Generating walking trajectory from {start_point} to {end_point} at height {height:.2f}")
    eyes = generate_walking_trajectory(start_point, end_point, height, steps)
    
    target = end_point 
    
    up = np.array([0, 0, 1])

    frames = []
    
    
    
    import io
    from PIL import Image
    
    for i, current_eye in enumerate(eyes):
        print(f"Rendering frame {i+1}/{steps}...")
        
        proj_points, valid_indices = project_points(points, current_eye, target, up, fov_deg=90)
        
        if len(proj_points) == 0:
            continue

        valid_colors = colors[valid_indices]
        
        # Sort
        dists = np.linalg.norm(points[valid_indices] - current_eye, axis=1)
        sort_order = np.argsort(dists)[::-1]
        
        proj_points_sorted = proj_points[sort_order]
        colors_sorted = valid_colors[sort_order]
        
        fig = plt.figure(figsize=(8, 6)) # Smaller for GIF
        plt.scatter(proj_points_sorted[:, 0], proj_points_sorted[:, 1], c=colors_sorted, s=2.0, edgecolors='none', alpha=0.9)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.title(f"Walking... Frame {i+1}")
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))
        
    if frames:
        print(f"Saving GIF to {output_gif}...")
        # Save as GIF
        frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
        print("Done.")
    else:
        print("No frames generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, default="inference_result.ply", help="Path to .ply file")
    parser.add_argument("--output", type=str, default="inference_preview_in_room.png", help="Output filename (or .gif)")
    parser.add_argument("--eye", type=float, nargs=3, default=None, help="Camera eye position (x y z)")
    parser.add_argument("--target", type=float, nargs=3, default=None, help="Camera target position (x y z)")
    parser.add_argument("--make_gif", action='store_true', help="Generate a walking GIF instead of a static image")
    
    args = parser.parse_args()
    
    # Convert lists to numpy arrays if provided
    eye = np.array(args.eye) if args.eye else None
    target = np.array(args.target) if args.target else None
    
    if args.make_gif:
        output_gif = args.output.replace('.png', '.gif') if args.output.endswith('.png') else args.output
        if not output_gif.endswith('.gif'): output_gif += '.gif'
        render_gif(args.ply_path, output_gif, eye)
    else:
        render_in_room(args.ply_path, args.output, eye, target)
