

import os
import json
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from plyfile import PlyData
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False



VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)


NYU40_TO_20 = {nyu: idx for idx, nyu in enumerate(VALID_CLASS_IDS_20)}


def compute_normals_from_mesh(path):
   
    if not HAS_O3D:
        return None          # caller must handle None
    try:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) == 0:
            return None      # no face data – can't compute normals
        mesh.compute_vertex_normals()
        return np.asarray(mesh.vertex_normals, dtype=np.float32)
    except Exception:
        return None


def read_ply_open3d(path):
   
    pcd = o3d.io.read_point_cloud(path)
    coord = np.asarray(pcd.points, dtype=np.float32)
    color = np.asarray(pcd.colors, dtype=np.float32) * 255.0  # [0,1] → [0,255]
    if pcd.has_normals():
        normal = np.asarray(pcd.normals, dtype=np.float32)
    else:
        # The PLY is a mesh – compute vertex normals from face geometry
        normal = compute_normals_from_mesh(path)
        if normal is None:
            normal = np.zeros_like(coord)
    return coord, color, normal


def read_ply_plyfile(path):
   
    plydata = PlyData.read(path)
    v = plydata['vertex']
    coord = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    # color: uint8 in PLY, keep as float32 [0,255]
    if 'red' in v.data.dtype.names:
        color = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.float32)
    else:
        color = np.zeros_like(coord)
    # normals: prefer stored values, but ScanNet mesh PLYs usually don't have them
    if 'nx' in v.data.dtype.names:
        normal = np.stack([v['nx'], v['ny'], v['nz']], axis=1).astype(np.float32)
    else:
        normal = None    # will be computed from mesh faces below

    # If normals are absent or identically zero, compute from mesh faces
    if normal is None or not np.any(normal):
        computed = compute_normals_from_mesh(path)
        if computed is not None and len(computed) == len(coord):
            normal = computed
        else:
            normal = np.zeros_like(coord)

    return coord, color, normal


def read_labels_ply(labels_ply_path):
   
    plydata = PlyData.read(labels_ply_path)
    v = plydata['vertex']
    if 'label' in v.data.dtype.names:
        labels = v['label'].astype(np.int32)
    else:
        labels = np.zeros(len(v['x']), dtype=np.int32)
    return labels


def nyu40_to_segment20(raw_labels):
   
    seg20 = np.full(len(raw_labels), 255, dtype=np.int64)
    for nyu, idx in NYU40_TO_20.items():
        seg20[raw_labels == nyu] = idx
    return seg20


def read_scene_meta(scene_dir, scene_id):
   
    txt_path = os.path.join(scene_dir, f"{scene_id}.txt")
    scene_type = "unknown"
    axis_align = None
    if not os.path.exists(txt_path):
        return scene_type, axis_align
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("sceneType"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    scene_type = parts[1].strip()
            elif line.startswith("axisAlignment"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    vals = list(map(float, parts[1].strip().split()))
                    axis_align = np.array(vals, dtype=np.float32).reshape(4, 4)
    return scene_type, axis_align


def process_scene(scene_id, scannet_root, output_root):
   
    scene_dir = os.path.join(scannet_root, scene_id)
    out_dir = os.path.join(output_root, scene_id)

    mesh_ply = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")
    labels_ply = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.labels.ply")

    if not os.path.exists(mesh_ply):
        return scene_id, False, f"Missing mesh PLY: {mesh_ply}"

    
    try:
        if HAS_PLYFILE:
            coord, color, normal = read_ply_plyfile(mesh_ply)
        elif HAS_O3D:
            coord, color, normal = read_ply_open3d(mesh_ply)
        else:
            return scene_id, False, "Neither plyfile nor open3d is available"
    except Exception as e:
        return scene_id, False, f"Error reading mesh PLY: {e}"

  
    scene_type, axis_align = read_scene_meta(scene_dir, scene_id)

    if axis_align is not None:
        ones = np.ones((coord.shape[0], 1), dtype=np.float32)
        coord_h = np.concatenate([coord, ones], axis=1)  # (N, 4)
        coord = (axis_align @ coord_h.T).T[:, :3].astype(np.float32)
        if np.any(normal != 0):
            normal = (axis_align[:3, :3] @ normal.T).T.astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "coord.npy"), coord)
    np.save(os.path.join(out_dir, "color.npy"), color)
    np.save(os.path.join(out_dir, "normal.npy"), normal)

    
    with open(os.path.join(out_dir, "scene_type.txt"), "w") as f:
        f.write(scene_type)

    
    if os.path.exists(labels_ply):
        try:
            if HAS_PLYFILE:
                raw_labels = read_labels_ply(labels_ply)
            else:
                
                raw_labels = None
        except Exception as e:
            raw_labels = None

        if raw_labels is not None:
            seg20 = nyu40_to_segment20(raw_labels)
            np.save(os.path.join(out_dir, "segment20.npy"), seg20)

    return scene_id, True, f"OK (sceneType={scene_type})"


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ScanNet scenes into npy format for Sonata inference."
    )
    parser.add_argument(
        "--scannet_root",
        type=str,
        default="",
        help="Path to scannet_all/scans/ directory",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Output directory (will be created if needed)",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Specific scene IDs to process (default: all found in scannet_root)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes",
    )
    args = parser.parse_args()

    if not HAS_PLYFILE and not HAS_O3D:
        raise RuntimeError(
            "Install at least one of: plyfile (`pip install plyfile`) or "
            "open3d (`pip install open3d`)"
        )

    
    if args.scenes:
        scene_ids = args.scenes
    else:
        scene_ids = sorted(
            d for d in os.listdir(args.scannet_root)
            if os.path.isdir(os.path.join(args.scannet_root, d))
            and d.startswith("scene")
        )

    print(f"Processing {len(scene_ids)} scenes → {args.output_root}")
    print(f"Using {'plyfile' if HAS_PLYFILE else 'open3d'} for PLY reading")

    os.makedirs(args.output_root, exist_ok=True)

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_scene, sid, args.scannet_root, args.output_root): sid
                for sid in scene_ids
            }
            for future in as_completed(futures):
                sid, ok, msg = future.result()
                status = "✓" if ok else "✗"
                print(f"  [{status}] {sid}: {msg}")
    else:
        for sid in scene_ids:
            _, ok, msg = process_scene(sid, args.scannet_root, args.output_root)
            status = "✓" if ok else "✗"
            print(f"  [{status}] {sid}: {msg}")

    print("Done.")


if __name__ == "__main__":
    main()
