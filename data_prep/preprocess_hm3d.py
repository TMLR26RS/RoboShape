"""
preprocess_hm3d.py
==================
Preprocess HM3D-Semantics scenes into the same .npy format produced by
preprocess_scannet.py so that Sonata inference can be run on both datasets
with the same loader.

Output per scene (mirrors ScanNet preprocessing):
    <output_root>/<split>/<scene_folder>/
        coord.npy        (N, 3) float32  -- vertex XYZ
        color.npy        (N, 3) float32  -- vertex RGB in [0, 255]
        normal.npy       (N, 3) float32  -- vertex normals
        segment20.npy    (N,)   int64    -- class index 0-19, 255 = unlabeled
        scene_type.txt                   -- always "hm3d" (no type metadata)

HM3D file layout (per scene folder):
    <scene_id>.basis.glb      -- geometry mesh (color = surface color)
    <scene_id>.semantic.glb   -- mesh whose vertex colors encode object IDs
    <scene_id>.semantic.txt   -- maps object_id → (hex_color, category_str, room_id)
        format:  <obj_id>,<RRGGBB>,"<label>",<room_id>

Label mapping strategy
----------------------
HM3D uses free-text category strings.  We map them to the same 20 classes
used in ScanNet so that the Sonata segmentation head (trained on ScanNet-20)
can be evaluated directly.

  HM3D_TO_SCANNET20:  dict[str → int]  (0-indexed, matches CLASS_LABELS_20)
  CLASS_LABELS_20 = [
      "wall","floor","cabinet","bed","chair","sofa","table","door","window",
      "bookshelf","picture","counter","desk","curtain","refrigerator",
      "shower curtain","toilet","sink","bathtub","otherfurniture"
  ]
  Anything not in the map → 255 (ignore / unlabeled)
"""

import os
import re
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


# ---------------------------------------------------------------------------
# ScanNet-20 target ontology  (same as inference_manual.py / preprocess_scannet.py)
# ---------------------------------------------------------------------------
CLASS_LABELS_20 = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
]
# index  0       1        2          3      4        5       6       7
#        8          9            10        11        12      13
#        14              15                  16       17      18
#        19

# ---------------------------------------------------------------------------
# HM3D free-text → ScanNet-20 index mapping
# Covers the most common categories seen across HM3D-Semantics scenes.
# Unknown / structurally ambiguous labels fall through to 255.
# ---------------------------------------------------------------------------
HM3D_TO_SCANNET20 = {
    # --- wall / structural surfaces ---
    "wall": 0,
    "wall /outside": 0,
    "kitchen wall": 0,
    "fireplace wall": 0,
    "shower wall": 0,
    "recessed wall": 0,
    "ceiling": 0,           # no ceiling class; map to wall (structural)
    "ceiling door": 0,
    "roof": 0,

    # --- floor ---
    "floor": 1,
    "floor /outside": 1,
    "shower floor": 1,
    "doormat": 1,
    "rug": 1,
    "carpet": 1,
    "platform": 1,
    "stair step": 1,

    # --- cabinet / storage ---
    "cabinet": 2,
    "kitchen cabinet": 2,
    "wardrobe": 2,
    "dresser": 2,
    "bathroom cabinet": 2,
    "bathroom shelf": 2,
    "kitchen shelf": 2,
    "shelf": 2,
    "wall cabinet": 2,
    "rack": 2,

    # --- bed ---
    "bed": 3,
    "bed stand": 3,

    # --- chair ---
    "chair": 4,
    "armchair": 4,
    "sofa chair": 4,
    "folding chair": 4,
    "seat": 4,
    "pouffe": 4,
    "bar": 4,               # bar stool area

    # --- sofa ---
    "sofa": 5,
    "couch": 5,
    "l-shaped sofa": 5,
    "pillow": 5,
    "blanket": 5,

    # --- table ---
    "table": 6,
    "kitchen top": 6,
    "stand": 6,
    "table stand": 6,
    "tray": 6,
    "bar table": 6,

    # --- door ---
    "door": 7,
    "door frame": 7,
    "door/window": 7,
    "door/window frame": 7,

    # --- window ---
    "window": 8,
    "window frame": 8,
    "window shutter": 8,
    "window shutters": 8,

    # --- bookshelf ---
    "bookshelf": 9,
    "book": 9,

    # --- picture ---
    "picture": 10,
    "mirror": 10,
    "decoration": 10,

    # --- counter ---
    "counter": 11,

    # --- desk ---
    "desk": 12,

    # --- curtain ---
    "curtain": 13,
    "curtain rail": 13,
    "shower door frame": 13,  # closest match

    # --- refrigerator ---
    "refrigerator": 14,
    "freezer": 14,

    # --- shower curtain ---
    "shower hose": 15,
    "shower hose/head": 15,

    # --- toilet ---
    "toilet": 16,
    "toilet brush": 16,

    # --- sink ---
    "sink": 17,
    "tap": 17,

    # --- bathtub ---
    "bathtub": 18,

    # --- otherfurniture (catch-all for identifiable furniture / appliances) ---
    "lamp": 19,
    "wall lamp": 19,
    "heater": 19,
    "fireplace": 19,
    "tv": 19,
    "television": 19,
    "microwave": 19,
    "oven and stove": 19,
    "range hood": 19,
    "coffee machine": 19,
    "washing machine": 19,
    "washer-dryer": 19,
    "dryer": 19,
    "kitchen appliance": 19,
    "ventilation": 19,
    "ventilation hood": 19,
    "plant": 19,
    "flowerpot": 19,
    "vase": 19,
    "ladder": 19,
    "ironing board": 19,
    "vacuum cleaner": 19,
    "board": 19,
    "grill": 19,
}

# Normalise keys once (lowercase, strip)
HM3D_TO_SCANNET20 = {k.strip().lower(): v for k, v in HM3D_TO_SCANNET20.items()}


# ---------------------------------------------------------------------------
# Utility: read .semantic.txt → {packed_rgb_int: label_str}
# ---------------------------------------------------------------------------
def parse_semantic_txt(txt_path: str) -> dict:
    """
    Returns a dict mapping packed RGB integer (R<<16 | G<<8 | B) → label string.
    Line format:  <obj_id>,<RRGGBB>,"<label>",<room_id>
    """
    color_to_label = {}
    pattern = re.compile(r'^\d+,([0-9A-Fa-f]{6}),"([^"]*)",\d+')
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if not m:
                continue
            hex_color = m.group(1).upper()
            label     = m.group(2).strip().lower()
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            packed = (r << 16) | (g << 8) | b
            color_to_label[packed] = label
    return color_to_label


# ---------------------------------------------------------------------------
# Utility: extract per-vertex labels from semantic GLB
# ---------------------------------------------------------------------------
def get_vertex_labels(semantic_glb_path: str,
                      color_to_label: dict,
                      n_vertices: int) -> np.ndarray:
    """
    Loads the semantic GLB, reads per-vertex colours (which encode object ID
    via the colour values in .semantic.txt), and returns a per-vertex label
    array of shape (N,) int64, values 0-19 or 255 (ignore).

    The semantic mesh and basis mesh have the same vertex count and ordering.
    """
    seg = np.full(n_vertices, 255, dtype=np.int64)

    sem_mesh = o3d.io.read_triangle_mesh(semantic_glb_path)
    if not sem_mesh.has_vertex_colors():
        return seg   # no colour info → all unlabeled

    vc = np.asarray(sem_mesh.vertex_colors)   # float64 in [0, 1]
    # Round to nearest uint8 to avoid float precision drift
    r = np.clip(np.round(vc[:, 0] * 255), 0, 255).astype(np.int32)
    g = np.clip(np.round(vc[:, 1] * 255), 0, 255).astype(np.int32)
    b = np.clip(np.round(vc[:, 2] * 255), 0, 255).astype(np.int32)
    packed = (r << 16) | (g << 8) | b   # shape (M,)

    # semantic mesh may have different vertex count (it's a separate file)
    # if counts match, use directly; else fall back to vertex-count mismatch handling
    if len(packed) != n_vertices:
        # Best-effort: try to use what we have up to the min length
        min_n = min(len(packed), n_vertices)
        packed_use = packed[:min_n]
        for i, pk in enumerate(packed_use):
            label_str = color_to_label.get(int(pk), None)
            if label_str is not None:
                seg[i] = HM3D_TO_SCANNET20.get(label_str, 255)
        return seg

    for i, pk in enumerate(packed):
        label_str = color_to_label.get(int(pk), None)
        if label_str is not None:
            seg[i] = HM3D_TO_SCANNET20.get(label_str, 255)

    return seg


# ---------------------------------------------------------------------------
# Utility: compute vertex normals from a triangle mesh
# ---------------------------------------------------------------------------
def compute_normals(mesh: "o3d.geometry.TriangleMesh") -> np.ndarray:
    if len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        return np.asarray(mesh.vertex_normals, dtype=np.float32)
    return np.zeros((len(mesh.vertices), 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------
def process_scene(scene_folder: str, hm3d_root: str, output_root: str, split: str):
    """
    scene_folder : e.g. "00023-zepmXAdrpjR"
    hm3d_root    : e.g. ""
    split        : "train" | "val" | "minival"
    """
    scene_dir = Path(hm3d_root) / split / scene_folder
    out_dir   = Path(output_root) / split / scene_folder

    # Scene ID is the part after the dash
    scene_id  = scene_folder.split("-", 1)[1] if "-" in scene_folder else scene_folder

    basis_glb    = scene_dir / f"{scene_id}.basis.glb"
    semantic_glb = scene_dir / f"{scene_id}.semantic.glb"
    semantic_txt = scene_dir / f"{scene_id}.semantic.txt"

    if not basis_glb.exists():
        return scene_folder, False, f"Missing basis GLB: {basis_glb}"

    # --- Load geometry mesh ---
    try:
        mesh = o3d.io.read_triangle_mesh(str(basis_glb))
    except Exception as e:
        return scene_folder, False, f"Error reading basis GLB: {e}"

    vertices  = np.asarray(mesh.vertices, dtype=np.float32)
    n_verts   = len(vertices)

    if n_verts == 0:
        return scene_folder, False, "Empty mesh (0 vertices)"

    # Per-vertex colour from basis mesh ([0,1] → [0,255])
    if mesh.has_vertex_colors():
        color = (np.asarray(mesh.vertex_colors, dtype=np.float32) * 255.0)
    else:
        color = np.zeros((n_verts, 3), dtype=np.float32)

    # Per-vertex normals
    normal = compute_normals(mesh)

    # --- Semantic labels ---
    seg20 = None
    if semantic_glb.exists() and semantic_txt.exists():
        try:
            color_to_label = parse_semantic_txt(str(semantic_txt))
            seg20 = get_vertex_labels(str(semantic_glb), color_to_label, n_verts)
        except Exception as e:
            seg20 = None
            print(f"  [warn] {scene_folder}: label extraction failed: {e}")

    # --- Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coord.npy",  vertices)
    np.save(out_dir / "color.npy",  color)
    np.save(out_dir / "normal.npy", normal)
    (out_dir / "scene_type.txt").write_text("hm3d")

    if seg20 is not None:
        np.save(out_dir / "segment20.npy", seg20)
        n_labeled = int((seg20 < 255).sum())
        label_info = f"labels={n_labeled}/{n_verts}"
    else:
        label_info = "no labels"

    return scene_folder, True, f"OK vertices={n_verts} {label_info}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HM3D-Semantics scenes into .npy format for Sonata."
    )
    parser.add_argument(
        "--hm3d_root",
        type=str,
        default="",
        help="Root of hm3d-bingCS dataset (contains train/ val/ minival/)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Output directory (will be created if needed)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "minival"],
        help="Which splits to process",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Specific scene folder names to process (default: all in each split)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes",
    )
    args = parser.parse_args()

    if not HAS_O3D:
        raise RuntimeError("open3d is required. Install with: pip install open3d")

    hm3d_root = Path(args.hm3d_root)

    all_tasks = []   # list of (scene_folder, split)
    for split in args.splits:
        split_dir = hm3d_root / split
        if not split_dir.exists():
            print(f"  [skip] split dir not found: {split_dir}")
            continue
        if args.scenes:
            folders = args.scenes
        else:
            folders = sorted(
                d.name for d in split_dir.iterdir()
                if d.is_dir()
            )
        for folder in folders:
            all_tasks.append((folder, split))

    print(f"Processing {len(all_tasks)} scenes → {args.output_root}")

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    process_scene, folder, str(hm3d_root), args.output_root, split
                ): (folder, split)
                for folder, split in all_tasks
            }
            for future in as_completed(futures):
                folder, ok, msg = future.result()
                status = "✓" if ok else "✗"
                print(f"  [{status}] {folder}: {msg}")
    else:
        for folder, split in all_tasks:
            _, ok, msg = process_scene(folder, str(hm3d_root), args.output_root, split)
            status = "✓" if ok else "✗"
            print(f"  [{status}] {folder}: {msg}")

    print("Done.")


if __name__ == "__main__":
    main()
