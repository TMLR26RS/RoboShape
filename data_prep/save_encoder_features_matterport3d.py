"""
save_encoder_features_matterport3d.py

For every scene in --data_dir:

  Task 1 – Encoder features
      Accumulate all per-scene coarsest-voxel features into one big array
      and save as:  <output_dir>/all_features.npy   shape (N_total, 512)

  Task 2 – Wall label tensor  (voxelwise_pred logic, Solution 2)
      For each coarsest voxel: 1 if predicted label == "wall" (idx 0), else 0
      Saved as:  <output_dir>/wall_labels.npy        shape (N_total,)  int32

  Task 3 – Bed scene-type tensor 
      For each coarsest voxel: 1 if its scene contains a "bed" in ground truth labels, else 0
      Saved as:  <output_dir>/bed_scene_labels.npy   shape (N_total,)  int32

All three arrays are aligned row-by-row (same voxel order).
A JSON log of included scenes is written to --json_log.
"""

import numpy as np
import sonata
import torch
import torch.nn as nn
import os
import glob
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


_SPLIT_SEED = 42
_TEST_SIZE  = 0.20   # 80 % train
_VAL_FRAC   = 0.50   # 50 % of 20 % temp → 10 % val, 10 % test

try:
    import flash_attn
except ImportError:
    flash_attn = None

CLASS_LABELS_20 = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "desk",
    "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)
WALL_IDX = CLASS_LABELS_20.index("wall")   # == 0
BED_IDX = CLASS_LABELS_20.index("bed")     # == 3


# ── Seg head ──────────────────────────────────────────────────────────────────
class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


# ── Data loader ───────────────────────────────────────────────────────────────
def load_custom_data(scene_path):
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene path not found: {scene_path}")
    point = {}
    point["coord"]  = np.load(os.path.join(scene_path, "coord.npy"))
    point["color"]  = np.load(os.path.join(scene_path, "color.npy"))
    point["normal"] = np.load(os.path.join(scene_path, "normal.npy"))
    
    seg20 = os.path.join(scene_path, "segment20.npy")
    seg = os.path.join(scene_path, "segment.npy")
    if os.path.exists(seg20):
        point["segment"] = np.load(seg20)
    elif os.path.exists(seg):
        point["segment"] = np.load(seg)
    return point


# ── Per-scene processing ──────────────────────────────────────────────────────
def process_scene(scene_path, model, seg_head, transform, softmax, min_voxels, max_voxels, max_input_points, non_bed_keep_ratio):
    """
    Returns (feat_np, wall_mask, is_bed_scene) or None on failure.
      feat_np      : (N_coarsest, 512)  float32  — encoder features
      wall_mask    : (N_coarsest,)      int32    — 1 where voxel pred == wall
      is_bed_scene : bool               — True if bed is present in GT
    """
    # ── Load & transform ──────────────────────────────────────────────────────
    try:
        raw = load_custom_data(scene_path)
    except Exception as e:
        print(f"  [SKIP load] {e}")
        return None

    # ── Task 3: bed scene type from GT ────────────────────────────────────────
    is_bed_scene = False
    if "segment" in raw:
        is_bed_scene = bool(np.any(raw["segment"] == BED_IDX))

    # To balance the ratio to ~25%, randomly drop non-bedroom scenes
    if not is_bed_scene and np.random.rand() > non_bed_keep_ratio:
        return None

    pt = transform(raw)
    for k, v in pt.items():
        if isinstance(v, torch.Tensor):
            pt[k] = v.cuda(non_blocking=True)

    try:
        with torch.inference_mode():
            # ── Check input size to prevent OOM ───────────────────────────────
            num_input_points = pt["coord"].shape[0] if "coord" in pt else 0
            if num_input_points > max_input_points:
                print(f"  [SKIP infer] Scene too large: {num_input_points:,} points > {max_input_points:,} limit.")
                return None

            # ── Encoder forward ───────────────────────────────────────────────
            out = model(pt)

            # ── Task 1: encoder features at bottleneck ────────────────────────
            feat_bottleneck = out.feat.cpu().numpy()   # (N_coarsest, 512)
            N_coarsest = feat_bottleneck.shape[0]

            # Filter by voxel count
            if not (min_voxels <= N_coarsest <= max_voxels):
                return None

            # ── Unpool to get finest-level features + inverse map ─────────────
            inverses = []
            while "pooling_parent" in out.keys():
                parent  = out.pop("pooling_parent")
                inverse = out.pop("pooling_inverse")
                inverses.append(inverse)
                parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
                out = parent

            feat_finest = out.feat          # (N_finest, C)
            N_final     = feat_finest.shape[0]

            # Compose inverses: finest point → coarsest voxel index
            idx = torch.arange(N_final, device=feat_finest.device)
            for inv in reversed(inverses):
                idx = inv[idx]
            idx_np = idx.cpu().numpy()     # (N_final,)

            # ── Task 2: voxel-level label via Solution 2 ──────────────────────
            probs = softmax(seg_head(feat_finest)).cpu().numpy()   # (N_final, 20)
            voxel_prob_sum = np.zeros((N_coarsest, 20), dtype=np.float32)
            np.add.at(voxel_prob_sum, idx_np, probs)

            voxel_pred = np.argmax(voxel_prob_sum, axis=1)         # (N_coarsest,)
            wall_mask  = (voxel_pred == WALL_IDX).astype(np.int32) # (N_coarsest,)

    except Exception as e:
        print(f"  [SKIP infer] {e}")
        return None

    return feat_bottleneck, wall_mask, is_bed_scene


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,
                        default="/path/to/dataset/matterport3d_processed/test",
                        help="Directory containing scene folders")
    parser.add_argument("--output_dir", type=str,
                        default="/path/to/dataset/matterport3d_tensors",
                        help="Directory to save output tensors")
    parser.add_argument("--json_log",   type=str,
                        default="/path/to/dataset/matterport3d_tensors/scene_log.json",
                        help="Path to JSON log of included scenes")
    parser.add_argument("--min_voxels", type=int, default=0,
                        help="Minimum number of coarsest voxels to include scene")
    parser.add_argument("--max_voxels", type=int, default=9999999,
                        help="Maximum number of coarsest voxels to include scene")
    parser.add_argument("--max_input_points", type=int, default=1500000,
                        help="Maximum number of input points (to avoid CUDA OOM). Default 1.5M.")
    parser.add_argument("--non_bed_keep_ratio", type=float, default=0.52,
                        help="Ratio of non-bed scenes to keep (to achieve ~25% positive ratio).")
    parser.add_argument("--save_test_labels", action="store_true",
                        help="Skip scene processing. Instead load the already-saved tensors, "
                             "extract the test-split rows , and save "
                             "matterport3d_test_private_label.npy + matterport3d_test_public_label.npy.")
    args = parser.parse_args()

    # ── Fast path: extract test-split labels from existing tensors ────────────
    if args.save_test_labels:
        tensor_dir = args.output_dir
        # Load the three tensors produced by the full pipeline
        
        #   train_label1_path → private  (matterport3D_private_label.npy)
        #   train_label2_path → public   (matterport3D_public_label.npy)
      
        private_label = np.load(os.path.join(tensor_dir, "matterport3D_private_label.npy"))
        public_label  = np.load(os.path.join(tensor_dir, "matterport3D_public_label.npy"))

        idx = np.arange(len(public_label))
        _, idx_temp = train_test_split(
            idx, test_size=_TEST_SIZE, random_state=_SPLIT_SEED, stratify=public_label
        )
        _, idx_test = train_test_split(
            idx_temp, test_size=_VAL_FRAC, random_state=_SPLIT_SEED, stratify=public_label[idx_temp]
        )

        priv_test   = private_label[idx_test]
        pub_test    = public_label[idx_test]

        os.makedirs(tensor_dir, exist_ok=True)
        priv_out = os.path.join(tensor_dir, "matterport3d_test_private_label.npy")
        pub_out  = os.path.join(tensor_dir, "matterport3d_test_public_label.npy")
        np.save(priv_out, priv_test)
        np.save(pub_out,  pub_test)

        print(f"Test split: {len(idx_test):,} rows (out of {len(public_label):,} total)")
        print(f"Saved private label → {priv_out}  shape={priv_test.shape}")
        print(f"Saved public  label → {pub_out}   shape={pub_test.shape}")
        print(f"  Bed-scene positives: {priv_test.sum():,} ({priv_test.mean()*100:.1f}%)")
        print(f"  Wall positives:      {pub_test.sum():,}  ({pub_test.mean()*100:.1f}%)")
        exit(0)

    os.makedirs(args.output_dir, exist_ok=True)
    sonata.utils.set_seed(24525867)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading Sonata encoder …")
    if flash_attn is not None:
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        model = sonata.load(
            "sonata", repo_id="facebook/sonata",
            custom_config=dict(enc_patch_size=[1024]*5, enable_flash=False)
        ).cuda()

    print("Loading seg head …")
    ckpt     = sonata.load("sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True)
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])
    model.eval(); seg_head.eval()

    transform = sonata.transform.default()
    softmax   = nn.Softmax(dim=-1)

    # Matterport3D directories are arbitrary IDs, not just "scene*"
    scene_paths = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                   if os.path.isdir(os.path.join(args.data_dir, d))]
    scene_paths = sorted(scene_paths)
    print(f"Found {len(scene_paths)} scenes in {args.data_dir}\n")

    # ── Accumulate across scenes ──────────────────────────────────────────────
    all_features  = []   # list of (N_i, 512) arrays
    all_wall      = []   # list of (N_i,) int32 arrays
    all_bed_scene = []   # list of (N_i,) int32 arrays
    scene_log     = []
    global_offset = 0    # running row index into the concatenated arrays

    for scene_path in tqdm(scene_paths, desc="Scenes"):
        scene_name = os.path.basename(scene_path)
        result = process_scene(scene_path, model, seg_head, transform, softmax, 
                               args.min_voxels, args.max_voxels, args.max_input_points, args.non_bed_keep_ratio)
        if result is None:
            continue

        feat_np, wall_mask, is_bed_scene = result
        N_i = feat_np.shape[0]

        all_features.append(feat_np)
        all_wall.append(wall_mask)
        all_bed_scene.append(np.full(N_i, int(is_bed_scene), dtype=np.int32))

        scene_log.append({
            "scene_name":        scene_name,
            "scene_path":        scene_path,
            "is_bed_scene":      is_bed_scene,
            "n_coarsest_voxels": N_i,
            "global_start":      global_offset,          # first row index in numpy arrays
            "global_end":        global_offset + N_i,    # exclusive end index
            "wall_voxels":       int(wall_mask.sum()),
            "wall_ratio":        round(float(wall_mask.mean()) * 100, 2),
        })
        global_offset += N_i

    if not all_features:
        print("No scenes passed the filter. Exiting.")
        exit(1)

    # ── Stack & save ──────────────────────────────────────────────────────────
    features_arr = np.concatenate(all_features, axis=0)   # (N_total, 512)
    wall_arr     = np.concatenate(all_wall,     axis=0)   # (N_total,)
    bed_arr      = np.concatenate(all_bed_scene,axis=0)   # (N_total,)

    feat_path    = os.path.join(args.output_dir, "all_features.npy")
    wall_path    = os.path.join(args.output_dir, "wall_labels.npy")
    bed_path     = os.path.join(args.output_dir, "bed_scene_labels.npy")

    np.save(feat_path, features_arr)
    np.save(wall_path, wall_arr)
    np.save(bed_path,  bed_arr)

    with open(args.json_log, "w") as jf:
        json.dump(scene_log, jf, indent=2)

    N_total = features_arr.shape[0]
    print(f"\n── Saved ──────────────────────────────────────────────────")
    print(f"  all_features.npy     : {features_arr.shape}  float32")
    print(f"  wall_labels.npy      : {wall_arr.shape}  int32  "
          f"({wall_arr.sum():,} wall voxels, {wall_arr.mean()*100:.1f}%)")
    print(f"  bed_scene_labels.npy : {bed_arr.shape}  int32  "
          f"({bed_arr.sum():,} bed-scene voxels, {bed_arr.mean()*100:.1f}%)")
    print(f"  scene_log.json       : {len(scene_log)} scenes")
    print(f"  Total voxels         : {N_total:,}")
    print("Done!")
