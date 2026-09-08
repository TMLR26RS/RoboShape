
import os
import sys
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── ScanNet-20 meta 
CLASS_LABELS_20 = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "desk",
    "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)
NUM_CLASSES = len(CLASS_LABELS_20)  # 20

# GT segment20.npy stores class indices 0-19 (255 = unlabelled)
UNLABELLED = 255


try:
    import flash_attn
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

try:
    import sonata
    HAS_SONATA = True
except ImportError:
    HAS_SONATA = False


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


def load_model(device):
    if not HAS_SONATA:
        raise ImportError("sonata package not found. Install it first.")

    sonata.utils.set_seed(24525867)

    if HAS_FLASH:
        model = sonata.load("sonata", repo_id="facebook/sonata").to(device)
    else:
        print("Flash Attention not found; using reduced patch size config.")
        custom_config = dict(
            enc_patch_size=[1024] * 5,
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).to(device)

    ckpt = sonata.load(
        "sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True
    )
    seg_head = SegHead(**ckpt["config"]).to(device)
    seg_head.load_state_dict(ckpt["state_dict"])

    transform = sonata.transform.default()
    return model, seg_head, transform


def load_scene(scene_path):
    point = {
        "coord":  np.load(os.path.join(scene_path, "coord.npy")),
        "color":  np.load(os.path.join(scene_path, "color.npy")),
        "normal": np.load(os.path.join(scene_path, "normal.npy")),
    }
    seg_path = os.path.join(scene_path, "segment20.npy")
    gt_labels = np.load(seg_path) if os.path.exists(seg_path) else None
    return point, gt_labels


@torch.inference_mode()
def infer_scene(point_raw, model, seg_head, transform, device):
    point = transform(dict(point_raw))           

   
    for k, v in point.items():
        if isinstance(v, torch.Tensor):
            point[k] = v.to(device, non_blocking=True)

    out = model(point)

    while "pooling_parent" in out.keys():
        parent = out.pop("pooling_parent")
        inverse = out.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
        out = parent

    pred = seg_head(out.feat).argmax(dim=-1).cpu().numpy()  
    return pred


def classes_present(labels, total_pts, min_point_frac, ignore_val=None):
    """Return set of class indices present above the threshold."""
    present = set()
    for cls in range(NUM_CLASSES):
        if ignore_val is not None and cls == ignore_val:
            continue
        count = np.sum(labels == cls)
        if count / total_pts >= min_point_frac:
            present.add(cls)
    return present


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_histograms(
    inference_pct,
    gt_pct,
    match_ratio,
    total_scenes,
    output_dir,
):
    plt.style.use("default")

    labels = CLASS_LABELS_20
    x = np.arange(NUM_CLASSES)
    bar_w = 0.35

    # ── Figure 1: Side-by-side presence histogram ─────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(18, 7))
    fig1.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    bars_gt   = ax1.bar(x - bar_w / 2, gt_pct,        bar_w, label="GT (segment20)",
                        color="#4ea8de", alpha=0.88, zorder=3)
    bars_inf  = ax1.bar(x + bar_w / 2, inference_pct, bar_w, label="Inference (Sonata)",
                        color="#f77f00", alpha=0.88, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=13, color="#111111")
    ax1.set_ylabel("% of scenes where class is present", color="#111111", fontsize=11)
    ax1.set_title(
        f"ScanNet-20 Furniture Presence per Scene\n"
        f"({total_scenes} scenes — GT vs Inference)",
        color="black", fontsize=14, pad=12,
    )
    ax1.set_ylim(0, 110)
    ax1.yaxis.set_tick_params(labelcolor="#111111")
    ax1.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax1.spines[:].set_color("#aaaaaa")
    ax1.legend(fontsize=10, labelcolor="black", facecolor="white", edgecolor="#aaaaaa")

    
    for bar in bars_gt:
        h = bar.get_height()
        if h > 1:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1,
                     f"{h:.0f}", ha="center", va="bottom", fontsize=11, color="#4ea8de")
    for bar in bars_inf:
        h = bar.get_height()
        if h > 1:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1,
                     f"{h:.0f}", ha="center", va="bottom", fontsize=11, color="#f77f00")

    fig1.tight_layout()
    p1 = os.path.join(output_dir, "furniture_presence_histogram.png")
    fig1.savefig(p1, dpi=150, facecolor=fig1.get_facecolor())
    print(f"Saved → {p1}")
    plt.close(fig1)

    # ── Figure 2: Match ratio ──────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(18, 5))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")

    colors = [
        "#43aa8b" if r >= 0.75 else ("#f9c74f" if r >= 0.50 else "#f94144")
        for r in match_ratio
    ]
    ax2.bar(x, [r * 100 for r in match_ratio], color=colors, alpha=0.9, zorder=3)
    ax2.axhline(75, color="#43aa8b", linestyle="--", linewidth=1, alpha=0.6, label="75% threshold")
    ax2.axhline(50, color="#f9c74f", linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=13, color="#111111")
    ax2.set_ylabel("Match ratio  (% of scenes where GT == Inference)", color="#111111", fontsize=10)
    ax2.set_title(
        "Per-Class Agreement Between GT Labels and Sonata Inference\n"
        "(scene-level: class 'present' or 'absent')",
        color="black", fontsize=14, pad=12,
    )
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_tick_params(labelcolor="#111111")
    ax2.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax2.spines[:].set_color("#aaaaaa")

    for i, r in enumerate(match_ratio):
        ax2.text(i, r * 100 + 1.5, f"{r*100:.1f}%",
                 ha="center", va="bottom", fontsize=11, color="black")

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#43aa8b", label="≥75%  Good"),
        Patch(color="#f9c74f", label="50-75%  Fair"),
    ]
    ax2.legend(handles=legend_patches, fontsize=9, labelcolor="black",
               facecolor="white", edgecolor="#aaaaaa")

    fig2.tight_layout()
    p2 = os.path.join(output_dir, "inference_vs_gt_match_ratio.png")
    fig2.savefig(p2, dpi=150, facecolor=fig2.get_facecolor())
    print(f"Saved → {p2}")
    plt.close(fig2)

    return p1, p2



def main():
    parser = argparse.ArgumentParser(
        description="Furniture presence histogram: inference vs GT labels over ScanNet."
    )
    parser.add_argument(
        "--data_root", type=str,
        default="",
        help="Path to scannet_all_processed/ directory",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="",
        help="Where to save plots and CSV",
    )
    parser.add_argument(
        "--max_scenes", type=int, default=None,
        help="Optional: process only the first N scenes (for quick tests)",
    )
    parser.add_argument(
        "--min_point_frac", type=float, default=0.001,
        help="A class is 'present' only if ≥ this fraction of points are that class "
             "(default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (cuda or cpu)",
    )
    parser.add_argument(
        "--gt_only", action="store_true",
        help="Skip inference, only compute GT histogram from segment20.npy files",
    )
    parser.add_argument(
        "--embeddings_dir", type=str,
        default="",
        help="Path to directory containing precomputed encoder feature .npy files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover scenes ────────────────────────────────────────────────────────
    all_scene_dirs = sorted(
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith("scene")
    )
    if args.max_scenes:
        all_scene_dirs = all_scene_dirs[: args.max_scenes]
    total_scenes = len(all_scene_dirs)
    print(f"Found {total_scenes} scenes in {args.data_root}")

    # ── Load model (unless --gt_only) ─────────────────────────────────────────
    if not args.gt_only:
        print("Loading Sonata model …")
        model, seg_head, transform = load_model(args.device)
        model.eval()
        seg_head.eval()
        print("Model loaded.")
    else:
        model = seg_head = transform = None

    # ── Per-scene counting ────────────────────────────────────────────────────
    scene_present_inf[c]  = []#scenes where inference says class c is present
    scene_present_gt[c]   = []#scenes where GT says class c is present
    scene_match[c]        = []#scenes where both agree (both present or both absent)

    scene_present_inf = np.zeros(NUM_CLASSES, dtype=np.int64)
    scene_present_gt  = np.zeros(NUM_CLASSES, dtype=np.int64)
    scene_match       = np.zeros(NUM_CLASSES, dtype=np.int64)
    scenes_with_gt    = 0

    per_scene_results = []   # for CSV export

    for idx, scene_name in enumerate(all_scene_dirs):
        scene_path = os.path.join(args.data_root, scene_name)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{total_scenes}] {scene_name}")

        point_raw, gt_labels = load_scene(scene_path)
        n_pts = len(point_raw["coord"])

        # Load precomputed encoder features to get embedding point count
        emb_path = os.path.join(
            args.embeddings_dir, f"{scene_name}_encoder_features.npy"
        )
        if os.path.exists(emb_path):
            emb = np.load(emb_path, mmap_mode="r")
            n_embedding_pts = int(emb.shape[0])
        else:
            print(f"    WARNING: embedding file not found for {scene_name}, setting n_points=-1")
            n_embedding_pts = -1

        gt_present_set = set()
        if gt_labels is not None:
            scenes_with_gt += 1
            valid_mask = gt_labels != UNLABELLED
            n_valid = valid_mask.sum()
            if n_valid > 0:
                gt_present_set = classes_present(
                    gt_labels[valid_mask], n_valid, args.min_point_frac
                )
            for c in gt_present_set:
                scene_present_gt[c] += 1

        inf_present_set = set()
        if not args.gt_only:
            try:
                pred = infer_scene(point_raw, model, seg_head, transform, args.device)
                inf_present_set = classes_present(pred, n_pts, args.min_point_frac)
                for c in inf_present_set:
                    scene_present_inf[c] += 1
            except Exception as e:
                print(f"    WARNING: inference failed for {scene_name}: {e}")

        if gt_labels is not None and not args.gt_only:
            for c in range(NUM_CLASSES):
                gt_has  = c in gt_present_set
                inf_has = c in inf_present_set
                if gt_has == inf_has:
                    scene_match[c] += 1

        # save per-scene detail
        per_scene_results.append({
            "scene": f"{scene_name}_encoder_features",
            "n_points": n_embedding_pts,
            "inf_classes": sorted(inf_present_set),
            "gt_classes":  sorted(gt_present_set),
        })

   
    inf_pct = scene_present_inf / total_scenes * 100
    gt_pct  = scene_present_gt  / max(scenes_with_gt, 1) * 100

    # match ratio is #agreements / #scenes_with_GT (for classes we can compare)
    match_ratio = scene_match / max(scenes_with_gt, 1)

    # ── Console summary ────────────────────────────────────────────────────────
    header = f"\n{'Class':<20} {'GT %':>8}  {'Inf %':>8}  {'Match':>8}"
    print(header)
    print("─" * len(header))
    for c, label in enumerate(CLASS_LABELS_20):
        gt_s  = f"{gt_pct[c]:7.1f}%" if scenes_with_gt > 0 else "  N/A  "
        inf_s = f"{inf_pct[c]:7.1f}%" if not args.gt_only else "  N/A  "
        ma_s  = f"{match_ratio[c]*100:7.1f}%" if (scenes_with_gt > 0 and not args.gt_only) else "  N/A  "
        print(f"  {label:<18} {gt_s}  {inf_s}  {ma_s}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("class,gt_pct,inf_pct,match_pct\n")
        for c, label in enumerate(CLASS_LABELS_20):
            f.write(f"{label},{gt_pct[c]:.2f},{inf_pct[c]:.2f},{match_ratio[c]*100:.2f}\n")
    print(f"\nCSV saved → {csv_path}")

    # ── Save per-scene JSON ────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "per_scene_classes.json")
    with open(json_path, "w") as f:
        json.dump(per_scene_results, f, indent=2)
    print(f"Per-scene JSON saved → {json_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_histograms(inf_pct, gt_pct, match_ratio, total_scenes, args.output_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
