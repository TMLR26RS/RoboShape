import numpy as np
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Root directory containing scene* folders.",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="Bedroom",
        help="Scene type to label as 1 (case-insensitive). Default: Bedroom",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="bedroom_scenes",
        help="Prefix for output files.",
    )
    args = parser.parse_args()

    scene_paths = sorted(glob.glob(os.path.join(args.data_dir, "scene*")))
    print(f"Found {len(scene_paths)} scenes.")

    target = args.target_type.strip().lower()
    scene_names = []
    labels = []
    type_counts = {}

    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        type_path  = os.path.join(scene_path, "scene_type.txt")

        if not os.path.exists(type_path):
            print(f"  [SKIP] {scene_name} — no scene_type.txt")
            scene_names.append(scene_name)
            labels.append(0)
            continue

        scene_type = open(type_path).read().strip()
        type_counts[scene_type] = type_counts.get(scene_type, 0) + 1

        label = int(target in scene_type.lower())
        labels.append(label)
        scene_names.append(scene_name)

    label_tensor = np.array(labels, dtype=np.int32)

    # Save binary tensor
    npy_path = f"{args.output_prefix}_labels.npy"
    np.save(npy_path, label_tensor)
    print(f"Saved label tensor -> {npy_path}  shape={label_tensor.shape}")

    # Save scene list with labels
    txt_path = f"{args.output_prefix}_scenes.txt"
    with open(txt_path, "w") as f:
        for name, lbl in zip(scene_names, label_tensor):
            f.write(f"{name}\t{lbl}\n")
    print(f"Saved scene list   -> {txt_path}")

    # Summary
    n_pos = int(label_tensor.sum())
    n_tot = len(label_tensor)
    print(f"\nSummary:")
    print(f"  Target type       : '{args.target_type}'")
    print(f"  Total scenes      : {n_tot}")
    print(f"  Matching scenes   : {n_pos}  ({n_pos/n_tot*100:.1f}%)")
    print(f"  Other scenes      : {n_tot - n_pos}  ({(n_tot-n_pos)/n_tot*100:.1f}%)")
    print(f"\nAll scene types found:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:4d}  {t}")

if __name__ == "__main__":
    main()
