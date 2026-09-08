from src.models.models_to_train import DenseEncoder
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


_SPLIT_SEED = 42
_TEST_SIZE  = 0.20   # 80 % train
_VAL_FRAC   = 0.50   # 50 % of the 20 % temp → 10 % val, 10 % test



DEFAULT_IN_DIM = 512
DEFAULT_HIDDEN_DIMS = [256, 128]
DEFAULT_OUT_DIM = 64


def build_encoder(in_dim: int, hidden_dims: list, out_dim: int) -> DenseEncoder:
    return DenseEncoder(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim)


def main():
    parser = argparse.ArgumentParser(description="Run trained encoder and save outputs.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the saved encoder state_dict (.pt file).",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to the input embeddings .npy file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="encoded_output.npy",
        help="Output path for the encoded embeddings (.npy). Default: encoded_output.npy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g. cuda, cuda:3, cpu). Default: cuda if available.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="If set, reconstruct the 80/10/10 split  and "
             "encode only that subset. Requires --label1 for stratification.",
    )
    parser.add_argument(
        "--label1",
        type=str,
        default=None,
        help="Path to the label1 .npy file used for stratified splitting (required when --split is set).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # 1. Build encoder with same architecture used during training
    encoder = build_encoder(DEFAULT_IN_DIM, DEFAULT_HIDDEN_DIMS, DEFAULT_OUT_DIM)
    print(encoder)

    # 2. Load saved weights
    weights_path = Path(args.weights)
    assert weights_path.exists(), f"Weights file not found: {weights_path}"
    state_dict = torch.load(weights_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    print(f"Loaded weights from: {weights_path}")

    # 3. Load input embeddings (and optionally slice to the requested split)
    embeddings_path = Path(args.embeddings)
    assert embeddings_path.exists(), f"Embeddings file not found: {embeddings_path}"
    raw = np.load(embeddings_path)

    if args.split is not None:
        assert args.label1 is not None, "--label1 is required when --split is set"
        label1_path = Path(args.label1)
        assert label1_path.exists(), f"Label1 file not found: {label1_path}"
        l1_np = np.load(label1_path)

        idx = np.arange(len(l1_np))
        idx_train, idx_temp = train_test_split(
            idx, test_size=_TEST_SIZE, random_state=_SPLIT_SEED, stratify=l1_np
        )
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=_VAL_FRAC, random_state=_SPLIT_SEED, stratify=l1_np[idx_temp]
        )
        split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
        chosen_idx = split_map[args.split]
        raw = raw[chosen_idx]
        print(f"Split '{args.split}': {len(chosen_idx):,} rows selected out of {len(l1_np):,} total")

    x = torch.tensor(raw, dtype=torch.float32).to(device)
    print(f"Input shape:  {x.shape}")

    # 4. Forward pass — no MINE, no gradients needed
    with torch.no_grad():
        encoded = encoder(x)

    print(f"Output shape: {encoded.shape}")

    # 5. Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, encoded.cpu().numpy())
    print(f"Saved encoded output to: {out_path}")


if __name__ == "__main__":
    main()