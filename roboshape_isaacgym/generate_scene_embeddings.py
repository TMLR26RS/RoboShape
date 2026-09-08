"""
generate_scene_embeddings.py
===================================
Runs Sonata + RoboShape on the 20 downloaded scenes to extract the spatial coordinates (x,y,z) 
and the corresponding 64-dim and 512-dim features for the Hybrid Lookup Approach.
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import sonata
import sys


try:
    import MinkowskiEngine as ME
except ImportError:
    print("MinkowskiEngine not found.")
    
def load_custom_data(scene_path):
    point = {}
    point["coord"]  = np.load(os.path.join(scene_path, "coord.npy"))
    point["color"]  = np.load(os.path.join(scene_path, "color.npy"))
    point["normal"] = np.load(os.path.join(scene_path, "normal.npy"))
    seg20 = os.path.join(scene_path, "segment20.npy")
    if os.path.exists(seg20):
        point["segment"] = np.load(seg20)
    return point

def main():
    scans_dir = "/pathto/scans"
    processed_dir = "/pathto/processed"
    
    # Get the 20 downloaded scenes
    scenes = [d for d in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, d)) and d.startswith("scene")]
    print(f"Found {len(scenes)} downloaded scenes.")
    
    # Load Models
    print("Loading Sonata...")
    try:
        import flash_attn
    except ImportError:
        flash_attn = None

    if flash_attn is not None:
        sonata_model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        sonata_model = sonata.load(
            "sonata", repo_id="facebook/sonata",
            custom_config=dict(enc_patch_size=[1024]*5, enable_flash=False)
        ).cuda()
    sonata_model.eval()
    
    print("Loading roboshape...")
    sys.path.append("/pathto/Roboshape")
    from src.models.utils import create_encoder_model
    roboshape = create_encoder_model(
        model_name="DenseEncoder",
        model_params={"in_dim": 512, "hidden_dims": [256, 128], "out_dim": 64}
    ).cuda()
    roboshape.load_state_dict(torch.load("pathto/modelweights.pt"))
    roboshape.eval()
    
    transform = sonata.transform.default()
    
    with torch.no_grad():
        for scene in tqdm(scenes):
            proc_path = os.path.join(processed_dir, scene)
            if not os.path.exists(proc_path):
                print(f"Missing processed data for {scene}")
                continue
                
            raw = load_custom_data(proc_path)
            pt = transform(raw)
            for k, v in pt.items():
                if isinstance(v, torch.Tensor):
                    pt[k] = v.cuda(non_blocking=True)
                    
            # Sonata (512-dim)
            out = sonata_model(pt)
            feat_orig = out.feat # (N, 512)
            coords = out.coord # (N, 3)
            
            # Since Sonata uses MinkowskiEngine, coordinates are typically scaled (e.g. by voxel size)
            # We must map them back to metric space. By default, Sonata uses voxel_size=0.02 or 0.05
            # We will save the raw coords and un-project them in Isaac Gym based on bounding boxes.
            
            
            feat_robo = roboshape(feat_orig)
            
            # Map original segment20 labels to the downsampled coords
            # raw["coord"] is (M, 3) and coords is (N, 3)
            # raw["segment"] is (M,)
            if "segment" in raw:
                from scipy.spatial import cKDTree
                tree = cKDTree(raw["coord"])
                _, idxs = tree.query(coords.cpu().numpy(), k=1)
                labels = raw["segment"][idxs]
            else:
                labels = np.zeros(coords.shape[0], dtype=np.int32)
            
            out_path = os.path.join(scans_dir, scene, "spatial_embeddings.pt")
            torch.save({
                "coords": coords.cpu(),
                "feat_orig": feat_orig.cpu(),
                "feat_robo": feat_robo.cpu(),
                "labels": torch.from_numpy(labels)
            }, out_path)
            
    print("Successfully generated spatial embeddings for all scenes!")

if __name__ == "__main__":
    main()
