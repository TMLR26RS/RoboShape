import numpy as np
import open3d as o3d
import sonata
import torch
import torch.nn as nn
import os

try:
    import flash_attn
except ImportError:
    flash_attn = None

# ScanNet Meta data 
VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 45, 28, 33, 34, 36, 39, 41, 42, 43, 44, 46, 47, 48, 49, 50)

VALID_CLASS_IDS_20_DEMO = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 45, 28, 33, 34, 36, 39, 41, 42, 43, 44, 46, 47, 48, 49, 50)


CLASS_LABELS_20 = ("wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture")

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


def load_custom_data(scene_path):
  
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene path not found: {scene_path}")
        
    print(f"Loading data from {scene_path}...")
    point = {}
    point["coord"] = np.load(os.path.join(scene_path, "coord.npy"))
    point["color"] = np.load(os.path.join(scene_path, "color.npy"))
    point["normal"] = np.load(os.path.join(scene_path, "normal.npy"))

    if os.path.exists(os.path.join(scene_path, "segment20.npy")):
        point["segment"] = np.load(os.path.join(scene_path, "segment20.npy"))
    elif os.path.exists(os.path.join(scene_path, "segment.npy")):
        point["segment"] = np.load(os.path.join(scene_path, "segment.npy"))
    
    return point

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, default="")
    args = parser.parse_args()

   
    sonata.utils.set_seed(24525867)
    
  
    if flash_attn is not None:
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        print("Flash Attention not found, using custom config...")
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).cuda()
        
  
    print("Loading Segmentation Head...")
    ckpt = sonata.load(
        "sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True
    )
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])
    
 
    transform = sonata.transform.default()
    
    
    scene_path = args.scene_path
    point = load_custom_data(scene_path)
    
  
    original_coord = point["coord"].copy()
    point_transformed = transform(point)

    for key in point_transformed.keys():
        if isinstance(point_transformed[key], torch.Tensor):

            point_transformed[key] = point_transformed[key].cuda(non_blocking=True)

    # Inference
    print("Running Inference...")
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        # model forward:
        point_output = model(point_transformed)
        print("point_output.feat.shape", point_output.feat.shape)
    
        
        # Save encoder features
        scene_name = os.path.basename(args.scene_path)
        feature_save_path = f"{scene_name}_encoder_features.npy"
        np.save(feature_save_path, point_output.feat.cpu().numpy())
        print(f"Saved encoder features to {feature_save_path}")

        while "pooling_parent" in point_output.keys():
            assert "pooling_inverse" in point_output.keys()
            parent = point_output.pop("pooling_parent")
            inverse = point_output.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point_output.feat[inverse]], dim=-1)
            point_output = parent
            print("point_output.feat.shape", point_output.feat.shape)
        feat = point_output.feat
        print("feat.shape", feat.shape)
        
        seg_logits = seg_head(feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()
    
       
        color_map_list = [SCANNET_COLOR_MAP_20.get(i, (0,0,0)) for i in range(max(SCANNET_COLOR_MAP_20.keys()) + 1)]
       
        
        # Re-construct visual map
        VALID_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39) # Top 20 semantic classes usually
        
        DEMO_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
       
        print(f"DEBUG: Prediction Array: {pred}")
        print(f"DEBUG: Unique IDs found: {np.unique(pred)}")
        
        
        DEMO_COLORS = []
        for uid in DEMO_IDS:
            DEMO_COLORS.append(SCANNET_COLOR_MAP_20.get(uid, (0,0,0)))
        
       
        color = np.array(DEMO_COLORS)[pred]

    
    print("Visualizing...")
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(point_transformed["coord"].cpu().detach().numpy())
    
    pcd.colors = o3d.utility.Vector3dVector(color / 255)
    
    output_filename = "inference_result.ply"
    o3d.io.write_point_cloud(output_filename, pcd)
    print(f"Saved result to {output_filename}")
