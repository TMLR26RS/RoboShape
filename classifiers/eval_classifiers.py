import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


# ── Reproduce the same 80/10/10 split as classifier2.py ───────
# Using identical seeds + stratify so idx_test is exactly the same.
np.random.seed(42)

orig_emb_all    = np.load("/path/to/dataset/arkitscenes_tensors_subset/all_features.npy")
robo_emb_all    = np.load("/path/to/dataset/arkitscenes_tensors_subset/arkitscenes_roboshape_embeddings.npy")
pub_labels_all  = np.load("/path/to/dataset/arkitscenes_tensors_subset/wall_labels.npy")
priv_labels_all = np.load("/path/to/dataset/arkitscenes_tensors_subset/bed_scene_labels.npy")

idx = np.arange(len(pub_labels_all))
idx_train, idx_temp = train_test_split(idx, test_size=0.20, random_state=42, stratify=pub_labels_all)
idx_val,   idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42,
                                        stratify=pub_labels_all[idx_temp])

X_orig_test  = torch.from_numpy(orig_emb_all[idx_test]).float()
X_robo_test  = torch.from_numpy(robo_emb_all[idx_test]).float()
y_pub_test   = torch.from_numpy(pub_labels_all[idx_test]).long()
y_priv_test  = torch.from_numpy(priv_labels_all[idx_test]).long()


# ── Load classifiers from saved weights ────────────────────────
original_public_classifier  = Classifier(512, 256, 2)
roboshape_public_classifier = Classifier(64,  32,  2) 
original_private_classifier = Classifier(512, 256, 2)
roboshape_private_classifier= Classifier(64,  32,  2)

original_public_classifier.load_state_dict(torch.load("/path/to/project/RoboShape/arkitscenes_private_public_classifier_weights/original_public_classifier_50.pt",   map_location="cpu"))
roboshape_public_classifier.load_state_dict(torch.load("/path/to/project/RoboShape/arkitscenes_private_public_classifier_weights/roboshape_public_classifier_950.pt",  map_location="cpu"))
original_private_classifier.load_state_dict(torch.load("/path/to/project/RoboShape/arkitscenes_private_public_classifier_weights/original_private_classifier_100.pt",  map_location="cpu"))
roboshape_private_classifier.load_state_dict(torch.load("/path/to/project/RoboShape/arkitscenes_private_public_classifier_weights/roboshape_private_classifier_950.pt", map_location="cpu"))

# ── Evaluate ───────────────────────────────────────────────────
acc_orig_pub  = accuracy(original_public_classifier,   X_orig_test, y_pub_test)
acc_robo_pub  = accuracy(roboshape_public_classifier,  X_robo_test, y_pub_test)
acc_orig_priv = accuracy(original_private_classifier,  X_orig_test, y_priv_test)
acc_robo_priv = accuracy(roboshape_private_classifier, X_robo_test, y_priv_test)

majority_pub  = max(y_pub_test.float().mean().item(),  1 - y_pub_test.float().mean().item())
majority_priv = max(y_priv_test.float().mean().item(), 1 - y_priv_test.float().mean().item())

print("=" * 55)
print("TEST ACCURACY SUMMARY")
print("=" * 55)
print(f"  Original  → public  (wall):    {acc_orig_pub:.3f} ")  
print(f"  Roboshape → public  (wall):    {acc_robo_pub:.3f}  ") 
print(f"  Original  → private (bedroom): {acc_orig_priv:.3f}"  )
print(f"  Roboshape → private (bedroom): {acc_robo_priv:.3f} ") 
print()

