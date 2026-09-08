import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import os
from sklearn.model_selection import train_test_split
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch.manual_seed(42)
np.random.seed(42)

CKPT_DIR = "/pathto/encoderweights"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Load all subset2 arrays ────────────────────────────────────
# All four arrays share the same row ordering: row i in embeddings
# corresponds to row i in both label arrays.
orig_emb_all    = np.load("/pathto/original_embeddings.npy")
robo_emb_all    = np.load("/pathto/roboshape_embeddings.npy")
pub_labels_all  = np.load("/pathto/public_label.npy")
priv_labels_all = np.load("/pathto/private_label.npy")

# ── 80 / 10 / 10 split ────────────────────────────────────────
# We split a single index array and then apply the SAME indices to
# all four arrays. This guarantees that row-level alignment between
# embeddings and labels is preserved after shuffling.
idx = np.arange(len(pub_labels_all))
idx_train, idx_temp = train_test_split(idx, test_size=0.20, random_state=42, stratify=pub_labels_all)
idx_val,   idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42,
                                        stratify=pub_labels_all[idx_temp])

print(f"Split sizes — train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

# Apply the same index arrays to every array so alignment is preserved
original_embeddings_train  = orig_emb_all[idx_train]
original_embeddings_val    = orig_emb_all[idx_val]
original_embeddings_test   = orig_emb_all[idx_test]

roboshape_embeddings_train = robo_emb_all[idx_train]
roboshape_embeddings_val   = robo_emb_all[idx_val]
roboshape_embeddings_test  = robo_emb_all[idx_test]

public_lables_train  = pub_labels_all[idx_train]
public_lables_val    = pub_labels_all[idx_val]
public_lables_test   = pub_labels_all[idx_test]

private_lables_train = priv_labels_all[idx_train]
private_lables_val   = priv_labels_all[idx_val]
private_lables_test  = priv_labels_all[idx_test]

# ── Pre-convert val / test splits to tensors ──────────────────
X_orig_val   = torch.from_numpy(original_embeddings_val)
X_orig_test  = torch.from_numpy(original_embeddings_test)
X_robo_val   = torch.from_numpy(roboshape_embeddings_val)
X_robo_test  = torch.from_numpy(roboshape_embeddings_test)
y_pub_val    = torch.from_numpy(public_lables_val).long()
y_pub_test   = torch.from_numpy(public_lables_test).long()
y_priv_val   = torch.from_numpy(private_lables_val).long()
y_priv_test  = torch.from_numpy(private_lables_test).long()


original_public_classifier = Classifier(512, 256, 2)   # out_dim=2: one score per class
roboshape_public_classifier = Classifier(64, 32, 2)
original_private_classifier = Classifier(512, 256, 2)
roboshape_private_classifier = Classifier(64, 32, 2)

criterion = nn.CrossEntropyLoss()
optimizer_original_public = torch.optim.Adam(original_public_classifier.parameters(), lr=0.001)
optimizer_roboshape_public = torch.optim.Adam(roboshape_public_classifier.parameters(), lr=0.001)
optimizer_original_private = torch.optim.Adam(original_private_classifier.parameters(), lr=0.001)
optimizer_roboshape_private = torch.optim.Adam(roboshape_private_classifier.parameters(), lr=0.001)

epochs_original = 500
epochs_roboshape = 1000
loss_original_public = []
loss_roboshape_public = []
loss_original_private = []
loss_roboshape_private = []

loss_original_public_val  = []
loss_roboshape_public_val = []
loss_original_private_val = []
loss_roboshape_private_val = []

loss_original_public_test = []
loss_roboshape_public_test = []
loss_original_private_test = []
loss_roboshape_private_test = []


for i in range(epochs_original):
    y_pred_original_public  = original_public_classifier.forward(torch.from_numpy(original_embeddings_train))
    y_pred_original_private = original_private_classifier.forward(torch.from_numpy(original_embeddings_train))

    # Keep tensors for .backward(), store float values for logging
    l_orig_pub  = criterion(y_pred_original_public,  torch.from_numpy(public_lables_train).long())
    l_orig_priv = criterion(y_pred_original_private, torch.from_numpy(private_lables_train).long())

    loss_original_public.append(l_orig_pub.item())
    loss_original_private.append(l_orig_priv.item())

    if i % 10 == 0:
        print("Epoch:", i, "Loss Original Public:", loss_original_public[-1], "Loss Original Private:", loss_original_private[-1])

    optimizer_original_public.zero_grad()
    optimizer_original_private.zero_grad()

    l_orig_pub.backward()
    l_orig_priv.backward()

    optimizer_original_public.step()
    optimizer_original_private.step()

    if i % 50 == 0:
        torch.save(original_public_classifier.state_dict(),  os.path.join(CKPT_DIR, "original_public_classifier_"  + str(i) + ".pt"))
        torch.save(original_private_classifier.state_dict(), os.path.join(CKPT_DIR, "original_private_classifier_" + str(i) + ".pt"))

    # Evaluate on val and test sets
    original_public_classifier.eval()
    original_private_classifier.eval()
    with torch.no_grad():
        # Validation
        logits_orig_pub_val  = original_public_classifier(X_orig_val)
        logits_orig_priv_val = original_private_classifier(X_orig_val)
        loss_original_public_val.append(criterion(logits_orig_pub_val,  y_pub_val).item())
        loss_original_private_val.append(criterion(logits_orig_priv_val, y_priv_val).item())
        # Test
        logits_orig_pub  = original_public_classifier(X_orig_test)
        logits_orig_priv = original_private_classifier(X_orig_test)
        loss_original_public_test.append(criterion(logits_orig_pub,  y_pub_test).item())
        loss_original_private_test.append(criterion(logits_orig_priv, y_priv_test).item())

    # Back to training mode for next epoch
    original_public_classifier.train()
    original_private_classifier.train()


for i in range(epochs_roboshape):
    y_pred_roboshape_public  = roboshape_public_classifier(torch.from_numpy(roboshape_embeddings_train))
    y_pred_roboshape_private = roboshape_private_classifier(torch.from_numpy(roboshape_embeddings_train))

    l_robo_pub  = criterion(y_pred_roboshape_public,  torch.from_numpy(public_lables_train).long())
    l_robo_priv = criterion(y_pred_roboshape_private, torch.from_numpy(private_lables_train).long())

    loss_roboshape_public.append(l_robo_pub.item())
    loss_roboshape_private.append(l_robo_priv.item())

    if i % 10 == 0:
        print("Epoch:", i, "Loss Roboshape Public:", loss_roboshape_public[-1], "Loss Roboshape Private:", loss_roboshape_private[-1])

    optimizer_roboshape_public.zero_grad()
    optimizer_roboshape_private.zero_grad()
    l_robo_pub.backward()
    l_robo_priv.backward()
    optimizer_roboshape_public.step()
    optimizer_roboshape_private.step()

    if i % 50 == 0:
        torch.save(roboshape_public_classifier.state_dict(),  os.path.join(CKPT_DIR, "roboshape_public_classifier_"  + str(i) + ".pt"))
        torch.save(roboshape_private_classifier.state_dict(), os.path.join(CKPT_DIR, "roboshape_private_classifier_" + str(i) + ".pt"))

    roboshape_public_classifier.eval()
    roboshape_private_classifier.eval()

    with torch.no_grad():
        # Validation
        logits_robo_pub_val  = roboshape_public_classifier(X_robo_val)
        logits_robo_priv_val = roboshape_private_classifier(X_robo_val)
        loss_roboshape_public_val.append(criterion(logits_robo_pub_val,  y_pub_val).item())
        loss_roboshape_private_val.append(criterion(logits_robo_priv_val, y_priv_val).item())
        # Test
        logits_robo_pub  = roboshape_public_classifier(X_robo_test)
        logits_robo_priv = roboshape_private_classifier(X_robo_test)
        loss_roboshape_public_test.append(criterion(logits_robo_pub,  y_pub_test).item())
        loss_roboshape_private_test.append(criterion(logits_robo_priv, y_priv_test).item())

    roboshape_public_classifier.train()
    roboshape_private_classifier.train()


# ── Plots ──────────────────────────────────────────────────────
orig_range = range(epochs_original)
robo_range = range(epochs_roboshape)

# Plot 1: Public (wall) label — train vs val loss
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(orig_range, loss_original_public,     label="Train", color="steelblue")
axes[0].plot(orig_range, loss_original_public_val, label="Val",   color="steelblue", linestyle="--")
axes[0].set_title("Original embeddings — Public (wall) loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("CrossEntropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(robo_range, loss_roboshape_public,     label="Train", color="darkorange")
axes[1].plot(robo_range, loss_roboshape_public_val, label="Val",   color="darkorange", linestyle="--")
axes[1].set_title("Roboshape embeddings — Public (wall) loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("CrossEntropy Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_public_arkitscenes_subset2.png", dpi=150)
plt.close()
print("Saved: loss_public_arkitscenes_subset2.png")

# Plot 2: Private (bedroom) label — train vs val loss
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(orig_range, loss_original_private,     label="Train", color="seagreen")
axes[0].plot(orig_range, loss_original_private_val, label="Val",   color="seagreen", linestyle="--")
axes[0].set_title("Original embeddings — Private (bedroom) loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("CrossEntropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(robo_range, loss_roboshape_private,     label="Train", color="crimson")
axes[1].plot(robo_range, loss_roboshape_private_val, label="Val",   color="crimson", linestyle="--")
axes[1].set_title("Roboshape embeddings — Private (bedroom) loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("CrossEntropy Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_private_arkitscenes_subset2.png", dpi=150)
plt.close()
print("Saved: loss_private_arkitscenes_subset2.png")
