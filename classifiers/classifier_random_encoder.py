import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Train random-encoder classifiers")
parser.add_argument(
    "--encoder_save_path",
    type=str,
    default=None,
    help="Directory where every-50-epoch checkpoint weights are saved. "
         "Defaults to the current directory if omitted.",
)
args = parser.parse_args()



class DenseEncoder(nn.Module):
    """
    in_dim=512, hidden_dims=[256, 128], out_dim=64
    
    """
    def __init__(self, in_dim, hidden_dims, out_dim, dropout_rate=0.1):
        super().__init__()
        layers = []
        prev_size = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(nn.ReLU())
            prev_size = hidden_dim
        layers.append(nn.Linear(prev_size, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.float())


def apply_xavier(module):
    """Apply Xavier uniform init to all Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


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


torch.manual_seed(42)

# ── Build randomly initialized (Xavier)  ────────
random_encoder = DenseEncoder(in_dim=512, hidden_dims=[256, 128], out_dim=64)
random_encoder.apply(apply_xavier)
random_encoder.eval()  # frozen — never trained

# ── Load Matterport3D data and split 80/10/10 ─────────────────
orig_emb_all    = np.load("/pathto/original_embeddings.npy")
pub_labels_all  = np.load("/pathto/public_label.npy")
priv_labels_all = np.load("/pathto/private_label.npy")

idx = np.arange(len(pub_labels_all))
idx_train, idx_temp = train_test_split(idx, test_size=0.20, random_state=42, stratify=pub_labels_all)
idx_val,   idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=pub_labels_all[idx_temp])

original_embeddings_train = torch.from_numpy(orig_emb_all[idx_train]).float()
public_labels_train       = torch.from_numpy(pub_labels_all[idx_train]).long()
private_labels_train      = torch.from_numpy(priv_labels_all[idx_train]).long()

# ── Pass training data through frozen random encoder ────────────
with torch.no_grad():
    random_embeddings_train = random_encoder(original_embeddings_train)

# ── Use test split ──────────────────────────────────────────────
original_embeddings_test = torch.from_numpy(orig_emb_all[idx_test]).float()
y_pub_test  = torch.from_numpy(pub_labels_all[idx_test]).long()
y_priv_test = torch.from_numpy(priv_labels_all[idx_test]).long()

with torch.no_grad():
    X_rand_test = random_encoder(original_embeddings_test)

# ── Classifiers ─────────────────────────────────────────────────

random_public_classifier  = Classifier(64, 32, 2)
random_private_classifier = Classifier(64, 32, 2)

criterion               = nn.CrossEntropyLoss()
optimizer_rand_public   = torch.optim.Adam(random_public_classifier.parameters(),  lr=0.001)
optimizer_rand_private  = torch.optim.Adam(random_private_classifier.parameters(), lr=0.001)

# ── Loss logs ────────────────────────────────────────────────────
epochs = 1000
loss_rand_public        = []
loss_rand_private       = []
loss_rand_public_test   = []
loss_rand_private_test  = []

# ── Training loop ────────────────────────────────────────────────
for i in range(epochs):

    y_pred_rand_public  = random_public_classifier(random_embeddings_train)
    y_pred_rand_private = random_private_classifier(random_embeddings_train)

    l_rand_pub  = criterion(y_pred_rand_public,  public_labels_train)
    l_rand_priv = criterion(y_pred_rand_private, private_labels_train)

    loss_rand_public.append(l_rand_pub.item())
    loss_rand_private.append(l_rand_priv.item())

    if i % 10 == 0:
        print(f"Epoch: {i}  Loss Rand Public: {loss_rand_public[-1]:.4f}"
              f"  Loss Rand Private: {loss_rand_private[-1]:.4f}")

    optimizer_rand_public.zero_grad()
    optimizer_rand_private.zero_grad()
    l_rand_pub.backward()
    l_rand_priv.backward()
    optimizer_rand_public.step()
    optimizer_rand_private.step()

    # Checkpoint every 50 epochs (train mode, right after optimizer step)
    if i % 50 == 0:
        ckpt_dir = args.encoder_save_path or "."
        os.makedirs(ckpt_dir, exist_ok=True)
        pub_ckpt  = os.path.join(ckpt_dir, f"random_public_classifier_{i}.pt")
        priv_ckpt = os.path.join(ckpt_dir, f"random_private_classifier_{i}.pt")
        torch.save(random_public_classifier.state_dict(),  pub_ckpt)
        torch.save(random_private_classifier.state_dict(), priv_ckpt)
        print(f"  [ckpt epoch {i}] {pub_ckpt}")
        print(f"  [ckpt epoch {i}] {priv_ckpt}")

    # Evaluate on test set
    random_public_classifier.eval()
    random_private_classifier.eval()

    with torch.no_grad():
        logits_rand_pub  = random_public_classifier(X_rand_test)
        logits_rand_priv = random_private_classifier(X_rand_test)

        loss_rand_public_test.append(criterion(logits_rand_pub,  y_pub_test).item())
        loss_rand_private_test.append(criterion(logits_rand_priv, y_priv_test).item())

    random_public_classifier.train()
    random_private_classifier.train()


# ── Plots ────────────────────────────────────────────────────────
epoch_range = range(epochs)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epoch_range, loss_rand_public,      label="Train", color="steelblue")
axes[0].plot(epoch_range, loss_rand_public_test, label="Test",  color="steelblue", linestyle="--")
axes[0].set_title("Random Encoder — Public (wall) loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("CrossEntropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epoch_range, loss_rand_private,      label="Train", color="seagreen")
axes[1].plot(epoch_range, loss_rand_private_test, label="Test",  color="seagreen", linestyle="--")
axes[1].set_title("Random Encoder — Private (bedroom) loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("CrossEntropy Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_random_encoder.png", dpi=150)
plt.close()
print("Saved: loss_random_encoder.png")
