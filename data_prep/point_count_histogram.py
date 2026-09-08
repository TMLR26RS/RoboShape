

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR   = ""
PROCESSED_DIR    = ""
OUTPUT_PATH      = ""
LEGEND_PATH      = ""
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

files = sorted(glob.glob(os.path.join(EMBEDDINGS_DIR, "*.npy")))
print(f"Found {len(files)} embedding files.")

# ── Load point counts & scene types ──────────────────────────────────────────
point_counts = []
scene_types  = []

for f in tqdm(files, desc="Loading embeddings"):
    arr = np.load(f, mmap_mode="r")   # mmap: no need to load full tensor
    point_counts.append(arr.shape[0])

    # Derive scene name from filename, e.g. scene0000_00_encoder_features.npy
    basename  = os.path.basename(f)                        # scene0000_00_encoder_features.npy
    scene_name = "_".join(basename.split("_")[:2])         # scene0000_00
    type_file  = os.path.join(PROCESSED_DIR, scene_name, "scene_type.txt")
    if os.path.exists(type_file):
        with open(type_file, "r") as tf:
            stype = tf.read().strip()
    else:
        stype = "Unknown"
    scene_types.append(stype)
    
    



point_counts = np.array(point_counts, dtype=np.int64)
scene_types  = np.array(scene_types)

# ── Statistics ────────────────────────────────────────────────────────────────
mean  = point_counts.mean()
std   = point_counts.std()
vmin  = point_counts.min()
vmax  = point_counts.max()
p25   = np.percentile(point_counts, 25)
p50   = np.percentile(point_counts, 50)
p75   = np.percentile(point_counts, 75)

print("\n── Point Count Statistics ──────────────────────────────")
print(f"  Count   : {len(point_counts)} scenes")
print(f"  Mean    : {mean:,.1f}")
print(f"  Std Dev : {std:,.1f}")
print(f"  Min     : {vmin:,}")
print(f"  Max     : {vmax:,}")
print(f"  P25     : {p25:,.1f}")
print(f"  Median  : {p50:,.1f}")
print(f"  P75     : {p75:,.1f}")
print("────────────────────────────────────────────────────────\n")

# ── Determine bins ─────────────────────────────────────────────────────────────
n_bins = max(30, int(np.ceil(np.log2(len(point_counts)) + 1)))
bin_edges = np.linspace(vmin, vmax, n_bins + 1)

# ── Scene-type palette ────────────────────────────────────────────────────────
# Sort scene types by overall frequency (most common first → bottom of stack)
unique_types, type_counts_all = np.unique(scene_types, return_counts=True)
order = np.argsort(-type_counts_all)          # descending frequency
unique_types = unique_types[order]

# Kelly's colors of maximum contrast — designed for maximum perceptual separation
BASE_COLORS = [
    "#F3C300",  # vivid yellow
    "#875692",  # strong purple
    "#F38400",  # vivid orange
    "#0067A5",  # strong blue
    "#BE0032",  # vivid red
    "#008856",  # vivid green
    "#604E97",  # strong violet
    "#F6A600",  # vivid orange-yellow
    "#B3446C",  # strong purplish-red
    "#8DB600",  # vivid yellow-green
    "#882D17",  # strong reddish-brown
    "#E25822",  # vivid reddish-orange
    "#2B3D26",  # dark olive green
    "#DCD300",  # vivid greenish-yellow
    "#E68FAC",  # strong purplish-pink
    "#A1CAF1",  # light blue
    "#C2B280",  # moderate buff/tan
    "#848482",  # medium grey
    "#F99379",  # light yellowish-pink
    "#222222",  # near-black
    "#007D8C",  # vivid teal
]

colors = {t: BASE_COLORS[i % len(BASE_COLORS)] for i, t in enumerate(unique_types)}

# ── Build per-type histograms ─────────────────────────────────────────────────
# type_hist[i] → array of per-bin scene counts for unique_types[i]
type_hists = {}
for stype in unique_types:
    mask = scene_types == stype
    counts_in_bins, _ = np.histogram(point_counts[mask], bins=bin_edges)
    type_hists[stype] = counts_in_bins

# Total counts per bin (for reference lines)
total_hist, _ = np.histogram(point_counts, bins=bin_edges)

# Bar centres & widths
bar_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bar_width   = bin_edges[1] - bin_edges[0]

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("#f6f8fa")

# Stacked bars — bottom accumulates as we layer each scene type
bottoms = np.zeros(len(bar_centres), dtype=np.float64)
for stype in unique_types:
    h = type_hists[stype].astype(np.float64)
    ax.bar(
        bar_centres,
        h,
        width=bar_width * 0.92,
        bottom=bottoms,
        color=colors[stype],
        label=stype,
        edgecolor="#0d1117",
        linewidth=0.3,
        alpha=0.92,
        zorder=2,
    )
    bottoms += h

# Mean & ±1 std lines
ax.axvline(mean,       color="#333333", linewidth=2,   linestyle="--",
           label=f"Mean = {mean:,.0f}", zorder=4)
ax.axvline(mean + std, color="#1a7f37", linewidth=1.5, linestyle=":",
           label=f"Mean ± 1σ  (σ={std:,.0f})", zorder=4)
ax.axvline(mean - std, color="#1a7f37", linewidth=1.5, linestyle=":", zorder=4)

# Shade ±1σ region
ax.axvspan(max(mean - std, vmin), mean + std,
           alpha=0.08, color="#1a7f37", zorder=1)

# Median line
ax.axvline(p50, color="#8250df", linewidth=1.5, linestyle="-.",
           label=f"Median = {p50:,.0f}", zorder=4)

# Stats text box — placed below the reference-line legend (upper-right)
# Use axes-fraction coords so it sits under the legend
stats_text = (
    f"n     = {len(point_counts):,}\n"
    f"Min   = {vmin:,}\n"
    f"Max   = {vmax:,}\n"
    f"P25   = {p25:,.0f}\n"
    f"P75   = {p75:,.0f}"
)
ax.text(
    0.98, 0.62, stats_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
              edgecolor="#cccccc", alpha=0.95),
    color="#333333",
    fontfamily="monospace",
    zorder=5,
)

# ── Legend — two sections: reference lines (upper-right inside) + scene types (outside right) ──
# Reference line handles
line_handles, line_labels = ax.get_legend_handles_labels()
ref_handles = [(h, l) for h, l in zip(line_handles, line_labels)
               if l.startswith(("Mean", "Median"))]

# Build scene-type patches legend FIRST so ax.legend() below can be the "current" one
type_patches = [
    mpatches.Patch(color=colors[t], label=f"{t}  ({type_counts_all[order[i]]:,})")
    for i, t in enumerate(unique_types)
]
legend_types = ax.legend(
    handles=type_patches,
    title="Scene Type (# scenes)",
    title_fontsize=9,
    fontsize=7.5,
    facecolor="white",
    edgecolor="#cccccc",
    labelcolor="#333333",
    framealpha=0.95,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    ncol=1,
    handlelength=1.2,
    handleheight=0.9,
    handletextpad=0.5,
    labelspacing=0.35,
)
ax.add_artist(legend_types)  # pin it so the next ax.legend() doesn't replace it

# Reference lines legend — inside the plot at upper right
ax.legend(
    handles=[h for h, _ in ref_handles],
    labels=[l for _, l in ref_handles],
    fontsize=9,
    facecolor="white",
    edgecolor="#cccccc",
    labelcolor="#333333",
    framealpha=0.95,
    loc="upper right",
)

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_title(
    "Distribution of Point Counts in ScanNet Encoder Embeddings\n"
    "(bars divided by scene type)",
    fontsize=15, fontweight="bold", color="#111111", pad=14,
)
ax.set_xlabel("Number of Points (a)",  fontsize=12, color="#555555")
ax.set_ylabel("Number of Scenes",      fontsize=12, color="#555555")
ax.tick_params(colors="#333333", labelsize=15)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for spine in ax.spines.values():
    spine.set_edgecolor("#cccccc")
ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)

plt.tight_layout(rect=[0, 0, 0.80, 1])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Histogram saved → {OUTPUT_PATH}")
plt.close()

# ── Standalone color-map legend PNG (horizontal grid) ────────────────────────
n_types  = len(unique_types)
n_cols   = min(6, n_types)          # up to 6 items per row
n_rows   = int(np.ceil(n_types / n_cols))

cell_w   = 2.4    # inches per column
cell_h   = 0.55   # inches per row
fig_w    = cell_w * n_cols + 0.4
fig_h    = cell_h * n_rows + 0.7   # extra headroom for title

fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")
ax2.set_axis_off()
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax2.set_title("Scene Type → Color", fontsize=13, fontweight="bold",
              color="#111111", pad=10)

# grid positions in axes-fraction coords
swatch_w  = 0.025          # swatch width  (fraction of axes)
swatch_h  = 0.045          # swatch height
gap_x     = 0.01           # gap between swatch and text
col_step  = 1.0 / n_cols
row_step  = (1.0 - 0.08) / n_rows   # 0.08 top margin

for i, stype in enumerate(unique_types):
    cnt  = type_counts_all[order[i]]
    col  = i % n_cols
    row  = i // n_cols

    x0   = col * col_step + 0.01
    y0   = 1.0 - (row + 1) * row_step + row_step * 0.25   # vertical center of row

    # color swatch
    ax2.add_patch(mpatches.FancyBboxPatch(
        (x0, y0 - swatch_h / 2), swatch_w, swatch_h,
        boxstyle="round,pad=0.003",
        facecolor=colors[stype], edgecolor="#aaaaaa", linewidth=0.7,
        transform=ax2.transAxes, clip_on=False,
    ))
    # label
    ax2.text(
        x0 + swatch_w + gap_x, y0,
        f"{stype} ({cnt:,})",
        transform=ax2.transAxes,
        fontsize=8.5, color="#333333",
        verticalalignment="center",
        clip_on=False,
    )

plt.tight_layout(pad=0.4)
plt.savefig(LEGEND_PATH, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Color map legend saved → {LEGEND_PATH}")
plt.show()
