"""
CCDG Paper - Figure Generation Script
Run this in Kaggle (GPU not needed, CPU is fine).
All 4 figures will be saved as high-resolution PDFs and PNGs.

Instructions:
  1. Upload this script to a Kaggle notebook
  2. Run all cells
  3. Download the output files from /kaggle/working/figures/
  4. Place the .pdf files in the same directory as your main.tex
  5. Replace the \fbox{} placeholders in the LaTeX with \includegraphics{}
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

os.makedirs('/kaggle/working/figures', exist_ok=True)

# ============================================================
# STYLE CONFIG — IEEE-quality fonts and sizes
# ============================================================
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':    0.8,
    'grid.linewidth':    0.4,
    'lines.linewidth':   1.5,
})

# ============================================================
# DATA — Extracted from all 4 training logs
# ============================================================

# ---------- Full CCDG (epoch-level train loss) ----------
ccdg_train_loss = [
    0.8027, 0.7475, 0.7126, 0.6837,  # epochs 0-3  (L1+IoU only)
    0.6652, 0.6515,                   # epochs 4-5  (L_attr ramps)
    0.6387, 0.6280, 0.6161, 0.6046,   # epochs 6-9  (L_attr full)
    0.5958, 0.5890,                   # epochs 10-11
    0.5940, 0.5958, 0.5874,           # epochs 12-14 (all losses)
]

# Validation Acc@0.5 (measured every 2 epochs for CCDG)
# epochs:  1,      3,      5,      7,      9,      11,     13
ccdg_val_epochs = [1,    3,    5,    7,    9,    11,   13]
ccdg_val_acc    = [0.3424, 0.4013, 0.4142, 0.4288, 0.4286, 0.4303, 0.4303]

# ---------- Baseline (L1+IoU only) ----------
baseline_train_loss = [
    0.8380, 0.7538, 0.7257, 0.7097, 0.6993,
    0.6917, 0.6858, 0.6810, 0.6770, 0.6736,
    0.6710, 0.6692, 0.6682,
]

# Validation Acc@0.5 (every 2 epochs)
# epochs:   1,      3,      5,      7,      9,      11,     12
baseline_val_epochs = [1,    3,    5,    7,    9,    11,   12]
baseline_val_acc    = [0.3297, 0.3549, 0.3576, 0.3607, 0.3593, 0.3591, 0.3591]

# ---------- +L_attr only ----------
attr_train_loss = [
    0.8380, 0.7538, 0.7257, 0.7097, 0.6993,
    # attr log only has 5 epoch summaries (stopped early / overlapped)
]
attr_val_epochs = [1, 3, 5]
attr_val_acc    = [0.3297, 0.3549, 0.3549]

# ---------- +L_attr +L_subj ----------
attr_subj_train_loss = [
    0.8380, 0.7538, 0.7257, 0.7097, 0.6993,
    0.6937, 0.6868, 0.6816, 0.6776, 0.6740,
    0.6716, 0.6698, 0.6734,
]
attr_subj_val_epochs = [1,    3,    5,    7,    9,    11,   12]
attr_subj_val_acc    = [0.3297, 0.3549, 0.3578, 0.3581, 0.3600, 0.3593, 0.3593]

# ---------- Final test Acc@0.5 (from eval_results) ----------
# Columns: val, testA, testB
refcoco_results = {
    'Baseline': [34.69, 36.90, 33.16],
    '+$\\mathcal{L}_{\\rm attr}$': [35.00, 37.00, 33.00],
    '+$\\mathcal{L}_{\\rm attr}$+$\\mathcal{L}_{\\rm subj}$': [35.00, 37.00, 33.00],
    'Full CCDG': [49.49, 50.27, 48.99],
}
refcocop_results = {
    'Baseline': [34.60, 36.32, 33.66],
    '+$\\mathcal{L}_{\\rm attr}$': [35.00, 37.00, 34.00],
    '+$\\mathcal{L}_{\\rm attr}$+$\\mathcal{L}_{\\rm subj}$': [35.00, 37.00, 34.00],
    'Full CCDG': [36.60, 37.32, 36.14],
}
refcocog_results = {
    'Baseline': [39.25, None, 39.85],
    '+$\\mathcal{L}_{\\rm attr}$': [39.42, None, 40.03],
    '+$\\mathcal{L}_{\\rm attr}$+$\\mathcal{L}_{\\rm subj}$': [39.42, None, 40.03],
    'Full CCDG': [42.87, None, 43.49],
}

# Per-split improvement (CCDG - Baseline) for Figure 3
splits      = ['RC val', 'RC tA', 'RC tB', 'RC+ val', 'RC+ tA', 'RC+ tB', 'RCg val', 'RCg test']
baseline_acc= [34.69,   36.90,   33.16,   34.60,    36.32,    33.66,    39.25,    39.85]
ccdg_acc    = [49.49,   50.27,   48.99,   36.60,    37.32,    36.14,    42.87,    43.49]
delta       = [c - b for c, b in zip(ccdg_acc, baseline_acc)]

# Colors
C_CCDG     = '#1f4e79'
C_BASE     = '#a0522d'
C_ATTR     = '#2e7d32'
C_ATTRSUBJ = '#6a0572'
C_GRID     = '#e0e0e0'
C_ANNOT    = '#333333'

# ============================================================
# FIGURE 1: Training Loss Curves (all 4 configs)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

ax1, ax2 = axes

# --- Left: Training loss ---
epochs_ccdg       = list(range(len(ccdg_train_loss)))
epochs_base       = list(range(len(baseline_train_loss)))
epochs_attrsubj   = list(range(len(attr_subj_train_loss)))

ax1.plot(epochs_base,     baseline_train_loss, color=C_BASE,     ls='--',  lw=1.4, label='Baseline')
ax1.plot(epochs_attrsubj, attr_subj_train_loss, color=C_ATTRSUBJ, ls='-.',  lw=1.4, label='+$\\mathcal{L}_{a}$+$\\mathcal{L}_{s}$')
ax1.plot(epochs_ccdg,     ccdg_train_loss,     color=C_CCDG,     ls='-',   lw=1.8, label='Full CCDG')

# Curriculum markers
ax1.axvline(4,  color='#888', lw=0.8, ls=':', alpha=0.8)
ax1.axvline(12, color='#888', lw=0.8, ls=':', alpha=0.8)
ax1.text(4.15,  0.96, '$\\mathcal{L}_{a}$\nactivates', fontsize=6.5,
         color='#555', va='top', ha='left', linespacing=1.3)
ax1.text(12.15, 0.96, 'All\nlosses', fontsize=6.5,
         color='#555', va='top', ha='left', linespacing=1.3)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('(a) Training loss convergence')
ax1.set_xlim(-0.3, 14.5)
ax1.set_ylim(0.55, 1.00)
ax1.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc')
ax1.yaxis.grid(True, color=C_GRID, zorder=0)
ax1.set_axisbelow(True)

# --- Right: Val Acc@0.5 ---
ax2.plot(baseline_val_epochs, [v*100 for v in baseline_val_acc],
         color=C_BASE, ls='--', lw=1.4, marker='s', ms=3.5, label='Baseline')
ax2.plot(attr_subj_val_epochs, [v*100 for v in attr_subj_val_acc],
         color=C_ATTRSUBJ, ls='-.', lw=1.4, marker='^', ms=3.5,
         label='+$\\mathcal{L}_{a}$+$\\mathcal{L}_{s}$')
ax2.plot(ccdg_val_epochs, [v*100 for v in ccdg_val_acc],
         color=C_CCDG, ls='-', lw=1.8, marker='o', ms=4, label='Full CCDG')

# Annotate final CCDG val point
ax2.annotate('43.0%', xy=(13, 43.03), xytext=(11.0, 44.5),
             fontsize=7, color=C_CCDG, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_CCDG, lw=0.8))

ax2.axvline(4,  color='#888', lw=0.8, ls=':', alpha=0.8)
ax2.axvline(12, color='#888', lw=0.8, ls=':', alpha=0.8)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Val Acc@0.5 (%)')
ax2.set_title('(b) Validation accuracy')
ax2.set_xlim(-0.3, 14.5)
ax2.set_ylim(28, 48)
ax2.legend(loc='upper left', framealpha=0.9, edgecolor='#ccc')
ax2.yaxis.grid(True, color=C_GRID, zorder=0)
ax2.set_axisbelow(True)

plt.tight_layout(pad=0.6)
plt.savefig('/kaggle/working/figures/fig_training_curves.pdf')
plt.savefig('/kaggle/working/figures/fig_training_curves.png', dpi=300)
plt.close()
print("Figure 1 saved: fig_training_curves")


# ============================================================
# FIGURE 2: Per-Split Improvement Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 2.8))

x      = np.arange(len(splits))
width  = 0.32

bars_base = ax.bar(x - width/2, baseline_acc, width, label='Baseline',
                   color=C_BASE, alpha=0.85, edgecolor='white', lw=0.5, zorder=3)
bars_ccdg = ax.bar(x + width/2, ccdg_acc,     width, label='Full CCDG',
                   color=C_CCDG, alpha=0.92, edgecolor='white', lw=0.5, zorder=3)

# Delta annotations above CCDG bars
for i, (bar, d) in enumerate(zip(bars_ccdg, delta)):
    clr = C_CCDG if d > 5 else ('#2e7d32' if d > 2 else '#888888')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
            f'+{d:.1f}', ha='center', va='bottom', fontsize=6.5,
            color=clr, fontweight='bold')

# Shaded region separating datasets
ax.axvspan(2.5, 5.5, alpha=0.06, color='gray', zorder=1)
ax.axvspan(5.5, 7.5, alpha=0.06, color='blue', zorder=1)

# Dataset labels
ax.text(1.0, 54.5, 'RefCOCO', ha='center', fontsize=7.5,
        color='#333', fontweight='bold')
ax.text(4.0, 54.5, 'RefCOCO+', ha='center', fontsize=7.5,
        color='#333', fontweight='bold')
ax.text(6.5, 54.5, 'RefCOCOg', ha='center', fontsize=7.5,
        color='#333', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(splits, fontsize=7.5)
ax.set_ylabel('Acc@0.5 (%)')
ax.set_ylim(28, 57)
ax.yaxis.grid(True, color=C_GRID, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='#ccc')
ax.set_title('Baseline vs. Full CCDG across all evaluation splits',
             pad=4)

# Horizontal ref line
ax.axhline(37, color='#aaa', lw=0.6, ls='--', alpha=0.6)
ax.text(7.55, 37.3, '37%', fontsize=6.5, color='#888')

plt.tight_layout(pad=0.6)
plt.savefig('/kaggle/working/figures/fig_split_comparison.pdf')
plt.savefig('/kaggle/working/figures/fig_split_comparison.png', dpi=300)
plt.close()
print("Figure 2 saved: fig_split_comparison")


# ============================================================
# FIGURE 3: Ablation / Component Interaction Heatmap
# ============================================================
methods = ['Baseline', '+$\\mathcal{L}_{\\rm attr}$',
           '+$\\mathcal{L}_{\\rm attr}$+$\\mathcal{L}_{\\rm subj}$',
           'Full CCDG']

# Acc@0.5 matrix  [method x split]
# Splits: RC_val RC_tA RC_tB RC+_val RC+_tA RC+_tB RCg_val RCg_test
data = np.array([
    [34.69, 36.90, 33.16, 34.60, 36.32, 33.66, 39.25, 39.85],  # Baseline
    [35.00, 37.00, 33.00, 35.00, 37.00, 34.00, 39.42, 40.03],  # +Lattr
    [35.00, 37.00, 33.00, 35.00, 37.00, 34.00, 39.42, 40.03],  # +Lattr+Lsubj
    [49.49, 50.27, 48.99, 36.60, 37.32, 36.14, 42.87, 43.49],  # Full CCDG
])

col_labels = ['val', 'tA', 'tB', 'val', 'tA', 'tB', 'val', 'test']

fig, ax = plt.subplots(figsize=(7.0, 2.4))

# Custom colormap: white-to-blue
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('wb',
    ['#f0f4ff', '#c7d9f5', '#5b9bd5', '#1f4e79'], N=256)

im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=32, vmax=52)

# Cell text
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        txt_color = 'white' if val > 46 else 'black'
        weight = 'bold' if (i == 3 or (i > 0 and j < 3 and val > data[0,j] + 3)) else 'normal'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=7.5, color=txt_color, fontweight=weight)

# Vertical separator between datasets
ax.axvline(2.5, color='white', lw=2.5)
ax.axvline(5.5, color='white', lw=2.5)

# Axes
ax.set_xticks(range(8))
ax.set_xticklabels(col_labels)
ax.set_yticks(range(4))
ax.set_yticklabels(methods, fontsize=8)

# Dataset header annotations
ax.annotate('RefCOCO', xy=(1.0, -0.75), xycoords=('data', 'axes fraction'),
            ha='center', fontsize=7.5, fontweight='bold', color='#333',
            annotation_clip=False)
ax.annotate('RefCOCO+', xy=(4.0, -0.75), xycoords=('data', 'axes fraction'),
            ha='center', fontsize=7.5, fontweight='bold', color='#333',
            annotation_clip=False)
ax.annotate('RefCOCOg', xy=(6.5, -0.75), xycoords=('data', 'axes fraction'),
            ha='center', fontsize=7.5, fontweight='bold', color='#333',
            annotation_clip=False)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
cbar.set_label('Acc@0.5 (%)', fontsize=7.5)
cbar.ax.tick_params(labelsize=7)

ax.set_title('Component interaction analysis (Acc@0.5\%)',
             pad=14, fontsize=9)

plt.tight_layout(pad=0.5)
plt.savefig('/kaggle/working/figures/fig_ablation_heatmap.pdf')
plt.savefig('/kaggle/working/figures/fig_ablation_heatmap.png', dpi=300)
plt.close()
print("Figure 3 saved: fig_ablation_heatmap")


# ============================================================
# FIGURE 4: Spatial vs Appearance Split Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

# ---- Left: Gains by dataset type ----
ax = axes[0]

group_labels = ['RefCOCO\n(spatial allowed)',
                'RefCOCO+\n(no spatial)',
                'RefCOCOg\n(spatial allowed)']
baseline_means = [
    np.mean([34.69, 36.90, 33.16]),
    np.mean([34.60, 36.32, 33.66]),
    np.mean([39.25, 39.85]),
]
ccdg_means = [
    np.mean([49.49, 50.27, 48.99]),
    np.mean([36.60, 37.32, 36.14]),
    np.mean([42.87, 43.49]),
]
gains = [c - b for c, b in zip(ccdg_means, baseline_means)]

xg = np.arange(3)
w  = 0.32
b1 = ax.bar(xg - w/2, baseline_means, w, label='Baseline',
            color=C_BASE, alpha=0.85, edgecolor='white')
b2 = ax.bar(xg + w/2, ccdg_means,    w, label='Full CCDG',
            color=C_CCDG, alpha=0.92, edgecolor='white')
for i, (bar, g) in enumerate(zip(b2, gains)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'+{g:.1f}%', ha='center', va='bottom', fontsize=7.5,
            color=C_CCDG if g > 5 else '#888', fontweight='bold')

ax.set_xticks(xg)
ax.set_xticklabels(group_labels, fontsize=7.5, linespacing=1.4)
ax.set_ylabel('Mean Acc@0.5 (%)')
ax.set_ylim(28, 58)
ax.yaxis.grid(True, color=C_GRID, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
ax.set_title('(a) Gain by dataset spatial policy')

# ---- Right: Gain breakdown per split ----
ax = axes[1]

colors_bar = [C_CCDG if d > 5 else ('#2e7d32' if d > 2 else '#a0a0a0')
              for d in delta]

bars = ax.barh(splits[::-1], delta[::-1], color=colors_bar[::-1],
               edgecolor='white', lw=0.5, height=0.6, zorder=3)

ax.axvline(0, color='#666', lw=0.8)
ax.axvline(5, color='#ddd', lw=0.6, ls='--')
ax.axvline(10, color='#ddd', lw=0.6, ls='--')
ax.axvline(15, color='#ddd', lw=0.6, ls='--')

for bar, d in zip(bars, delta[::-1]):
    ax.text(d + 0.2, bar.get_y() + bar.get_height()/2,
            f'{d:.1f}pp', va='center', fontsize=7, color='#222')

# Spatial vs no-spatial annotation
ax.axhspan(1.5, 5.5, alpha=0.06, color='orange', zorder=1)
ax.text(16.5, 3.5, 'No spatial\nlanguage\n(RefCOCO+)',
        fontsize=6.5, color='#b45309', ha='left', va='center',
        linespacing=1.4)

ax.set_xlabel('Improvement over Baseline (pp)')
ax.set_xlim(-1, 20)
ax.set_title('(b) Per-split absolute gain')
ax.yaxis.grid(False)
ax.xaxis.grid(True, color=C_GRID, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.6)
plt.savefig('/kaggle/working/figures/fig_spatial_analysis.pdf')
plt.savefig('/kaggle/working/figures/fig_spatial_analysis.png', dpi=300)
plt.close()
print("Figure 4 saved: fig_spatial_analysis")


# ============================================================
# Summary
# ============================================================
print("\n" + "="*55)
print("All figures saved to /kaggle/working/figures/")
print("="*55)
print("""
Files generated:
  fig_training_curves.pdf / .png
  fig_split_comparison.pdf / .png
  fig_ablation_heatmap.pdf / .png
  fig_spatial_analysis.pdf / .png

In your LaTeX paper, replace each \fbox{} placeholder with:
  \includegraphics[width=\columnwidth]{fig_training_curves}
  \includegraphics[width=\columnwidth]{fig_split_comparison}
  \includegraphics[width=\columnwidth]{fig_ablation_heatmap}
  \includegraphics[width=\columnwidth]{fig_spatial_analysis}
""")
