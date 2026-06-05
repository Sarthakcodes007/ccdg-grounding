import subprocess
subprocess.run(["pip","install","-q","transformers==4.38.0","datasets","timm"],
               capture_output=True)

import os, random, warnings
from io import BytesIO
from itertools import combinations
import requests
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from datasets import load_dataset

warnings.filterwarnings('ignore')
random.seed(42)
torch.manual_seed(42)

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS = '/kaggle/input/datasets/sarthakpandey007/ccdg-weights/ccdg_weights.pth'
OUT     = '/kaggle/working'
print(f"Device: {DEVICE} | Weights: {os.path.exists(WEIGHTS)}")

# ── Dataset ───────────────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset("jxu124/refcoco", split="test")
print(f"Loaded {len(ds)} samples")

# ── Image fetch ───────────────────────────────────────────────────
def fetch_image(image_id):
    url = (f"http://images.cocodataset.org/train2014/"
           f"COCO_train2014_{int(image_id):012d}.jpg")
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content)).convert('RGB')
    except: pass
    return None

# ── Helpers ───────────────────────────────────────────────────────
def get_phrase(s):
    sents = s.get('sentences', [])
    if sents:
        x = sents[0]
        return (x.get('sent') or x.get('raw') or '') if isinstance(x,dict) else str(x)
    return ''

def get_gt(s):
    b = s['bbox']
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])

def iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    da=(a[2]-a[0])*(a[3]-a[1]); db=(b[2]-b[0])*(b[3]-b[1])
    return inter/(da+db-inter+1e-8)

SPATIAL = {'left','right','top','bottom','above','below','near',
           'next','beside','behind','front','middle','center',
           'between','leftmost','rightmost','corner','first','last'}

def has_spatial(p):
    return bool(set(p.lower().split()) & SPATIAL)

# ── Models ────────────────────────────────────────────────────────
print("Loading models...")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

ccdg = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
state = torch.load(WEIGHTS, map_location='cpu')
if 'model_state_dict' in state: state = state['model_state_dict']
elif 'state_dict' in state:     state = state['state_dict']
ccdg.load_state_dict(state, strict=False)
ccdg.eval().to(DEVICE)

base = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
base.eval().to(DEVICE)
print("Models ready.")

def predict(mdl, img, phrase):
    inp = processor(text=[[phrase]], images=img, return_tensors='pt')
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = mdl(**inp)
    logits = out.logits[0, :, 0]
    boxes  = out.pred_boxes[0]
    best   = logits.argmax().item()
    W, H   = img.size
    cx,cy,w,h = boxes[best].cpu().tolist()
    return (cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H

# ── Collect 6 clean success examples ─────────────────────────────
print("\nCollecting 6 success examples (scanning up to 500 samples)...")
idxs = list(range(len(ds)))
random.shuffle(idxs)
success = []

for rank, idx in enumerate(idxs):
    if len(success) >= 6: break
    if rank >= 500: print("500-sample limit reached."); break
    try:
        s  = ds[idx]
        ph = get_phrase(s)
        if not ph or not has_spatial(ph): continue
        im = fetch_image(s['image_id'])
        if im is None: continue
        gt = get_gt(s)
        cb = predict(ccdg, im, ph)
        bb = predict(base, im, ph)
        ci = iou(cb, gt)
        bi = iou(bb, gt)
        if ci >= 0.5 and bi < 0.4:
            success.append(dict(
                img=im, phrase=ph, gt=gt,
                ccdg=cb, base=bb, ci=ci, bi=bi
            ))
            print(f"  [{len(success)}] '{ph[:50]}'  "
                  f"CCDG={ci:.2f}  Base={bi:.2f}")
    except: continue

print(f"\nCollected {len(success)} examples.")

# ── Plot function: clean 2×3 grid ─────────────────────────────────
def make_grid(cases, filename):
    """
    cases: list of 3 example dicts
    Layout:
      Row 0 (top):    Baseline predictions
      Row 1 (bottom): CCDG predictions
      Cols 0,1,2:     Three different examples
    """
    assert len(cases) == 3, "Need exactly 3 cases"

    fig, axes = plt.subplots(
        2, 3,
        figsize=(11, 5.2),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.12}
    )

    plt.rcParams.update({'font.family': 'serif', 'font.size': 9})

    row_labels  = ['Baseline', 'CCDG (ours)']
    row_colors  = ['#c62828', '#1565c0']

    for col, case in enumerate(cases):
        img_np = np.array(case['img'])

        for row in range(2):
            ax    = axes[row][col]
            box   = case['base']  if row == 0 else case['ccdg']
            ival  = case['bi']    if row == 0 else case['ci']
            color = row_colors[row]

            ax.imshow(img_np)
            ax.set_xticks([]); ax.set_yticks([])

            # Colored border matching row color
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(2.0)

            # Ground truth (dashed green)
            gx1,gy1,gx2,gy2 = case['gt']
            ax.add_patch(patches.Rectangle(
                (gx1,gy1), gx2-gx1, gy2-gy1,
                lw=2.0, edgecolor='#2e7d32',
                facecolor='none', ls='--', zorder=3))

            # Predicted box
            bx1,by1,bx2,by2 = box
            ax.add_patch(patches.Rectangle(
                (bx1,by1), bx2-bx1, by2-by1,
                lw=2.5, edgecolor=color,
                facecolor='none', zorder=4))

            # IoU badge top-left
            check = '✓' if ival >= 0.5 else '✗'
            ax.text(0.03, 0.96, f'{check} IoU={ival:.2f}',
                    transform=ax.transAxes, fontsize=8.5,
                    color=color, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', alpha=0.92,
                              ec=color, lw=0.9))

            # Row label on leftmost column only
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10,
                              fontweight='bold', color=color,
                              labelpad=6)

            # Phrase caption below bottom row only
            if row == 1:
                ph = case['phrase']
                if len(ph) > 32: ph = ph[:30] + '...'
                ax.set_xlabel(f'"{ph}"', fontsize=8.5,
                              color='#333333', labelpad=4,
                              fontstyle='italic')

    # ── REMOVED: "Example 1/2/3" column headers ──────────────────
    # ── REMOVED: fig.suptitle with option number ─────────────────

    # Legend only
    fig.legend(handles=[
        Line2D([0],[0],color='#2e7d32',lw=2.0,ls='--',
               label='Ground truth (GT)'),
        Line2D([0],[0],color='#c62828',lw=2.5,ls='-',
               label='Baseline prediction'),
        Line2D([0],[0],color='#1565c0',lw=2.5,ls='-',
               label='CCDG prediction (ours)'),
    ], loc='lower center', ncol=3, fontsize=9,
       framealpha=0.95, edgecolor='#cccccc',
       bbox_to_anchor=(0.5, -0.04))

    path_pdf = f'{OUT}/{filename}.pdf'
    path_png = f'{OUT}/{filename}.png'
    plt.savefig(path_pdf, bbox_inches='tight', dpi=200)
    plt.savefig(path_png, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}.pdf")

# ── Generate combinations ─────────────────────────────────────────
if len(success) < 3:
    print(f"\nOnly {len(success)} examples found. Need at least 3.")
    print("  - Increase scan limit from 500 to 800 in the loop above")
    print("  - Lower the baseline threshold from bi<0.4 to bi<0.5")
else:
    n = len(success)
    combos = list(combinations(range(n), 3))[:5]
    print(f"\nGenerating {len(combos)} clean PDF options...")

    for i, (a, b_, c) in enumerate(combos):
        fname = f'fig_qualitative_option{i+1}'
        trio  = [success[a], success[b_], success[c]]
        make_grid(trio, fname)
        print(f"    phrases: {trio[0]['phrase'][:28]} | "
              f"{trio[1]['phrase'][:28]} | "
              f"{trio[2]['phrase'][:28]}")

    print(f"\nDone. Files saved to /kaggle/working/")
    print("Download all, pick the best, rename to fig_qualitative.pdf")
    print("In main.tex replace \\fbox{{...}} in Section VIII with:")
    print("  \\includegraphics[width=\\textwidth]{fig_qualitative}")
