#!/bin/bash
# =============================================================
# setup_ablation.sh — Setup and run one ablation on A40
# =============================================================
# USAGE:
#   bash setup_ablation.sh baseline       ← Run 1
#   bash setup_ablation.sh attr           ← Run 2
#   bash setup_ablation.sh attr_subj      ← Run 3
#
# Upload this script + all 12 files to /workspace/ first.
# =============================================================

set -e

ABLATION=${1:-baseline}
echo "=============================================="
echo "  CCDG Ablation: $ABLATION"
echo "=============================================="

# Validate argument
if [[ "$ABLATION" != "baseline" && "$ABLATION" != "attr" && "$ABLATION" != "attr_subj" ]]; then
    echo "ERROR: argument must be baseline, attr, or attr_subj"
    exit 1
fi

# ── Step 1: Install dependencies ───────────────────────────
echo "[1/5] Installing dependencies..."
pip install -q transformers==4.40.0
pip install -q torch torchvision
pip install -q spacy
python -m spacy download en_core_web_sm -q
echo "  Done."

# ── Step 2: Create folders ──────────────────────────────────
echo "[2/5] Creating folders..."
mkdir -p /container/coco
mkdir -p /container/refer
mkdir -p /workspace/outputs/ablation_${ABLATION}/checkpoints
mkdir -p /workspace/ccdg/{models,losses,data,neg_cache}
touch /workspace/ccdg/{models,losses,data}/__init__.py
echo "  Done."

# ── Step 3: Download COCO 2014 images ──────────────────────
echo "[3/5] Downloading COCO 2014 train images (~13GB)..."
if [ "$(ls /container/coco/train2014 2>/dev/null | wc -l)" -gt "1000" ]; then
    echo "  Already downloaded. Skipping."
else
    wget -q --show-progress \
        http://images.cocodataset.org/zips/train2014.zip \
        -O /container/train2014.zip
    python3 -c "
import zipfile, os
print('Extracting...')
with zipfile.ZipFile('/container/train2014.zip') as z:
    z.extractall('/container/coco/')
print('Done:', len(os.listdir('/container/coco/train2014')), 'images')
"
    rm /container/train2014.zip
fi

# ── Step 4: Download RefCOCO annotations ───────────────────
echo "[4/5] Downloading RefCOCO annotations..."
for DS in refcoco refcoco+ refcocog; do
    DS_DIR="/container/refer/$DS"
    if [ -d "$DS_DIR" ] && [ "$(ls -A $DS_DIR)" ]; then
        echo "  $DS already exists. Skipping."
    else
        mkdir -p $DS_DIR
        if [ "$DS" = "refcoco" ]; then
            URL="https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"
        elif [ "$DS" = "refcoco+" ]; then
            URL="https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"
        else
            URL="https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"
        fi
        wget -q --show-progress "$URL" -O /tmp/${DS}.zip
        python3 -c "
import zipfile
with zipfile.ZipFile('/tmp/${DS}.zip') as z:
    z.extractall('/container/refer/')
print('${DS}: done')
"
        rm /tmp/${DS}.zip
    fi
done

# ── Step 5: Set up code ─────────────────────────────────────
echo "[5/5] Setting up code..."

# Copy files
cp /workspace/train.py       /workspace/ccdg/
cp /workspace/eval.py        /workspace/ccdg/
cp /workspace/ccdg.py        /workspace/ccdg/models/
cp /workspace/contrastive.py /workspace/ccdg/losses/
cp /workspace/phrase_cache.json /workspace/ccdg/neg_cache/
cp /workspace/neg_cache.json    /workspace/ccdg/neg_cache/

# Copy the correct ablation config as config.py
cp /workspace/config_ablation_${ABLATION}.py /workspace/ccdg/config.py
echo "  Using config: config_ablation_${ABLATION}.py"

# Create minimal phrase_parser.py
cat > /workspace/ccdg/data/phrase_parser.py << 'PARSER'
from dataclasses import dataclass

@dataclass
class PhraseComponents:
    original:      str
    subject:       str
    attribute:     str = ""
    spatial:       str = ""
    has_attribute: bool = False
    has_spatial:   bool = False
PARSER

# Create minimal refcoco_dataset.py
cat > /workspace/ccdg/data/refcoco_dataset.py << 'DATASET'
import os, json, pickle
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
from dataclasses import dataclass

@dataclass
class PhraseComponents:
    original: str
    subject: str
    attribute: str = ""
    spatial: str = ""
    has_attribute: bool = False
    has_spatial: bool = False

class RefCOCODataset(Dataset):
    def __init__(self, dataset_name, split, processor,
                 neg_cache=None, phrase_cache=None,
                 cfg_data=None, cfg_paths=None):
        self.dataset_name = dataset_name
        self.split = split
        self.processor = processor
        self.neg_cache = neg_cache or {}
        self.phrase_cache = phrase_cache or {}
        self.max_len = 16
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import DataConfig, Paths
        self.cfg = cfg_data or DataConfig()
        self.paths = cfg_paths or Paths()
        self.refs, self.anns, self.imgs = self._load()
        self.samples = self._build()
        print(f"[RefCOCODataset] {dataset_name}/{split}: {len(self.samples)} samples")

    def _load(self):
        data_dir = Path(self.paths.refcoco_root) / self.dataset_name
        for fname in ["refs(unc).p", "refs(umd).p", "refs(google).p"]:
            ref_file = data_dir / fname
            if ref_file.exists():
                break
        with open(ref_file, "rb") as f:
            all_refs = pickle.load(f)
        refs = [r for r in all_refs if r["split"] == self.split]
        with open(data_dir / "instances.json") as f:
            inst = json.load(f)
        anns = {a["id"]: a for a in inst["annotations"]}
        imgs = {i["id"]: i for i in inst["images"]}
        return refs, anns, imgs

    def _build(self):
        samples = []
        for ref in self.refs:
            ann_id = ref["ann_id"]
            image_id = ref["image_id"]
            if ann_id not in self.anns or image_id not in self.imgs:
                continue
            ann = self.anns[ann_id]
            img_info = self.imgs[image_id]
            for s in ref["sentences"]:
                samples.append({
                    "sent_id": s["sent_id"], "sent": s["sent"].lower().strip(),
                    "ann_id": ann_id, "image_id": image_id,
                    "image_file": img_info["file_name"], "bbox": ann["bbox"],
                    "img_w": img_info["width"], "img_h": img_info["height"],
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def tok(self, text):
        encoding = self.processor.tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=self.max_len, truncation=True, add_special_tokens=True,
        )
        ids  = encoding["input_ids"][0][:self.max_len]
        mask = encoding["attention_mask"][0][:self.max_len]
        if ids.shape[0] < self.max_len:
            pad = self.max_len - ids.shape[0]
            ids  = torch.nn.functional.pad(ids,  (0, pad))
            mask = torch.nn.functional.pad(mask, (0, pad))
        return ids, mask

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(os.path.join(self.paths.coco_images, s["image_file"])).convert("RGB")
        proc_out = self.processor(images=image, text=["dummy"], return_tensors="pt")
        pixel_values = proc_out["pixel_values"].squeeze(0)
        key = (self.dataset_name, s["sent_id"])
        comp = self.phrase_cache.get(key)
        if comp is None:
            comp = PhraseComponents(original=s["sent"], subject=s["sent"])
        full_phrase = comp.original
        ids_full, mask_full = self.tok(full_phrase)
        ids_subj, mask_subj = self.tok(comp.subject or full_phrase)
        ids_attr, mask_attr = self.tok(comp.attribute if comp.attribute else full_phrase)
        ids_spat, mask_spat = self.tok(comp.spatial   if comp.spatial   else full_phrase)
        x, y, w, h = s["bbox"]
        iw, ih = s["img_w"], s["img_h"]
        gt_box = torch.tensor([(x+w/2)/iw,(y+h/2)/ih,w/iw,h/ih],dtype=torch.float32).clamp(0,1)
        neg_key = (self.dataset_name, s["ann_id"])
        negs = self.neg_cache.get(neg_key, [])[:self.cfg.num_negatives]
        num_neg = self.cfg.num_negatives
        pad_ids, pad_mask = self.tok("")
        neg_ids_list, neg_mask_list = [], []
        for neg in negs:
            t = neg.get("text","") if isinstance(neg,dict) else str(neg)
            ni, nm = self.tok(t)
            neg_ids_list.append(ni)
            neg_mask_list.append(nm)
        while len(neg_ids_list) < num_neg:
            neg_ids_list.append(pad_ids.clone())
            neg_mask_list.append(pad_mask.clone())
        neg_ids_t  = torch.stack(neg_ids_list)
        neg_mask_t = torch.stack(neg_mask_list)
        return {
            "pixel_values": pixel_values,
            "input_ids_full": ids_full,  "attn_mask_full": mask_full,
            "input_ids_subj": ids_subj,  "attn_mask_subj": mask_subj,
            "input_ids_attr": ids_attr,  "attn_mask_attr": mask_attr,
            "input_ids_spat": ids_spat,  "attn_mask_spat": mask_spat,
            "gt_box": gt_box,
            "neg_attr_ids": neg_ids_t, "neg_attr_masks": neg_mask_t,
            "neg_subj_ids": neg_ids_t, "neg_subj_masks": neg_mask_t,
            "neg_spat_ids": neg_ids_t, "neg_spat_masks": neg_mask_t,
            "has_attribute": torch.tensor(comp.has_attribute, dtype=torch.bool),
            "has_spatial":   torch.tensor(comp.has_spatial,   dtype=torch.bool),
            "image_id": s["image_id"], "ann_id": s["ann_id"], "sent": s["sent"],
        }
DATASET

echo "  Code ready."

# Import test
python3 -c "
import sys
sys.path.insert(0, '/workspace/ccdg')
from models.ccdg import CCDGModel
from losses.contrastive import CCDGLoss
from config import Paths, ModelConfig, TrainConfig
print('  All imports OK.')
"

# ── Start training ──────────────────────────────────────────
echo ""
echo "=============================================="
echo "  STARTING ABLATION: $ABLATION"
echo "  Checkpoints: /workspace/outputs/ablation_${ABLATION}/"
echo "  Resume if needed:"
echo "    python train.py --resume /workspace/outputs/ablation_${ABLATION}/checkpoints/checkpoint_epochN.pth"
echo "=============================================="
echo ""

cd /workspace/ccdg
TOKENIZERS_PARALLELISM=false python train.py \
    2>&1 | grep -v "Unused or unrecognized" \
         | tee /workspace/outputs/ablation_${ABLATION}/training_log.txt
