#!/bin/bash
# =============================================================
# setup_runpod.sh — One-shot setup and training for CCDG
# =============================================================
# USAGE:
#   1. Start a RunPod instance (A100 80GB or A6000 48GB)
#   2. Upload this script + your 7 files to /workspace
#   3. Run: bash setup_runpod.sh
#
# WHAT THIS SCRIPT DOES:
#   Step 1 — Install all Python dependencies
#   Step 2 — Download COCO 2014 train images
#   Step 3 — Download RefCOCO/+/g annotations
#   Step 4 — Set up folder structure
#   Step 5 — Start training
# =============================================================

set -e  # stop immediately if any command fails

echo "=============================================="
echo "  CCDG RunPod Setup"
echo "=============================================="

# ── Paths ──────────────────────────────────────────────────
WORKSPACE="/workspace"
DATA_DIR="/workspace/data"
COCO_DIR="$DATA_DIR/coco/train2014"
REFER_DIR="$DATA_DIR/refer"
OUTPUT_DIR="/workspace/outputs/ccdg"
CODE_DIR="/workspace/ccdg"

# ── Step 1: Install dependencies ───────────────────────────
echo ""
echo "[1/5] Installing dependencies..."
pip install -q transformers==4.40.0
pip install -q torch torchvision
pip install -q spacy
python -m spacy download en_core_web_sm -q
echo "  Dependencies installed."

# ── Step 2: Create folder structure ────────────────────────
echo ""
echo "[2/5] Creating folder structure..."
mkdir -p $COCO_DIR
mkdir -p $REFER_DIR
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $CODE_DIR/models
mkdir -p $CODE_DIR/losses
mkdir -p $CODE_DIR/data
mkdir -p $CODE_DIR/neg_cache

# ── Step 3: Download COCO 2014 images ──────────────────────
echo ""
echo "[3/5] Downloading COCO 2014 train images (~13GB)..."
if [ "$(ls -A $COCO_DIR 2>/dev/null | wc -l)" -gt "1000" ]; then
    echo "  Already downloaded ($( ls $COCO_DIR | wc -l) images found). Skipping."
else
    wget -q --show-progress \
        http://images.cocodataset.org/zips/train2014.zip \
        -O /tmp/train2014.zip
    echo "  Unzipping..."
    unzip -q /tmp/train2014.zip -d $DATA_DIR/coco/
    rm /tmp/train2014.zip
    echo "  COCO images ready: $(ls $COCO_DIR | wc -l) images"
fi

# ── Step 4: Download RefCOCO annotations ───────────────────
echo ""
echo "[4/5] Downloading RefCOCO annotations..."
for DS in refcoco refcoco+ refcocog; do
    DS_DIR="$REFER_DIR/$DS"
    if [ -d "$DS_DIR" ] && [ "$(ls -A $DS_DIR)" ]; then
        echo "  $DS already exists. Skipping."
    else
        mkdir -p $DS_DIR
        ENCODED=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$DS+'))" 2>/dev/null || echo "$DS")
        if [ "$DS" = "refcoco+" ]; then
            wget -q --show-progress \
                "https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip" \
                -O /tmp/${DS}.zip
        elif [ "$DS" = "refcoco" ]; then
            wget -q --show-progress \
                "https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip" \
                -O /tmp/${DS}.zip
        else
            wget -q --show-progress \
                "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip" \
                -O /tmp/${DS}.zip
        fi
        unzip -q /tmp/${DS}.zip -d $REFER_DIR/
        rm /tmp/${DS}.zip
        echo "  $DS ready: $(ls $DS_DIR)"
    fi
done

# ── Step 5: Copy code files into place ─────────────────────
echo ""
echo "[5/5] Setting up code..."

# Copy Python files from workspace root to code directory
cp $WORKSPACE/config.py      $CODE_DIR/
cp $WORKSPACE/train.py       $CODE_DIR/
cp $WORKSPACE/eval.py        $CODE_DIR/
cp $WORKSPACE/ccdg.py        $CODE_DIR/models/
cp $WORKSPACE/contrastive.py $CODE_DIR/losses/
cp $WORKSPACE/phrase_cache.json $CODE_DIR/neg_cache/
cp $WORKSPACE/neg_cache.json    $CODE_DIR/neg_cache/

# Create __init__.py files
touch $CODE_DIR/models/__init__.py
touch $CODE_DIR/losses/__init__.py
touch $CODE_DIR/data/__init__.py

# Create phrase_parser.py stub in data/
# (parser already ran offline — we just need the PhraseComponents class)
cat > $CODE_DIR/data/phrase_parser.py << 'PARSER_EOF'
from dataclasses import dataclass

@dataclass
class PhraseComponents:
    original:      str
    subject:       str
    attribute:     str = ""
    spatial:       str = ""
    has_attribute: bool = False
    has_spatial:   bool = False
PARSER_EOF

echo "  Code structure ready."

# ── Update config paths to match RunPod ────────────────────
python3 - << 'PYEOF'
import re

config_path = '/workspace/ccdg/config.py'
with open(config_path, 'r') as f:
    content = f.read()

# Update all paths to RunPod locations
replacements = {
    'coco_images   = "/data/coco2014/images/train2014"':
        'coco_images   = "/workspace/data/coco/train2014"',
    'refcoco_root  = "/data/refer/data"':
        'refcoco_root  = "/workspace/data/refer"',
    'checkpoint_dir = "/outputs/ccdg/checkpoints"':
        'checkpoint_dir = "/workspace/outputs/ccdg/checkpoints"',
    'neg_cache_dir  = "/outputs/ccdg/neg_cache"':
        'neg_cache_dir  = "/workspace/ccdg/neg_cache"',
}

for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
        print(f'  Updated: {new.strip()}')
    else:
        print(f'  WARNING: could not find: {old.strip()}')

with open(config_path, 'w') as f:
    f.write(content)

print('  config.py paths updated for RunPod.')
PYEOF

# ── Verify everything ───────────────────────────────────────
echo ""
echo "=============================================="
echo "  VERIFICATION"
echo "=============================================="
echo ""
echo "COCO images:    $(ls $COCO_DIR | wc -l) files"
echo "RefCOCO:        $(ls $REFER_DIR/refcoco)"
echo "RefCOCO+:       $(ls $REFER_DIR/refcoco+)"
echo "RefCOCOg:       $(ls $REFER_DIR/refcocog)"
echo "Phrase cache:   $(du -sh $CODE_DIR/neg_cache/phrase_cache.json 2>/dev/null || echo 'MISSING')"
echo "Neg cache:      $(du -sh $CODE_DIR/neg_cache/neg_cache.json 2>/dev/null || echo 'MISSING')"
echo ""
echo "Code structure:"
find $CODE_DIR -name "*.py" | sort
echo ""

# ── Quick import test ───────────────────────────────────────
echo "Running import test..."
python3 - << 'PYEOF'
import sys
sys.path.insert(0, '/workspace/ccdg')
try:
    from models.ccdg import CCDGModel
    from losses.contrastive import CCDGLoss
    from config import Paths, ModelConfig, TrainConfig
    print('  All imports OK.')
except Exception as e:
    print(f'  Import error: {e}')
    exit(1)
PYEOF

# ── Start training ──────────────────────────────────────────
echo ""
echo "=============================================="
echo "  STARTING TRAINING"
echo "=============================================="
echo ""
echo "Logs will print every 100 steps."
echo "Checkpoints saved every 2 epochs to:"
echo "  $OUTPUT_DIR/checkpoints/"
echo ""
echo "To resume if interrupted:"
echo "  python train.py --resume $OUTPUT_DIR/checkpoints/checkpoint_epochN.pth"
echo ""

cd $CODE_DIR
python train.py 2>&1 | tee $OUTPUT_DIR/training_log.txt
