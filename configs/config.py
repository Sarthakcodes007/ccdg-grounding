"""
config.py — Central configuration for CCDG training.

Every hyperparameter lives here. The reason we do this instead of
hardcoding numbers in individual files: when you run ablations (which
the paper requires), you change ONE number here and every file picks
it up automatically. No hunting through 6 files to find where you
wrote 0.5.
"""

import os

# ─────────────────────────────────────────────
# PATHS  (change these for your RunPod setup)
# ─────────────────────────────────────────────
class Paths:
    # COCO 2014 images — download from https://cocodataset.org
    coco_images   = "/data/coco2014/images/train2014"

    # RefCOCO annotation files (downloaded via refer API)
    # Each dataset has its own folder with refs(unc).p and instances.json
    refcoco_root  = "/data/refer/data"

    # Where we save model checkpoints
    checkpoint_dir = "/outputs/ccdg/checkpoints"

    # Where we save hard negative cache (generated once, reused)
    neg_cache_dir  = "/outputs/ccdg/neg_cache"

    # Pretrained OWL-ViT weights from HuggingFace
    owlvit_model   = "google/owlvit-base-patch32"


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
class DataConfig:
    # Which datasets to train on
    # RefCOCO  → general grounding
    # RefCOCO+ → attribute-heavy (no spatial words) — most important for us
    # RefCOCOg → longer, more complex sentences
    datasets = ["refcoco", "refcoco+", "refcocog"]

    # Image resize target. OWL-ViT was pretrained at 768×768.
    # Keeping it the same means pretrained features transfer cleanly.
    image_size = 768

    # Text tokenizer max length. OWL-ViT uses 16 tokens.
    # Phrases in RefCOCO are short enough that 16 covers >99% of them.
    max_text_len = 16

    # How many hard negatives to pair with each positive during training.
    # More = stronger contrastive signal but slower batches.
    num_negatives = 3

    # Train/val split ratio (RefCOCO already has official splits,
    # but for RefCOCOg which has no official test, we use this)
    val_ratio = 0.2


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class ModelConfig:
    # Shared embedding dimension (OWL-ViT projects everything here)
    embed_dim = 512

    # Whether to freeze the vision encoder backbone.
    # We set False — full fine-tuning adapts both vision and text
    # for tight localization, which your project already showed works.
    freeze_vision = False
    freeze_text   = False

    # The alignment module maps component text embeddings to visual
    # features. This MLP projects 512 → 512 with one hidden layer.
    alignment_hidden_dim = 256

    # Temperature for cosine similarity scoring (learnable in OWL-ViT)
    # Lower temperature = sharper, more confident predictions
    temperature_init = 0.07


# ─────────────────────────────────────────────
# LOSS WEIGHTS  (λ values from the paper)
# These are starting points — ablation will tune them.
# ─────────────────────────────────────────────
class LossConfig:
    # Base localization losses (always active)
    lambda_l1  = 1.0   # L1 box regression
    lambda_iou = 1.0   # 1 - IoU

    # Compositional contrastive losses
    # Subject loss: pushes wrong-category regions away
    lambda_subj = 0.5

    # Attribute loss: pushes wrong-colour/texture patches away
    # Slightly higher because RefCOCO+ is attribute-heavy
    lambda_attr = 0.5

    # Spatial loss: pushes wrong-position regions away
    # Slightly lower because spatial is harder to learn stably
    lambda_spat = 0.3

    # Triplet margin: the minimum gap we enforce between
    # d(query, correct) and d(query, decoy).
    # Tune over {0.2, 0.3, 0.5} in ablation.
    triplet_margin = 0.3


# ─────────────────────────────────────────────
# CURRICULUM SCHEDULE
# This controls WHEN each loss term activates.
# The key insight: don't add hard losses before the model
# has learned basic localization — it destabilises training.
# ─────────────────────────────────────────────
class CurriculumConfig:
    # Phase 1: basic localization only (L1 + IoU)
    phase1_end_epoch = 4     # epochs 0–3

    # Phase 2: add attribute-level contrastive loss
    phase2_end_epoch = 11    # epochs 4–10

    # Phase 3: add subject + spatial contrastive losses
    # (full model from epoch 12 onwards)
    phase3_start_epoch = 12

    # Warmup ramp: instead of switching loss terms on like a switch,
    # we linearly ramp lambda from 0 to its target value over
    # this many epochs. Smoother = more stable training.
    ramp_epochs = 2


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
class TrainConfig:
    # On RunPod A100 (80GB), batch_size=8 fits comfortably.
    # On A6000 (48GB), use batch_size=4.
    # Effective batch = batch_size × grad_accum_steps
    batch_size = 8
    grad_accum_steps = 4   # effective batch = 32

    total_epochs = 20

    # AdamW with a low learning rate: OWL-ViT pretrained weights
    # are good — we want to fine-tune, not overwrite.
    lr = 5e-6
    weight_decay = 1e-4

    # Cosine LR schedule: LR starts at lr, decays to lr_min by epoch 20
    lr_min = 1e-7

    # Mixed precision (AMP) — halves GPU memory, ~2× speed.
    # Safe to use here since our losses are numerically stable.
    use_amp = True

    # Save a checkpoint every N epochs
    save_every = 2

    # Number of DataLoader workers per GPU
    num_workers = 4

    # Random seed for reproducibility
    seed = 42


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
class EvalConfig:
    # Primary metric: Acc@0.5
    # A prediction is correct if IoU(pred_box, gt_box) >= 0.5
    iou_threshold = 0.5

    # We also report Acc@0.25 and Acc@0.75 for completeness
    iou_thresholds = [0.25, 0.5, 0.75]

    # Evaluate on validation set every N epochs during training
    eval_every = 2
