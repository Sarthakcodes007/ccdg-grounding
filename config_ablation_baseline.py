"""
config_ablation_baseline.py
Ablation Run 1 — Baseline: L1 + IoU only, zero contrastive losses.

This is the comparison point for the ablation table.
It trains vanilla OWL-ViT fine-tuning with no compositional
supervision whatsoever — the exact same setup as prior work.
"""

# ── Paths (updated automatically by setup script) ──────────
class Paths:
    coco_images    = "/container/coco/train2014"
    refcoco_root   = "/container/refer"
    checkpoint_dir = "/workspace/outputs/ablation_baseline/checkpoints"
    neg_cache_dir  = "/workspace/ccdg/neg_cache"
    owlvit_model   = "google/owlvit-base-patch32"

# ── Data ───────────────────────────────────────────────────
class DataConfig:
    datasets      = ["refcoco", "refcoco+", "refcocog"]
    image_size    = 768
    max_text_len  = 16
    num_negatives = 3
    val_ratio     = 0.2

# ── Model ──────────────────────────────────────────────────
class ModelConfig:
    embed_dim           = 512
    freeze_vision       = False
    freeze_text         = False
    alignment_hidden_dim = 256
    temperature_init    = 0.07

# ── Loss weights ───────────────────────────────────────────
# ALL contrastive weights set to 0 — pure localization only
class LossConfig:
    lambda_l1    = 1.0
    lambda_iou   = 1.0
    lambda_subj  = 0.0   # OFF
    lambda_attr  = 0.0   # OFF
    lambda_spat  = 0.0   # OFF
    triplet_margin = 0.3

# ── Curriculum ─────────────────────────────────────────────
# All phases keep contrastive losses at zero
class CurriculumConfig:
    phase1_end_epoch   = 999   # never activates contrastive
    phase2_end_epoch   = 999
    phase3_start_epoch = 999
    ramp_epochs        = 2

# ── Training ───────────────────────────────────────────────
class TrainConfig:
    batch_size        = 4     # reduced for A40 48GB
    grad_accum_steps  = 8     # effective batch = 32
    total_epochs      = 13
    lr                = 2e-6
    weight_decay      = 1e-4
    lr_min            = 5e-8
    use_amp           = True
    save_every        = 2
    num_workers       = 8
    seed              = 42

# ── Evaluation ─────────────────────────────────────────────
class EvalConfig:
    iou_threshold  = 0.5
    iou_thresholds = [0.25, 0.5, 0.75]
    eval_every     = 2
