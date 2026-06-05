"""
config_ablation_attr.py
Ablation Run 2 — +L_attr: adds attribute contrastive loss only.

Shows the isolated contribution of colour/texture/action
discrimination. Compares against baseline to prove L_attr helps.
Subject and spatial losses remain OFF.
"""

class Paths:
    coco_images    = "/container/coco/train2014"
    refcoco_root   = "/container/refer"
    checkpoint_dir = "/workspace/outputs/ablation_attr/checkpoints"
    neg_cache_dir  = "/workspace/ccdg/neg_cache"
    owlvit_model   = "google/owlvit-base-patch32"

class DataConfig:
    datasets      = ["refcoco", "refcoco+", "refcocog"]
    image_size    = 768
    max_text_len  = 16
    num_negatives = 3
    val_ratio     = 0.2

class ModelConfig:
    embed_dim            = 512
    freeze_vision        = False
    freeze_text          = False
    alignment_hidden_dim = 256
    temperature_init     = 0.07

class LossConfig:
    lambda_l1      = 1.0
    lambda_iou     = 1.0
    lambda_subj    = 0.0   # OFF
    lambda_attr    = 0.5   # ON — attribute contrastive only
    lambda_spat    = 0.0   # OFF
    triplet_margin = 0.3

class CurriculumConfig:
    phase1_end_epoch   = 4    # epochs 0-3: L1+IoU only
    phase2_end_epoch   = 999  # attr activates at epoch 4, subj/spat never
    phase3_start_epoch = 999  # never activates subj/spat
    ramp_epochs        = 2

class TrainConfig:
    batch_size       = 4
    grad_accum_steps = 8
    total_epochs     = 13
    lr               = 2e-6
    weight_decay     = 1e-4
    lr_min           = 5e-8
    use_amp          = True
    save_every       = 2
    num_workers      = 8
    seed             = 42

class EvalConfig:
    iou_threshold  = 0.5
    iou_thresholds = [0.25, 0.5, 0.75]
    eval_every     = 2
