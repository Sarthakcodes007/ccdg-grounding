"""
config_ablation_attr_subj.py
Ablation Run 3 — +L_attr +L_subj: attribute + subject contrastive.

Shows the additional gain from subject-level discrimination
on top of attribute. Spatial loss remains OFF.
Comparing Run 3 vs Run 2 isolates L_subj's contribution.
Comparing Full CCDG vs Run 3 isolates L_spat's contribution.
"""

class Paths:
    coco_images    = "/container/coco/train2014"
    refcoco_root   = "/container/refer"
    checkpoint_dir = "/workspace/outputs/ablation_attr_subj/checkpoints"
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
    lambda_subj    = 0.5   # ON
    lambda_attr    = 0.5   # ON
    lambda_spat    = 0.0   # OFF — spatial still excluded
    triplet_margin = 0.3

class CurriculumConfig:
    phase1_end_epoch   = 4    # epochs 0-3: L1+IoU only
    phase2_end_epoch   = 11   # epochs 4-10: add L_attr
    phase3_start_epoch = 12   # epoch 12+: add L_subj (no L_spat)
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
