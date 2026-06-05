"""
train.py — Full training loop for CCDG.

WHAT THIS FILE DOES:
─────────────────────
Orchestrates the entire training process:
  1. Loads datasets (RefCOCO/+/g) with phrase + negative caches
  2. Builds the model and optimizer
  3. Runs the training loop with:
     - AMP (mixed precision) for speed and memory efficiency
     - Gradient accumulation for effective large batch size
     - Curriculum scheduling (which loss terms are active per epoch)
     - Validation every N epochs
     - Checkpoint saving (best model + resume checkpoint)
  4. Logs everything cleanly so you can monitor training

KEY TRAINING DECISIONS EXPLAINED:
───────────────────────────────────
AdamW optimizer:
  Standard for transformer fine-tuning. The 'W' means weight decay is
  applied correctly (not to bias/LayerNorm params). LR=5e-6 is low
  intentionally — OWL-ViT pretrained weights are good, we want to
  adapt them, not overwrite them.

Cosine LR schedule:
  LR starts at 5e-6, smoothly decays to 1e-7 by epoch 20.
  Prevents overshooting in late training when the model is nearly converged.

AMP (Automatic Mixed Precision):
  Computes forward pass in float16, keeps weights in float32.
  Roughly halves GPU memory and speeds up training ~2x on A100.
  GradScaler handles the numerical scaling to prevent float16 underflow.

Gradient accumulation:
  Effective batch = batch_size × grad_accum_steps = 8 × 4 = 32.
  We accumulate gradients over 4 mini-batches before each optimizer step.
  This simulates a batch of 32 without needing 32 samples in GPU memory.

USAGE:
───────
  # Full training on RunPod A100:
  python train.py

  # Quick smoke test on Kaggle T4 (1 epoch, tiny subset):
  python train.py --smoke_test

  # Resume from checkpoint:
  python train.py --resume /outputs/ccdg/checkpoints/checkpoint_epoch5.pth
"""

import os
import sys
import time
import json
import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.amp import GradScaler, autocast
from transformers import OwlViTProcessor

sys.path.insert(0, str(Path(__file__).parent))
from config import Paths, DataConfig, ModelConfig, TrainConfig, EvalConfig, CurriculumConfig
from models.ccdg import CCDGModel
from losses.contrastive import CCDGLoss
from data.refcoco_dataset import RefCOCODataset
from eval import evaluate   # we write this next


def parse_args():
    parser = argparse.ArgumentParser(description='Train CCDG')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run 1 epoch on 200 samples to verify everything works')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override total epochs from config')
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# DATASET BUILDER
# ──────────────────────────────────────────────────────────────

def build_datasets(
    processor,
    neg_cache: dict,
    phrase_cache: dict,
    cfg_data: DataConfig,
    cfg_paths: Paths,
    smoke_test: bool = False,
):
    """
    Build train and val datasets by combining RefCOCO/+/g.

    WHY COMBINE ALL THREE?
    ───────────────────────
    Each dataset tests a different skill:
    - RefCOCO:  general grounding, balanced attributes and spatial
    - RefCOCO+: attribute-heavy (no spatial words) → stress-tests L_attr
    - RefCOCOg: long complex phrases → tests compositional reasoning

    Training on all three gives the model exposure to all difficulty levels
    and prevents overfitting to one distribution.

    For evaluation we use the OFFICIAL splits of each dataset separately,
    which lets us report per-dataset numbers in the paper's ablation table.
    """
    dataset_configs = [
        ('refcoco',  'refs(unc).p',    'train'),
        ('refcoco+', 'refs(unc).p',    'train'),
        ('refcocog', 'refs(umd).p',    'train'),
    ]
    val_configs = [
        ('refcoco',  'refs(unc).p',    'val'),
        ('refcoco+', 'refs(unc).p',    'val'),
        ('refcocog', 'refs(umd).p',    'val'),
    ]

    train_datasets, val_datasets = [], []

    for ds_name, _, split in dataset_configs:
        ds = RefCOCODataset(
            dataset_name=ds_name,
            split='train',
            processor=processor,
            neg_cache=neg_cache,
            phrase_cache=phrase_cache,
            cfg_data=cfg_data,
            cfg_paths=cfg_paths,
        )
        train_datasets.append(ds)

    for ds_name, _, split in val_configs:
        ds = RefCOCODataset(
            dataset_name=ds_name,
            split='val',
            processor=processor,
            neg_cache=None,       # no negatives needed for eval
            phrase_cache=phrase_cache,
            cfg_data=cfg_data,
            cfg_paths=cfg_paths,
        )
        val_datasets.append(ds)

    train_combined = ConcatDataset(train_datasets)
    val_combined   = ConcatDataset(val_datasets)

    # Smoke test: use tiny subset to verify the pipeline
    if smoke_test:
        train_combined = Subset(train_combined, range(200))
        val_combined   = Subset(val_combined,   range(50))
        print(f"[SMOKE TEST] Using 200 train / 50 val samples")

    return train_combined, val_combined


# ──────────────────────────────────────────────────────────────
# COLLATE FUNCTION
# ──────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Stack a list of sample dicts into a batched dict.

    WHY A CUSTOM COLLATE?
    ──────────────────────
    PyTorch's default collate handles simple tensors fine, but our batch
    contains mixed types (tensors + strings + ints). We handle tensors
    manually and keep strings as lists.

    Also: has_attribute and has_spatial are bool tensors — we stack them
    into [B] bool tensors here so the loss function can use them as masks.
    """
    tensor_keys = [
        'pixel_values',
        'input_ids_full', 'attn_mask_full',
        'input_ids_subj', 'attn_mask_subj',
        'input_ids_attr', 'attn_mask_attr',
        'input_ids_spat', 'attn_mask_spat',
        'gt_box',
        'neg_attr_ids', 'neg_attr_masks',
        'neg_subj_ids', 'neg_subj_masks',
        'neg_spat_ids', 'neg_spat_masks',
        'has_attribute', 'has_spatial',
    ]
    str_keys = ['sent']
    int_keys = ['image_id', 'ann_id']

    result = {}
    for k in tensor_keys:
        if k in batch[0]:
            result[k] = torch.stack([s[k] for s in batch])
    for k in str_keys + int_keys:
        if k in batch[0]:
            result[k] = [s[k] for s in batch]

    return result


# ──────────────────────────────────────────────────────────────
# OPTIMIZER + SCHEDULER
# ──────────────────────────────────────────────────────────────

def build_optimizer_and_scheduler(model, cfg_train: TrainConfig, num_steps: int):
    """
    Build AdamW optimizer with cosine LR schedule.

    PARAMETER GROUPS:
    ──────────────────
    We use different learning rates for different parts of the model:
    - OWL-ViT pretrained weights: lr=5e-6 (low — preserve pretrained knowledge)
    - Our new components (visual_extractor, box_head): lr=1e-4 (higher — learn from scratch)

    This is called "discriminative fine-tuning" and prevents the pretrained
    backbone from being overwritten by the randomly-initialized new heads.

    WHY NO WEIGHT DECAY ON BIAS/LAYERNORM?
    ────────────────────────────────────────
    Weight decay regularizes by shrinking weights toward zero. Applying it
    to bias terms and LayerNorm parameters hurts performance because these
    parameters calibrate the scale of activations — they should be free to
    find whatever value works best.
    """
    # Separate pretrained params from new params
    pretrained_params, new_params = [], []
    no_decay_names = {'bias', 'LayerNorm.weight', 'layer_norm.weight'}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new = ('visual_extractor' in name or 'box_head' in name)
        no_decay = any(nd in name for nd in no_decay_names)

        if is_new:
            new_params.append((name, param, no_decay))
        else:
            pretrained_params.append((name, param, no_decay))

    param_groups = [
        # Pretrained backbone — low LR, with weight decay
        {
            'params': [p for _, p, nd in pretrained_params if not nd],
            'lr': cfg_train.lr,
            'weight_decay': cfg_train.weight_decay,
            'name': 'pretrained_decay',
        },
        # Pretrained backbone — low LR, no weight decay (bias/norm)
        {
            'params': [p for _, p, nd in pretrained_params if nd],
            'lr': cfg_train.lr,
            'weight_decay': 0.0,
            'name': 'pretrained_nodecay',
        },
        # New components — higher LR, with weight decay
        {
            'params': [p for _, p, nd in new_params if not nd],
            'lr': cfg_train.lr * 20,   # 20× higher for new components
            'weight_decay': cfg_train.weight_decay,
            'name': 'new_decay',
        },
        # New components — higher LR, no weight decay
        {
            'params': [p for _, p, nd in new_params if nd],
            'lr': cfg_train.lr * 20,
            'weight_decay': 0.0,
            'name': 'new_nodecay',
        },
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    optimizer = torch.optim.AdamW(param_groups)

    # Cosine annealing: LR decays smoothly from initial value to lr_min
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=cfg_train.lr_min,
    )

    return optimizer, scheduler


# ──────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ──────────────────────────────────────────────────────────────

def save_checkpoint(
    model, optimizer, scheduler, scaler,
    epoch, best_val_loss, cfg_paths, suffix=''
):
    """Save training state for resuming."""
    Path(cfg_paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    state = {
        'epoch':         epoch,
        'model':         model.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'scheduler':     scheduler.state_dict(),
        'scaler':        scaler.state_dict(),
        'best_val_loss': best_val_loss,
    }

    path = os.path.join(
        cfg_paths.checkpoint_dir,
        f'checkpoint_epoch{epoch}{suffix}.pth'
    )
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """Load training state from checkpoint."""
    print(f"Loading checkpoint from {path}...")
    state = torch.load(path, map_location='cpu')

    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])

    print(f"  Resumed from epoch {state['epoch']}")
    return state['epoch'], state['best_val_loss']


def save_best_model(model, processor, cfg_paths):
    """Save the best model weights + processor for inference."""
    best_dir = os.path.join(cfg_paths.checkpoint_dir, 'best_model')
    Path(best_dir).mkdir(parents=True, exist_ok=True)
    model.owlvit.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)
    torch.save(model.state_dict(), os.path.join(best_dir, 'ccdg_weights.pth'))
    print(f"  Best model saved to {best_dir}")


# ──────────────────────────────────────────────────────────────
# TRAINING STEP
# ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler,
    epoch, cfg_train, device,
):
    """
    Run one full epoch of training.

    GRADIENT ACCUMULATION EXPLAINED:
    ──────────────────────────────────
    We want an effective batch size of 32, but GPU memory limits us to
    batch_size=8. The solution: accumulate gradients over 4 steps before
    calling optimizer.step(). This is mathematically equivalent to a
    batch of 32.

    The key: divide the loss by grad_accum_steps before backward pass.
    This ensures the accumulated gradient has the same scale as if we
    had processed the full effective batch at once.

    AMP EXPLAINED:
    ──────────────
    Inside `autocast()`, PyTorch automatically casts certain operations
    to float16. Matrix multiplications (the expensive part of transformers)
    run in float16 — 2x faster on modern GPUs. Loss and weight updates
    stay in float32 for numerical stability.

    GradScaler multiplies the loss by a large factor before backward()
    to prevent float16 gradients from underflowing to zero. It then
    unscales them before the optimizer step.
    """
    model.train()
    criterion.train()

    accum_steps = cfg_train.grad_accum_steps
    total_loss  = 0.0
    loss_components = {'l1': 0, 'iou': 0, 'attr': 0, 'subj': 0, 'spat': 0}
    num_batches = 0
    t_start     = time.time()

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        # Move all tensors to GPU
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # ── Forward pass with AMP ────────────────────────────────
        with autocast('cuda', enabled=cfg_train.use_amp):
            model_out = model(batch)
            loss_dict = criterion(model_out, batch, epoch)
            # Divide by accumulation steps — critical for correct gradient scale
            loss = loss_dict['total'] / accum_steps

        # ── Backward pass with gradient scaling ─────────────────
        scaler.scale(loss).backward()

        # ── Optimizer step every accum_steps batches ─────────────
        if (step + 1) % accum_steps == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Gradient clipping: prevents exploding gradients
            # Clips the global norm of all gradients to max_norm=1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # ── Accumulate stats ─────────────────────────────────────
        total_loss += loss_dict['total'].item()
        for k in loss_components:
            loss_components[k] += loss_dict[k].item()
        num_batches += 1

        # ── Progress logging every 100 steps ────────────────────
        if (step + 1) % 100 == 0:
            elapsed  = time.time() - t_start
            avg_loss = total_loss / num_batches
            w = loss_dict['curriculum']
            print(
                f"  Epoch {epoch} | step {step+1}/{len(loader)} | "
                f"loss={avg_loss:.4f} | "
                f"l1={loss_components['l1']/num_batches:.3f} | "
                f"iou={loss_components['iou']/num_batches:.3f} | "
                f"attr={loss_components['attr']/num_batches:.3f}(w={w['attr']:.2f}) | "
                f"subj={loss_components['subj']/num_batches:.3f}(w={w['subj']:.2f}) | "
                f"elapsed={elapsed:.0f}s"
            )

    # Epoch summary
    avg = {k: v / num_batches for k, v in loss_components.items()}
    avg['total'] = total_loss / num_batches
    avg['time']  = time.time() - t_start

    return avg


# ──────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Configs ────────────────────────────────────────────────
    cfg_paths = Paths()
    cfg_data  = DataConfig()
    cfg_model = ModelConfig()
    cfg_train = TrainConfig()
    cfg_eval  = EvalConfig()
    cfg_curr  = CurriculumConfig()

    if args.epochs:
        cfg_train.total_epochs = args.epochs

    # ── Reproducibility ────────────────────────────────────────
    torch.manual_seed(cfg_train.seed)

    # ── Device ─────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Load caches ─────────────────────────────────────────────
    print("\nLoading preprocessing caches...")

    with open(os.path.join(cfg_paths.neg_cache_dir, 'neg_cache.json')) as f:
        raw_neg = json.load(f)
    neg_cache = {
        tuple(k.split('__', 1)): v
        for k, v in raw_neg.items()
    }
    # Fix ann_id back to int
    neg_cache = {(k[0], int(k[1])): v for k, v in neg_cache.items()}
    print(f"  Neg cache:    {len(neg_cache):,} entries")

    with open(os.path.join(cfg_paths.neg_cache_dir, 'phrase_cache.json')) as f:
        raw_phrase = json.load(f)

    # Reconstruct PhraseComponents from JSON
    from data.phrase_parser import PhraseComponents
    phrase_cache = {}
    for k, v in raw_phrase.items():
        ds_name, sent_id = k.split('__', 1)
        phrase_cache[(ds_name, int(sent_id))] = PhraseComponents(**v)
    print(f"  Phrase cache: {len(phrase_cache):,} entries")

    # ── Processor ──────────────────────────────────────────────
    print(f"\nLoading OWL-ViT processor from {cfg_paths.owlvit_model}...")
    processor = OwlViTProcessor.from_pretrained(cfg_paths.owlvit_model)

    # ── Datasets + DataLoaders ──────────────────────────────────
    print("\nBuilding datasets...")
    train_ds, val_ds = build_datasets(
        processor, neg_cache, phrase_cache,
        cfg_data, cfg_paths,
        smoke_test=args.smoke_test,
    )
    print(f"  Train: {len(train_ds):,} samples")
    print(f"  Val:   {len(val_ds):,} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_train.batch_size,
        shuffle=True,
        num_workers=cfg_train.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,    # avoids issues with small final batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_train.batch_size * 2,   # larger batch for eval (no grad)
        shuffle=False,
        num_workers=cfg_train.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Model ───────────────────────────────────────────────────
    print("\nBuilding model...")
    model = CCDGModel(cfg_model, cfg_paths).to(device)

    # ── Loss ────────────────────────────────────────────────────
    criterion = CCDGLoss(cfg_curr=cfg_curr)

    # ── Optimizer + Scheduler ───────────────────────────────────
    steps_per_epoch = len(train_loader) // cfg_train.grad_accum_steps
    total_steps     = steps_per_epoch * cfg_train.total_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg_train, total_steps
    )
    print(f"\n  Steps per epoch:  {steps_per_epoch}")
    print(f"  Total steps:      {total_steps}")

    # ── AMP Scaler ──────────────────────────────────────────────
    scaler = GradScaler('cuda', enabled=cfg_train.use_amp)

    # ── Resume from checkpoint ───────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float('inf')

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch += 1   # continue from next epoch

    # ── Training Loop ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting training: {cfg_train.total_epochs} epochs")
    print(f"Effective batch size: "
          f"{cfg_train.batch_size} × {cfg_train.grad_accum_steps} = "
          f"{cfg_train.batch_size * cfg_train.grad_accum_steps}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, cfg_train.total_epochs):
        print(f"\nEpoch {epoch}/{cfg_train.total_epochs-1}")
        print(f"  Curriculum: {get_curriculum_info(epoch, cfg_curr)}")

        # ── Train ────────────────────────────────────────────────
        train_stats = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler,
            epoch, cfg_train, device,
        )

        print(f"\n  Train summary:")
        print(f"    total={train_stats['total']:.4f} | "
              f"l1={train_stats['l1']:.4f} | "
              f"iou={train_stats['iou']:.4f} | "
              f"attr={train_stats['attr']:.4f} | "
              f"subj={train_stats['subj']:.4f} | "
              f"spat={train_stats['spat']:.4f} | "
              f"time={train_stats['time']:.0f}s")

        # ── Validate ─────────────────────────────────────────────
        if (epoch + 1) % cfg_eval.eval_every == 0 or epoch == 0:
            print(f"\n  Running validation...")
            val_stats = evaluate(
                model, val_loader, criterion, epoch, device, cfg_eval
            )
            print(f"  Val: loss={val_stats['loss']:.4f} | "
                  f"Acc@0.5={val_stats['acc_50']:.4f} | "
                  f"Acc@0.25={val_stats['acc_25']:.4f}")

            # Track best model
            if val_stats['loss'] < best_val_loss:
                best_val_loss = val_stats['loss']
                save_best_model(model, processor, cfg_paths)
                print(f"  ★ New best val loss: {best_val_loss:.4f}")

        # ── Save checkpoint ──────────────────────────────────────
        if (epoch + 1) % cfg_train.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, best_val_loss, cfg_paths,
            )

    print(f"\n{'='*60}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {cfg_paths.checkpoint_dir}/best_model/")


def get_curriculum_info(epoch, cfg):
    """Human-readable string showing what losses are active."""
    from losses.contrastive import get_curriculum_weights
    w = get_curriculum_weights(epoch, cfg)
    active = ['L1+IoU']
    if w['attr'] > 0:  active.append(f"L_attr(w={w['attr']:.2f})")
    if w['subj'] > 0:  active.append(f"L_subj(w={w['subj']:.2f})")
    if w['spat'] > 0:  active.append(f"L_spat(w={w['spat']:.2f})")
    return ' + '.join(active)


if __name__ == '__main__':
    main()
