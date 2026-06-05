"""
eval.py — Evaluation with Acc@0.5 across all RefCOCO splits.

WHAT THIS FILE DOES:
─────────────────────
Two modes:

1. eval() — called during training every N epochs.
   Runs on the combined validation set.
   Returns loss + Acc@0.5 quickly.

2. full_eval() — called once after training is complete.
   Runs on ALL official splits of all three datasets:
     RefCOCO:  val, testA, testB
     RefCOCO+: val, testA, testB
     RefCOCOg: val, test
   This is the table that goes into the paper.

WHAT IS Acc@0.5?
─────────────────
For each test sample:
  1. Model predicts a bounding box.
  2. Compute IoU between predicted box and GT box.
  3. If IoU >= 0.5 → correct. Else → wrong.
Acc@0.5 = (number correct) / (total samples)

This is the standard metric for REC in all baseline papers
(TransVG, MDETR, OFA) so our numbers are directly comparable.

We also report Acc@0.25 and Acc@0.75:
  Acc@0.25 = lenient (partial overlap counts)
  Acc@0.75 = strict (tight overlap required)
"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from config import Paths, DataConfig, EvalConfig, TrainConfig
from losses.contrastive import CCDGLoss, box_cxcywh_to_xyxy


def compute_iou_batch(pred_boxes: torch.Tensor,
                      gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample IoU between predicted and GT boxes.

    Parameters
    ----------
    pred_boxes : [B, 4]  (cx, cy, w, h) normalized
    gt_boxes   : [B, 4]  (cx, cy, w, h) normalized

    Returns
    -------
    iou : [B]  per-sample IoU values
    """
    # Convert to (x1, y1, x2, y2)
    pred = box_cxcywh_to_xyxy(pred_boxes.clamp(0, 1))
    gt   = box_cxcywh_to_xyxy(gt_boxes.clamp(0, 1))

    # Intersection
    inter_x1 = torch.max(pred[:, 0], gt[:, 0])
    inter_y1 = torch.max(pred[:, 1], gt[:, 1])
    inter_x2 = torch.min(pred[:, 2], gt[:, 2])
    inter_y2 = torch.min(pred[:, 3], gt[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h

    # Areas
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    gt_area   = (gt[:, 2]   - gt[:, 0])   * (gt[:, 3]   - gt[:, 1])

    union = pred_area + gt_area - intersection + 1e-6  # epsilon for stability

    return intersection / union   # [B]


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    epoch: int,
    device,
    cfg_eval: EvalConfig = None,
) -> dict:
    """
    Evaluate on a DataLoader. Called during training.

    Returns
    -------
    dict with keys: loss, acc_25, acc_50, acc_75
    """
    if cfg_eval is None:
        cfg_eval = EvalConfig()

    model.eval()

    total_loss   = 0.0
    all_iou      = []
    num_batches  = 0

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with autocast('cuda'):
            # During eval we use the full forward pass
            # but don't need hard negatives
            # Use predict() for clean inference
            pred_boxes = model.predict(
                pixel_values   = batch['pixel_values'],
                input_ids      = batch['input_ids_full'],
                attention_mask = batch['attn_mask_full'],
            )   # [B, 4]

        # Compute IoU for each sample
        iou = compute_iou_batch(pred_boxes.cpu(), batch['gt_box'].cpu())
        all_iou.append(iou)

        # Also compute loss for monitoring (optional during eval)
        try:
            with autocast('cuda'):
                model_out = model(batch)
                loss_dict = criterion(model_out, batch, epoch)
            total_loss += loss_dict['total'].item()
        except Exception:
            pass   # skip loss if batch doesn't have all required fields

        num_batches += 1

    # Stack all IoU values
    all_iou = torch.cat(all_iou)   # [N_total]

    # Compute accuracy at each threshold
    results = {
        'loss':   total_loss / max(num_batches, 1),
        'acc_25': (all_iou >= 0.25).float().mean().item(),
        'acc_50': (all_iou >= 0.50).float().mean().item(),
        'acc_75': (all_iou >= 0.75).float().mean().item(),
        'mean_iou': all_iou.mean().item(),
        'n_samples': len(all_iou),
    }

    return results


@torch.no_grad()
def full_eval(
    model,
    processor,
    neg_cache: dict,
    phrase_cache: dict,
    cfg_paths: Paths = None,
    cfg_data: DataConfig = None,
    cfg_eval: EvalConfig = None,
    cfg_train: TrainConfig = None,
    device = None,
) -> dict:
    """
    Full evaluation across ALL official splits.
    Run this ONCE after training is complete to get paper numbers.

    Returns a nested dict:
    {
      'refcoco':  {'val': {...}, 'testA': {...}, 'testB': {...}},
      'refcoco+': {'val': {...}, 'testA': {...}, 'testB': {...}},
      'refcocog': {'val': {...}, 'test':  {...}},
    }
    """
    from data.refcoco_dataset import RefCOCODataset
    from train import collate_fn

    if cfg_paths is None: cfg_paths = Paths()
    if cfg_data  is None: cfg_data  = DataConfig()
    if cfg_eval  is None: cfg_eval  = EvalConfig()
    if cfg_train is None: cfg_train = TrainConfig()
    if device    is None: device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # All splits to evaluate on
    eval_splits = {
        'refcoco':  ['val', 'testA', 'testB'],
        'refcoco+': ['val', 'testA', 'testB'],
        'refcocog': ['val', 'test'],
    }

    all_results = {}
    criterion   = CCDGLoss()

    print("\n" + "="*60)
    print("FULL EVALUATION ACROSS ALL SPLITS")
    print("="*60)

    for ds_name, splits in eval_splits.items():
        all_results[ds_name] = {}

        for split in splits:
            print(f"\n  {ds_name} / {split}...")

            ds = RefCOCODataset(
                dataset_name = ds_name,
                split        = split,
                processor    = processor,
                neg_cache    = None,      # no negatives for eval
                phrase_cache = phrase_cache,
                cfg_data     = cfg_data,
                cfg_paths    = cfg_paths,
            )

            loader = DataLoader(
                ds,
                batch_size  = cfg_train.batch_size * 2,
                shuffle     = False,
                num_workers = cfg_train.num_workers,
                collate_fn  = collate_fn,
                pin_memory  = (device.type == 'cuda'),
            )

            results = evaluate(
                model, loader, criterion, epoch=0, device=device, cfg_eval=cfg_eval
            )
            all_results[ds_name][split] = results

            print(f"    Acc@0.25={results['acc_25']:.4f} | "
                  f"Acc@0.50={results['acc_50']:.4f} | "
                  f"Acc@0.75={results['acc_75']:.4f} | "
                  f"n={results['n_samples']}")

    # Print paper-ready table
    print("\n" + "="*60)
    print("RESULTS TABLE (Acc@0.5)")
    print("="*60)
    print(f"{'Dataset':<12} {'Split':<8} {'Acc@0.25':>10} {'Acc@0.50':>10} {'Acc@0.75':>10}")
    print("-"*52)
    for ds_name, splits in all_results.items():
        for split, res in splits.items():
            print(f"{ds_name:<12} {split:<8} "
                  f"{res['acc_25']:>10.2f} "
                  f"{res['acc_50']:>10.2f} "
                  f"{res['acc_75']:>10.2f}")
    print("="*60)

    # Save results to JSON
    out_path = os.path.join(cfg_paths.checkpoint_dir, 'eval_results.json')
    Path(cfg_paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == '__main__':
    """
    Run full evaluation on a saved checkpoint.
    Usage: python eval.py --checkpoint /path/to/best_model/
    """
    import argparse
    from transformers import OwlViTProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    cfg_paths = Paths()
    cfg_data  = DataConfig()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    from models.ccdg import CCDGModel
    model = CCDGModel()
    weights = torch.load(
        os.path.join(args.checkpoint, 'ccdg_weights.pth'),
        map_location='cpu'
    )
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    processor = OwlViTProcessor.from_pretrained(args.checkpoint)

    # Load caches
    with open(os.path.join(cfg_paths.neg_cache_dir, 'phrase_cache.json')) as f:
        raw = json.load(f)
    from data.phrase_parser import PhraseComponents
    phrase_cache = {
        tuple(k.split('__', 1)): PhraseComponents(**v)
        for k, v in raw.items()
    }
    phrase_cache = {(k[0], int(k[1])): v for k, v in phrase_cache.items()}

    full_eval(model, processor, {}, phrase_cache,
              cfg_paths, cfg_data, device=device)
