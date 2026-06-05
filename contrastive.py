"""
losses/contrastive.py — Multi-level compositional contrastive loss.

FULL LOSS FUNCTION:
───────────────────
L_total = L1 + (1 - IoU)
        + α(t) × [λ₁·L_subj + λ₂·L_attr + λ₃·L_spat]

Where α(t) is the curriculum weight that ramps each term in
progressively as training proceeds.

EACH TERM EXPLAINED:
─────────────────────
L1      : Mean absolute error between predicted and GT box coordinates.
          Penalises being far from the right location.

IoU     : 1 minus the Intersection over Union of predicted and GT box.
          Penalises poor overlap even when L1 is small.
          (Two boxes can have similar L1 distance but very different IoU
          if one is shifted vs the other being the wrong size.)

L_subj  : Triplet loss on subject component.
          Pushes f_subj (subject phrase embedding) TOWARD region-level
          visual features of the correct referent, AWAY from embeddings
          of subject-swapped phrases ("man" → "woman").

L_attr  : Triplet loss on attribute component.
          Pushes f_attr TOWARD attribute-aware visual features,
          AWAY from colour/texture-swapped phrases ("red" → "blue").

L_spat  : Triplet loss on spatial component.
          Pushes f_spat TOWARD scene-level features,
          AWAY from direction-swapped phrases ("left" → "right").

TRIPLET LOSS FORMULA:
──────────────────────
L_triplet = max(0, margin − cos(anchor, positive) + cos(anchor, negative))

Where:
  anchor   = visual feature at the matching scale
  positive = text embedding of the CORRECT component phrase
  negative = text embedding of the HARD NEGATIVE component phrase

Intuition: force the gap between (anchor, positive) and (anchor, negative)
similarity to be at least `margin`. If it already is, loss = 0.

WHY COSINE SIMILARITY (not L2 distance)?
──────────────────────────────────────────
All our embeddings are L2-normalized (done in the model's encode_text).
For normalized vectors, cosine similarity is equivalent to negative
squared Euclidean distance up to a constant, but cosine is more
numerically stable and scales naturally to [-1, 1].

CURRICULUM SCHEDULE:
─────────────────────
Phase 1 (epochs 0-3):   α=0            — only L1+IoU
Phase 2 (epochs 4-10):  α ramps 0→1   — add L_attr
Phase 3 (epochs 11+):   α=1           — add L_subj + L_spat

Why? In early training the model hasn't learned to localize at all.
Adding contrastive losses before localization stabilises would push
embeddings in random directions and destabilise training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LossConfig, CurriculumConfig


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) → (x1, y1, x2, y2).

    WHY WE NEED THIS:
    torchvision's box_iou expects (x1,y1,x2,y2) format.
    Our model outputs and GT boxes are in (cx,cy,w,h).
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def l1_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Mean L1 distance between predicted and GT box coordinates.

    pred_boxes : [B, 4]  (cx, cy, w, h) normalized
    gt_boxes   : [B, 4]  (cx, cy, w, h) normalized

    Returns scalar loss.
    """
    return F.l1_loss(pred_boxes, gt_boxes, reduction='mean')


def iou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    1 - IoU loss averaged over the batch.

    WHY IoU ON TOP OF L1?
    ─────────────────────
    L1 treats all coordinate errors equally — being off by 0.1 in cx
    is penalised the same as being off by 0.1 in w. But a small shift
    in a tiny box is catastrophic (IoU drops to 0) while the same shift
    in a large box barely matters. IoU loss captures this scale-sensitivity.

    Combined L1 + IoU is the standard approach in modern detection papers.
    """
    # Convert to xyxy for torchvision's box_iou
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes.clamp(0, 1))
    gt_xyxy   = box_cxcywh_to_xyxy(gt_boxes.clamp(0, 1))

    # box_iou expects [N, 4] and returns [N, N] pairwise IoU matrix
    # We want the diagonal (each prediction vs its own GT)
    iou_matrix = box_iou(pred_xyxy, gt_xyxy)      # [B, B]
    iou_diag   = iou_matrix.diagonal()             # [B]

    return (1.0 - iou_diag).mean()


def triplet_loss(
    anchor:   torch.Tensor,   # [B, 512] — visual feature at matching scale
    positive: torch.Tensor,   # [B, 512] — correct component phrase embedding
    negative: torch.Tensor,   # [B, K, 512] — hard negative embeddings
    margin:   float = 0.3,
) -> torch.Tensor:
    """
    Batch hard triplet loss with multiple negatives per anchor.

    For each sample we have K hard negatives. We take the HARDEST
    negative (highest cosine similarity to anchor) for the loss.
    This is called "batch hard" mining and is more stable than
    random negative selection.

    Formula: L = mean over batch of max(0, margin - sim_pos + sim_hardneg)

    Parameters
    ----------
    anchor   : [B, 512]    visual features (already L2-normalized)
    positive : [B, 512]    positive text embeddings (L2-normalized)
    negative : [B, K, 512] hard negative embeddings (L2-normalized)
    margin   : float       minimum required gap between pos/neg similarity

    Returns
    -------
    loss : scalar tensor
    """
    # Similarity between anchor and positive phrase
    # Both normalized → dot product = cosine similarity ∈ [-1, 1]
    sim_pos = (anchor * positive).sum(dim=-1)   # [B]

    # Similarity between anchor and each of K negatives
    # anchor: [B, 512] → [B, 1, 512] for broadcasting
    sim_neg = torch.einsum(
        'bd,bkd->bk',
        anchor,    # [B, 512]
        negative,  # [B, K, 512]
    )              # [B, K]

    # Take the HARDEST negative (highest similarity = hardest to distinguish)
    sim_hardneg = sim_neg.max(dim=-1).values   # [B]

    # Triplet loss: penalise when positive is not far enough from negative
    loss = F.relu(margin - sim_pos + sim_hardneg)   # [B]

    return loss.mean()


def get_curriculum_weights(
    epoch: int,
    cfg: CurriculumConfig = None,
) -> dict:
    """
    Returns the curriculum weight for each loss term at a given epoch.

    Returns dict: {
        'attr':  float ∈ [0, 1],
        'subj':  float ∈ [0, 1],
        'spat':  float ∈ [0, 1],
    }

    Phase 1 (0 to phase1_end_epoch):
        All contrastive weights = 0. Only L1 + IoU active.
        Model learns basic localization first.

    Phase 2 (phase1_end_epoch to phase2_end_epoch):
        L_attr ramps linearly from 0 → 1.
        Attribute discrimination introduced first because it's the
        most common component (57% of phrases have attributes).

    Phase 3 (phase2_end_epoch onwards):
        L_attr = 1 (fully active).
        L_subj and L_spat ramp linearly from 0 → 1.
        Subject and spatial discrimination added last.

    The linear ramp (instead of a hard switch) prevents loss spikes
    that would destabilise the optimizer state.
    """
    if cfg is None:
        cfg = CurriculumConfig()

    def linear_ramp(current, start, end):
        """Linearly ramp from 0→1 between start and end epochs."""
        if current < start:
            return 0.0
        if current >= end:
            return 1.0
        return (current - start) / (end - start)

    ramp_end_attr = cfg.phase1_end_epoch + cfg.ramp_epochs
    ramp_end_subj = cfg.phase2_end_epoch + cfg.ramp_epochs
    ramp_end_spat = cfg.phase2_end_epoch + cfg.ramp_epochs

    return {
        'attr': linear_ramp(epoch, cfg.phase1_end_epoch, ramp_end_attr),
        'subj': linear_ramp(epoch, cfg.phase2_end_epoch, ramp_end_subj),
        'spat': linear_ramp(epoch, cfg.phase2_end_epoch, ramp_end_spat),
    }


class CCDGLoss(nn.Module):
    """
    Full CCDG loss function.

    L_total = L1 + λ_iou·(1-IoU)
            + α_attr(t)·λ_attr·L_attr
            + α_subj(t)·λ_subj·L_subj
            + α_spat(t)·λ_spat·L_spat

    Parameters
    ----------
    cfg_loss       : LossConfig
    cfg_curriculum : CurriculumConfig
    """

    def __init__(
        self,
        cfg_loss:       LossConfig       = None,
        cfg_curriculum: CurriculumConfig = None,
    ):
        super().__init__()
        self.lc  = cfg_loss       or LossConfig()
        self.cc  = cfg_curriculum or CurriculumConfig()

    def forward(
        self,
        model_out: dict,
        batch:     dict,
        epoch:     int,
    ) -> dict:
        """
        Compute full loss.

        Parameters
        ----------
        model_out : dict   output of CCDGModel.forward()
        batch     : dict   the raw batch from DataLoader
        epoch     : int    current training epoch (for curriculum)

        Returns
        -------
        dict with keys:
            total        : scalar  — total loss (backprop on this)
            l1           : scalar  — L1 component
            iou          : scalar  — IoU component
            attr         : scalar  — attribute contrastive
            subj         : scalar  — subject contrastive
            spat         : scalar  — spatial contrastive
            curriculum   : dict    — current curriculum weights
        """
        pred_boxes = model_out['pred_boxes']     # [B, 4]
        gt_boxes   = batch['gt_box']             # [B, 4]

        # ── Localization losses (always active) ─────────────────────
        loss_l1  = l1_loss(pred_boxes, gt_boxes)
        loss_iou = iou_loss(pred_boxes, gt_boxes)

        loc_loss = (self.lc.lambda_l1  * loss_l1 +
                    self.lc.lambda_iou * loss_iou)

        # ── Curriculum weights for this epoch ───────────────────────
        weights = get_curriculum_weights(epoch, self.cc)

        # ── Attribute contrastive loss ───────────────────────────────
        loss_attr = torch.tensor(0.0, device=pred_boxes.device)
        if weights['attr'] > 0:
            # Only compute for samples where attribute component is active
            has_attr = batch['has_attribute']   # [B] bool
            if has_attr.any():
                loss_attr = triplet_loss(
                    anchor   = model_out['f_visual_attr'][has_attr],
                    positive = model_out['f_attr'][has_attr],
                    negative = model_out['neg_f_attr'][has_attr],
                    margin   = self.lc.triplet_margin,
                )

        # ── Subject contrastive loss ─────────────────────────────────
        loss_subj = torch.tensor(0.0, device=pred_boxes.device)
        if weights['subj'] > 0:
            loss_subj = triplet_loss(
                anchor   = model_out['f_visual_reg'],
                positive = model_out['f_subj'],
                negative = model_out['neg_f_subj'],
                margin   = self.lc.triplet_margin,
            )

        # ── Spatial contrastive loss ─────────────────────────────────
        loss_spat = torch.tensor(0.0, device=pred_boxes.device)
        if weights['spat'] > 0:
            has_spat = batch['has_spatial']   # [B] bool
            if has_spat.any():
                loss_spat = triplet_loss(
                    anchor   = model_out['f_visual_sce'][has_spat],
                    positive = model_out['f_spat'][has_spat],
                    negative = model_out['neg_f_spat'][has_spat],
                    margin   = self.lc.triplet_margin,
                )

        # ── Combine ──────────────────────────────────────────────────
        contrastive_loss = (
            weights['attr'] * self.lc.lambda_attr * loss_attr +
            weights['subj'] * self.lc.lambda_subj * loss_subj +
            weights['spat'] * self.lc.lambda_spat * loss_spat
        )

        total = loc_loss + contrastive_loss

        return {
            'total':      total,
            'l1':         loss_l1.detach(),
            'iou':        loss_iou.detach(),
            'attr':       loss_attr.detach(),
            'subj':       loss_subj.detach(),
            'spat':       loss_spat.detach(),
            'curriculum': weights,
        }
