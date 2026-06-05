"""
models/ccdg.py — Compositional Contrastive Dense Grounding model.

ARCHITECTURE OVERVIEW:
──────────────────────
We build on top of OWL-ViT (ViT-B/32) and add three novel components:

  1. Multi-scale visual feature extraction
     - Patch-level features   : local colour/texture  → for attribute alignment
     - Region-level features  : RoI-pooled object box → for subject alignment
     - Scene-level features   : global CLS token      → for spatial alignment

  2. Compositional text encoders
     We pass the subject / attribute / spatial component phrases through
     the SAME OWL-ViT text encoder (no new parameters added).
     Each component gets its own 512-D embedding.

  3. Compositional alignment module
     A lightweight cross-attention layer aligns each text component
     with its matching visual scale. This is where the novelty lives.

HOW OWL-ViT WORKS INTERNALLY (so this makes sense):
─────────────────────────────────────────────────────
OwlViTModel has two sub-models:

  vision_model  : ViT-B/32
    Input : pixel_values [B, 3, 768, 768]
    Output: last_hidden_state [B, 577, 768]
              index 0    = CLS token  (global image summary)
              index 1..576 = patch tokens (local 32×32 regions)

  text_model    : 12-layer text transformer
    Input : input_ids [B, 16], attention_mask [B, 16]
    Output: pooler_output [B, 512]  ← the phrase embedding we use

Then a visual_projection and text_projection layer map both to a
shared 512-D space for computing cosine similarity scores.

WE ADD ON TOP:
──────────────
  patch_proj   : Linear(768 → 512) for patch tokens
  scene_proj   : Linear(768 → 512) for CLS token
  attr_attn    : MultiheadAttention for attribute-aware patch selection
  box_head     : MLP(512 → 4) predicts (cx, cy, w, h) ∈ [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OwlViTModel, OwlViTProcessor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ModelConfig, Paths


class MultiScaleVisualExtractor(nn.Module):
    """
    Extracts three levels of visual features from OWL-ViT's patch tokens.

    WHY THREE LEVELS?
    ─────────────────
    Language is compositional — "woman in red dress on the left" describes
    three different things simultaneously. The visual counterparts of each
    component live at different spatial scales:

      Subject ("woman")       → region level: the whole person bounding box
      Attribute ("red dress") → patch level: local colour/texture pixels
      Spatial ("on the left") → scene level: global position in the image

    Matching each text component to its natural visual scale is the key
    insight of CCDG. No prior work does this.

    Parameters
    ----------
    vision_hidden : int
        Hidden dimension of OWL-ViT vision encoder (768 for ViT-B/32).
    embed_dim : int
        Target embedding dimension (512, matching text encoder).
    """

    def __init__(self, vision_hidden: int = 768, embed_dim: int = 512):
        super().__init__()

        # Project patch tokens from 768 → 512
        # Used for BOTH patch-level and region-level features
        # (region = mean-pool of patches within a box)
        self.patch_proj = nn.Linear(vision_hidden, embed_dim)

        # Project CLS token from 768 → 512
        # CLS token = global scene summary = spatial relation features
        self.scene_proj = nn.Linear(vision_hidden, embed_dim)

        # Cross-attention for attribute-aware patch selection.
        # WHAT THIS DOES: instead of naively averaging all patches
        # (which blurs colour/texture across the whole image),
        # we use the attribute text embedding as a QUERY to focus
        # attention on patches that actually contain the attribute.
        #
        # Example: for "red dress", attention focuses on the patches
        # covering the dress region, not the background.
        self.attr_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )

        # LayerNorm after cross-attention (standard transformer practice)
        self.attr_norm = nn.LayerNorm(embed_dim)

    def forward(self, vision_hidden_states, f_attr, gt_box):
        """
        Extract multi-scale visual features.

        Parameters
        ----------
        vision_hidden_states : Tensor [B, 577, 768]
            Raw output from OWL-ViT vision encoder.
            Index 0 = CLS token, index 1..576 = patch tokens.

        f_attr : Tensor [B, 512]
            Attribute phrase embedding. Used as cross-attention query
            to select attribute-relevant patches.

        gt_box : Tensor [B, 4]
            Ground truth box in normalized (cx, cy, w, h) format.
            Used to identify which patches overlap the referent object.

        Returns
        -------
        f_scene  : Tensor [B, 512]  — scene-level (spatial features)
        f_attr_v : Tensor [B, 512]  — patch-level (attribute features)
        f_region : Tensor [B, 512]  — region-level (subject features)
        """
        # ── CLS token → scene features ─────────────────────────────
        # Shape: [B, 768] → [B, 512]
        cls_token = vision_hidden_states[:, 0, :]
        f_scene   = self.scene_proj(cls_token)

        # ── Patch tokens → projected ────────────────────────────────
        # Shape: [B, 576, 768] → [B, 576, 512]
        patches   = vision_hidden_states[:, 1:, :]   # remove CLS
        f_patches = self.patch_proj(patches)          # [B, 576, 512]

        # ── Attribute-aware patch selection via cross-attention ──────
        # Query  = attribute embedding  [B, 1, 512]
        # Key/Value = all patch features [B, 576, 512]
        # Output = weighted sum of patches, focused on attribute regions
        query    = f_attr.unsqueeze(1)   # [B, 1, 512]
        attn_out, _ = self.attr_cross_attn(
            query=query,
            key=f_patches,
            value=f_patches,
        )                                # [B, 1, 512]
        f_attr_v = self.attr_norm(attn_out.squeeze(1))  # [B, 512]

        # ── Region features via GT box spatial pooling ───────────────
        # WHY: the subject component ("woman") should match the visual
        # features of the WHOLE object, not just texture patches.
        # We find which of the 24×24 patches fall inside the GT box
        # and mean-pool them → gives the object-level representation.
        f_region = self._pool_region_features(f_patches, gt_box)

        return f_scene, f_attr_v, f_region

    def _pool_region_features(
        self, f_patches: torch.Tensor, gt_box: torch.Tensor
    ) -> torch.Tensor:
        """
        Spatially pool patch features within the GT bounding box.

        OWL-ViT uses a 24×24 patch grid (768px image / 32px patch = 24).
        We convert the GT box coordinates to patch indices and mean-pool
        the patches within the box.

        Parameters
        ----------
        f_patches : Tensor [B, 576, 512]
        gt_box    : Tensor [B, 4]  normalized (cx, cy, w, h)

        Returns
        -------
        region_feats : Tensor [B, 512]
        """
        B, _, D = f_patches.shape
        GRID    = 24   # 24×24 patch grid for ViT-B/32 at 768px

        # Convert (cx, cy, w, h) → (x1, y1, x2, y2)
        cx, cy = gt_box[:, 0], gt_box[:, 1]
        w,  h  = gt_box[:, 2], gt_box[:, 3]
        x1 = (cx - w / 2).clamp(0.0, 1.0)
        y1 = (cy - h / 2).clamp(0.0, 1.0)
        x2 = (cx + w / 2).clamp(0.0, 1.0)
        y2 = (cy + h / 2).clamp(0.0, 1.0)

        # Reshape patches to spatial grid: [B, 24, 24, 512]
        f_grid = f_patches.reshape(B, GRID, GRID, D)

        region_feats = []
        for i in range(B):
            # Convert normalized [0,1] coords to patch indices [0,23]
            r1 = int(y1[i].item() * GRID)
            r2 = max(int(y2[i].item() * GRID), r1 + 1)   # ensure ≥1 patch
            c1 = int(x1[i].item() * GRID)
            c2 = max(int(x2[i].item() * GRID), c1 + 1)

            # Clamp to valid range
            r1, r2 = max(0, r1), min(GRID, r2)
            c1, c2 = max(0, c1), min(GRID, c2)

            # Mean pool patches within region: [rows, cols, 512] → [512]
            region = f_grid[i, r1:r2, c1:c2, :]     # [r, c, 512]
            region_feats.append(region.mean(dim=(0, 1)))

        return torch.stack(region_feats)   # [B, 512]


class BoxHead(nn.Module):
    """
    Predicts a single bounding box from aligned text-image features.

    WHAT IT DOES:
    ─────────────
    After computing cosine similarity between the text query and every
    image patch, we have a score for each of the 576 patches. The box head
    converts these attended features into (cx, cy, w, h) coordinates.

    This is simpler than OWL-ViT's full detection head (which predicts
    a box per patch) — we predict ONE box per query, matching the
    RefCOCO task (one phrase → one box).

    Architecture: 3-layer MLP with ReLU, final Sigmoid to keep output ∈ [0,1]
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 4),
            nn.Sigmoid(),   # output in [0, 1] for normalized box coords
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, 512]
            Attended image features (text-image similarity weighted).

        Returns
        -------
        boxes : Tensor [B, 4]  (cx, cy, w, h) all ∈ [0, 1]
        """
        return self.mlp(x)


class CCDGModel(nn.Module):
    """
    Compositional Contrastive Dense Grounding model.

    Combines OWL-ViT's strong vision-language backbone with our
    novel compositional alignment and contrastive training.

    FORWARD PASS OVERVIEW:
    ──────────────────────
    Input: image + full phrase + 3 component phrases + GT box + hard negatives

    Step 1: Encode image → multi-scale visual features
            (scene, attribute-aware, region)

    Step 2: Encode text → 4 phrase embeddings
            (full, subject, attribute, spatial)

    Step 3: Compute similarity for localization
            full_phrase ↔ image patches → attended features → box prediction

    Step 4: Return all features for loss computation
            (actual loss is in losses/contrastive.py)

    Parameters
    ----------
    cfg_model : ModelConfig
    cfg_paths : Paths
    """

    def __init__(
        self,
        cfg_model: ModelConfig = None,
        cfg_paths: Paths = None,
    ):
        super().__init__()
        self.cfg   = cfg_model or ModelConfig()
        self.paths = cfg_paths or Paths()

        # ── Load pretrained OWL-ViT ────────────────────────────────
        # We use OwlViTModel (not ForObjectDetection) because we want
        # direct access to vision and text encoders separately.
        print(f"Loading OWL-ViT from {self.paths.owlvit_model}...")
        self.owlvit = OwlViTModel.from_pretrained(self.paths.owlvit_model)

        # Optionally freeze encoders (we keep both unfrozen — full fine-tune)
        if self.cfg.freeze_vision:
            for p in self.owlvit.vision_model.parameters():
                p.requires_grad = False
        if self.cfg.freeze_text:
            for p in self.owlvit.text_model.parameters():
                p.requires_grad = False

        # ── Novel components ────────────────────────────────────────
        self.visual_extractor = MultiScaleVisualExtractor(
            vision_hidden=768,
            embed_dim=self.cfg.embed_dim,
        )
        self.box_head = BoxHead(embed_dim=self.cfg.embed_dim)

        # Temperature for cosine similarity (learnable, like CLIP)
        # Initialized to 0.07 — standard value from CLIP paper
        self.logit_scale = nn.Parameter(
            torch.tensor(self.cfg.temperature_init).log()
        )

        print("CCDG model initialized.")
        print(f"  Vision encoder: {'frozen' if self.cfg.freeze_vision else 'trainable'}")
        print(f"  Text encoder:   {'frozen' if self.cfg.freeze_text else 'trainable'}")
        self._print_param_count()

    # ────────────────────────────────────────────────────────────────
    # ENCODING METHODS
    # ────────────────────────────────────────────────────────────────

    def encode_image(self, pixel_values: torch.Tensor):
        """
        Run image through OWL-ViT vision encoder.

        Returns raw hidden states [B, 577, 768].
        Index 0 = CLS, index 1..576 = patches.
        Also returns projected image embeddings [B, 576, 512]
        in the shared space (used for similarity with text).
        """
        vision_out = self.owlvit.vision_model(pixel_values=pixel_values)
        hidden     = vision_out.last_hidden_state   # [B, 577, 768]

        # OWL-ViT's visual_projection maps 768 → 512
        # We apply it to patch tokens for the main similarity computation
        patches_768 = hidden[:, 1:, :]              # [B, 576, 768]
        patches_512 = self.owlvit.visual_projection(patches_768)  # [B, 576, 512]

        # L2 normalize (standard for cosine similarity)
        patches_512 = F.normalize(patches_512, dim=-1)

        return hidden, patches_512

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run text through OWL-ViT text encoder.

        Returns the CLS-token embedding projected to 512-D.
        This is the standard phrase embedding used in OWL-ViT.

        We call this once per phrase type (full, subject, attribute, spatial),
        reusing the same encoder weights each time — no extra parameters.
        """
        text_out = self.owlvit.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # pooler_output = CLS token after projection → [B, 512]
        text_emb = text_out.pooler_output
        # Apply OWL-ViT's text projection (keeps us in the same space as images)
        text_emb = self.owlvit.text_projection(text_emb)  # [B, 512]
        return F.normalize(text_emb, dim=-1)

    # ────────────────────────────────────────────────────────────────
    # MAIN FORWARD PASS
    # ────────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        """
        Full forward pass.

        WHAT HAPPENS HERE (step by step):
        ──────────────────────────────────
        1. Image → [CLS token, 576 patch tokens] via ViT-B/32
        2. Full phrase → 1 text embedding via text transformer
        3. Three component phrases → 3 text embeddings (same encoder, 3 forward passes)
        4. Hard negative phrases → embeddings for contrastive loss
        5. Multi-scale visual features extracted for alignment
        6. Similarity computed between full phrase and all patches
        7. Attended image features → box prediction via BoxHead
        8. All features returned for loss computation

        Parameters
        ----------
        batch : dict (keys defined in refcoco_dataset.py __getitem__)

        Returns
        -------
        out : dict with keys:
            pred_boxes    : [B, 4]   predicted boxes (cx,cy,w,h) ∈ [0,1]
            f_full        : [B, 512] full phrase embedding
            f_subj        : [B, 512] subject component embedding
            f_attr        : [B, 512] attribute component embedding
            f_spat        : [B, 512] spatial component embedding
            f_visual_attr : [B, 512] attribute-scale visual features
            f_visual_reg  : [B, 512] region-scale visual features
            f_visual_sce  : [B, 512] scene-scale visual features
            neg_f_attr    : [B, K, 512] attribute negative embeddings
            neg_f_subj    : [B, K, 512] subject negative embeddings
            neg_f_spat    : [B, K, 512] spatial negative embeddings
            patches_512   : [B, 576, 512] all patch embeddings (for ablations)
        """
        pixel_values = batch['pixel_values']
        B = pixel_values.shape[0]
        device = pixel_values.device

        # ── Step 1: Encode image ────────────────────────────────────
        # hidden:       [B, 577, 768]   raw ViT features
        # patches_512:  [B, 576, 512]   projected + normalized patch embeddings
        hidden, patches_512 = self.encode_image(pixel_values)

        # ── Step 2: Encode full phrase ──────────────────────────────
        f_full = self.encode_text(
            batch['input_ids_full'], batch['attn_mask_full']
        )   # [B, 512]

        # ── Step 3: Encode component phrases ───────────────────────
        # Same encoder, three separate forward passes.
        # Cost: 3× text encoder inference — acceptable since text encoder
        # is much smaller than vision encoder (512 vs 768 hidden dim)
        f_subj = self.encode_text(
            batch['input_ids_subj'], batch['attn_mask_subj']
        )   # [B, 512]

        f_attr = self.encode_text(
            batch['input_ids_attr'], batch['attn_mask_attr']
        )   # [B, 512]

        f_spat = self.encode_text(
            batch['input_ids_spat'], batch['attn_mask_spat']
        )   # [B, 512]

        # ── Step 4: Multi-scale visual features ────────────────────
        f_visual_sce, f_visual_attr, f_visual_reg = \
            self.visual_extractor(hidden, f_attr, batch['gt_box'])
        # Each: [B, 512]

        # ── Step 5: Compute similarity for localization ─────────────
        # Dot product between full phrase embedding and each patch.
        # logit_scale is a learnable temperature (from CLIP).
        # f_full:      [B, 512]    → [B, 1, 512]
        # patches_512: [B, 576, 512]
        scale = self.logit_scale.exp().clamp(max=100.0)   # stability clamp
        sim   = scale * torch.einsum(
            'bd,bpd->bp',
            f_full,          # [B, 512]
            patches_512,     # [B, 576, 512]
        )                    # [B, 576] — similarity score per patch

        # Weighted sum of patches by similarity scores → attended features
        # This is equivalent to soft-attention over all patches
        # Shape: [B, 512]
        weights        = F.softmax(sim, dim=-1)          # [B, 576]
        attended_feats = torch.einsum(
            'bp,bpd->bd', weights, patches_512
        )                                                 # [B, 512]

        # ── Step 6: Predict box ─────────────────────────────────────
        pred_boxes = self.box_head(attended_feats)        # [B, 4]

        # ── Step 7: Encode hard negatives ──────────────────────────
        # neg_attr_ids: [B, K, max_len] — K negatives per sample
        # We encode each negative phrase and return [B, K, 512]
        neg_f_attr = self._encode_negatives(
            batch['neg_attr_ids'], batch['neg_attr_masks']
        )   # [B, K, 512]

        neg_f_subj = self._encode_negatives(
            batch['neg_subj_ids'], batch['neg_subj_masks']
        )   # [B, K, 512]

        neg_f_spat = self._encode_negatives(
            batch['neg_spat_ids'], batch['neg_spat_masks']
        )   # [B, K, 512]

        return {
            'pred_boxes':    pred_boxes,     # [B, 4]
            'f_full':        f_full,         # [B, 512]
            'f_subj':        f_subj,         # [B, 512]
            'f_attr':        f_attr,         # [B, 512]
            'f_spat':        f_spat,         # [B, 512]
            'f_visual_attr': f_visual_attr,  # [B, 512]
            'f_visual_reg':  f_visual_reg,   # [B, 512]
            'f_visual_sce':  f_visual_sce,   # [B, 512]
            'neg_f_attr':    neg_f_attr,     # [B, K, 512]
            'neg_f_subj':    neg_f_subj,     # [B, K, 512]
            'neg_f_spat':    neg_f_spat,     # [B, K, 512]
            'patches_512':   patches_512,    # [B, 576, 512]
            'sim_scores':    sim,            # [B, 576]
        }

    def _encode_negatives(
        self,
        neg_ids:   torch.Tensor,   # [B, K, max_len]
        neg_masks: torch.Tensor,   # [B, K, max_len]
    ) -> torch.Tensor:             # [B, K, 512]
        """
        Encode K hard negative phrases per sample.

        We reshape to [B*K, max_len], run through text encoder once,
        then reshape back to [B, K, 512].
        This is more efficient than K separate encoder calls.
        """
        B, K, L = neg_ids.shape
        # Flatten batch and negatives dimensions
        ids_flat   = neg_ids.reshape(B * K, L)    # [B*K, max_len]
        masks_flat = neg_masks.reshape(B * K, L)  # [B*K, max_len]

        embs_flat = self.encode_text(ids_flat, masks_flat)  # [B*K, 512]

        return embs_flat.reshape(B, K, 512)   # [B, K, 512]

    # ────────────────────────────────────────────────────────────────
    # INFERENCE (no ground truth box needed)
    # ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        input_ids:    torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference-only forward pass.
        No component phrases, no GT box, no hard negatives needed.

        Returns predicted box [B, 4] in normalized (cx, cy, w, h).
        Used for evaluation and demo.
        """
        hidden, patches_512 = self.encode_image(pixel_values)
        f_full = self.encode_text(input_ids, attention_mask)

        scale  = self.logit_scale.exp().clamp(max=100.0)
        sim    = scale * torch.einsum('bd,bpd->bp', f_full, patches_512)

        weights        = F.softmax(sim, dim=-1)
        attended_feats = torch.einsum('bp,bpd->bd', weights, patches_512)

        return self.box_head(attended_feats)   # [B, 4]

    # ────────────────────────────────────────────────────────────────
    # UTILITIES
    # ────────────────────────────────────────────────────────────────

    def _print_param_count(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params:     {total/1e6:.1f}M")
        print(f"  Trainable params: {trainable/1e6:.1f}M")
