"""
HyperbolicTransformerLM — LLaMA-style Transformer with curved token embeddings.

Architecture overview
---------------------
Token IDs
    ↓
HyperbolicTransformerLM.tok_emb              ← geometry embedding (Phase 1)
    ↓  forward() returns x_spatial [B, T, D]
    ↓  (tangent-space projection at manifold origin — Phase 1: identity)
Existing Euclidean Transformer (UNCHANGED)
    ↓  Attention, RMSNorm, RoPE, SwiGLU FFN, KV-cache, FlashAttention
    ↓
self.norm + self.output                       ← Euclidean output head (unchanged)
    ↓
Logits [B, T, vocab_size]

Design principles
-----------------
1. The Transformer core (attention, residual stream, RMSNorm, FFN) is UNTOUCHED.
   This preserves FlashAttention dispatch, KV-cache, gradient checkpointing,
   torch.compile compatibility, and bf16 performance.

2. Geometry is introduced only at the embedding boundary.  The model receives
   standard [B, T, D] float tensors regardless of geometry_type.

3. HyperbolicTransformerLM subclasses TransformerLM to reuse 100% of:
   - TransformerBlock, Attention, FeedForward, RMSNorm, RoPE
   - forward(), generate(), enable/disable_gradient_checkpointing()
   - count_parameters(), summary(), _init_weights() (partially overridden)

4. Weight tying is handled post-embedding-install to avoid stale references.
   The parent is always constructed with tie_embeddings=False; the subclass
   re-applies tying after replacing tok_emb.

Comparison workflow
-------------------
To compare Euclidean vs. Hyperbolic on identical architectures:

    from src.model import TransformerLM
    from src.hyperbolic_model import HyperbolicTransformerLM
    from src.config import get_model_config, get_hyperbolic_model_config

    eucl_model  = TransformerLM(get_model_config("tiny"))
    hyp_model   = HyperbolicTransformerLM(get_hyperbolic_model_config("tiny"))
    # Identical dims, identical Transformer core, different tok_emb.

Phase 2+ hooks
--------------
The geometry stubs in LorentzEmbedding / SphericalEmbedding are designed for:
  Phase 2 → hyperbolic contrastive / geodesic loss (call tok_emb.lorentz_distance)
  Phase 3 → geodesic retrieval (lorentz_points / sphere_points)
  Phase 4 → tangent-space attention projection (log_map before attention)
  Phase 5 → manifold-native components (exp_map on residual stream)
"""

import dataclasses
import math
from typing import Optional

import torch
import torch.nn as nn

from .config import HyperbolicModelConfig
from .model import TransformerLM
from .geometry import LorentzEmbedding, SphericalEmbedding


class HyperbolicTransformerLM(TransformerLM):
    """
    Decoder-only Transformer with curved token embeddings.

    Subclasses TransformerLM, replacing only tok_emb with a geometry-aware
    version.  All other components and methods are fully inherited.

    Args:
        config: HyperbolicModelConfig — extends ModelConfig with geometry fields.
    """

    def __init__(self, config: HyperbolicModelConfig):
        # ── Step 1: Build the full Euclidean Transformer ────────────────────
        # We defer weight tying (tie_embeddings=False) because:
        #   a) The parent would tie output.weight → tok_emb.weight pointing to the
        #      nn.Embedding we are about to *replace*.
        #   b) After replacement, that reference would be stale (dangling pointer
        #      to a no-longer-used Parameter).
        # We re-apply tying ourselves in Step 3.
        eucl_config = dataclasses.replace(config, tie_embeddings=False)
        super().__init__(eucl_config)

        # Store the full config (with geometry fields) — overwrites the parent's
        # reference to the temporary eucl_config.
        self.config = config

        # ── Step 2: Install geometry embedding ──────────────────────────────
        # Replace the nn.Embedding created by the parent with the appropriate
        # curved embedding.  Both LorentzEmbedding and SphericalEmbedding expose
        # the same forward(input_ids) → [B, T, dim] interface, so the inherited
        # forward() method works without modification.
        if config.geometry_type == "lorentz":
            self.tok_emb = LorentzEmbedding(
                vocab_size=config.vocab_size,
                dim=config.dim,
                curvature=config.curvature,
                init_scale=config.embed_init_scale,
            )
        elif config.geometry_type == "spherical":
            self.tok_emb = SphericalEmbedding(
                vocab_size=config.vocab_size,
                dim=config.dim,
                curvature=config.curvature,
                init_scale=config.embed_init_scale,
            )
        else:
            # geometry_type == "euclidean": keep the nn.Embedding from parent.
            # Re-initialize with standard std=0.02 (parent already did this,
            # but we explicitly re-apply to be consistent with any future changes).
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        # ── Step 3: Re-apply weight tying ────────────────────────────────────
        if config.tie_embeddings:
            if config.geometry_type in ("lorentz", "spherical"):
                # Tie output logit weights to spatial coordinates.
                # Geometric interpretation: logit for token j ∝ ⟨h, x_spatial_j⟩
                # where h is the Transformer hidden state.
                # This is experimental — it provides an inductive bias that
                # tokens with similar manifold positions produce similar logits.
                self.output.weight = self.tok_emb.spatial_coords.weight
            else:
                # Standard Euclidean tie
                self.output.weight = self.tok_emb.weight

        # ── Step 4: Re-initialize residual projections ───────────────────────
        # The parent's _init_weights() already scaled wo and w_down.
        # We do NOT reinitialize here — the inherited scaling is correct.
        # The geometry embedding was initialized with embed_init_scale in its
        # own __init__, so no additional init is needed.

    # ─────────────────────────────── properties ──────────────────────────────

    @property
    def geometry_type(self) -> str:
        return self.config.geometry_type

    @property
    def curvature(self) -> float:
        return self.config.curvature

    # ─────────────────────────────── summary ─────────────────────────────────

    def summary(self) -> str:
        """Extend parent summary with geometry information."""
        base = super().summary()
        geom_lines = [
            f"  Geometry:       {self.config.geometry_type}",
            f"  Curvature K:    {self.config.curvature}",
            f"  Embed init σ:   {self.config.embed_init_scale}",
            f"  Weight tying:   {self.config.tie_embeddings}",
        ]
        # Insert geometry block after the first line of the parent summary
        lines = base.splitlines()
        # Find a safe insertion point — after the header line
        insert_after = 1
        return "\n".join(lines[:insert_after] + geom_lines + lines[insert_after:])

    # ─────────────────────────────── Phase 2 hooks (stubs) ───────────────────

    def get_lorentz_points(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return full Lorentz manifold coordinates for monitoring / Phase 2 losses.

        Returns None for non-Lorentz geometries.

        Args:
            input_ids: (batch, seq_len) token IDs.
        Returns:
            (batch, seq_len, dim+1) Lorentz points, or None.
        """
        if isinstance(self.tok_emb, LorentzEmbedding):
            return self.tok_emb.lorentz_points(input_ids)
        return None

    def get_sphere_points(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return full sphere coordinates for monitoring / Phase 2 losses.

        Returns None for non-spherical geometries.

        Args:
            input_ids: (batch, seq_len) token IDs.
        Returns:
            (batch, seq_len, dim+1) sphere points, or None.
        """
        if isinstance(self.tok_emb, SphericalEmbedding):
            return self.tok_emb.sphere_points(input_ids)
        return None
