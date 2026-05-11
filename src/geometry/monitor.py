"""
Geometry Monitor — Non-intrusive diagnostic tool for curved embedding spaces.

Purpose
-------
Geometry systems fail *silently* before they fail *catastrophically*.
Common failure modes:
  - Embeddings drift to the boundary of the manifold (exploding norms)
  - Lorentz/Sphere constraints are violated (numerical instability)
  - Gradients on spatial_coords explode or vanish
  - NaN/Inf appear in embedding weights before appearing in loss

This monitor catches all of the above by sampling a small subset of vocab
entries at each logging step, computing manifold-specific statistics, and
returning them as a flat dict ready for wandb/console logging.

Design constraints
------------------
- **No state**: GeometryMonitor is instantiated once and never stores tensors.
  All computation is local to each call.  This is safe for torch.compile.
- **Cheap by design**: Samples ≤ max_sample_vocab entries (default 1000).
  On a 32K vocab this is a 3% sample — negligible overhead.
- **No gradient interference**: All stats computed under torch.no_grad().
- **dtype-safe**: Casts to float32 for statistics, never modifies the embedding.
"""

import math
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn

from .lorentz import LorentzEmbedding
from .spherical import SphericalEmbedding


# Maximum number of vocab entries to sample per monitoring call.
# 1000 / 32000 ≈ 3% — negligible GPU time, representative statistics.
_DEFAULT_MAX_SAMPLE: int = 1_000


class GeometryMonitor:
    """
    Stateless geometry diagnostic tool.

    Usage::

        monitor = GeometryMonitor()

        # In training loop after logging interval:
        stats = monitor.compute_stats(model.tok_emb, "lorentz", curvature=1.0)
        tracker.log(stats, step)

        # After each backward pass (when grads are available):
        grad_stats = monitor.compute_grad_norms(model.tok_emb)
        tracker.log(grad_stats, step)
    """

    def __init__(self, max_sample_vocab: int = _DEFAULT_MAX_SAMPLE):
        self.max_sample_vocab = max_sample_vocab

    # ─────────────────────────────── main stats ───────────────────────────────

    @torch.no_grad()
    def compute_stats(
        self,
        tok_emb: nn.Module,
        geometry_type: str,
        curvature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute geometry-specific statistics for the token embedding.

        Returned keys (all prefixed "geometry/"):

        Universal:
          geometry/spatial_norm_mean      — mean ||x_spatial|| over sampled vocab
          geometry/spatial_norm_std       — std  ||x_spatial||
          geometry/spatial_norm_max       — max  ||x_spatial||
          geometry/has_nan                — 1.0 if any NaN in sampled weights
          geometry/has_inf                — 1.0 if any Inf in sampled weights
          geometry/sample_size            — number of vocab entries sampled

        Lorentz-specific:
          geometry/lorentz_x0_mean        — mean x₀ (should be > 0)
          geometry/lorentz_constraint_violation_mean  — mean |x₀² - ||x||² - 1/K|
          geometry/lorentz_constraint_violation_max   — max  |x₀² - ||x||² - 1/K|

        Spherical-specific:
          geometry/spherical_x0_mean      — mean x₀ (should be > 0, ≤ 1/√K)
          geometry/spherical_constraint_violation_mean  — mean |x₀² + ||x||² - 1/K|
          geometry/spherical_constraint_violation_max   — max  |x₀² + ||x||² - 1/K|
          geometry/spherical_invalid_count — entries where clamping activated (x₀²→ε)

        Euclidean fallback (nn.Embedding):
          geometry/euclidean_norm_mean
          geometry/euclidean_norm_std
          geometry/euclidean_norm_max

        Args:
            tok_emb:       The token embedding module (LorentzEmbedding,
                           SphericalEmbedding, or nn.Embedding).
            geometry_type: "lorentz" | "spherical" | "euclidean".
            curvature:     Curvature K > 0 (used to compute constraint target 1/K).
        Returns:
            Dict[str, float] ready for tracker.log().
        """
        inv_K = 1.0 / curvature
        stats: Dict[str, float] = {}

        # ── extract weight ──────────────────────────────────────────────────
        if geometry_type in ("lorentz", "spherical"):
            weight = tok_emb.spatial_coords.weight   # (vocab_size, dim)
        else:
            weight = tok_emb.weight                  # (vocab_size, dim)

        vocab_size = weight.shape[0]
        n_sample = min(self.max_sample_vocab, vocab_size)
        stats["geometry/sample_size"] = float(n_sample)

        # Sample without replacement, deterministic within a monitoring call
        # (no seed setting — we do NOT want to disturb the global RNG state).
        if n_sample < vocab_size:
            idx = torch.randperm(vocab_size, device=weight.device)[:n_sample]
            w = weight[idx].float()
        else:
            w = weight.float()

        # ── NaN / Inf guard ─────────────────────────────────────────────────
        stats["geometry/has_nan"] = 1.0 if torch.isnan(w).any().item() else 0.0
        stats["geometry/has_inf"] = 1.0 if torch.isinf(w).any().item() else 0.0

        # Replace NaN/Inf for remaining stats to avoid propagation
        w_clean = w.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # ── spatial norms ────────────────────────────────────────────────────
        norms = w_clean.norm(dim=-1)   # (n_sample,)
        stats["geometry/spatial_norm_mean"] = norms.mean().item()
        stats["geometry/spatial_norm_std"]  = norms.std().item()
        stats["geometry/spatial_norm_max"]  = norms.max().item()

        # ── geometry-specific stats ──────────────────────────────────────────
        if geometry_type == "lorentz":
            stats.update(self._lorentz_stats(w_clean, inv_K))
        elif geometry_type == "spherical":
            stats.update(self._spherical_stats(w_clean, inv_K))
        else:
            # Euclidean — just alias the norm stats under euclidean keys
            stats["geometry/euclidean_norm_mean"] = stats["geometry/spatial_norm_mean"]
            stats["geometry/euclidean_norm_std"]  = stats["geometry/spatial_norm_std"]
            stats["geometry/euclidean_norm_max"]  = stats["geometry/spatial_norm_max"]

        return stats

    # ─────────────────────────────── gradient norms ───────────────────────────

    @torch.no_grad()
    def compute_grad_norms(
        self,
        tok_emb: nn.Module,
        geometry_type: str = "euclidean",
    ) -> Dict[str, float]:
        """
        Compute gradient norms for the token embedding weight.

        Call this after loss.backward() and before optimizer.zero_grad().
        Returns empty dict if no gradient is available (e.g., first step or
        embedding is frozen).

        Returned keys:
          geometry/embed_grad_norm    — global L2 norm of the embedding gradient
          geometry/embed_grad_max     — max absolute value in the gradient
          geometry/embed_has_nan_grad — 1.0 if any NaN in gradient
          geometry/embed_has_inf_grad — 1.0 if any Inf in gradient

        Args:
            tok_emb:       Token embedding module.
            geometry_type: "lorentz" | "spherical" | "euclidean" — determines
                           which sub-parameter to inspect.
        Returns:
            Dict[str, float] — empty if gradient is not available.
        """
        if geometry_type in ("lorentz", "spherical"):
            param = tok_emb.spatial_coords.weight
        else:
            param = tok_emb.weight

        if param.grad is None:
            return {}

        g = param.grad.float()
        return {
            "geometry/embed_grad_norm":     g.norm().item(),
            "geometry/embed_grad_max":      g.abs().max().item(),
            "geometry/embed_has_nan_grad":  1.0 if torch.isnan(g).any().item() else 0.0,
            "geometry/embed_has_inf_grad":  1.0 if torch.isinf(g).any().item() else 0.0,
        }

    # ─────────────────────────────── private helpers ──────────────────────────

    def _lorentz_stats(
        self,
        x_spatial: torch.Tensor,   # (n_sample, dim), float32, NaN-clean
        inv_K: float,
    ) -> Dict[str, float]:
        """Compute Lorentz manifold-specific statistics."""
        sq_norms = x_spatial.pow(2).sum(dim=-1)       # (n_sample,)
        x0       = (inv_K + sq_norms).sqrt()           # (n_sample,)

        # Lorentz constraint:  x₀² - ||x_spatial||² = 1/K
        # (by construction this should be ≈ 0, but we track it for numerical health)
        constraint = (x0.pow(2) - sq_norms - inv_K).abs()

        return {
            "geometry/lorentz_x0_mean":                   x0.mean().item(),
            "geometry/lorentz_constraint_violation_mean": constraint.mean().item(),
            "geometry/lorentz_constraint_violation_max":  constraint.max().item(),
        }

    def _spherical_stats(
        self,
        x_spatial: torch.Tensor,   # (n_sample, dim), float32, NaN-clean
        inv_K: float,
    ) -> Dict[str, float]:
        """Compute spherical manifold-specific statistics."""
        sq_norms = x_spatial.pow(2).sum(dim=-1)          # (n_sample,)

        # Reconstruct x₀ with the same clamping used in SphericalEmbedding
        radicand = (inv_K - sq_norms).clamp(min=1e-8)
        x0       = radicand.sqrt()                        # (n_sample,)

        # Count entries where clamping was non-trivial (point left the sphere)
        invalid_mask    = (inv_K - sq_norms) < 1e-8      # (n_sample,)
        invalid_count   = invalid_mask.sum().item()

        # Sphere constraint:  x₀² + ||x_spatial||² = 1/K
        constraint = (x0.pow(2) + sq_norms - inv_K).abs()

        return {
            "geometry/spherical_x0_mean":                   x0.mean().item(),
            "geometry/spherical_constraint_violation_mean": constraint.mean().item(),
            "geometry/spherical_constraint_violation_max":  constraint.max().item(),
            "geometry/spherical_invalid_count":             float(invalid_count),
        }
