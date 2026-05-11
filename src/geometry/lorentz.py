"""
Lorentz (Hyperboloid) Embedding — Hyperbolic geometry with negative curvature.

Mathematical background
-----------------------
The hyperboloid model H^n_K of curvature K > 0 (radius 1/√K) is the upper sheet of:

    -x₀² + x₁² + ... + xₙ² = -1/K,    x₀ > 0

Points are represented in (n+1)-dimensional Minkowski space with the metric:

    ⟨x, y⟩_L  =  -x₀y₀  +  Σᵢ xᵢyᵢ

Geodesic distance between two points:

    d(x, y)  =  (1/√K) · arccosh(-K · ⟨x, y⟩_L)

Implementation strategy (Phase 1)
----------------------------------
Store ONLY the spatial coordinates x₁..xₙ.
The time-like coordinate x₀ is reconstructed on-the-fly:

    x₀  =  sqrt(1/K + ||x_spatial||²)

This construction is:
  - Automatically on the manifold (no constraint to enforce)
  - Memory-efficient (vocab_size × dim, not vocab_size × (dim+1))
  - Numerically stable for ||x_spatial|| small (regime where curvature effects dominate)

Tangent space projection (Phase 1 simplified)
----------------------------------------------
For a point p = (x₀, x_spatial) ∈ H^n_K, the tangent space at p has an orthogonal
complement in the ambient Minkowski space.  At the base point o = (1/√K, 0, ..., 0)
(origin on the manifold) the tangent space T_o H^n_K  ≅  Rⁿ  is simply the hyperplane
x₀ = 0 — i.e., the spatial coordinates themselves.

This means the tangent projection at the origin is the identity map on x_spatial:
    π(x)  =  x_spatial

This avoids the expensive logarithmic map (log_o: H → T_o H) without losing
the key property: the Transformer receives a standard Euclidean tensor [B, T, dim].

Phase 2+ will replace this with the full log-map when geometric losses are introduced.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LorentzEmbedding(nn.Module):
    """
    Token embedding on the Lorentz (Hyperboloid) manifold H^dim_K.

    Stores spatial coordinates only; reconstructs the time-like coordinate
    on-the-fly.  Feeds the Euclidean Transformer via a tangent-space projection
    that is, in Phase 1, the identity on spatial coords.

    Args:
        vocab_size:   Vocabulary size.
        dim:          Spatial dimension (= model hidden dim).
        curvature:    Positive curvature magnitude K.  Manifold radius = 1/√K.
                      Larger K → tighter manifold → stronger hierarchical bias.
        init_scale:   Std-dev for Gaussian init of spatial coords.  Small values
                      keep embeddings near the origin where the manifold is nearly
                      flat and training is most stable.
        padding_idx:  Optional padding index (passed through to nn.Embedding).
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        curvature: float = 1.0,
        init_scale: float = 0.01,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        assert curvature > 0, f"Lorentz curvature must be positive, got {curvature}"
        assert dim >= 1, f"dim must be >= 1, got {dim}"

        self.vocab_size = vocab_size
        self.dim = dim
        self.curvature = curvature
        self.init_scale = init_scale

        # Spatial coordinates: shape (vocab_size, dim).
        # Small init keeps points near the hyperboloid apex (origin),
        # where the manifold is locally nearly flat — safe starting regime.
        self.spatial_coords = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        nn.init.normal_(self.spatial_coords.weight, mean=0.0, std=init_scale)
        if padding_idx is not None:
            with torch.no_grad():
                self.spatial_coords.weight[padding_idx].zero_()

        # Reciprocal of curvature — used in time-coord reconstruction and monitoring.
        # Not a parameter; derived from curvature.
        self._inv_K: float = 1.0 / curvature

    # ─────────────────────────────── manifold helpers ────────────────────────

    def time_coord(self, x_spatial: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the time-like coordinate from spatial coordinates.

        x₀ = sqrt(1/K + ||x_spatial||²)

        This is always real and > 0, so no clamping is needed.
        The Lorentz constraint  -x₀² + ||x_spatial||² = -1/K  is satisfied
        by construction:  -x₀² + ||x||²  =  -(1/K + ||x||²) + ||x||²  =  -1/K.

        Args:
            x_spatial: (..., dim) spatial coordinates.
        Returns:
            (..., 1) time-like coordinate.
        """
        sq_norm = x_spatial.float().pow(2).sum(dim=-1, keepdim=True)
        return torch.sqrt(self._inv_K + sq_norm).type_as(x_spatial)

    def lorentz_points(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Return full Lorentz manifold coordinates (x₀, x_spatial) for input tokens.

        Shape: (*, seq_len, dim+1)
        x₀ is placed in position 0 (convention: time-like first).

        Primarily used for:
          - Geometry monitoring (constraint violation checks)
          - Phase 2 geometric losses (Lorentz distance, contrastive objectives)
          - Visualization of radial distribution on the manifold
        """
        x_s = self.spatial_coords(input_ids)          # (..., T, dim)
        x_t = self.time_coord(x_s)                     # (..., T, 1)
        return torch.cat([x_t, x_s], dim=-1)           # (..., T, dim+1)

    # ─────────────────────────────── forward ─────────────────────────────────

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens and project to tangent space at the manifold origin.

        Phase 1 simplified projection:
            π(x) = x_spatial

        This is the exact tangent space T_o H^n_K at the apex o = (1/√K, 0,...,0),
        because the tangent hyperplane at o is {x₀=0} in Minkowski space.
        No logarithmic map is required.

        Args:
            input_ids: (batch, seq_len) or (...) integer token IDs.
        Returns:
            (batch, seq_len, dim) Euclidean tensor ready for the Transformer.
        """
        return self.spatial_coords(input_ids)

    # ─────────────────────────────── Phase 2 stubs ───────────────────────────

    @staticmethod
    def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Minkowski inner product  ⟨x, y⟩_L  =  -x₀y₀  +  Σᵢ xᵢyᵢ.

        Expects full manifold coords (dim+1) in the last dimension,
        time-like coordinate in position 0.

        Args:
            x, y: (..., dim+1) Lorentz points.
        Returns:
            (...,) scalar inner products.
        """
        time_part  = -x[..., :1] * y[..., :1]
        space_part =  x[..., 1:] * y[..., 1:]
        return (time_part + space_part).sum(dim=-1)

    def lorentz_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Geodesic distance on H^n_K between Lorentz points x and y.

        d(x, y)  =  (1/√K) · arccosh(clamp(-K · ⟨x, y⟩_L, min=1))

        The clamp to ≥ 1 is required because:
          - Numerically ⟨x,y⟩_L may be slightly > -1/K (manifold origin distance),
            giving -K·⟨x,y⟩_L slightly < 1 due to floating-point error.
          - arccosh is defined only for arguments ≥ 1.

        NOTE: This stub is provided for Phase 2 geometric losses.
              It is not called during Phase 1 training.

        Args:
            x, y: (..., dim+1) Lorentz manifold points (from lorentz_points()).
        Returns:
            (...,) non-negative geodesic distances.
        """
        inner  = self.lorentz_inner(x, y)
        arg    = (-self.curvature * inner).clamp(min=1.0 + 1e-7)
        return torch.acosh(arg) / math.sqrt(self.curvature)

    def exp_map(
        self,
        base: torch.Tensor,
        tangent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exponential map on H^n_K: maps a tangent vector at `base` to the manifold.

        exp_p(v)  =  cosh(√K·||v||_L) · p  +  sinh(√K·||v||_L) / (√K·||v||_L) · v

        where ||v||_L² = ⟨v, v⟩_L (always non-negative for tangent vectors).

        NOTE: Stub for Phase 2 hyperbolic output projection.

        Args:
            base:    (..., dim+1) base point on the manifold.
            tangent: (..., dim+1) tangent vector at base (must satisfy ⟨base, v⟩_L = 0).
        Returns:
            (..., dim+1) point on the manifold.
        """
        sqrt_K = math.sqrt(self.curvature)
        # Lorentz norm of tangent vector (non-negative for tangent vectors)
        v_norm = self.lorentz_inner(tangent, tangent).clamp(min=0.0).sqrt()  # (...,)
        v_norm_scaled = (sqrt_K * v_norm).unsqueeze(-1)  # √K·‖v‖_L, (..., 1)
        # Safe division: where v_norm ≈ 0 use identity (sinh(x)/x → 1 as x→0)
        safe_norm = v_norm_scaled.clamp(min=1e-8)
        return (
            torch.cosh(v_norm_scaled) * base
            + torch.sinh(v_norm_scaled) / safe_norm * tangent
        )

    def log_map(
        self,
        base: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Logarithmic map on H^n_K: maps a manifold point to the tangent space at `base`.

        log_p(q)  =  d(p, q) / sinh(√K · d(p, q)) · (q + K·⟨p,q⟩_L · p)

        NOTE: Stub for Phase 3+ full tangent-space pipeline.
              Phase 1 uses the simplified identity projection instead.

        Args:
            base:  (..., dim+1) base point on the manifold.
            point: (..., dim+1) target point on the manifold.
        Returns:
            (..., dim+1) tangent vector at `base`.
        """
        sqrt_K   = math.sqrt(self.curvature)
        dist     = self.lorentz_distance(base, point)   # (...,)
        inner    = self.lorentz_inner(base, point)       # (...,)
        alpha    = (-self.curvature * inner).unsqueeze(-1)   # (..., 1)
        denom    = torch.sinh(sqrt_K * dist).unsqueeze(-1).clamp(min=1e-8)
        return (dist.unsqueeze(-1) / denom) * (point + alpha * base)

    # ─────────────────────────────── extras ──────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, dim={self.dim}, "
            f"curvature={self.curvature:.4f}, init_scale={self.init_scale}"
        )
