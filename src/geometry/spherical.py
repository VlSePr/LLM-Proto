"""
Spherical Embedding — Elliptic geometry with positive curvature.

Mathematical background
-----------------------
The n-sphere S^n_K of curvature K > 0 (radius r = 1/√K) is embedded in R^(n+1) as:

    x₀² + x₁² + ... + xₙ² = 1/K

with the standard Riemannian metric inherited from R^(n+1).

The inner product between two points:

    ⟨x, y⟩  =  Σᵢ xᵢyᵢ  (standard Euclidean)

Geodesic (great-circle) distance:

    d(x, y)  =  (1/√K) · arccos(K · ⟨x, y⟩)

NOTE: Unlike hyperbolic space, spherical space is *compact* and *positively* curved.
This makes it natural for modelling *cyclic* or *compositional* structure,
while Lorentz space is better for hierarchical tree-like structure.

Implementation strategy (Phase 1)
----------------------------------
Store ONLY the spatial coordinates x₁..xₙ ("south-hemisphere" parameterisation).
The "north" coordinate x₀ is reconstructed on-the-fly:

    x₀  =  sqrt((1/K - ||x_spatial||²).clamp(min=ε))

Validity requirement:  ||x_spatial||² < 1/K.

With init_scale=0.01 and K=1.0 this holds trivially:
    E[||x||²] = dim × init_scale²  =  dim × 10⁻⁴  ≪  1.

The clamp guards against numerical drift — if ||x||² ≥ 1/K the point has left
the sphere (north pole region unreachable).  The `invalid_count` monitor stat
tracks how often the clamp activates.

Tangent space projection (Phase 1 simplified)
----------------------------------------------
At the "north pole" o = (1/√K, 0, ..., 0) the tangent space T_o S^n_K
is the hyperplane {x₀=0} — exactly the spatial coordinates x_spatial.

So the Phase 1 tangent projection at the north pole is again the identity:
    π(x)  =  x_spatial

This is the symmetric twin of the Lorentz tangent projection at the apex.

Phase 2+ will replace this with the spherical logarithmic map when geometric
losses are introduced.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Minimum squared spatial norm margin below 1/K.
# Prevents x₀ from hitting 0 (equator / south pole singularity).
_CLAMP_EPS: float = 1e-8


class SphericalEmbedding(nn.Module):
    """
    Token embedding on the n-sphere S^dim_K with positive curvature K.

    Stores spatial coordinates only; reconstructs the north-pole coordinate
    on-the-fly.  Feeds the Euclidean Transformer via a tangent-space projection
    that is, in Phase 1, the identity on spatial coords.

    Args:
        vocab_size:   Vocabulary size.
        dim:          Spatial dimension (= model hidden dim).
        curvature:    Positive curvature magnitude K.  Sphere radius = 1/√K.
                      Larger K → smaller sphere → more tightly packed embeddings.
        init_scale:   Std-dev for Gaussian init of spatial coords.  Must satisfy
                      dim × init_scale² ≪ 1/K to start safely on the manifold.
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
        assert curvature > 0, f"Spherical curvature must be positive, got {curvature}"
        assert dim >= 1, f"dim must be >= 1, got {dim}"

        self.vocab_size = vocab_size
        self.dim = dim
        self.curvature = curvature
        self.init_scale = init_scale

        # Safety check: initial expected ||x||² = dim * init_scale²
        # Should be well below 1/K so all points start on the valid hemisphere.
        expected_sq_norm = dim * (init_scale ** 2)
        inv_K = 1.0 / curvature
        if expected_sq_norm >= inv_K * 0.5:
            import warnings
            warnings.warn(
                f"SphericalEmbedding: expected initial ||x||² ≈ {expected_sq_norm:.4f} "
                f"is close to or exceeds 0.5/K = {0.5*inv_K:.4f}. "
                "Consider reducing init_scale or curvature to keep embeddings near the north pole.",
                UserWarning,
                stacklevel=2,
            )

        self.spatial_coords = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        nn.init.normal_(self.spatial_coords.weight, mean=0.0, std=init_scale)
        if padding_idx is not None:
            with torch.no_grad():
                self.spatial_coords.weight[padding_idx].zero_()

        self._inv_K: float = inv_K

    # ─────────────────────────────── manifold helpers ────────────────────────

    def north_coord(self, x_spatial: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the north-pole coordinate from spatial coordinates.

        x₀  =  sqrt((1/K - ||x_spatial||²).clamp(min=ε))

        The clamp prevents NaN gradients when ||x_spatial||² → 1/K.
        The `invalid_count` geometry monitor stat tracks how often this activates.

        Args:
            x_spatial: (..., dim) spatial coordinates.
        Returns:
            (..., 1) north-pole coordinate.
        """
        sq_norm = x_spatial.float().pow(2).sum(dim=-1, keepdim=True)
        radicand = (self._inv_K - sq_norm).clamp(min=_CLAMP_EPS)
        return torch.sqrt(radicand).type_as(x_spatial)

    def sphere_points(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Return full sphere coordinates (x₀, x_spatial) for input tokens.

        Shape: (*, seq_len, dim+1)
        x₀ is placed in position 0 (convention: north coord first).

        Primarily used for:
          - Geometry monitoring (constraint violation, invalid_count)
          - Phase 2 geometric losses (geodesic distance, contrastive objectives)
          - Visualization of distribution on the sphere surface
        """
        x_s = self.spatial_coords(input_ids)   # (..., T, dim)
        x_t = self.north_coord(x_s)             # (..., T, 1)
        return torch.cat([x_t, x_s], dim=-1)   # (..., T, dim+1)

    # ─────────────────────────────── forward ─────────────────────────────────

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens and project to tangent space at the north pole.

        Phase 1 simplified projection:
            π(x)  =  x_spatial

        The tangent space T_o S^n_K at the north pole o = (1/√K, 0,...,0) is
        the hyperplane {x₀=0}, so the spatial coordinates ARE the tangent coords.
        No logarithmic map required.

        Args:
            input_ids: (batch, seq_len) or (...) integer token IDs.
        Returns:
            (batch, seq_len, dim) Euclidean tensor ready for the Transformer.
        """
        return self.spatial_coords(input_ids)

    # ─────────────────────────────── Phase 2 stubs ───────────────────────────

    @staticmethod
    def spherical_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Standard Euclidean inner product ⟨x, y⟩ = Σᵢ xᵢyᵢ on R^(n+1).

        Args:
            x, y: (..., dim+1) sphere points.
        Returns:
            (...,) scalar inner products.
        """
        return (x * y).sum(dim=-1)

    def spherical_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Geodesic (great-circle) distance on S^n_K between sphere points x and y.

        d(x, y)  =  (1/√K) · arccos(clamp(K · ⟨x, y⟩, min=-1, max=1))

        The clamp is required because:
          - Numerical error can push K·⟨x,y⟩ outside [-1, 1].
          - arccos is defined only on [-1, 1].
          - arccos(-1) = π, arccos(1) = 0 (antipodal and identical points).

        NOTE: Stub for Phase 2 geometric losses.  Not called during Phase 1 training.

        Args:
            x, y: (..., dim+1) sphere manifold points (from sphere_points()).
        Returns:
            (...,) non-negative geodesic distances in [0, π/√K].
        """
        inner = self.spherical_inner(x, y)
        arg   = (self.curvature * inner).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return torch.acos(arg) / math.sqrt(self.curvature)

    def exp_map(
        self,
        base: torch.Tensor,
        tangent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exponential map on S^n_K: maps a tangent vector at `base` to the manifold.

        exp_p(v)  =  cos(√K·||v||) · p  +  sin(√K·||v||) / (√K·||v||) · v

        where ||v|| is the Euclidean norm of the tangent vector.

        NOTE: Stub for Phase 2 hyperbolic output projection.

        Args:
            base:    (..., dim+1) base point on the sphere.
            tangent: (..., dim+1) tangent vector at base (must satisfy ⟨base, v⟩ = 0).
        Returns:
            (..., dim+1) point on the sphere.
        """
        sqrt_K  = math.sqrt(self.curvature)
        v_norm  = tangent.norm(dim=-1, keepdim=True)              # (..., 1)
        v_scaled = sqrt_K * v_norm                                # √K·‖v‖
        safe_v  = v_scaled.clamp(min=1e-8)                        # avoid ÷0
        return (
            torch.cos(v_scaled) * base
            + torch.sin(v_scaled) / safe_v * tangent
        )

    def log_map(
        self,
        base: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Logarithmic map on S^n_K: maps a manifold point to the tangent space at `base`.

        log_p(q)  =  θ / sin(θ) · (q - cos(θ) · p)
        where θ = d_S(p, q) · √K  ∈ [0, π)

        NOTE: Stub for Phase 3+ full tangent-space pipeline.
              Phase 1 uses the simplified identity projection instead.

        Args:
            base:  (..., dim+1) base point on the sphere.
            point: (..., dim+1) target point on the sphere.
        Returns:
            (..., dim+1) tangent vector at `base`.
        """
        sqrt_K = math.sqrt(self.curvature)
        dist   = self.spherical_distance(base, point)   # (...,)
        theta  = (sqrt_K * dist).unsqueeze(-1)          # (..., 1)
        cos_t  = torch.cos(theta)
        # sin(theta)/theta → 1 as theta → 0; avoid division by zero
        sin_t  = torch.sin(theta).clamp(min=1e-8)
        return (theta / sin_t) * (point - cos_t * base)

    # ─────────────────────────────── extras ──────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, dim={self.dim}, "
            f"curvature={self.curvature:.4f}, init_scale={self.init_scale}"
        )
