"""
src/geometry — Curved embedding space primitives for LLM-Proto.

Phase 1 contents
----------------
LorentzEmbedding   — token embeddings on the hyperboloid H^n_K (negative curvature)
SphericalEmbedding — token embeddings on the n-sphere S^n_K (positive curvature)
GeometryMonitor    — non-intrusive runtime diagnostic tool for manifold health

All embeddings expose the same interface as nn.Embedding:
    forward(input_ids: LongTensor) -> FloatTensor[B, T, dim]

They feed the unchanged Euclidean Transformer via a tangent-space projection
at the manifold origin (Phase 1: identity on spatial coords — no log-map cost).

Phase 2+ stubs (present but not called during Phase 1 training):
    LorentzEmbedding.lorentz_distance()
    LorentzEmbedding.exp_map() / .log_map()
    SphericalEmbedding.spherical_distance()
    SphericalEmbedding.exp_map() / .log_map()
"""

from .lorentz import LorentzEmbedding
from .spherical import SphericalEmbedding
from .monitor import GeometryMonitor

__all__ = [
    "LorentzEmbedding",
    "SphericalEmbedding",
    "GeometryMonitor",
]
