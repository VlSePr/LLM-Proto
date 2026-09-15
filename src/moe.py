"""
Sparse Mixture-of-Experts (MoE) layers and the helpers that graft them onto a
pre-trained dense ``TransformerLM``.

The workflow this module supports is *post-hoc* MoE fine-tuning:

1. Train (or load) a dense model.
2. ``graft_moe`` replaces the FFN of selected blocks with a ``SparseMoE`` whose
   experts are warm-started from the dense FFN weights, so the model's output is
   unchanged at the moment of grafting.
3. ``freeze_for_moe`` freezes everything except the MoE parameters.
4. Train with ``src.train.train(model=...)``; ``TransformerLM.forward`` exposes the
   summed load-balance loss as ``out["aux_loss"]`` and the trainer adds
   ``aux_loss_coeff * aux_loss`` to the LM loss.
5. Optionally grow the model round by round with ``expand_moe_experts``.

Checkpoints written by ``save_checkpoint`` carry ``moe_metadata(model)`` under
``ckpt["moe"]`` so ``build_model_from_checkpoint`` can rebuild the same shape
before loading weights. Fresh MoE *pre-training* (MoE in ``__init__``) is out of
scope; presets and ``ModelConfig`` always describe the dense skeleton.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoEConfig


class ExpertFFN(nn.Module):
    """A single SwiGLU expert; identical layout to ``model.FeedForward``.

    The parameter names (``w_gate``, ``w_up``, ``w_down``) match ``FeedForward`` on
    purpose so ``expert.load_state_dict(base_ffn.state_dict())`` warm-starts an expert.
    """

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class SparseMoE(nn.Module):
    """Sparse Mixture-of-Experts FFN: a drop-in replacement for ``FeedForward``.

    Every token is routed independently to its ``top_k`` highest-scoring experts
    (softmax over a bias-free linear router); the outputs are combined with the
    renormalised router probabilities.

    During training the layer also computes the Switch-Transformer load-balance
    loss ``n_experts * sum_i(f_i * P_i)`` (``f_i`` = fraction of tokens dispatched to
    expert ``i``, ``P_i`` = mean router probability for expert ``i``) and stores it in
    ``self.aux_loss``. ``TransformerLM.forward`` sums these into ``out["aux_loss"]``.
    """

    def __init__(self, dim: int, ffn_dim: int, n_experts: int, top_k: int, dropout: float = 0.0):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k ({top_k}) must be between 1 and n_experts ({n_experts})")
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(dim, ffn_dim, dropout) for _ in range(n_experts)])
        # Overwritten on every forward; read by TransformerLM.forward.
        self.aux_loss: torch.Tensor = torch.zeros(())

    @property
    def dim(self) -> int:
        return self.router.in_features

    @property
    def ffn_dim(self) -> int:
        return self.experts[0].w_gate.out_features

    @property
    def dropout_p(self) -> float:
        d = self.experts[0].dropout
        return d.p if isinstance(d, nn.Dropout) else 0.0

    def route(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Router probabilities, renormalised top-k weights and top-k expert indices for ``(N, dim)`` tokens."""
        router_probs = F.softmax(self.router(tokens), dim=-1)          # (N, n_experts)
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # (N, top_k), sums to 1 per token
        return router_probs, topk_probs, topk_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        tokens = x.reshape(-1, D)
        N = tokens.shape[0]
        router_probs, topk_probs, topk_idx = self.route(tokens)

        if self.training:
            with torch.no_grad():
                dispatch = torch.zeros(N, self.n_experts, device=x.device, dtype=router_probs.dtype)
                dispatch.scatter_(1, topk_idx, 1.0)
                f = dispatch.mean(dim=0)                  # fraction of tokens per expert (no grad)
            P = router_probs.mean(dim=0)                  # mean soft probability per expert (grad)
            self.aux_loss = self.n_experts * (f * P).sum()
        else:
            self.aux_loss = torch.zeros((), device=x.device)

        output = torch.zeros_like(tokens)
        for expert_idx, expert in enumerate(self.experts):
            slot_mask = topk_idx == expert_idx            # (N, top_k)
            routed = slot_mask.any(dim=1)                 # (N,)
            if not routed.any():
                continue
            expert_out = expert(tokens[routed])                          # (n_routed, dim)
            weights = topk_probs[routed][slot_mask[routed]].unsqueeze(-1)  # (n_routed, 1)
            output[routed] += weights * expert_out
        return output.view(B, T, D)


# ──────────────────────────────────────────────
# Grafting / freezing / growing
# ──────────────────────────────────────────────

def moe_layers(model: nn.Module) -> list[tuple[int, SparseMoE]]:
    """``(layer_index, SparseMoE)`` pairs for every block whose FFN is a ``SparseMoE``."""
    return [(i, layer.ffn) for i, layer in enumerate(model.layers) if isinstance(layer.ffn, SparseMoE)]


def graft_moe(model: nn.Module, cfg: MoEConfig, *, warm_start: bool | None = None) -> list[int]:
    """Replace ``model.layers[i].ffn`` with a ``SparseMoE`` for every ``i`` in ``cfg.expert_layers``.

    With ``warm_start`` (default ``cfg.warm_start``) every expert is initialised from the
    dense FFN it replaces, so the model output is unchanged right after grafting
    (renormalised top-k weights sum to 1). Returns the grafted layer indices.
    """
    warm = cfg.warm_start if warm_start is None else warm_start
    n_layers = len(model.layers)
    for idx in cfg.expert_layers:
        if not 0 <= idx < n_layers:
            raise ValueError(f"expert_layers entry {idx} out of range for a {n_layers}-layer model")
        block = model.layers[idx]
        base = block.ffn
        if isinstance(base, SparseMoE):
            raise ValueError(f"layer {idx} already has a SparseMoE FFN")
        w = base.w_gate.weight
        dropout = base.dropout.p if isinstance(base.dropout, nn.Dropout) else 0.0
        moe = SparseMoE(w.shape[1], w.shape[0], cfg.n_experts, cfg.top_k, dropout).to(device=w.device, dtype=w.dtype)
        if warm:
            for expert in moe.experts:
                expert.load_state_dict(base.state_dict())
        block.ffn = moe
    return list(cfg.expert_layers)


def freeze_for_moe(model: nn.Module) -> dict[str, int]:
    """Freeze every parameter except those inside ``SparseMoE`` layers.

    Returns ``{"total", "trainable", "frozen"}`` parameter counts.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for _, moe in moe_layers(model):
        for p in moe.parameters():
            p.requires_grad_(True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def param_budget_report(model: nn.Module) -> str:
    """Human-readable parameter budget, including the per-layer MoE breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        "Parameter budget",
        f"  Total     : {total:>14,}  ({total / 1e6:.1f}M)",
        f"  Trainable : {trainable:>14,}  ({trainable / 1e6:.1f}M, {100 * trainable / max(total, 1):.1f}%)",
        f"  Frozen    : {total - trainable:>14,}",
    ]
    layers = moe_layers(model)
    if layers:
        _, first = layers[0]
        per_expert = sum(p.numel() for p in first.experts[0].parameters())
        lines.append(
            f"  MoE layers: {[i for i, _ in layers]} (n_experts={first.n_experts}, top_k={first.top_k}, "
            f"{per_expert / 1e6:.1f}M params per expert)"
        )
    return "\n".join(lines)


def expand_moe_experts(
    model: nn.Module,
    n_new: int = 1,
    *,
    warm_start: bool = True,
    warm_start_from: int = 0,
    top_k: int | None = None,
) -> int:
    """Grow every ``SparseMoE`` layer by ``n_new`` experts for an incremental-growth round.

    - Existing experts are frozen; the new ones are trainable and (with ``warm_start``)
      copied from ``experts[warm_start_from]``.
    - The router is widened in place: old rows are preserved, new rows are N(0, 1e-3)
      so the router starts neutral toward the new experts. The router stays trainable.
    - ``top_k`` (if given) is applied to every layer; it is never bumped implicitly.

    Returns the new number of experts per layer. Raises if the model has no MoE layers.
    """
    layers = moe_layers(model)
    if not layers:
        raise ValueError("expand_moe_experts: model has no SparseMoE layers (call graft_moe first)")
    if n_new < 1:
        raise ValueError("n_new must be >= 1")
    new_n = None
    for _, moe in layers:
        old_n = moe.n_experts
        new_n = old_n + n_new
        if top_k is not None and not 1 <= top_k <= new_n:
            raise ValueError(f"top_k ({top_k}) must be between 1 and n_experts ({new_n})")
        src = moe.experts[warm_start_from]
        w = moe.router.weight
        for p in moe.experts.parameters():
            p.requires_grad_(False)
        for _ in range(n_new):
            expert = ExpertFFN(moe.dim, moe.ffn_dim, moe.dropout_p).to(device=w.device, dtype=w.dtype)
            if warm_start:
                expert.load_state_dict(src.state_dict())
            expert.requires_grad_(True)
            moe.experts.append(expert)
        router = nn.Linear(moe.dim, new_n, bias=False).to(device=w.device, dtype=w.dtype)
        with torch.no_grad():
            router.weight[:old_n] = w
            nn.init.normal_(router.weight[old_n:], mean=0.0, std=1e-3)
        moe.router = router
        moe.n_experts = new_n
        if top_k is not None:
            moe.top_k = top_k
    return new_n


# ──────────────────────────────────────────────
# Metadata (checkpoint schema) and diagnostics
# ──────────────────────────────────────────────

def moe_metadata(model: nn.Module) -> dict | None:
    """MoE shape of a live model as plain Python, or ``None`` for a dense model.

    Stored in checkpoints as ``ckpt["moe"]`` and fed back through
    ``config_from_dict(MoEConfig, ...)`` by ``build_model_from_checkpoint``. Values are
    derived from the modules themselves (``len(experts)``, ``top_k``), so an expanded
    model is described correctly.
    """
    layers = moe_layers(model)
    if not layers:
        return None
    shapes = {(len(moe.experts), moe.top_k) for _, moe in layers}
    if len(shapes) != 1:
        raise ValueError(f"MoE layers disagree on (n_experts, top_k): {sorted(shapes)}")
    ((n_experts, top_k),) = shapes
    return {"expert_layers": [i for i, _ in layers], "n_experts": n_experts, "top_k": top_k}


@torch.no_grad()
def measure_expert_load(
    model: nn.Module,
    batches: Iterable,
    n_batches: int = 30,
    device: torch.device | None = None,
) -> dict[int, torch.Tensor]:
    """Tally how many token-slots each expert receives over ``n_batches`` batches.

    ``batches`` yields ``(input_ids, targets)`` pairs (a DataLoader works). Returns
    ``{layer_index: counts}`` with a CPU float tensor of length ``n_experts`` per MoE
    layer; each tensor sums to ``tokens_seen * top_k``.
    """
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    counts: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module: SparseMoE, inp, out):
            tokens = inp[0].reshape(-1, inp[0].shape[-1])
            _, _, topk_idx = module.route(tokens)
            c = torch.bincount(topk_idx.reshape(-1), minlength=module.n_experts).to(torch.float32).cpu()
            counts[layer_idx] = counts.get(layer_idx, torch.zeros(module.n_experts)) + c
        return hook

    for idx, moe in moe_layers(model):
        hooks.append(moe.register_forward_hook(make_hook(idx)))
    try:
        for i, batch in enumerate(batches):
            if i >= n_batches:
                break
            ids = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(ids.to(device))
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()
    return counts
