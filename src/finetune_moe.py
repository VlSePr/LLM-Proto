"""
Mixture-of-Experts fine-tuning on top of a pre-trained dense checkpoint.

    python -m src.finetune_moe --base_checkpoint checkpoints/latest.pt \\
        --expert_layers 24-31 --n_experts 1 --top_k 1 --config configs/finetune_moe.yaml

Incremental growth (freeze the trained experts, add one more, train only the new one):

    python -m src.finetune_moe --base_checkpoint checkpoints/latest.pt \\
        --expand_from checkpoints/expert/latest.pt --n_new 1 --top_k 2 \\
        --config configs/finetune_moe.yaml --checkpoint_dir checkpoints/expert_r2

Everything after grafting is the ordinary ``src.train.train`` loop: AMP, gradient
accumulation with OOM replay, token-weighted validation, ``save_checkpoint`` with
``latest.pt``/``best.pt``, optional Drive backup. Checkpoints carry ``ckpt["moe"]`` so
``python -m src.generate`` / ``src.evaluate`` load them unchanged.

Set ``PYTORCH_ALLOC_CONF=expandable_segments:True`` in the environment *before*
launching when VRAM is tight; it must precede the first CUDA allocation.
"""

from __future__ import annotations

import os

import torch

from .config import MoEConfig, TrainConfig
from .model import TransformerLM
from .moe import expand_moe_experts, freeze_for_moe, graft_moe, moe_metadata, param_budget_report
from .train import add_train_override_args, train, train_config_from_args
from .utils import build_model_from_checkpoint, get_device


def parse_layer_spec(spec: str) -> list[int]:
    """``"24-31"``, ``"1,3,5"`` or ``"0-3,8"`` -> sorted list of layer indices."""
    layers: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError(f"empty layer spec: {spec!r}")
    return sorted(layers)


def finetune_moe(
    base_checkpoint: str,
    moe_config: MoEConfig | None,
    train_config: TrainConfig,
    *,
    expand_from: str | None = None,
    n_new: int = 1,
    tokenizer_from_checkpoint: bool = True,
    device: torch.device | None = None,
    stop_after_step: int | None = None,
) -> tuple[TransformerLM, list[dict]]:
    """Graft (or grow) MoE layers onto a checkpointed model and fine-tune them.

    Round 1 (``expand_from`` is None): load ``base_checkpoint`` (dense), graft
    ``moe_config`` onto it with warm-started experts, freeze everything else.

    Growth round (``expand_from`` given): load that MoE checkpoint instead, freeze it,
    then ``expand_moe_experts(n_new, top_k=moe_config.top_k)``; ``moe_config`` may be
    ``None`` to keep the checkpoint's ``top_k``.

    ``tokenizer_from_checkpoint`` makes ``train_config.tokenizer_path`` follow the base
    checkpoint's stored value (the tokenizer the model was trained with), unless the
    caller passed an explicit path.

    Returns ``(model, history)`` where ``history`` is the trainer's metric log.
    """
    if device is None:
        device = get_device()

    if expand_from:
        model, ckpt = build_model_from_checkpoint(expand_from, device)
        if moe_metadata(model) is None:
            raise ValueError(f"--expand_from checkpoint {expand_from} has no MoE layers")
        freeze_for_moe(model)
        top_k = moe_config.top_k if moe_config is not None else None
        n = expand_moe_experts(model, n_new, warm_start=(moe_config.warm_start if moe_config else True), top_k=top_k)
        print(f"Expanded MoE layers to {n} experts (top_k={moe_metadata(model)['top_k']})")
    else:
        if moe_config is None:
            raise ValueError("moe_config is required for a first fine-tuning round")
        model, ckpt = build_model_from_checkpoint(base_checkpoint, device)
        if moe_metadata(model) is not None:
            raise ValueError(f"{base_checkpoint} already has MoE layers; use expand_from to grow it")
        graft_moe(model, moe_config)
        freeze_for_moe(model)

    if tokenizer_from_checkpoint:
        stored = (ckpt.get("train_config") or {}).get("tokenizer_path")
        if stored and os.path.isdir(stored):
            train_config.tokenizer_path = stored

    print(param_budget_report(model))
    history = train(model.config, train_config, model=model, stop_after_step=stop_after_step)
    return model, history


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MoE fine-tuning on a pre-trained checkpoint")
    parser.add_argument("--base_checkpoint", type=str, required=True,
                        help="Dense checkpoint to graft experts onto (.pt path)")
    parser.add_argument("--expert_layers", type=str, default=None,
                        help="Layer indices to convert, e.g. '24-31' or '1,3,5' (round 1 only)")
    parser.add_argument("--n_experts", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=None,
                        help="Experts per token (default 1; on --expand_from, default keeps the checkpoint's)")
    parser.add_argument("--no_warm_start", action="store_true", help="Random experts instead of copies of the FFN")
    parser.add_argument("--expand_from", type=str, default=None,
                        help="MoE checkpoint to grow instead of grafting from scratch")
    parser.add_argument("--n_new", type=int, default=1, help="Experts to add per layer with --expand_from")
    add_train_override_args(parser)
    args = parser.parse_args()

    train_config = train_config_from_args(args, parser)

    if args.expand_from:
        moe_config = None
        if args.top_k is not None:
            layers = parse_layer_spec(args.expert_layers) if args.expert_layers else [0]
            # only top_k / warm_start are read from this config in a growth round
            moe_config = MoEConfig(expert_layers=layers, n_experts=max(args.top_k, 1), top_k=args.top_k,
                                   warm_start=not args.no_warm_start)
    else:
        if not args.expert_layers:
            parser.error("--expert_layers is required unless --expand_from is given")
        moe_config = MoEConfig(
            expert_layers=parse_layer_spec(args.expert_layers),
            n_experts=args.n_experts,
            top_k=args.top_k if args.top_k is not None else 1,
            warm_start=not args.no_warm_start,
        )

    finetune_moe(
        args.base_checkpoint, moe_config, train_config,
        expand_from=args.expand_from, n_new=args.n_new,
        tokenizer_from_checkpoint=args.tokenizer_path is None,
    )


if __name__ == "__main__":
    main()
