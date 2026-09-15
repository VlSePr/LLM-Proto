"""
Evaluation: validation loss and perplexity.
"""

import math
import torch
from contextlib import nullcontext
from typing import Optional

from .model import IGNORE_INDEX
from .data import create_dataloader
from .utils import get_device, get_dtype, build_model_from_checkpoint


@torch.no_grad()
def compute_val_metrics(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    amp_ctx=None,
    max_batches: int = 200,
) -> dict:
    """Compute token-weighted validation loss and perplexity.

    The model's own loss (a mean over non-ignored tokens, computed inside the
    autocast context) is re-weighted by the number of valid tokens in each batch
    so the result is an exact global per-token average, not a mean of batch means.
    Accumulation happens on-device; the host is synchronised once at the end.
    """
    if amp_ctx is None:
        amp_ctx = nullcontext()
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = torch.zeros((), dtype=torch.int64, device=device)
    n_batches = 0

    for input_ids, targets in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp_ctx:
            out = model(input_ids, targets=targets)

        n_valid = (targets != IGNORE_INDEX).sum()
        total_loss += out["loss"].detach().double() * n_valid
        total_tokens += n_valid
        n_batches += 1

    if was_training:
        model.train()

    tokens_evaluated = int(total_tokens.item())
    avg_loss = float(total_loss.item() / max(tokens_evaluated, 1))
    # Perplexity = exp(avg_loss). Intuitively, it's the effective vocabulary size
    # the model is "confused" among per token. Lower is better.
    # Clamp avg_loss at 20 to avoid exp(>20) overflow (~485 million).
    perplexity = math.exp(min(avg_loss, 20))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "tokens_evaluated": tokens_evaluated,
        "batches": n_batches,
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    model_config_name: Optional[str] = None,
    data_dir: str = "data",
    batch_size: int = 32,
    max_batches: int = 200,
    num_workers: int = 0,
):
    """Evaluate a saved checkpoint on validation data.

    The architecture is read from the checkpoint; ``model_config_name`` is only a
    fallback for checkpoints that predate the ``model_config`` entry.
    """
    device = get_device()
    dtype = get_dtype()

    model, _ = build_model_from_checkpoint(checkpoint_path, device, model_config_name)
    model_config = model.config

    # Data
    val_loader = create_dataloader(
        data_dir, model_config.max_seq_len, batch_size, "val", num_workers=num_workers, shuffle=False,
    )

    # Evaluate
    use_amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()

    metrics = compute_val_metrics(model, val_loader, device, amp_ctx, max_batches)

    print("\nEvaluation Results:")
    print(f"  Loss:       {metrics['loss']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Tokens:     {metrics['tokens_evaluated']:,}")
    return metrics


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model config name or YAML path (only needed for checkpoints without model_config)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint, args.model, args.data_dir,
        args.batch_size, args.max_batches, args.num_workers,
    )


if __name__ == "__main__":
    main()
