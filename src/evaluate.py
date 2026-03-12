"""
Evaluation: validation loss, perplexity, and LM benchmarks (via lm-eval-harness).
"""

import os
import math
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from .config import ModelConfig, TrainConfig, get_model_config, load_model_config
from .model import TransformerLM
from .tokenizer import LLMTokenizer
from .data import create_dataloader
from .utils import get_device, get_dtype, load_checkpoint


@torch.no_grad()
def compute_val_metrics(
    model: TransformerLM,
    val_loader,
    device: torch.device,
    amp_ctx,
    max_batches: int = 200,
) -> dict:
    """Compute full validation metrics."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for input_ids, targets in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp_ctx:
            out = model(input_ids, targets=targets)

        # Use reduction="sum" (not "mean") to get the true per-token loss.
        # "mean" would average over each batch independently, weighting shorter
        # sequences equally with longer ones. "sum" lets us accumulate total loss
        # and total token count, then divide once for an exact global average.
        logits = out["logits"]
        loss_flat = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,   # Skip padding tokens (target=-1) so they don't dilute the loss
            reduction="sum",
        )
        n_valid = (targets != -1).sum().item()
        total_loss += loss_flat.item()
        total_tokens += n_valid
        n_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    # Perplexity = exp(avg_loss). Intuitively, it's the effective vocabulary size
    # the model is "confused" among per token. Lower is better.
    # Clamp avg_loss at 20 to avoid exp(>20) overflow (~485 million).
    perplexity = math.exp(min(avg_loss, 20))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "tokens_evaluated": total_tokens,
        "batches": n_batches,
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    model_config_name: str = "tiny",
    data_dir: str = "data",
    tokenizer_path: str = "tokenizer_data",
    batch_size: int = 32,
    max_batches: int = 200,
):
    """Evaluate a saved checkpoint on validation data."""
    device = get_device()
    dtype = get_dtype()

    # Load model config
    if os.path.isfile(model_config_name):
        model_config = load_model_config(model_config_name)
    else:
        model_config = get_model_config(model_config_name)

    # Build model
    model = TransformerLM(model_config).to(device)

    # Load checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    resume_name = os.path.basename(checkpoint_path).replace(".pt", "")
    load_checkpoint(checkpoint_dir, resume_name, model, device=device)

    # Data
    val_loader = create_dataloader(data_dir, model_config.max_seq_len, batch_size, "val", shuffle=False)

    # Evaluate
    use_amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()

    metrics = compute_val_metrics(model, val_loader, device, amp_ctx, max_batches)

    print(f"\nEvaluation Results:")
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
    parser.add_argument("--model", type=str, default="tiny", help="Model config name or YAML path")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer_data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=200)

    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint, args.model, args.data_dir,
        args.tokenizer_path, args.batch_size, args.max_batches,
    )


if __name__ == "__main__":
    main()
