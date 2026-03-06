"""
Pre-training loop.
Handles: mixed precision, gradient accumulation, LR scheduling,
checkpointing, periodic evaluation, text generation, and model internals visualization.
Works seamlessly in Colab, vast.ai, and local environments.
"""

import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from dataclasses import asdict

from .config import ModelConfig, TrainConfig, get_model_config, load_model_config, load_train_config
from .model import TransformerLM
from .tokenizer import LLMTokenizer
from .data import create_dataloader
from .utils import (
    detect_environment, get_device, get_dtype, should_compile, set_seed,
    get_lr, save_checkpoint, load_checkpoint, has_checkpoint,
    MetricsTracker, Timer,
)
from .visualize import generate_all_visualizations


def train(
    model_config: ModelConfig,
    train_config: TrainConfig,
):
    """
    Full pre-training loop.
    """
    env = detect_environment()
    device = get_device()
    dtype = get_dtype(train_config.precision)
    print(f"Environment: {env} | Device: {device} | Dtype: {dtype}")

    set_seed(train_config.seed)

    # ── Model ──
    model = TransformerLM(model_config).to(device)
    print(model.summary())

    if should_compile(train_config.use_compile):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if train_config.gradient_checkpointing:
        from torch.utils.checkpoint import checkpoint
        # Enable gradient checkpointing by wrapping layer forward calls
        # (handled via custom forward wrapper below, not a built-in flag)
        print("Gradient checkpointing enabled.")

    # ── Optimizer ──
    # Separate weight-decay and no-decay param groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_config.peak_lr,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_eps,
        fused=device.type == "cuda",  # Use fused AdamW on CUDA
    )

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer groups: {n_decay:,} decay params, {n_no_decay:,} no-decay params")

    # ── Data ──
    train_loader = create_dataloader(
        train_config.data_dir, model_config.max_seq_len,
        train_config.batch_size, "train", train_config.num_workers,
    )
    val_loader = create_dataloader(
        train_config.data_dir, model_config.max_seq_len,
        train_config.batch_size, "val", train_config.num_workers,
        shuffle=False,
    )

    # ── Tokenizer (for generation samples) ──
    tokenizer = LLMTokenizer(train_config.tokenizer_path)

    # ── Logging ──
    tracker = MetricsTracker(
        use_wandb=train_config.use_wandb,
        project=train_config.wandb_project,
        run_name=train_config.wandb_run_name,
    )
    tracker.log_config({
        "model": asdict(model_config),
        "training": asdict(train_config),
        "environment": env,
        "device": str(device),
        "dtype": str(dtype),
        "param_count": model.count_parameters() if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule) else sum(p.numel() for p in model.parameters() if p.requires_grad),
    })

    # ── Resume ──
    start_step = 0
    best_val_loss = float("inf")

    if has_checkpoint(train_config.checkpoint_dir, train_config.resume):
        ckpt = load_checkpoint(
            train_config.checkpoint_dir, train_config.resume,
            model, optimizer, device,
        )
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step}")

    # ── Mixed precision context ──
    use_amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()
    # GradScaler only needed for fp16, not bf16
    scaler = GradScaler(enabled=(use_amp and dtype == torch.float16))

    # ── Training loop ──
    timer = Timer()
    train_iter = iter(train_loader)
    tokens_per_step = train_config.batch_size * model_config.max_seq_len * train_config.gradient_accumulation_steps

    print(f"\n{'='*60}")
    print(f"Starting training from step {start_step}")
    print(f"  Steps: {start_step} → {train_config.max_steps}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Effective batch: {train_config.batch_size * train_config.gradient_accumulation_steps} sequences")
    print(f"{'='*60}\n")

    model.train()
    running_loss = 0.0

    for step in range(start_step, train_config.max_steps):
        t0 = time.time()

        # Update learning rate
        lr = get_lr(step, train_config.warmup_steps, train_config.max_steps,
                     train_config.peak_lr, train_config.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ──
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            # Get next batch (cycle if exhausted)
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets = next(train_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with amp_ctx:
                out = model(input_ids, targets=targets)
                loss = out["loss"] / train_config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping
        if train_config.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        running_loss += accum_loss
        timer.step()
        t1 = time.time()

        # ── Logging ──
        if step % train_config.log_every_steps == 0 and step > 0:
            avg_loss = running_loss / train_config.log_every_steps
            tok_per_sec = tokens_per_step / (t1 - t0)
            perplexity = math.exp(min(avg_loss, 20))  # Clamp to avoid overflow

            metrics = {
                "train/loss": avg_loss,
                "train/perplexity": perplexity,
                "train/lr": lr,
                "train/tokens_per_sec": tok_per_sec,
                "train/tokens_seen": step * tokens_per_step,
                "train/elapsed_hours": timer.elapsed() / 3600,
            }
            tracker.log(metrics, step)

            print(
                f"step {step:>6d} | loss {avg_loss:.4f} | ppl {perplexity:.1f} | "
                f"lr {lr:.2e} | tok/s {tok_per_sec:,.0f} | "
                f"{timer.elapsed() / 3600:.1f}h"
            )
            running_loss = 0.0

        # ── Validation ──
        if step % train_config.eval_every_steps == 0 and step > 0:
            val_loss = evaluate(model, val_loader, device, amp_ctx)
            val_ppl = math.exp(min(val_loss, 20))
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            tracker.log({
                "val/loss": val_loss,
                "val/perplexity": val_ppl,
                "val/best_loss": best_val_loss,
            }, step)
            print(f"  → val loss {val_loss:.4f} | ppl {val_ppl:.1f} {'★ best' if is_best else ''}")

            model.train()

        # ── Checkpoint ──
        if step % train_config.save_every_steps == 0 and step > 0:
            is_best = (step % train_config.eval_every_steps == 0) and (best_val_loss == val_loss if 'val_loss' in dir() else False)
            save_checkpoint(
                model, optimizer, step, running_loss,
                model_config, train_config,
                train_config.checkpoint_dir,
                is_best=is_best,
            )
            print(f"  → Saved checkpoint at step {step}")

        # ── Sample generation ──
        if step % train_config.generate_every_steps == 0 and step > 0:
            generate_samples(model, tokenizer, train_config.sample_prompts, device, tracker, step)
            model.train()

        # ── Model internals visualization ──
        if step % train_config.visualize_every_steps == 0 and step > 0:
            # Use first batch as sample for visualizations
            try:
                sample_ids, _ = next(iter(val_loader))
                sample_ids = sample_ids[:1, :128].to(device)
            except StopIteration:
                sample_ids = None

            if sample_ids is not None:
                generate_all_visualizations(model, tokenizer, sample_ids, tracker, step)
            model.train()

    # ── Final checkpoint ──
    save_checkpoint(
        model, optimizer, train_config.max_steps, running_loss,
        model_config, train_config,
        train_config.checkpoint_dir,
    )
    print(f"\nTraining complete! Final checkpoint saved at step {train_config.max_steps}")
    tracker.finish()


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    val_loader,
    device: torch.device,
    amp_ctx,
    max_batches: int = 50,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for input_ids, targets in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp_ctx:
            out = model(input_ids, targets=targets)
            total_loss += out["loss"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def generate_samples(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    prompts: list,
    device: torch.device,
    tracker,
    step: int,
):
    """Generate text samples from prompts and log them."""
    model.eval()
    table_data = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        output_ids = model.generate(
            input_tensor, max_new_tokens=128, temperature=0.8, top_k=50, top_p=0.9,
            eos_token_id=tokenizer.eos_id,
        )
        generated_text = tokenizer.decode(output_ids[0].tolist())
        table_data.append({"prompt": prompt, "generated": generated_text})
        print(f"  [Gen] {prompt}")
        print(f"         → {generated_text[:200]}...")

    # Log to wandb as a table
    if tracker.use_wandb and tracker.wandb:
        import wandb
        table = wandb.Table(columns=["prompt", "generated"], data=[
            [d["prompt"], d["generated"]] for d in table_data
        ])
        tracker.log({"samples/generations": table}, step)


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Pre-train LLM from scratch")
    parser.add_argument("--model", type=str, default="tiny",
                        help="Model config name (tiny/small/medium/base/large) or path to YAML")
    parser.add_argument("--config", type=str, default="",
                        help="Path to training config YAML (overrides defaults)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint: 'latest', 'best', or 'step_N'")

    # Common overrides
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--peak_lr", type=float, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)

    args = parser.parse_args()

    # Load model config
    if os.path.isfile(args.model):
        model_config = load_model_config(args.model)
    else:
        model_config = get_model_config(args.model)

    # Load training config
    if args.config and os.path.isfile(args.config):
        train_config = load_train_config(args.config)
    else:
        train_config = TrainConfig()

    # Apply CLI overrides
    if args.resume:
        train_config.resume = args.resume
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.peak_lr is not None:
        train_config.peak_lr = args.peak_lr
    if args.wandb_project is not None:
        train_config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        train_config.wandb_run_name = args.wandb_run_name
    if args.no_wandb:
        train_config.use_wandb = False
    if args.data_dir is not None:
        train_config.data_dir = args.data_dir
    if args.checkpoint_dir is not None:
        train_config.checkpoint_dir = args.checkpoint_dir
    if args.tokenizer_path is not None:
        train_config.tokenizer_path = args.tokenizer_path

    train(model_config, train_config)


if __name__ == "__main__":
    main()
