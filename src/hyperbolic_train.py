"""
Hyperbolic pre-training loop.

Mirrors src/train.py exactly, with three additions:
  1. Uses HyperbolicTransformerLM instead of TransformerLM.
  2. Instantiates GeometryMonitor and logs geometry stats every log_every_steps.
  3. Logs embedding gradient norms after each backward pass.

Everything else — mixed precision, gradient accumulation, LR scheduling,
OOM retry, checkpointing, evaluation, generation, visualisation — is
identical to the Euclidean training loop.

This file is intentionally self-contained (no imports from train.py) so that:
  a) The two pipelines can be run independently and compared.
  b) Changes to one loop do not inadvertently affect the other.
  c) Checkpoint directories can be kept separate.
"""

import dataclasses
import os
import gc
import sys
import time
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from dataclasses import asdict

from .config import HyperbolicModelConfig, TrainConfig, load_hyperbolic_model_config, load_train_config, get_hyperbolic_model_config
from .hyperbolic_model import HyperbolicTransformerLM
from .geometry import GeometryMonitor
from .tokenizer import LLMTokenizer
from .data import create_dataloader
from .utils import (
    detect_environment, get_device, get_dtype, should_compile, set_seed,
    get_lr, save_checkpoint, load_checkpoint, has_checkpoint,
    MetricsTracker, Timer,
)
from .visualize import generate_all_visualizations


def train_hyperbolic(
    model_config: HyperbolicModelConfig,
    train_config: TrainConfig,
):
    """
    Full pre-training loop for HyperbolicTransformerLM.

    Identical to train() in train.py except:
      - Model class: HyperbolicTransformerLM
      - GeometryMonitor is instantiated and called at logging intervals
      - Geometry stats are included in wandb / console output
      - Checkpoint metadata includes geometry_type and curvature
    """
    env = detect_environment()
    device = get_device()
    dtype = get_dtype(train_config.precision)
    print(
        f"[Hyperbolic] Environment: {env} | Device: {device} | Dtype: {dtype}\n"
        f"[Hyperbolic] Geometry: {model_config.geometry_type} "
        f"| Curvature K={model_config.curvature} "
        f"| Init σ={model_config.embed_init_scale}"
    )

    set_seed(train_config.seed)

    # ── Model ──
    model = HyperbolicTransformerLM(model_config).to(device)
    print(model.summary())

    if should_compile(train_config.use_compile):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if train_config.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # ── Geometry monitor ──
    monitor = GeometryMonitor(max_sample_vocab=1_000)

    # ── Optimizer ──
    # Same decay / no-decay split as the Euclidean loop.
    # spatial_coords.weight is 2D → enters decay group.
    # This may be overridden in Phase 2 via embed_weight_decay_override.
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
        fused=device.type == "cuda",
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

    # ── Tokenizer ──
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
        "geometry_type": model_config.geometry_type,
        "curvature": model_config.curvature,
        "embed_init_scale": model_config.embed_init_scale,
        "param_count": (
            model.count_parameters()
            if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
            else sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    })

    # ── Resume ──
    start_step = 0
    best_val_loss = float("inf")
    val_loss = None

    gdrive_kw = dict(
        gdrive_folder_id=train_config.gdrive_folder_id if train_config.backup_to_gdrive else "",
        gdrive_credentials_path=train_config.gdrive_credentials_path,
    )

    if has_checkpoint(train_config.checkpoint_dir, train_config.resume, **gdrive_kw):
        ckpt = load_checkpoint(
            train_config.checkpoint_dir, train_config.resume,
            model, optimizer, device, **gdrive_kw,
        )
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step}")

    # ── Mixed precision setup ──
    use_amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
    # GradScaler is only meaningful for fp16; bf16 has enough range without scaling.
    scaler = GradScaler(enabled=(dtype == torch.float16))
    amp_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if use_amp
        else nullcontext()
    )

    # ── Training state ──
    tokens_per_step = (
        train_config.batch_size
        * model_config.max_seq_len
        * train_config.gradient_accumulation_steps
    )

    print(f"\n{'='*60}")
    print(f"[Hyperbolic] Starting training from step {start_step}")
    print(f"  Geometry:        {model_config.geometry_type}  K={model_config.curvature}")
    print(f"  Steps:           {start_step} → {train_config.max_steps}")
    print(f"  Tokens/step:     {tokens_per_step:,}")
    print(f"  Effective batch: {train_config.batch_size * train_config.gradient_accumulation_steps} sequences")
    print(f"{'='*60}\n")

    model.train()
    running_loss = 0.0
    train_iter = iter(train_loader)
    timer = Timer()

    for step in range(start_step, train_config.max_steps):
        t0 = time.time()

        # Learning rate schedule (cosine with linear warmup — identical to Euclidean loop)
        lr = get_lr(
            step, train_config.warmup_steps, train_config.max_steps,
            train_config.peak_lr, train_config.min_lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ──
        micro_batches = []
        for _ in range(train_config.gradient_accumulation_steps):
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets = next(train_iter)
            micro_batches.append((input_ids, targets))

        # OOM-safe accumulation loop (mirrors train.py exactly)
        seq_cap = model_config.max_seq_len
        oom_retries = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            try:
                for ids, tgts in micro_batches:
                    ids  = ids[:, :seq_cap].to(device, non_blocking=True)
                    tgts = tgts[:, :seq_cap].to(device, non_blocking=True)
                    with amp_ctx:
                        out  = model(ids, targets=tgts)
                        loss = out["loss"] / train_config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    accum_loss += loss.item()
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                new_cap = max(train_config.min_seq_len, seq_cap * 3 // 4)
                if new_cap == seq_cap:
                    raise RuntimeError(
                        f"OOM at step {step} even with minimum seq_len={seq_cap}. "
                        "Enable gradient_checkpointing or reduce batch_size."
                    )
                oom_retries += 1
                seq_cap = new_cap
                print(f"  ⚠ OOM at step {step} — retry {oom_retries} with seq_cap={seq_cap}")

        # ── Geometry: gradient norms (before optimizer.zero_grad) ────────────
        # Gradients are available here (after backward, before zeroing).
        # We sample these cheaply — no manifold operations needed.
        grad_stats = monitor.compute_grad_norms(
            tok_emb=model.tok_emb if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
                    else model._orig_mod.tok_emb,
            geometry_type=model_config.geometry_type,
        )

        # ── Gradient clipping ──
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
            perplexity = math.exp(min(avg_loss, 20))

            # ── Geometry: manifold stats ─────────────────────────────────────
            _tok_emb = (
                model.tok_emb
                if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
                else model._orig_mod.tok_emb
            )
            geo_stats = monitor.compute_stats(
                tok_emb=_tok_emb,
                geometry_type=model_config.geometry_type,
                curvature=model_config.curvature,
            )

            metrics = {
                "train/loss":          avg_loss,
                "train/perplexity":    perplexity,
                "train/lr":            lr,
                "train/tokens_per_sec": tok_per_sec,
                "train/tokens_seen":   step * tokens_per_step,
                "train/elapsed_hours": timer.elapsed() / 3600,
                **geo_stats,
                **grad_stats,
            }
            tracker.log(metrics, step)

            print(
                f"step {step:>6d} | loss {avg_loss:.4f} | ppl {perplexity:.1f} | "
                f"lr {lr:.2e} | tok/s {tok_per_sec:,.0f} | "
                f"emb_norm {geo_stats.get('geometry/spatial_norm_mean', 0):.4f} | "
                f"nan={geo_stats.get('geometry/has_nan', 0):.0f} "
                f"inf={geo_stats.get('geometry/has_inf', 0):.0f} | "
                f"{timer.elapsed() / 3600:.1f}h"
            )
            running_loss = 0.0

        # ── Validation ──
        if step % train_config.eval_every_steps == 0 and step > 0:
            val_loss = _evaluate(model, val_loader, device, amp_ctx)
            val_ppl = math.exp(min(val_loss, 20))
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            tracker.log({
                "val/loss":       val_loss,
                "val/perplexity": val_ppl,
                "val/best_loss":  best_val_loss,
            }, step)
            print(f"  → val loss {val_loss:.4f} | ppl {val_ppl:.1f} {'★ best' if is_best else ''}")

            model.train()

        # ── Checkpoint ──
        if step % train_config.save_every_steps == 0 and step > 0:
            is_best = (
                step % train_config.eval_every_steps == 0
                and val_loss is not None
                and best_val_loss == val_loss
            )
            save_checkpoint(
                model, optimizer, step, running_loss,
                model_config, train_config,
                train_config.checkpoint_dir,
                is_best=is_best,
            )
            print(f"  → Saved checkpoint at step {step}")

        # ── Sample generation ──
        if step % train_config.generate_every_steps == 0 and step > 0:
            _generate_samples(model, tokenizer, train_config.sample_prompts, device, tracker, step)
            model.train()

        # ── Model internals visualisation ──
        if step % train_config.visualize_every_steps == 0 and step > 0:
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
    print(f"\n[Hyperbolic] Training complete! Final checkpoint at step {train_config.max_steps}")
    tracker.finish()


# ─────────────────────────────── helpers ──────────────────────────────────────

@torch.no_grad()
def _evaluate(
    model: HyperbolicTransformerLM,
    val_loader,
    device: torch.device,
    amp_ctx,
    max_batches: int = 50,
) -> float:
    """Run validation and return average cross-entropy loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for input_ids, targets in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets   = targets.to(device, non_blocking=True)

        with amp_ctx:
            out = model(input_ids, targets=targets)
            total_loss += out["loss"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _generate_samples(
    model: HyperbolicTransformerLM,
    tokenizer: LLMTokenizer,
    prompts: list,
    device: torch.device,
    tracker,
    step: int,
) -> None:
    """Generate text samples from prompts and log them."""
    model.eval()
    table_data = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        output_ids = model.generate(
            input_tensor, max_new_tokens=128, temperature=0.8,
            top_k=50, top_p=0.9, eos_token_id=tokenizer.eos_id,
        )
        generated_text = tokenizer.decode(output_ids[0].tolist())
        table_data.append({"prompt": prompt, "generated": generated_text})
        print(f"  [Gen] {prompt}")
        print(f"         → {generated_text[:200]}...")

    if tracker.use_wandb and tracker.wandb:
        import wandb
        table = wandb.Table(
            columns=["prompt", "generated"],
            data=[[d["prompt"], d["generated"]] for d in table_data],
        )
        tracker.log({"samples/generations": table}, step)


# ─────────────────────────────── CLI entry point ──────────────────────────────

def main():
    """CLI entry point for hyperbolic pre-training."""
    import argparse

    parser = argparse.ArgumentParser(description="Pre-train HyperbolicTransformerLM")
    parser.add_argument(
        "--model", type=str, default="tiny",
        help="Hyperbolic model config name (tiny/small/medium/base/large) or path to YAML",
    )
    parser.add_argument("--config", type=str, default="",
                        help="Path to training config YAML")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint: 'latest', 'best', or 'step_N'")

    # Geometry overrides
    parser.add_argument("--geometry", type=str, default=None,
                        help="Override geometry_type: lorentz | spherical | euclidean")
    parser.add_argument("--curvature", type=float, default=None,
                        help="Override curvature K > 0")

    # Standard training overrides
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
        model_config = load_hyperbolic_model_config(args.model)
    else:
        model_config = get_hyperbolic_model_config(args.model)

    # Apply geometry CLI overrides
    if args.geometry is not None:
        model_config = dataclasses.replace(model_config, geometry_type=args.geometry)
    if args.curvature is not None:
        model_config = dataclasses.replace(model_config, curvature=args.curvature)

    # Load training config
    if args.config and os.path.isfile(args.config):
        train_config = load_train_config(args.config)
    else:
        train_config = TrainConfig()

    # Apply standard CLI overrides
    if args.resume is not None and args.resume:
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

    train_hyperbolic(model_config, train_config)


if __name__ == "__main__":
    main()
