"""
Pre-training loop.
Handles: mixed precision, gradient accumulation, LR scheduling,
checkpointing, periodic evaluation, text generation, and model internals visualization.
Works seamlessly in Colab, vast.ai, and local environments.
"""

import gc
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from .config import ModelConfig, TrainConfig, get_model_config, load_model_config, load_train_config, save_config
from .data import create_dataloader
from .evaluate import compute_val_metrics
from .model import TransformerLM
from .moe import moe_metadata
from .tokenizer import LLMTokenizer
from .utils import (
    MetricsTracker,
    Timer,
    detect_environment,
    get_device,
    get_dtype,
    get_lr,
    has_checkpoint,
    load_checkpoint,
    make_grad_scaler,
    save_checkpoint,
    save_metrics_history,
    set_seed,
    should_compile,
    tokenizer_fingerprint,
    unwrap_model,
    upload_run_artifact,
    warn_if_tokenizer_mismatch,
)
from .visualize import generate_all_visualizations


def train(
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    model: TransformerLM | None = None,
    stop_after_step: int | None = None,
) -> list[dict]:
    """
    Full training loop (pre-training, or fine-tuning when ``model`` is given).

    Checkpoint semantics: a checkpoint's ``step`` is the last *completed* optimizer
    step, and a resumed run continues at ``step + 1``. The final checkpoint is
    therefore written with ``step = max_steps - 1``.

    Args:
        model: An already-built model to train instead of constructing one from
            ``model_config`` (e.g. a base model with MoE layers grafted on and the
            rest frozen; see ``src/finetune_moe.py``). Only parameters with
            ``requires_grad`` are optimised. If the model exposes ``out["aux_loss"]``
            it is added to the LM loss with weight ``train_config.aux_loss_coeff``.
        stop_after_step: If set, save a checkpoint and return once this step has
            completed (simulates an interruption; used by tests and budgeted runs).

    Returns:
        The list of logged metric dicts (``MetricsTracker.history``), one per log call.
    """
    env = detect_environment()
    device = get_device()
    dtype = get_dtype(train_config.precision)
    print(f"Environment: {env} | Device: {device} | Dtype: {dtype}")

    set_seed(train_config.seed)

    # Sequence length used for batches; fine-tuning may use less than the model's context.
    seq_len = train_config.seq_len or model_config.max_seq_len
    if not 1 <= seq_len <= model_config.max_seq_len:
        raise ValueError(f"seq_len={seq_len} must be in [1, max_seq_len={model_config.max_seq_len}]")

    # ── Tokenizer (for generation samples) — loaded first so we can validate vocab ──
    tokenizer = LLMTokenizer(train_config.tokenizer_path)
    if tokenizer.vocab_size > model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size} exceeds model vocab_size={model_config.vocab_size}. "
            "Set ModelConfig.vocab_size >= tokenizer vocab size."
        )
    if tokenizer.vocab_size != model_config.vocab_size:
        print(f"Note: tokenizer vocab ({tokenizer.vocab_size}) < model vocab ({model_config.vocab_size}); "
              f"{model_config.vocab_size - tokenizer.vocab_size} embedding rows will be unused.")
    tok_fingerprint = tokenizer_fingerprint(train_config.tokenizer_path)

    # ── Model ──
    if model is None:
        model = TransformerLM(model_config).to(device)
    else:
        model = model.to(device)
    print(model.summary())

    if train_config.gradient_checkpointing:
        # Enable before torch.compile so the compiled graph includes checkpointing.
        # Saves ~60% activation memory at cost of ~33% compute.
        model.enable_gradient_checkpointing()

    if should_compile(train_config.use_compile):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    base_model = unwrap_model(model)  # plain module for generate()/visualizations
    moe_meta = moe_metadata(base_model)  # None for dense models; stored in every checkpoint

    # ── Optimizer ──
    # Separate parameters into two groups:
    # 1. "Decay" params (2D+ weight matrices): get weight decay to prevent overfitting.
    # 2. "No-decay" params (1D: biases, norms, embeddings): excluded from decay because
    #    - LayerNorm/RMSNorm scale params should be free to grow.
    #    - Biases have too few params for regularization to matter.
    #    - Decaying norms can destabilize training.
    decay_params = []
    no_decay_params = []
    for name, param in base_model.named_parameters():
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
    # AdamW (decoupled weight decay) is chosen over Adam + L2 regularization because:
    # - In standard Adam, L2 penalty interacts with the adaptive learning rate per-param,
    #   making effective regularization inconsistent across parameters.
    # - AdamW applies weight decay *directly* to weights (w = w - lr*wd*w), independent of
    #   Adam's moment estimates, giving uniform regularization.
    # - This is the standard optimizer for all modern LLMs (GPT-3, LLaMA, Chinchilla).
    #
    # beta1=0.9: momentum decay — standard value, averages ~10 recent gradients.
    # beta2=0.95: second moment decay — lower than the default 0.999 because LLM training
    #   with large batch sizes produces noisier gradient variance estimates; 0.95 adapts
    #   faster to changing gradient magnitudes (used in GPT-3, Chinchilla, LLaMA papers).
    # fused=True on CUDA: uses a single fused kernel that updates all params in one pass,
    #   avoiding multiple kernel launches — typically 5-15% faster.
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

    # ── Mixed precision context ──
    # Automatic Mixed Precision (AMP) runs most operations in bf16/fp16 while keeping
    # master weights in fp32. This halves memory usage and doubles throughput on modern GPUs.
    use_amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()
    # GradScaler is only needed for fp16 — it scales the loss up before backward to prevent
    # gradient underflow in fp16, then unscales before optimizer step.
    # bf16 has the same exponent range as fp32 (8 bits), so it doesn't need scaling.
    scaler = make_grad_scaler(enabled=(use_amp and dtype == torch.float16))

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
        "param_count": base_model.count_parameters(),
    })

    # ── Human-readable config snapshot next to the checkpoints ──
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    model_config_yaml = os.path.join(train_config.checkpoint_dir, "model_config.yaml")
    train_config_yaml = os.path.join(train_config.checkpoint_dir, "train_config.yaml")
    save_config(model_config, model_config_yaml)
    save_config(train_config, train_config_yaml)
    run_gdrive_folder = train_config.gdrive_folder_id if train_config.backup_to_gdrive else ""
    upload_run_artifact(model_config_yaml, run_gdrive_folder, train_config.gdrive_credentials_path)
    upload_run_artifact(train_config_yaml, run_gdrive_folder, train_config.gdrive_credentials_path)

    tokens_per_full_step = train_config.batch_size * seq_len * train_config.gradient_accumulation_steps

    # ── Resume ──
    start_step = 0
    best_val_loss = float("inf")
    epoch = 0
    batches_in_epoch = 0   # global train batches consumed in the current epoch
    tokens_seen = 0

    gdrive_kw = dict(
        gdrive_folder_id=train_config.gdrive_folder_id if train_config.backup_to_gdrive else "",
        gdrive_credentials_path=train_config.gdrive_credentials_path,
    )

    if train_config.resume:
        if has_checkpoint(train_config.checkpoint_dir, train_config.resume, **gdrive_kw):
            ckpt = load_checkpoint(
                train_config.checkpoint_dir, train_config.resume,
                model, optimizer, device, scaler=scaler, **gdrive_kw,
            )
            start_step = ckpt["step"] + 1
            best_val_loss = ckpt.get("best_val_loss") or float("inf")
            epoch = int(ckpt.get("epoch") or 0)
            batches_in_epoch = int(ckpt.get("batches_in_epoch") or 0)
            tokens_seen = int(ckpt.get("tokens_seen") or start_step * tokens_per_full_step)
            print(f"Continuing at step {start_step} (epoch {epoch}, batch {batches_in_epoch} in epoch)")
            warn_if_tokenizer_mismatch(ckpt, train_config.tokenizer_path)
        elif train_config.resume_if_exists:
            print(f"No checkpoint '{train_config.resume}' found in {train_config.checkpoint_dir}; "
                  "starting from scratch (resume_if_exists=True)")
        else:
            raise FileNotFoundError(
                f"resume='{train_config.resume}' but no such checkpoint in "
                f"{train_config.checkpoint_dir}. Use --resume '' for a fresh start or set "
                "resume_if_exists: true to fall back automatically."
            )

    # ── Data ──
    train_loader = create_dataloader(
        train_config.data_dir, seq_len,
        train_config.batch_size, "train", train_config.num_workers,
        seed=train_config.seed,
    )
    val_loader = create_dataloader(
        train_config.data_dir, seq_len,
        train_config.batch_size, "val", train_config.num_workers,
        shuffle=False,
    )
    train_dataset = train_loader.dataset
    train_dataset.set_epoch(epoch)
    train_dataset.skip_batches = batches_in_epoch   # continue exactly where we stopped
    train_iter = iter(train_loader)

    def next_batch():
        """Fetch the next micro-batch, rolling over to a new epoch when exhausted."""
        nonlocal train_iter, epoch, batches_in_epoch
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            batches_in_epoch = 0
            train_dataset.set_epoch(epoch)
            train_dataset.skip_batches = 0
            print(f"  -> Epoch {epoch} (reshuffling shards)")
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batches_in_epoch += 1
        return batch

    # ── Training loop ──
    timer = Timer()

    print(f"\n{'='*60}")
    print(f"Starting training from step {start_step}")
    print(f"  Steps: {start_step} -> {train_config.max_steps}")
    print(f"  Tokens/step: {tokens_per_full_step:,}")
    print(f"  Effective batch: {train_config.batch_size * train_config.gradient_accumulation_steps} sequences")
    print(f"{'='*60}\n")

    model.train()
    running_loss = 0.0
    running_aux = 0.0
    has_aux = False
    steps_in_window = 0
    last_grad_norm = 0.0
    last_logged_loss: float | None = None
    last_val_loss: float | None = None
    improved_since_save = False   # did validation improve since the last checkpoint?

    def checkpoint(step: int):
        nonlocal improved_since_save
        save_checkpoint(
            model, optimizer, step, last_logged_loss,
            model_config, train_config,
            train_config.checkpoint_dir,
            is_best=improved_since_save,
            best_val_loss=best_val_loss if best_val_loss != float("inf") else None,
            val_loss=last_val_loss,
            scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
            epoch=epoch,
            batches_in_epoch=batches_in_epoch,
            tokens_seen=tokens_seen,
            tokenizer_fingerprint=tok_fingerprint,
            extra={"moe": moe_meta} if moe_meta else None,
        )
        save_metrics_history(
            tracker.history, train_config.checkpoint_dir,
            gdrive_folder_id=run_gdrive_folder,
            gdrive_credentials_path=train_config.gdrive_credentials_path,
        )
        improved_since_save = False

    for step in range(start_step, train_config.max_steps):
        t0 = time.time()

        # Update learning rate following a cosine decay schedule with linear warmup.
        # Warmup: gradually ramp LR from 0 to peak_lr to stabilize early training
        #   when gradients are large and poorly directed.
        # Cosine decay: smoothly decrease LR toward min_lr, which empirically gives
        #   better final loss than step decay or linear decay.
        lr = get_lr(step, train_config.warmup_steps, train_config.max_steps,
                     train_config.peak_lr, train_config.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ──
        # Simulates a larger effective batch without requiring more GPU memory.
        # Instead of one big batch, process N micro-batches, accumulate gradients,
        # then do a single optimizer step. This is essential when the target effective
        # batch size doesn't fit in GPU memory (e.g., batch=32 × accum=4 = 128 effective).

        # Buffer all micro-batches for this step BEFORE running any forward passes.
        # This allows OOM retry: if any micro-step runs OOM we can replay the exact
        # same data with a shorter sequence cap rather than silently skipping batches.
        micro_batches = [next_batch() for _ in range(train_config.gradient_accumulation_steps)]

        # OOM-safe accumulation: retry the entire step with a shorter sequence cap
        # if CUDA runs out of memory.  Each retry shrinks the cap by 25% until it
        # either succeeds or hits train_config.min_seq_len.
        seq_cap = seq_len
        oom_retries = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = torch.zeros((), device=device)
            accum_aux = torch.zeros((), device=device)
            try:
                for ids, tgts in micro_batches:
                    ids  = ids[:, :seq_cap].to(device, non_blocking=True)
                    tgts = tgts[:, :seq_cap].to(device, non_blocking=True)
                    with amp_ctx:
                        out  = model(ids, targets=tgts)
                        # Divide loss by accumulation steps so the sum of micro-batch
                        # gradients equals the gradient of the full effective batch.
                        lm_loss = out["loss"] / train_config.gradient_accumulation_steps
                        # MoE models also return a router load-balance loss (src/moe.py).
                        aux = out.get("aux_loss")
                        loss = lm_loss
                        if aux is not None:
                            aux = aux / train_config.gradient_accumulation_steps
                            loss = lm_loss + train_config.aux_loss_coeff * aux
                    scaler.scale(loss).backward()
                    accum_loss += lm_loss.detach().float()
                    if aux is not None:
                        accum_aux += aux.detach().float()
                        has_aux = True
                break  # all micro-steps succeeded
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                new_cap = max(train_config.min_seq_len, seq_cap * 3 // 4)
                if new_cap == seq_cap:
                    raise RuntimeError(
                        f"OOM at step {step} even with minimum seq_len={seq_cap}. "
                        "Enable gradient_checkpointing or reduce batch_size."
                    ) from None
                oom_retries += 1
                seq_cap = new_cap
                print(f"  ! OOM at step {step} -- retry {oom_retries} with seq_cap={seq_cap}")

        # Gradient clipping: cap the global L2 norm of all gradients to prevent
        # exploding gradients, which can destabilize training (especially early on
        # or after a bad batch). max_grad_norm=1.0 is the standard for LLM training.
        if train_config.max_grad_norm > 0:
            scaler.unscale_(optimizer)  # Unscale gradients before clipping (required for fp16)
            last_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.max_grad_norm
            ).item()

        scaler.step(optimizer)
        scaler.update()

        tokens_this_step = (
            micro_batches[0][0].shape[0] * min(seq_cap, micro_batches[0][0].shape[1])
            * train_config.gradient_accumulation_steps
        )
        tokens_seen += tokens_this_step
        running_loss += accum_loss.item()   # single device sync per step
        if has_aux:
            running_aux += accum_aux.item()
        steps_in_window += 1
        timer.step()
        t1 = time.time()

        # ── Logging ──
        if step % train_config.log_every_steps == 0 and step > 0:
            avg_loss = running_loss / max(steps_in_window, 1)
            last_logged_loss = avg_loss
            tok_per_sec = tokens_this_step / max(t1 - t0, 1e-9)
            perplexity = math.exp(min(avg_loss, 20))  # Clamp to avoid overflow

            metrics = {
                "train/loss": avg_loss,
                "train/perplexity": perplexity,
                "train/lr": lr,
                "train/tokens_per_sec": tok_per_sec,
                "train/tokens_seen": tokens_seen,
                "train/epoch": epoch,
                "train/elapsed_hours": timer.elapsed() / 3600,
            }
            if train_config.max_grad_norm > 0:
                metrics["train/grad_norm"] = last_grad_norm
            if has_aux:
                metrics["train/aux_loss"] = running_aux / max(steps_in_window, 1)
            tracker.log(metrics, step)

            print(
                f"step {step:>6d} | loss {avg_loss:.4f} | ppl {perplexity:.1f} | "
                f"lr {lr:.2e} | tok/s {tok_per_sec:,.0f} | "
                f"{timer.elapsed() / 3600:.1f}h"
            )
            running_loss = 0.0
            running_aux = 0.0
            steps_in_window = 0

        # ── Validation ──
        if step % train_config.eval_every_steps == 0 and step > 0:
            val_metrics = compute_val_metrics(
                model, val_loader, device, amp_ctx, max_batches=train_config.eval_max_batches,
            )
            val_loss = val_metrics["loss"]
            last_val_loss = val_loss
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                improved_since_save = True

            tracker.log({
                "val/loss": val_loss,
                "val/perplexity": val_metrics["perplexity"],
                "val/best_loss": best_val_loss,
            }, step)
            print(f"  -> val loss {val_loss:.4f} | ppl {val_metrics['perplexity']:.1f} {'* best' if is_best else ''}")

            model.train()

        # ── Checkpoint ──
        # best.pt is refreshed whenever validation improved since the previous save.
        # (Align save_every_steps with eval_every_steps so best.pt holds the evaluated weights.)
        saved_this_step = False
        if step % train_config.save_every_steps == 0 and step > 0:
            checkpoint(step)
            saved_this_step = True
            print(f"  -> Saved checkpoint at step {step}")

        if stop_after_step is not None and step >= stop_after_step:
            if not saved_this_step:
                checkpoint(step)
            print(f"\nStopping after step {step} (stop_after_step); resume with --resume latest")
            tracker.finish()
            return tracker.history

        # ── Sample generation ──
        if step % train_config.generate_every_steps == 0 and step > 0:
            generate_samples(base_model, tokenizer, train_config.sample_prompts, device, tracker, step)
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
                generate_all_visualizations(base_model, tokenizer, sample_ids, tracker, step)
            model.train()

    # ── Final checkpoint ──
    # The loop runs steps [start_step, max_steps), so the last completed step is max_steps - 1.
    final_step = train_config.max_steps - 1
    if final_step >= start_step:
        checkpoint(final_step)
        print(f"\nTraining complete! Final checkpoint saved at step {final_step} "
              f"({train_config.max_steps} steps total)")
    else:
        print(f"\nNothing to do: checkpoint already at step {start_step - 1} >= max_steps - 1")
    tracker.finish()
    return tracker.history


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
        generated_text = tokenizer.decode(output_ids[0, len(input_ids):].tolist())
        table_data.append({"prompt": prompt, "generated": generated_text})
        print(f"  [Gen] {prompt}")
        print(f"         -> {generated_text[:200]}...")

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

_OVERRIDE_FIELDS = (
    "batch_size", "gradient_accumulation_steps", "max_steps", "peak_lr", "precision",
    "use_compile", "num_workers", "seed", "seq_len", "aux_loss_coeff", "wandb_project",
    "wandb_run_name", "data_dir", "checkpoint_dir", "tokenizer_path",
)


def add_train_override_args(parser) -> None:
    """Add ``--config``, ``--resume`` and the common ``TrainConfig`` overrides to ``parser``.

    Shared by ``python -m src.train`` and ``python -m src.finetune_moe``.
    """
    parser.add_argument("--config", type=str, default="",
                        help="Path to training config YAML (overrides defaults)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint: 'latest', 'best', 'step_N', or a name. "
                             "Pass '' to force a fresh start even if the config sets resume.")
    for name in ("batch_size", "gradient_accumulation_steps", "max_steps", "num_workers", "seed", "seq_len"):
        parser.add_argument(f"--{name}", type=int, default=None)
    for name in ("peak_lr", "aux_loss_coeff"):
        parser.add_argument(f"--{name}", type=float, default=None)
    parser.add_argument("--precision", type=str, default=None, help="auto | bf16 | fp16 | fp32")
    parser.add_argument("--use_compile", type=str, default=None, help="auto | true | false")
    for name in ("wandb_project", "wandb_run_name", "data_dir", "checkpoint_dir", "tokenizer_path"):
        parser.add_argument(f"--{name}", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_gdrive", action="store_true", help="Disable Google Drive backup")


def train_config_from_args(args, parser) -> TrainConfig:
    """Build a ``TrainConfig`` from ``--config`` (or defaults) and apply the CLI overrides."""
    if args.config:
        if not os.path.isfile(args.config):
            parser.error(f"--config file not found: {args.config}")
        train_config = load_train_config(args.config)
    else:
        train_config = TrainConfig()

    if args.resume is not None:
        train_config.resume = args.resume
    for name in _OVERRIDE_FIELDS:
        value = getattr(args, name)
        if value is not None:
            setattr(train_config, name, value)
    if args.no_wandb:
        train_config.use_wandb = False
    if args.no_gdrive:
        train_config.backup_to_gdrive = False
    return train_config


def resolve_run_gdrive_folder(train_config: TrainConfig, model_label: str) -> None:
    """For a fresh run with Drive backup configured, nest checkpoints under a per-model subfolder.

    Only applies when ``train_config.resume`` is empty: a resumed run must keep uploading to
    whatever folder its earlier checkpoints already live in, so it never auto-generates a new one
    (same contract as ``checkpoint_dir``/``resume`` themselves — the user carries it forward).
    Mutates ``train_config.gdrive_folder_id`` in place; best-effort, never raises.
    """
    if train_config.resume or not (train_config.backup_to_gdrive and train_config.gdrive_folder_id):
        return
    try:
        from .gdrive import default_run_folder_name, resolve_subfolder

        label = os.path.splitext(os.path.basename(model_label))[0] if os.path.isfile(model_label) else model_label
        run_folder = default_run_folder_name(label)
        resolved = resolve_subfolder(
            train_config.gdrive_folder_id, run_folder, train_config.gdrive_credentials_path,
        )
        if resolved:
            train_config.gdrive_folder_id = resolved
            print(f"Google Drive: using run folder '{run_folder}' under the configured root")
    except Exception as e:
        print(f"  ! Google Drive run-folder setup failed: {e}")


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Pre-train LLM from scratch")
    parser.add_argument("--model", type=str, default="tiny",
                        help="Model config name (tiny/small/medium/base/large) or path to YAML")
    add_train_override_args(parser)
    args = parser.parse_args()

    if os.path.isfile(args.model):
        model_config = load_model_config(args.model)
    else:
        model_config = get_model_config(args.model)

    train_config = train_config_from_args(args, parser)
    resolve_run_gdrive_folder(train_config, args.model)
    train(model_config, train_config)


if __name__ == "__main__":
    main()
