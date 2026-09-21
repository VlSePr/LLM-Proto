"""
Human-readable summaries for the training notebook: what a run is about to do, what data it will
see, and how it went. Each function returns a string (so it is testable); the notebook prints it.
"""

import os
from typing import Any

from .config import ModelConfig, TrainConfig
from .data import read_manifest
from .progress import human_size


def dataset_summary(data_dir: str) -> str:
    """Shard count, token counts and size of the tokenized dataset in *data_dir* (from its manifest)."""
    manifest = read_manifest(data_dir)
    if manifest is None:
        return f"Dataset: no manifest.json in {data_dir}"
    files = manifest.get("files", [])
    train_shards = [f for f in files if f["name"].startswith("train_")]
    total_bytes = sum(f.get("bytes", 0) for f in files)
    train_tokens = manifest.get("train_tokens", sum(f.get("tokens", 0) for f in train_shards))
    val_tokens = manifest.get("val_tokens", sum(f.get("tokens", 0) for f in files if f["name"] == "val.bin"))
    vocab = (manifest.get("tokenizer") or {}).get("vocab_size")
    parts = [
        f"{len(train_shards)} train shard(s)",
        f"{train_tokens:,} train tokens",
        f"{val_tokens:,} val tokens",
        human_size(total_bytes),
    ]
    if vocab:
        parts.append(f"vocab {vocab:,}")
    lines = [f"Dataset ({data_dir}): " + " | ".join(parts)]
    if manifest.get("created"):
        lines.append(f"  built {manifest['created']} | fingerprint {str(manifest.get('fingerprint', ''))[:12]}")
    return "\n".join(lines)


def run_summary(
    model_config: ModelConfig,
    train_config: TrainConfig,
    start_step: int,
    n_steps: int,
    *,
    model_name: str = "",
    drive_folder: str = "",
    data_dir: str = "",
) -> str:
    """What the coming ``train()`` call will do: batch/token budget, schedule, save cadence, folders."""
    seq_len = train_config.seq_len or model_config.max_seq_len
    seqs_per_step = train_config.batch_size * train_config.gradient_accumulation_steps
    tokens_per_step = seqs_per_step * seq_len
    session_tokens = tokens_per_step * n_steps
    resume = f"resuming from '{train_config.resume}'" if start_step else "fresh run"

    lines = [
        f"Model      : {model_name or 'custom'} | ~{model_config.param_count_estimate() / 1e6:.0f}M params | "
        f"dim {model_config.dim}, {model_config.n_layers} layers, "
        f"{model_config.n_heads}/{model_config.n_kv_heads} heads",
        f"Batch      : {train_config.batch_size} x {train_config.gradient_accumulation_steps} accum = "
        f"{seqs_per_step} sequences x {seq_len} tokens = {tokens_per_step:,} tokens/step",
        f"Steps      : {start_step:,} -> {train_config.max_steps:,} ({n_steps:,} more; {resume})",
        f"This session: {session_tokens:,} tokens",
        f"Schedule   : warmup {train_config.warmup_steps:,} steps, LR {train_config.peak_lr:g} -> "
        f"{train_config.min_lr:g} (cosine)",
        f"Cadence    : log/{train_config.log_every_steps}  eval/{train_config.eval_every_steps}  "
        f"generate/{train_config.generate_every_steps}  save/{train_config.save_every_steps} steps",
        f"Checkpoints: {train_config.checkpoint_dir}/ (keep last {train_config.keep_last_n_checkpoints})",
        f"Drive      : {drive_folder if train_config.backup_to_gdrive and drive_folder else 'disabled'}",
    ]
    manifest = read_manifest(data_dir) if data_dir else None
    if manifest and manifest.get("train_tokens"):
        epochs = session_tokens / manifest["train_tokens"]
        lines.append(f"Data       : {manifest['train_tokens']:,} train tokens "
                     f"-> this session covers ~{epochs:.2f} epoch(s)")
    return "\n".join(lines)


def training_summary(history: list[dict[str, Any]], wall_seconds: float | None = None) -> str:
    """Headline numbers from ``train()``'s metric history: final/best val loss, perplexity, tokens, time."""
    if not history:
        return "Training summary: no metrics were logged."

    def last(key: str):
        for row in reversed(history):
            if key in row:
                return row[key]
        return None

    val_losses = [(row["val/loss"], row["step"]) for row in history if "val/loss" in row]
    lines = [f"Last logged step: {history[-1].get('step', '?')}"]
    if last("train/loss") is not None:
        train_ppl = last("train/perplexity") or float("nan")
        lines.append(f"Train loss : {last('train/loss'):.4f} (perplexity {train_ppl:.1f})")
    if val_losses:
        best_loss, best_step = min(val_losses)
        lines.append(f"Val loss   : {val_losses[-1][0]:.4f} (perplexity {last('val/perplexity') or float('nan'):.1f})"
                     f" | best {best_loss:.4f} at step {best_step}")
    else:
        lines.append("Val loss   : none logged (run shorter than eval_every_steps?)")
    if last("train/tokens_seen") is not None:
        lines.append(f"Tokens seen: {int(last('train/tokens_seen')):,}")
    if wall_seconds is not None:
        lines.append(f"Wall time  : {wall_seconds / 60:.1f} min")
    return "\n".join(lines)


def list_outputs_summary(output_dir: str) -> str:
    """One line per file in *output_dir* with its size (used by the notebook's wrap-up cell)."""
    from .outputs import list_outputs
    files = list_outputs(output_dir)
    if not files:
        return f"No files in {output_dir} yet."
    width = max(len(name) for name, _ in files)
    rows = [f"  {name:{width}s}  {human_size(size)}" for name, size in files]
    return "\n".join([f"{os.path.abspath(output_dir)}:", *rows])
