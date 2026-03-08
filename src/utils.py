"""
Utilities: checkpointing, logging, environment detection, learning rate scheduling.
"""

import os
import sys
import time
import json
import random
import glob
import torch
import numpy as np
from typing import Optional
from dataclasses import asdict

from .config import ModelConfig, TrainConfig


# ──────────────────────────────────────────────
# Environment detection
# ──────────────────────────────────────────────

def detect_environment() -> str:
    """Detect if running in Colab, vast.ai, or local."""
    if "google.colab" in sys.modules:
        return "colab"
    if os.path.exists("/workspace") or os.environ.get("VAST_CONTAINERLABEL"):
        return "vastai"
    return "local"


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(precision: str = "auto") -> torch.dtype:
    """Get the compute dtype based on precision setting and hardware."""
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    # Auto-detect
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def should_compile(setting: str = "auto") -> bool:
    """Determine if torch.compile should be used."""
    if setting == "true":
        return True
    if setting == "false":
        return False
    # Auto: use compile on CUDA with PyTorch 2.x
    return (
        torch.cuda.is_available()
        and hasattr(torch, "compile")
        and detect_environment() != "colab"  # Colab T4 has issues with compile
    )


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# Learning rate schedule
# ──────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float, min_lr: float) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


# ──────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """
    Save a full training checkpoint.
    Includes: model, optimizer, step, loss, configs, RNG states.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()

    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    torch.save(checkpoint, path)

    # Also save as latest (for easy resume)
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(checkpoint, best_path)

    # Cleanup old checkpoints (keep last N + best)
    cleanup_checkpoints(checkpoint_dir, train_config.keep_last_n_checkpoints)

    # ── Google Drive backup ──
    if train_config.backup_to_gdrive and train_config.gdrive_folder_id:
        try:
            from .gdrive import upload_to_gdrive, cleanup_remote_checkpoints

            upload_to_gdrive(path, train_config.gdrive_folder_id, train_config.gdrive_credentials_path)
            upload_to_gdrive(latest_path, train_config.gdrive_folder_id, train_config.gdrive_credentials_path)
            if is_best:
                upload_to_gdrive(best_path, train_config.gdrive_folder_id, train_config.gdrive_credentials_path)

            if train_config.gdrive_cleanup_remote:
                cleanup_remote_checkpoints(
                    train_config.gdrive_folder_id,
                    train_config.keep_last_n_checkpoints,
                    train_config.gdrive_credentials_path,
                )
            print(f"  → Backed up checkpoint to Google Drive")
        except Exception as e:
            print(f"  ⚠ Google Drive backup failed: {e}")

    return path


def cleanup_checkpoints(checkpoint_dir: str, keep_n: int):
    """Keep only the last N step checkpoints + best + latest."""
    pattern = os.path.join(checkpoint_dir, "step_*.pt")
    step_files = sorted(glob.glob(pattern), key=os.path.getmtime)

    # Remove old ones, keeping the last N
    for f in step_files[:-keep_n]:
        os.remove(f)


def load_checkpoint(
    checkpoint_dir: str,
    resume: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    gdrive_folder_id: str = "",
    gdrive_credentials_path: str = "",
) -> dict:
    """
    Load a checkpoint and restore model/optimizer state.

    If the checkpoint is not found locally but gdrive_folder_id is set,
    it is downloaded from Google Drive first.

    Args:
        checkpoint_dir: Directory containing checkpoints
        resume: "latest", "best", or "step_N"
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load tensors to
        gdrive_folder_id: Google Drive folder to fetch from (optional)
        gdrive_credentials_path: Service-account JSON path (optional)

    Returns:
        Checkpoint dict with step, loss, etc.
    """
    if resume == "latest":
        filename = "latest.pt"
    elif resume == "best":
        filename = "best.pt"
    else:
        filename = f"{resume}.pt"

    path = os.path.join(checkpoint_dir, filename)

    # Try downloading from Google Drive when the local file is missing
    if not os.path.exists(path) and gdrive_folder_id:
        try:
            from .gdrive import download_from_gdrive
            print(f"Checkpoint not found locally, downloading '{filename}' from Google Drive...")
            path = download_from_gdrive(
                filename, gdrive_folder_id, checkpoint_dir, gdrive_credentials_path,
            )
            print(f"  → Downloaded to {path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Checkpoint '{filename}' not found locally or on Google Drive: {e}"
            )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore RNG states
    rng = checkpoint.get("rng_state", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.random.set_rng_state(rng["torch"].cpu().to(torch.uint8))
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])

    print(f"Resumed from step {checkpoint['step']} (loss={checkpoint['loss']:.4f})")
    return checkpoint


def has_checkpoint(checkpoint_dir: str, resume: str, gdrive_folder_id: str = "", gdrive_credentials_path: str = "") -> bool:
    """Check if a resumable checkpoint exists locally or on Google Drive."""
    if not resume:
        return False

    if resume == "latest":
        filename = "latest.pt"
    elif resume == "best":
        filename = "best.pt"
    else:
        filename = f"{resume}.pt"

    if os.path.exists(os.path.join(checkpoint_dir, filename)):
        return True

    # Check Google Drive
    if gdrive_folder_id:
        try:
            from .gdrive import _get_service, _find_file
            service = _get_service(gdrive_credentials_path)
            return _find_file(service, filename, gdrive_folder_id) is not None
        except Exception:
            return False

    return False


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, use_wandb: bool = True, project: str = "llm-proto", run_name: str = ""):
        self.use_wandb = use_wandb
        self.history = []

        if use_wandb:
            try:
                import wandb
                wandb.init(project=project, name=run_name or None)
                self.wandb = wandb
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def log(self, metrics: dict, step: int):
        """Log metrics to wandb and internal history."""
        metrics["step"] = step
        self.history.append(metrics)
        if self.use_wandb and self.wandb:
            self.wandb.log(metrics, step=step)

    def log_image(self, key: str, image, step: int):
        """Log an image (matplotlib figure or PIL image) to wandb."""
        if self.use_wandb and self.wandb:
            self.wandb.log({key: self.wandb.Image(image)}, step=step)

    def log_config(self, config: dict):
        """Log config to wandb."""
        if self.use_wandb and self.wandb:
            self.wandb.config.update(config)

    def finish(self):
        """Finish the logging run."""
        if self.use_wandb and self.wandb:
            self.wandb.finish()


class Timer:
    """Simple timer for measuring throughput."""

    def __init__(self):
        self.start_time = time.time()
        self.step_times = []

    def step(self):
        now = time.time()
        self.step_times.append(now)

    def tokens_per_sec(self, tokens_since_last: int) -> float:
        if len(self.step_times) < 2:
            return 0.0
        dt = self.step_times[-1] - self.step_times[-2]
        return tokens_since_last / dt if dt > 0 else 0.0

    def elapsed(self) -> float:
        return time.time() - self.start_time
