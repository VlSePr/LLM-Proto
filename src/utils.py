"""
Utilities: checkpointing, logging, environment detection, learning rate scheduling.
"""

import glob
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .config import ModelConfig, TrainConfig, config_from_dict


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file's contents, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

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


def default_num_workers(max_workers: int = 4) -> int:
    """DataLoader worker count that is safe for the current process.

    Returns 0 on Windows and inside Jupyter/IPython kernels: both spawn worker
    processes that re-import ``__main__``, which has no ``if __name__`` guard in a
    notebook. Elsewhere returns ``max_workers``.
    """
    if sys.platform.startswith("win") or "ipykernel" in sys.modules:
        return 0
    return max_workers


def get_dtype(precision: str = "auto") -> torch.dtype:
    """Get the compute dtype based on precision setting and hardware."""
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    # Auto-detect: prefer bf16 (same range as fp32, no GradScaler needed) on Ampere+ GPUs,
    # fall back to fp16 (needs GradScaler) on older GPUs, fp32 on CPU.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def should_compile(setting: str = "auto") -> bool:
    """Determine if torch.compile should be used."""
    setting = str(setting).lower()
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


def make_grad_scaler(enabled: bool):
    """GradScaler that works across torch versions (torch.amp.GradScaler is 2.3+)."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    from torch.cuda.amp import GradScaler  # pragma: no cover (older torch)
    return GradScaler(enabled=enabled)


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.
    All four RNG sources must be seeded to ensure identical results across runs:
    Python's random (for data shuffling), NumPy (for data loading), PyTorch CPU, and CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# Learning rate schedule
# ──────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float, min_lr: float) -> float:
    """
    Cosine decay with linear warmup — the standard LR schedule for LLM pre-training.

    Phase 1 (warmup): LR ramps linearly from 0 to peak_lr over warmup_steps.
      This prevents large, poorly-directed gradient updates before the optimizer's
      moment estimates (Adam's m and v) have warmed up.

    Phase 2 (cosine decay): LR follows a half-cosine curve from peak_lr down to min_lr.
      Cosine decay is smoother than step decay and empirically gives ~0.5-1% better
      final loss than linear decay (Loshchilov & Hutter, 2016).

    Formula: lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
    """
    if step < warmup_steps:
        # Linear warmup: LR increases proportionally with step
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay: progress goes from 0.0 to 1.0 over the decay phase
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))  # Ranges from 1.0 down to 0.0
    return float(min_lr + (peak_lr - min_lr) * cosine)


# ──────────────────────────────────────────────
# Model wrappers (torch.compile)
# ──────────────────────────────────────────────

_COMPILE_PREFIX = "_orig_mod."


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module of a ``torch.compile``d model (or the model itself)."""
    return getattr(model, "_orig_mod", model)


def strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove the ``_orig_mod.`` prefix that ``torch.compile`` adds to state-dict keys."""
    return {
        (k[len(_COMPILE_PREFIX):] if k.startswith(_COMPILE_PREFIX) else k): v
        for k, v in state_dict.items()
    }


# ──────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────

def resolve_checkpoint_filename(resume: str) -> str:
    """Map a resume spec ('latest', 'best', 'step_N', or a custom name) to a filename."""
    if resume == "latest":
        return "latest.pt"
    if resume == "best":
        return "best.pt"
    if resume.endswith(".pt"):
        return resume
    return f"{resume}.pt"


def tokenizer_fingerprint(tokenizer_path: str) -> str | None:
    """Best-effort SHA-256 of ``<tokenizer_path>/tokenizer.json``, or ``None`` if unreadable."""
    try:
        return sha256_file(os.path.join(tokenizer_path, "tokenizer.json"))
    except OSError:
        return None


def warn_if_tokenizer_mismatch(ckpt: dict[str, Any], tokenizer_path: str) -> None:
    """Print a warning when the checkpoint's tokenizer no longer matches the one on disk.

    Never raises: a missing fingerprint on either side (legacy checkpoints, or a tokenizer
    that can't be read) silently skips the check instead of blocking resume/inference.
    """
    saved = ckpt.get("tokenizer_fingerprint")
    if not saved:
        return
    current = tokenizer_fingerprint(tokenizer_path)
    if current and current != saved:
        print(
            f"  ! Warning: tokenizer at '{tokenizer_path}' does not match the tokenizer this "
            "checkpoint was trained with. Embeddings may no longer align with token ids."
        )


def _rng_state_for_save() -> dict[str, Any]:
    """Collect RNG states in a form that ``torch.load(weights_only=True)`` accepts."""
    np_state = np.random.get_state()  # ('MT19937', ndarray[624] uint32, pos, has_gauss, cached_gaussian)
    state = {
        "python": random.getstate(),
        "numpy": (np_state[0], torch.from_numpy(np_state[1].copy()), int(np_state[2]),
                  int(np_state[3]), float(np_state[4])),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        # Save RNG state for ALL GPUs (even if single-GPU) to support future multi-GPU resume
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _to_tuple(obj):
    """Recursively convert lists back to tuples (random.setstate needs tuples)."""
    if isinstance(obj, (list, tuple)):
        return tuple(_to_tuple(o) for o in obj)
    return obj


def _restore_rng_state(rng: dict[str, Any]) -> None:
    if "python" in rng:
        random.setstate(_to_tuple(rng["python"]))
    if "numpy" in rng:
        np_state = rng["numpy"]
        keys = np_state[1]
        if isinstance(keys, torch.Tensor):
            keys = keys.cpu().numpy()
        np.random.set_state((np_state[0], np.asarray(keys, dtype=np.uint32), int(np_state[2]),
                             int(np_state[3]), float(np_state[4])))
    if "torch" in rng:
        # Cast to uint8 in case the state was saved on a different device/dtype
        torch.random.set_rng_state(rng["torch"].cpu().to(torch.uint8))
    if "cuda" in rng and torch.cuda.is_available():
        saved = [s.cpu().to(torch.uint8) for s in rng["cuda"]]
        n = min(len(saved), torch.cuda.device_count())
        for i in range(n):
            torch.cuda.set_rng_state(saved[i], device=i)


def load_checkpoint_file(path: str, device: torch.device | str = "cpu") -> dict:
    """``torch.load`` a checkpoint, preferring the safe ``weights_only=True`` path."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as e:  # older checkpoints contain numpy objects
        print(f"  ! weights_only load failed ({type(e).__name__}); falling back to full unpickling "
              f"for {path}. Only load checkpoints you trust.")
        return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    loss: float | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_dir: str,
    is_best: bool = False,
    *,
    best_val_loss: float | None = None,
    val_loss: float | None = None,
    scaler_state: dict | None = None,
    epoch: int = 0,
    batches_in_epoch: int = 0,
    tokens_seen: int | None = None,
    tokenizer_fingerprint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    Save a full training checkpoint.
    Includes: model, optimizer, GradScaler, step/epoch/data position, best val loss,
    configs, and RNG states — everything needed for an exact resume.

    The model state dict is always saved *without* the ``torch.compile`` prefix so the
    file loads into a plain ``TransformerLM`` (inference CLIs, Colab without compile).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "batches_in_epoch": batches_in_epoch,
        "tokens_seen": tokens_seen,
        "loss": float(loss) if loss is not None else None,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "model_state_dict": strip_compile_prefix(unwrap_model(model).state_dict()),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler_state,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        # RNG states let a resumed run continue the *exact* same random sequence
        # (dropout masks, sampling), making results reproducible across interruptions.
        "rng_state": _rng_state_for_save(),
        # Ties this checkpoint to the exact tokenizer.json it was trained with, so a later
        # retrain/swap of the tokenizer can be detected instead of silently misaligning ids.
        "tokenizer_fingerprint": tokenizer_fingerprint,
    }
    if extra:
        checkpoint.update(extra)

    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)  # atomic: never leave a half-written step_N.pt behind

    # "latest.pt" is a convenience alias: always points to the most recent checkpoint.
    # This way, resume="latest" always works without knowing the exact step number.
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    shutil.copyfile(path, latest_path)

    best_path = os.path.join(checkpoint_dir, "best.pt")
    if is_best:
        shutil.copyfile(path, best_path)

    # Cleanup old checkpoints (keep last N + best + latest)
    cleanup_checkpoints(checkpoint_dir, train_config.keep_last_n_checkpoints)

    # ── Google Drive backup ──
    if train_config.backup_to_gdrive and train_config.gdrive_folder_id:
        try:
            from .gdrive import cleanup_remote_checkpoints, upload_to_gdrive

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
            print("  -> Backed up checkpoint to Google Drive")
        except Exception as e:
            print(f"  ! Google Drive backup failed: {e}")

    return path


def upload_run_artifact(path: str, gdrive_folder_id: str, gdrive_credentials_path: str = "") -> None:
    """Best-effort upload of a single run artifact (metrics history, config snapshot, ...) to Drive."""
    if not gdrive_folder_id:
        return
    try:
        from .gdrive import upload_to_gdrive
        upload_to_gdrive(path, gdrive_folder_id, gdrive_credentials_path)
    except Exception as e:
        print(f"  ! Google Drive upload of {os.path.basename(path)} failed: {e}")


def save_metrics_history(
    history: list[dict[str, Any]],
    checkpoint_dir: str,
    *,
    gdrive_folder_id: str = "",
    gdrive_credentials_path: str = "",
) -> str:
    """Write ``MetricsTracker.history`` to ``<checkpoint_dir>/metrics_history.json`` (atomic write).

    This is the only durable copy of the loss curve when ``use_wandb`` is off. Best-effort
    uploaded to Drive alongside checkpoints when a folder id is given.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "metrics_history.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(history, f)
    os.replace(tmp_path, path)
    upload_run_artifact(path, gdrive_folder_id, gdrive_credentials_path)
    return path


def cleanup_checkpoints(checkpoint_dir: str, keep_n: int):
    """Keep only the last N step checkpoints (+ best + latest). keep_n <= 0 keeps everything."""
    if keep_n is None or keep_n <= 0:
        return
    pattern = os.path.join(checkpoint_dir, "step_*.pt")
    step_files = sorted(glob.glob(pattern), key=os.path.getmtime)

    # Remove old ones, keeping the last N
    for f in step_files[:-keep_n]:
        os.remove(f)


def resolve_checkpoint_path(
    checkpoint_dir: str,
    resume: str,
    gdrive_folder_id: str = "",
    gdrive_credentials_path: str = "",
) -> str:
    """Local path of the checkpoint named by ``resume``, downloading it from Drive if needed.

    ``resume`` is ``"latest"``, ``"best"``, ``"step_N"``, a custom name, or a ``.pt``
    filename (see ``resolve_checkpoint_filename``). Raises ``FileNotFoundError`` when
    the file is neither local nor (with ``gdrive_folder_id``) on Google Drive.
    """
    filename = resolve_checkpoint_filename(resume)
    path = os.path.join(checkpoint_dir, filename)

    # Try downloading from Google Drive when the local file is missing
    if not os.path.exists(path) and gdrive_folder_id:
        try:
            from .gdrive import download_from_gdrive
            print(f"Checkpoint not found locally, downloading '{filename}' from Google Drive...")
            path = download_from_gdrive(
                filename, gdrive_folder_id, checkpoint_dir, gdrive_credentials_path,
            )
            print(f"  -> Downloaded to {path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Checkpoint '{filename}' not found locally or on Google Drive: {e}"
            ) from e

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(
    checkpoint_dir: str,
    resume: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    gdrive_folder_id: str = "",
    gdrive_credentials_path: str = "",
    scaler=None,
    restore_rng: bool = True,
) -> dict:
    """
    Load a checkpoint and restore model/optimizer/scaler state.

    If the checkpoint is not found locally but gdrive_folder_id is set,
    it is downloaded from Google Drive first. The model must already have the
    right shape (for MoE checkpoints, graft first; see ``build_model_from_checkpoint``).

    Args:
        checkpoint_dir: Directory containing checkpoints
        resume: "latest", "best", "step_N", or a custom name
        model: Model to load weights into (plain or torch.compile'd)
        optimizer: Optimizer to load state into (optional)
        device: Device to load tensors to
        gdrive_folder_id: Google Drive folder to fetch from (optional)
        gdrive_credentials_path: Service-account JSON path (optional)
        scaler: GradScaler to restore (optional)
        restore_rng: Restore RNG states (True for training resume; False for inference)

    Returns:
        Checkpoint dict with step, loss, etc.
    """
    path = resolve_checkpoint_path(checkpoint_dir, resume, gdrive_folder_id, gdrive_credentials_path)

    print(f"Loading checkpoint from {path}...")
    checkpoint = load_checkpoint_file(path, device)

    unwrap_model(model).load_state_dict(strip_compile_prefix(checkpoint["model_state_dict"]))
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if restore_rng:
        _restore_rng_state(checkpoint.get("rng_state", {}))

    loss = checkpoint.get("loss")
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "n/a"
    print(f"Resumed from step {checkpoint['step']} (loss={loss_str})")
    return checkpoint


def _legacy_moe_metadata(ckpt: dict) -> dict | None:
    """MoE shape from the pre-``ckpt["moe"]`` notebook schema (top-level keys), or ``None``."""
    if "expert_layers" in ckpt and "n_experts" in ckpt:
        return {
            "expert_layers": list(ckpt["expert_layers"]),
            "n_experts": int(ckpt["n_experts"]),
            "top_k": int(ckpt.get("top_k", 1)),
        }
    return None


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
    model_config_name: str | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Build a ``TransformerLM`` from a checkpoint file for inference/evaluation.

    The architecture is taken from ``ckpt["model_config"]`` (always present for
    checkpoints written by ``save_checkpoint``). ``model_config_name`` (a preset
    name or YAML path) is only used as a fallback for files without one.
    Mixture-of-Experts checkpoints carry ``ckpt["moe"]`` (see ``moe.moe_metadata``);
    the same layers are grafted onto the dense skeleton before the weights load,
    so dense and MoE checkpoints go through this one path.
    Returns ``(model, checkpoint_dict)``; the checkpoint is loaded exactly once
    and RNG states are *not* restored.
    """
    from .config import MoEConfig, get_model_config, load_model_config
    from .model import TransformerLM

    if device is None:
        device = get_device()
    ckpt = load_checkpoint_file(checkpoint_path, device)

    if ckpt.get("model_config"):
        model_config = config_from_dict(ModelConfig, dict(ckpt["model_config"]), checkpoint_path)
    elif model_config_name:
        if os.path.isfile(model_config_name):
            model_config = load_model_config(model_config_name)
        else:
            model_config = get_model_config(model_config_name)
    else:
        raise ValueError(
            f"{checkpoint_path} has no 'model_config' entry; pass --model <preset|yaml> explicitly."
        )

    model = TransformerLM(model_config).to(device)
    moe = ckpt.get("moe") or _legacy_moe_metadata(ckpt)
    if moe:
        from .moe import graft_moe
        graft_moe(model, config_from_dict(MoEConfig, dict(moe), checkpoint_path), warm_start=False)
    model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
    model.eval()
    return model, ckpt


def has_checkpoint(
    checkpoint_dir: str, resume: str, gdrive_folder_id: str = "", gdrive_credentials_path: str = "",
) -> bool:
    """Check if a resumable checkpoint exists locally or on Google Drive."""
    if not resume:
        return False

    filename = resolve_checkpoint_filename(resume)
    if os.path.exists(os.path.join(checkpoint_dir, filename)):
        return True

    # Check Google Drive
    if gdrive_folder_id:
        try:
            from .gdrive import _find_file, _get_service
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
        metrics = dict(metrics)
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
        if len(self.step_times) > 1000:
            del self.step_times[:-1000]

    def tokens_per_sec(self, tokens_since_last: int) -> float:
        if len(self.step_times) < 2:
            return 0.0
        dt = self.step_times[-1] - self.step_times[-2]
        return tokens_since_last / dt if dt > 0 else 0.0

    def elapsed(self) -> float:
        return time.time() - self.start_time
