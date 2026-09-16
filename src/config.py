"""
Configuration dataclasses for model architecture and training.
Change model size by swapping configs — zero code changes needed.
"""

import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """All model architecture parameters. Change these to scale from 30M to 1B+."""

    # Core dimensions
    vocab_size: int = 32_000       # 32K is the sweet spot: covers English + code well with BPE
    dim: int = 512                 # Hidden dimension (d_model) — the "width" of the model
    n_layers: int = 6              # Number of transformer blocks — the "depth" of the model
    n_heads: int = 8               # Number of attention heads (each sees dim/n_heads dimensions)
    n_kv_heads: int = 4            # GQA: fewer KV heads → less KV-cache memory at inference time
    max_seq_len: int = 2048        # Context window — max tokens the model can attend to at once

    # Feed-forward network
    # SwiGLU FFN hidden dim. If None, computed as 2/3 × 4 × dim, rounded to multiple of 256
    # (see __post_init__ for the formula and rationale)
    ffn_dim: int | None = None

    # Regularization
    dropout: float = 0.0           # 0 for pre-training (data is diverse enough); >0 for fine-tuning

    # Normalization
    norm_eps: float = 1e-5         # RMSNorm epsilon — prevents division by zero in normalization

    # Positional encoding
    rope_theta: float = 10_000.0   # RoPE base frequency — controls the wavelength spectrum of rotations

    # Weight tying: share embedding and output projection weights
    # Reduces param count, and provides a useful learning signal (see model.py for details)
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.head_dim % 2 != 0:
            # RoPE rotates pairs of dimensions, so head_dim must be even.
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")
        if self.vocab_size >= 65_535:
            # Token shards are stored as uint16 (see data.py); 65535 is reserved.
            raise ValueError(f"vocab_size ({self.vocab_size}) must be < 65535 (uint16 shard format)")

        if self.ffn_dim is None:
            # SwiGLU uses 3 matrices (gate, up, down) instead of 2 in standard FFN.
            # To keep total FFN params ≈ same as standard 4×dim FFN (2 matrices of dim×4dim),
            # we use: hidden = 2/3 × 4 × dim ≈ 2.67× dim.
            # Rounding to multiple of 256 aligns with GPU tensor core tile sizes (8, 16, 32...),
            # ensuring matrix dimensions are evenly divisible for maximum CUDA kernel efficiency.
            raw = int(2 / 3 * 4 * self.dim)
            self.ffn_dim = ((raw + 255) // 256) * 256

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def param_count_estimate(self) -> int:
        """Parameter count from the architecture (matches TransformerLM.count_parameters)."""
        embed = self.vocab_size * self.dim
        # Each transformer block: attn(Q,K,V,O) + FFN(gate,up,down) + 2 norms
        attn = self.dim * (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        attn += self.n_heads * self.head_dim * self.dim  # output projection
        ffn = 3 * self.dim * self.ffn_dim  # gate, up, down projections
        norm = 2 * self.dim
        block = attn + ffn + norm
        total = embed + self.n_layers * block + self.dim  # + final norm
        if not self.tie_embeddings:
            total += self.vocab_size * self.dim  # output projection
        return total


@dataclass
class TrainConfig:
    """All training hyperparameters."""

    # --- Data ---
    # Data sources are defined in configs/data.yaml (see python -m src.data).
    data_dir: str = "data"
    tokenizer_path: str = "tokenizer_data"

    # --- Optimization ---
    batch_size: int = 32       # Per-device micro batch size (sequences per forward pass)
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size × grad_accum × seq_len tokens
    max_steps: int = 50_000    # Total optimization steps (not epochs — step-based is standard for LLMs)
    warmup_steps: int = 2_000  # Linear warmup: prevents early instability from large initial gradients

    # Learning rate — peak_lr=6e-4 is the Chinchilla-optimal range for models under 1B params.
    # Higher LR for smaller models, lower for larger (scales roughly as 1/sqrt(model_size)).
    peak_lr: float = 6e-4
    min_lr: float = 6e-5       # Cosine decay target — 10× below peak is a common ratio
    weight_decay: float = 0.1  # AdamW weight decay — 0.1 is the GPT-3/LLaMA standard

    # Adam hyperparameters
    adam_beta1: float = 0.9    # Momentum — standard value, no reason to change
    adam_beta2: float = 0.95   # Lower than default 0.999 — better for LLM training stability
                               # (see train.py comments for full rationale)
    adam_eps: float = 1e-8     # Numerical stability in Adam denominator

    # Gradient clipping — 1.0 is the universal standard for LLM pre-training
    max_grad_norm: float = 1.0

    # --- Mixed precision ---
    precision: str = "auto"    # "auto" picks bf16 if supported (Ampere+), fp16 otherwise, fp32 on CPU

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 500        # Frequent saves protect against crashes mid-training
    keep_last_n_checkpoints: int = 5   # Disk budget: keep last 5 + best validation checkpoint

    # --- Google Drive backup ---
    # On by default, but a no-op until gdrive_folder_id names an actual folder (see save_checkpoint).
    backup_to_gdrive: bool = True              # Upload checkpoints to Google Drive
    gdrive_folder_id: str = ""                 # Google Drive folder ID (API) or folder name (Colab)
    gdrive_credentials_path: str = ""          # Path to service account JSON or OAuth credentials
    gdrive_cleanup_remote: bool = True         # Remove old remote checkpoints (mirrors keep_last_n)

    # --- Logging ---
    log_every_steps: int = 10           # Log metrics every N steps
    eval_every_steps: int = 500         # Run validation every N steps
    eval_max_batches: int = 50          # Validation batches per in-training eval
    generate_every_steps: int = 1000    # Generate sample text every N steps
    visualize_every_steps: int = 1000   # Log model internals every N steps

    # --- Wandb ---
    wandb_project: str = "llm-proto"
    wandb_run_name: str = ""            # Auto-generated if empty
    use_wandb: bool = True

    # --- Performance ---
    num_workers: int = 4               # DataLoader workers for async data prefetching
    use_compile: str = "auto"          # torch.compile: fuses operations for 10-30% speedup on CUDA
    gradient_checkpointing: bool = False  # Trade compute for memory: recompute activations in backward
                                          # instead of storing them. Essential for 1B+ models.
    min_seq_len: int = 256             # OOM safety floor: never truncate sequences below this length.
                                       # Shorter sequences lose enough context to hurt gradient quality.
    seq_len: int | None = None         # Training sequence length; None = model max_seq_len.
                                       # Fine-tuning can use less than the context the model supports.

    # --- Mixture-of-Experts fine-tuning (src/moe.py, src/finetune_moe.py) ---
    aux_loss_coeff: float = 0.01       # Weight of the router load-balance loss; ignored for dense models

    # --- Reproducibility ---
    seed: int = 42  # Fixed seed for deterministic initialization and data shuffling

    # --- Resume ---
    resume: str = ""  # "" = fresh start, "latest" = latest checkpoint, "step_5000" = specific
    resume_if_exists: bool = False  # If True, a missing `resume` checkpoint falls back to a fresh start

    # --- Generation sampling ---
    sample_prompts: list = field(default_factory=lambda: [
        "The meaning of life is",
        "In the year 2050, artificial intelligence",
        "Once upon a time in a land far away",
        "The theory of relativity explains",
    ])


@dataclass
class MoEConfig:
    """Shape of a post-hoc Mixture-of-Experts graft (see ``src/moe.py``).

    Kept separate from ``ModelConfig`` on purpose: ``TransformerLM(config)`` always builds
    the dense skeleton (so a base checkpoint's weights load first), and the MoE layers are
    grafted on afterwards. Stored in checkpoints under ``ckpt["moe"]``.
    """

    expert_layers: list          # block indices whose FFN becomes a SparseMoE, e.g. list(range(24, 32))
    n_experts: int = 1           # experts per MoE layer
    top_k: int = 1               # experts activated per token (<= n_experts)
    warm_start: bool = True      # initialise every expert from the dense FFN it replaces

    def __post_init__(self):
        if not isinstance(self.expert_layers, (list, tuple)) or not self.expert_layers:
            raise ValueError("expert_layers must be a non-empty list of layer indices")
        layers = [int(i) for i in self.expert_layers]
        if any(i < 0 for i in layers):
            raise ValueError(f"expert_layers must be non-negative, got {layers}")
        if len(set(layers)) != len(layers):
            raise ValueError(f"expert_layers contains duplicates: {layers}")
        self.expert_layers = sorted(layers)
        if self.n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {self.n_experts}")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError(f"top_k ({self.top_k}) must be between 1 and n_experts ({self.n_experts})")


# ──────────────────────────────────────────────
# Preset model configs
# ──────────────────────────────────────────────

MODEL_CONFIGS = {
    "tiny": ModelConfig(
        dim=512, n_layers=6, n_heads=8, n_kv_heads=4,
        max_seq_len=2048,
    ),
    "small": ModelConfig(
        dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
        max_seq_len=2048,
    ),
    "medium": ModelConfig(
        dim=1024, n_layers=24, n_heads=16, n_kv_heads=4,
        max_seq_len=2048,
    ),
    "base": ModelConfig(
        dim=1280, n_layers=24, n_heads=20, n_kv_heads=4,
        max_seq_len=2048,
    ),
    "large": ModelConfig(
        dim=2048, n_layers=32, n_heads=32, n_kv_heads=8,
        max_seq_len=4096,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Get a *copy* of a preset model config by name (safe to mutate)."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model config: {name}. Available: {list(MODEL_CONFIGS.keys())}")
    return replace(MODEL_CONFIGS[name])


# ──────────────────────────────────────────────
# YAML loading with validation
# ──────────────────────────────────────────────

_SCALAR_TYPES = (int, float, bool, str)


def _coerce(value: Any, target: type, key: str, path: str) -> Any:
    """Coerce a YAML scalar to the dataclass field type.

    YAML 1.1 (PyYAML) parses `3e-4` as the *string* "3e-4" because it lacks a
    decimal point, so numeric fields are explicitly converted here.
    """
    if value is None or target not in _SCALAR_TYPES:
        return value
    if target is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no", "on", "off"):
            return value.strip().lower() in ("true", "yes", "on")
        raise ValueError(f"{path}: field '{key}' expects a bool, got {value!r}")
    if target is float and isinstance(value, (int, float, str)) and not isinstance(value, bool):
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"{path}: field '{key}' expects a float, got {value!r}") from None
    if target is int and not isinstance(value, bool):
        if isinstance(value, int):
            return value
        if isinstance(value, (float, str)):
            try:
                as_float = float(value)
            except ValueError:
                raise ValueError(f"{path}: field '{key}' expects an int, got {value!r}") from None
            if as_float != int(as_float):
                raise ValueError(f"{path}: field '{key}' expects an int, got {value!r}")
            return int(as_float)
    if target is str and not isinstance(value, str):
        return str(value)
    return value


def config_from_dict(cls, data: dict[str, Any] | None, path: str = "<dict>"):
    """Build a config dataclass from a dict, rejecting unknown keys and coercing scalars."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(
            f"{path}: unknown {cls.__name__} field(s): {', '.join(unknown)}. "
            f"Known fields: {', '.join(sorted(known))}"
        )
    coerced = {k: _coerce(v, known[k].type, k, path) for k, v in data.items()}
    return cls(**coerced)


def save_config(config, path: str):
    """Save a config dataclass to YAML."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)


def load_model_config(path: str) -> ModelConfig:
    """Load a ModelConfig from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return config_from_dict(ModelConfig, data, path)


def load_train_config(path: str) -> TrainConfig:
    """Load a TrainConfig from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return config_from_dict(TrainConfig, data, path)
