"""
Configuration dataclasses for model architecture and training.
Change model size by swapping configs — zero code changes needed.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml
import os


@dataclass
class ModelConfig:
    """All model architecture parameters. Change these to scale from 30M to 1B+."""

    # Core dimensions
    vocab_size: int = 32_000
    dim: int = 512            # Hidden dimension (d_model)
    n_layers: int = 6         # Number of transformer blocks
    n_heads: int = 8          # Number of attention heads
    n_kv_heads: int = 4       # Number of key-value heads (for GQA)
    max_seq_len: int = 2048   # Maximum sequence length

    # Feed-forward network
    # SwiGLU FFN hidden dim. If None, computed as 2/3 * 4 * dim, rounded to multiple of 256
    ffn_dim: Optional[int] = None

    # Regularization
    dropout: float = 0.0      # Dropout rate (0 for pre-training, >0 for fine-tuning)

    # Normalization
    norm_eps: float = 1e-5    # RMSNorm epsilon

    # Positional encoding
    rope_theta: float = 10_000.0  # RoPE base frequency

    # Weight tying: share embedding and output projection weights
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        if self.ffn_dim is None:
            # SwiGLU: 2/3 * 4 * dim, rounded to nearest multiple of 256
            raw = int(2 / 3 * 4 * self.dim)
            self.ffn_dim = ((raw + 255) // 256) * 256

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def param_count_estimate(self) -> int:
        """Rough parameter count estimate."""
        embed = self.vocab_size * self.dim
        # Each transformer block: attn(Q,K,V,O) + FFN(gate,up,down) + 2 norms
        attn = self.dim * (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
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
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"  # 10B token sample
    data_dir: str = "data"
    tokenizer_path: str = "tokenizer_data"

    # --- Optimization ---
    batch_size: int = 32       # Per-device micro batch size (sequences)
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size * grad_accum * seq_len tokens
    max_steps: int = 50_000    # Total optimization steps
    warmup_steps: int = 2_000  # Linear warmup steps

    # Learning rate
    peak_lr: float = 6e-4
    min_lr: float = 6e-5       # Minimum LR (cosine decay target)
    weight_decay: float = 0.1

    # Adam
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Gradient clipping
    max_grad_norm: float = 1.0

    # --- Mixed precision ---
    precision: str = "auto"    # "auto", "bf16", "fp16", "fp32"

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 500        # Save checkpoint every N steps
    keep_last_n_checkpoints: int = 5   # Keep last N + best validation
    save_to_hf_hub: bool = False       # Upload checkpoints to HF Hub
    hf_repo_id: str = ""               # HF Hub repo for checkpoint backup

    # --- Logging ---
    log_every_steps: int = 10           # Log metrics every N steps
    eval_every_steps: int = 500         # Run validation every N steps
    generate_every_steps: int = 1000    # Generate sample text every N steps
    visualize_every_steps: int = 1000   # Log model internals every N steps

    # --- Wandb ---
    wandb_project: str = "llm-proto"
    wandb_run_name: str = ""            # Auto-generated if empty
    use_wandb: bool = True

    # --- Performance ---
    num_workers: int = 4
    use_compile: str = "auto"          # "auto", "true", "false"
    gradient_checkpointing: bool = False  # Trade compute for memory (for 1B+ models)

    # --- Reproducibility ---
    seed: int = 42

    # --- Resume ---
    resume: str = ""  # "" = fresh start, "latest" = latest checkpoint, "step_5000" = specific

    # --- Generation sampling ---
    sample_prompts: list = field(default_factory=lambda: [
        "The meaning of life is",
        "In the year 2050, artificial intelligence",
        "Once upon a time in a land far away",
        "The theory of relativity explains",
    ])


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
    """Get a preset model config by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model config: {name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]


def save_config(config, path: str):
    """Save a config dataclass to YAML."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)


def load_model_config(path: str) -> ModelConfig:
    """Load a ModelConfig from YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


def load_train_config(path: str) -> TrainConfig:
    """Load a TrainConfig from YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)
