# LLM-Proto — Complete Codebase Documentation

> A from-scratch LLaMA-style Transformer language model framework.  
> Supports training models from **30M to 1B+ parameters** using modern techniques:  
> RoPE, SwiGLU, Grouped Query Attention, Flash Attention, mixed precision, and KV-cache inference.

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Directory Structure](#2-directory-structure)
3. [Configuration System (`src/config.py`)](#3-configuration-system-srcconfigpy)
4. [Model Architecture (`src/model.py`)](#4-model-architecture-srcmodelpy)
5. [Tokenizer (`src/tokenizer.py`)](#5-tokenizer-srctokenizerpy)
6. [Data Pipeline (`src/data.py`)](#6-data-pipeline-srcdatapy)
7. [Training Loop (`src/train.py`)](#7-training-loop-srctrainpy)
8. [Inference & Generation (`src/generate.py`)](#8-inference--generation-srcgeneratepy)
9. [Evaluation (`src/evaluate.py`)](#9-evaluation-srcevaluatepy)
10. [Visualization (`src/visualize.py`)](#10-visualization-srcvisualizepy)
11. [Utilities (`src/utils.py`)](#11-utilities-srcutilspy)
12. [Google Drive Integration (`src/gdrive.py`)](#12-google-drive-integration-srcgdrivepy)
13. [Configuration Files (`configs/`)](#13-configuration-files-configs)
14. [Scripts (`scripts/`)](#14-scripts-scripts)
15. [Training Data Flow — End to End](#15-training-data-flow--end-to-end)
16. [Model Size Presets](#16-model-size-presets)
17. [Dependencies](#17-dependencies)

---

## 1. Project Overview & Architecture

LLM-Proto is a **complete, self-contained framework** for pre-training decoder-only Transformer language models from scratch. It follows the **LLaMA architecture** and is designed to run in three environments:

| Environment | Use Case |
|---|---|
| **Google Colab** | Quick prototyping with free/paid GPUs (T4, A100) |
| **vast.ai** | Affordable cloud GPU rentals for longer training runs |
| **Local** | Development and testing on personal hardware |

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    configs/*.yaml                         │
│         (Model architecture + Training hyperparams)       │
└──────────────┬───────────────────────┬───────────────────┘
               │                       │
               ▼                       ▼
┌──────────────────────┐   ┌───────────────────────┐
│   src/config.py      │   │   src/data.py          │
│   ModelConfig        │   │   Multi-source pipeline │
│   TrainConfig        │   │   HuggingFace / txt /  │
│   5 preset sizes     │   │   jsonl → binary shards │
└──────────┬───────────┘   └───────────┬────────────┘
           │                           │
           ▼                           ▼
┌──────────────────────┐   ┌───────────────────────┐
│   src/model.py       │   │   src/tokenizer.py     │
│   TransformerLM      │   │   BPE tokenizer        │
│   RMSNorm + RoPE +   │   │   32K vocab, byte      │
│   SwiGLU + GQA       │   │   fallback, special    │
│                      │   │   tokens                │
└──────────┬───────────┘   └───────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│                    src/train.py                           │
│   Full training loop: mixed precision, gradient accum,   │
│   cosine LR schedule, checkpointing, eval, generation    │
└──────────┬───────────────────────┬───────────────────────┘
           │                       │
           ▼                       ▼
┌──────────────────────┐   ┌───────────────────────┐
│   src/evaluate.py    │   │   src/visualize.py     │
│   Validation loss,   │   │   Attention heatmaps,  │
│   perplexity, LM     │   │   embedding t-SNE,     │
│   benchmarks         │   │   weight distributions  │
└──────────────────────┘   └───────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│   src/generate.py — Inference with KV-cache              │
│   Single prompt & interactive chat modes                 │
│   Temperature, top-k, top-p sampling                     │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
LLM-Proto/
├── configs/                    # All YAML configuration files
│   ├── data.yaml               # Data sources, tokenizer settings, processing params
│   ├── training.yaml           # Training hyperparameters (LR, batch size, etc.)
│   ├── model_tiny.yaml         #  ~30M params  — prototyping
│   ├── model_small.yaml        # ~125M params  — mid-range
│   ├── model_medium.yaml       # ~350M params  — GPT-2 scale
│   ├── model_base.yaml         # ~500M params  — production-ready
│   └── model_large.yaml        #   ~1B params  — full scale
│
├── src/                        # Core Python source code
│   ├── __init__.py             # Package marker (empty)
│   ├── config.py               # Dataclasses for model & training config
│   ├── model.py                # Full Transformer model implementation
│   ├── tokenizer.py            # BPE tokenizer training & inference
│   ├── data.py                 # Multi-source data pipeline → binary shards
│   ├── train.py                # Training loop with all bells and whistles
│   ├── evaluate.py             # Validation loss / perplexity / benchmarks
│   ├── generate.py             # Text generation with KV-cache
│   ├── visualize.py            # Model internals visualization
│   ├── utils.py                # Checkpointing, LR schedule, logging, env detection
│   └── gdrive.py               # Google Drive upload/download for checkpoints
│
├── scripts/                    # Shell/Python scripts for automation
│   ├── run_training.sh         # Launch training in tmux (vast.ai-friendly)
│   ├── setup_vastai.sh         # One-time vast.ai instance setup
│   └── train_tokenizer.py      # Standalone tokenizer training script
│
├── data/                       # Tokenized binary data (generated)
│   └── custom/                 # User's own data (txt/ and jsonl/ subdirs)
│       ├── README.md           # Instructions for adding custom data
│       ├── txt/                # Plain .txt files
│       └── jsonl/              # JSONL files with {"text": "..."} entries
│
├── tokenizer_data/             # Trained tokenizer output
│   └── tokenizer.json          # The BPE tokenizer model file
│
├── checkpoints/                # Saved model checkpoints (generated)
│
├── LLM-proto.ipynb             # Main training notebook (Colab-ready)
├── LLM-inference.ipynb         # Inference/demo notebook
└── requirements.txt            # Python dependencies
```

---

## 3. Configuration System (`src/config.py`)

The configuration system uses Python **dataclasses** so that every parameter has a typed default, is easily serializable to YAML, and requires **zero code changes** to switch between model sizes.

### 3.1 `ModelConfig` — Architecture Parameters

Controls the entire model shape. Changing these values creates a completely different model.

```python
@dataclass
class ModelConfig:
    vocab_size: int = 32_000       # Vocabulary size (must match tokenizer)
    dim: int = 512                 # Hidden dimension (d_model) — the width of every layer
    n_layers: int = 6              # Number of stacked Transformer blocks
    n_heads: int = 8               # Query attention heads
    n_kv_heads: int = 4            # Key-Value heads (fewer = GQA memory savings)
    max_seq_len: int = 2048        # Maximum context window (tokens)
    ffn_dim: Optional[int] = None  # SwiGLU FFN hidden dim (auto-computed if None)
    dropout: float = 0.0           # Dropout rate (0 for pre-training)
    norm_eps: float = 1e-5         # RMSNorm epsilon for numerical stability
    rope_theta: float = 10_000.0   # RoPE base frequency
    tie_embeddings: bool = True    # Share input embedding & output projection weights
```

**Key design decisions:**

| Parameter | Why It Matters |
|---|---|
| `n_kv_heads < n_heads` | Enables **Grouped Query Attention (GQA)** — fewer K,V heads are shared across multiple Q heads, reducing memory by ~2-4x during inference with minimal quality loss. LLaMA 2 70B uses this. |
| `ffn_dim = None` | Auto-computes as `round_to_256(2/3 × 4 × dim)` — the **SwiGLU** formula. The 2/3 factor compensates for SwiGLU having 3 weight matrices instead of 2. |
| `tie_embeddings = True` | Shares the embedding matrix with the output projection (saves `vocab_size × dim` parameters, ~16M for 32K vocab). Standard in LLaMA. |
| `rope_theta = 10,000` | Standard RoPE frequency base. Higher values (e.g., 500K in LLaMA 3) extend context length. |

**Auto-computed properties:**
- `head_dim = dim // n_heads` — dimension of each attention head
- `param_count_estimate()` — rough parameter count without instantiating the model

### 3.2 `TrainConfig` — Training Hyperparameters

All training settings in one place, loadable from `configs/training.yaml`.

```python
@dataclass
class TrainConfig:
    # --- Data ---
    dataset_name: str = "HuggingFaceFW/fineweb-edu"  # Default dataset
    data_dir: str = "data"                            # Where binary shards live
    tokenizer_path: str = "tokenizer_data"

    # --- Optimization ---
    batch_size: int = 32                    # Per-device micro batch size
    gradient_accumulation_steps: int = 4    # Effective batch = 32 × 4 = 128 sequences
    max_steps: int = 50_000                 # Total optimizer steps
    warmup_steps: int = 2_000               # Linear LR warmup

    # --- Learning Rate ---
    peak_lr: float = 6e-4                   # Peak learning rate
    min_lr: float = 6e-5                    # Cosine decay floor (10% of peak)
    weight_decay: float = 0.1               # AdamW weight decay

    # --- Adam Optimizer ---
    adam_beta1: float = 0.9                 # Momentum
    adam_beta2: float = 0.95                # Variance (lower than default 0.999 for stability)
    adam_eps: float = 1e-8

    # --- Gradient Clipping ---
    max_grad_norm: float = 1.0              # Global L2 norm clipping

    # --- Mixed Precision ---
    precision: str = "auto"                 # "auto" picks bf16 or fp16 based on GPU

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 500
    keep_last_n_checkpoints: int = 5        # Auto-cleanup old checkpoints

    # --- Google Drive Backup ---
    backup_to_gdrive: bool = False
    gdrive_folder_id: str = ""
    gdrive_credentials_path: str = ""

    # --- Logging ---
    log_every_steps: int = 10
    eval_every_steps: int = 500
    generate_every_steps: int = 1000        # Sample generation for qualitative check
    visualize_every_steps: int = 1000       # Model internals plots

    # --- Wandb ---
    use_wandb: bool = True
    wandb_project: str = "llm-proto"

    # --- Resume ---
    resume: str = ""                        # "", "latest", "best", or "step_N"
```

### 3.3 Preset Model Configs

Five pre-defined model sizes, selectable by name:

| Name | dim | Layers | Heads | KV Heads | Seq Len | ~Params |
|------|-----|--------|-------|----------|---------|---------|
| `tiny` | 512 | 6 | 8 | 4 | 2048 | **30M** |
| `small` | 768 | 12 | 12 | 4 | 2048 | **125M** |
| `medium` | 1024 | 24 | 16 | 4 | 2048 | **350M** |
| `base` | 1280 | 24 | 20 | 4 | 2048 | **500M** |
| `large` | 2048 | 32 | 32 | 8 | 4096 | **1B** |

### 3.4 Config I/O

```python
save_config(config, "path.yaml")       # Serialize dataclass → YAML
load_model_config("path.yaml")         # YAML → ModelConfig
load_train_config("path.yaml")         # YAML → TrainConfig
get_model_config("tiny")               # Preset name → ModelConfig
```

---

## 4. Model Architecture (`src/model.py`)

This file implements a **complete LLaMA-style decoder-only Transformer** from scratch using pure PyTorch. Every component is explained below.

### 4.1 `RMSNorm` — Root Mean Square Normalization

```python
class RMSNorm(nn.Module):
    norm = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * norm).type_as(x) * self.weight
```

**What it does:** Normalizes activations by their root mean square (not mean and variance like LayerNorm).

**Why RMSNorm over LayerNorm:**
- ~15% faster — skips the mean subtraction step
- Same quality — empirically validated in LLaMA and other modern LLMs
- Learnable scale parameter `weight` (no bias) preserves the ability to shift distributions

**Math:** Given input $x \in \mathbb{R}^d$:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

where $\gamma$ is the learnable scale vector and $\epsilon$ prevents division by zero.

### 4.2 Rotary Positional Embeddings (RoPE)

```python
def precompute_rope_freqs(dim, max_seq_len, theta=10_000.0)
def apply_rope(x, freqs)
```

**What it does:** Encodes position information directly into query and key vectors by rotating pairs of dimensions.

**How it works:**
1. **Precompute frequencies:** For each dimension pair $k$ and position $m$:
   $$\theta_k = \frac{1}{\theta^{2k/d}}, \quad f_{m,k} = e^{i \cdot m \cdot \theta_k}$$
   This creates a complex-valued frequency table of shape `(max_seq_len, head_dim/2)`.

2. **Apply rotation:** Reshape each head's vector into pairs, treat each pair as a complex number, multiply by the precomputed frequency, and convert back to real.

**Why RoPE over absolute/sinusoidal embeddings:**
- Relative position is encoded via the angle between Q and K rotations — attention naturally decays with distance
- No extra parameters (computed from a formula, not learned)
- Extrapolates better to longer sequences than seen during training
- Compatible with KV-cache — position is baked into the Q/K vectors, not added later

### 4.3 `Attention` — Grouped Query Attention (GQA) with Flash Attention

```python
class Attention(nn.Module):
    def __init__(self, config):
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)       # Query projection
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)    # Key projection (fewer heads)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)    # Value projection (fewer heads)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)       # Output projection
```

**GQA explained:**  
Standard Multi-Head Attention (MHA) uses the same number of Q, K, and V heads. **Grouped Query Attention** uses fewer K and V heads — each KV head is shared by `n_heads / n_kv_heads` query heads.

```
Standard MHA (8 heads):     Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈
                            K₁ K₂ K₃ K₄ K₅ K₆ K₇ K₈    (8 KV heads)

GQA (8Q, 4KV):             Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈
                            K₁  K₁  K₂  K₂  K₃  K₃  K₄  K₄    (4 KV heads, each shared by 2 Q heads)
```

**Benefits:** 2-4x less KV-cache memory during inference, minimal quality loss.

**Forward pass step by step:**
1. Project input to Q, K, V tensors
2. Apply RoPE to Q and K (not V — position only affects attention routing)
3. If KV-cache exists (inference), concatenate with previous K, V
4. Repeat KV heads to match Q heads count (GQA expansion)
5. Call `F.scaled_dot_product_attention()` — this automatically uses **Flash Attention** on compatible GPUs (PyTorch 2.x), which is:
   - **Memory-efficient:** O(N) instead of O(N²)
   - **Hardware-optimized:** Fuses computation into a single GPU kernel
   - **Automatic causal masking** when `is_causal=True`
6. Reshape and project output through `wo`

### 4.4 `FeedForward` — SwiGLU FFN

```python
class FeedForward(nn.Module):
    def forward(self, x):
        return w_down(SiLU(w_gate(x)) * w_up(x))
```

**SwiGLU** (Swish-Gated Linear Unit) replaces the standard ReLU FFN:

$$\text{FFN}(x) = W_{down} \cdot (\text{SiLU}(W_{gate} \cdot x) \odot W_{up} \cdot x)$$

Where $\odot$ is element-wise multiplication and $\text{SiLU}(x) = x \cdot \sigma(x)$.

**Three weight matrices:**
| Matrix | Shape | Purpose |
|--------|-------|---------|
| `w_gate` | `(dim, ffn_dim)` | Produces the gating signal (passed through SiLU activation) |
| `w_up` | `(dim, ffn_dim)` | Produces the value signal |
| `w_down` | `(ffn_dim, dim)` | Projects back to model dimension |

**Why SwiGLU over ReLU:**
- Consistently better quality at same parameter count (PaLM, LLaMA papers)
- The gating mechanism lets the network learn which information to pass through
- The `2/3 × 4d` hidden size compensates for the extra matrix (same total parameters as a standard `4d` FFN)

### 4.5 `TransformerBlock` — Pre-Norm Residual Block

```python
class TransformerBlock(nn.Module):
    def forward(self, x, rope_freqs, mask, kv_cache):
        h, new_kv = self.attn(self.attn_norm(x), rope_freqs, mask, kv_cache)  # Norm → Attn
        x = x + h                                                              # Residual
        x = x + self.ffn(self.ffn_norm(x))                                     # Norm → FFN → Residual
        return x, new_kv
```

**Pre-norm** (normalize before sublayer, not after):
```
x ──→ RMSNorm ──→ Attention ──→ (+) ──→ RMSNorm ──→ FFN ──→ (+) ──→ output
  └────────────────────────────┘   └──────────────────────┘
           Residual connection           Residual connection
```

**Why pre-norm:** More stable training, especially for deep models. The residual stream acts as a "highway" for gradients to flow back through many layers without vanishing.

### 4.6 `TransformerLM` — Full Model

```python
class TransformerLM(nn.Module):
    def __init__(self, config):
        self.tok_emb = nn.Embedding(vocab_size, dim)        # Token → vector
        self.layers = [TransformerBlock(config)] × n_layers  # N transformer blocks
        self.norm = RMSNorm(dim)                             # Final normalization
        self.output = nn.Linear(dim, vocab_size)             # Vector → logits
        # Weight tying: output.weight = tok_emb.weight
```

**Full forward pass:**
```
input_ids: [The, cat, sat, on]
     │
     ▼
Token Embedding  →  [512-dim vectors]
     │
     ▼ (× n_layers)
┌─────────────────┐
│ TransformerBlock │  RMSNorm → GQA Attention (+ RoPE) → Residual → RMSNorm → SwiGLU FFN → Residual
└─────────────────┘
     │
     ▼
Final RMSNorm
     │
     ▼
Output Projection  →  [32000-dim logits per position]
     │
     ▼
Cross-Entropy Loss (if targets provided)
```

**Weight initialization:**
- All weights: Normal distribution with std=0.02 (GPT-2 convention)
- Residual projections (`wo` and `w_down`): Scaled by $\frac{1}{\sqrt{2 \times n\_layers}}$ for training stability — prevents the variance from growing with depth

**Generation (`model.generate()`):**
- Uses KV-cache: first pass processes the full prompt, subsequent passes process one token at a time
- Supports temperature scaling, top-k filtering, top-p (nucleus) sampling, and greedy decoding
- Stops at EOS token or `max_new_tokens`

---

## 5. Tokenizer (`src/tokenizer.py`)

A **BPE (Byte-Pair Encoding) tokenizer** built on HuggingFace's Rust-backed `tokenizers` library for maximum speed.

### 5.1 Design Choices

| Choice | Detail |
|--------|--------|
| **Algorithm** | BPE with byte-level fallback — can encode ANY text, no "unknown" tokens |
| **Vocabulary** | 32,000 tokens (same as LLaMA) |
| **Pre-tokenization** | ByteLevel — splits on UTF-8 bytes before BPE, enabling multilingual support |
| **Special tokens** | `<\|bos\|>`, `<\|eos\|>`, `<\|pad\|>`, `<\|im_start\|>`, `<\|im_end\|>` |

### 5.2 Special Tokens

```python
SPECIAL_TOKENS = {
    "bos": "<|bos|>",       # Beginning of sequence — prepended to every input
    "eos": "<|eos|>",       # End of sequence — signals generation should stop
    "pad": "<|pad|>",       # Padding — for batching variable-length sequences
    "im_start": "<|im_start|>",  # Chat template: start of message turn
    "im_end": "<|im_end|>",      # Chat template: end of message turn
}
```

### 5.3 Training Pipeline

```python
LLMTokenizer.train(
    texts=text_iterator(),    # Any iterator of strings
    vocab_size=32_000,
    save_path="tokenizer_data",
    min_frequency=2,          # Only merge pairs seen ≥2 times
)
```

The BPE algorithm:
1. Start with individual bytes as the initial vocabulary
2. Count all adjacent token pairs in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat until `vocab_size` is reached

### 5.4 Encode / Decode API

```python
tok = LLMTokenizer("tokenizer_data")

tok.encode("Hello world")              # → [bos_id, 1234, 5678]
tok.encode("Hello", add_bos=False)     # → [1234]
tok.decode([1234, 5678])               # → "Hello world"
tok.encode_batch(["text1", "text2"])    # Batch encoding (parallelized in Rust)
```

---

## 6. Data Pipeline (`src/data.py`)

Handles everything from raw text to GPU-ready batches.

### 6.1 Multi-Source Text Ingestion

Three source types, all configured in `configs/data.yaml`:

| Source Type | Input | How It Works |
|---|---|---|
| `huggingface` | Dataset name + subset | Streams from HuggingFace Hub (no local storage needed) |
| `text_dir` | Directory of `.txt` files | Recursively scans directory, each file = one document |
| `jsonl` | `.jsonl` file or directory | Each line is `{"text": "..."}`, reads `text_field` key |

```python
_SOURCE_ITERATORS = {
    "huggingface": _iter_huggingface,
    "text_dir": _iter_text_dir,
    "jsonl": _iter_jsonl,
}
```

All sources yield plain text strings. Documents shorter than 50 characters are skipped.

### 6.2 Tokenization & Binary Shard Creation

```
Raw Text → BPE Tokenizer → uint16 token IDs → Binary .bin shard files
```

The `tokenize_and_save()` function:

1. **Reads text** from all configured sources sequentially
2. **Tokenizes** each document (adding BOS + EOS tokens)
3. **Splits** into train/val: every Nth document goes to validation (controlled by `val_every`)
4. **Packs** token IDs into fixed-size binary shards (`shard_size` tokens each, default 100M)
5. **Saves** as `train_0000.bin`, `train_0001.bin`, ... and `val.bin`

**Binary format:** Flat array of `uint16` values. Why uint16? Vocabulary is 32K, which fits in 16 bits (max 65,535). This halves storage compared to int32.

### 6.3 Memory-Mapped Dataset

```python
class ShardedTokenDataset(Dataset):
```

At training time, data is NOT loaded into RAM. Instead:

1. **`np.memmap`** maps each `.bin` file directly into virtual memory
2. The OS transparently pages data from disk as needed
3. **Zero-copy access** — the GPU gets data without intermediate copies

Each sample is a contiguous chunk of `seq_len` tokens:
```
Shard:  [tok₁, tok₂, tok₃, tok₄, tok₅, tok₆, tok₇, tok₈, tok₉, ...]
         └──── sample 0 ────┘  └──── sample 1 ────┘  └──── ...
         input:  [tok₁, tok₂, tok₃, tok₄]    ← input_ids
         target: [tok₂, tok₃, tok₄, tok₅]    ← shifted by 1 (next-token prediction)
```

### 6.4 DataLoader Creation

```python
create_dataloader(data_dir, seq_len, batch_size, split, num_workers, shuffle)
```

Returns a standard PyTorch `DataLoader` with:
- `pin_memory=True` — faster CPU→GPU transfer
- `drop_last=True` — avoids variable batch sizes
- `num_workers=4` — parallel data loading

---

## 7. Training Loop (`src/train.py`)

The `train()` function orchestrates the entire pre-training process.

### 7.1 Setup Phase

```python
def train(model_config, train_config):
    # 1. Detect environment (Colab / vast.ai / local)
    # 2. Select device (CUDA > MPS > CPU) and dtype (bf16 > fp16 > fp32)
    # 3. Set random seeds for reproducibility
    # 4. Build model and move to device
    # 5. Optionally torch.compile() for ~20% speedup
    # 6. Build optimizer with separate weight-decay groups
    # 7. Create train + val dataloaders
    # 8. Initialize Wandb logging
    # 9. Resume from checkpoint if configured
```

### 7.2 Optimizer Configuration

The optimizer separates parameters into two groups:

| Group | Parameters | Weight Decay |
|---|---|---|
| **Decay** | 2D+ weight matrices (attention projections, FFN weights) | 0.1 |
| **No-decay** | 1D params (norms, biases) | 0.0 |

Uses **fused AdamW** on CUDA (a single kernel instead of multiple, ~5-10% faster).

### 7.3 Learning Rate Schedule

**Linear warmup** → **Cosine decay**:

```
LR
 ▲
 │     ╱‾‾‾‾‾╲
 │    ╱        ╲
 │   ╱          ╲
 │  ╱            ╲___________
 │ ╱                          min_lr
 └─────────────────────────────→ Steps
   │warmup│      cosine decay
```

$$\text{LR}(t) = \begin{cases} 
\text{peak\_lr} \cdot \frac{t}{\text{warmup\_steps}} & t < \text{warmup\_steps} \\
\text{min\_lr} + \frac{1}{2}(\text{peak\_lr} - \text{min\_lr})(1 + \cos(\pi \cdot \frac{t - w}{T - w})) & \text{otherwise}
\end{cases}$$

### 7.4 Training Step (with Gradient Accumulation)

Each optimizer step actually processes multiple micro-batches:

```
Step:
 ├─ micro_batch 1 → forward → loss/grad_accum → backward
 ├─ micro_batch 2 → forward → loss/grad_accum → backward
 ├─ micro_batch 3 → forward → loss/grad_accum → backward
 ├─ micro_batch 4 → forward → loss/grad_accum → backward
 │
 ├─ Gradient clipping (max_norm = 1.0)
 └─ Optimizer step
```

**Effective batch size** = `batch_size × gradient_accumulation_steps × seq_len` tokens per step.

With defaults: `32 × 4 × 2048 = 262,144 tokens/step`.

### 7.5 Mixed Precision Training

```python
amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
scaler = GradScaler(enabled=(dtype == torch.float16))
```

- **bf16 (A100, H100):** No loss scaling needed — bf16 has the same exponent range as fp32
- **fp16 (T4, V100):** Requires `GradScaler` to prevent underflow in gradients
- **fp32:** Full precision fallback for CPU or debugging

### 7.6 Periodic Actions

| Action | Frequency | What Happens |
|---|---|---|
| **Logging** | Every 10 steps | Loss, perplexity, LR, tokens/sec → Wandb + console |
| **Validation** | Every 500 steps | Run model on val set, compute avg loss + perplexity |
| **Checkpoint** | Every 500 steps | Save model + optimizer + RNG state + configs |
| **Generation** | Every 1000 steps | Generate text from sample prompts (qualitative check) |
| **Visualization** | Every 1000 steps | Attention maps, weight distributions, activation stats |

### 7.7 Checkpoint Contents

Each `.pt` checkpoint file contains:

```python
{
    "step": 5000,
    "loss": 3.2145,
    "model_state_dict": {...},          # All model weights
    "optimizer_state_dict": {...},       # Adam momentum + variance buffers
    "model_config": {...},               # Architecture params (for reconstruction)
    "train_config": {...},               # Training hyperparams
    "rng_state": {                       # For exact reproducibility on resume
        "python": ...,
        "numpy": ...,
        "torch": ...,
        "cuda": ...,
    },
}
```

### 7.8 CLI Entry Point

```bash
python -m src.train --model tiny --config configs/training.yaml
python -m src.train --model configs/model_large.yaml --resume latest --no_wandb
python -m src.train --model small --batch_size 16 --peak_lr 3e-4
```

Supports overriding any `TrainConfig` field from the command line.

---

## 8. Inference & Generation (`src/generate.py`)

### 8.1 Model Loading

```python
model, tokenizer = load_model_for_inference("tiny", "checkpoints/latest.pt")
```

Loads model config from the checkpoint, rebuilds the architecture, loads weights, and loads the tokenizer.

### 8.2 Generation with KV-Cache

The KV-cache avoids recomputing attention for previously generated tokens:

```
Without cache:  Generate token 100 → recompute attention over all 100 tokens
With cache:     Generate token 100 → recompute attention over 1 token, reuse 99 cached K,V
```

This makes generation **O(n)** instead of **O(n²)** in sequence length.

### 8.3 Sampling Strategies

```python
generate_text(model, tokenizer, prompt,
    temperature=0.8,   # Controls randomness (0 = greedy, 1 = standard, >1 = more random)
    top_k=50,          # Keep only top 50 most likely tokens
    top_p=0.9,         # Keep tokens until cumulative probability reaches 90%
)
```

**Temperature** scales the logits: $p_i = \text{softmax}(z_i / T)$
- $T < 1$: Sharper distribution → more deterministic
- $T = 1$: Standard sampling
- $T > 1$: Flatter distribution → more creative/random

**Top-k:** Zero out all logits except the k highest.

**Top-p (nucleus):** Sort tokens by probability, keep the smallest set whose cumulative probability ≥ p. More adaptive than top-k.

### 8.4 Interactive Chat Mode

```python
interactive_chat(model, tokenizer, temperature=0.8, top_k=50)
```

A terminal-based chat interface with:
- Multi-turn conversation
- Live parameter tuning via commands: `/temp 0.5`, `/topk 100`, `/topp 0.95`
- `clear` to reset, `quit` to exit

---

## 9. Evaluation (`src/evaluate.py`)

### 9.1 Validation Metrics

```python
compute_val_metrics(model, val_loader, device, amp_ctx, max_batches=200)
```

Returns:
- **Loss:** Average per-token cross-entropy loss
- **Perplexity:** $\text{PPL} = e^{\text{loss}}$ — measures how "surprised" the model is. Lower = better. A perplexity of 20 means the model is as uncertain as choosing uniformly among 20 tokens.
- **Tokens evaluated:** Total token count for reference

### 9.2 Standalone Evaluation

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --model tiny --max_batches 500
```

---

## 10. Visualization (`src/visualize.py`)

Six visualization types for understanding model internals, all logged to Wandb:

### 10.1 Attention Heatmap

`plot_attention_heatmap(model, input_ids, tokenizer, layer_idx, head_idx)`

Shows what each token "attends to" — a 2D heatmap where position (i, j) shows how much token i attends to token j. The causal mask makes this lower-triangular (tokens can only attend to past tokens).

### 10.2 Embedding Space (t-SNE)

`plot_embedding_space(model, tokenizer, n_tokens=500)`

Projects the 512+ dimensional token embeddings into 2D using t-SNE. Reveals clustering structure — semantically similar tokens should cluster together after training.

### 10.3 Weight Distributions

`plot_weight_distributions(model)`

Histograms of weight standard deviations per layer type. Useful for detecting:
- **Vanishing weights** (std ≈ 0) — layer isn't learning
- **Exploding weights** (very large std) — training instability

### 10.4 Activation Statistics

`plot_activation_stats(model, input_ids)`

Mean and standard deviation of activations at each layer's output. Healthy training shows:
- Mean near 0 (residual stream centered)
- Std roughly constant across layers (no explosion/collapse)

### 10.5 Per-Token Loss Heatmap

`plot_token_loss_heatmap(model, input_ids, tokenizer)`

Shows which specific tokens are hardest for the model to predict. Useful for understanding what the model has/hasn't learned.

### 10.6 Gradient Norms

`plot_gradient_norms(grad_norms_by_layer)`

Bar chart of gradient L2 norms per layer. Detects:
- **Vanishing gradients** — early layers have near-zero gradients
- **Exploding gradients** — some layers have disproportionately large gradients

### 10.7 Orchestration

`generate_all_visualizations()` is called during training every `visualize_every_steps` and produces all plots at once, logging them to Wandb with proper step tracking. Embedding space plots are generated less frequently (every 5000 steps) because t-SNE is expensive.

---

## 11. Utilities (`src/utils.py`)

### 11.1 Environment Detection

```python
detect_environment()  # → "colab" | "vastai" | "local"
get_device()          # → torch.device("cuda") or "mps" or "cpu"
get_dtype("auto")     # → bf16 (A100+) or fp16 (T4/V100) or fp32 (CPU)
should_compile("auto")# → True on CUDA + PyTorch 2.x (not Colab T4)
```

### 11.2 Checkpointing System

```python
save_checkpoint(model, optimizer, step, loss, model_config, train_config, dir, is_best)
```

Saves:
- `step_N.pt` — numbered checkpoint
- `latest.pt` — always points to most recent (easy resume)
- `best.pt` — best validation loss so far
- Auto-cleans old checkpoints, keeping last N

```python
load_checkpoint(dir, resume, model, optimizer, device, gdrive_folder_id)
```

Loads checkpoint, restoring:
- Model weights
- Optimizer state (momentum, variance buffers)
- RNG states (Python, NumPy, PyTorch, CUDA) for exact reproducibility
- Falls back to **Google Drive download** if checkpoint not found locally

```python
has_checkpoint(dir, resume, gdrive_folder_id)  # Check existence locally or on Drive
```

### 11.3 Metrics & Logging

```python
class MetricsTracker:
    log(metrics, step)          # Log to Wandb + internal history
    log_image(key, figure, step) # Log matplotlib figures to Wandb
    log_config(config)           # Log config dict to Wandb
    finish()                     # Close Wandb run
```

```python
class Timer:
    step()                 # Record timestamp
    tokens_per_sec(n)      # Compute throughput
    elapsed()              # Total wall-clock time
```

### 11.4 Learning Rate Schedule

```python
get_lr(step, warmup_steps, max_steps, peak_lr, min_lr)
```

Linear warmup followed by cosine decay to `min_lr`.

### 11.5 Reproducibility

```python
set_seed(42)  # Seeds: Python random, NumPy, PyTorch (CPU + all CUDA devices)
```

---

## 12. Google Drive Integration (`src/gdrive.py`)

Two operational modes for checkpoint backup/restore:

### 12.1 Colab Mode

When running in Google Colab (`google.colab` in `sys.modules`):
- **Mounts** Google Drive to `/content/drive/MyDrive/`
- `folder_id` is a **folder name** (e.g., `"LLM"` → `/content/drive/MyDrive/LLM/`)
- Uses simple `shutil.copy2()` — no API credentials needed

### 12.2 API Mode (Local / vast.ai)

For non-Colab environments:
- Uses the **Google Drive REST API v3**
- `folder_id` is the **real Drive folder ID** from the URL (e.g., `1AbCdEfGhIjK...`)
- Requires either:
  - A **service account JSON** file (set `gdrive_credentials_path`)
  - Or **Application Default Credentials** (ADC)
- Supports **resumable uploads** for large checkpoint files
- Caches the Drive service client (singleton pattern)

### 12.3 Available Functions

| Function | Purpose |
|---|---|
| `upload_to_gdrive(local_path, folder_id, creds)` | Upload a file (create or update) |
| `download_from_gdrive(filename, folder_id, local_dir, creds)` | Download a file by name |
| `list_remote_checkpoints(folder_id, creds)` | List all `.pt` files in folder |
| `cleanup_remote_checkpoints(folder_id, keep_n, creds)` | Remove old `step_*.pt` files |
| `reset_service()` | Clear cached API client (re-authenticate) |

---

## 13. Configuration Files (`configs/`)

### 13.1 `data.yaml` — Data Pipeline

```yaml
tokenizer:
  vocab_size: 32000              # BPE vocabulary size
  num_samples: 50000             # Texts used to train the tokenizer
  save_path: tokenizer_data

processing:
  output_dir: data               # Where binary shards go
  max_tokens:                    # null = process everything
  shard_size: 100000000          # 100M tokens per shard file
  val_ratio: 0.005               # 0.5% of docs → validation

sources:                         # List of data sources
  - type: huggingface
    name: HuggingFaceFW/fineweb-edu
    subset: sample-10BT
    split: train
    text_field: text
    weight: 1.0
```

Sources can be mixed. The `weight` field controls relative sampling ratios when combining multiple sources.

### 13.2 `training.yaml` — Training Hyperparameters

All `TrainConfig` fields as YAML. CLI flags override these.

### 13.3 `model_*.yaml` — Architecture Presets

One file per model size. Each contains the `ModelConfig` fields for that size.

---

## 14. Scripts (`scripts/`)

### 14.1 `train_tokenizer.py`

Standalone script to train the BPE tokenizer:

```bash
python scripts/train_tokenizer.py                    # Uses configs/data.yaml
python scripts/train_tokenizer.py --config my.yaml   # Custom config
python scripts/train_tokenizer.py --force             # Retrain even if exists
```

Reads sources from `data.yaml`, streams up to `num_samples` texts, trains BPE, saves to `tokenizer_data/tokenizer.json`.

### 14.2 `run_training.sh`

Launches training in a **tmux session** so it survives SSH disconnects (critical for vast.ai):

```bash
bash scripts/run_training.sh              # tiny model
bash scripts/run_training.sh small        # small model
bash scripts/run_training.sh medium --resume latest  # resume medium
```

Creates a named tmux session `train_<model_size>` and logs to `logs/`.

### 14.3 `setup_vastai.sh`

One-time setup for vast.ai GPU instances:

```bash
bash scripts/setup_vastai.sh
```

Installs system packages (git, tmux, htop), Python dependencies, and creates necessary directories.

---

## 15. Training Data Flow — End to End

```
Step 1: Configure sources
    configs/data.yaml → HuggingFace datasets + local txt/jsonl files

Step 2: Train tokenizer (once)
    python scripts/train_tokenizer.py
    Raw text → BPE algorithm → tokenizer_data/tokenizer.json

Step 3: Tokenize & shard data
    (Handled in notebook or via data.py's tokenize_and_save())
    Text sources → tokenizer.encode() → packed uint16 binary shards
    data/train_0000.bin, data/train_0001.bin, ..., data/val.bin

Step 4: Train model
    python -m src.train --model tiny --config configs/training.yaml
    Binary shards → memory-mapped DataLoader → Transformer → loss → backprop

Step 5: Evaluate
    python -m src.evaluate --checkpoint checkpoints/best.pt --model tiny

Step 6: Generate text
    python -m src.generate --checkpoint checkpoints/best.pt --model tiny --prompt "Hello"
```

---

## 16. Model Size Presets

| Config | dim | Layers | Heads (Q/KV) | FFN dim | Seq Len | ~Parameters | GPU Memory (bf16) | Use Case |
|--------|-----|--------|-------------|---------|---------|------------|-------------------|---------|
| **tiny** | 512 | 6 | 8/4 | 1536 | 2048 | ~30M | ~256 MB | Debugging, prototyping |
| **small** | 768 | 12 | 12/4 | 2048 | 2048 | ~125M | ~1 GB | Colab T4, learning |
| **medium** | 1024 | 24 | 16/4 | 2816 | 2048 | ~350M | ~2.5 GB | GPT-2 scale experiments |
| **base** | 1280 | 24 | 20/4 | 3584 | 2048 | ~500M | ~4 GB | Production-quality |
| **large** | 2048 | 32 | 32/8 | 5632 | 4096 | ~1B | ~8 GB | Full scale, A100 recommended |

---

## 17. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.1.0 | Model, training, CUDA, Flash Attention |
| `tokenizers` | ≥0.15.0 | Fast BPE tokenizer (Rust backend) |
| `datasets` | ≥2.16.0 | HuggingFace dataset streaming |
| `wandb` | ≥0.16.0 | Experiment tracking & visualization |
| `safetensors` | ≥0.4.0 | Safe model weight serialization |
| `lm-eval` | ≥0.4.0 | LM evaluation harness (benchmarks) |
| `matplotlib` | ≥3.8.0 | Visualization plots |
| `seaborn` | ≥0.13.0 | Heatmap styling |
| `scikit-learn` | ≥1.3.0 | t-SNE for embedding visualization |
| `numpy` | ≥1.24.0 | Array operations, memory mapping |
| `pyyaml` | ≥6.0 | YAML config parsing |
| `tqdm` | ≥4.66.0 | Progress bars |
| `google-api-python-client` | ≥2.100.0 | Google Drive API |
| `google-auth` | ≥2.23.0 | Google authentication |
