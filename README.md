# LLM-Proto

A from-scratch LLaMA-style Transformer language model framework built with PyTorch. Train decoder-only language models from **30M to 1B+ parameters** using modern techniques.

## Features

- **LLaMA Architecture** — RMSNorm, Rotary Positional Embeddings (RoPE), SwiGLU FFN, Grouped Query Attention (GQA)
- **Flash Attention** — via PyTorch 2.x `scaled_dot_product_attention`
- **Mixed Precision** — automatic bf16/fp16 selection based on hardware
- **KV-Cache Inference** — efficient autoregressive generation
- **Multi-Source Data Pipeline** — HuggingFace streaming, local `.txt`, `.jsonl` files → packed binary shards
- **BPE Tokenizer** — 32K vocabulary with byte-level fallback (HuggingFace `tokenizers` backend)
- **Google Drive Backup** — automatic checkpoint sync (Colab mount or REST API)
- **Weights & Biases** — experiment tracking, loss curves, sample generations, model visualizations
- **5 Model Presets** — tiny (30M), small (125M), medium (350M), base (500M), large (1B)
- **Environment Support** — Google Colab, vast.ai, local GPU

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the tokenizer

```bash
python scripts/train_tokenizer.py
```

### 3. Train a model

```bash
# Train the tiny model (good for prototyping)
python -m src.train --model tiny --config configs/training.yaml

# Train a larger model with custom settings
python -m src.train --model small --batch_size 16 --peak_lr 3e-4

# Resume from a checkpoint
python -m src.train --model medium --resume latest
```

### 4. Generate text

```bash
# Single prompt
python -m src.generate --checkpoint checkpoints/best.pt --model tiny --prompt "Once upon a time"

# Interactive chat
python -m src.generate --checkpoint checkpoints/best.pt --model tiny
```

### 5. Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --model tiny
```

## Project Structure

```
LLM-Proto/
├── configs/                 # YAML configuration files
│   ├── data.yaml            # Data sources & tokenizer settings
│   ├── training.yaml        # Training hyperparameters
│   └── model_*.yaml         # Model architecture presets (tiny → large)
├── src/                     # Core source code
│   ├── model.py             # Transformer model (RMSNorm, RoPE, GQA, SwiGLU)
│   ├── tokenizer.py         # BPE tokenizer training & inference
│   ├── data.py              # Multi-source data pipeline → binary shards
│   ├── train.py             # Full training loop
│   ├── generate.py          # Text generation with KV-cache
│   ├── evaluate.py          # Validation metrics (loss, perplexity)
│   ├── visualize.py         # Model internals visualization
│   ├── config.py            # Configuration dataclasses & presets
│   ├── utils.py             # Checkpointing, LR schedule, environment detection
│   └── gdrive.py            # Google Drive checkpoint backup
├── scripts/                 # Automation scripts
│   ├── train_tokenizer.py   # Standalone tokenizer training
│   ├── run_training.sh      # tmux-based training launcher
│   └── setup_vastai.sh      # vast.ai instance setup
├── data/                    # Tokenized binary data (generated)
│   └── custom/              # Your own txt/ and jsonl/ data
├── tokenizer_data/          # Trained tokenizer output
├── checkpoints/             # Model checkpoints (generated)
├── LLM-proto.ipynb          # Training notebook (Colab-ready)
├── LLM-inference.ipynb      # Inference notebook
└── requirements.txt         # Python dependencies
```

## Model Presets

| Preset | Params | Dim | Layers | Heads (Q/KV) | Context | Recommended GPU |
|--------|--------|-----|--------|--------------|---------|-----------------|
| `tiny` | ~30M | 512 | 6 | 8/4 | 2048 | Any (T4, etc.) |
| `small` | ~125M | 768 | 12 | 12/4 | 2048 | T4 16GB |
| `medium` | ~350M | 1024 | 24 | 16/4 | 2048 | A10 24GB |
| `base` | ~500M | 1280 | 24 | 20/4 | 2048 | A100 40GB |
| `large` | ~1B | 2048 | 32 | 32/8 | 4096 | A100 80GB |

## Custom Data

Place your data in `data/custom/`:

- **Plain text:** Add `.txt` files to `data/custom/txt/`
- **JSONL:** Add `.jsonl` files (one `{"text": "..."}` per line) to `data/custom/jsonl/`

Then enable the corresponding source in `configs/data.yaml`.

## Training on vast.ai

```bash
# One-time setup
bash scripts/setup_vastai.sh

# Launch training (survives SSH disconnect via tmux)
bash scripts/run_training.sh small
```

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for a comprehensive technical reference of the entire codebase, including architecture details, math, and diagrams.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
