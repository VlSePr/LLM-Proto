# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch LLaMA-style decoder-only LM in PyTorch (RMSNorm, RoPE, GQA, SwiGLU), with a tokenizer
trainer, a fingerprinted shard-building data pipeline, an exactly-resumable training loop, and inference
/ evaluation CLIs. All logic lives in `src/`; the notebooks (`LLM_proto.ipynb` for Colab training,
`LLM-inference.ipynb`) are thin drivers that `from src... import` — put new behaviour in `src/`, not in
notebook cells. `DOCUMENTATION.md` is a long technical reference (math, diagrams) worth consulting
before changing `model.py` or `train.py`.

## Commands

The project venv is `.venv/` (Python 3.12, CPU-only torch on this machine). On Windows call
`.venv/Scripts/python`. No `pyproject.toml`: there is no repo-level pytest or ruff config.

```bash
pip install -r requirements-dev.txt          # includes requirements.txt + pytest, ruff, nbstripout

# Tests (CPU only, ~6 s). PYTHONIOENCODING is needed on Windows because src/ prints non-ASCII.
PYTHONIOENCODING=utf-8 python -m pytest -q
python -m pytest tests/test_train_resume.py -q                       # one file
python -m pytest tests/test_model.py::test_forward_shapes_and_loss -q  # one test

ruff check src tests scripts    # NOTE: the tree is not ruff-clean (~180 findings, mostly UP/I
                                # annotation & import-order rules). Do not run `ruff --fix` wholesale.

# Full CPU smoke pipeline on the toy corpus in tests/fixtures/corpus (writes to gitignored smoke_run/)
python scripts/train_tokenizer.py --config tests/fixtures/data_smoke.yaml --force
python -m src.data --config tests/fixtures/data_smoke.yaml
python -m src.train --model tests/fixtures/model_smoke.yaml --config tests/fixtures/training_smoke.yaml --max_steps 10
python -m src.train --model tests/fixtures/model_smoke.yaml --config tests/fixtures/training_smoke.yaml --max_steps 20 --resume latest
python -m src.generate --checkpoint smoke_run/checkpoints/latest.pt --prompt "Alice was" --max_tokens 16
python -m src.evaluate --checkpoint smoke_run/checkpoints/latest.pt --data_dir smoke_run/data --batch_size 2

# Real pipeline (see README for flags)
python scripts/train_tokenizer.py                       # -> tokenizer_data/ (committed)
python -m src.data --config configs/data.yaml           # -> data/train_NNNN.bin, val.bin, manifest.json
python -m src.train --model tiny --config configs/training.yaml --no_wandb --no_gdrive
```

Notebook outputs are stripped on commit via `.gitattributes` (`nbstripout --install` once after cloning).

## Architecture: the things that span files

### Configuration (`src/config.py`)
- `ModelConfig` and `TrainConfig` are dataclasses. `config_from_dict` **rejects unknown keys** and
  coerces scalars (PyYAML parses `3e-4` as a *string*, so numeric fields are converted explicitly).
  Adding a field means adding it to the dataclass; YAML files may omit fields but never add extras.
- `--model` accepts a preset name (`tiny/small/medium/base/large`, defined in `MODEL_CONFIGS`) **or** a
  YAML path. Precedence is CLI flag > `--config` YAML > dataclass defaults (`train.main()`).
- Machine-specific values (Drive folder, resume target) must not go into `configs/training.yaml`;
  use CLI flags or a gitignored `configs/local.yaml`.

### Data pipeline (`src/data.py`) — training never tokenizes
- `configs/data.yaml` lists sources (`huggingface` streaming, `text_dir`, `jsonl`). `python -m src.data`
  runs `ensure_tokenized_data`, which computes a **fingerprint** of tokenizer + source files + processing
  params, checks `data/manifest.json` (and shard sizes), optionally falls back to a Google Drive cache at
  `<folder>/tokenized/<fingerprint>/`, and only then tokenizes into uint16 `train_NNNN.bin` shards +
  `val.bin` (every `val_every`-th document). Changing the tokenizer or any source file changes the
  fingerprint and triggers a rebuild.
- `src.train` only calls `create_dataloader(data_dir, ...)` and expects shards to already exist.
- Train split uses `IterableShardDataset` (streaming, epoch-seeded shard shuffle, per-epoch random window
  offset, shuffle buffer, worker-aware). Its sample order is a **deterministic function of
  (seed, epoch, worker layout)**, and `skip_batches` fast-forwards on the index stream. This is what makes
  exact resume work; the trainer calls `set_epoch()` and re-creates `iter(loader)` at each epoch boundary,
  so the train loader deliberately has no `persistent_workers`. Val uses the map-style
  `ShardedTokenDataset`.

### Checkpoints (`src/utils.py`) and resume (`src/train.py`)
- A checkpoint's `step` is the **last completed** optimizer step; resume continues at `step + 1`, and the
  final checkpoint is `step = max_steps - 1`. It stores model, optimizer, GradScaler, RNG (Python/NumPy/
  torch/CUDA), `epoch`, `batches_in_epoch`, `tokens_seen`, `best_val_loss`, and both configs as dicts.
- `save_checkpoint` writes `step_N.pt` atomically (tmp + `os.replace`), copies to `latest.pt` and
  (on improvement) `best.pt`, prunes to `keep_last_n_checkpoints`, then does a best-effort Drive upload.
  Weights are always saved **without** the `torch.compile` `_orig_mod.` prefix; use `unwrap_model` /
  `strip_compile_prefix` whenever touching state dicts.
- `resume` in config: `''` fresh, `latest`, `best`, `step_N`, or a custom name. A missing checkpoint
  **raises** unless `resume_if_exists: true`; `--resume ''` on the CLI forces a fresh start.
- `generate.py` / `evaluate.py` build the model via `build_model_from_checkpoint`, taking the architecture
  from `ckpt["model_config"]` and the tokenizer path from `ckpt["train_config"]`; `--model` is only a
  fallback for old checkpoints. Loading prefers `torch.load(weights_only=True)`, so anything added to a
  checkpoint must be tensors / plain Python containers (see `_rng_state_for_save`).

### Training loop specifics (`src/train.py`)
- AdamW with two param groups (no weight decay on 1-D params, norms, biases); cosine LR with linear
  warmup from `utils.get_lr`; `precision: auto` picks bf16 on Ampere+, fp16 (+GradScaler) on older CUDA,
  fp32 on CPU; `use_compile: auto` compiles only on CUDA outside Colab.
- Gradient checkpointing must be enabled **before** `torch.compile`.
- OOM handling: micro-batches for a step are held so the whole step can be replayed with a sequence
  cap shrunk by 25% per retry, down to `min_seq_len`.
- `train(..., stop_after_step=N)` saves and returns after step N; tests use it to simulate interruption.

### Tokenizer (`src/tokenizer.py`)
- HuggingFace `tokenizers` BPE with byte fallback. Special tokens `<|bos|> <|eos|> <|pad|>
  <|im_start|> <|im_end|>` (the last two are ChatML markers for later chat fine-tuning). The trained
  `tokenizer_data/tokenizer.json` is committed. Training validates `tokenizer.vocab_size <=
  model_config.vocab_size`.

### Google Drive (`src/gdrive.py`)
Two modes chosen at runtime: on Colab, `gdrive_folder_id` is a folder *name* under `MyDrive` on the
mounted drive; elsewhere it is a real Drive folder *ID* used through the REST API with a service-account
JSON. All Drive calls in the data cache and checkpoint paths are best-effort and wrapped in `try/except`.

### Tests (`tests/`)
`conftest.py` provides `tiny_cfg` (vocab 512, dim 64, 2 layers), `write_shards`/`tmp_data` (random
uint16 shards in `tmp_path`), and a session-scoped tiny tokenizer. Tests are CPU-only and never touch the
network or real Drive (`test_gdrive.py` uses a fake Drive v3 service object). Smoke-run YAMLs live in
`tests/fixtures/` and mirror the real configs at toy scale.
