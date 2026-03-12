"""
Data pipeline for pre-training and fine-tuning.
Supports streaming from HuggingFace, local .txt / .jsonl files,
tokenization, packing into binary files, and memory-mapped DataLoader
for training.

Data sources are configured via configs/data.yaml.
"""

import glob
import json
import os
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Iterator, Dict, Any
from tqdm import tqdm


class TokenizedDataset(Dataset):
    """
    Memory-mapped dataset of packed token sequences.
    Reads from a .bin file (uint16 token IDs) with zero-copy access.
    Each sample is a contiguous chunk of ``seq_len`` tokens.
    """

    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len

        # Memory-map: the OS maps the file into virtual memory without loading it all into RAM.
        # This enables datasets larger than available RAM — only accessed pages are loaded.
        # uint16 supports vocab sizes up to 65535, which is enough for typical BPE tokenizers
        # (32K-64K vocab) while using 2 bytes/token instead of 4 (int32) — halving disk and memory usage.
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        # -1 because the last token of each sequence is used as the target for the
        # previous token (next-token prediction: input[i] predicts target[i] = input[i+1])
        self.n_samples = (self.n_tokens - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 to have the next-token target for position seq_len-1
        # Cast to int64 for PyTorch — nn.Embedding requires LongTensor (int64), not uint16
        chunk = self.data[start:end].astype(np.int64)

        # Standard next-token prediction setup:
        # input_ids  = [t0, t1, t2, ..., t_{n-1}]
        # targets    = [t1, t2, t3, ..., t_n]
        input_ids = torch.from_numpy(chunk[:-1])   # (seq_len,)
        targets = torch.from_numpy(chunk[1:])       # (seq_len,)
        return input_ids, targets

    def __repr__(self):
        return (f"TokenizedDataset(tokens={self.n_tokens:,}, "
                f"samples={self.n_samples:,}, seq_len={self.seq_len})")


# ──────────────────────────────────────────────
# Multi-source text iterators
# ──────────────────────────────────────────────

def _iter_huggingface(source: Dict[str, Any]) -> Iterator[str]:
    """Yield texts from a HuggingFace streaming dataset."""
    from datasets import load_dataset

    name = source["name"]
    subset = source.get("subset")
    split = source.get("split", "train")
    text_field = source.get("text_field", "text")

    # Streaming mode: no download needed — data is fetched on-the-fly in small chunks.
    # Essential for large datasets (e.g., 10B tokens) that won't fit on disk.
    ds = load_dataset(name, subset, split=split, streaming=True)
    for sample in ds:
        text = sample.get(text_field, "")
        # Skip very short documents — they add noise without meaningful context
        # and waste tokenizer overhead (BOS/EOS tokens per doc)
        if text and len(text) >= 50:
            yield text


def _iter_text_dir(source: Dict[str, Any]) -> Iterator[str]:
    """Yield texts from .txt files in a directory (recursive)."""
    path = source["path"]
    if not os.path.isdir(path):
        print(f"  Warning: text_dir path does not exist: {path}")
        return

    txt_files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
    print(f"  Found {len(txt_files)} .txt files in {path}")

    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if text and len(text) >= 50:
            yield text


def _iter_jsonl(source: Dict[str, Any]) -> Iterator[str]:
    """Yield texts from .jsonl files (single file or directory)."""
    path = source["path"]
    text_field = source.get("text_field", "text")

    if os.path.isfile(path):
        jsonl_files = [path]
    elif os.path.isdir(path):
        jsonl_files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
    else:
        print(f"  Warning: jsonl path does not exist: {path}")
        return

    print(f"  Found {len(jsonl_files)} .jsonl files")

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_field, "")
                if text and len(text) >= 50:
                    yield text


_SOURCE_ITERATORS = {
    "huggingface": _iter_huggingface,
    "text_dir": _iter_text_dir,
    "jsonl": _iter_jsonl,
}


def iter_texts_from_sources(sources: List[Dict[str, Any]]) -> Iterator[str]:
    """
    Yield texts from a list of data source dicts (as defined in data.yaml).
    Sources are iterated sequentially in config order.
    """
    for i, src in enumerate(sources):
        src_type = src.get("type", "huggingface")
        iterator_fn = _SOURCE_ITERATORS.get(src_type)
        if iterator_fn is None:
            raise ValueError(f"Unknown source type: {src_type}. "
                             f"Supported: {list(_SOURCE_ITERATORS.keys())}")
        print(f"  Source {i}: {src_type} — {src.get('name') or src.get('path')}")
        yield from iterator_fn(src)


def load_data_config(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    """Load and return the data pipeline config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def tokenize_and_save(
    tokenizer_path: str = "tokenizer_data",
    output_dir: str = "data",
    max_tokens: Optional[int] = None,
    shard_size: int = 100_000_000,
    val_every: int = 200,
    sources: Optional[List[Dict[str, Any]]] = None,
    config_path: Optional[str] = None,
    # Legacy single-source args (used when sources is None)
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_subset: str = "sample-10BT",
    split: str = "train",
):
    """
    Tokenize text from one or more sources and save as packed binary shards.

    Data sources can be specified in three ways (highest priority first):
      1. ``sources`` — list of source dicts directly
      2. ``config_path`` — path to a data.yaml config file
      3. Legacy args (dataset_name/subset/split) — single HuggingFace dataset

    Each .bin file contains uint16 token IDs, tightly packed (no padding or separators
    between documents). This "packing" approach maximizes GPU utilization — every token
    in a batch contributes to the loss, unlike padded batches where pad tokens are wasted.

    Sharding into ~100M-token files keeps individual files manageable and enables
    parallel loading from multiple workers.

    Train/val split: every ``val_every``-th document goes to validation.
    This interleaved split ensures the val set is representative of the full data
    distribution, rather than being biased by document ordering.
    """
    from .tokenizer import LLMTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tok = LLMTokenizer(tokenizer_path)
    print(f"Loaded tokenizer (vocab_size={tok.vocab_size})")

    # Resolve sources
    if sources is None and config_path is not None:
        cfg = load_data_config(config_path)
        sources = cfg.get("sources", [])
        proc = cfg.get("processing", {})
        max_tokens = max_tokens or proc.get("max_tokens")
        shard_size = proc.get("shard_size", shard_size)
        output_dir = proc.get("output_dir", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    if sources:
        print(f"Tokenizing from {len(sources)} source(s)...")
        text_iter = iter_texts_from_sources(sources)
    else:
        # Legacy: single HuggingFace dataset
        from datasets import load_dataset
        print(f"Streaming {dataset_name}/{dataset_subset}...")
        ds = load_dataset(dataset_name, dataset_subset, split=split, streaming=True)
        text_iter = (s.get("text", "") for s in ds)

    # Tokenize and write to shards
    shard_idx = 0
    token_count = 0
    # Pre-allocate a large numpy buffer and fill it token-by-token.
    # Writing to disk only when a shard is full minimizes I/O operations.
    buffer = np.empty(shard_size, dtype=np.uint16)
    buf_pos = 0

    val_buffer: list = []
    val_tokens = 0

    for doc_idx, text in enumerate(tqdm(text_iter, desc="Tokenizing")):
        if not text or len(text) < 50:
            continue

        ids = tok.encode(text, add_bos=True, add_eos=True)

        # Skip tokens that overflow uint16 (token ID >= 65535).
        # This is a safety check for tokenizers with very large vocabularies.
        if max(ids) >= 65535:
            continue

        # Interleaved train/val split: every val_every-th document → validation.
        # Using document index ensures deterministic splitting regardless of text content.
        is_val = (doc_idx % val_every == 0)

        if is_val:
            val_buffer.extend(ids)
            val_tokens += len(ids)
        else:
            for token_id in ids:
                buffer[buf_pos] = token_id
                buf_pos += 1
                token_count += 1

                if buf_pos >= shard_size:
                    shard_path = os.path.join(output_dir, f"train_{shard_idx:04d}.bin")
                    buffer[:buf_pos].tofile(shard_path)
                    print(f"  Saved {shard_path} ({buf_pos:,} tokens)")
                    shard_idx += 1
                    buf_pos = 0

                if max_tokens and token_count >= max_tokens:
                    break

        if max_tokens and token_count >= max_tokens:
            break

    # Flush remaining train tokens
    if buf_pos > 0:
        shard_path = os.path.join(output_dir, f"train_{shard_idx:04d}.bin")
        buffer[:buf_pos].tofile(shard_path)
        print(f"  Saved {shard_path} ({buf_pos:,} tokens)")

    # Save validation set
    if val_buffer:
        val_path = os.path.join(output_dir, "val.bin")
        np.array(val_buffer, dtype=np.uint16).tofile(val_path)
        print(f"  Saved {val_path} ({val_tokens:,} tokens)")

    total = token_count + val_tokens
    print(f"Done! Total tokens: {total:,} (train: {token_count:,}, val: {val_tokens:,})")
    print(f"Shards: {shard_idx + 1} train + 1 val")


class ShardedTokenDataset(Dataset):
    """
    Dataset that reads from multiple binary shards.
    Useful when data is split across multiple .bin files.

    All shards are memory-mapped at init (cheap — no data loaded until accessed).
    An index maps each sample to its (shard, offset) pair for O(1) random access.
    """

    def __init__(self, data_dir: str, seq_len: int, split: str = "train"):
        self.seq_len = seq_len

        # Find all shards for this split
        if split == "val":
            shard_files = [os.path.join(data_dir, "val.bin")]
        else:
            shard_files = sorted([
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.startswith("train_") and f.endswith(".bin")
            ])

        shard_files = [f for f in shard_files if os.path.exists(f)]

        if not shard_files:
            raise FileNotFoundError(f"No {split} shards found in {data_dir}")

        # Memory-map all shards at once — the OS handles page faults on demand,
        # so only the data actually read during training touches RAM.
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in shard_files]
        self.shard_lengths = [len(s) for s in self.shards]
        self.total_tokens = sum(self.shard_lengths)

        # Build index: for each sample idx, which shard and offset?
        self.n_samples = 0
        self.shard_offsets = []  # (shard_idx, start_offset) for each sample
        for si, shard in enumerate(self.shards):
            n = (len(shard) - 1) // seq_len
            for j in range(n):
                self.shard_offsets.append((si, j * seq_len))
            self.n_samples += n

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        shard_idx, offset = self.shard_offsets[idx]
        chunk = self.shards[shard_idx][offset:offset + self.seq_len + 1].astype(np.int64)

        input_ids = torch.from_numpy(chunk[:-1])
        targets = torch.from_numpy(chunk[1:])
        return input_ids, targets


def create_dataloader(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    split: str = "train",
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from tokenized binary shards."""
    dataset = ShardedTokenDataset(data_dir, seq_len, split)
    print(f"Created {split} DataLoader: {dataset.n_samples:,} samples, "
          f"{dataset.total_tokens:,} tokens")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=True,   # Pre-copy data to CUDA-pinned RAM for faster GPU transfers
        drop_last=True,    # Drop incomplete last batch to keep batch size consistent
    )
