"""
Data pipeline for pre-training and fine-tuning.
Supports streaming from HuggingFace, tokenization, packing into binary files,
and memory-mapped DataLoader for training.
"""

import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List
from tqdm import tqdm


class TokenizedDataset(Dataset):
    """
    Memory-mapped dataset of packed token sequences.
    Reads from a .bin file (uint16 token IDs) with zero-copy access.
    Each sample is a contiguous chunk of `seq_len` tokens.
    """

    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len

        # Memory-map the binary file
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        self.n_samples = (self.n_tokens - 1) // seq_len  # -1 because targets are shifted by 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for the target shift
        chunk = self.data[start:end].astype(np.int64)

        input_ids = torch.from_numpy(chunk[:-1])   # (seq_len,)
        targets = torch.from_numpy(chunk[1:])       # (seq_len,)
        return input_ids, targets

    def __repr__(self):
        return (f"TokenizedDataset(tokens={self.n_tokens:,}, "
                f"samples={self.n_samples:,}, seq_len={self.seq_len})")


def tokenize_and_save(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_subset: str = "sample-10BT",
    tokenizer_path: str = "tokenizer_data",
    output_dir: str = "data",
    max_tokens: Optional[int] = None,
    split: str = "train",
    shard_size: int = 100_000_000,  # 100M tokens per shard
):
    """
    Stream a HuggingFace dataset, tokenize with the trained tokenizer,
    and save as packed binary files (.bin).

    Each .bin file contains uint16 token IDs, tightly packed.
    Creates train and validation splits (val = 0.5% of data).

    Args:
        dataset_name: HuggingFace dataset name
        dataset_subset: Dataset subset
        tokenizer_path: Path to trained tokenizer
        output_dir: Directory to save binary shards
        max_tokens: Maximum total tokens to process (None = all)
        split: Dataset split to use
        shard_size: Tokens per shard file
    """
    from datasets import load_dataset
    from .tokenizer import LLMTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tok = LLMTokenizer(tokenizer_path)
    print(f"Loaded tokenizer (vocab_size={tok.vocab_size})")

    # Stream dataset
    print(f"Streaming {dataset_name}/{dataset_subset}...")
    ds = load_dataset(dataset_name, dataset_subset, split=split, streaming=True)

    # Tokenize and write to shards
    shard_idx = 0
    token_count = 0
    buffer = np.empty(shard_size, dtype=np.uint16)
    buf_pos = 0
    val_every = 200  # Every 200th document goes to validation

    # Open validation buffer
    val_buffer = []
    val_tokens = 0

    for doc_idx, sample in enumerate(tqdm(ds, desc="Tokenizing")):
        text = sample.get("text", "")
        if not text or len(text) < 50:
            continue

        ids = tok.encode(text, add_bos=True, add_eos=True)

        if max(ids) >= 65535:
            # uint16 overflow safety — skip (shouldn't happen with 32K vocab)
            continue

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
                    # Flush shard
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

        # Memory-map all shards
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
        pin_memory=True,
        drop_last=True,
    )
