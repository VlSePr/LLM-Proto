"""
Data pipeline for pre-training and fine-tuning.
Supports streaming from HuggingFace, local .txt / .jsonl files,
tokenization, packing into binary files, and memory-mapped DataLoader
for training.

Data sources are configured via configs/data.yaml.
Run ``python -m src.data --config configs/data.yaml`` to build the shards.
"""

import glob
import itertools
import json
import os
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Iterator, Dict, Any, Tuple
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
    """Yield texts from files in a directory (recursive).

    Reads all non-hidden files regardless of extension, so extension-less
    books (e.g. ``alice-in-wonderland``) are included alongside ``*.txt``.
    """
    path = source["path"]
    if not os.path.isdir(path):
        abs_path = os.path.abspath(path)
        print(f"  Warning: text_dir path does not exist: {path}")
        print(f"    (resolved to: {abs_path}, cwd: {os.getcwd()})")
        return

    all_files = []
    for root, _, files in os.walk(path):
        for fname in sorted(files):
            if not fname.startswith("."):
                all_files.append(os.path.join(root, fname))
    all_files.sort()
    print(f"  Found {len(all_files)} files in {path}")

    for fpath in all_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
        except (IOError, OSError) as e:
            print(f"  Warning: could not read {fpath}: {e}")
            continue
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
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  Warning: skipping malformed JSON at {fpath}:{line_no}: {e}")
                    continue
                text = obj.get(text_field, "") if isinstance(obj, dict) else ""
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
        print(f"  Source {i}: {src_type} - {src.get('name') or src.get('path')}")
        yield from iterator_fn(src)


def load_data_config(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    """Load and return the data pipeline config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _val_every_from_processing(proc: Dict[str, Any], default: int) -> int:
    """Resolve the train/val split interval from a ``processing`` config block.

    Accepts ``val_every: N`` (every N-th document is validation) or the older
    ``val_ratio: r`` (converted to ``round(1 / r)``).
    """
    if proc.get("val_every"):
        return int(proc["val_every"])
    if proc.get("val_ratio"):
        return max(1, int(round(1.0 / float(proc["val_ratio"]))))
    return default


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
    if tok.vocab_size >= 65_535:
        raise ValueError(f"Tokenizer vocab_size={tok.vocab_size} does not fit the uint16 shard format")

    # Resolve sources
    if sources is None and config_path is not None:
        cfg = load_data_config(config_path)
        sources = cfg.get("sources", [])
        proc = cfg.get("processing", {}) or {}
        max_tokens = max_tokens or proc.get("max_tokens")
        shard_size = int(proc.get("shard_size", shard_size))
        output_dir = proc.get("output_dir", output_dir)
        val_every = _val_every_from_processing(proc, val_every)
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

        # Interleaved train/val split: every val_every-th document → validation.
        # Using document index ensures deterministic splitting regardless of text content.
        is_val = (doc_idx % val_every == 0)

        if is_val:
            val_buffer.extend(ids)
            val_tokens += len(ids)
        else:
            ids_arr = np.array(ids, dtype=np.uint16)
            # Clip to remaining token budget before writing
            if max_tokens:
                ids_arr = ids_arr[: max(0, max_tokens - token_count)]
            # Vectorised fill: write token blocks into the buffer, flushing
            # complete shards to disk as each one fills up.  This replaces a
            # Python for-loop over individual tokens, which is ~50–100× slower
            # for long documents (numpy slice assignment is a single C memcpy).
            pos = 0
            while pos < len(ids_arr):
                space = shard_size - buf_pos
                take = min(len(ids_arr) - pos, space)
                buffer[buf_pos : buf_pos + take] = ids_arr[pos : pos + take]
                buf_pos += take
                token_count += take
                pos += take
                if buf_pos >= shard_size:
                    shard_path = os.path.join(output_dir, f"train_{shard_idx:04d}.bin")
                    buffer[:buf_pos].tofile(shard_path)
                    print(f"  Saved {shard_path} ({buf_pos:,} tokens)")
                    shard_idx += 1
                    buf_pos = 0

        if max_tokens and token_count >= max_tokens:
            break

    # Flush remaining train tokens
    n_train_shards = shard_idx
    if buf_pos > 0:
        shard_path = os.path.join(output_dir, f"train_{shard_idx:04d}.bin")
        buffer[:buf_pos].tofile(shard_path)
        print(f"  Saved {shard_path} ({buf_pos:,} tokens)")
        n_train_shards += 1

    # Save validation set
    if val_buffer:
        val_path = os.path.join(output_dir, "val.bin")
        np.array(val_buffer, dtype=np.uint16).tofile(val_path)
        print(f"  Saved {val_path} ({val_tokens:,} tokens)")

    total = token_count + val_tokens
    print(f"Done! Total tokens: {total:,} (train: {token_count:,}, val: {val_tokens:,})")
    print(f"Shards: {n_train_shards} train + {1 if val_buffer else 0} val")


def find_train_shards(data_dir: str) -> List[str]:
    """Sorted list of ``train_NNNN.bin`` shard paths in ``data_dir``."""
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("train_") and f.endswith(".bin")
    )


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
            shard_files = find_train_shards(data_dir)

        shard_files = [f for f in shard_files if os.path.exists(f)]

        if not shard_files:
            raise FileNotFoundError(f"No {split} shards found in {data_dir}")

        # Memory-map all shards at once — the OS handles page faults on demand,
        # so only the data actually read during training touches RAM.
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in shard_files]
        self.shard_lengths = [len(s) for s in self.shards]
        self.total_tokens = sum(self.shard_lengths)

        # Build index using compact numpy arrays instead of a Python list of tuples.
        # At 1B tokens / seq_len=1024 ≈ 1M samples:
        #   list-of-tuples  ≈ 56 MB  (56 bytes per CPython tuple of two ints)
        #   numpy int32 pair ≈  8 MB  (4+4 bytes per entry)  → 7× smaller
        # numpy also gives faster random-index lookup (C array vs. pointer-chased heap).
        shard_idx_list = []
        offset_list = []
        self.n_samples = 0
        for si, shard in enumerate(self.shards):
            n = (len(shard) - 1) // seq_len
            if n == 0:
                continue
            shard_idx_list.append(np.full(n, si, dtype=np.int32))
            offset_list.append(np.arange(n, dtype=np.int32) * seq_len)
            self.n_samples += n
        if self.n_samples > 0:
            self._shard_idx = np.concatenate(shard_idx_list)  # (N,) int32
            self._offsets   = np.concatenate(offset_list)     # (N,) int32
        else:
            self._shard_idx = np.empty(0, dtype=np.int32)
            self._offsets   = np.empty(0, dtype=np.int32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        shard_idx = int(self._shard_idx[idx])
        offset    = int(self._offsets[idx])
        chunk     = self.shards[shard_idx][offset : offset + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


class IterableShardDataset(torch.utils.data.IterableDataset):
    """
    Streaming iterable dataset for multi-shard binary token files.

    Designed for training on billions of tokens without ever materialising a
    per-sample index in RAM.  Key differences from ``ShardedTokenDataset``:

    * **Zero index memory** — no ``(shard_idx, offset)`` table is built.
      Memory footprint is a fixed constant regardless of dataset size.
    * **Shard-level shuffle** — shard order is randomised each epoch via an
      epoch-seeded RNG, ensuring the model sees data in a different global
      order every epoch.  Call :meth:`set_epoch` before each epoch.
    * **Random window offset** — each shard gets a per-epoch offset in
      ``[0, seq_len)`` so sequence boundaries move between epochs instead of
      always being aligned at ``i * seq_len``.
    * **Sequential shard reads** — each shard is consumed end-to-end, which
      is optimal for memory-mapped files (sequential page-faults < random seeks).
    * **Shuffle buffer** — a small fixed-size reservoir provides fine-grained
      sample randomness without requiring a full global permutation.
      Effective shuffle window ≈ ``shuffle_buffer_size × seq_len`` tokens.
    * **Worker-aware** — when ``num_workers > 0``, each DataLoader worker
      receives a disjoint slice of the shard list.  When there are fewer shards
      than workers, every worker reads every shard but takes a strided subset
      of samples, so no worker sits idle.
    * **Resumable** — set :attr:`skip_batches` (global batches already consumed
      in this epoch) before iterating to continue exactly where a previous run
      stopped.  Skipping is done on the index stream, so it costs no I/O.

    The sample order is a deterministic function of ``(seed, epoch, worker
    layout)``, which is what makes exact resume possible.
    """

    def __init__(
        self,
        shard_files: List[str],
        seq_len: int,
        shuffle_shards: bool = True,
        shuffle_buffer_size: int = 1_000,
        seed: int = 42,
        random_offset: bool = True,
    ):
        self.shard_files = list(shard_files)
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.random_offset = random_offset
        self._epoch = 0

        # Resume support (set by the trainer / create_dataloader).
        self.skip_batches: int = 0   # global batches already consumed in the current epoch
        self.batch_size: int = 1     # needed to convert skip_batches into samples per worker

        # Estimate total tokens / samples without loading any data.
        # os.path.getsize is a fast syscall; uint16 = 2 bytes per token.
        self.total_tokens = sum(os.path.getsize(f) // 2 for f in shard_files)
        self.n_samples_approx = max(0, (self.total_tokens - 1) // seq_len)

    def __len__(self) -> int:
        return self.n_samples_approx

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch counter to change shard shuffle order.

        Call before each training epoch so the model sees shards in a
        different order every time::

            for epoch in range(n_epochs):
                dataset.set_epoch(epoch)
                for batch in loader:
                    ...
        """
        self._epoch = int(epoch)

    # ── internal helpers ─────────────────────────────────────────────

    def _worker_layout(self) -> Tuple[int, int, List[Tuple[str, int]], int, int]:
        """Decide which (shard, offset) pairs and which sample stride this worker handles."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # Shard-level shuffle: deterministic but different each epoch.
        rng = np.random.default_rng(self.seed + self._epoch)
        shard_files = list(self.shard_files)
        if self.shuffle_shards:
            rng.shuffle(shard_files)
        # Per-shard window offsets, drawn from the same epoch RNG so every worker agrees.
        if self.random_offset:
            offsets = [int(rng.integers(self.seq_len)) for _ in shard_files]
        else:
            offsets = [0] * len(shard_files)
        shards = list(zip(shard_files, offsets))

        if len(shards) >= num_workers:
            # Worker-aware shard splitting: worker i reads shards i, i+W, i+2W, …
            # This gives disjoint, full coverage with no inter-worker coordination.
            my_shards = shards[worker_id::num_workers]
            sample_start, sample_step = 0, 1
        else:
            # Fewer shards than workers: every worker reads every shard but takes a
            # strided subset of samples, so all workers stay busy and coverage is exact.
            my_shards = shards
            sample_start, sample_step = worker_id, num_workers
        return worker_id, num_workers, my_shards, sample_start, sample_step

    def _index_stream(
        self, worker_id: int, my_shards: List[Tuple[str, int]], sample_start: int, sample_step: int,
    ) -> Iterator[Tuple[str, int]]:
        """Yield ``(shard_path, token_start)`` pairs in final (shuffled) order. No I/O."""
        # Per-worker shuffle-buffer RNG — different seed per worker per epoch
        # so workers don't produce the same random order.
        buf_rng = np.random.default_rng(self.seed + self._epoch * 10_000 + worker_id)
        buf: List[Tuple[str, int]] = []
        buf_size = self.shuffle_buffer_size

        for shard_path, offset in my_shards:
            n_tokens = os.path.getsize(shard_path) // 2
            n_samples = (n_tokens - 1 - offset) // self.seq_len
            for i in range(sample_start, n_samples, sample_step):
                item = (shard_path, offset + i * self.seq_len)
                if buf_size <= 1:
                    # No shuffle buffer: yield directly (deterministic order)
                    yield item
                    continue
                buf.append(item)
                if len(buf) >= buf_size:
                    # Reservoir emit: swap a random slot with the last element
                    # and pop — O(1), no list reallocation.
                    idx = int(buf_rng.integers(len(buf)))
                    yield buf[idx]
                    buf[idx] = buf[-1]
                    buf.pop()

        # Flush remaining buffer in a random order
        if buf:
            for idx in buf_rng.permutation(len(buf)):
                yield buf[int(idx)]

    def _worker_skip_samples(self, worker_id: int, num_workers: int) -> int:
        """Samples this worker must skip so the loader resumes after ``skip_batches``."""
        skip_batches = int(self.skip_batches or 0)
        if skip_batches <= 0:
            return 0
        if num_workers <= 1:
            return skip_batches * self.batch_size
        # The DataLoader takes batches from workers round-robin, so worker w produced
        # batches w, w+W, w+2W, … — count how many of those are below skip_batches.
        if skip_batches <= worker_id:
            return 0
        my_batches = (skip_batches - worker_id + num_workers - 1) // num_workers
        return my_batches * self.batch_size

    def __iter__(self):
        worker_id, num_workers, my_shards, sample_start, sample_step = self._worker_layout()
        stream = self._index_stream(worker_id, my_shards, sample_start, sample_step)

        skip = self._worker_skip_samples(worker_id, num_workers)
        if skip > 0:
            stream = itertools.islice(stream, skip, None)

        mmaps: Dict[str, np.memmap] = {}
        seq_len = self.seq_len
        for shard_path, start in stream:
            data = mmaps.get(shard_path)
            if data is None:
                data = mmaps[shard_path] = np.memmap(shard_path, dtype=np.uint16, mode="r")
            # Read seq_len+1 tokens; the extra token is the next-token target
            # for position seq_len-1.  The .astype copy detaches from the mmap.
            chunk = data[start : start + seq_len + 1].astype(np.int64)
            yield torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def create_dataloader(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    split: str = "train",
    num_workers: int = 4,
    shuffle: bool = True,
    shuffle_buffer_size: int = 1_000,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader from tokenized binary shards.

    For the *train* split this returns a streaming :class:`IterableShardDataset`
    that shuffles at shard granularity and uses a shuffle buffer for fine-grained
    randomness — no global index, so memory cost is constant regardless of how
    many tokens are on disk.  The map-style :class:`ShardedTokenDataset` is kept
    for the *val* split (deterministic, single-file).

    Args:
        data_dir: Directory containing ``train_NNNN.bin`` and ``val.bin`` shards.
        seq_len: Sequence length (tokens per sample).
        batch_size: Samples per batch.
        split: ``"train"`` or ``"val"``.
        num_workers: DataLoader worker processes.  Use 0 for notebooks / debugging.
            For production training 2–4 workers are recommended.
        shuffle: Shuffle shard order each epoch and apply the shuffle buffer
            (train only; always ``False`` for val).
        shuffle_buffer_size: Samples held in the shuffle buffer (only their
            indices are buffered, so memory cost is negligible).
        seed: Base RNG seed for reproducible shuffling.
    """
    if split == "val":
        dataset = ShardedTokenDataset(data_dir, seq_len, split="val")
        print(f"Created val DataLoader: {dataset.n_samples:,} samples, "
              f"{dataset.total_tokens:,} tokens")
        _pin = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=_pin,           # page-lock host RAM only when a GPU is present
            pin_memory_device="cuda" if _pin else "",
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    # ── Train split: streaming IterableShardDataset ──
    shard_files = [f for f in find_train_shards(data_dir) if os.path.exists(f)]
    if not shard_files:
        raise FileNotFoundError(f"No train shards found in {data_dir}")

    dataset = IterableShardDataset(
        shard_files,
        seq_len,
        shuffle_shards=shuffle,
        shuffle_buffer_size=shuffle_buffer_size if shuffle else 0,
        seed=seed,
        random_offset=shuffle,
    )
    dataset.batch_size = batch_size
    print(
        f"Created train DataLoader (streaming): "
        f"~{dataset.n_samples_approx:,} samples, "
        f"{dataset.total_tokens:,} tokens across {len(shard_files)} shard(s)"
    )
    _pin = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=_pin,           # page-lock host RAM only when a GPU is present
        pin_memory_device="cuda" if _pin else "",
        drop_last=True,
        # No persistent_workers for the train iterable: the trainer calls
        # iter(loader) again at each epoch boundary after set_epoch(), so fresh
        # workers pick up the new epoch seed and skip position.
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize data sources into binary shards")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Data config YAML")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Tokenizer directory (default: tokenizer.save_path from the config)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Override processing.max_tokens")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    tokenizer_path = args.tokenizer_path or (cfg.get("tokenizer", {}) or {}).get("save_path", "tokenizer_data")
    tokenize_and_save(
        tokenizer_path=tokenizer_path,
        max_tokens=args.max_tokens,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
