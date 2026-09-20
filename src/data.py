"""
Data pipeline for pre-training and fine-tuning.
Supports streaming from HuggingFace, local .txt / .jsonl files,
tokenization, packing into binary files, and memory-mapped DataLoader
for training.

Data sources are configured via configs/data.yaml.
Run ``python -m src.data --config configs/data.yaml`` to build the shards.
"""

import copy
import glob
import hashlib
import itertools
import json
import os
import re
import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import sha256_file


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

def _iter_huggingface(source: dict[str, Any], token: str | None = None) -> Iterator[str]:
    """Yield texts from a HuggingFace streaming dataset.

    ``token`` authenticates the request to the Hub; falls back to the ``HF_TOKEN``
    environment variable when not given. Optional for public datasets, but an
    authenticated request avoids the lower anonymous rate limit.
    """
    from datasets import load_dataset

    name = source["name"]
    subset = source.get("subset")
    split = source.get("split", "train")
    text_field = source.get("text_field", "text")
    token = token or os.environ.get("HF_TOKEN")

    # Streaming mode: no download needed — data is fetched on-the-fly in small chunks.
    # Essential for large datasets (e.g., 10B tokens) that won't fit on disk.
    ds = load_dataset(name, subset, split=split, streaming=True, token=token)
    for sample in ds:
        text = sample.get(text_field, "")
        # Skip very short documents — they add noise without meaningful context
        # and waste tokenizer overhead (BOS/EOS tokens per doc)
        if text and len(text) >= 50:
            yield text


def _iter_text_dir(source: dict[str, Any]) -> Iterator[str]:
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
            with open(fpath, encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
        except OSError as e:
            print(f"  Warning: could not read {fpath}: {e}")
            continue
        if text and len(text) >= 50:
            yield text


def _iter_jsonl(source: dict[str, Any]) -> Iterator[str]:
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
        with open(fpath, encoding="utf-8", errors="replace") as f:
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


def iter_texts_from_sources(sources: list[dict[str, Any]], hf_token: str | None = None) -> Iterator[str]:
    """
    Yield texts from a list of data source dicts (as defined in data.yaml).
    Sources are iterated sequentially in config order. ``hf_token`` is forwarded to
    ``huggingface``-type sources only (see ``_iter_huggingface``).
    """
    for i, src in enumerate(sources):
        src_type = src.get("type", "huggingface")
        iterator_fn = _SOURCE_ITERATORS.get(src_type)
        if iterator_fn is None:
            raise ValueError(f"Unknown source type: {src_type}. "
                             f"Supported: {list(_SOURCE_ITERATORS.keys())}")
        print(f"  Source {i}: {src_type} - {src.get('name') or src.get('path')}")
        if src_type == "huggingface":
            yield from iterator_fn(src, token=hf_token)
        else:
            yield from iterator_fn(src)


def load_data_config(config_path: str = "configs/data.yaml") -> dict[str, Any]:
    """Load and return the data pipeline config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_sources(
    cfg: dict[str, Any],
    *,
    include_huggingface: bool = True,
    base_dir: str | None = None,
) -> list[dict[str, Any]]:
    """The ``sources`` list of a data config, optionally without HuggingFace entries.

    With ``base_dir`` every relative ``text_dir`` / ``jsonl`` path is made absolute
    against it, so a notebook that changes directory (Colab clones the repo into
    ``/content/LLM-Proto``) still points at the right files. The cache fingerprint
    uses paths relative to each source root, so this never changes the cache key.
    """
    sources = copy.deepcopy(cfg.get("sources", []) or [])
    if not include_huggingface:
        sources = [s for s in sources if s.get("type", "huggingface") != "huggingface"]
    if base_dir:
        for src in sources:
            path = src.get("path", "")
            if src.get("type") in ("text_dir", "jsonl") and path and not os.path.isabs(path):
                src["path"] = os.path.join(base_dir, path)
    return sources


def _val_every_from_processing(proc: dict[str, Any], default: int) -> int:
    """Resolve the train/val split interval from a ``processing`` config block.

    Accepts ``val_every: N`` (every N-th document is validation) or the older
    ``val_ratio: r`` (converted to ``round(1 / r)``).
    """
    if proc.get("val_every"):
        return int(proc["val_every"])
    if proc.get("val_ratio"):
        return max(1, int(round(1.0 / float(proc["val_ratio"]))))
    return default


def processing_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve the ``processing`` block of a data config into tokenizer kwargs.

    Returns ``{"output_dir", "max_tokens", "shard_size", "val_every"}`` with the
    same defaults ``tokenize_and_save`` uses.  Shared by the CLI and the notebook
    so both read ``val_every`` / ``shard_size`` the same way.
    """
    proc = cfg.get("processing", {}) or {}
    max_tokens = proc.get("max_tokens")
    return {
        "output_dir": proc.get("output_dir", "data"),
        "max_tokens": int(max_tokens) if max_tokens else None,
        "shard_size": int(proc.get("shard_size", 100_000_000)),
        "val_every": _val_every_from_processing(proc, 200),
    }


# ──────────────────────────────────────────────
# Cache fingerprint & manifest
# ──────────────────────────────────────────────
#
# Tokenized shards are expensive to build, so they are cached locally and on
# Google Drive.  A cache is identified by a *fingerprint*: a hash of everything
# that determines the shard contents (tokenizer, source files, split/shard
# parameters).  ``manifest.json`` next to the shards records the fingerprint and
# the expected size of every file, which is what a cache lookup validates.

FORMAT_VERSION = 1            # Bump when the .bin layout, doc filter, BOS/EOS policy or split rule changes
MANIFEST_NAME = "manifest.json"
GDRIVE_CACHE_SUBDIR = "tokenized"

# Legacy default used when a caller passes no sources at all.
DEFAULT_HF_DATASET = ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train")


def default_sources(
    dataset_name: str = DEFAULT_HF_DATASET[0],
    dataset_subset: str = DEFAULT_HF_DATASET[1],
    split: str = DEFAULT_HF_DATASET[2],
) -> list[dict[str, Any]]:
    """The single HuggingFace source tokenized when no sources are configured.

    Every entry point that accepts an empty source list must substitute this
    *before* computing the cache fingerprint, so the local manifest, the Drive
    cache folder and the tokenization all describe the same data.
    """
    return [{"type": "huggingface", "name": dataset_name, "subset": dataset_subset, "split": split}]
_SHARD_RE = re.compile(r"^train_\d{4}\.bin$")


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _describe_files(base: str, paths: list[str]) -> list[dict[str, Any]]:
    """Content descriptors for *paths*, with names relative to *base* (posix separators)."""
    out = []
    for p in paths:
        rel = os.path.relpath(p, base).replace(os.sep, "/")
        out.append({"path": rel, "bytes": os.path.getsize(p), "sha256": sha256_file(p)})
    return out


def normalize_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reduce a list of source dicts to what actually determines the tokenized output.

    Local sources (``text_dir`` / ``jsonl``) are described by the relative name,
    size and content hash of every file they contain — the absolute path is
    deliberately left out so a corpus cloned into ``/content/LLM-Proto`` on Colab
    and ``C:\\...\\LLM-Proto`` locally produce the same fingerprint.  HuggingFace
    sources are described by their identity only (name/subset/split/text_field).
    Order is preserved because the train/val split depends on document order.
    """
    normalized = []
    for src in sources:
        src_type = src.get("type", "huggingface")
        if src_type == "huggingface":
            normalized.append({
                "type": "huggingface",
                "name": src.get("name"),
                "subset": src.get("subset"),
                "split": src.get("split", "train"),
                "text_field": src.get("text_field", "text"),
            })
        elif src_type == "text_dir":
            path = src.get("path", "")
            if not os.path.isdir(path):
                normalized.append({"type": "text_dir", "missing": True, "files": []})
                continue
            files = []
            for root, _, names in os.walk(path):
                for fname in names:
                    if not fname.startswith("."):
                        files.append(os.path.join(root, fname))
            files.sort()
            normalized.append({"type": "text_dir", "files": _describe_files(path, files)})
        elif src_type == "jsonl":
            path = src.get("path", "")
            text_field = src.get("text_field", "text")
            if os.path.isfile(path):
                base, files = os.path.dirname(path), [path]
            elif os.path.isdir(path):
                base = path
                files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
            else:
                normalized.append({"type": "jsonl", "missing": True, "files": [], "text_field": text_field})
                continue
            normalized.append({"type": "jsonl", "text_field": text_field,
                               "files": _describe_files(base, files)})
        else:
            raise ValueError(f"Unknown source type: {src_type}. "
                             f"Supported: {list(_SOURCE_ITERATORS.keys())}")
    return normalized


def compute_fingerprint(
    tokenizer_path: str,
    sources: list[dict[str, Any]],
    shard_size: int,
    val_every: int,
    max_tokens: int | None,
) -> tuple[str, dict[str, Any]]:
    """
    Return ``(fingerprint, inputs)`` for a tokenization run.

    The fingerprint is a sha256 over the canonical JSON of *inputs*; any change
    to the tokenizer file, the source data or the shard/split parameters yields
    a different value.  Cost is one sequential read of the local corpus, which is
    negligible next to BPE encoding.
    """
    inputs = {
        "format_version": FORMAT_VERSION,
        "tokenizer_sha256": sha256_file(os.path.join(tokenizer_path, "tokenizer.json")),
        "shard_size": int(shard_size),
        "val_every": int(val_every),
        "max_tokens": int(max_tokens) if max_tokens else None,
        "sources": normalize_sources(sources),
    }
    return _canonical_hash(inputs), inputs


def write_manifest(output_dir: str, manifest: dict[str, Any]) -> str:
    path = os.path.join(output_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(data_dir: str) -> dict[str, Any] | None:
    """Parsed ``manifest.json`` from *data_dir*, or None if missing/unreadable."""
    path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def validate_local_cache(data_dir: str, fingerprint: str) -> tuple[bool, str]:
    """Check that *data_dir* holds a complete cache for *fingerprint*.

    Returns ``(True, "ok")`` or ``(False, reason)``.
    """
    manifest = read_manifest(data_dir)
    if manifest is None:
        return False, "no manifest"
    if manifest.get("format_version") != FORMAT_VERSION:
        return False, "format_version mismatch"
    if manifest.get("fingerprint") != fingerprint:
        return False, "fingerprint mismatch"
    for entry in manifest.get("files", []):
        path = os.path.join(data_dir, entry["name"])
        if not os.path.isfile(path):
            return False, f"missing {entry['name']}"
        actual = os.path.getsize(path)
        if actual != entry["bytes"]:
            return False, f"size mismatch {entry['name']}: {actual} != {entry['bytes']}"
    return True, "ok"


def _clear_shards(output_dir: str) -> int:
    """Delete shard files and the manifest from *output_dir*; return the count removed.

    Only ``train_NNNN.bin``, ``val.bin`` and ``manifest.json`` are touched so
    unrelated files (raw corpora, notes) in the same directory survive.
    """
    if not os.path.isdir(output_dir):
        return 0
    removed = 0
    for name in os.listdir(output_dir):
        if _SHARD_RE.match(name) or name in ("val.bin", MANIFEST_NAME):
            os.remove(os.path.join(output_dir, name))
            removed += 1
    return removed


def tokenize_and_save(
    tokenizer_path: str = "tokenizer_data",
    output_dir: str = "data",
    max_tokens: int | None = None,
    shard_size: int = 100_000_000,
    val_every: int = 200,
    sources: list[dict[str, Any]] | None = None,
    config_path: str | None = None,
    # Legacy single-source args (used when sources is None)
    dataset_name: str = DEFAULT_HF_DATASET[0],
    dataset_subset: str = DEFAULT_HF_DATASET[1],
    split: str = DEFAULT_HF_DATASET[2],
    write_manifest_file: bool = True,
    hf_token: str | None = None,
) -> dict[str, Any]:
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

    Any shards left in ``output_dir`` from a previous run are removed first, so a
    smaller corpus can never leave stale higher-numbered shards behind.  A
    ``manifest.json`` describing the result (fingerprint, file sizes, token
    counts) is written last and also returned.
    """
    from .tokenizer import LLMTokenizer

    # Load tokenizer
    tok = LLMTokenizer(tokenizer_path)
    print(f"Loaded tokenizer (vocab_size={tok.vocab_size})")
    if tok.vocab_size >= 65_535:
        raise ValueError(f"Tokenizer vocab_size={tok.vocab_size} does not fit the uint16 shard format")

    # Resolve sources
    if sources is None and config_path is not None:
        cfg = load_data_config(config_path)
        sources = cfg.get("sources", [])
        proc = processing_params(cfg)
        max_tokens = max_tokens or proc["max_tokens"]
        shard_size = proc["shard_size"]
        output_dir = proc["output_dir"]
        val_every = proc["val_every"]
    if not sources:
        # Legacy: single HuggingFace dataset, routed through the regular iterator
        sources = default_sources(dataset_name, dataset_subset, split)

    fingerprint, inputs = compute_fingerprint(tokenizer_path, sources, shard_size, val_every, max_tokens)

    os.makedirs(output_dir, exist_ok=True)
    stale = _clear_shards(output_dir)
    if stale:
        print(f"Removed {stale} stale file(s) from {output_dir}")

    print(f"Tokenizing from {len(sources)} source(s)...")
    text_iter = iter_texts_from_sources(sources, hf_token=hf_token)

    # Tokenize and write to shards
    shard_idx = 0
    token_count = 0
    # Pre-allocate a large numpy buffer and fill it token-by-token.
    # Writing to disk only when a shard is full minimizes I/O operations.
    buffer = np.empty(shard_size, dtype=np.uint16)
    buf_pos = 0

    val_buffer: list = []
    val_tokens = 0
    files: list[dict[str, Any]] = []   # manifest entries, in write order

    def _flush_shard() -> None:
        nonlocal shard_idx, buf_pos
        name = f"train_{shard_idx:04d}.bin"
        shard_path = os.path.join(output_dir, name)
        buffer[:buf_pos].tofile(shard_path)
        print(f"  Saved {shard_path} ({buf_pos:,} tokens)")
        files.append({"name": name, "bytes": os.path.getsize(shard_path), "tokens": buf_pos})
        shard_idx += 1
        buf_pos = 0

    # Documents are tokenized in batches via `encode_batch`, which hands the whole
    # batch to the Rust-backed tokenizer at once so it can encode across CPU cores
    # in parallel — a single-document-at-a-time `encode()` loop never leaves Python,
    # so it can only ever use one core.
    TOKENIZE_BATCH_SIZE = 64
    progress = tqdm(desc="Tokenizing", unit="doc")
    stop = False
    indexed_texts = enumerate(text_iter)
    while not stop:
        batch = list(itertools.islice(indexed_texts, TOKENIZE_BATCH_SIZE))
        if not batch:
            break
        valid = [(doc_idx, text) for doc_idx, text in batch if text and len(text) >= 50]
        encoded = tok.encode_batch([text for _, text in valid], add_bos=True, add_eos=True) if valid else []

        for (doc_idx, _text), ids in zip(valid, encoded, strict=True):
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
                        _flush_shard()

            if max_tokens and token_count >= max_tokens:
                stop = True
                break
        progress.update(len(batch))
    progress.close()

    # Flush remaining train tokens
    if buf_pos > 0:
        _flush_shard()
    n_train_shards = shard_idx

    # Save validation set
    if val_buffer:
        val_path = os.path.join(output_dir, "val.bin")
        np.array(val_buffer, dtype=np.uint16).tofile(val_path)
        print(f"  Saved {val_path} ({val_tokens:,} tokens)")
        files.append({"name": "val.bin", "bytes": os.path.getsize(val_path), "tokens": val_tokens})

    total = token_count + val_tokens
    print(f"Done! Total tokens: {total:,} (train: {token_count:,}, val: {val_tokens:,})")
    print(f"Shards: {n_train_shards} train + {1 if val_buffer else 0} val")

    manifest = {
        "format_version": FORMAT_VERSION,
        "fingerprint": fingerprint,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer": {"vocab_size": tok.vocab_size, "sha256": inputs["tokenizer_sha256"]},
        "params": {"shard_size": int(shard_size), "val_every": int(val_every),
                   "max_tokens": inputs["max_tokens"]},
        "sources": inputs["sources"],
        "files": files,
        "train_tokens": token_count,
        "val_tokens": val_tokens,
        "n_train_shards": n_train_shards,
    }
    if write_manifest_file:
        print(f"  Wrote {write_manifest(output_dir, manifest)}")
    return manifest


# ──────────────────────────────────────────────
# Cache orchestration: local → Google Drive → tokenize
# ──────────────────────────────────────────────

def _remote_cache_folder(fingerprint: str, root_folder_id: str, credentials_path: str,
                         create: bool) -> str | None:
    """Drive handle for ``<root>/tokenized/<fp12>``, or None when absent and not creating."""
    from . import gdrive
    sub = gdrive.resolve_subfolder(root_folder_id, GDRIVE_CACHE_SUBDIR, credentials_path, create=create)
    if sub is None:
        return None
    return gdrive.resolve_subfolder(sub, fingerprint[:12], credentials_path, create=create)


def _download_remote_manifest(folder: str, credentials_path: str) -> dict[str, Any]:
    """Read ``manifest.json`` from the Drive *folder*; ``FileNotFoundError`` if absent or unreadable."""
    from . import gdrive
    with tempfile.TemporaryDirectory() as td:
        gdrive.download_from_gdrive(MANIFEST_NAME, folder, td, credentials_path)
        remote = read_manifest(td)
    if remote is None:
        raise FileNotFoundError("remote manifest is unreadable")
    return remote


def _download_cache_from_gdrive(fingerprint: str, output_dir: str, root_folder_id: str,
                                credentials_path: str) -> dict[str, Any]:
    """Fetch a cached tokenization from Drive into *output_dir*.

    Raises ``FileNotFoundError`` when Drive has no matching cache and
    ``RuntimeError`` when a downloaded file does not match its manifest entry.
    The local manifest is written last, so a partially fetched directory is
    never mistaken for a valid cache.
    """
    folder = _remote_cache_folder(fingerprint, root_folder_id, credentials_path, create=False)
    if folder is None:
        raise FileNotFoundError(f"no '{GDRIVE_CACHE_SUBDIR}/{fingerprint[:12]}' folder on Drive")

    remote = _download_remote_manifest(folder, credentials_path)
    if remote.get("fingerprint") != fingerprint or remote.get("format_version") != FORMAT_VERSION:
        raise FileNotFoundError("remote manifest does not match the requested fingerprint")
    return _download_folder(folder, remote, output_dir, credentials_path)


def _download_folder(folder: str, remote: dict[str, Any], output_dir: str,
                     credentials_path: str) -> dict[str, Any]:
    """Download every file listed in *remote* (the manifest of Drive *folder*) into *output_dir*.

    Existing shards in *output_dir* are cleared first and the manifest is written
    last. ``RuntimeError`` when a downloaded file's size differs from the manifest.
    """
    from . import gdrive
    os.makedirs(output_dir, exist_ok=True)
    _clear_shards(output_dir)
    files = remote.get("files", [])
    total_bytes = 0
    with tqdm(total=sum(entry["bytes"] for entry in files), unit="B", unit_scale=True,
              unit_divisor=1024, desc="[data cache] Downloading from Google Drive") as pbar:
        for entry in files:
            pbar.set_postfix_str(entry["name"])
            path = gdrive.download_from_gdrive(entry["name"], folder, output_dir, credentials_path)
            actual = os.path.getsize(path)
            if actual != entry["bytes"]:
                raise RuntimeError(f"size mismatch after download: {entry['name']} ({actual} != {entry['bytes']})")
            total_bytes += actual
            pbar.update(actual)
    write_manifest(output_dir, remote)
    print(f"[data cache] Downloaded {len(files)} file(s), {total_bytes / 1e6:.1f} MB from Google Drive")
    return remote


def _upload_cache_to_gdrive(manifest: dict[str, Any], output_dir: str, root_folder_id: str,
                            credentials_path: str) -> None:
    """Upload shards then the manifest (last) to ``<root>/tokenized/<fp12>``."""
    from . import gdrive
    folder = _remote_cache_folder(manifest["fingerprint"], root_folder_id, credentials_path, create=True)
    for entry in manifest["files"]:
        gdrive.upload_to_gdrive(os.path.join(output_dir, entry["name"]), folder, credentials_path)
    gdrive.upload_to_gdrive(os.path.join(output_dir, MANIFEST_NAME), folder, credentials_path)
    print(f"[data cache] Uploaded {len(manifest['files'])} file(s) to Google Drive "
          f"({GDRIVE_CACHE_SUBDIR}/{manifest['fingerprint'][:12]})")


def ensure_tokenized_data(
    tokenizer_path: str,
    output_dir: str,
    sources: list[dict[str, Any]],
    max_tokens: int | None = None,
    shard_size: int = 100_000_000,
    val_every: int = 200,
    gdrive_folder_id: str = "",
    gdrive_credentials_path: str = "",
    force: bool = False,
    hf_token: str | None = None,
) -> str:
    """
    Make sure *output_dir* holds tokenized shards for the given inputs.

    Lookup order:
      1. Local cache — ``manifest.json`` in *output_dir* matches the fingerprint
         and every listed file is present with the right size.
      2. Google Drive — ``<gdrive_folder_id>/tokenized/<fingerprint>/`` (only when
         ``gdrive_folder_id`` is set); downloaded into *output_dir*.
      3. Tokenize from scratch, then upload the result to Drive (best effort —
         a failed upload is reported but never aborts the run).

    An empty *sources* list falls back to :func:`default_sources` **here**, before
    the fingerprint is computed: ``tokenize_and_save`` applies the same fallback,
    and fingerprinting the empty list instead would make every cache lookup miss
    while the upload lands in a folder named after the substituted source.

    ``force=True`` skips both caches and re-tokenizes.  ``hf_token`` authenticates
    HuggingFace source requests (falls back to the ``HF_TOKEN`` env var) and is
    deliberately excluded from the fingerprint — it doesn't change the tokenized
    output, only how fast/reliably it's fetched.  Returns *output_dir*.
    """
    if not sources:
        sources = default_sources()
        print(f"[data cache] ! No data sources configured — using the default HuggingFace dataset "
              f"{sources[0]['name']} ({sources[0]['subset']}). Configure sources in data.yaml "
              f"(or enable the HuggingFace ones) to make this explicit.")
    fingerprint, _ = compute_fingerprint(tokenizer_path, sources, shard_size, val_every, max_tokens)
    print(f"[data cache] Fingerprint {fingerprint[:12]} for {len(sources)} source(s)")

    if not force:
        ok, reason = validate_local_cache(output_dir, fingerprint)
        if ok:
            print(f"[data cache] Local cache hit in {output_dir} — skipping tokenization")
            return output_dir
        print(f"[data cache] Local cache miss ({reason})")

        if gdrive_folder_id:
            try:
                _download_cache_from_gdrive(fingerprint, output_dir, gdrive_folder_id, gdrive_credentials_path)
                ok, reason = validate_local_cache(output_dir, fingerprint)
                if ok:
                    print(f"[data cache] Google Drive cache hit — restored into {output_dir}")
                    return output_dir
                print(f"[data cache] Downloaded cache failed validation ({reason})")
            except FileNotFoundError as e:
                print(f"[data cache] Not on Google Drive ({e})")
            except Exception as e:
                print(f"[data cache] ! Google Drive download failed: {e}")
    else:
        print("[data cache] force=True — re-tokenizing")

    print("[data cache] Tokenizing from sources...")
    manifest = tokenize_and_save(
        tokenizer_path=tokenizer_path, output_dir=output_dir, sources=sources,
        max_tokens=max_tokens, shard_size=shard_size, val_every=val_every,
        hf_token=hf_token,
    )
    if manifest["n_train_shards"] == 0:
        raise RuntimeError(
            f"Tokenization produced no training tokens — check the data sources: {sources}"
        )

    if gdrive_folder_id:
        try:
            _upload_cache_to_gdrive(manifest, output_dir, gdrive_folder_id, gdrive_credentials_path)
        except Exception as e:
            print(f"[data cache] ! Google Drive upload failed: {e}")
    return output_dir


def ensure_tokenized_data_from_config(
    cfg_or_path: str | dict[str, Any] = "configs/data.yaml",
    tokenizer_path: str | None = None,
    *,
    include_huggingface: bool = True,
    max_tokens: int | None = None,
    gdrive_folder_id: str | None = None,
    gdrive_credentials_path: str | None = None,
    force: bool = False,
    base_dir: str | None = None,
    hf_token: str | None = None,
) -> str:
    """``ensure_tokenized_data`` driven by a ``data.yaml`` (path or loaded dict).

    ``tokenizer_path`` defaults to ``tokenizer.save_path`` from the config; ``max_tokens``
    overrides ``processing.max_tokens``; ``gdrive_folder_id`` / ``gdrive_credentials_path``
    override the ``cache`` block (pass ``""`` to disable Drive). ``base_dir`` absolutises
    relative source paths and the output dir (see ``resolve_sources``). ``hf_token``
    authenticates HuggingFace source requests (falls back to the ``HF_TOKEN`` env var).
    Returns the shard directory. This is what ``python -m src.data`` and the notebooks call.
    """
    cfg = load_data_config(cfg_or_path) if isinstance(cfg_or_path, str) else cfg_or_path
    tokenizer_path = tokenizer_path or (cfg.get("tokenizer", {}) or {}).get("save_path", "tokenizer_data")
    proc = processing_params(cfg)
    cache = cfg.get("cache", {}) or {}
    folder_id = cache.get("gdrive_folder_id", "") if gdrive_folder_id is None else gdrive_folder_id
    credentials = (cache.get("gdrive_credentials_path", "") if gdrive_credentials_path is None
                   else gdrive_credentials_path)
    output_dir = proc["output_dir"]
    if base_dir and not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)

    return ensure_tokenized_data(
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        sources=resolve_sources(cfg, include_huggingface=include_huggingface, base_dir=base_dir),
        max_tokens=max_tokens if max_tokens is not None else proc["max_tokens"],
        shard_size=proc["shard_size"],
        val_every=proc["val_every"],
        gdrive_folder_id=folder_id or "",
        gdrive_credentials_path=credentials or "",
        force=force,
        hf_token=hf_token,
    )


def dataset_output_dir(base_dir: str, gdrive_folder: str) -> str:
    """Local directory for a pre-tokenized Drive folder: ``<base_dir>/<last folder name>``.

    Each dataset gets its own directory, so switching datasets never clears the shards
    of another one (``tokenize_and_save`` and the Drive download both wipe their target).
    """
    name = gdrive_folder.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not name:
        raise ValueError(f"cannot derive a dataset name from Drive folder {gdrive_folder!r}")
    return os.path.join(base_dir, name)


def fetch_prebuilt_dataset(
    tokenizer_path: str,
    output_dir: str,
    gdrive_folder: str,
    gdrive_credentials_path: str = "",
    force: bool = False,
) -> str:
    """
    Use an already-tokenized dataset from a Google Drive folder, exactly as it is.

    Unlike :func:`ensure_tokenized_data` this never computes an expected fingerprint and
    never tokenizes: the folder's own ``manifest.json`` says what it holds. *gdrive_folder*
    must contain ``train_NNNN.bin`` / ``val.bin`` / ``manifest.json`` directly (Colab: a
    folder path under MyDrive such as ``"LLM/gutenberg"``; elsewhere: the Drive folder ID).

    The only compatibility check is the tokenizer: the manifest's tokenizer sha256 must match
    ``<tokenizer_path>/tokenizer.json``, otherwise the token ids in the shards are meaningless
    (``ValueError``). A missing manifest raises ``FileNotFoundError``. If *output_dir* already
    holds an identical, complete copy (same fingerprint, all file sizes match) nothing is
    downloaded, unless ``force=True``. Returns *output_dir*.
    """
    if not gdrive_folder:
        raise ValueError("gdrive_folder is required")
    try:
        remote = _download_remote_manifest(gdrive_folder, gdrive_credentials_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Drive folder '{gdrive_folder}' has no usable {MANIFEST_NAME} ({e}). A pre-tokenized "
            f"dataset folder must hold train_NNNN.bin, val.bin and {MANIFEST_NAME} directly."
        ) from e

    if remote.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Drive folder '{gdrive_folder}' has manifest format_version "
                         f"{remote.get('format_version')!r}, expected {FORMAT_VERSION}")
    remote_sha = (remote.get("tokenizer") or {}).get("sha256")
    local_sha = sha256_file(os.path.join(tokenizer_path, "tokenizer.json"))
    if remote_sha != local_sha:
        raise ValueError(
            f"Drive folder '{gdrive_folder}' was tokenized with a different tokenizer "
            f"(sha256 {str(remote_sha)[:12]} != {local_sha[:12]} of {tokenizer_path}/tokenizer.json); "
            f"its token ids would be meaningless to this model."
        )

    fingerprint = remote.get("fingerprint", "")
    if not force:
        ok, reason = validate_local_cache(output_dir, fingerprint)
        if ok:
            print(f"[data cache] {gdrive_folder} already in {output_dir} — skipping download")
            return output_dir
        print(f"[data cache] Local copy in {output_dir} not usable ({reason})")

    print(f"[data cache] Using pre-tokenized dataset from Drive folder '{gdrive_folder}' "
          f"({remote.get('train_tokens', 0):,} train tokens)")
    _download_folder(gdrive_folder, remote, output_dir, gdrive_credentials_path)
    ok, reason = validate_local_cache(output_dir, fingerprint)
    if not ok:
        raise RuntimeError(f"Downloaded dataset from '{gdrive_folder}' failed validation ({reason})")
    return output_dir


def find_train_shards(data_dir: str) -> list[str]:
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
        shard_files: list[str],
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

    def _worker_layout(self) -> tuple[int, int, list[tuple[str, int]], int, int]:
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
        shards = list(zip(shard_files, offsets, strict=True))

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
        self, worker_id: int, my_shards: list[tuple[str, int]], sample_start: int, sample_step: int,
    ) -> Iterator[tuple[str, int]]:
        """Yield ``(shard_path, token_start)`` pairs in final (shuffled) order. No I/O."""
        # Per-worker shuffle-buffer RNG — different seed per worker per epoch
        # so workers don't produce the same random order.
        buf_rng = np.random.default_rng(self.seed + self._epoch * 10_000 + worker_id)
        buf: list[tuple[str, int]] = []
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

        mmaps: dict[str, np.memmap] = {}
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
    parser.add_argument("--gdrive_folder_id", type=str, default=None,
                        help="Override cache.gdrive_folder_id (Colab: folder name; local: Drive folder ID)")
    parser.add_argument("--gdrive_credentials", type=str, default=None,
                        help="Override cache.gdrive_credentials_path (service-account JSON)")
    parser.add_argument("--no_gdrive", action="store_true", help="Skip the Google Drive cache")
    parser.add_argument("--force", action="store_true", help="Re-tokenize even if a valid cache exists")
    parser.add_argument("--prebuilt_folder", type=str, default=None,
                        help="Use an already-tokenized Drive folder (shards + manifest.json) as-is instead of "
                             "tokenizing; never tokenizes. Needs --output_dir")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Local directory to restore --prebuilt_folder into")
    args = parser.parse_args()

    if args.prebuilt_folder:
        if not args.output_dir:
            parser.error("--prebuilt_folder requires --output_dir")
        cfg = load_data_config(args.config)
        tokenizer_path = args.tokenizer_path or (cfg.get("tokenizer", {}) or {}).get("save_path", "tokenizer_data")
        credentials = (args.gdrive_credentials if args.gdrive_credentials is not None
                       else (cfg.get("cache", {}) or {}).get("gdrive_credentials_path", ""))
        fetch_prebuilt_dataset(tokenizer_path, args.output_dir, args.prebuilt_folder,
                               credentials or "", force=args.force)
        return

    ensure_tokenized_data_from_config(
        args.config,
        args.tokenizer_path,
        max_tokens=args.max_tokens,
        gdrive_folder_id="" if args.no_gdrive else args.gdrive_folder_id,
        gdrive_credentials_path=args.gdrive_credentials,
        force=args.force,
    )


if __name__ == "__main__":
    main()
