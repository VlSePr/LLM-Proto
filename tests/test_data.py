import os
from types import SimpleNamespace

import numpy as np
import torch

from src.data import (
    IterableShardDataset,
    _val_every_from_processing,
    create_dataloader,
    find_train_shards,
    tokenize_and_save,
)
from tests.conftest import TINY_VOCAB, write_shards

SEQ = 16


def _samples(ds):
    return [(a.clone(), b.clone()) for a, b in ds]


def _starts(ds):
    """(shard, start) index stream exactly as __iter__ consumes it (no I/O)."""
    wid, nw, shards, s0, step = ds._worker_layout()
    return list(ds._index_stream(wid, shards, s0, step))


def _with_worker(monkeypatch, worker_id, num_workers):
    info = SimpleNamespace(id=worker_id, num_workers=num_workers)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)


def test_every_sample_is_a_valid_window(tmp_data):
    ds = IterableShardDataset(find_train_shards(tmp_data), SEQ, shuffle_buffer_size=8)
    shards = {p: np.fromfile(p, dtype=np.uint16) for p in ds.shard_files}
    n = 0
    for path, start in _starts(ds):
        chunk = shards[path][start:start + SEQ + 1]
        assert len(chunk) == SEQ + 1
        n += 1
    assert n == len(_samples(ds))
    # targets are inputs shifted by one
    x, y = next(iter(ds))
    assert torch.equal(x[1:], y[:-1])


def test_set_epoch_changes_order_and_offsets(tmp_data):
    ds = IterableShardDataset(find_train_shards(tmp_data), SEQ, shuffle_buffer_size=8)
    ds.set_epoch(0)
    e0 = _starts(ds)
    ds.set_epoch(1)
    e1 = _starts(ds)
    assert e0 != e1
    # window offsets differ between epochs (starts are not all multiples of SEQ)
    assert any(s % SEQ != 0 for _, s in e0 + e1)
    # deterministic: same epoch -> same stream
    ds.set_epoch(1)
    assert _starts(ds) == e1


def test_no_shuffle_is_sequential(tmp_data):
    ds = IterableShardDataset(find_train_shards(tmp_data), SEQ, shuffle_shards=False,
                              shuffle_buffer_size=0, random_offset=False)
    starts = _starts(ds)
    assert [s for _, s in starts[:3]] == [0, SEQ, 2 * SEQ]


def test_skip_batches_single_worker_resumes_exactly(tmp_data):
    ds = IterableShardDataset(find_train_shards(tmp_data), SEQ, shuffle_buffer_size=8)
    ds.batch_size = 4
    full = _samples(ds)
    ds.skip_batches = 3
    wid, nw, shards, s0, step = ds._worker_layout()
    assert ds._worker_skip_samples(wid, nw) == 12
    resumed = _samples(ds)
    assert len(resumed) == len(full) - 12
    for (a, b), (c, d) in zip(resumed, full[12:], strict=True):
        assert torch.equal(a, c) and torch.equal(b, d)


def test_skip_batches_multi_worker_matches_dataloader_order(tmp_data, monkeypatch):
    """With W workers the DataLoader interleaves batches round-robin; skipping k global
    batches must leave each worker exactly the batches it would have produced after k."""
    paths = find_train_shards(tmp_data)
    W, bs, k = 2, 4, 5
    ds = IterableShardDataset(paths, SEQ, shuffle_buffer_size=8)
    ds.batch_size = bs

    # Per-worker streams before resume, chunked into batches (drop_last)
    per_worker = []
    for w in range(W):
        _with_worker(monkeypatch, w, W)
        s = _starts(ds)
        per_worker.append([s[i:i + bs] for i in range(0, len(s) - bs + 1, bs)])
    # Global loader order (round-robin while every worker still has batches)
    loader = []
    i = 0
    while all(i < len(b) for b in per_worker):
        for w in range(W):
            loader.append((w, per_worker[w][i]))
        i += 1
    assert len(loader) > k
    expected = {w: [b for ww, b in loader[k:] if ww == w] for w in range(W)}

    ds.skip_batches = k
    for w in range(W):
        _with_worker(monkeypatch, w, W)
        wid, nw, shards, s0, step = ds._worker_layout()
        skip = ds._worker_skip_samples(wid, nw)
        s = _starts(ds)[skip:]
        got = [s[i:i + bs] for i in range(0, len(s) - bs + 1, bs)]
        assert got[:len(expected[w])] == expected[w]


def test_fewer_shards_than_workers_covers_everything_once(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "one")
    write_shards(data_dir, sizes=(2000,), val_size=100)
    ds = IterableShardDataset(find_train_shards(data_dir), SEQ, shuffle_buffer_size=4)
    W = 3
    seen = []
    for w in range(W):
        _with_worker(monkeypatch, w, W)
        s = _starts(ds)
        assert s, f"worker {w} got no samples"
        seen += s
    assert len(seen) == len(set(seen))  # disjoint
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    assert set(seen) == set(_starts(ds))  # complete


def test_create_dataloader_batches(tmp_data):
    loader = create_dataloader(tmp_data, SEQ, batch_size=4, split="train", num_workers=0)
    assert loader.dataset.batch_size == 4
    x, y = next(iter(loader))
    assert x.shape == (4, SEQ) and y.shape == (4, SEQ)
    val = create_dataloader(tmp_data, SEQ, batch_size=2, split="val", num_workers=0)
    x, y = next(iter(val))
    assert x.shape == (2, SEQ)


def test_val_every_from_processing():
    assert _val_every_from_processing({"val_every": 50}, 200) == 50
    assert _val_every_from_processing({"val_ratio": 0.005}, 200) == 200
    assert _val_every_from_processing({}, 123) == 123


def test_tokenize_and_save_from_text_dir(tmp_path, tmp_tokenizer_dir):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(6):
        (corpus / f"doc{i}.txt").write_text(f"document number {i}: " + "the quick brown fox " * 20)
    out = tmp_path / "out"
    tokenize_and_save(
        tokenizer_path=tmp_tokenizer_dir, output_dir=str(out), shard_size=200, val_every=3,
        sources=[{"type": "text_dir", "path": str(corpus)}],
    )
    shards = find_train_shards(str(out))
    assert len(shards) >= 1
    assert os.path.exists(out / "val.bin")
    tokens = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in shards])
    assert tokens.max() < TINY_VOCAB
    # docs 0 and 3 are validation (every 3rd starting at 0) -> 4 train docs
    from src.tokenizer import LLMTokenizer
    tok = LLMTokenizer(tmp_tokenizer_dir)
    assert int((tokens == tok.bos_id).sum()) == 4
