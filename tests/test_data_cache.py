"""Fingerprint, manifest and local/Drive cache behaviour of the tokenization pipeline."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import data
from src import gdrive
from src.data import (
    FORMAT_VERSION, compute_fingerprint, ensure_tokenized_data,
    find_train_shards, normalize_sources, read_manifest, tokenize_and_save,
    validate_local_cache,
)


# ──────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────

def _corpus(path: Path, n=6, prefix="doc") -> str:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (path / f"{prefix}{i}.txt").write_text(f"document number {i}: " + "the quick brown fox " * 20)
    return str(path)


def _src(path):
    return [{"type": "text_dir", "path": str(path)}]


def _fp(tok, sources, shard_size=200, val_every=3, max_tokens=None):
    return compute_fingerprint(tok, sources, shard_size, val_every, max_tokens)[0]


def _run(tok, out, sources, **kw):
    return ensure_tokenized_data(tokenizer_path=tok, output_dir=str(out), sources=sources,
                                 shard_size=200, val_every=3, **kw)


@pytest.fixture
def corpus(tmp_path):
    return _corpus(tmp_path / "corpus")


@pytest.fixture
def fake_drive(monkeypatch, tmp_path):
    """Pretend we are on Colab with Drive mounted at tmp_path/drive (exercises the real copy code)."""
    root = tmp_path / "drive"
    monkeypatch.setattr(gdrive, "_is_colab", lambda: True)

    def _folder(folder_id, create=True):
        p = root / folder_id
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return str(p)

    monkeypatch.setattr(gdrive, "_colab_folder", _folder)
    return root


def _forbid_tokenize(monkeypatch):
    monkeypatch.setattr(data, "tokenize_and_save",
                        lambda *a, **k: pytest.fail("tokenize_and_save should not have been called"))


# ──────────────────────────────────────────────
# Fingerprint
# ──────────────────────────────────────────────

def test_fingerprint_is_stable_across_corpus_location(tmp_path, tmp_tokenizer_dir, corpus):
    copy = tmp_path / "elsewhere" / "nested" / "corpus"
    shutil.copytree(corpus, copy)
    assert _fp(tmp_tokenizer_dir, _src(corpus)) == _fp(tmp_tokenizer_dir, _src(copy))
    for entry in normalize_sources(_src(corpus))[0]["files"]:
        assert "\\" not in entry["path"]


def test_fingerprint_changes_when_file_content_changes(tmp_tokenizer_dir, corpus):
    before = _fp(tmp_tokenizer_dir, _src(corpus))
    with open(os.path.join(corpus, "doc0.txt"), "a") as f:
        f.write(" extra words")
    assert _fp(tmp_tokenizer_dir, _src(corpus)) != before


def test_fingerprint_changes_when_file_added_or_renamed(tmp_tokenizer_dir, corpus):
    before = _fp(tmp_tokenizer_dir, _src(corpus))
    os.rename(os.path.join(corpus, "doc1.txt"), os.path.join(corpus, "doc1-renamed.txt"))
    renamed = _fp(tmp_tokenizer_dir, _src(corpus))
    assert renamed != before
    _corpus(Path(corpus), n=1, prefix="new")
    assert _fp(tmp_tokenizer_dir, _src(corpus)) != renamed


@pytest.mark.parametrize("kw", [{"shard_size": 300}, {"val_every": 4}, {"max_tokens": 100}])
def test_fingerprint_changes_with_processing_params(tmp_tokenizer_dir, corpus, kw):
    assert _fp(tmp_tokenizer_dir, _src(corpus), **kw) != _fp(tmp_tokenizer_dir, _src(corpus))


def test_fingerprint_changes_with_tokenizer(tmp_path, tmp_tokenizer_dir, corpus):
    other = tmp_path / "tok2"
    shutil.copytree(tmp_tokenizer_dir, other)
    with open(other / "tokenizer.json", "a") as f:
        f.write("\n")   # still valid JSON, different bytes
    assert _fp(str(other), _src(corpus)) != _fp(tmp_tokenizer_dir, _src(corpus))


def test_fingerprint_changes_with_source_order(tmp_path, tmp_tokenizer_dir):
    a, b = _corpus(tmp_path / "a", prefix="a"), _corpus(tmp_path / "b", prefix="b")
    assert _fp(tmp_tokenizer_dir, _src(a) + _src(b)) != _fp(tmp_tokenizer_dir, _src(b) + _src(a))


def test_fingerprint_huggingface_uses_metadata_only(tmp_tokenizer_dir):
    hf = {"type": "huggingface", "name": "org/ds", "subset": "x", "split": "train"}
    base = _fp(tmp_tokenizer_dir, [hf])
    assert _fp(tmp_tokenizer_dir, [dict(hf)]) == base
    assert _fp(tmp_tokenizer_dir, [dict(hf, subset="y")]) != base
    assert _fp(tmp_tokenizer_dir, [dict(hf, text_field="content")]) != base
    assert normalize_sources([hf])[0]["text_field"] == "text"


def test_normalize_sources_missing_dir_and_unknown_type(tmp_path):
    missing = normalize_sources([{"type": "text_dir", "path": str(tmp_path / "nope")}])
    assert missing == [{"type": "text_dir", "missing": True, "files": []}]
    with pytest.raises(ValueError):
        normalize_sources([{"type": "parquet", "path": "x"}])


def test_normalize_jsonl_single_file_and_dir(tmp_path):
    d = tmp_path / "j"
    d.mkdir()
    (d / "a.jsonl").write_text('{"text": "hello"}\n')
    single = normalize_sources([{"type": "jsonl", "path": str(d / "a.jsonl")}])[0]
    whole = normalize_sources([{"type": "jsonl", "path": str(d), "text_field": "body"}])[0]
    assert single["files"][0]["path"] == "a.jsonl" and whole["files"][0]["path"] == "a.jsonl"
    assert single["text_field"] == "text" and whole["text_field"] == "body"


# ──────────────────────────────────────────────
# tokenize_and_save: manifest + stale-shard cleanup
# ──────────────────────────────────────────────

def test_tokenize_and_save_writes_manifest(tmp_path, tmp_tokenizer_dir, corpus):
    out = tmp_path / "out"
    manifest = tokenize_and_save(tokenizer_path=tmp_tokenizer_dir, output_dir=str(out),
                                 shard_size=200, val_every=3, sources=_src(corpus))
    on_disk = read_manifest(str(out))
    assert on_disk == json.loads(json.dumps(manifest))
    assert on_disk["format_version"] == FORMAT_VERSION
    assert on_disk["fingerprint"] == _fp(tmp_tokenizer_dir, _src(corpus))
    from src.tokenizer import LLMTokenizer
    assert on_disk["tokenizer"]["vocab_size"] == LLMTokenizer(tmp_tokenizer_dir).vocab_size
    names = [f["name"] for f in on_disk["files"]]
    assert names[-1] == "val.bin"
    assert names[:-1] == [os.path.basename(p) for p in find_train_shards(str(out))]
    for f in on_disk["files"]:
        assert f["bytes"] == 2 * f["tokens"] == os.path.getsize(out / f["name"])
    assert sum(f["tokens"] for f in on_disk["files"][:-1]) == on_disk["train_tokens"]
    assert on_disk["n_train_shards"] == len(names) - 1
    assert validate_local_cache(str(out), on_disk["fingerprint"]) == (True, "ok")


def test_tokenize_and_save_clears_stale_shards(tmp_path, tmp_tokenizer_dir, corpus):
    out = tmp_path / "out"
    out.mkdir()
    for name in ("train_0007.bin", "val.bin", "manifest.json", "notes.txt"):
        (out / name).write_bytes(b"garbage")
    tokenize_and_save(tokenizer_path=tmp_tokenizer_dir, output_dir=str(out),
                      shard_size=200, val_every=3, sources=_src(corpus))
    assert not (out / "train_0007.bin").exists()
    assert (out / "notes.txt").read_bytes() == b"garbage"
    assert np.fromfile(out / "val.bin", dtype=np.uint16).size > 0


def test_validate_local_cache_reasons(tmp_path, tmp_tokenizer_dir, corpus):
    out = tmp_path / "out"
    fp = _fp(tmp_tokenizer_dir, _src(corpus))
    assert validate_local_cache(str(out), fp) == (False, "no manifest")
    tokenize_and_save(tokenizer_path=tmp_tokenizer_dir, output_dir=str(out),
                      shard_size=200, val_every=3, sources=_src(corpus))
    assert validate_local_cache(str(out), "0" * 64)[1] == "fingerprint mismatch"
    with open(out / "val.bin", "ab") as f:
        f.write(b"\x00\x00")
    assert validate_local_cache(str(out), fp)[1].startswith("size mismatch val.bin")
    os.remove(out / "train_0000.bin")
    assert validate_local_cache(str(out), fp) == (False, "missing train_0000.bin")


# ──────────────────────────────────────────────
# ensure_tokenized_data: local cache
# ──────────────────────────────────────────────

def test_ensure_local_hit_skips_tokenization(tmp_path, tmp_tokenizer_dir, corpus, monkeypatch):
    out = tmp_path / "out"
    _run(tmp_tokenizer_dir, out, _src(corpus))
    before = {p: os.path.getmtime(p) for p in find_train_shards(str(out))}
    _forbid_tokenize(monkeypatch)
    assert _run(tmp_tokenizer_dir, out, _src(corpus)) == str(out)
    assert {p: os.path.getmtime(p) for p in find_train_shards(str(out))} == before


def test_ensure_local_miss_after_corpus_change_retokenizes(tmp_path, tmp_tokenizer_dir, corpus):
    out = tmp_path / "out"
    _run(tmp_tokenizer_dir, out, _src(corpus))
    old_fp = read_manifest(str(out))["fingerprint"]
    with open(os.path.join(corpus, "doc2.txt"), "a") as f:
        f.write(" more text")
    _run(tmp_tokenizer_dir, out, _src(corpus))
    assert read_manifest(str(out))["fingerprint"] != old_fp


def test_ensure_force_retokenizes(tmp_path, tmp_tokenizer_dir, corpus, monkeypatch):
    out = tmp_path / "out"
    _run(tmp_tokenizer_dir, out, _src(corpus))
    calls = []
    real = data.tokenize_and_save

    def _spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(data, "tokenize_and_save", _spy)
    _run(tmp_tokenizer_dir, out, _src(corpus))                 # cache hit: no call
    _run(tmp_tokenizer_dir, out, _src(corpus), force=True)     # forced: one call
    assert calls == [1]


def test_ensure_raises_when_no_tokens_produced(tmp_path, tmp_tokenizer_dir):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError):
        _run(tmp_tokenizer_dir, tmp_path / "out", _src(empty))


def test_ensure_no_gdrive_calls_when_folder_empty(tmp_path, tmp_tokenizer_dir, corpus, monkeypatch):
    monkeypatch.setattr(gdrive, "upload_to_gdrive", lambda *a, **k: pytest.fail("upload called"))
    monkeypatch.setattr(gdrive, "resolve_subfolder", lambda *a, **k: pytest.fail("resolve called"))
    _run(tmp_tokenizer_dir, tmp_path / "out", _src(corpus), gdrive_folder_id="")


# ──────────────────────────────────────────────
# ensure_tokenized_data: Google Drive via a fake Colab mount
# ──────────────────────────────────────────────

def _remote(fake_drive, out):
    fp = read_manifest(str(out))["fingerprint"]
    return fake_drive / "LLM" / "tokenized" / fp[:12]


def test_ensure_uploads_to_drive_layout(tmp_path, tmp_tokenizer_dir, corpus, fake_drive):
    out = tmp_path / "out"
    _run(tmp_tokenizer_dir, out, _src(corpus), gdrive_folder_id="LLM")
    remote = _remote(fake_drive, out)
    manifest = read_manifest(str(out))
    assert read_manifest(str(remote)) == manifest
    for f in manifest["files"]:
        assert (remote / f["name"]).read_bytes() == (out / f["name"]).read_bytes()


def test_ensure_drive_hit_restores_local_dir(tmp_path, tmp_tokenizer_dir, corpus, fake_drive, monkeypatch):
    out = tmp_path / "out"
    _run(tmp_tokenizer_dir, out, _src(corpus), gdrive_folder_id="LLM")
    manifest = read_manifest(str(out))
    shutil.rmtree(out)
    _forbid_tokenize(monkeypatch)
    _run(tmp_tokenizer_dir, out, _src(corpus), gdrive_folder_id="LLM")
    assert read_manifest(str(out)) == manifest
    assert validate_local_cache(str(out), manifest["fingerprint"]) == (True, "ok")
    assert not list(out.glob("*.part"))


def test_ensure_drive_truncated_shard_falls_back_to_tokenize(tmp_path, tmp_tokenizer_dir, corpus, fake_drive):
    out = tmp_path / "out"
    _run(tmp_tokenizer_dir, out, _src(corpus), gdrive_folder_id="LLM")
    remote = _remote(fake_drive, out)
    good = (remote / "train_0000.bin").read_bytes()
    (remote / "train_0000.bin").write_bytes(good[:-2])
    shutil.rmtree(out)
    _run(tmp_tokenizer_dir, out, _src(corpus), gdrive_folder_id="LLM")
    assert (out / "train_0000.bin").read_bytes() == good
    assert (remote / "train_0000.bin").read_bytes() == good      # re-uploaded
    assert validate_local_cache(str(out), read_manifest(str(out))["fingerprint"]) == (True, "ok")


def test_ensure_drive_folder_without_manifest_is_a_miss(tmp_path, tmp_tokenizer_dir, corpus, fake_drive):
    fp = _fp(tmp_tokenizer_dir, _src(corpus))
    stray = fake_drive / "LLM" / "tokenized" / fp[:12]
    stray.mkdir(parents=True)
    (stray / "train_0000.bin").write_bytes(b"\x00\x00")
    out = tmp_path / "out"
    _run(tmp_tokenizer_dir, out, _src(corpus), gdrive_folder_id="LLM")
    assert validate_local_cache(str(out), fp) == (True, "ok")
    assert read_manifest(str(stray))["fingerprint"] == fp       # remote repaired by the upload


def test_ensure_drive_probe_creates_no_directories(tmp_path, tmp_tokenizer_dir, corpus, fake_drive, monkeypatch):
    monkeypatch.setattr(data, "_upload_cache_to_gdrive", lambda *a, **k: None)
    _run(tmp_tokenizer_dir, tmp_path / "out", _src(corpus), gdrive_folder_id="LLM")
    assert not (fake_drive / "LLM").exists()


def test_ensure_upload_failure_does_not_raise(tmp_path, tmp_tokenizer_dir, corpus, fake_drive, monkeypatch):
    def _fail(*a, **k):
        raise OSError("quota exceeded")
    monkeypatch.setattr(gdrive, "upload_to_gdrive", _fail)
    out = tmp_path / "out"
    _run(tmp_tokenizer_dir, out, _src(corpus), gdrive_folder_id="LLM")
    assert validate_local_cache(str(out), _fp(tmp_tokenizer_dir, _src(corpus))) == (True, "ok")


def test_ensure_download_failure_falls_back_to_tokenize(tmp_path, tmp_tokenizer_dir, corpus, fake_drive, monkeypatch):
    out = tmp_path / "out"
    _run(tmp_tokenizer_dir, out, _src(corpus), gdrive_folder_id="LLM")
    shutil.rmtree(out)

    def _fail(*a, **k):
        raise OSError("drive unreachable")
    monkeypatch.setattr(gdrive, "download_from_gdrive", _fail)
    _run(tmp_tokenizer_dir, out, _src(corpus), gdrive_folder_id="LLM")
    assert validate_local_cache(str(out), _fp(tmp_tokenizer_dir, _src(corpus))) == (True, "ok")


def test_download_from_gdrive_leaves_no_partial_file(tmp_path, fake_drive, monkeypatch):
    remote = fake_drive / "LLM"
    remote.mkdir(parents=True)
    (remote / "big.bin").write_bytes(b"x" * 100)

    def _fail(src, dst):
        with open(dst, "wb") as f:
            f.write(b"xx")
        raise OSError("disconnected")
    monkeypatch.setattr(shutil, "copy2", _fail)
    local = tmp_path / "local"
    with pytest.raises(OSError):
        gdrive.download_from_gdrive("big.bin", "LLM", str(local))
    assert not (local / "big.bin").exists() and not (local / "big.bin.part").exists()
    with pytest.raises(FileNotFoundError):
        gdrive.download_from_gdrive("nope.bin", "LLM", str(local))
