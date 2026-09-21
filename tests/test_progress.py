"""Transfer progress: chunked copy, byte callbacks in the Drive primitives, bar thresholds, outputs helpers."""
import os
import sys
import types

import pytest

from src import gdrive, outputs, progress
from src.progress import copy_with_progress, human_size, transfer_progress
from src.utils import load_metrics_history

# ──────────────────────────────────────────────
# src/progress.py
# ──────────────────────────────────────────────


def test_copy_with_progress_copies_exactly_and_reports_every_byte(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(os.urandom(10_000))
    os.utime(src, (1_000_000_000, 1_000_000_000))
    seen = []
    dst = copy_with_progress(str(src), str(tmp_path / "b.bin"), seen.append, chunk_bytes=4096)
    assert (tmp_path / "b.bin").read_bytes() == src.read_bytes()
    assert sum(seen) == 10_000 and len(seen) == 3
    assert os.path.getmtime(dst) == pytest.approx(1_000_000_000)      # copy2 semantics: timestamps kept


def test_copy_with_progress_handles_empty_file(tmp_path):
    (tmp_path / "e").write_bytes(b"")
    seen = []
    copy_with_progress(str(tmp_path / "e"), str(tmp_path / "f"), seen.append)
    assert (tmp_path / "f").read_bytes() == b"" and seen == []


def test_transfer_progress_bar_only_for_large_transfers(monkeypatch):
    bars = []

    class _Bar:
        def __init__(self, **kw):
            bars.append(kw)
            self.n = 0

        def update(self, n):
            self.n += n

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(progress, "tqdm", _Bar)
    with transfer_progress(True, progress.MIN_BAR_BYTES - 1, "small") as bump:
        bump(5)
    with transfer_progress(False, progress.MIN_BAR_BYTES * 5, "silenced") as bump:
        bump(5)
    assert bars == []                                                 # small or disabled: no bar
    with transfer_progress(True, progress.MIN_BAR_BYTES, "big") as bump:
        bump(7)
    assert len(bars) == 1 and bars[0]["total"] == progress.MIN_BAR_BYTES and bars[0]["unit"] == "B"


def test_transfer_progress_passes_a_callable_through():
    calls = []
    with transfer_progress(calls.append, 1, "x") as bump:
        bump(3)
    assert calls == [3]                                               # even tiny files feed a caller's own bar


def test_human_size():
    assert human_size(0) == "0 B"
    assert human_size(1500) == "1.5 KB"
    assert human_size(2_500_000) == "2.5 MB"
    assert human_size(3_200_000_000) == "3.2 GB"


# ──────────────────────────────────────────────
# gdrive primitives feed the callback (Colab-style filesystem copy)
# ──────────────────────────────────────────────

@pytest.fixture
def colab_drive(tmp_path, monkeypatch):
    root = tmp_path / "drive"
    root.mkdir()
    monkeypatch.setattr(gdrive, "_is_colab", lambda: True)

    def _folder(folder_id, create=True):
        path = root / folder_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return str(path)
    monkeypatch.setattr(gdrive, "_colab_folder", _folder)
    return root


def test_colab_upload_and_download_report_bytes(tmp_path, colab_drive):
    src = tmp_path / "ckpt.pt"
    src.write_bytes(os.urandom(50_000))
    up, down = [], []
    gdrive.upload_to_gdrive(str(src), "run", progress=up.append)
    assert (colab_drive / "run" / "ckpt.pt").read_bytes() == src.read_bytes()
    out = gdrive.download_from_gdrive("ckpt.pt", "run", str(tmp_path / "local"), progress=down.append)
    assert open(out, "rb").read() == src.read_bytes()
    assert sum(up) == sum(down) == 50_000
    assert not list((tmp_path / "local").glob("*.part"))


# ──────────────────────────────────────────────
# gdrive primitives feed the callback (REST API mode, fake service)
# ──────────────────────────────────────────────

class _Status:
    def __init__(self, done, total):
        self.resumable_progress = done
        self.total_size = total


class _UploadRequest:
    def __init__(self, total):
        self.total, self.pos = total, 0

    def next_chunk(self):
        self.pos = min(self.total, self.pos + 4000)
        if self.pos < self.total:
            return _Status(self.pos, self.total), None
        return None, {"id": "file-123"}


class _UploadFiles:
    def __init__(self, total):
        self.total, self.calls = total, []

    def create(self, **kw):
        self.calls.append("create")
        return _UploadRequest(self.total)

    def update(self, **kw):
        self.calls.append("update")
        return _UploadRequest(self.total)


class _Service:
    def __init__(self, files):
        self._files = files

    def files(self):
        return self._files


@pytest.mark.parametrize("existing, verb", [(None, "create"), ("old-id", "update")])
def test_api_upload_reports_bytes_and_returns_id(tmp_path, monkeypatch, existing, verb):
    pytest.importorskip("googleapiclient")
    src = tmp_path / "shard.bin"
    src.write_bytes(b"x" * 10_000)
    files = _UploadFiles(10_000)
    monkeypatch.setattr(gdrive, "_is_colab", lambda: False)
    monkeypatch.setattr(gdrive, "_get_service", lambda creds: _Service(files))
    monkeypatch.setattr(gdrive, "_find_file", lambda service, name, folder: existing)
    seen = []
    assert gdrive.upload_to_gdrive(str(src), "FOLDER", progress=seen.append) == "file-123"
    assert files.calls == [verb]
    assert sum(seen) == 10_000                                        # chunk deltas plus the final remainder


def test_api_download_reports_bytes_and_moves_part_into_place(tmp_path, monkeypatch):
    googleapiclient_http = pytest.importorskip("googleapiclient.http")

    class _Downloader:
        def __init__(self, fh, request, chunksize):
            self.fh, self.pos, self.total = fh, 0, 9_000

        def next_chunk(self):
            step = min(4_000, self.total - self.pos)
            self.fh.write(b"y" * step)
            self.pos += step
            return _Status(self.pos, self.total), self.pos >= self.total

    class _GetMedia:
        def get_media(self, fileId):
            return object()

    monkeypatch.setattr(googleapiclient_http, "MediaIoBaseDownload", _Downloader)
    monkeypatch.setattr(gdrive, "_is_colab", lambda: False)
    monkeypatch.setattr(gdrive, "_get_service", lambda creds: _Service(_GetMedia()))
    monkeypatch.setattr(gdrive, "_find_file", lambda service, name, folder: "id-1")
    seen = []
    path = gdrive.download_from_gdrive("shard.bin", "FOLDER", str(tmp_path), progress=seen.append)
    assert open(path, "rb").read() == b"y" * 9_000
    assert sum(seen) == 9_000 and not os.path.exists(path + ".part")


# ──────────────────────────────────────────────
# metrics history loader
# ──────────────────────────────────────────────

def test_load_metrics_history(tmp_path):
    assert load_metrics_history(str(tmp_path)) == []                  # nothing saved yet
    (tmp_path / "metrics_history.json").write_text('[{"step": 1, "train/loss": 2.0}]')
    assert load_metrics_history(str(tmp_path)) == [{"step": 1, "train/loss": 2.0}]
    (tmp_path / "metrics_history.json").write_text("{not json")
    assert load_metrics_history(str(tmp_path)) == []                  # corrupt file: no crash, no curves


def test_load_metrics_history_falls_back_to_drive(tmp_path, monkeypatch):
    def fake_download(filename, folder, local_dir, creds="", **kw):
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, filename), "w") as f:
            f.write('[{"step": 9}]')
    monkeypatch.setattr(gdrive, "download_from_gdrive", fake_download)
    assert load_metrics_history(str(tmp_path / "ck"), gdrive_folder_id="LLM/run") == [{"step": 9}]


# ──────────────────────────────────────────────
# src/outputs.py
# ──────────────────────────────────────────────

def test_run_output_dir_sits_next_to_checkpoints(tmp_path):
    out = outputs.run_output_dir(str(tmp_path / "run" / "checkpoints"))
    assert out == str(tmp_path / "run" / "outputs") and os.path.isdir(out)


def test_publish_output_copies_to_drive_outputs_subfolder(tmp_path, colab_drive, monkeypatch, capsys):
    monkeypatch.setattr(gdrive, "resolve_subfolder", lambda parent, name, creds="", create=True: f"{parent}/{name}")
    f = tmp_path / "emb.html"
    f.write_text("<html></html>")
    assert outputs.publish_output(str(f), "LLM/run", download=False) == str(f)
    assert (colab_drive / "LLM" / "run" / "outputs" / "emb.html").read_text() == "<html></html>"
    assert "Saved" in capsys.readouterr().out


def test_publish_output_survives_drive_failure(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("drive down")
    monkeypatch.setattr(gdrive, "resolve_subfolder", boom)
    f = tmp_path / "emb.html"
    f.write_text("x")
    assert outputs.publish_output(str(f), "LLM/run", download=False) == str(f)
    assert "Google Drive copy failed" in capsys.readouterr().out


def test_offer_download_is_noop_off_colab_and_calls_colab_files_on_colab(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    assert outputs.offer_download(str(tmp_path / "x")) is False

    downloaded = []
    files_mod = types.SimpleNamespace(download=downloaded.append)
    colab = types.ModuleType("google.colab")
    colab.files = files_mod
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    assert outputs.offer_download("some/file.html") is True
    assert downloaded == ["some/file.html"]


def test_zip_outputs_round_trip(tmp_path):
    import zipfile
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "a.html").write_text("A")
    (out / "b.png").write_bytes(b"B")
    zpath = outputs.zip_outputs(str(out))
    assert zpath == str(tmp_path / "outputs.zip")
    with zipfile.ZipFile(zpath) as zf:
        assert sorted(zf.namelist()) == ["a.html", "b.png"]
        assert zf.read("a.html") == b"A"
    assert outputs.list_outputs(str(out)) == [("a.html", 1), ("b.png", 1)]
    with pytest.raises(FileNotFoundError):
        outputs.zip_outputs(str(tmp_path / "empty"))


# ──────────────────────────────────────────────
# src/report.py
# ──────────────────────────────────────────────

def test_run_summary_states_the_token_budget(tmp_path):
    from src.config import ModelConfig, TrainConfig
    from src.report import run_summary
    cfg = ModelConfig(vocab_size=512, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    tc = TrainConfig(batch_size=4, gradient_accumulation_steps=2, max_steps=110, warmup_steps=5,
                     resume="latest", checkpoint_dir="ck", backup_to_gdrive=True)
    text = run_summary(cfg, tc, start_step=10, n_steps=100, model_name="tiny", drive_folder="LLM/run")
    assert "8 sequences x 64 tokens = 512 tokens/step" in text
    assert "10 -> 110 (100 more; resuming from 'latest')" in text
    assert "This session: 51,200 tokens" in text
    assert "LLM/run" in text
    fresh = run_summary(cfg, tc, start_step=0, n_steps=100)
    assert "fresh run" in fresh
    assert "disabled" in run_summary(cfg, tc, 0, 100)                 # no folder given


def test_dataset_and_training_summaries(tmp_path):
    import json

    from src.report import dataset_summary, training_summary
    assert "no manifest" in dataset_summary(str(tmp_path))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "fingerprint": "abcdef1234567890", "created": "2026-09-21T10:00:00+00:00",
        "tokenizer": {"vocab_size": 32000},
        "files": [{"name": "train_0000.bin", "bytes": 2_000_000, "tokens": 1_000_000},
                  {"name": "val.bin", "bytes": 20_000, "tokens": 10_000}],
        "train_tokens": 1_000_000, "val_tokens": 10_000}))
    text = dataset_summary(str(tmp_path))
    assert "1 train shard(s)" in text and "1,000,000 train tokens" in text and "abcdef123456" in text

    history = [{"step": 10, "train/loss": 5.0, "train/perplexity": 148.4, "train/tokens_seen": 1000},
               {"step": 50, "val/loss": 4.0, "val/perplexity": 54.6},
               {"step": 100, "train/loss": 3.5, "train/perplexity": 33.1, "train/tokens_seen": 5000},
               {"step": 100, "val/loss": 4.2, "val/perplexity": 66.7}]
    out = training_summary(history, wall_seconds=120)
    assert "Last logged step: 100" in out and "best 4.0000 at step 50" in out
    assert "Tokens seen: 5,000" in out and "2.0 min" in out
    assert "no metrics" in training_summary([])


def test_list_outputs_summary(tmp_path):
    from src.report import list_outputs_summary
    assert "No files" in list_outputs_summary(str(tmp_path / "none"))
    (tmp_path / "a.html").write_text("x" * 2000)
    assert "a.html" in list_outputs_summary(str(tmp_path)) and "2.0 KB" in list_outputs_summary(str(tmp_path))


def test_save_figure_writes_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = outputs.save_figure(fig, "curve.png", str(tmp_path / "out"), dpi=50)
    plt.close(fig)
    assert path == str(tmp_path / "out" / "curve.png")
    assert open(path, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
