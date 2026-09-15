import os
import random

import numpy as np
import torch

from src.config import TrainConfig
from src.model import TransformerLM
from src.utils import (
    save_checkpoint, load_checkpoint, has_checkpoint, build_model_from_checkpoint,
    strip_compile_prefix, unwrap_model, resolve_checkpoint_filename, cleanup_checkpoints,
)


class FakeCompiled(torch.nn.Module):
    """Mimics torch.compile's OptimizedModule: wraps `_orig_mod` and prefixes state-dict keys."""

    def __init__(self, mod):
        super().__init__()
        self._orig_mod = mod

    def forward(self, *a, **k):
        return self._orig_mod(*a, **k)


def _train_cfg(tmp_path, **kw):
    cfg = TrainConfig(use_wandb=False, checkpoint_dir=str(tmp_path / "ckpt"), keep_last_n_checkpoints=2)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_round_trip_through_compile_prefix(tiny_cfg, tmp_path, seed):
    model = TransformerLM(tiny_cfg)
    wrapped = FakeCompiled(model)
    assert all(k.startswith("_orig_mod.") for k in wrapped.state_dict())
    assert unwrap_model(wrapped) is model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tcfg = _train_cfg(tmp_path)

    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    path = save_checkpoint(
        wrapped, opt, 7, 1.23, tiny_cfg, tcfg, tcfg.checkpoint_dir, is_best=True,
        best_val_loss=0.5, val_loss=0.6, scaler_state={"scale": 1.0}, epoch=2,
        batches_in_epoch=11, tokens_seen=999,
    )
    assert os.path.basename(path) == "step_7.pt"
    for name in ("step_7.pt", "latest.pt", "best.pt"):
        assert os.path.exists(os.path.join(tcfg.checkpoint_dir, name))

    # Saved keys have no compile prefix and load with weights_only=True
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert not any(k.startswith("_orig_mod.") for k in raw["model_state_dict"])
    assert raw["best_val_loss"] == 0.5 and raw["epoch"] == 2 and raw["batches_in_epoch"] == 11
    assert raw["tokens_seen"] == 999 and raw["scaler_state_dict"] == {"scale": 1.0}

    # Load into a fresh *uncompiled* model
    fresh = TransformerLM(tiny_cfg)
    ckpt = load_checkpoint(tcfg.checkpoint_dir, "latest", fresh, device=torch.device("cpu"))
    assert ckpt["step"] == 7
    for (ka, va), (kb, vb) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)

    # RNG states were restored: the next draws match the ones right after saving
    expected = (random.random(), float(np.random.rand()), torch.rand(1).item())
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    # (save_checkpoint captured the state *after* seeding; re-seed and re-capture to compare)
    load_checkpoint(tcfg.checkpoint_dir, "latest", fresh, device=torch.device("cpu"))
    assert (random.random(), float(np.random.rand()), torch.rand(1).item()) == expected


def test_load_legacy_prefixed_state_dict(tiny_cfg, tmp_path):
    """Checkpoints written by older code with `_orig_mod.` keys still load."""
    model = TransformerLM(tiny_cfg)
    sd = {"_orig_mod." + k: v for k, v in model.state_dict().items()}
    fresh = TransformerLM(tiny_cfg)
    fresh.load_state_dict(strip_compile_prefix(sd))
    assert torch.equal(fresh.tok_emb.weight, model.tok_emb.weight)


def test_build_model_from_checkpoint_uses_saved_config(tiny_cfg, tmp_path):
    model = TransformerLM(tiny_cfg)
    tcfg = _train_cfg(tmp_path)
    path = save_checkpoint(model, None, 1, None, tiny_cfg, tcfg, tcfg.checkpoint_dir)
    loaded, ckpt = build_model_from_checkpoint(path, torch.device("cpu"), model_config_name="large")
    assert loaded.config == tiny_cfg  # checkpoint wins over the (wrong) preset name
    assert not loaded.training
    assert ckpt["step"] == 1


def test_resolve_checkpoint_filename():
    assert resolve_checkpoint_filename("latest") == "latest.pt"
    assert resolve_checkpoint_filename("best") == "best.pt"
    assert resolve_checkpoint_filename("step_5") == "step_5.pt"
    assert resolve_checkpoint_filename("Large.pt") == "Large.pt"


def test_has_checkpoint_and_cleanup(tiny_cfg, tmp_path):
    model = TransformerLM(tiny_cfg)
    tcfg = _train_cfg(tmp_path, keep_last_n_checkpoints=2)
    assert not has_checkpoint(tcfg.checkpoint_dir, "latest")
    for step in (1, 2, 3):
        save_checkpoint(model, None, step, None, tiny_cfg, tcfg, tcfg.checkpoint_dir)
    files = sorted(f for f in os.listdir(tcfg.checkpoint_dir) if f.startswith("step_"))
    assert files == ["step_2.pt", "step_3.pt"]
    assert has_checkpoint(tcfg.checkpoint_dir, "latest")
    assert has_checkpoint(tcfg.checkpoint_dir, "step_3")
    assert not has_checkpoint(tcfg.checkpoint_dir, "step_1")
    # keep_n <= 0 keeps everything
    cleanup_checkpoints(tcfg.checkpoint_dir, 0)
    assert len([f for f in os.listdir(tcfg.checkpoint_dir) if f.startswith("step_")]) == 2
