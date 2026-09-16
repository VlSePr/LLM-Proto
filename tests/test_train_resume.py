"""End-to-end: a run interrupted at step 20 and resumed to 30 must match a single 30-step run."""
import os

import pytest
import torch

from src.config import ModelConfig, TrainConfig
from src.train import train
from src.utils import load_checkpoint_file
from tests.conftest import TINY_VOCAB, write_shards


def _cfg(tmp_path, tag, **kw):
    cfg = TrainConfig(
        data_dir=str(tmp_path / "data"),
        checkpoint_dir=str(tmp_path / f"ckpt_{tag}"),
        batch_size=2, gradient_accumulation_steps=2, max_steps=30, warmup_steps=2,
        peak_lr=1e-3, min_lr=1e-4, precision="fp32", use_compile="false", num_workers=0,
        log_every_steps=5, eval_every_steps=10, eval_max_batches=3, save_every_steps=10,
        generate_every_steps=10, visualize_every_steps=10_000, use_wandb=False, seed=7,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def small_model():
    return ModelConfig(vocab_size=TINY_VOCAB, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)


@pytest.fixture
def train_env(tmp_path, tmp_tokenizer_dir):
    # 2 tiny shards so the epoch rolls over several times within 30 steps
    write_shards(str(tmp_path / "data"), sizes=(700, 500), val_size=300, seed=3)
    return tmp_path


def test_resume_matches_uninterrupted_run(train_env, tmp_tokenizer_dir, small_model):
    tp = train_env
    # Reference: one uninterrupted 30-step run
    history = train(small_model, _cfg(tp, "ref", tokenizer_path=tmp_tokenizer_dir))
    ref = load_checkpoint_file(os.path.join(str(tp / "ckpt_ref"), "latest.pt"))
    assert ref["step"] == 29 and ref["epoch"] >= 1  # data rolled over at least once
    assert any("train/grad_norm" in h for h in history)  # max_grad_norm defaults to 1.0

    # Interrupted: same 30-step schedule, stopped right after step 20, then resumed
    train(small_model, _cfg(tp, "res", tokenizer_path=tmp_tokenizer_dir), stop_after_step=20)
    mid = load_checkpoint_file(os.path.join(str(tp / "ckpt_res"), "latest.pt"))
    assert mid["step"] == 20 and mid["best_val_loss"] is not None
    train(small_model, _cfg(tp, "res", tokenizer_path=tmp_tokenizer_dir, resume="latest"))
    res = load_checkpoint_file(os.path.join(str(tp / "ckpt_res"), "latest.pt"))

    assert res["step"] == 29
    assert res["epoch"] == ref["epoch"]
    assert res["batches_in_epoch"] == ref["batches_in_epoch"]
    assert res["tokens_seen"] == ref["tokens_seen"]
    assert res["best_val_loss"] == pytest.approx(ref["best_val_loss"], abs=1e-6)
    assert res["loss"] == pytest.approx(ref["loss"], abs=1e-6)
    for k in ref["model_state_dict"]:
        assert torch.allclose(ref["model_state_dict"][k], res["model_state_dict"][k], atol=1e-6), k

    # Resuming a finished run is a no-op that does not crash
    train(small_model, _cfg(tp, "res", tokenizer_path=tmp_tokenizer_dir, resume="latest"))


def test_missing_resume_raises_unless_resume_if_exists(train_env, tmp_tokenizer_dir, small_model):
    tp = train_env
    with pytest.raises(FileNotFoundError):
        train(small_model, _cfg(tp, "miss", tokenizer_path=tmp_tokenizer_dir, resume="latest", max_steps=1))
    train(small_model, _cfg(tp, "miss", tokenizer_path=tmp_tokenizer_dir, resume="latest",
                            resume_if_exists=True, max_steps=2))
    assert os.path.exists(os.path.join(str(tp / "ckpt_miss"), "latest.pt"))


def test_vocab_mismatch_is_rejected(train_env, tmp_tokenizer_dir):
    too_small = ModelConfig(vocab_size=256, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32)
    with pytest.raises(ValueError, match="vocab_size"):
        train(too_small, _cfg(train_env, "vocab", tokenizer_path=tmp_tokenizer_dir, max_steps=1))
