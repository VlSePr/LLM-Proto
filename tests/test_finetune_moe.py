"""MoE fine-tuning through src.train.train(model=...) and the src.finetune_moe orchestration."""
import os

import pytest
import torch

from src.config import ModelConfig, MoEConfig, TrainConfig
from src.finetune_moe import finetune_moe, parse_layer_spec
from src.model import TransformerLM
from src.moe import SparseMoE, freeze_for_moe, graft_moe
from src.train import train
from src.utils import build_model_from_checkpoint, load_checkpoint_file
from tests.conftest import TINY_VOCAB, write_shards


def _cfg(tmp_path, tag, **kw):
    cfg = TrainConfig(
        data_dir=str(tmp_path / "data"),
        checkpoint_dir=str(tmp_path / f"ckpt_{tag}"),
        batch_size=2, gradient_accumulation_steps=2, max_steps=4, warmup_steps=1, seq_len=16,
        peak_lr=1e-3, min_lr=1e-4, precision="fp32", use_compile="false", num_workers=0,
        log_every_steps=1, eval_every_steps=2, eval_max_batches=2, save_every_steps=2,
        generate_every_steps=1000, visualize_every_steps=10_000, use_wandb=False, seed=7,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def small_model():
    return ModelConfig(vocab_size=TINY_VOCAB, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)


@pytest.fixture
def train_env(tmp_path):
    write_shards(str(tmp_path / "data"), sizes=(600, 400), val_size=200, seed=3)
    return tmp_path


def test_parse_layer_spec():
    assert parse_layer_spec("24-27") == [24, 25, 26, 27]
    assert parse_layer_spec("1,3, 5") == [1, 3, 5]
    assert parse_layer_spec("0-1,8") == [0, 1, 8]
    with pytest.raises(ValueError):
        parse_layer_spec(",")


def test_train_with_frozen_moe_only_updates_experts(train_env, tmp_tokenizer_dir, small_model):
    model = TransformerLM(small_model)
    graft_moe(model, MoEConfig(expert_layers=[1], n_experts=2, top_k=1))
    freeze_for_moe(model)
    base_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    moe_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    cfg = _cfg(train_env, "moe", tokenizer_path=tmp_tokenizer_dir)
    history = train(model.config, cfg, model=model)

    for n, p in model.named_parameters():
        if n in base_before:
            assert torch.equal(p, base_before[n]), f"frozen param changed: {n}"
    assert any(not torch.equal(p, moe_before[n]) for n, p in model.named_parameters() if n in moe_before)

    ckpt = load_checkpoint_file(os.path.join(cfg.checkpoint_dir, "latest.pt"))
    assert ckpt["moe"] == {"expert_layers": [1], "n_experts": 2, "top_k": 1}
    assert ckpt["step"] == 3
    logged = [h for h in history if "train/aux_loss" in h]
    assert logged and all(h["train/aux_loss"] > 0 for h in logged)
    assert "seq_len" in ckpt["train_config"] and ckpt["train_config"]["seq_len"] == 16


def test_dense_training_has_no_aux_loss_and_returns_history(train_env, tmp_tokenizer_dir, small_model):
    history = train(small_model, _cfg(train_env, "dense", tokenizer_path=tmp_tokenizer_dir, max_steps=2))
    assert history and all("train/aux_loss" not in h for h in history)
    ckpt = load_checkpoint_file(os.path.join(str(train_env / "ckpt_dense"), "latest.pt"))
    assert "moe" not in ckpt


def test_finetune_moe_end_to_end_and_expansion_round(train_env, tmp_tokenizer_dir, small_model):
    tp = train_env
    # Base pre-training run -> dense checkpoint
    train(small_model, _cfg(tp, "base", tokenizer_path=tmp_tokenizer_dir, max_steps=2))
    base = os.path.join(str(tp / "ckpt_base"), "latest.pt")

    # Round 1: graft one expert into layer 1 and fine-tune it
    model1, hist1 = finetune_moe(base, MoEConfig(expert_layers=[1], n_experts=1, top_k=1),
                                 _cfg(tp, "r1", tokenizer_path=tmp_tokenizer_dir))
    assert isinstance(model1.layers[1].ffn, SparseMoE) and model1.layers[1].ffn.n_experts == 1
    assert hist1 and any("train/aux_loss" in h for h in hist1)
    r1 = os.path.join(str(tp / "ckpt_r1"), "latest.pt")
    expert0_r1 = {k: v.clone() for k, v in model1.layers[1].ffn.experts[0].state_dict().items()}

    # Round 2: grow from the round-1 checkpoint; old expert frozen, new one trained, top_k -> 2
    model2, _ = finetune_moe(base, MoEConfig(expert_layers=[1], n_experts=2, top_k=2),
                             _cfg(tp, "r2", tokenizer_path=tmp_tokenizer_dir), expand_from=r1, n_new=1)
    moe = model2.layers[1].ffn
    assert moe.n_experts == 2 and moe.top_k == 2
    for k, v in moe.experts[0].state_dict().items():
        assert torch.equal(v, expert0_r1[k])

    reloaded, ckpt = build_model_from_checkpoint(os.path.join(str(tp / "ckpt_r2"), "latest.pt"), torch.device("cpu"))
    assert ckpt["moe"] == {"expert_layers": [1], "n_experts": 2, "top_k": 2}
    assert reloaded.layers[1].ffn.n_experts == 2
    ids = torch.randint(0, TINY_VOCAB, (1, 8))
    with torch.no_grad():
        assert torch.allclose(reloaded(ids)["logits"], model2.eval()(ids)["logits"], atol=1e-6)

    # Grafting onto an MoE checkpoint without expand_from is refused
    with pytest.raises(ValueError, match="already has MoE"):
        finetune_moe(r1, MoEConfig(expert_layers=[1]), _cfg(tp, "bad", tokenizer_path=tmp_tokenizer_dir))
