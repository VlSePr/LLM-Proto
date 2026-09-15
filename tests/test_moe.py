"""SparseMoE layer, grafting onto a dense TransformerLM, freezing, growth and diagnostics."""
import pytest
import torch

from src.config import MoEConfig, config_from_dict
from src.model import FeedForward, TransformerLM
from src.moe import (
    SparseMoE,
    expand_moe_experts,
    freeze_for_moe,
    graft_moe,
    measure_expert_load,
    moe_layers,
    moe_metadata,
    param_budget_report,
)


def test_sparse_moe_forward_shape_and_aux_loss(seed):
    moe = SparseMoE(64, 128, n_experts=4, top_k=2)
    x = torch.randn(2, 16, 64)
    moe.train()
    out = moe(x)
    assert out.shape == x.shape
    assert moe.aux_loss.item() > 0
    assert moe.aux_loss.requires_grad
    moe.eval()
    moe(x)
    assert moe.aux_loss.item() == 0.0 and not moe.aux_loss.requires_grad


def test_sparse_moe_rejects_bad_top_k():
    with pytest.raises(ValueError, match="top_k"):
        SparseMoE(8, 16, n_experts=2, top_k=3)


@pytest.mark.parametrize("n_experts,top_k", [(2, 1), (3, 3)])
def test_graft_with_warm_start_preserves_output(tiny_cfg, seed, n_experts, top_k):
    model = TransformerLM(tiny_cfg).eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 12))
    with torch.no_grad():
        before = model(ids)["logits"]
    grafted = graft_moe(model, MoEConfig(expert_layers=[1], n_experts=n_experts, top_k=top_k))
    assert grafted == [1]
    assert isinstance(model.layers[1].ffn, SparseMoE) and isinstance(model.layers[0].ffn, FeedForward)
    with torch.no_grad():
        after = model(ids)["logits"]
    # identical experts + weights summing to 1 reproduce the dense FFN exactly
    assert torch.allclose(before, after, atol=1e-5)


def test_graft_without_warm_start_changes_output(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg).eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 12))
    with torch.no_grad():
        before = model(ids)["logits"]
    graft_moe(model, MoEConfig(expert_layers=[0, 1], n_experts=2, top_k=1, warm_start=False))
    with torch.no_grad():
        after = model(ids)["logits"]
    assert not torch.allclose(before, after, atol=1e-5)


def test_graft_rejects_bad_layers(tiny_cfg):
    model = TransformerLM(tiny_cfg)
    with pytest.raises(ValueError, match="out of range"):
        graft_moe(model, MoEConfig(expert_layers=[5]))
    graft_moe(model, MoEConfig(expert_layers=[1]))
    with pytest.raises(ValueError, match="already"):
        graft_moe(model, MoEConfig(expert_layers=[1]))


def test_freeze_for_moe_only_experts_trainable(tiny_cfg):
    model = TransformerLM(tiny_cfg)
    graft_moe(model, MoEConfig(expert_layers=[1], n_experts=2, top_k=1))
    report = freeze_for_moe(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("layers.1.ffn.") for n in trainable)
    assert report["total"] == report["trainable"] + report["frozen"]
    assert report["trainable"] == sum(p.numel() for p in model.layers[1].ffn.parameters())
    assert "MoE layers: [1]" in param_budget_report(model)
    assert "MoE layers: [1] (n_experts=2, top_k=1)" in model.summary()


def test_expand_moe_experts_preserves_old(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg)
    graft_moe(model, MoEConfig(expert_layers=[0, 1], n_experts=1, top_k=1))
    freeze_for_moe(model)
    moe = model.layers[1].ffn
    old_router = moe.router.weight.detach().clone()
    old_expert = {k: v.clone() for k, v in moe.experts[0].state_dict().items()}

    n = expand_moe_experts(model, 1, top_k=2)
    assert n == 2 and moe.n_experts == 2 and moe.top_k == 2
    assert moe.router.weight.shape == (2, tiny_cfg.dim)
    assert torch.equal(moe.router.weight[:1], old_router)
    assert moe.router.weight.requires_grad
    assert all(not p.requires_grad for p in moe.experts[0].parameters())
    assert all(p.requires_grad for p in moe.experts[1].parameters())
    for k, v in moe.experts[0].state_dict().items():
        assert torch.equal(v, old_expert[k])
        assert torch.equal(moe.experts[1].state_dict()[k], old_expert[k])  # warm-started copy
    model.train()
    out = model(torch.randint(0, tiny_cfg.vocab_size, (2, 8)))
    assert out["logits"].shape == (2, 8, tiny_cfg.vocab_size)
    assert moe_metadata(model) == {"expert_layers": [0, 1], "n_experts": 2, "top_k": 2}


def test_expand_requires_moe_layers(tiny_cfg):
    with pytest.raises(ValueError, match="no SparseMoE"):
        expand_moe_experts(TransformerLM(tiny_cfg))


def test_model_forward_exposes_aux_loss_and_backprops_with_grad_ckpt(tiny_cfg, seed):
    dense = TransformerLM(tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    assert "aux_loss" not in dense(ids, targets=ids)

    model = TransformerLM(tiny_cfg)
    graft_moe(model, MoEConfig(expert_layers=[1], n_experts=2, top_k=1))
    freeze_for_moe(model)
    model.enable_gradient_checkpointing()
    model.train()
    out = model(ids, targets=ids)
    assert out["aux_loss"].requires_grad
    (out["loss"] + out["aux_loss"]).backward()
    router = model.layers[1].ffn.router.weight
    assert router.grad is not None and torch.isfinite(router.grad).all()
    assert model.tok_emb.weight.grad is None  # frozen


def test_measure_expert_load_counts_sum_to_tokens_times_top_k(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg)
    graft_moe(model, MoEConfig(expert_layers=[0, 1], n_experts=3, top_k=2))
    batches = [(torch.randint(0, tiny_cfg.vocab_size, (2, 8)), None) for _ in range(5)]
    counts = measure_expert_load(model, batches, n_batches=3)
    assert sorted(counts) == [0, 1]
    for c in counts.values():
        assert c.shape == (3,) and c.sum().item() == 3 * 2 * 8 * 2
    assert len(moe_layers(model)) == 2


def test_moe_metadata_roundtrip_and_validation(tiny_cfg):
    assert moe_metadata(TransformerLM(tiny_cfg)) is None
    model = TransformerLM(tiny_cfg)
    graft_moe(model, MoEConfig(expert_layers=[1, 0], n_experts=2, top_k=2))
    meta = moe_metadata(model)
    cfg = config_from_dict(MoEConfig, meta, "ckpt")
    assert (cfg.expert_layers, cfg.n_experts, cfg.top_k) == ([0, 1], 2, 2)
    with pytest.raises(ValueError, match="top_k"):
        MoEConfig(expert_layers=[0], n_experts=2, top_k=3)
    with pytest.raises(ValueError, match="non-empty"):
        MoEConfig(expert_layers=[])
    with pytest.raises(ValueError, match="duplicates"):
        MoEConfig(expert_layers=[1, 1])
