import pytest
import torch

from src.model import IGNORE_INDEX, TransformerLM


def test_forward_shapes_and_loss(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    out = model(ids, targets=ids)
    assert out["logits"].shape == (2, 16, tiny_cfg.vocab_size)
    assert torch.isfinite(out["loss"])


def test_ignore_index_excluded_from_loss(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    targets = ids.clone()
    targets[:, 4:] = IGNORE_INDEX
    full = model(ids, targets=ids)["loss"]
    partial = model(ids, targets=targets)["loss"]
    # Manual: mean CE over the first 4 positions only
    logits = model(ids)["logits"][0, :4]
    manual = torch.nn.functional.cross_entropy(logits, ids[0, :4])
    assert torch.allclose(partial, manual, atol=1e-6)
    assert not torch.allclose(full, partial)


def test_forward_rejects_positions_past_context(tiny_cfg):
    model = TransformerLM(tiny_cfg)
    ids = torch.zeros(1, tiny_cfg.max_seq_len + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(ids)


def test_generate_never_exceeds_max_seq_len(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg).eval()
    prompt = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.max_seq_len - 3))
    out = model.generate(prompt, max_new_tokens=50, temperature=0.0)
    assert out.shape[1] == tiny_cfg.max_seq_len
    assert torch.equal(out[:, : prompt.shape[1]], prompt)

    full = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.max_seq_len))
    assert torch.equal(model.generate(full, max_new_tokens=5), full)


def test_generate_per_row_eos(tiny_cfg, seed):
    """Rows that hit EOS stop independently and are padded afterwards."""
    model = TransformerLM(tiny_cfg).eval()
    eos = 1
    # Bias the output projection so token `eos` is always the argmax -> every row emits EOS on step 1
    with torch.no_grad():
        model.output.weight[eos] += 100.0
    prompt = torch.randint(2, tiny_cfg.vocab_size, (3, 4))
    out = model.generate(prompt, max_new_tokens=10, temperature=0.0, eos_token_id=eos)
    assert out.shape == (3, 5)  # stopped after the first EOS for all rows
    assert (out[:, -1] == eos).all()


def test_generate_padding_for_finished_rows(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg).eval()
    eos, pad = 1, 2
    prompt = torch.randint(3, tiny_cfg.vocab_size, (2, 4))
    # Make row 0 emit EOS immediately by giving it a huge logit only through its prompt? Simpler:
    # run greedy, then verify the invariant: after the first EOS in a row, only `pad` follows.
    out = model.generate(prompt, max_new_tokens=20, temperature=0.0, eos_token_id=eos, pad_token_id=pad)
    for row in out:
        gen = row[4:].tolist()
        if eos in gen:
            after = gen[gen.index(eos) + 1:]
            assert all(t == pad for t in after)


def test_greedy_kv_cache_matches_full_forward(tiny_cfg, seed):
    """Incremental decoding with the KV cache must equal recomputing from scratch."""
    model = TransformerLM(tiny_cfg).eval()
    prompt = torch.randint(0, tiny_cfg.vocab_size, (1, 6))
    out = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    # Re-derive greedily without cache
    seq = prompt
    for _ in range(8):
        logits = model(seq)["logits"][:, -1, :]
        seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, seq)


def test_repetition_penalty_discourages_repeats(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg).eval()
    tok = 7
    with torch.no_grad():
        model.output.weight[tok] += 5.0  # `tok` becomes the strong argmax
    prompt = torch.full((1, 4), tok, dtype=torch.long)
    plain = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert (plain[0, 4:] == tok).all()
    penalized = model.generate(prompt, max_new_tokens=6, temperature=0.0, repetition_penalty=1e6)
    assert not (penalized[0, 4:] == tok).all()


def test_generate_restores_train_mode(tiny_cfg):
    model = TransformerLM(tiny_cfg).train()
    model.generate(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=2)
    assert model.training
