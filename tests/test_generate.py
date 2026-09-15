import torch

from src.config import TrainConfig
from src.generate import ChatSession, clean_generated_text, generate_ids, generate_text, load_model_for_inference
from src.model import TransformerLM
from src.utils import save_checkpoint


def test_clean_generated_text_only_strips_special_markup():
    s = "keep <b>html</b>, a < b > c, x|y> and <|eot_id|> gone <|im_end|>\n  indent kept"
    out = clean_generated_text(s)
    assert "<|" not in out
    assert "<b>html</b>" in out and "a < b > c" in out and "x|y>" in out
    assert "  indent kept" in out


def test_generate_text_returns_continuation_only(tiny_cfg, tokenizer, seed):
    model = TransformerLM(tiny_cfg).eval()
    prompt = "the quick brown"
    text = generate_text(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0)
    assert "<|" not in text
    with_prompt = generate_text(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0, include_prompt=True)
    assert with_prompt.startswith(prompt)


def test_generate_ids_truncates_long_context(tiny_cfg, tokenizer, seed):
    model = TransformerLM(tiny_cfg).eval()
    prompt_ids = list(range(3, 3 + tiny_cfg.max_seq_len + 20))
    new_ids = generate_ids(model, tokenizer, prompt_ids, max_new_tokens=8, temperature=0.0)
    assert 0 < len(new_ids) <= 8


def test_load_model_for_inference_reads_config_from_checkpoint(tiny_cfg, tokenizer, tmp_tokenizer_dir, tmp_path):
    model = TransformerLM(tiny_cfg)
    tcfg = TrainConfig(use_wandb=False, tokenizer_path=tmp_tokenizer_dir, checkpoint_dir=str(tmp_path))
    path = save_checkpoint(model, None, 3, None, tiny_cfg, tcfg, tcfg.checkpoint_dir)
    loaded, tok = load_model_for_inference(path, device=torch.device("cpu"))
    assert loaded.config == tiny_cfg
    assert tok.vocab_size == tokenizer.vocab_size
    # legacy argument order still works
    loaded2, _ = load_model_for_inference("tiny", path, device=torch.device("cpu"))
    assert loaded2.config == tiny_cfg


def test_chat_session_keeps_bounded_history(tiny_cfg, tokenizer, seed):
    model = TransformerLM(tiny_cfg).eval()
    session = ChatSession(model, tokenizer, max_new_tokens=8, temperature=0.0)
    assert session.max_history == tiny_cfg.max_seq_len - 8 - 1
    first = session.reply("hello there")
    assert isinstance(first, str) and "<|" not in first
    assert session.turns == [("hello there", first)]
    n1 = len(session.history)
    assert 0 < n1 <= session.max_history
    for _ in range(6):
        session.reply("and then what happened next in the story")
    assert len(session.history) <= session.max_history
    assert len(session.turns) == 7
    session.clear()
    assert session.history == [] and session.turns == []
