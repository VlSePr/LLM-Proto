import pytest
import torch

from src.config import TrainConfig
from src.generate import (
    ChatSession,
    build_gradio_chat_demo,
    chat_widget,
    clean_generated_text,
    generate_ids,
    generate_text,
    gradio_chat_clear,
    gradio_chat_send,
    load_model_for_inference,
)
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


def test_chat_widget_drives_session(tiny_cfg, tokenizer, seed):
    pytest.importorskip("ipywidgets")
    model = TransformerLM(tiny_cfg).eval()
    session = ChatSession(model, tokenizer, max_new_tokens=8, temperature=0.0)
    ui = chat_widget(session)
    textarea, buttons = ui.children[1], ui.children[2]
    send_btn, clear_btn = buttons.children
    textarea.value = "hello"
    send_btn.click()
    assert len(session.turns) == 1 and textarea.value == ""
    clear_btn.click()
    assert session.turns == [] and session.history == []


def test_gradio_chat_send_isolates_visitors(tiny_cfg, tokenizer, seed):
    pytest.importorskip("gradio")
    model = TransformerLM(tiny_cfg).eval()

    session_a, history_a, cleared_a = gradio_chat_send(
        None, "hello from visitor A", [], model, tokenizer, 8, 0.0, 50, 0.9, 1.0)
    session_b, history_b, cleared_b = gradio_chat_send(
        None, "hello from visitor B", [], model, tokenizer, 8, 0.0, 50, 0.9, 1.0)

    assert session_a is not session_b
    assert cleared_a == "" and cleared_b == ""
    assert session_a.turns == [("hello from visitor A", session_a.turns[0][1])]
    assert session_b.turns == [("hello from visitor B", session_b.turns[0][1])]
    assert history_a[0] == {"role": "user", "content": "hello from visitor A"}
    assert history_b[0] == {"role": "user", "content": "hello from visitor B"}
    # Mutating A's history must never leak into B's -- independent conversation state per visitor.
    session_a.history.append(999)
    assert 999 not in session_b.history


def test_gradio_chat_send_accumulates_within_one_session(tiny_cfg, tokenizer, seed):
    pytest.importorskip("gradio")
    model = TransformerLM(tiny_cfg).eval()

    session, history, cleared = gradio_chat_send(None, "first turn", [], model, tokenizer, 8, 0.0, 50, 0.9, 1.0)
    assert cleared == "" and len(history) == 2
    assert history[0] == {"role": "user", "content": "first turn"}
    assert history[1]["role"] == "assistant"

    session, history, cleared = gradio_chat_send(
        session, "second turn", history, model, tokenizer, 8, 0.0, 50, 0.9, 1.0)
    assert len(history) == 4 and len(session.turns) == 2


def test_gradio_chat_clear_resets_only_that_session(tiny_cfg, tokenizer, seed):
    pytest.importorskip("gradio")
    model = TransformerLM(tiny_cfg).eval()

    session_a, _, _ = gradio_chat_send(None, "hi from A", [], model, tokenizer, 8, 0.0, 50, 0.9, 1.0)
    session_b, _, _ = gradio_chat_send(None, "hi from B", [], model, tokenizer, 8, 0.0, 50, 0.9, 1.0)

    cleared_state, cleared_history = gradio_chat_clear(session_a)
    assert cleared_state is session_a
    assert cleared_state.turns == [] and cleared_state.history == []
    assert cleared_history == []
    assert session_b.turns != [] and session_b.history != []   # untouched


def test_build_gradio_chat_demo_returns_blocks_with_queue(tiny_cfg, tokenizer):
    gr = pytest.importorskip("gradio")
    model = TransformerLM(tiny_cfg).eval()

    demo = build_gradio_chat_demo(model, tokenizer, concurrency_limit=2, queue_max_size=200)
    assert isinstance(demo, gr.Blocks)
    assert demo._queue.max_size == 200
    demo.close()
