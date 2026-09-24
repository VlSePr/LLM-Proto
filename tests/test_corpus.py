"""Fine-tuning corpus cleanup and the JSONL -> shards path."""
import json
import os

import pytest

from src.corpus import (
    build_finetune_corpus,
    clean_corpus_text,
    collect_hf_texts,
    extract_text,
    format_chat_pair,
    normalize_whitespace,
    strip_special_token_text,
    write_jsonl,
)
from src.data import _iter_jsonl, read_manifest


def test_strip_special_token_text_keeps_everything_else():
    s = "keep <b>html</b>, a < b > c, x|y> and <|eot_id|> gone <|im_end|> <|end-of-text|>"
    out = strip_special_token_text(s)
    assert "<|" not in out and "<b>html</b>" in out and "a < b > c" in out and "x|y>" in out


def test_clean_corpus_text_llama3_keeps_assistant_span():
    s = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an NPC.<|eot_id|>"
         "<|start_header_id|>user<|end_header_id|>Who are you?<|eot_id|>"
         "<|start_header_id|>assistant<|end_header_id|>\n\nI am the Firekeeper.<|eot_id|>")
    assert clean_corpus_text(s) == "I am the Firekeeper."


def test_clean_corpus_text_strips_system_block_and_tags():
    s = "<s>Act as an NPC from Dark Souls.</s>  The Abyss   swallows <br> all.</s>\n\n\n\nLight fades."
    out = clean_corpus_text(s)
    assert out == "The Abyss swallows all.\n\nLight fades."


def test_clean_corpus_text_preserves_inequalities_and_paragraphs():
    s = "if a < b and c > d then\tx<y holds.\n\nSecond paragraph <|im_end|> here."
    out = clean_corpus_text(s)
    assert "a < b and c > d" in out and "x<y" in out
    assert "\n\n" in out and "<|" not in out
    kept = clean_corpus_text("line <br> here", strip_bare_tags=False)
    assert "<br>" in kept


def test_normalize_whitespace():
    assert normalize_whitespace("a  \t b \n  c\n\n\n\nd ") == "a b\nc\n\nd"


def test_extract_text_priority_and_fields():
    row = {"id": 1, "text": "short", "description": "a description that is long enough to count"}
    assert extract_text(row) == row["description"]
    assert extract_text({"a": "x", "b": 2, "c": "y"}) == "x y"
    assert extract_text({"instruction": "Q?", "output": "A.", "junk": "no"}, ["instruction", "output"]) == "Q?\nA."


def test_format_chat_pair_wraps_in_chatml_turns():
    out = format_chat_pair("Who are you?", "I am the Firekeeper.")
    assert out == "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\nI am the Firekeeper.<|im_end|>"


def test_collect_hf_texts_chat_roles_cleans_before_wrapping(monkeypatch):
    rows = [
        {"instruction": "<s>Act as an NPC.</s>Who are you, traveler of the Abyss?",
         "output": "<|start_header_id|>assistant<|end_header_id|>I am the Firekeeper of legend.<|eot_id|>"},
        {"instruction": "tiny", "output": "also tiny"},  # dropped: below min_chars after cleaning
    ]

    def fake_load_dataset(name, subset, split):
        return rows

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    texts = collect_hf_texts(["fake/dataset"], chat_roles=("instruction", "output"))
    assert len(texts) == 1
    out = texts[0]
    # Cleaning ran before wrapping: the Llama-3/system-block markup is gone, but the
    # <|im_start|>/<|im_end|> markers this function just added are intact.
    assert out.startswith("<|im_start|>user\n") and out.endswith("<|im_end|>")
    assert "<s>" not in out and "<|start_header_id|>" not in out and "<|eot_id|>" not in out
    assert "Who are you, traveler of the Abyss?" in out
    assert "I am the Firekeeper of legend." in out


def test_build_finetune_corpus_chat_roles_survives_cleanup_pass(monkeypatch, tmp_path, tmp_tokenizer_dir):
    rows = [{"instruction": f"Question number {i} about the ruins?",
             "output": f"Answer number {i} about the ruins, told at length."} for i in range(8)]

    def fake_load_dataset(name, subset, split):
        return rows

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    bins = build_finetune_corpus(
        ["fake/dataset"], jsonl_path=str(tmp_path / "jsonl" / "chat.jsonl"), tokenizer_path=tmp_tokenizer_dir,
        bins_dir=str(tmp_path / "chat_bins"), shard_size=512, val_every=2, preview=0,
        chat_roles=("instruction", "output"),
    )
    manifest = read_manifest(bins)
    assert manifest and manifest["n_train_shards"] >= 1
    with open(tmp_path / "jsonl" / "chat.jsonl", encoding="utf-8") as f:
        written = [json.loads(line)["text"] for line in f]
    assert len(written) == 8
    assert all(r.startswith("<|im_start|>user\n") and r.endswith("<|im_end|>") for r in written)
    assert all("<|im_start|>assistant\n" in r for r in written)


def test_write_jsonl_round_trips_through_data_pipeline(tmp_path):
    texts = ["x" * 60, "y" * 70]
    path = str(tmp_path / "c.jsonl")
    assert write_jsonl(texts, path) == 2
    with open(path, encoding="utf-8") as f:
        assert [json.loads(line)["text"] for line in f] == texts
    assert list(_iter_jsonl({"type": "jsonl", "path": path})) == texts


def test_build_finetune_corpus_end_to_end(tmp_path, tmp_tokenizer_dir):
    raw = [
        "<s>Act as an NPC.</s>" + "The Abyss swallows all who wander too far from the flame. " * 4,
        "<|start_header_id|>assistant<|end_header_id|>" + "Lordran lies in ruin, its lords long faded. " * 4
        + "<|eot_id|>",
        "tiny",  # dropped by the min_chars filter
    ] * 4
    bins = build_finetune_corpus(
        texts=raw, jsonl_path=str(tmp_path / "jsonl" / "c.jsonl"), tokenizer_path=tmp_tokenizer_dir,
        bins_dir=str(tmp_path / "bins"), shard_size=512, val_every=2, preview=2,
    )
    assert bins == str(tmp_path / "bins")
    manifest = read_manifest(bins)
    assert manifest and manifest["n_train_shards"] >= 1
    assert os.path.exists(os.path.join(bins, "val.bin"))
    with open(tmp_path / "jsonl" / "c.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line)["text"] for line in f]
    assert len(rows) == 8 and all("<" not in r for r in rows)

    with pytest.raises(ValueError, match="empty"):
        build_finetune_corpus(texts=["tiny"], jsonl_path=str(tmp_path / "e.jsonl"),
                              tokenizer_path=tmp_tokenizer_dir, bins_dir=str(tmp_path / "e"))
