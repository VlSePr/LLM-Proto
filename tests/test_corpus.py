"""Fine-tuning corpus cleanup and the JSONL -> shards path."""
import json
import os

import pytest

from src.corpus import (
    build_finetune_corpus,
    clean_corpus_text,
    extract_text,
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
