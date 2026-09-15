"""Tokenizer training from data.yaml sources and the local -> Drive -> train resolution."""
import os

import pytest

from src.tokenizer import LLMTokenizer, ensure_tokenizer, train_tokenizer_from_sources

CORPUS = os.path.join(os.path.dirname(__file__), "fixtures", "corpus")
SOURCES = [{"type": "text_dir", "path": CORPUS}]


def test_train_tokenizer_from_sources_respects_num_samples(tmp_path, capsys):
    tok = train_tokenizer_from_sources(SOURCES, vocab_size=300, save_path=str(tmp_path / "tok"), num_samples=3)
    assert os.path.isfile(tmp_path / "tok" / "tokenizer.json")
    assert tok.vocab_size <= 300 and tok.bos_id is not None
    assert "Used 3 text samples" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no data sources"):
        train_tokenizer_from_sources([], save_path=str(tmp_path / "none"))


def test_ensure_tokenizer_prefers_local_then_trains(tmp_path, tmp_tokenizer_dir):
    # local hit: nothing is written
    before = os.path.getmtime(os.path.join(tmp_tokenizer_dir, "tokenizer.json"))
    tok = ensure_tokenizer(tmp_tokenizer_dir, SOURCES, vocab_size=300)
    assert isinstance(tok, LLMTokenizer)
    assert os.path.getmtime(os.path.join(tmp_tokenizer_dir, "tokenizer.json")) == before

    # missing + no sources -> error; missing + sources -> trained in place
    with pytest.raises(FileNotFoundError):
        ensure_tokenizer(str(tmp_path / "missing"))
    tok = ensure_tokenizer(str(tmp_path / "new"), SOURCES, vocab_size=300, num_samples=4)
    assert os.path.isfile(tmp_path / "new" / "tokenizer.json") and tok.vocab_size <= 300

    # force retrains even when the file exists
    ensure_tokenizer(str(tmp_path / "new"), SOURCES, vocab_size=280, num_samples=4, force=True)
    assert LLMTokenizer(str(tmp_path / "new")).vocab_size <= 280
