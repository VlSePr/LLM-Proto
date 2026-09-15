"""Shared fixtures: a tiny model config, small token shards, and a tiny tokenizer."""
import os
import sys

import numpy as np
import pytest
import torch

# Make `src` importable when pytest is run from the repo root or from tests/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import ModelConfig  # noqa: E402
from src.tokenizer import LLMTokenizer  # noqa: E402

TINY_VOCAB = 512


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=TINY_VOCAB, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)


@pytest.fixture
def seed():
    torch.manual_seed(0)
    np.random.seed(0)
    return 0


def write_shards(data_dir: str, sizes=(3000, 1500), val_size=700, vocab: int = TINY_VOCAB, seed: int = 0):
    """Write random uint16 token shards (train_0000.bin, ...) plus val.bin."""
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i, n in enumerate(sizes):
        p = os.path.join(data_dir, f"train_{i:04d}.bin")
        rng.integers(0, vocab, size=n, dtype=np.uint16).tofile(p)
        paths.append(p)
    rng.integers(0, vocab, size=val_size, dtype=np.uint16).tofile(os.path.join(data_dir, "val.bin"))
    return paths


@pytest.fixture
def tmp_data(tmp_path):
    data_dir = str(tmp_path / "data")
    write_shards(data_dir)
    return data_dir


@pytest.fixture(scope="session")
def tmp_tokenizer_dir(tmp_path_factory):
    """Train a tiny BPE tokenizer once per session."""
    path = str(tmp_path_factory.mktemp("tok"))
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "hello world, this is a tiny corpus for a tiny tokenizer",
        "language models predict the next token given the previous tokens",
    ] * 50
    LLMTokenizer.train(corpus, vocab_size=TINY_VOCAB, save_path=path)
    return path


@pytest.fixture
def tokenizer(tmp_tokenizer_dir) -> LLMTokenizer:
    return LLMTokenizer(tmp_tokenizer_dir)
