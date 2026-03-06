#!/usr/bin/env python3
"""
Train BPE tokenizer and save to tokenizer_data/.
Run once, then commit the output to the repository.

Usage:
    python scripts/train_tokenizer.py                     # uses configs/data.yaml
    python scripts/train_tokenizer.py --config my.yaml    # custom config
    python scripts/train_tokenizer.py --force              # retrain even if exists
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from src.tokenizer import LLMTokenizer
from src.data import iter_texts_from_sources


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--config", default="configs/data.yaml", help="Data config YAML")
    parser.add_argument("--force", action="store_true", help="Retrain even if tokenizer exists")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    tok_cfg = cfg["tokenizer"]
    save_path = tok_cfg["save_path"]
    vocab_size = tok_cfg["vocab_size"]
    num_samples = tok_cfg.get("num_samples", 50_000)

    if os.path.exists(os.path.join(save_path, "tokenizer.json")) and not args.force:
        print(f"Tokenizer already exists at {save_path}/tokenizer.json")
        print("Use --force to retrain.")
        return

    sources = cfg.get("sources", [])

    print(f"Training BPE tokenizer (vocab={vocab_size:,}, samples≤{num_samples:,})")
    print(f"Sources: {len(sources)}")

    def text_iterator():
        count = 0
        for text in iter_texts_from_sources(sources):
            if count >= num_samples:
                break
            if text and len(text) > 50:
                yield text
                count += 1
        print(f"  Used {count:,} text samples for tokenizer training")

    tokenizer = LLMTokenizer.train(
        texts=text_iterator(),
        vocab_size=vocab_size,
        save_path=save_path,
    )

    print(f"\nTokenizer saved to {save_path}/")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Quick test
    test_text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    print(f"Test: '{test_text}' → {len(ids)} tokens → '{decoded}'")


if __name__ == "__main__":
    main()
