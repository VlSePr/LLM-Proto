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

from src.data import load_data_config
from src.tokenizer import train_tokenizer_from_sources


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--config", default="configs/data.yaml", help="Data config YAML")
    parser.add_argument("--force", action="store_true", help="Retrain even if tokenizer exists")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    tok_cfg = cfg["tokenizer"]
    save_path = tok_cfg["save_path"]

    if os.path.exists(os.path.join(save_path, "tokenizer.json")) and not args.force:
        print(f"Tokenizer already exists at {save_path}/tokenizer.json")
        print("Use --force to retrain.")
        return

    train_tokenizer_from_sources(
        cfg.get("sources", []),
        vocab_size=tok_cfg["vocab_size"],
        save_path=save_path,
        num_samples=tok_cfg.get("num_samples", 50_000),
    )


if __name__ == "__main__":
    main()
