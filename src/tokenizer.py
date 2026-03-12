"""
BPE Tokenizer: training and inference wrapper.
Uses HuggingFace tokenizers library (fast, Rust-backed).
Optimized for English text with byte-fallback.
"""

import os
from typing import List, Optional

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


# Special tokens used by the model:
# - bos/eos: mark sequence boundaries (critical for generation to know when to stop)
# - pad: used to align variable-length sequences in a batch
# - im_start/im_end: chat template markers (for fine-tuning into a chat model later)
SPECIAL_TOKENS = {
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "pad": "<|pad|>",
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}

SPECIAL_TOKEN_LIST = list(SPECIAL_TOKENS.values())


class LLMTokenizer:
    """Wrapper around HuggingFace tokenizer with training and encode/decode methods."""

    def __init__(self, tokenizer_path: Optional[str] = None):
        self.tokenizer: Optional[Tokenizer] = None
        if tokenizer_path and os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
            self.load(tokenizer_path)

    @staticmethod
    def train(
        texts,
        vocab_size: int = 32_000,
        save_path: str = "tokenizer_data",
        min_frequency: int = 2,
    ) -> "LLMTokenizer":
        """
        Train a BPE tokenizer on an iterator of texts.

        Args:
            texts: Iterator/generator of strings
            vocab_size: Vocabulary size
            save_path: Directory to save tokenizer files
            min_frequency: Minimum frequency for BPE merges

        Returns:
            Trained LLMTokenizer instance
        """
        # BPE with byte_fallback=True: when the tokenizer encounters an unknown character
        # (e.g., rare Unicode), it falls back to raw byte tokens (0x00–0xFF) instead of <UNK>.
        # This guarantees the tokenizer can represent *any* input — no information is ever lost.
        tokenizer = Tokenizer(models.BPE(byte_fallback=True))

        # ByteLevel pre-tokenization: splits text into bytes before BPE merges.
        # This is the GPT-2/GPT-4 approach — the tokenizer operates on UTF-8 byte sequences,
        # making it language-agnostic (works for any script without special handling).
        # add_prefix_space=False: don't add a leading space (we handle whitespace naturally).
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # ByteLevel decoder reverses the byte-level encoding back to readable text
        tokenizer.decoder = decoders.ByteLevel()

        # Post-processing: add BOS token at start
        # (can be customized later for chat templates)

        # BPE trainer: iteratively merges the most frequent byte/token pairs.
        # min_frequency=2: a merge must occur at least twice to be learned,
        #   preventing overfitting to typos or single-occurrence strings.
        # initial_alphabet=ByteLevel.alphabet(): start with all 256 byte tokens
        #   as the base vocabulary, ensuring full Unicode coverage.
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKEN_LIST,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # Train
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Save
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save(os.path.join(save_path, "tokenizer.json"))

        # Return wrapped instance
        wrapper = LLMTokenizer()
        wrapper.tokenizer = tokenizer
        wrapper._setup_special_token_ids()
        return wrapper

    def load(self, path: str):
        """Load a trained tokenizer from directory."""
        self.tokenizer = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self._setup_special_token_ids()

    def _setup_special_token_ids(self):
        """Cache special token IDs for fast access."""
        self.bos_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["bos"])
        self.eos_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["eos"])
        self.pad_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["pad"])
        self.im_start_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["im_start"])
        self.im_end_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["im_end"])

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        ids = self.tokenizer.encode(text).ids
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special:
            special_ids = {self.bos_id, self.eos_id, self.pad_id, self.im_start_id, self.im_end_id}
            ids = [i for i in ids if i not in special_ids]
        return self.tokenizer.decode(ids)

    def encode_batch(self, texts: List[str], add_bos: bool = True, add_eos: bool = False) -> List[List[int]]:
        """Batch encode multiple texts. Uses the Rust-backed parallel encoder for speed."""
        results = self.tokenizer.encode_batch(texts)
        encoded = [enc.ids for enc in results]
        if add_bos and self.bos_id is not None:
            encoded = [[self.bos_id] + ids for ids in encoded]
        if add_eos and self.eos_id is not None:
            encoded = [ids + [self.eos_id] for ids in encoded]
        return encoded

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def id_to_token(self, id: int) -> Optional[str]:
        return self.tokenizer.id_to_token(id)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token)


def train_tokenizer_from_dataset(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_subset: str = "sample-10BT",
    vocab_size: int = 32_000,
    save_path: str = "tokenizer_data",
    num_samples: int = 500_000,
):
    """
    Train a BPE tokenizer from a HuggingFace dataset (streaming).

    Args:
        dataset_name: HuggingFace dataset name
        dataset_subset: Dataset subset/config
        vocab_size: Target vocabulary size
        save_path: Where to save tokenizer files
        num_samples: Number of text samples to use for training
    """
    from datasets import load_dataset
    from itertools import islice

    print(f"Loading dataset {dataset_name}/{dataset_subset} (streaming)...")
    ds = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)

    def text_iterator():
        for sample in islice(ds, num_samples):
            text = sample.get("text", "")
            if text and len(text) > 50:  # Skip very short texts
                yield text

    print(f"Training BPE tokenizer (vocab_size={vocab_size}, samples={num_samples})...")
    tokenizer = LLMTokenizer.train(
        texts=text_iterator(),
        vocab_size=vocab_size,
        save_path=save_path,
    )

    print(f"Tokenizer saved to {save_path}/")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Quick test
    test_text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    print(f"Test encode: '{test_text}' -> {len(ids)} tokens")
    print(f"Test decode: '{decoded}'")

    return tokenizer
