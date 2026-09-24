"""
Fine-tuning corpus preparation: collect texts (e.g. from HuggingFace datasets),
clean chat-template markup out of them, write a JSONL file and tokenize it into
shards with the regular data pipeline.

    from src.corpus import build_finetune_corpus
    bins_dir = build_finetune_corpus(
        ["SantaBot/dark-souls-lore", {"name": "ArenaRune/EldenRingQA", "fields": ["instruction", "output"]}],
        jsonl_path="data/custom/jsonl/my_corpus.jsonl",
        tokenizer_path="tokenizer_data",
        bins_dir="data/custom/bins",
    )

Pass ``chat_roles=("instruction", "output")`` to wrap each row in ``<|im_start|>``/
``<|im_end|>`` ChatML turns instead of a flat field join, so a checkpoint fine-tuned on the
result actually learns the same turn-boundary markers ``generate.ChatSession`` builds its
prompts with:

    build_finetune_corpus(
        [{"name": "ArenaRune/EldenRingQA", "fields": ["instruction", "output"]}],
        jsonl_path="data/custom/jsonl/chat_corpus.jsonl",
        tokenizer_path="tokenizer_data",
        bins_dir="data/custom/bins",
        chat_roles=("instruction", "output"),
    )

Two different cleaners live in this project on purpose:

* ``strip_special_token_text`` removes only ``<|name|>`` markup. ``generate.clean_generated_text``
  uses it on model output, where ordinary angle brackets and code must survive.
* ``clean_corpus_text`` is for *training data* scraped from chat transcripts: it keeps
  only the assistant span of Llama-3 style prompts, drops ``<s>...</s>`` system blocks,
  ``<|...|>`` tokens and tag-shaped leftovers such as ``<s>``/``<br>``, and normalises
  whitespace. It never touches inequalities like ``a < b`` because the tag pattern
  requires a letter right after ``<`` and no spaces.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Iterable, Sequence
from typing import Any

# Chat-template special tokens written out as text, e.g. <|eot_id|>, <|im_end|>.
SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z0-9_.\-]+\|>")
# Tag-shaped leftovers: <s>, </s>, <br>, <INST>, <br/>. A letter must follow "<" and there
# are no spaces inside, so "a < b > c" and "x<y" are left alone.
BARE_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9_:\-]{0,30}\s*/?>")
# Llama-3 chat format: keep only what the assistant said.
LLAMA3_ASSISTANT_RE = re.compile(
    r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*?)(?:<\|eot_id\|>|$)", re.DOTALL,
)
# "<s>Act as an NPC ...</s>" system-prompt wrappers.
SYSTEM_BLOCK_RE = re.compile(r"<s>[^<]*?</s>\s*")

DEFAULT_TEXT_FIELDS = ("text", "description", "lore", "content", "body")


def strip_special_token_text(text: str) -> str:
    """Remove ``<|name|>`` markup only; everything else, including bare tags, is kept."""
    return SPECIAL_TOKEN_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, trim line ends, and cap blank lines at one (paragraphs survive)."""
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_corpus_text(text: str, *, strip_bare_tags: bool = True, collapse_whitespace: bool = True) -> str:
    """Clean one training document scraped from chat-formatted data (see module docstring)."""
    m = LLAMA3_ASSISTANT_RE.search(text)
    if m:
        text = m.group(1)
    text = SYSTEM_BLOCK_RE.sub("", text)
    text = strip_special_token_text(text)
    if strip_bare_tags:
        text = BARE_TAG_RE.sub("", text)
    if collapse_whitespace:
        text = normalize_whitespace(text)
    return text.strip()


def format_chat_pair(user_text: str, assistant_text: str) -> str:
    """Wrap one instruction/output pair in ChatML turns matching ``generate.ChatSession``.

    Call this on already-``clean_corpus_text``-cleaned text: ``clean_corpus_text`` strips
    any ``<|name|>`` markup (including these markers), so cleaning must happen *before*
    wrapping, never after.
    """
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{assistant_text}<|im_end|>"


def extract_text(row: dict[str, Any], fields: Sequence[str] | None = None, *, join: str = "\n",
                 min_chars: int = 20) -> str:
    """Pull the training text out of one dataset row.

    With ``fields`` the named string columns are joined with ``join`` (QA datasets:
    ``["instruction", "output"]``). Otherwise the first of ``DEFAULT_TEXT_FIELDS`` longer
    than ``min_chars`` wins, falling back to every string column joined by a space.
    """
    if fields:
        parts = [str(row.get(f, "")).strip() for f in fields]
        return join.join(p for p in parts if p)
    for key in DEFAULT_TEXT_FIELDS:
        v = row.get(key, "")
        if isinstance(v, str) and len(v) > min_chars:
            return v
    return " ".join(str(v) for v in row.values() if isinstance(v, str))


def collect_hf_texts(datasets: Iterable[str | dict[str, Any]], *, split: str = "train",
                     min_chars: int = 20, chat_roles: tuple[str, str] | None = None) -> list[str]:
    """Download HuggingFace datasets and return their texts (best effort per dataset).

    Each entry is a dataset id (``"SantaBot/dark-souls-lore"``) or a dict with ``name`` and
    optional ``subset``, ``split`` and ``fields`` (see ``extract_text``). A dataset that
    fails to load is reported and skipped so one bad id does not abort the corpus.

    With ``chat_roles=(user_field, assistant_field)``, each row's two named columns are
    cleaned individually and wrapped with ``format_chat_pair`` instead of joined by
    ``extract_text`` -- rows missing either field (after cleaning) are dropped. This
    replaces, rather than composes with, per-entry ``fields``.
    """
    from datasets import load_dataset

    texts: list[str] = []
    for entry in datasets:
        spec = {"name": entry} if isinstance(entry, str) else dict(entry)
        name = spec["name"]
        print(f"Loading {name} ...")
        try:
            ds = load_dataset(name, spec.get("subset"), split=spec.get("split", split))
            if chat_roles:
                user_field, assistant_field = chat_roles
                rows = []
                for r in ds:
                    user_text = clean_corpus_text(str(r.get(user_field, "")).strip())
                    assistant_text = clean_corpus_text(str(r.get(assistant_field, "")).strip())
                    if len(user_text) > min_chars and len(assistant_text) > min_chars:
                        rows.append(format_chat_pair(user_text, assistant_text))
            else:
                rows = [extract_text(r, spec.get("fields"), min_chars=min_chars) for r in ds]
                rows = [t.strip() for t in rows if t and len(t.strip()) > min_chars]
            texts.extend(rows)
            print(f"  collected {len(rows):,} samples (total {len(texts):,})")
        except Exception as e:
            print(f"  FAILED {name}: {e}")
    return texts


def write_jsonl(texts: Iterable[str], path: str, text_field: str = "text") -> int:
    """Write one ``{text_field: text}`` object per line; returns the number written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps({text_field: t}, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_finetune_corpus(
    datasets: Iterable[str | dict[str, Any]] = (),
    *,
    jsonl_path: str,
    tokenizer_path: str,
    bins_dir: str,
    texts: Iterable[str] | None = None,
    clean: bool = True,
    min_chars: int = 20,
    shard_size: int = 10_000_000,
    val_every: int = 50,
    preview: int = 8,
    force: bool = False,
    seed: int = 0,
    chat_roles: tuple[str, str] | None = None,
) -> str:
    """Collect, clean, write JSONL and tokenize a fine-tuning corpus; returns ``bins_dir``.

    ``texts`` bypasses the HuggingFace download (tests, local corpora) -- and, with it,
    ``chat_roles``, since that option needs per-row dataset dict access. Tokenization goes
    through ``data.ensure_tokenized_data``, so a manifest is written and an unchanged
    corpus is a cache hit. Note the shard builder skips documents shorter than 50
    characters, the same rule as every other source.

    ``chat_roles=(user_field, assistant_field)`` wraps each row in ChatML turns (see
    ``format_chat_pair``) instead of the plain field join. Those rows are already cleaned
    per-field by ``collect_hf_texts``, so the generic ``clean`` pass below -- which would
    strip the ``<|im_start|>``/``<|im_end|>`` markers it just added -- is skipped for them.
    """
    from .data import ensure_tokenized_data

    rows = (
        list(texts) if texts is not None
        else collect_hf_texts(datasets, min_chars=min_chars, chat_roles=chat_roles)
    )
    print(f"Total samples collected: {len(rows):,}")
    if clean and not chat_roles:
        rows = [clean_corpus_text(t) for t in rows]
        rows = [t for t in rows if len(t) > min_chars]
        print(f"After cleanup: {len(rows):,}")
    if not rows:
        raise ValueError("fine-tuning corpus is empty after collection/cleanup")

    if preview:
        rng = random.Random(seed)
        for i, t in enumerate(rng.sample(rows, min(preview, len(rows))), 1):
            snippet = t.replace("\n", " ")[:200]
            print(f"[{i}] {snippet}{'...' if len(t) > 200 else ''}")

    n = write_jsonl(rows, jsonl_path)
    print(f"Wrote {n:,} entries -> {jsonl_path}")
    return ensure_tokenized_data(
        tokenizer_path=tokenizer_path,
        output_dir=bins_dir,
        sources=[{"type": "jsonl", "path": jsonl_path, "text_field": "text"}],
        shard_size=shard_size,
        val_every=val_every,
        force=force,
    )
