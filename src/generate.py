"""
Interactive text generation / inference CLI.
Supports single-prompt and multi-turn chat modes.
Uses KV-cache for efficient autoregressive generation.
"""

import re
import torch
from typing import List, Optional, Tuple

from .model import TransformerLM
from .tokenizer import LLMTokenizer
from .utils import get_device, build_model_from_checkpoint


# Safety net for special-token *text* the model may have learned to emit verbatim
# (e.g. "<|eot_id|>" from scraped chat transcripts). Real special tokens are already
# dropped by tokenizer.decode(skip_special=True); this only matches the <|name|> shape,
# so ordinary angle brackets, pipes and code stay intact.
_SPECIAL_TOKEN_TEXT = re.compile(r"<\|[A-Za-z0-9_]+\|>")


def clean_generated_text(text: str) -> str:
    """Remove literal special-token markup and trim surrounding whitespace."""
    return _SPECIAL_TOKEN_TEXT.sub("", text).strip()


def load_model_for_inference(
    checkpoint_path: str,
    model_config_name: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TransformerLM, LLMTokenizer]:
    """Load model and tokenizer for inference. Returns (model, tokenizer).

    The architecture comes from the checkpoint's ``model_config``; ``model_config_name``
    is only used for old checkpoints that lack it. The checkpoint is read once.
    """
    # Backward compatibility: the old signature was (model_config_name, checkpoint_path).
    if model_config_name and model_config_name.endswith(".pt") and not checkpoint_path.endswith(".pt"):
        checkpoint_path, model_config_name = model_config_name, checkpoint_path

    if device is None:
        device = get_device()

    model, ckpt = build_model_from_checkpoint(checkpoint_path, device, model_config_name)
    tokenizer_path = (ckpt.get("train_config") or {}).get("tokenizer_path", "tokenizer_data")
    tokenizer = LLMTokenizer(tokenizer_path)
    return model, tokenizer


def generate_ids(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    prompt_ids: List[int],
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
    repetition_penalty: float = 1.0,
) -> List[int]:
    """Generate continuation token IDs for ``prompt_ids`` (prompt not included)."""
    if device is None:
        device = next(model.parameters()).device
    max_seq_len = model.config.max_seq_len
    # Keep room for the requested continuation by dropping the oldest context.
    keep = max(1, max_seq_len - max_new_tokens)
    if len(prompt_ids) > keep:
        prompt_ids = prompt_ids[-keep:]
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        # model.generate() uses KV-cache internally: the full prompt is processed
        # in one pass (prefill), then each new token only computes attention over
        # cached K/V plus the single new query — O(T) per step instead of O(T²).
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_id,
            repetition_penalty=repetition_penalty,
        )
    return output_ids[0, len(prompt_ids):].tolist()


def generate_text(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
    repetition_penalty: float = 1.0,
    include_prompt: bool = False,
) -> str:
    """Generate text from a prompt. Returns only the continuation unless ``include_prompt``."""
    # Encode with BOS so the model sees a clean sequence start signal.
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    new_ids = generate_ids(
        model, tokenizer, prompt_ids,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty, device=device,
    )
    text = clean_generated_text(tokenizer.decode(new_ids, skip_special=True))
    if include_prompt:
        return prompt + text
    return text


def interactive_chat(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
):
    """Interactive multi-turn chat loop in the terminal.

    The conversation so far (prompts and replies) is kept as token IDs and fed back
    as context on every turn, truncated from the left to fit the model's context
    window. ``clear`` resets the history.
    """
    device = next(model.parameters()).device
    history: List[int] = []

    print("=" * 60)
    print("Interactive Chat (type 'quit' to exit, 'clear' to reset)")
    print(f"  Model: {model.config.dim}d, {model.config.n_layers}L, "
          f"{model.count_parameters() / 1e6:.1f}M params")
    print(f"  Temp: {temperature}, Top-k: {top_k}, Top-p: {top_p}, Rep-penalty: {repetition_penalty}")
    print("  Commands: /temp X  /topk N  /topp X  /rep X")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("Goodbye!")
            break
        if prompt.lower() == "clear":
            history = []
            print("Chat cleared.")
            continue

        # Slash-commands let users adjust sampling parameters at runtime without
        # restarting the session.
        if prompt.startswith("/"):
            parts = prompt.split()
            try:
                if parts[0] == "/temp":
                    temperature = float(parts[1]); print(f"Temperature set to {temperature}")
                elif parts[0] == "/topk":
                    top_k = int(parts[1]); print(f"Top-k set to {top_k}")
                elif parts[0] == "/topp":
                    top_p = float(parts[1]); print(f"Top-p set to {top_p}")
                elif parts[0] == "/rep":
                    repetition_penalty = float(parts[1]); print(f"Repetition penalty set to {repetition_penalty}")
                else:
                    print("Unknown command. Use /temp, /topk, /topp, /rep")
            except (IndexError, ValueError):
                print(f"Usage: {parts[0]} <value>")
            continue

        # Context = BOS + conversation so far + new prompt (+ a newline separator).
        turn_ids = tokenizer.encode(prompt + "\n", add_bos=False)
        context_ids = [tokenizer.bos_id] + history + turn_ids
        new_ids = generate_ids(
            model, tokenizer, context_ids,
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, device=device,
        )
        history = history + turn_ids + new_ids
        # Keep the rolling history bounded to the model's context window.
        max_history = max(1, model.config.max_seq_len - max_new_tokens - 1)
        if len(history) > max_history:
            history = history[-max_history:]

        response = clean_generated_text(tokenizer.decode(new_ids, skip_special=True))
        print(f"\nModel: {response}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with trained LLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model config name or YAML path (only needed for checkpoints without model_config)")
    parser.add_argument("--prompt", type=str, default="", help="Single prompt (omit for interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    model, tokenizer = load_model_for_inference(args.checkpoint, args.model)

    if args.prompt:
        # Single generation
        text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(text)
    else:
        # Interactive mode
        interactive_chat(
            model, tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )


if __name__ == "__main__":
    main()
