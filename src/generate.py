"""
Interactive text generation / inference CLI.
Supports single-prompt and multi-turn chat modes.
Uses KV-cache for efficient autoregressive generation.
"""

import os
import re
import torch

from .config import ModelConfig, get_model_config, load_model_config
from .model import TransformerLM
from .tokenizer import LLMTokenizer
from .utils import get_device, get_dtype, load_checkpoint


def load_model_for_inference(
    model_config_name: str,
    checkpoint_path: str,
    device: torch.device = None,
) -> tuple:
    """Load model and tokenizer for inference. Returns (model, tokenizer)."""
    if device is None:
        device = get_device()

    # Load model config
    if os.path.isfile(model_config_name):
        model_config = load_model_config(model_config_name)
    else:
        model_config = get_model_config(model_config_name)

    # Build model
    model = TransformerLM(model_config).to(device)

    # Load checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    resume_name = os.path.basename(checkpoint_path).replace(".pt", "")
    load_checkpoint(checkpoint_dir, resume_name, model, device=device)

    # Load tokenizer
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer_path = ckpt.get("train_config", {}).get("tokenizer_path", "tokenizer_data")
    tokenizer = LLMTokenizer(tokenizer_path)

    model.eval()
    return model, tokenizer


def generate_text(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = None,
) -> str:
    """Generate text from a prompt."""
    if device is None:
        device = get_device()

    # Encode with BOS so the model sees a clean sequence start signal.
    input_ids = tokenizer.encode(prompt, add_bos=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
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
        )

    text = tokenizer.decode(output_ids[0].tolist())
    # Pass 1: fully-formed special tokens  <|end|>  <|start_header_id|>  <|>  etc.
    text = re.sub(r'<[^>]*>', '', text)
    # Pass 2: dangling opens without closing >  e.g. <|starter  <|e_
    text = re.sub(r'<\|[^\s>]*', '', text)
    # Pass 3: dangling closes — optionally preceded by a word fragment
    #   catches:  |end|>   e|end|>   assistant_header_head|end|>   assistant_id|end|end|>
    #   pattern:  optional word-chars  PIPE  one-or-more (word-chars or PIPE)  >
    text = re.sub(r'[a-zA-Z_0-9]*\|[a-zA-Z_0-9|]+>', '', text)
    # Pass 4: collapse runs of blank lines left by removals (3+ → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Pass 5: collapse multiple spaces
    text = re.sub(r' +', ' ', text).strip()
    return text



def interactive_chat(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """Interactive chat loop in the terminal."""
    device = next(model.parameters()).device

    print("=" * 60)
    print("Interactive Chat (type 'quit' to exit, 'clear' to reset)")
    print(f"  Model: {model.config.dim}d, {model.config.n_layers}L, "
          f"{model.count_parameters() / 1e6:.1f}M params")
    print(f"  Temp: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
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
            print("Chat cleared.")
            continue

        # Slash-commands let users adjust sampling parameters at runtime without
        # restarting the session. This is useful for exploring how temperature,
        # top-k, and top-p affect the model's creativity vs. coherence trade-off.
        if prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
            except (IndexError, ValueError):
                print("Usage: /temp <value>")
            continue
        if prompt.startswith("/topk "):
            try:
                top_k = int(prompt.split()[1])
                print(f"Top-k set to {top_k}")
            except (IndexError, ValueError):
                print("Usage: /topk <value>")
            continue
        if prompt.startswith("/topp "):
            try:
                top_p = float(prompt.split()[1])
                print(f"Top-p set to {top_p}")
            except (IndexError, ValueError):
                print("Usage: /topp <value>")
            continue

        response = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        # model.generate() returns the full sequence (prompt + completion).
        # Strip the prompt prefix so we only display the model's new tokens.
        if response.startswith(prompt):
            response = response[len(prompt):]
        print(f"\nModel: {response.strip()}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with trained LLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--model", type=str, default="tiny", help="Model config name or YAML path")
    parser.add_argument("--prompt", type=str, default="", help="Single prompt (omit for interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    model, tokenizer = load_model_for_inference(args.model, args.checkpoint)

    if args.prompt:
        # Single generation
        text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        )
        print(text)
    else:
        # Interactive mode
        interactive_chat(
            model, tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
