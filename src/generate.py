"""
Interactive text generation / inference CLI.
Supports single-prompt and multi-turn chat modes.
Uses KV-cache for efficient autoregressive generation.
"""

import re

import torch

from .corpus import strip_special_token_text
from .model import TransformerLM
from .tokenizer import LLMTokenizer
from .utils import build_model_from_checkpoint, get_device, warn_if_tokenizer_mismatch

# A model with no learned chat-template turn boundary will sometimes, past the real
# answer, free-associate a fabricated continuation shaped like the next turn (bare role
# words, not real <|im_start|>/<|im_end|> tokens -- those are already gone by the time this
# runs). Cut the text there so a hallucinated turn never reaches the user or gets stored
# back into ChatSession history. Only matches on its own line so ordinary prose mentioning
# "the user" or "as an assistant" is untouched.
ROLE_LEAK_RE = re.compile(r"\n\s*(?:user|assistant|system)\s*\n", re.IGNORECASE)


def clean_generated_text(text: str) -> str:
    """Remove literal special-token markup (``<|name|>``) and trim surrounding whitespace.

    Real special tokens are already dropped by ``tokenizer.decode(skip_special=True)``;
    this is a safety net for models that learned to emit the *text* of chat-template
    tokens (e.g. ``<|eot_id|>``). Ordinary angle brackets, pipes and code are untouched
    (the training-data cleaner in ``src/corpus.py`` is deliberately more aggressive).

    Also truncates at the first sign of a hallucinated new turn (see ``ROLE_LEAK_RE``) --
    a model with no trained turn-boundary signal can ramble past its real answer into a
    fabricated ``user``/``assistant`` continuation.
    """
    text = strip_special_token_text(text)
    m = ROLE_LEAK_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def load_model_for_inference(
    checkpoint_path: str,
    model_config_name: str | None = None,
    device: torch.device | None = None,
) -> tuple[TransformerLM, LLMTokenizer]:
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
    warn_if_tokenizer_mismatch(ckpt, tokenizer_path)
    return model, tokenizer


def generate_ids(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device | None = None,
    repetition_penalty: float = 1.0,
    extra_stop_ids: set[int] | None = None,
    repetition_penalty_window: int | None = None,
) -> list[int]:
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
            extra_stop_ids=extra_stop_ids,
            repetition_penalty_window=repetition_penalty_window,
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
    device: torch.device | None = None,
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


class ChatSession:
    """Multi-turn chat state: a rolling token history plus the sampling settings.

    Shared by the terminal ``interactive_chat`` and notebook widgets. Each turn is wrapped
    in ChatML markers (``<|im_start|>user ... <|im_end|><|im_start|>assistant ...
    <|im_end|>``) so the model gets a consistent, structural turn-boundary cue even though
    (unless the checkpoint was fine-tuned with ``corpus.build_finetune_corpus(...,
    chat_roles=...)``) it never learned one. ``reply`` feeds BOS + history + the new turn
    to the model, generation stops at ``<|eos|>`` or ``<|im_end|>``, and only the
    *cleaned* answer (hallucinated continuations trimmed by ``clean_generated_text``) is
    appended to history -- never the model's raw output -- so a rambling turn can't poison
    later turns. History is then truncated from the left so it always fits the model's
    context window together with ``max_new_tokens``.
    """

    def __init__(
        self,
        model: TransformerLM,
        tokenizer: LLMTokenizer,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        repetition_penalty_window: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_window = repetition_penalty_window
        self.device = next(model.parameters()).device
        self.history: list[int] = []
        self.turns: list[tuple[str, str]] = []

    @property
    def max_history(self) -> int:
        return max(1, self.model.config.max_seq_len - self.max_new_tokens - 1)

    def clear(self) -> None:
        self.history = []
        self.turns = []

    def reply(self, prompt: str) -> str:
        """Generate the model's answer to ``prompt`` in the context of the conversation so far."""
        tok = self.tokenizer
        im_start, im_end = tok.im_start_id, tok.im_end_id
        user_span = tok.encode(f"user\n{prompt}", add_bos=False)
        assistant_span = tok.encode("assistant\n", add_bos=False)
        turn_ids = [im_start] + user_span + [im_end] + [im_start] + assistant_span
        context_ids = [tok.bos_id] + self.history + turn_ids

        stop_ids = {i for i in (tok.eos_id, im_end) if i is not None}
        raw_new_ids = generate_ids(
            self.model, tok, context_ids,
            max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            top_k=self.top_k, top_p=self.top_p, repetition_penalty=self.repetition_penalty,
            repetition_penalty_window=self.repetition_penalty_window,
            extra_stop_ids=stop_ids or None, device=self.device,
        )
        response = clean_generated_text(tok.decode(raw_new_ids, skip_special=True))
        clean_new_ids = tok.encode(response, add_bos=False) + [im_end]
        self.history = (self.history + turn_ids + clean_new_ids)[-self.max_history:]
        self.turns.append((prompt, response))
        return response


def chat_widget(session: ChatSession, *, title: str = "LLM-Proto Chat"):
    """ipywidgets chat UI over a ``ChatSession`` (Jupyter / Colab).

    Sliders edit the session's sampling settings live; "Send" appends a turn to the
    conversation (history is kept by the session), "Clear" resets it. Returns the
    top-level widget; ``display()`` it. ``ipywidgets`` is imported here so the rest
    of this module stays usable without it.
    """
    import ipywidgets as widgets

    prompt_input = widgets.Textarea(placeholder="Type your message...",
                                    layout=widgets.Layout(width="100%", height="80px"))
    sliders = {
        "max_new_tokens": widgets.IntSlider(value=session.max_new_tokens, min=16, max=1024, step=16,
                                            description="Max tokens:", style={"description_width": "110px"}),
        "temperature": widgets.FloatSlider(value=session.temperature, min=0.1, max=2.0, step=0.05,
                                           description="Temperature:", style={"description_width": "110px"}),
        "top_k": widgets.IntSlider(value=session.top_k, min=1, max=200, step=1,
                                   description="Top-K:", style={"description_width": "110px"}),
        "top_p": widgets.FloatSlider(value=session.top_p, min=0.1, max=1.0, step=0.05,
                                     description="Top-P:", style={"description_width": "110px"}),
        "repetition_penalty": widgets.FloatSlider(value=session.repetition_penalty, min=1.0, max=2.0, step=0.05,
                                                  description="Rep. penalty:", style={"description_width": "110px"}),
    }
    send_btn = widgets.Button(description="Send", button_style="primary", layout=widgets.Layout(width="120px"))
    clear_btn = widgets.Button(description="Clear", button_style="warning", layout=widgets.Layout(width="120px"))
    output = widgets.Output(layout=widgets.Layout(width="100%", border="1px solid #ccc", min_height="200px",
                                                  max_height="600px", overflow="auto", padding="10px"))

    def on_send(_):
        prompt = prompt_input.value.strip()
        if not prompt:
            return
        for name, slider in sliders.items():
            setattr(session, name, slider.value)
        prompt_input.value = ""
        with output:
            print(f"You:   {prompt}")
            print(f"Model: {session.reply(prompt)}")
            print("-" * 60)

    def on_clear(_):
        session.clear()
        output.clear_output()

    send_btn.on_click(on_send)
    clear_btn.on_click(on_clear)
    return widgets.VBox([
        widgets.HTML(f"<h3>{title}</h3>"),
        prompt_input,
        widgets.HBox([send_btn, clear_btn]),
        widgets.VBox(list(sliders.values())),
        widgets.HTML("<hr>"),
        output,
    ])


def get_or_create_chat_session(
    session_state: "ChatSession | None",
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
) -> ChatSession:
    """Return ``session_state`` unchanged if already a session, else build one bound to ``model``/``tokenizer``.

    This is the lazy per-visitor init step for the Gradio UI: each browser session's ``gr.State`` starts as
    ``None`` and only becomes a real ``ChatSession`` on that visitor's first message.
    """
    if session_state is None:
        return ChatSession(
            model, tokenizer, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        )
    return session_state


def gradio_chat_send(
    session_state: "ChatSession | None",
    message: str,
    chat_history: list[dict] | None,
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> tuple[ChatSession, list[dict], str]:
    """One visitor's chat turn: lazily create their session, apply the current slider values, generate a reply,
    and append it to their ``chat_history`` (``gr.Chatbot(type="messages")`` format).

    Plain function with no Gradio event plumbing, so it can be called directly (e.g. with
    ``session_state=None`` to simulate a new visitor) both by ``build_gradio_chat_demo`` and by tests.
    Returns ``(updated_session_state, updated_chat_history, "")`` — the last value clears the textbox.
    """
    chat_history = list(chat_history or [])
    message = (message or "").strip()
    if not message:
        return session_state, chat_history, ""

    session = get_or_create_chat_session(
        session_state, model, tokenizer, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
    )
    session.max_new_tokens, session.temperature = max_new_tokens, temperature
    session.top_k, session.top_p, session.repetition_penalty = top_k, top_p, repetition_penalty

    response = session.reply(message)
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return session, chat_history, ""


def gradio_chat_clear(session_state: "ChatSession | None") -> tuple["ChatSession | None", list]:
    """Reset one visitor's conversation only. Safe to call before that visitor ever sent a message."""
    if session_state is not None:
        session_state.clear()
    return session_state, []


def gradio_version_ok() -> bool:
    """True if an installed gradio satisfies the pin (4.44.0 <= version < 6.0.0).

    Colab ships its own pre-installed ``gradio`` that can predate the ``type="messages"`` kwarg
    ``build_gradio_chat_demo`` relies on; a presence-only check (``find_spec``) would miss that and let
    the mismatched version through, so callers must check the version, not just whether it's installed.
    """
    import importlib.util

    if importlib.util.find_spec("gradio") is None:
        return False
    from importlib.metadata import version

    try:
        parts = tuple(int(p) for p in version("gradio").split(".")[:3] if p.isdigit())
    except Exception:
        return False
    return (4, 44, 0) <= parts < (6, 0, 0)


def build_gradio_chat_demo(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    *,
    title: str = "LLM-Proto Chat",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    concurrency_limit: int = 1,
    queue_max_size: int | None = None,
):
    """Multi-visitor Gradio chat UI over one shared ``model``/``tokenizer``.

    Each browser session gets its own ``ChatSession``: a Gradio ``gr.State`` is instantiated fresh per client
    by default, so it starts as ``None`` and is lazily filled in on that visitor's first message — visitors
    never see or affect each other's conversation, and the sampling sliders are likewise per-session
    automatically. ``gradio`` is imported here, not at module top, so this module stays importable without it.

    ``concurrency_limit`` caps how many replies generate at once; ``queue_max_size`` caps how many *waiting*
    requests are accepted beyond that (``None`` = unbounded) — once full, Gradio tells new visitors the app is
    busy instead of leaving them in an ever-growing queue. Visitors within the queue see their live position
    and ETA automatically (built into Gradio's ``.queue()``, no extra wiring needed here).

    Returns a ready ``gr.Blocks`` with ``.queue(...)`` already applied. Does **not** call ``.launch()`` — the
    caller decides ``share``/``server_port`` (run-time choices, not build-time ones).
    """
    import gradio as gr

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}")
        session_state = gr.State(None)  # fresh None per browser session; becomes a ChatSession on first Send
        chatbot = gr.Chatbot(type="messages", height=400, label=title)
        msg = gr.Textbox(placeholder="Type your message...", label="Message", lines=2)
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")
        with gr.Accordion("Sampling settings", open=False):
            max_new_tokens_s = gr.Slider(16, 1024, value=max_new_tokens, step=16, label="Max tokens")
            temperature_s = gr.Slider(0.1, 2.0, value=temperature, step=0.05, label="Temperature")
            top_k_s = gr.Slider(1, 200, value=top_k, step=1, label="Top-K")
            top_p_s = gr.Slider(0.1, 1.0, value=top_p, step=0.05, label="Top-P")
            repetition_penalty_s = gr.Slider(1.0, 2.0, value=repetition_penalty, step=0.05, label="Repetition penalty")
        sliders = [max_new_tokens_s, temperature_s, top_k_s, top_p_s, repetition_penalty_s]

        def _send(state, message, history, *slider_values):
            return gradio_chat_send(state, message, history, model, tokenizer, *slider_values)

        def _clear(state):
            return gradio_chat_clear(state)

        send_btn.click(_send, [session_state, msg, chatbot, *sliders], [session_state, chatbot, msg])
        msg.submit(_send, [session_state, msg, chatbot, *sliders], [session_state, chatbot, msg])
        clear_btn.click(_clear, [session_state], [session_state, chatbot])

    return demo.queue(default_concurrency_limit=concurrency_limit, max_size=queue_max_size)


def interactive_chat(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
):
    """Interactive multi-turn chat loop in the terminal (a front-end for ``ChatSession``).

    ``clear`` resets the history; ``/temp X`` ``/topk N`` ``/topp X`` ``/rep X`` adjust
    sampling at runtime; ``quit`` exits.
    """
    session = ChatSession(
        model, tokenizer, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
    )
    commands = {"/temp": ("temperature", float), "/topk": ("top_k", int),
                "/topp": ("top_p", float), "/rep": ("repetition_penalty", float)}

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
            session.clear()
            print("Chat cleared.")
            continue

        if prompt.startswith("/"):
            parts = prompt.split()
            spec = commands.get(parts[0])
            if spec is None:
                print("Unknown command. Use /temp, /topk, /topp, /rep")
                continue
            attr, cast = spec
            try:
                setattr(session, attr, cast(parts[1]))
                print(f"{attr} set to {getattr(session, attr)}")
            except (IndexError, ValueError):
                print(f"Usage: {parts[0]} <value>")
            continue

        print(f"\nModel: {session.reply(prompt)}")


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
