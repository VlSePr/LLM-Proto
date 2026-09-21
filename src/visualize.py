"""
Visualization of model internals and training metrics.
Generates matplotlib figures logged to Wandb.
"""

import os
import string
import sys

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

# Use the non-interactive "Agg" backend on headless servers (vast.ai, remote SSH)
# so matplotlib renders to in-memory buffers instead of trying to open a window.
# In notebooks/Colab the interactive backend is auto-detected via IPython modules.
if not any(k in sys.modules for k in ("IPython", "ipykernel", "google.colab")):
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import seaborn as sns

from .config import TrainConfig
from .model import TransformerLM
from .tokenizer import LLMTokenizer


def plot_attention_heatmap(
    model: TransformerLM,
    input_ids: torch.Tensor,
    tokenizer: LLMTokenizer,
    layer_idx: int = 0,
    head_idx: int = 0,
    max_len: int = 64,
) -> plt.Figure:
    """
    Visualize attention weights for a specific layer and head.
    Diagnostic purpose: reveals what the model "looks at" when predicting
    each token — early layers often attend locally; deeper layers show
    long-range or semantic patterns.
    Returns a matplotlib figure.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids[:1, :max_len].to(device)  # Single sample, truncate
    B, T = input_ids.shape

    # Hook to capture attention weights
    attn_weights = {}

    def hook_fn(module, input, output):
        # We need to recompute attention manually to get weights
        pass

    # Manual forward through target layer to get attention weights
    with torch.no_grad():
        x = model.tok_emb(input_ids)
        rope_freqs = model.rope_freqs[:T]

        for i, layer in enumerate(model.layers):
            h_norm = layer.attn_norm(x)
            attn = layer.attn
            q = attn.wq(h_norm).view(B, T, attn.n_heads, attn.head_dim)
            k = attn.wk(h_norm).view(B, T, attn.n_kv_heads, attn.head_dim)

            from .model import apply_rope
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)

            if attn.n_rep > 1:
                k = k.unsqueeze(3).expand(B, T, attn.n_kv_heads, attn.n_rep, attn.head_dim)
                k = k.reshape(B, T, attn.n_heads, attn.head_dim)

            q_t = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
            k_t = k.transpose(1, 2)

            if i == layer_idx:
                # Compute raw attention scores
                scale = attn.head_dim ** -0.5
                scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
                # Causal mask
                causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                weights = F.softmax(scores, dim=-1)
                attn_weights["weights"] = weights[0, head_idx].cpu().numpy()
                break

            # Continue forward pass
            out = model.layers[i](x, rope_freqs)
            x = out[0]

    if "weights" not in attn_weights:
        return _empty_figure("Could not extract attention weights")

    # Decode tokens for labels
    tokens = [tokenizer.id_to_token(int(tid)) or f"[{tid}]" for tid in input_ids[0].cpu()]

    fig, ax = plt.subplots(figsize=(min(12, T * 0.3 + 2), min(10, T * 0.3 + 2)))
    sns.heatmap(
        attn_weights["weights"],
        xticklabels=tokens, yticklabels=tokens,
        cmap="viridis", ax=ax, square=True,
    )
    ax.set_title(f"Attention — Layer {layer_idx}, Head {head_idx}")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    return fig


EMBEDDING_METHODS = ("pca", "tsne")


def _select_token_ids(model: TransformerLM, tokenizer: LLMTokenizer, n_tokens: int, selection: str) -> list[int]:
    """Token ids to plot.

    ``"merged"`` (default) skips the special tokens *and* the 256 single-character byte-level alphabet
    tokens that the BPE trainer puts first, so the plot shows learned sub-words (ids are in merge order,
    i.e. roughly most-frequent first). ``"all"`` takes the first ids after the special tokens.
    """
    if selection not in ("merged", "all"):
        raise ValueError(f"selection must be 'merged' or 'all', got {selection!r}")
    limit = min(model.tok_emb.weight.shape[0], tokenizer.vocab_size)
    special = tokenizer.special_ids()
    ids: list[int] = []
    for i in range(limit):
        if i in special:
            continue
        token = tokenizer.id_to_token(i)
        if token is None or (selection == "merged" and len(token) <= 1):
            continue
        ids.append(i)
        if len(ids) >= n_tokens:
            break
    if len(ids) < 4:
        raise ValueError(f"need at least 4 tokens to project embeddings, found {len(ids)}")
    return ids


def _embedding_subset(model: TransformerLM, tokenizer: LLMTokenizer, n_tokens: int, selection: str):
    """``(vectors, ids, raw_tokens)`` for the selected tokens of ``model.tok_emb``."""
    model.eval()
    ids = _select_token_ids(model, tokenizer, n_tokens, selection)
    weights = model.tok_emb.weight.detach().float().cpu().numpy()
    raw = [tokenizer.id_to_token(i) or f"[{i}]" for i in ids]
    return weights[ids], ids, raw


def _project_embeddings(vectors: np.ndarray, method: str, n_components: int):
    """Reduce *vectors* to *n_components* dims; returns ``(coords, axis_titles)``."""
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        coords = reducer.fit_transform(vectors)
        titles = [f"PC{i + 1} ({ratio:.0%})" for i, ratio in enumerate(reducer.explained_variance_ratio_)]
        return coords, titles
    if method == "tsne":
        from sklearn.manifold import TSNE
        # perplexity ~ "effective number of neighbors"; must be < n_samples, 30 is the usual sweet spot.
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(vectors) - 1))
        return reducer.fit_transform(vectors), [f"t-SNE {i + 1}" for i in range(n_components)]
    raise ValueError(f"method must be one of {EMBEDDING_METHODS}, got {method!r}")


def _display_token(raw: str) -> str:
    """Make a byte-level BPE token readable: ``'Ġthe'`` -> ``'␣the'``, newline marker -> ``'⏎'``."""
    return raw.replace("Ġ", "␣").replace("Ċ", "⏎")


def _token_category(raw: str) -> str:
    """Coarse class of a token, used to colour the 3D plot."""
    spaced = raw.startswith("Ġ")
    body = raw[1:] if spaced else raw
    if body.isalpha():
        return "word (with leading space)" if spaced else "word piece"
    if body.isdigit():
        return "digits"
    if body and all(c in string.punctuation for c in body):
        return "punctuation"
    return "other"


def plot_embedding_space(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    n_tokens: int = 500,
    method: str = "tsne",
    selection: str = "merged",
) -> plt.Figure:
    """
    Visualize token embedding space in 2D (``method`` = ``"tsne"`` or ``"pca"``).
    Diagnostic purpose: well-trained embeddings cluster semantically similar
    tokens (e.g., digits together, punctuation together). Uniform blobs
    suggest undertrained embeddings. See ``plot_embedding_space_3d`` for the
    interactive, downloadable version.
    """
    subset, _ids, raw = _embedding_subset(model, tokenizer, n_tokens, selection)
    coords, axes = _project_embeddings(subset, method, 2)
    labels = [_display_token(r) for r in raw]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=10)

    # Label some points
    step = max(1, len(labels) // 50)
    for i in range(0, len(labels), step):
        ax.annotate(labels[i], (coords[i, 0], coords[i, 1]), fontsize=5, alpha=0.7)

    ax.set_title(f"Token Embedding Space ({method.upper()}, {len(labels)} tokens)")
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    plt.tight_layout()
    return fig


def plot_embedding_space_3d(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    n_tokens: int = 1000,
    method: str = "pca",
    selection: str = "merged",
    out_path: str | None = None,
):
    """
    Interactive 3D token-embedding scatter (Plotly): rotate, zoom, hover to read a token,
    click a legend entry to hide a token class.

    ``method`` is ``"pca"`` (fast, deterministic; axis titles show explained variance) or
    ``"tsne"`` (slower, better local clusters). ``selection`` is ``"merged"`` (learned sub-words,
    default) or ``"all"`` (see ``_select_token_ids``). With ``out_path`` the figure is also written
    as ONE self-contained ``.html`` (plotly.js embedded, ~4 MB) that opens offline in any browser.
    Returns the ``plotly.graph_objects.Figure``.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover - plotly is in requirements.txt
        raise ImportError("plot_embedding_space_3d needs plotly: pip install plotly") from e

    subset, ids, raw = _embedding_subset(model, tokenizer, n_tokens, selection)
    coords, axes = _project_embeddings(subset, method, 3)
    labels = [_display_token(r) for r in raw]
    categories = [_token_category(r) for r in raw]

    fig = go.Figure()
    for category in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == category]
        fig.add_trace(go.Scatter3d(
            x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2],
            mode="markers", name=category,
            marker={"size": 3, "opacity": 0.8},
            text=[labels[i] for i in idx],
            customdata=[ids[i] for i in idx],
            hovertemplate="<b>%{text}</b><br>token id %{customdata}<extra>" + category + "</extra>",
        ))
    # A sparse set of always-visible labels; every token stays readable on hover.
    step = max(1, len(ids) // 40)
    shown = list(range(0, len(ids), step))
    fig.add_trace(go.Scatter3d(
        x=coords[shown, 0], y=coords[shown, 1], z=coords[shown, 2],
        mode="text", text=[labels[i] for i in shown], textfont={"size": 9},
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        title=f"Token embedding space ({method.upper()}, {len(ids)} tokens)",
        scene={"xaxis_title": axes[0], "yaxis_title": axes[1], "zaxis_title": axes[2]},
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        height=760,
    )

    if out_path:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    return fig


def plot_weight_distributions(model: TransformerLM) -> plt.Figure:
    """
    Plot weight standard deviations grouped by layer type.
    Diagnostic purpose: large std disparity across layers may indicate
    exploding/vanishing gradients or poor initialization. All stds should
    be roughly similar for a well-conditioned model.
    """
    weight_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.detach().cpu().float().numpy().flatten()
            weight_stats[name] = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
            }

    # Group by layer type
    groups = {}
    for name, stats in weight_stats.items():
        # Extract the layer type (e.g., "attn.wq", "ffn.w_gate")
        parts = name.split(".")
        if "layers" in parts:
            key = ".".join(parts[2:])  # e.g., "attn.wq.weight"
        else:
            key = name
        if key not in groups:
            groups[key] = []
        groups[key].append(stats)

    n_groups = len(groups)
    cols = min(4, n_groups)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (key, stats_list) in enumerate(sorted(groups.items())):
        ax = axes[idx]
        means = [s["mean"] for s in stats_list]
        stds = [s["std"] for s in stats_list]
        ax.bar(range(len(means)), stds, alpha=0.7, label="std")
        ax.set_title(key, fontsize=8)
        ax.tick_params(labelsize=6)

    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Weight Standard Deviations by Layer", fontsize=12)
    plt.tight_layout()
    return fig


def plot_activation_stats(
    model: TransformerLM,
    input_ids: torch.Tensor,
) -> plt.Figure:
    """
    Plot mean and std of activations per layer.
    Diagnostic purpose: means drifting away from 0 suggest normalization issues;
    shrinking stds ("activation collapse") or growing stds ("activation explosion")
    indicate training instability. RMSNorm + pre-norm should keep these stable.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids[:1, :512].to(device)

    layer_means = []
    layer_stds = []
    layer_names = []

    with torch.no_grad():
        x = model.tok_emb(input_ids)
        rope_freqs = model.rope_freqs[:input_ids.shape[1]]

        for i, layer in enumerate(model.layers):
            x, _ = layer(x, rope_freqs)
            act = x.float()
            layer_means.append(act.mean().item())
            layer_stds.append(act.std().item())
            layer_names.append(f"L{i}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(layer_names, layer_means, color="steelblue", alpha=0.8)
    ax1.set_title("Activation Mean per Layer")
    ax1.set_xlabel("Layer")
    ax1.tick_params(labelsize=7)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    ax2.bar(layer_names, layer_stds, color="coral", alpha=0.8)
    ax2.set_title("Activation Std per Layer")
    ax2.set_xlabel("Layer")
    ax2.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


def plot_token_loss_heatmap(
    model: TransformerLM,
    input_ids: torch.Tensor,
    tokenizer: LLMTokenizer,
    max_len: int = 128,
) -> plt.Figure:
    """
    Per-token loss heatmap: shows which tokens are hardest to predict.
    Diagnostic purpose: function words ("the", "is") should have low loss;
    content words or rare tokens will have high loss. Patterns here reveal
    whether the model has learned syntax vs. semantics.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids[:1, :max_len].to(device)

    with torch.no_grad():
        out = model(input_ids[:, :-1], targets=None)
        logits = out["logits"]  # (1, T-1, vocab)
        targets = input_ids[:, 1:]  # (1, T-1)
        losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view(1, -1)

    losses_np = losses[0].cpu().numpy()
    tokens = [tokenizer.id_to_token(int(tid)) or f"[{tid}]" for tid in input_ids[0, 1:].cpu()]

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.15), 2))
    sns.heatmap(
        losses_np.reshape(1, -1),
        xticklabels=tokens,
        yticklabels=["loss"],
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Cross-entropy loss"},
    )
    ax.set_title("Per-token Loss")
    plt.xticks(rotation=90, fontsize=5)
    plt.tight_layout()
    return fig


def plot_gradient_norms(grad_norms_by_layer: dict) -> plt.Figure:
    """
    Plot gradient L2 norms per layer.
    Diagnostic purpose: near-zero gradients signal vanishing gradient layers;
    very large norms indicate potential exploding gradients. Gradient clipping
    should cap these at the configured threshold (typically 1.0).
    """
    names = list(grad_norms_by_layer.keys())
    values = list(grad_norms_by_layer.values())

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.3), 4))
    ax.bar(range(len(names)), values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_title("Gradient Norms by Layer")
    ax.set_ylabel("L2 Norm")
    plt.tight_layout()
    return fig


def generate_all_visualizations(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    sample_input_ids: torch.Tensor,
    tracker,
    step: int,
):
    """Generate all model internals visualizations and log to tracker."""
    try:
        # 1. Attention heatmap (layer 0, head 0)
        fig = plot_attention_heatmap(model, sample_input_ids, tokenizer, layer_idx=0, head_idx=0)
        tracker.log_image("internals/attention_L0_H0", fig, step)
        plt.close(fig)

        # 2. Attention heatmap (last layer, head 0)
        last_layer = model.config.n_layers - 1
        fig = plot_attention_heatmap(model, sample_input_ids, tokenizer, layer_idx=last_layer, head_idx=0)
        tracker.log_image(f"internals/attention_L{last_layer}_H0", fig, step)
        plt.close(fig)

        # 3. Weight distributions
        fig = plot_weight_distributions(model)
        tracker.log_image("internals/weight_distributions", fig, step)
        plt.close(fig)

        # 4. Activation stats
        fig = plot_activation_stats(model, sample_input_ids)
        tracker.log_image("internals/activation_stats", fig, step)
        plt.close(fig)

        # 5. Per-token loss
        fig = plot_token_loss_heatmap(model, sample_input_ids, tokenizer)
        tracker.log_image("internals/token_loss", fig, step)
        plt.close(fig)

        # 6. Embedding space (expensive, do less frequently)
        if step % 5000 == 0 or step == 0:
            fig = plot_embedding_space(model, tokenizer, n_tokens=300)
            tracker.log_image("internals/embedding_space", fig, step)
            plt.close(fig)

    except Exception as e:
        print(f"Warning: Visualization failed at step {step}: {e}")


def plot_lr_schedule(train_config: TrainConfig) -> plt.Figure:
    """Cosine-decay-with-warmup LR schedule for a TrainConfig — no training needed.

    Diagnostic purpose: sanity-check warmup length vs. total steps and the decay shape
    before spending any compute (e.g. catch warmup_steps that's accidentally most of
    max_steps).
    """
    from .utils import get_lr

    steps = list(range(train_config.max_steps))
    lrs = [
        get_lr(s, train_config.warmup_steps, train_config.max_steps, train_config.peak_lr, train_config.min_lr)
        for s in steps
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, lrs)
    ax.axvline(train_config.warmup_steps, color="red", linestyle="--", alpha=0.5, label="end of warmup")
    ax.set_title("Learning Rate Schedule (cosine decay with warmup)")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_param_breakdown(breakdowns: dict[str, dict[str, int]]) -> plt.Figure:
    """Stacked bar chart of parameter count by component, one bar per named config.

    ``breakdowns`` maps a config name (e.g. a MODEL_CONFIGS key) to the dict returned by
    ModelConfig.param_count_breakdown(). Diagnostic purpose: shows how the embedding/
    attention/feed_forward/norm split shifts with scale — embedding dominates small
    tied-embedding models, feed_forward dominates at scale (SwiGLU's 3-matrix design).
    """
    names = list(breakdowns.keys())
    components = sorted({c for b in breakdowns.values() for c in b})
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bottom = np.zeros(len(names))
    for comp in components:
        vals = np.array([breakdowns[n].get(comp, 0) / 1e6 for n in names])
        ax.bar(names, vals, bottom=bottom, label=comp)
        bottom += vals
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Parameter Count by Component")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_training_curves(history: list, keys: tuple = ("train/loss", "val/loss", "train/aux_loss")) -> plt.Figure:
    """Loss curves from the metric history returned by ``train()`` (one panel per present key)."""
    present = [k for k in keys if any(k in h for h in history)]
    if not present:
        return _empty_figure("no metrics logged yet")
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)
    for ax, key in zip(axes[0], present, strict=True):
        pts = [(h["step"], h[key]) for h in history if key in h]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="." if len(pts) < 50 else None)
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_expert_load(load_counts: dict) -> plt.Figure:
    """Fraction of token-slots routed to each expert, one panel per MoE layer.

    ``load_counts`` is ``moe.measure_expert_load``'s output; the dashed line marks a
    perfectly uniform load.
    """
    if not load_counts:
        return _empty_figure("no MoE layers")
    layers = sorted(load_counts)
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4), squeeze=False)
    for ax, layer_idx in zip(axes[0], layers, strict=True):
        counts = torch.as_tensor(load_counts[layer_idx], dtype=torch.float32)
        fracs = (counts / counts.sum().clamp(min=1)).numpy()
        ax.bar(range(len(fracs)), fracs, color="steelblue")
        ax.axhline(1 / len(fracs), color="red", linestyle="--", label="uniform")
        ax.set_title(f"Layer {layer_idx} expert load")
        ax.set_xlabel("expert")
        ax.set_ylabel("fraction of tokens")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _empty_figure(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    return fig
