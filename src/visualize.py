"""
Visualization of model internals and training metrics.
Generates matplotlib figures logged to Wandb.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
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
from typing import Optional, List

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


def plot_embedding_space(
    model: TransformerLM,
    tokenizer: LLMTokenizer,
    n_tokens: int = 500,
    method: str = "tsne",
) -> plt.Figure:
    """
    Visualize token embedding space using t-SNE (or UMAP).
    Diagnostic purpose: well-trained embeddings cluster semantically similar
    tokens (e.g., digits together, punctuation together). Uniform blobs
    suggest undertrained embeddings.
    """
    model.eval()
    embeddings = model.tok_emb.weight.detach().cpu().numpy()

    # Select a subset of tokens (skip special tokens, take first n_tokens regular tokens)
    n_tokens = min(n_tokens, embeddings.shape[0])
    indices = list(range(5, 5 + n_tokens))  # Skip 5 special tokens
    subset = embeddings[indices]
    labels = [tokenizer.id_to_token(i) or f"[{i}]" for i in indices]

    if method == "tsne":
        from sklearn.manifold import TSNE
        # perplexity ≈ "effective number of neighbors"; must be < n_samples.
        # 30 is the default/sweet spot; we clamp to n_tokens-1 for small datasets.
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_tokens - 1))
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_tokens - 1))

    coords = reducer.fit_transform(subset)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=10)

    # Label some points
    step = max(1, n_tokens // 50)
    for i in range(0, len(labels), step):
        ax.annotate(labels[i], (coords[i, 0], coords[i, 1]), fontsize=5, alpha=0.7)

    ax.set_title(f"Token Embedding Space ({method.upper()}, {n_tokens} tokens)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
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
    T = input_ids.shape[1]

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


def _empty_figure(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    return fig
