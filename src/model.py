"""
LLaMA-style decoder-only Transformer.
Components: RMSNorm, RoPE, SwiGLU, Grouped Query Attention, KV-cache.
Fully parameterized by ModelConfig — same code for 30M to 1B+.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is preferred over LayerNorm in modern LLMs (LLaMA, Mistral) because:
    - It skips the mean-centering step, making it ~10-15% faster.
    - Empirically matches LayerNorm quality for Transformer pre-training.
    - Formula: x * rsqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable per-channel scale (no bias — RMSNorm only re-scales, unlike LayerNorm)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability during norm computation,
        # then cast back to the original dtype (e.g., bf16) for the rest of the model.
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10_000.0, device: torch.device = None) -> torch.Tensor:
    """
    Precompute complex RoPE frequencies: e^(i * m * theta_k) for all positions m and freq indices k.

    RoPE (Rotary Position Embedding) encodes position by *rotating* query/key vectors
    in 2D subspaces. Unlike absolute or learned positional embeddings:
    - The dot product q·k naturally decays with relative distance → built-in locality bias.
    - Generalizes to unseen sequence lengths better than learned embeddings.
    - No extra parameters — positions are encoded via rotation matrices.

    The base frequency theta=10000 controls the wavelength range:
      lower dims rotate fast (capture local patterns), higher dims rotate slowly (long-range).
    """
    # theta_k = 1 / (theta^(2k/dim)) — geometric sequence of frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # t = [0, 1, 2, ..., max_seq_len-1] — position indices
    t = torch.arange(max_seq_len, device=device).float()
    # Outer product gives angles: angle[m, k] = m * theta_k
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim//2)
    # Convert to complex exponentials e^(i*angle) for efficient rotation via complex multiplication
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64: (max_seq_len, dim//2)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to input tensor."""
    # x: (batch, seq_len, n_heads, head_dim)
    # Reinterpret consecutive pairs of dims as complex numbers for rotation.
    # E.g., head_dim=64 → 32 complex numbers per head.
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
    # Complex multiplication rotates each 2D subspace by the position-dependent angle.
    # This is equivalent to the 2x2 rotation matrix [cos,-sin; sin,cos] but much faster.
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x)


class Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) and KV-cache support.

    GQA (Grouped Query Attention) uses fewer KV heads than Q heads:
    - Standard MHA: n_kv_heads == n_heads (every Q head has its own KV pair).
    - GQA: n_kv_heads < n_heads — multiple Q heads share one KV head group.
    - MQA: n_kv_heads == 1 (extreme sharing).
    Benefits: drastically reduces KV-cache memory during inference (important for
    long sequences and large batch sizes) with minimal quality loss.

    All linear projections omit bias — standard in modern LLMs (LLaMA, Mistral)
    as bias adds parameters without measurable quality improvement.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        # How many Q heads share each KV head (e.g., 8 Q heads / 4 KV heads = 2× sharing)
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repetition factor

        # Q has full n_heads projections; K and V have fewer (n_kv_heads) — this is the GQA savings
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # KV cache for inference: concatenate past keys/values with current ones.
        # This avoids recomputing attention over the entire sequence at each generation step,
        # reducing complexity from O(T²) to O(T) per new token.
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        new_kv_cache = (k, v)

        # GQA: repeat KV heads to match Q heads
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, k.shape[1], self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, k.shape[1], self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, v.shape[1], self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, v.shape[1], self.n_heads, self.head_dim)

        # Transpose for attention: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch's F.scaled_dot_product_attention automatically dispatches to:
        #   - Flash Attention 2  (O(T) memory, fused CUDA kernel — fastest)
        #   - Memory-efficient attention (xformers-style)
        #   - Standard math fallback
        # This gives us Flash Attention speed without an explicit dependency.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None and kv_cache is None),  # Use causal mask for training
            dropout_p=0.0,  # Dropout handled separately for compatibility
        )

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.wo(out)
        out = self.dropout(out)
        return out, new_kv_cache


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    SwiGLU replaces the traditional ReLU FFN (2 matrices) with a gated design (3 matrices):
      output = W_down * (SiLU(W_gate * x) ⊙ W_up * x)

    Why SwiGLU over standard ReLU FFN?
    - The gating mechanism (element-wise multiply of two projections) lets the network
      learn which features to amplify or suppress, improving expressiveness.
    - SiLU (Swish) activation is smooth and non-monotonic, avoiding ReLU's "dead neuron" problem.
    - Empirically outperforms ReLU and GELU FFNs in LLM benchmarks (PaLM, LLaMA papers).
    - The 3-matrix design uses 3 × dim × ffn_dim parameters instead of 2 × dim × 4×dim,
      so ffn_dim is set to 2/3 × 4 × dim to keep total parameter count roughly equivalent.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Three projections instead of two: gate, up-project, and down-project
        self.w_gate = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.w_up = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.w_down = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) acts as a learned soft switch; up-project provides the values to gate
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block: RMSNorm → Attention → residual → RMSNorm → FFN → residual.

    Pre-norm (normalize before sublayer) is used instead of post-norm because:
    - Training is more stable, especially for deep models (24+ layers).
    - Gradients flow more directly through the residual stream.
    - Removes the need for careful learning rate warmup that post-norm requires.
    All modern LLMs (GPT-3, LLaMA, Mistral, Gemma) use pre-norm.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h, new_kv = self.attn(self.attn_norm(x), rope_freqs, mask, kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer Language Model.
    Architecture: LLaMA-style with RMSNorm, RoPE, SwiGLU, GQA.
    Fully parameterized by ModelConfig.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output projection.
        # This reduces parameters by vocab_size × dim (e.g., 32K × 512 = 16M params saved)
        # and provides a useful inductive bias — tokens with similar embeddings produce
        # similar output logits, which improves convergence in practice.
        if config.tie_embeddings:
            self.output.weight = self.tok_emb.weight

        # Precompute RoPE frequencies once and register as a non-persistent buffer.
        # "Non-persistent" means they won't be saved in state_dict (recomputed on load),
        # but they will move to the correct device (CPU/GPU) with the model.
        rope_freqs = precompute_rope_freqs(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights following GPT-2 / LLaMA conventions.

        std=0.02 is the standard init for Transformers (from GPT-2).
        It keeps activations in a reasonable range at init, preventing
        vanishing/exploding signals before training begins.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections (wo and w_down) by 1/sqrt(2*n_layers).
        # Each layer adds to the residual stream, so with N layers the variance
        # would grow ~N× without scaling. The factor 2 accounts for two residual
        # additions per block (attention + FFN). This is the GPT-2 convention,
        # also used in LLaMA and Megatron-LM.
        for layer in self.layers:
            nn.init.normal_(layer.attn.wo.weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * self.config.n_layers))
            nn.init.normal_(layer.ffn.w_down.weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        start_pos: int = 0,
    ) -> dict:
        """
        Forward pass.
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs for loss computation
            kv_cache: List of (k, v) tuples per layer, or None
            start_pos: Position offset for RoPE (used with KV cache during inference)
        Returns:
            dict with 'logits', optionally 'loss', and 'kv_cache'
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)

        # Slice RoPE frequencies for current positions
        rope_freqs = self.rope_freqs[start_pos:start_pos + T]

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x, new_kv = layer(x, rope_freqs, kv_cache=layer_kv)
            new_kv_cache.append(new_kv)

        x = self.norm(x)
        logits = self.output(x)

        result = {"logits": logits, "kv_cache": new_kv_cache}

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-cache, temperature, top-k, and top-p sampling.
        """
        self.eval()
        B, T = input_ids.shape
        kv_cache = None
        generated = input_ids

        for i in range(max_new_tokens):
            if kv_cache is None:
                # First pass: process full prompt
                out = self.forward(generated, kv_cache=None, start_pos=0)
                kv_cache = out["kv_cache"]
            else:
                # Subsequent passes: process only the last token
                out = self.forward(generated[:, -1:], kv_cache=kv_cache, start_pos=generated.shape[1] - 1)
                kv_cache = out["kv_cache"]

            logits = out["logits"][:, -1, :]  # (B, vocab_size)

            # Temperature controls randomness: <1 = more deterministic, >1 = more creative.
            # It scales logits before softmax: lower temp → sharper distribution → less surprise.
            if temperature > 0:
                logits = logits / temperature

                # Top-k: keep only the k most likely tokens, zero out the rest.
                # Prevents sampling from the long tail of unlikely tokens.
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_vals[:, -1:]] = float("-inf")

                # Top-p (nucleus) filtering: keep the smallest set of tokens whose
                # cumulative probability ≥ top_p. Adapts dynamically — when the model
                # is confident, fewer tokens are kept; when uncertain, more are allowed.
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Remove tokens with cumulative probability above top_p
                    remove_mask = cumulative_probs > top_p
                    # Shift right so that first token above threshold is kept
                    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                    remove_mask[:, 0] = False
                    sorted_logits[remove_mask] = float("-inf")
                    # Scatter back
                    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        """Human-readable model summary."""
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        lines = [
            f"TransformerLM Summary",
            f"  Layers: {self.config.n_layers}",
            f"  Hidden dim: {self.config.dim}",
            f"  Attention heads: {self.config.n_heads} (KV heads: {self.config.n_kv_heads})",
            f"  Head dim: {self.config.head_dim}",
            f"  FFN dim: {self.config.ffn_dim}",
            f"  Vocab size: {self.config.vocab_size}",
            f"  Max seq len: {self.config.max_seq_len}",
            f"  Tie embeddings: {self.config.tie_embeddings}",
            f"  Total params: {total:,} ({total / 1e6:.1f}M)",
            f"  Trainable params: {trainable:,} ({trainable / 1e6:.1f}M)",
        ]
        return "\n".join(lines)
