"""Notebook plotting helpers render without a display."""
import matplotlib
import torch

matplotlib.use("Agg")

from src.config import TrainConfig  # noqa: E402
from src.visualize import (  # noqa: E402
    plot_expert_load,
    plot_lr_schedule,
    plot_param_breakdown,
    plot_training_curves,
)


def test_plot_training_curves_one_panel_per_present_key():
    history = [{"step": 1, "train/loss": 2.0}, {"step": 2, "train/loss": 1.5, "train/aux_loss": 0.1},
               {"step": 2, "val/loss": 1.7}]
    fig = plot_training_curves(history)
    assert len(fig.axes) == 3
    matplotlib.pyplot.close(fig)
    fig = plot_training_curves([])
    assert len(fig.axes) == 1  # placeholder figure
    matplotlib.pyplot.close(fig)


def test_plot_expert_load_panels():
    fig = plot_expert_load({0: torch.tensor([3.0, 1.0]), 1: torch.tensor([2.0, 2.0])})
    assert len(fig.axes) == 2
    matplotlib.pyplot.close(fig)
    fig = plot_expert_load({})
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


def test_plot_lr_schedule_returns_figure():
    cfg = TrainConfig(max_steps=50, warmup_steps=5, peak_lr=1e-3, min_lr=1e-4)
    fig = plot_lr_schedule(cfg)
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


def test_plot_param_breakdown_returns_figure():
    breakdowns = {
        "tiny": {"embedding": 16_000_000, "attention": 5_000_000, "feed_forward": 12_000_000, "norm": 10_000},
        "small": {"embedding": 16_000_000, "attention": 20_000_000, "feed_forward": 48_000_000, "norm": 20_000},
    }
    fig = plot_param_breakdown(breakdowns)
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


# ──────────────────────────────────────────────
# Embedding space: 2D matplotlib and interactive 3D Plotly
# ──────────────────────────────────────────────

import pytest  # noqa: E402

from src.model import TransformerLM  # noqa: E402
from src.visualize import (  # noqa: E402
    _display_token,
    _select_token_ids,
    _token_category,
    plot_embedding_space,
    plot_embedding_space_3d,
)


@pytest.fixture
def small_model(tiny_cfg, seed):
    return TransformerLM(tiny_cfg)


def test_select_token_ids_merged_skips_specials_and_byte_alphabet(small_model, tokenizer):
    ids = _select_token_ids(small_model, tokenizer, 50, "merged")
    assert len(ids) == 50
    assert not set(ids) & tokenizer.special_ids()
    assert all(len(tokenizer.id_to_token(i)) > 1 for i in ids)
    everything = _select_token_ids(small_model, tokenizer, 50, "all")
    assert any(len(tokenizer.id_to_token(i)) == 1 for i in everything)
    with pytest.raises(ValueError):
        _select_token_ids(small_model, tokenizer, 50, "bogus")


def test_token_labels_and_categories():
    assert _display_token("Ġthe") == "␣the"
    assert _token_category("Ġthe") == "word (with leading space)"
    assert _token_category("ing") == "word piece"
    assert _token_category("123") == "digits"
    assert _token_category(".,") == "punctuation"
    assert _token_category("a1") == "other"


@pytest.mark.parametrize("method", ["pca", "tsne"])
def test_plot_embedding_space_2d_methods_render(small_model, tokenizer, method):
    fig = plot_embedding_space(small_model, tokenizer, n_tokens=40, method=method)
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


def test_plot_embedding_space_2d_rejects_unknown_method(small_model, tokenizer):
    with pytest.raises(ValueError):   # used to silently run t-SNE for "umap"/"pca"
        plot_embedding_space(small_model, tokenizer, n_tokens=40, method="umap")


@pytest.mark.parametrize("method", ["pca", "tsne"])
def test_plot_embedding_space_3d_points_and_hover(small_model, tokenizer, method):
    pytest.importorskip("plotly")
    fig = plot_embedding_space_3d(small_model, tokenizer, n_tokens=40, method=method)
    marker_traces = [t for t in fig.data if t.mode == "markers"]
    assert sum(len(t.x) for t in marker_traces) == 40          # every token is a point, in exactly one class
    assert all(len(t.x) == len(t.y) == len(t.z) == len(t.text) for t in marker_traces)


def test_plot_embedding_space_3d_writes_self_contained_html(small_model, tokenizer, tmp_path):
    pytest.importorskip("plotly")
    out = tmp_path / "nested" / "emb.html"
    plot_embedding_space_3d(small_model, tokenizer, n_tokens=30, out_path=str(out))
    html = out.read_text(encoding="utf-8")
    # plotly.js is inlined (no remote <script src>), so the single file opens offline.
    assert "Plotly.newPlot" in html and "<script src=" not in html
    assert out.stat().st_size > 1_000_000


def test_plot_embedding_space_3d_rejects_unknown_method(small_model, tokenizer):
    pytest.importorskip("plotly")
    with pytest.raises(ValueError):
        plot_embedding_space_3d(small_model, tokenizer, n_tokens=30, method="umap")
