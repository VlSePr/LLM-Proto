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
