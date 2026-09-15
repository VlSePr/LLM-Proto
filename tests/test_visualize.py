"""Notebook plotting helpers render without a display."""
import matplotlib
import torch

matplotlib.use("Agg")

from src.visualize import plot_expert_load, plot_training_curves  # noqa: E402


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
