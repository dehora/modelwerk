"""Training curves.

Loss and accuracy plots over training epochs.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — render to files, not windows
import matplotlib.pyplot as plt


def plot_loss_curve(losses: list[float], title: str = "Loss") -> plt.Figure:
    """Plot training loss over epochs. Returns a matplotlib Figure."""
    # fig = the overall image, ax = the drawing area inside it
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    return fig


def plot_accuracy_curve(accuracies: list[float], title: str = "Accuracy") -> plt.Figure:
    """Plot training accuracy over epochs. Returns a matplotlib Figure."""
    fig, ax = plt.subplots()
    ax.plot(accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    return fig
