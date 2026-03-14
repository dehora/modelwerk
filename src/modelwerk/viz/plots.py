"""Training curves.

Loss and accuracy plots over training epochs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curve(losses: list[float], title: str = "Loss") -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    return fig


def plot_accuracy_curve(accuracies: list[float], title: str = "Accuracy") -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    return fig
