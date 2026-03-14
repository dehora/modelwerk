"""Attention visualization.

Heatmaps of attention weights showing which tokens
attend to which other tokens.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Matrix = list[list[float]]


def plot_attention_weights(
    attention_weights: list[Matrix],
    tokens: list[str],
    title: str = "Attention Weights",
    filepath: str | None = None,
) -> plt.Figure:
    """Plot attention weight heatmaps, one per head.

    attention_weights: list of (seq_len, seq_len) matrices, one per head.
    tokens: list of token strings for axis labels.
    """
    num_heads = len(attention_weights)
    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 5))
    if num_heads == 1:
        axes = [axes]

    for h, (weights, ax) in enumerate(zip(attention_weights, axes)):
        seq_len = len(weights)
        im = ax.imshow(weights, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"Head {h + 1}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

        # Show token labels if sequence is short enough
        if seq_len <= 40:
            display_tokens = [t if t != "\n" else "\\n" for t in tokens[:seq_len]]
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(display_tokens, rotation=90, fontsize=7)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(display_tokens, fontsize=7)

    fig.suptitle(title, fontsize=14)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Attention weight")

    if filepath:
        fig.savefig(filepath, dpi=150, bbox_inches="tight")

    return fig
