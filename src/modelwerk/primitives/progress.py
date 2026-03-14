"""Level 0: Training progress display.

Terminal progress bar for long-running training loops. Updates a
single line in place using carriage return — no dependencies beyond
sys and the scalar primitives.
"""

import sys

from modelwerk.primitives import scalar


def progress_bar(
    epoch: int,
    total: int,
    loss: float,
    width: int = 30,
    stream=sys.stderr,
) -> None:
    """Display a training progress bar that updates in place.

    Overwrites the current line each call. Use progress_done() after
    the loop to finalize with a newline.

    Args:
        epoch: Current epoch (1-based).
        total: Total number of epochs.
        loss: Current loss value.
        width: Width of the bar in characters.
        stream: Output stream (stderr so it doesn't mix with lesson output).
    """
    fraction = scalar.multiply(float(epoch), scalar.inverse(float(total)))
    filled = int(scalar.multiply(fraction, float(width)))
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = int(scalar.multiply(fraction, 100.0))
    line = f"\r  Training: epoch {epoch}/{total}  loss={loss:.4f}  [{bar}] {pct}%"
    stream.write(line)
    stream.flush()


def progress_done(stream=sys.stderr) -> None:
    """Finalize the progress bar with a newline."""
    stream.write("\n")
    stream.flush()
