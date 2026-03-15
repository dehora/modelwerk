"""Decision boundary visualization.

2D plots showing how a model partitions input space
into classification regions.
"""

import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — render to files, not windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_boundary_2d(
    predict_fn,
    data: list[list[float]],
    labels: list[float],
    title: str = "Decision Boundary",
    resolution: int = 100,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Evaluate predict_fn on a grid and show classification regions."""
    # Find data bounds with padding
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
    y_min, y_max = min(ys) - 0.5, max(ys) + 0.5

    x_step = (x_max - x_min) / resolution
    y_step = (y_max - y_min) / resolution

    # Evaluate on grid
    grid_preds: list[list[float]] = []
    for grid_row in range(resolution):
        row = []
        for grid_col in range(resolution):
            x = x_min + grid_col * x_step
            y = y_min + grid_row * y_step
            row.append(float(predict_fn([x, y])))
        grid_preds.append(row)

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Use contourf instead of imshow to avoid row-ordering issues.
    # Two regions: light gray = predicts 0, teal = predicts 1.
    xs_grid = [x_min + grid_col * x_step for grid_col in range(resolution)]
    ys_grid = [y_min + grid_row * y_step for grid_row in range(resolution)]
    # levels define the decision boundary: values < 0.5 map to class 0, > 0.5 to class 1
    ax.contourf(
        xs_grid, ys_grid, grid_preds,
        levels=[-0.5, 0.5, 1.5],
        colors=["#E0E0E0", "#5CB8B2"],
        alpha=0.5,
    )

    # Plot data points: hollow = class 0 (false), filled = class 1 (true)
    # Track whether we've added a legend entry for each class (only need one per class)
    added_label_0 = False
    added_label_1 = False
    for x, y, label in zip(xs, ys, labels):
        if label == 0:
            lbl = "class 0 (false)" if not added_label_0 else None
            added_label_0 = True
            ax.scatter(x, y, facecolors="white", edgecolors="black", s=100, zorder=5, linewidths=1.5, label=lbl)
        else:
            lbl = "class 1 (true)" if not added_label_1 else None
            added_label_1 = True
            ax.scatter(x, y, facecolors="black", edgecolors="black", s=100, zorder=5, linewidths=1.5, label=lbl)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    return fig


def _use_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ANSI 256-color codes for more saturated output
_GREEN_DIM = "\033[38;5;65m"       # muted olive for · background
_GREEN_BRIGHT = "\033[38;5;118m"   # vivid green for # regions
_RED = "\033[38;5;196m"            # bright red for misclassified
_WHITE = "\033[1;37m"              # bright white for correct data points
_GREEN_FRAME = "\033[38;5;149m"    # yellow-green for box frame and title
_RESET = "\033[0m"


def ascii_decision_boundary_2d(
    predict_fn,
    data: list[list[float]],
    labels: list[float],
    title: str = "Decision Boundary",
    width: int = 40,
    height: int = 20,
) -> str:
    """Render a decision boundary as ASCII art.

    Uses · for class 0 regions, # for class 1 regions.
    Data points are shown as O (class 0) and X (class 1).
    With color: green CRT effect, bright data points, red for misclassified.
    Falls back to plain text when output is not a terminal.
    """
    color = _use_color()

    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
    y_min, y_max = min(ys) - 0.5, max(ys) + 0.5

    x_step = (x_max - x_min) / width
    y_step = (y_max - y_min) / height

    # Map data points to grid positions, storing (label, grid_x, grid_y)
    point_map: dict[tuple[int, int], float] = {}
    for (px, py), label in zip(data, labels):
        grid_col = round((px - x_min) / x_step)
        grid_row = round((py - y_min) / y_step)
        grid_col = max(0, min(width - 1, grid_col))
        grid_row = max(0, min(height - 1, grid_row))
        point_map[(grid_col, grid_row)] = label

    # Braille characters: single-width in all terminals, no alignment issues
    _FILL = "⣿"  # all 8 dots — dense textured fill
    _EMPTY = "·"

    lines = []
    if color:
        lines.append(f"  {_GREEN_BRIGHT}{title}{_RESET}")
        lines.append(f"  {_GREEN_FRAME}┌{'─' * width}┐{_RESET}")
    else:
        lines.append(f"  {title}")
        lines.append("  ┌" + "─" * width + "┐")

    # Top to bottom (high y to low y)
    has_misclassified = False
    for grid_row in range(height - 1, -1, -1):
        row = []
        for grid_col in range(width):
            if (grid_col, grid_row) in point_map:
                label = point_map[(grid_col, grid_row)]
                char = "F" if label == 0 else "T"
                if color:
                    x = x_min + grid_col * x_step
                    y = y_min + grid_row * y_step
                    pred = predict_fn([x, y])
                    correct = (pred == int(label))
                    if correct:
                        row.append(f"{_WHITE}{char}{_RESET}")
                    else:
                        has_misclassified = True
                        row.append(f"{_RED}{char}{_RESET}")
                else:
                    row.append(char)
            else:
                x = x_min + grid_col * x_step
                y = y_min + grid_row * y_step
                pred = predict_fn([x, y])
                if color:
                    if pred == 0:
                        row.append(f"{_GREEN_DIM}{_EMPTY}{_RESET}")
                    else:
                        row.append(f"{_GREEN_BRIGHT}{_FILL}{_RESET}")
                else:
                    row.append(_EMPTY if pred == 0 else _FILL)

        if color:
            lines.append(f"  {_GREEN_FRAME}│{_RESET}{''.join(row)}{_GREEN_FRAME}│{_RESET}")
        else:
            lines.append("  │" + "".join(row) + "│")

    if color:
        lines.append(f"  {_GREEN_FRAME}└{'─' * width}┘{_RESET}")
        lines.append(f"  - {_GREEN_DIM}{_EMPTY}{_RESET} predicts 0 (false)    {_GREEN_BRIGHT}{_FILL}{_RESET} predicts 1 (true)")
        lines.append(f"  - {_WHITE}F{_RESET} false (0)      {_WHITE}T{_RESET} true (1)")
        if has_misclassified:
            lines.append(f"  - {_RED}T{_RESET}/{_RED}F{_RESET} misclassified")
    else:
        lines.append("  └" + "─" * width + "┘")
        lines.append(f"  - {_EMPTY} predicts 0 (false)    {_FILL} predicts 1 (true)")
        lines.append("  - F false (0)      T true (1)")

    return "\n".join(lines)


def plot_points_2d(
    data: list[list[float]],
    labels: list[float],
    title: str = "Data Points",
) -> plt.Figure:
    """Scatter plot of labeled 2D data points. Returns a matplotlib Figure."""
    fig, ax = plt.subplots()
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    for x, y, label in zip(xs, ys, labels):
        if label == 0:
            ax.scatter(x, y, facecolors="white", edgecolors="black", s=100, linewidths=1.5)
        else:
            ax.scatter(x, y, facecolors="black", edgecolors="black", s=100, linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    return fig
