"""Plot downsampled magnitudes of two exported PEC system matrices."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def block_max_magnitude(matrix, step):
    """Keep the strongest entry in each block, including off-diagonal spikes."""
    size = matrix.shape[0]
    starts = np.arange(0, size, step)
    reduced = np.empty((len(starts), len(starts)), dtype=float)
    for row, start in enumerate(starts):
        block = np.abs(matrix[start:min(start + step, size), :])
        reduced[row] = np.maximum.reduceat(block, starts, axis=1).max(axis=0)
    return reduced


def save_plot(data, size, edge, output, logarithmic):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, (label, values) in zip(axes, data):
        if logarithmic:
            positive = values[values > 0]
            norm = LogNorm(vmin=np.percentile(positive, 1), vmax=positive.max())
            image = ax.imshow(values, origin="upper", extent=(0, size, size, 0),
                              cmap="magma", norm=norm, interpolation="nearest")
        else:
            vmax = np.percentile(values, 99.9)
            image = ax.imshow(values, origin="upper", extent=(0, size, size, 0),
                              cmap="magma", vmin=0, vmax=vmax,
                              interpolation="nearest")
        ax.axhline(edge, color="cyan", lw=0.5, alpha=0.8)
        ax.axvline(edge, color="cyan", lw=0.5, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Source edge / column")
        ax.set_ylabel("Test edge / row")
        fig.colorbar(image, ax=ax, shrink=0.8, label="max |matrix entry| per block")
    scale = "log" if logarithmic else "linear (top 0.1% clipped)"
    fig.suptitle(f"PEC system matrix magnitude, {scale}; edge {edge} marked")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-matrix", type=Path, required=True)
    parser.add_argument("--n-matrix", type=Path, required=True)
    parser.add_argument("--edge", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    t_matrix = np.load(args.t_matrix, mmap_mode="r")
    n_matrix = np.load(args.n_matrix, mmap_mode="r")
    if t_matrix.ndim != 2 or t_matrix.shape != n_matrix.shape or t_matrix.shape[0] != t_matrix.shape[1]:
        parser.error("the two matrices must have the same square shape")
    size = t_matrix.shape[0]
    if not 0 <= args.edge < size:
        parser.error("--edge must be a valid zero-based matrix index")
    step = max(1, (size + 1199) // 1200)
    data = [("T-EFIE", block_max_magnitude(t_matrix, step)),
            ("N-EFIE Form A", block_max_magnitude(n_matrix, step))]
    save_plot(data, size, args.edge, args.output_dir / "matrix_magnitude_linear.png", False)
    save_plot(data, size, args.edge, args.output_dir / "matrix_magnitude_log.png", True)


if __name__ == "__main__":
    main()
