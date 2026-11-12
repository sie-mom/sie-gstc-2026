"""Plot one row and column of two exported PEC system matrices."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-matrix", type=Path, required=True)
    parser.add_argument("--n-matrix", type=Path, required=True)
    parser.add_argument("--edge", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    t_matrix = np.load(args.t_matrix, mmap_mode="r")
    n_matrix = np.load(args.n_matrix, mmap_mode="r")
    if t_matrix.shape != n_matrix.shape or t_matrix.ndim != 2:
        parser.error("the two matrices must have the same 2D shape")
    if t_matrix.shape[0] != t_matrix.shape[1]:
        parser.error("the matrices must be square")
    if not 0 <= args.edge < t_matrix.shape[0]:
        parser.error("--edge must be a valid zero-based matrix index")

    edge = args.edge
    indices = np.arange(t_matrix.shape[0])
    first = max(0, edge - 100)
    last = min(len(indices) - 1, edge + 40)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharey=True,
                             constrained_layout=True)
    sections = (
        (t_matrix[edge, :], n_matrix[edge, :],
         f"Row {edge}: one test equation across all source edges", "Column index"),
        (t_matrix[:, edge], n_matrix[:, edge],
         f"Column {edge}: one source edge to all test equations", "Row index"),
    )

    for ax, (t_values, n_values, title, x_label) in zip(axes, sections):
        ax.plot(indices, np.abs(t_values), color="#2166ac", lw=1,
                label="T-EFIE")
        ax.plot(indices, np.abs(n_values), color="#b2182b", lw=1,
                label="N-EFIE Form A")
        ax.axvline(edge, color="0.3", ls="--", lw=0.9)
        ax.set_xlim(first, last)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Matrix entry magnitude")
    axes[0].legend(loc="upper right")
    fig.suptitle("Assembled PEC system matrices")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
