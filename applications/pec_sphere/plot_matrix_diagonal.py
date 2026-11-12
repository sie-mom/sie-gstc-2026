"""Plot the diagonal magnitudes of an exported PEC system matrix."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--highlight-edge", type=int)
    args = parser.parse_args()

    matrix = np.load(args.matrix, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        parser.error("--matrix must contain a square 2D array")

    magnitude = np.abs(np.diag(matrix))
    edges = np.arange(len(magnitude))
    if args.highlight_edge is not None and not 0 <= args.highlight_edge < len(edges):
        parser.error("--highlight-edge must be a valid zero-based matrix index")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             constrained_layout=True)
    for ax in axes:
        ax.plot(edges, magnitude, color="#2166ac", lw=0.8)
        ax.set_ylabel("|A[i, i]|")
        ax.grid(alpha=0.25)
        if args.highlight_edge is not None:
            edge = args.highlight_edge
            ax.plot(edge, magnitude[edge], "o", color="#b2182b",
                    markersize=5, label=f"edge {edge}")
            ax.legend(loc="upper right")

    axes[0].set_title("Diagonal magnitude (linear scale)")
    axes[1].set_yscale("log")
    axes[1].axhline(1e-6, color="0.4", ls="--", lw=0.9,
                    label="1e-6 reference")
    axes[1].set_title("Diagonal magnitude (log scale)")
    axes[1].set_xlabel("Edge / matrix diagonal index")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(0, len(edges) - 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")
    print(f"Entries with |A[i,i]| < 1e-6: {np.count_nonzero(magnitude < 1e-6)} "
          f"of {len(magnitude)}")


if __name__ == "__main__":
    main()
