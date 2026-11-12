#!/usr/bin/env python3
"""Plot saved radial magnitudes and the phi=0 far-field cut without solving again."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot(run_dir):
    report = json.loads((run_dir / "metrics.json").read_text())
    destination = run_dir / "plots"
    destination.mkdir(exist_ok=False)
    a = report["physical"]["radius_m"]
    for case_id in report["cases"]:
        with np.load(run_dir / case_id / "radial_fields.npz") as data:
            fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
            for column, field in enumerate(("E", "H")):
                for component, coordinate in enumerate("xyz"):
                    ax = axes[component, column]
                    numeric = abs(data[f"{field}_cartesian"][:, component])
                    target_key = f"target_{field}_cartesian"
                    target = abs(data[target_key][:, component]) if target_key in data else None
                    peak = max(
                        float(numeric.max()), float(target.max()) if target is not None else 0
                    )
                    floor = max(peak * 1e-10, 1e-30)
                    ax.semilogy(
                        data["radius_m"] / a,
                        np.maximum(numeric, floor),
                        label="NGSolve",
                    )
                    if target is not None:
                        ax.semilogy(
                            data["radius_m"] / a,
                            np.maximum(target, floor),
                            "--",
                            label="Prescribed target",
                        )
                    ax.axvline(1, color="gray", linewidth=0.8)
                    ax.set_ylabel(f"|{field}{coordinate}| (original normalization)")
                    ax.grid(alpha=0.25)
            axes[0, 0].legend()
            for ax in axes[-1]:
                ax.set_xlabel("r/a")
            fig.suptitle(f"{case_id}: magnitudes, display floor = panel peak x 1e-10")
            fig.tight_layout()
            fig.savefig(destination / f"{case_id}_radial.png", dpi=150)
            plt.close(fig)
        with np.load(run_dir / case_id / "far_field_phi0.npz") as data:
            theta = data["theta_rad"]
            theta_hat = np.stack((np.cos(theta), np.zeros_like(theta), -np.sin(theta)), axis=1)
            e = data["E_far_normalized"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(
                np.rad2deg(theta), abs(np.sum(e * theta_hat, axis=1)), label="theta polarization"
            )
            ax.plot(np.rad2deg(theta), abs(e[:, 1]), label="phi polarization")
            ax.set(
                xlabel="theta (degrees), phi=0",
                ylabel="|normalized far-field amplitude|",
                title=case_id,
            )
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(destination / f"{case_id}_far_field.png", dpi=150)
            plt.close(fig)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    plot(parser.parse_args().run_dir)
