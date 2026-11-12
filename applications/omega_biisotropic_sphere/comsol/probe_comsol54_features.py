"""Manually probe COMSOL 5.4 Wave Optics feature availability.

This is an integration diagnostic, not an automated pytest test. It starts a
licensed COMSOL session and checks the feature types needed by the volumetric
GSTC implementation.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import mph
from volumetric_impedance_matching import _patch_mph_for_legacy_comsol


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _require_compute_context(allow_outside_slurm: bool) -> None:
    if os.environ.get("SLURM_JOB_ID") or allow_outside_slurm:
        return
    raise RuntimeError(
        "Refusing to start COMSOL outside a Slurm allocation. Pass "
        "--allow-outside-slurm for a deliberate workstation probe."
    )


def probe_features(*, version: str, cores: int) -> bool:
    """Return whether COMSOL exposes both Case-B domain feature types."""
    _patch_mph_for_legacy_comsol(version)
    client = mph.Client(version=version, cores=cores)
    model = client.create("probe_comsol_features")

    try:
        component = model.java.component().create("comp1", True)
        geometry = component.geom().create("geom1", 3)
        geometry.create("sph1", "Sphere")
        geometry.run("fin")

        physics = component.physics().create(
            "ewfd", "ElectromagneticWavesFrequencyDomain", "geom1"
        )
        print("Default features in ewfd:")
        for tag in physics.feature().tags():
            print(f" - {tag}: {physics.feature(tag).getType()}")

        failures: list[str] = []
        for tag, feature_type in (
            ("weq2", "WaveEquationElectric"),
            ("wcon1", "WeakContribution"),
        ):
            try:
                feature = physics.create(tag, feature_type, 3)
                print(f"Created {feature_type} ({tag}) successfully.")
                try:
                    child_tags = list(feature.feature().tags())
                except Exception:
                    child_tags = []
                if child_tags:
                    print(f"   Child feature tags: {child_tags}")
            except Exception as error:
                failures.append(feature_type)
                print(f"Failed to create {feature_type}: {error}")

        if failures:
            print(f"Unavailable required feature types: {', '.join(failures)}")
            return False

        print("All required COMSOL feature types are available.")
        return True
    finally:
        client.remove(model)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="5.4")
    parser.add_argument("--cores", type=_positive_int, default=1)
    parser.add_argument(
        "--allow-outside-slurm",
        action="store_true",
        help="Allow this licensed integration probe to run on a workstation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    _require_compute_context(arguments.allow_outside_slurm)
    available = probe_features(version=arguments.version, cores=arguments.cores)
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())
