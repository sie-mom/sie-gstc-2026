"""COMSOL validation driver for the omega-biisotropic sphere example."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import mph
import numpy as np
from jpype.types import JArray, JDouble, JInt
from scipy.constants import c, epsilon_0, mu_0


APPLICATION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APPLICATION_DIR))

from dipole_impedance_matching import (  # noqa: E402
    prescribed_total_fields,
    spherical_to_cartesian,
)
from workflow import (  # noqa: E402
    default_run_directory,
    load_workflow_configuration,
    write_field_csv,
)


DEFAULT_CONFIGURATION = (
    APPLICATION_DIR / "configurations" / "budhu_10ghz_eps7.json"
)
DEFAULT_CASE_ID = "budhu_10ghz_eps7"
FIELD_EXPRESSIONS = (
    "emw.Ex",
    "emw.Ey",
    "emw.Ez",
    "emw.Hx",
    "emw.Hy",
    "emw.Hz",
)


def _java_double_array(values: Sequence[float]) -> Any:
    """Convert numeric values to an unambiguous Java ``double[]``."""
    return JArray(JDouble)(values)


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _default_cores() -> int:
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(allocated) if allocated else 1


def _require_compute_context(allow_outside_slurm: bool) -> None:
    if os.environ.get("SLURM_JOB_ID") or allow_outside_slurm:
        return
    raise RuntimeError(
        "Refusing to start COMSOL outside a Slurm allocation. Submit the "
        "TinkerCliffs job or pass --allow-outside-slurm on a workstation."
    )


def _detected_backends() -> list[dict[str, Any]]:
    return [
        {
            "name": backend["name"],
            "build": backend["build"],
            "root": str(backend["root"]),
        }
        for backend in mph.discovery.find_backends()
    ]


def _find_modern_jvm() -> str | None:
    """Find a Java >= 9 libjvm.so on Linux to satisfy modern JPype requirements."""
    try:
        import jpype

        jvm = jpype.getDefaultJVMPath()
        if jvm and Path(jvm).exists():
            return str(jvm)
    except Exception:
        pass

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        for candidate in Path(java_home).glob("**/libjvm.so"):
            if candidate.is_file():
                return str(candidate)

    java_bin = shutil.which("java") or "/usr/bin/java"
    if os.path.exists(java_bin):
        try:
            real_bin = os.path.realpath(java_bin)
            home = Path(real_bin).resolve().parent.parent
            for candidate in home.glob("**/libjvm.so"):
                if candidate.is_file():
                    return str(candidate)
        except Exception:
            pass

    for base in ("/usr/lib/jvm", "/usr/java"):
        base_path = Path(base)
        if base_path.exists():
            for candidate in sorted(base_path.glob("**/libjvm.so"), reverse=True):
                if candidate.is_file():
                    return str(candidate)
    return None


def _patch_mph_for_legacy_comsol(version: str) -> None:
    """Allow modern JPype (requiring Java >= 9) to run with legacy COMSOL (5.x).

    COMSOL 5.4 bundled Java 8. Modern JPype (>=1.5) strictly requires Java >= 9.
    We route MPh to the system's modern JVM (Java 11/17) and pass appropriate
    reflection options so COMSOL's Java API client can run seamlessly.
    """
    if not str(version).startswith("5."):
        return
    try:
        import jpype

        modern_jvm = _find_modern_jvm()
        if not modern_jvm:
            print("[COMSOL 5.4 patch] Warning: No modern JVM found on system.", flush=True)
            return

        # 1. Update any cached backends in find_backends()
        try:
            backends = mph.discovery.find_backends()
            for b in backends:
                if str(b.get("major", "")) <= "5":
                    b["jvm"] = Path(modern_jvm)
        except Exception:
            pass

        # 2. Patch mph.discovery.backend
        orig_backend = mph.discovery.backend

        def _patched_backend(v: str | None = None):
            b = orig_backend(v)
            if str(b.get("major", "")) <= "5":
                b["jvm"] = Path(modern_jvm)
            return b

        mph.discovery.backend = _patched_backend

        # 3. Patch jpype.startJVM to pass reflection options and compatibility jars needed by modern Java
        orig_startJVM = jpype.startJVM

        def _patched_startJVM(*args, **kwargs):
            new_args = list(args)
            jvm_opts = [
                "--add-opens=java.base/java.lang=ALL-UNNAMED",
                "--add-opens=java.base/java.util=ALL-UNNAMED",
                "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
                "--add-opens=java.base/java.io=ALL-UNNAMED",
                "--add-opens=java.base/java.nio=ALL-UNNAMED",
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
                "--add-opens=java.base/java.security=ALL-UNNAMED",
                "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
                "--add-opens=java.base/java.text=ALL-UNNAMED",
                "--add-opens=java.desktop/java.awt=ALL-UNNAMED",
                "--add-opens=java.desktop/java.awt.color=ALL-UNNAMED",
                "--add-opens=java.desktop/sun.awt=ALL-UNNAMED",
            ]
            for opt in jvm_opts:
                if opt not in new_args:
                    new_args.append(opt)

            # Java 11+ removed Java EE modules (JAXB, Activation, Annotations).
            # We inject compatibility jars (such as jakarta.xml.bind-api) so
            # legacy COMSOL 5.4 classes can load javax.xml.bind.DatatypeConverter.
            jars_dir = Path(__file__).resolve().parent / "jars"
            compat_jars = sorted(jars_dir.glob("*.jar")) if jars_dir.is_dir() else []
            orig_classpath = kwargs.get("classpath")
            if orig_classpath is not None:
                if isinstance(orig_classpath, (list, tuple)):
                    cp_list = list(orig_classpath)
                else:
                    cp_list = [orig_classpath]
                for jar_file in compat_jars:
                    if str(jar_file) not in cp_list:
                        cp_list.append(str(jar_file))
                kwargs["classpath"] = cp_list
            elif compat_jars:
                kwargs["classpath"] = [str(j) for j in compat_jars]

            if compat_jars:
                print(
                    f"[COMSOL 5.4 patch] Injected Java EE compatibility jars: {[j.name for j in compat_jars]}",
                    flush=True,
                )
            return orig_startJVM(*new_args, **kwargs)

        jpype.startJVM = _patched_startJVM
        print(f"[COMSOL 5.4 patch] Redirected JVM to modern runtime: {modern_jvm}", flush=True)
    except Exception as exc:
        print(f"[COMSOL 5.4 patch] Warning: failed to apply JVM patch: {exc}", flush=True)


def _start_comsol(
    *, version: str, cores: int, runtime_dir: Path
) -> tuple[mph.Server, mph.Client]:
    _patch_mph_for_legacy_comsol(version)
    configuration_dir = runtime_dir / "configuration"
    temporary_dir = runtime_dir / "tmp"
    configuration_dir.mkdir(parents=True)
    temporary_dir.mkdir()
    server = mph.Server(
        cores=cores,
        version=version,
        port=0,
        arguments=[
            "-configuration",
            str(configuration_dir),
            "-tmpdir",
            str(temporary_dir),
        ],
    )
    try:
        client = mph.Client(cores=cores, version=version, port=server.port)
        # Primitive spheres and blocks do not require a CAD Import license.
        try:
            client.java.setDefaultGeometryKernel("comsol")
        except Exception:
            pass
    except Exception:
        server.stop()
        raise
    return server, client


def _stop_comsol(server: mph.Server, client: mph.Client) -> None:
    if client.port:
        client.disconnect()
    if server.running():
        server.stop()


def run_smoke_test(client: mph.Client) -> dict[str, Any]:
    """Exercise model creation and removal without solving."""
    model = client.create("omega_sphere_smoke")
    try:
        model.java.label("Omega sphere COMSOL smoke test")
        model_names = client.names()
    finally:
        client.remove(model)
    return {"model_names_during_test": model_names}


def _build_homogeneous_model(
    client: mph.Client,
    *,
    frequency_hz: float,
    outer_radius_lambda0: float = 1.8,
    mesh_preset: int = 3,
    source_current_moment_am: float,
) -> mph.Model:
    """Build a free-space point-dipole model with a spherical radiation boundary."""
    model = client.create("omega_sphere_homogeneous")
    java = model.java
    java.label("Homogeneous electric dipole validation")

    java.param().set("f0", f"{frequency_hz:.17g}[Hz]")
    java.param().set("lambda0", "c_const/f0")
    java.param().set("r_outer", f"{outer_radius_lambda0:.17g}*lambda0")
    java.param().set("pI", f"{source_current_moment_am:.17g}[A*m]")

    java.component().create("comp1", True)
    component = java.component("comp1")
    component.geom().create("geom1", 3)
    geometry = component.geom("geom1")
    geometry.lengthUnit("m")
    geometry.create("sph1", "Sphere")
    geometry.feature("sph1").label("Free-space domain")
    geometry.feature("sph1").set("r", "r_outer")
    geometry.feature("sph1").set("selresult", True)
    geometry.create("pt1", "Point")
    geometry.feature("pt1").label("Dipole location")
    geometry.feature("pt1").set("p", _java_double_array((0.0, 0.0, 0.0)))
    geometry.feature("pt1").set("selresult", True)
    geometry.run()

    # Select by location: the sphere also has geometric points on its surface.
    # A Box selection creates no geometry and stays valid if entity IDs change.
    component.selection().create("dipole_origin", "Box")
    origin = component.selection("dipole_origin")
    origin.label("Dipole point at the origin")
    origin.set("entitydim", JInt(0))
    selection_tolerance_m = 1e-6 * c / frequency_hz
    for axis in ("x", "y", "z"):
        origin.set(f"{axis}min", JDouble(-selection_tolerance_m))
        origin.set(f"{axis}max", JDouble(selection_tolerance_m))
    source_points = [int(point) for point in origin.entities(JInt(0))]
    if len(source_points) != 1:
        raise RuntimeError(
            "Expected exactly one geometric point near the origin within "
            f"+/-{selection_tolerance_m:.6g} m per axis; got {source_points}"
        )

    component.material().create("mat1", "Common")
    material = component.material("mat1")
    material.label("Free space")
    material.selection().all()
    material.propertyGroup("def").set("relpermittivity", ["1"])
    material.propertyGroup("def").set("relpermeability", ["1"])
    material.propertyGroup("def").set("electricconductivity", ["0[S/m]"])

    component.physics().create("emw", "ElectromagneticWaves", "geom1")
    emw = component.physics("emw")
    emw.selection().all()
    emw.create("sctr1", "Scattering", 2)
    emw.feature("sctr1").label("Spherical radiation boundary")
    emw.feature("sctr1").selection().all()
    emw.feature("sctr1").set("WaveType", "SphericalWave")
    emw.create("epd1", "ElectricPointDipole", 0)
    emw.feature("epd1").label("Centered z-directed electric dipole")
    emw.feature("epd1").selection().named("dipole_origin")
    applied_points = [
        int(point) for point in emw.feature("epd1").selection().entities(JInt(0))
    ]
    if applied_points != source_points:
        raise RuntimeError(
            f"Dipole selection mismatch: expected {source_points}, got {applied_points}"
        )
    print(f"Dipole source: exactly one point selected, ID {source_points[0]}", flush=True)
    emw.feature("epd1").set("normpI", "pI")
    emw.feature("epd1").set("enpI", _java_double_array((0.0, 0.0, 1.0)))

    component.mesh().create("mesh1")
    mesh = component.mesh("mesh1")
    mesh.autoMeshSize(mesh_preset)
    mesh.run()

    java.study().create("std1")
    java.study("std1").create("freq", "Frequency")
    java.study("std1").feature("freq").set("plist", "f0")
    java.study("std1").feature("freq").setSolveFor("/physics/emw", True)
    return model


def _interpolate_fields(
    model: mph.Model,
    coordinates: np.ndarray,
    field_expressions: Sequence[str] | None = None,
) -> np.ndarray:
    """Return Cartesian complex E and H components at arbitrary coordinates."""
    if field_expressions is None:
        field_expressions = FIELD_EXPRESSIONS
    if coordinates.ndim != 2 or coordinates.shape[0] != 3:
        raise ValueError("coordinates must have shape (3, number_of_points)")
    numerical = model.java.result().numerical()
    numerical.create("interp1", "Interp")
    try:
        interpolation = model.java.result().numerical("interp1")
        interpolation.set("data", "dset1")
        interpolation.set("expr", list(field_expressions))
        # COMSOL 6.1 Interp ignores edim for solution datasets; select domains.
        interpolation.selection().geom("geom1", JInt(3))
        interpolation.selection().all()
        java_coordinates = JArray(JDouble, 2)(coordinates.tolist())
        interpolation.setInterpolationCoordinates(java_coordinates)
        real = np.asarray(interpolation.getData(), dtype=float)
        imaginary = np.asarray(interpolation.getImagData(), dtype=float)
    finally:
        numerical.remove("interp1")
    values = real + 1j * imaginary
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    expected_shape = (len(field_expressions), coordinates.shape[1])
    if values.shape != expected_shape:
        raise RuntimeError(
            f"unexpected COMSOL interpolation shape {values.shape}; "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(values)):
        print("Warning: COMSOL returned nonfinite values (expected at the dipole origin singularity).", flush=True)
    return values



def _write_radial_comparison_plot(
    path: Path,
    radius: np.ndarray,
    wavelength: float,
    theta: np.ndarray,
    phi: np.ndarray,
    computed_electric: np.ndarray,
    computed_magnetic: np.ndarray,
    reference_electric: np.ndarray,
    reference_magnetic: np.ndarray,
) -> None:
    """Plot the dominant tangential fields on the radial sample line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)

    def tangential(
        electric: np.ndarray, magnetic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        electric_theta = (
            electric[0] * cos_theta * cos_phi
            + electric[1] * cos_theta * sin_phi
            - electric[2] * sin_theta
        )
        magnetic_phi = -magnetic[0] * sin_phi + magnetic[1] * cos_phi
        return electric_theta, magnetic_phi

    computed_e_theta, computed_h_phi = tangential(
        computed_electric, computed_magnetic
    )
    reference_e_theta, reference_h_phi = tangential(
        reference_electric, reference_magnetic
    )

    path.parent.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    for axis, computed, reference, ylabel in (
        (axes[0], computed_e_theta, reference_e_theta, r"$|E_\theta|$ (V/m)"),
        (axes[1], computed_h_phi, reference_h_phi, r"$|H_\phi|$ (A/m)"),
    ):
        axis.plot(radius / wavelength, np.abs(reference), label="Analytical")
        axis.plot(
            radius / wavelength,
            np.abs(computed),
            "--",
            label="COMSOL",
        )
        axis.set_yscale("log")
        axis.set_xlabel(r"$r/\lambda_0$")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
        axis.legend()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _write_zx_comparison_plot(
    path: Path,
    x: np.ndarray,
    z: np.ndarray,
    grid_shape: tuple[int, int],
    wavelength: float,
    computed_electric: np.ndarray,
    computed_magnetic: np.ndarray,
    reference_electric: np.ndarray,
    reference_magnetic: np.ndarray,
) -> None:
    """Plot 2D comparison in the zx plane (y=0) for |E_theta|, |H_phi|, and |E_x|."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radius = np.sqrt(x**2 + z**2)
    radius_safe = np.where(radius == 0.0, 1.0, radius)
    theta = np.arccos(np.clip(z / radius_safe, -1.0, 1.0))
    phi = np.where(x >= 0.0, 0.0, np.pi)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)

    def extract_components(
        electric: np.ndarray, magnetic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e_theta = (
            electric[0] * cos_theta * cos_phi
            + electric[1] * cos_theta * sin_phi
            - electric[2] * sin_theta
        )
        h_phi = -magnetic[0] * sin_phi + magnetic[1] * cos_phi
        e_x = electric[0]
        return e_theta, h_phi, e_x

    comp_e_theta, comp_h_phi, comp_e_x = extract_components(
        computed_electric, computed_magnetic
    )
    ref_e_theta, ref_h_phi, ref_e_x = extract_components(
        reference_electric, reference_magnetic
    )

    def to_db(values: np.ndarray) -> np.ndarray:
        mag = np.abs(values)
        finite = mag[np.isfinite(mag)]
        floor = (
            max(float(np.max(finite)) * 1e-12, 1e-30)
            if finite.size > 0
            else 1e-30
        )
        return 20.0 * np.log10(np.maximum(mag, floor))

    ref_db = [to_db(ref_e_theta), to_db(ref_h_phi), to_db(ref_e_x)]
    comp_db = [to_db(comp_e_theta), to_db(comp_h_phi), to_db(comp_e_x)]

    x_norm = (x / wavelength).reshape(grid_shape)
    z_norm = (z / wavelength).reshape(grid_shape)

    column_titles = (r"$|E_\theta|$ (dB)", r"$|H_\phi|$ (dB)", r"$|E_x|$ (dB)")
    path.parent.mkdir(exist_ok=True)
    figure, axes = plt.subplots(
        2, 3, figsize=(15, 9), constrained_layout=True, sharex=True, sharey=True
    )
    row_labels = ("Analytical", "COMSOL")

    for col in range(3):
        combined = np.concatenate((ref_db[col], comp_db[col]))
        finite = combined[np.isfinite(combined)]
        if finite.size > 0:
            vmin, vmax = float(np.percentile(finite, 2.0)), float(
                np.percentile(finite, 98.0)
            )
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = -60.0, 0.0

        for row, data_set in enumerate((ref_db, comp_db)):
            ax = axes[row, col]
            img = ax.pcolormesh(
                x_norm,
                z_norm,
                data_set[col].reshape(grid_shape),
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_aspect("equal")
            ax.set_title(f"{row_labels[row]} {column_titles[col]}")
            ax.set_xlabel(r"$x/\lambda_0$")
            ax.set_ylabel(r"$z/\lambda_0$")
            figure.colorbar(img, ax=ax, shrink=0.85)

    figure.savefig(path, dpi=200)
    plt.close(figure)


def _relative_error(computed: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(computed - reference) / np.linalg.norm(reference))


def run_homogeneous_validation(
    client: mph.Client,
    *,
    configuration_path: Path,
    run_dir: Path,
    outer_radius_lambda0: float = 1.8,
    mesh_preset: int = 3,
    sample_min_lambda0: float = 0.20,
    sample_max_lambda0: float = 1.00,
    sample_points: int = 81,
    zx_grid_size: int = 80,
    zx_span_lambda0: float = 1.0,
    source_current_moment_am: float = 1.0,
) -> dict[str, Any]:
    """Solve and compare a free-space electric dipole against the analytic field."""
    configuration = load_workflow_configuration(configuration_path)
    frequency_hz = configuration.case.frequency
    wavelength = c / frequency_hz
    wavenumber = 2.0 * np.pi / wavelength
    impedance = np.sqrt(mu_0 / epsilon_0)

    model = _build_homogeneous_model(
        client,
        frequency_hz=frequency_hz,
        outer_radius_lambda0=outer_radius_lambda0,
        mesh_preset=mesh_preset,
        source_current_moment_am=source_current_moment_am,
    )
    try:
        print("Solving homogeneous dipole model...", flush=True)
        model.solve("Study 1")
        model_path = run_dir / "model.mph"
        model.save(model_path)
        print(f"Solved model saved: {model_path}", flush=True)

        radius = np.linspace(
            sample_min_lambda0 * wavelength,
            sample_max_lambda0 * wavelength,
            sample_points,
        )
        theta = np.full(sample_points, np.deg2rad(45.0))
        phi = np.zeros(sample_points)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        coordinates = np.vstack((x, y, z))
        print(f"Extracting fields at {sample_points} radial points...", flush=True)
        computed = _interpolate_fields(model, coordinates)
        computed_electric = computed[:3]
        computed_magnetic = computed[3:]

        effective_moment = (
            -1j
            * impedance
            * source_current_moment_am
            / (4.0 * np.pi * wavenumber)
        )
        from sie_mom.excitation import evaluate_dipole_fields

        reference_spherical = evaluate_dipole_fields(
            wavenumber,
            radius,
            theta,
            phi,
            impedance,
            effective_moment=(0.0, 0.0, effective_moment),
        )
        reference_electric = spherical_to_cartesian(
            reference_spherical[0], theta, phi
        )
        reference_magnetic = spherical_to_cartesian(
            reference_spherical[1], theta, phi
        )

        fields_dir = run_dir / "fields"
        fields_dir.mkdir(exist_ok=True)
        region = np.ones(sample_points, dtype=int)
        computed_path = fields_dir / "comsol_radial.csv"
        reference_path = fields_dir / "analytical_radial.csv"
        write_field_csv(
            computed_path,
            x,
            y,
            z,
            region,
            computed_electric,
            computed_magnetic,
        )
        write_field_csv(
            reference_path,
            x,
            y,
            z,
            region,
            reference_electric,
            reference_magnetic,
        )

        plot_path = run_dir / "plots" / "homogeneous_radial.png"
        _write_radial_comparison_plot(
            plot_path,
            radius,
            wavelength,
            theta,
            phi,
            computed_electric,
            computed_magnetic,
            reference_electric,
            reference_magnetic,
        )

        # 2D zx-plane extraction
        x_axis = np.linspace(
            -zx_span_lambda0 * wavelength, zx_span_lambda0 * wavelength, zx_grid_size
        )
        z_axis = np.linspace(
            -zx_span_lambda0 * wavelength, zx_span_lambda0 * wavelength, zx_grid_size
        )
        x_grid, z_grid = np.meshgrid(x_axis, z_axis, indexing="xy")
        x_2d = x_grid.ravel()
        y_2d = np.zeros_like(x_2d)
        z_2d = z_grid.ravel()
        coordinates_2d = np.vstack((x_2d, y_2d, z_2d))
        radius_2d = np.sqrt(x_2d**2 + y_2d**2 + z_2d**2)
        radius_safe_2d = np.where(radius_2d == 0.0, 1.0, radius_2d)
        theta_2d = np.arccos(np.clip(z_2d / radius_safe_2d, -1.0, 1.0))
        phi_2d = np.arctan2(y_2d, x_2d)

        print(
            f"Extracting 2D fields at {zx_grid_size}x{zx_grid_size} "
            f"({coordinates_2d.shape[1]} points) in the zx plane...",
            flush=True,
        )
        computed_2d = _interpolate_fields(model, coordinates_2d)
        computed_electric_2d = computed_2d[:3]
        computed_magnetic_2d = computed_2d[3:]

        reference_spherical_2d = evaluate_dipole_fields(
            wavenumber,
            radius_2d,
            theta_2d,
            phi_2d,
            impedance,
            effective_moment=(0.0, 0.0, effective_moment),
        )
        reference_electric_2d = spherical_to_cartesian(
            reference_spherical_2d[0], theta_2d, phi_2d
        )
        reference_magnetic_2d = spherical_to_cartesian(
            reference_spherical_2d[1], theta_2d, phi_2d
        )

        region_2d = np.ones(x_2d.size, dtype=int)
        computed_zx_path = fields_dir / "comsol_zx.csv"
        reference_zx_path = fields_dir / "analytical_zx.csv"
        write_field_csv(
            computed_zx_path,
            x_2d,
            y_2d,
            z_2d,
            region_2d,
            computed_electric_2d,
            computed_magnetic_2d,
        )
        write_field_csv(
            reference_zx_path,
            x_2d,
            y_2d,
            z_2d,
            region_2d,
            reference_electric_2d,
            reference_magnetic_2d,
        )

        plot_zx_path = run_dir / "plots" / "homogeneous_zx.png"
        _write_zx_comparison_plot(
            plot_zx_path,
            x_2d,
            z_2d,
            (zx_grid_size, zx_grid_size),
            wavelength,
            computed_electric_2d,
            computed_magnetic_2d,
            reference_electric_2d,
            reference_magnetic_2d,
        )

        resolved_configuration = configuration.to_dict()
        resolved_configuration["comsol"] = {
            "stage": "homogeneous",
            "outer_radius_lambda0": outer_radius_lambda0,
            "mesh_preset": mesh_preset,
            "boundary": "spherical_scattering",
            "source_current_moment_am": source_current_moment_am,
            "source_selection": "dipole_origin",
            "source_position_m": [0.0, 0.0, 0.0],
            "source_selection_half_width_m": 1e-6 * wavelength,
            "sample_min_lambda0": sample_min_lambda0,
            "sample_max_lambda0": sample_max_lambda0,
            "sample_points": sample_points,
            "sample_theta_degrees": 45.0,
            "sample_phi_degrees": 0.0,
            "zx_grid_size": zx_grid_size,
            "zx_span_lambda0": zx_span_lambda0,
        }
        with (run_dir / "configuration.json").open("w", encoding="utf-8") as stream:
            json.dump(resolved_configuration, stream, indent=2)
            stream.write("\n")

        mesh = model.java.component("comp1").mesh("mesh1")
        return {
            "frequency_hz": frequency_hz,
            "wavelength_m": wavelength,
            "outer_radius_lambda0": outer_radius_lambda0,
            "mesh_preset": mesh_preset,
            "source_current_moment_am": source_current_moment_am,
            "source_point_ids": [
                int(point)
                for point in model.java.component("comp1").physics("emw")
                .feature("epd1").selection().entities(JInt(0))
            ],
            "analytical_effective_moment": {
                "real": float(np.real(effective_moment)),
                "imag": float(np.imag(effective_moment)),
            },
            "mesh_elements": int(mesh.getNumElem()),
            "mesh_vertices": int(mesh.getNumVertex()),
            "relative_l2_error_electric": _relative_error(
                computed_electric, reference_electric
            ),
            "relative_l2_error_magnetic": _relative_error(
                computed_magnetic, reference_magnetic
            ),
            "relative_l2_error_electric_zx_masked": _relative_error(
                computed_electric_2d[:, radius_2d >= sample_min_lambda0 * wavelength],
                reference_electric_2d[:, radius_2d >= sample_min_lambda0 * wavelength],
            ),
            "relative_l2_error_magnetic_zx_masked": _relative_error(
                computed_magnetic_2d[:, radius_2d >= sample_min_lambda0 * wavelength],
                reference_magnetic_2d[:, radius_2d >= sample_min_lambda0 * wavelength],
            ),
            "files": {
                "configuration": "configuration.json",
                "model": "model.mph",
                "computed_radial": str(computed_path.relative_to(run_dir)),
                "reference_radial": str(reference_path.relative_to(run_dir)),
                "radial_plot": str(plot_path.relative_to(run_dir)),
                "computed_zx": str(computed_zx_path.relative_to(run_dir)),
                "reference_zx": str(reference_zx_path.relative_to(run_dir)),
                "zx_plot": str(plot_zx_path.relative_to(run_dir)),
            },
        }
    finally:
        client.remove(model)


def _write_omega_sphere_radial_plot(
    path: Path,
    radius: np.ndarray,
    sphere_radius: float,
    theta_deg: float,
    phi_deg: float,
    computed_electric: np.ndarray,
    computed_magnetic: np.ndarray,
    reference_electric: np.ndarray,
    reference_magnetic: np.ndarray,
) -> None:
    """Plot radial comparison along (theta, phi) for the omega-biisotropic sphere."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    sin_theta, cos_theta = np.sin(theta_rad), np.cos(theta_rad)
    sin_phi, cos_phi = np.sin(phi_rad), np.cos(phi_rad)

    def tangential(
        electric: np.ndarray, magnetic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        e_theta = (
            electric[0] * cos_theta * cos_phi
            + electric[1] * cos_theta * sin_phi
            - electric[2] * sin_theta
        )
        h_phi = -magnetic[0] * sin_phi + magnetic[1] * cos_phi
        return e_theta, h_phi

    comp_e_theta, comp_h_phi = tangential(computed_electric, computed_magnetic)
    ref_e_theta, ref_h_phi = tangential(reference_electric, reference_magnetic)
    comp_e_x = computed_electric[0]
    ref_e_x = reference_electric[0]
    comp_e_z = computed_electric[2]
    ref_e_z = reference_electric[2]

    normalized_r = radius / sphere_radius
    path.parent.mkdir(exist_ok=True)

    is_near_pole = abs(sin_theta) < 1e-4

    if is_near_pole:
        # At theta=0 deg, E_theta and H_phi are nulls for a z-dipole; E_z is dominant.
        figure, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
        panels = (
            (axes[0], np.abs(comp_e_z), np.abs(ref_e_z), r"$|E_z|$ (V/m) [Dominant]", True),
            (axes[1], np.abs(comp_e_theta), np.abs(ref_e_theta), r"$|E_\theta|$ (V/m) [Null check]", False),
            (axes[2], np.abs(comp_h_phi), np.abs(ref_h_phi), r"$|H_\phi|$ (A/m) [Null check]", False),
        )
    else:
        figure, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
        panels = (
            (axes[0], np.abs(comp_e_theta), np.abs(ref_e_theta), r"$|E_\theta|$ (V/m)", True),
            (axes[1], np.abs(comp_h_phi), np.abs(ref_h_phi), r"$|H_\phi|$ (A/m)", True),
            (axes[2], np.real(comp_e_x) / 1e5, np.real(ref_e_x) / 1e5, r"$\operatorname{Re}(E_x)$ ($10^5$ V/m)", False),
        )

    for axis, comp_vals, ref_vals, ylabel, log_scale in panels:
        axis.plot(normalized_r, ref_vals, "k-", linewidth=2.0, label="Analytical Prescribed")
        axis.plot(normalized_r, comp_vals, "r--", linewidth=1.8, label="COMSOL")
        axis.axvline(1.0, color="tab:blue", linestyle=":", label="Sheet (r=a)")
        if log_scale:
            axis.set_yscale("log")
        axis.set_xlabel(r"$r/a$")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
        axis.legend()

    axes[0].set_title(
        rf"Omega Sphere Radial Cut ($\theta = {theta_deg:.1f}^\circ, \phi = {phi_deg:.1f}^\circ$)"
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _write_omega_sphere_zx_plot(
    path: Path,
    x: np.ndarray,
    z: np.ndarray,
    grid_shape: tuple[int, int],
    sphere_radius: float,
    computed_electric: np.ndarray,
    computed_magnetic: np.ndarray,
    reference_electric: np.ndarray,
    reference_magnetic: np.ndarray,
) -> None:
    """Plot 2D comparison in the zx plane (y=0) for |E_theta|, |H_phi|, and Re(E_x)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radius = np.sqrt(x**2 + z**2)
    radius_safe = np.where(radius == 0.0, 1.0, radius)
    theta = np.arccos(np.clip(z / radius_safe, -1.0, 1.0))
    phi = np.where(x >= 0.0, 0.0, np.pi)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)

    def extract_components(
        electric: np.ndarray, magnetic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e_theta = (
            electric[0] * cos_theta * cos_phi
            + electric[1] * cos_theta * sin_phi
            - electric[2] * sin_theta
        )
        h_phi = -magnetic[0] * sin_phi + magnetic[1] * cos_phi
        e_x = electric[0]
        return e_theta, h_phi, e_x

    comp_e_theta, comp_h_phi, comp_e_x = extract_components(
        computed_electric, computed_magnetic
    )
    ref_e_theta, ref_h_phi, ref_e_x = extract_components(
        reference_electric, reference_magnetic
    )

    def to_db(values: np.ndarray) -> np.ndarray:
        mag = np.abs(values)
        finite = mag[np.isfinite(mag)]
        floor = (
            max(float(np.max(finite)) * 1e-12, 1e-30)
            if finite.size > 0
            else 1e-30
        )
        return 20.0 * np.log10(np.maximum(mag, floor))

    ref_e_db = to_db(ref_e_theta)
    comp_e_db = to_db(comp_e_theta)
    ref_h_db = to_db(ref_h_phi)
    comp_h_db = to_db(comp_h_phi)
    ref_re_ex = np.real(ref_e_x) / 1e5
    comp_re_ex = np.real(comp_e_x) / 1e5

    x_norm = (x / sphere_radius).reshape(grid_shape)
    z_norm = (z / sphere_radius).reshape(grid_shape)

    path.parent.mkdir(exist_ok=True)
    figure, axes = plt.subplots(
        2, 3, figsize=(15, 9), constrained_layout=True, sharex=True, sharey=True
    )
    row_labels = ("Analytical Prescribed", "COMSOL")
    column_titles = (
        r"$|E_\theta|$ (dB)",
        r"$|H_\phi|$ (dB)",
        r"$\operatorname{Re}(E_x)$ ($10^5$ V/m)",
    )

    e_comb = np.concatenate((ref_e_db, comp_e_db))
    e_finite = e_comb[np.isfinite(e_comb)]
    e_vmin = float(np.percentile(e_finite, 2.0)) if e_finite.size > 0 else -60.0
    e_vmax = float(np.percentile(e_finite, 98.0)) if e_finite.size > 0 else 0.0

    h_comb = np.concatenate((ref_h_db, comp_h_db))
    h_finite = h_comb[np.isfinite(h_comb)]
    h_vmin = float(np.percentile(h_finite, 2.0)) if h_finite.size > 0 else -60.0
    h_vmax = float(np.percentile(h_finite, 98.0)) if h_finite.size > 0 else 0.0

    ex_lim = 1.0

    cols_data = [
        ([ref_e_db, comp_e_db], "viridis", e_vmin, e_vmax),
        ([ref_h_db, comp_h_db], "viridis", h_vmin, h_vmax),
        ([ref_re_ex, comp_re_ex], "RdBu_r", -ex_lim, ex_lim),
    ]

    for col, (data_pair, cmap, vmin, vmax) in enumerate(cols_data):
        for row, data_set in enumerate(data_pair):
            ax = axes[row, col]
            img = ax.pcolormesh(
                x_norm,
                z_norm,
                data_set.reshape(grid_shape),
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.add_patch(
                plt.Circle(
                    (0.0, 0.0),
                    1.0,
                    fill=False,
                    color="white" if cmap == "viridis" else "black",
                    linestyle="--",
                )
            )
            ax.set_aspect("equal")
            ax.set_title(f"{row_labels[row]} {column_titles[col]}")
            ax.set_xlabel(r"$x/a$")
            ax.set_ylabel(r"$z/a$")
            figure.colorbar(img, ax=ax, shrink=0.85)

    figure.savefig(path, dpi=200)
    plt.close(figure)


def _build_omega_sphere_model(
    client: mph.Client,
    *,
    frequency_hz: float,
    radius_lambda0: float = 0.7,
    epsilon_r_interior: float = 7.0,
    source_current_moment_am: float = 1.0,
    chi_em: float = 0.38123,
    y_ee_imag: float = -0.0036679,
    z_mm_imag: float = -184.0435,
    air_extent_lambda0: float = 2.0,
    pml_thickness_lambda0: float = 0.25,
    mesh_preset: int = 4,
    boundary_elements_per_wavelength: float = 30.0,
    linear_solver: str = "iterative",
    physics_tag: str = "emw",
) -> mph.Model:
    """Build the omega-biisotropic sphere model with GSTC boundary and PML block."""
    model = client.create("omega_sphere_physical")
    java = model.java
    java.label("Omega-biisotropic sphere GSTC simulation")

    wavelength = c / frequency_hz
    radius_m = radius_lambda0 * wavelength
    air_extent_m = air_extent_lambda0 * wavelength

    # Parameters
    java.param().set("f0", f"{frequency_hz:.17g}[Hz]")
    java.param().set("lambda0", "c_const/f0")
    java.param().set("k0", "2*pi/lambda0")
    java.param().set("a", f"{radius_lambda0:.17g}*lambda0")
    java.param().set("b", f"{air_extent_lambda0:.17g}*lambda0")
    java.param().set("t_pml", f"{pml_thickness_lambda0:.17g}*lambda0")
    java.param().set("er", f"{epsilon_r_interior:.17g}")
    java.param().set("pI", f"{source_current_moment_am:.17g}[A*m]")
    java.param().set("Xem", f"{chi_em:.17g}")
    java.param().set("Yee", f"-j*{abs(y_ee_imag):.17g}[S]")
    java.param().set("Zmm", f"-j*{abs(z_mm_imag):.17g}[ohm]")

    component = java.component().create("comp1", True)
    geometry = component.geom().create("geom1", 3)
    geometry.lengthUnit("m")

    # 1. Metasurface sphere
    geometry.create("sph1", "Sphere")
    sph1 = geometry.feature("sph1")
    sph1.label("Omega metasurface sphere")
    sph1.set("r", "a")
    sph1.set("pos", _java_double_array((0.0, 0.0, 0.0)))

    # 2. Outer block with PML layer
    geometry.create("blk1", "Block")
    blk1 = geometry.feature("blk1")
    blk1.label("Outer air box with PML layer")
    blk1.set("size", ["2*(b + t_pml)", "2*(b + t_pml)", "2*(b + t_pml)"])
    blk1.set("base", "center")
    blk1.set("pos", _java_double_array((0.0, 0.0, 0.0)))
    blk1.set("layername", ["pml"])
    blk1.setIndex("layer", "t_pml", 0)
    for side in ("layerleft", "layerright", "layerbottom", "layertop", "layerfront", "layerback"):
        blk1.set(side, True)

    # 3. Dipole source point at the origin
    geometry.create("pt1", "Point")
    pt1 = geometry.feature("pt1")
    pt1.label("Dipole source point")
    pt1.set("p", _java_double_array((0.0, 0.0, 0.0)))

    geometry.run("fin")

    # Explicit Selections:
    # 1. Dipole source origin point
    component.selection().create("dipole_origin", "Ball")
    origin = component.selection("dipole_origin")
    origin.label("Dipole origin point")
    origin.set("entitydim", JInt(0))
    origin.set("r", JDouble(1e-6 * wavelength))
    origin.set("posx", JDouble(0.0))
    origin.set("posy", JDouble(0.0))
    origin.set("posz", JDouble(0.0))
    source_points = [int(p) for p in origin.entities(JInt(0))]
    if len(source_points) != 1:
        raise RuntimeError(f"Expected 1 origin point for dipole source, got {source_points}")

    # 2. Inner sphere domain
    component.selection().create("sphere_domain", "Ball")
    sphere_dom = component.selection("sphere_domain")
    sphere_dom.label("Inner sphere domain")
    sphere_dom.set("entitydim", JInt(3))
    sphere_dom.set("r", JDouble(radius_m * 0.9))
    sphere_dom.set("posx", JDouble(0.0))
    sphere_dom.set("posy", JDouble(0.0))
    sphere_dom.set("posz", JDouble(0.0))
    sphere_domains = [int(d) for d in sphere_dom.entities(JInt(3))]
    if not sphere_domains:
        raise RuntimeError("Could not find sphere domain via Ball selection")
    print(f"Sphere domain selected: {sphere_domains}", flush=True)

    # 3. Metasurface sphere boundary (surface)
    component.selection().create("sphere_boundary", "Ball")
    sphere_bnd_sel = component.selection("sphere_boundary")
    sphere_bnd_sel.label("Metasurface sphere boundary")
    sphere_bnd_sel.set("entitydim", JInt(2))
    try:
        sphere_bnd_sel.set("condition", "allvertices")
    except Exception:
        sphere_bnd_sel.set("condition", "all")
    sphere_bnd_sel.set("r", JDouble(radius_m * 1.01))
    sphere_bnd_sel.set("posx", JDouble(0.0))
    sphere_bnd_sel.set("posy", JDouble(0.0))
    sphere_bnd_sel.set("posz", JDouble(0.0))
    sphere_bnds = [int(b) for b in sphere_bnd_sel.entities(JInt(2))]
    if not sphere_bnds:
        raise RuntimeError("Could not find sphere boundary via Ball selection")
    print(f"Sphere boundaries selected: {len(sphere_bnds)} faces ({sphere_bnds})", flush=True)

    # 4. PML domains selection:
    component.selection().create("inner_box", "Box")
    inner_box = component.selection("inner_box")
    inner_box.label("Inner box excluding PML")
    inner_box.set("entitydim", JInt(3))
    # The default "intersects" also includes every PML domain touching the box.
    inner_box.set("condition", "inside")
    box_tol = air_extent_m + 1e-4 * wavelength
    for axis in ("x", "y", "z"):
        inner_box.set(f"{axis}min", JDouble(-box_tol))
        inner_box.set(f"{axis}max", JDouble(box_tol))

    component.selection().create("pml_domains", "Complement")
    pml_sel = component.selection("pml_domains")
    pml_sel.label("PML layer domains")
    pml_sel.set("entitydim", JInt(3))
    pml_sel.set("input", ["inner_box"])
    pml_domains = [int(d) for d in pml_sel.entities(JInt(3))]
    print(f"PML domains selected: {len(pml_domains)} domains ({pml_domains})", flush=True)
    if len(pml_domains) != 26:
        raise RuntimeError(
            f"Expected 26 PML domains around the layered block, got {pml_domains}"
        )

    # Coordinate system: PML
    component.coordSystem().create("pml1", "PML")
    component.coordSystem("pml1").label("PML Coordinate System")
    component.coordSystem("pml1").selection().named("pml_domains")

    # Materials
    component.material().create("mat1", "Common")
    mat1 = component.material("mat1")
    mat1.label("Air")
    mat1.selection().all()
    mat1.propertyGroup("def").set("relpermittivity", ["1"])
    mat1.propertyGroup("def").set("relpermeability", ["1"])
    mat1.propertyGroup("def").set("electricconductivity", ["0[S/m]"])
    # Refractive index required by Wave Optics (ewfd)
    try:
        mat1.propertyGroup().create("RefractiveIndex", "Refractive index")
        mat1.propertyGroup("RefractiveIndex").set("n", ["1"])
        mat1.propertyGroup("RefractiveIndex").set("ki", ["0"])
    except Exception:
        pass

    component.material().create("mat2", "Common")
    mat2 = component.material("mat2")
    mat2.label(f"Dielectric Core (eps_r={epsilon_r_interior})")
    mat2.selection().named("sphere_domain")
    mat2.propertyGroup("def").set("relpermittivity", ["er"])
    mat2.propertyGroup("def").set("relpermeability", ["1"])
    mat2.propertyGroup("def").set("electricconductivity", ["0[S/m]"])
    # Refractive index required by Wave Optics (ewfd): n = sqrt(er)
    try:
        mat2.propertyGroup().create("RefractiveIndex", "Refractive index")
        mat2.propertyGroup("RefractiveIndex").set("n", ["sqrt(er)"])
        mat2.propertyGroup("RefractiveIndex").set("ki", ["0"])
    except Exception:
        pass

    # Physics: Electromagnetic Waves (emw for RF, ewfd for Wave Optics)
    physics_type = (
        "ElectromagneticWavesFrequencyDomain"
        if physics_tag == "ewfd"
        else "ElectromagneticWaves"
    )
    component.physics().create(physics_tag, physics_type, "geom1")
    phys = component.physics(physics_tag)
    phys.selection().all()
    if physics_tag == "ewfd":
        try:
            phys.feature("wee1").set("DisplacementFieldModel", "RelativePermittivity")
        except Exception:
            pass

    # Scattering boundary condition on outer block faces
    phys.create("sctr1", "Scattering", 2)
    phys.feature("sctr1").label("Exterior Scattering Boundary")
    phys.feature("sctr1").selection().all()

    # Point dipole source (handles feature type naming across COMSOL versions)
    for dipole_type in ("ElectricPointDipole", "PointDipole", "PointDipoleSource"):
        try:
            phys.create("epd1", dipole_type, 0)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Failed to create point dipole feature in physics interface '{physics_tag}'")
    phys.feature("epd1").label("Centered z-directed electric dipole")
    phys.feature("epd1").selection().named("dipole_origin")
    phys.feature("epd1").set("normpI", "pI")
    phys.feature("epd1").set("enpI", _java_double_array((0.0, 0.0, 1.0)))

    # Surface Electric Current (scu1)
    phys.create("scu1", "SurfaceCurrent", 2)
    phys.feature("scu1").label("GSTC Surface Current (Js)")
    phys.feature("scu1").selection().named("sphere_boundary")
    phys.feature("scu1").set(
        "Js0",
        [
            f"Yee*0.5*(up({physics_tag}.Ex)+down({physics_tag}.Ex)) - Xem*(ny*0.5*(up({physics_tag}.Hz)+down({physics_tag}.Hz)) - nz*0.5*(up({physics_tag}.Hy)+down({physics_tag}.Hy)))",
            f"Yee*0.5*(up({physics_tag}.Ey)+down({physics_tag}.Ey)) + Xem*(nx*0.5*(up({physics_tag}.Hz)+down({physics_tag}.Hz)) - nz*0.5*(up({physics_tag}.Hx)+down({physics_tag}.Hx)))",
            f"Yee*0.5*(up({physics_tag}.Ez)+down({physics_tag}.Ez)) - Xem*(nx*0.5*(up({physics_tag}.Hy)+down({physics_tag}.Hy)) - ny*0.5*(up({physics_tag}.Hx)+down({physics_tag}.Hx)))",
        ],
    )

    # Surface Magnetic Current Density (smcd1)
    phys.create("smcd1", "SurfaceMagneticCurrentDensity", 2)
    phys.feature("smcd1").label("GSTC Surface Magnetic Current Density (Ms)")
    phys.feature("smcd1").selection().named("sphere_boundary")
    phys.feature("smcd1").set(
        "Jms0",
        [
            f"Zmm*0.5*(up({physics_tag}.Hx)+down({physics_tag}.Hx)) - Xem*(ny*0.5*(up({physics_tag}.Ez)+down({physics_tag}.Ez)) - nz*0.5*(up({physics_tag}.Ey)+down({physics_tag}.Ey)))",
            f"Zmm*0.5*(up({physics_tag}.Hy)+down({physics_tag}.Hy)) + Xem*(nx*0.5*(up({physics_tag}.Ez)+down({physics_tag}.Ez)) - nz*0.5*(up({physics_tag}.Ex)+down({physics_tag}.Ex)))",
            f"Zmm*0.5*(up({physics_tag}.Hz)+down({physics_tag}.Hz)) - Xem*(nx*0.5*(up({physics_tag}.Ey)+down({physics_tag}.Ey)) - ny*0.5*(up({physics_tag}.Ex)+down({physics_tag}.Ex)))",
        ],
    )

    # Study and linear solver sequence (created before mesh so operating frequency f0 is known)
    java.study().create("std1")
    java.study("std1").create("freq", "Frequency")
    java.study("std1").feature("freq").set("plist", "f0")
    try:
        java.study("std1").feature("freq").setSolveFor(f"/physics/{physics_tag}", True)
    except Exception:
        pass
    try:
        java.study("std1").feature("freq").activate(physics_tag, True)
    except Exception:
        pass

    # Mesh
    component.mesh().create("mesh1")
    mesh = component.mesh("mesh1")
    if physics_tag == "ewfd":
        # In COMSOL's Wave Optics module, automatic physics-controlled meshing defaults
        # to optical wavelengths (~1 um), inducing runaway element density on microwave models.
        # Use user-controlled mesh with explicit microwave lambda0 element sizing.
        mesh.automatic(False)
        try:
            mesh.feature("size").set("custom", True)
        except Exception:
            pass
        mesh.feature("size").set("hmax", "lambda0/4")
        mesh.feature("size").set("hmin", "lambda0/20")
        if boundary_elements_per_wavelength > 0.0:
            mesh.create("size_sphere", "Size")
            size_sp = mesh.feature("size_sphere")
            size_sp.label("Metasurface sphere boundary refinement")
            size_sp.selection().geom("geom1", JInt(2))
            size_sp.selection().named("sphere_boundary")
            try:
                size_sp.set("custom", True)
            except Exception:
                pass
            size_sp.set("hmaxactive", True)
            size_sp.set("hmax", f"lambda0/{boundary_elements_per_wavelength:.17g}")
            try:
                mesh.feature().move("size_sphere", 1)
            except Exception:
                pass
        mesh.run()
    else:
        mesh.autoMeshSize(mesh_preset)
        mesh.run()
        if boundary_elements_per_wavelength > 0.0:
            mesh.automatic(False)
            mesh.create("size_sphere", "Size")
            size_sp = mesh.feature("size_sphere")
            size_sp.label("Metasurface sphere boundary refinement")
            size_sp.selection().geom("geom1", JInt(2))
            size_sp.selection().named("sphere_boundary")
            size_sp.set("custom", True)
            size_sp.set("hmaxactive", True)
            size_sp.set("hmax", f"lambda0/{boundary_elements_per_wavelength:.17g}")
            try:
                mesh.feature().move("size_sphere", 1)
            except Exception:
                pass
            mesh.run()

    element_count = int(mesh.getNumElem())
    print(f"Omega sphere mesh: {element_count} elements", flush=True)
    if element_count == 0:
        raise RuntimeError("Omega sphere mesh is empty; refusing to solve")

    sol = java.sol().create("sol1")
    sol.study("std1")
    sol.create("st1", "StudyStep")
    sol.feature("st1").set("study", "std1")
    sol.feature("st1").set("studystep", "freq")
    sol.create("v1", "Variables")
    sol.feature("v1").set("control", "freq")
    sol.create("s1", "Stationary")
    sol.feature("s1").set("control", "freq")
    sol.feature("s1").feature("aDef").set("complexfun", True)
    sol.feature("s1").create("fc1", "FullyCoupled")
    direct_engine = "pardiso" if linear_solver == "pardiso" else "mumps"
    sol.feature("s1").create("d1", "Direct")
    sol.feature("s1").feature("d1").set("linsolver", direct_engine)
    sol.feature("s1").feature("d1").label(f"Suggested Direct Solver ({direct_engine})")

    sol.feature("s1").create("i1", "Iterative")
    sol.feature("s1").feature("i1").set("linsolver", "gmres")
    sol.feature("s1").feature("i1").set("prefuntype", "left")
    sol.feature("s1").feature("i1").label(f"Suggested Iterative Solver ({physics_tag})")
    try:
        sol.feature("s1").feature("i1").create("mg1", "Multigrid")
        sol.feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector")
        sol.feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec")
        sol.feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("sorvecdof", ["comp1_E"])
        sol.feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector")
        sol.feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1").set("prefun", "soruvec")
        sol.feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1").set("sorvecdof", ["comp1_E"])
    except Exception:
        try:
            sol.feature("s1").feature("i1").create("sv1", "SORVector")
            sol.feature("s1").feature("i1").feature("sv1").set("iter", JInt(1))
            sol.feature("s1").feature("i1").feature("sv1").set("relax", JDouble(1.0))
            sol.feature("s1").feature("i1").feature("sv1").set("seconditer", JInt(3))
        except Exception:
            pass

    active_solver = "i1" if linear_solver in ("iterative", "bicgstab", "gmres") else "d1"
    sol.feature("s1").feature("fc1").set("linsolver", active_solver)
    try:
        sol.feature("s1").feature().remove("fcDef")
    except Exception:
        pass
    sol.attach("std1")
    chosen_desc = direct_engine if active_solver == "d1" else "gmres"
    print(
        f"Configured study solver: sol1/s1 -> {active_solver} "
        f"({chosen_desc} for {physics_tag})",
        flush=True,
    )

    return model


def run_omega_sphere_simulation(
    client: mph.Client,
    *,
    configuration_path: Path,
    run_dir: Path,
    air_extent_lambda0: float = 2.0,
    pml_thickness_lambda0: float = 0.25,
    mesh_preset: int = 4,
    boundary_elements_per_wavelength: float = 30.0,
    linear_solver: str = "iterative",
    sample_theta_degrees: float = 45.0,
    sample_phi_degrees: float = 0.0,
    sample_min_radii: float = 0.05,
    sample_max_radii: float = 2.0,
    sample_points: int = 101,
    zx_grid_size: int = 80,
    zx_span_radii: float = 2.0,
    source_current_moment_am: float = 1.0,
    chi_em: float = 0.38123,
    y_ee_imag: float = -0.0036679,
    z_mm_imag: float = -184.0435,
    mesh_only: bool = False,
    physics_tag: str = "emw",
) -> dict[str, Any]:
    """Solve the physical omega-biisotropic sphere model and validate against prescribed theory."""
    configuration = load_workflow_configuration(configuration_path)
    case = configuration.case
    frequency_hz = case.frequency
    wavelength = c / frequency_hz
    radius_m = case.radius
    radius_lambda0 = radius_m / wavelength
    epsilon_r_interior = case.epsilon_interior / epsilon_0

    field_expressions = tuple(
        f"{physics_tag}.{comp}" for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )

    model = _build_omega_sphere_model(
        client,
        frequency_hz=frequency_hz,
        radius_lambda0=radius_lambda0,
        epsilon_r_interior=epsilon_r_interior,
        source_current_moment_am=source_current_moment_am,
        chi_em=chi_em,
        y_ee_imag=y_ee_imag,
        z_mm_imag=z_mm_imag,
        air_extent_lambda0=air_extent_lambda0,
        pml_thickness_lambda0=pml_thickness_lambda0,
        mesh_preset=mesh_preset,
        boundary_elements_per_wavelength=boundary_elements_per_wavelength,
        linear_solver=linear_solver,
        physics_tag=physics_tag,
    )

    try:
        if mesh_only:
            print("Mesh-only requested. Saving model and skipping solve.", flush=True)
            model_path = run_dir / "model.mph"
            model.save(model_path)
            mesh = model.java.component("comp1").mesh("mesh1")
            return {
                "mesh_elements": int(mesh.getNumElem()),
                "mesh_vertices": int(mesh.getNumVertex()),
                "status": "mesh_only",
                "physics_interface": physics_tag,
                "files": {"model": "model.mph"}
            }

        print("Solving omega-biisotropic sphere model...", flush=True)
        model.solve("Study 1")
        model_path = run_dir / "model.mph"
        model.save(model_path)
        print(f"Solved model saved: {model_path}", flush=True)

        fields_dir = run_dir / "fields"
        plots_dir = run_dir / "plots"
        fields_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Scale factor converting arbitrary prescribed moment to physical I*Delta_z = 1 A*m
        scale_factor = -1j * case.eta_interior / (4.0 * np.pi * case.k_interior)

        # 1. Radial cut extraction
        theta_rad = np.deg2rad(sample_theta_degrees)
        phi_rad = np.deg2rad(sample_phi_degrees)
        radius = np.linspace(
            sample_min_radii * radius_m,
            sample_max_radii * radius_m,
            sample_points,
        )
        x_rad = radius * np.sin(theta_rad) * np.cos(phi_rad)
        y_rad = radius * np.sin(theta_rad) * np.sin(phi_rad)
        z_rad = radius * np.cos(theta_rad)
        coordinates_rad = np.vstack((x_rad, y_rad, z_rad))

        print(
            f"Extracting fields at {sample_points} radial points "
            f"(theta={sample_theta_degrees:.1f} deg, phi={sample_phi_degrees:.1f} deg)...",
            flush=True,
        )
        computed_rad = _interpolate_fields(
            model, coordinates_rad, field_expressions=field_expressions
        )
        comp_e_rad = computed_rad[:3]
        comp_h_rad = computed_rad[3:]

        prescribed_e_sph_rad, prescribed_h_sph_rad = prescribed_total_fields(
            radius,
            np.full_like(radius, theta_rad),
            np.full_like(radius, phi_rad),
            case,
        )
        prescribed_e_sph_rad *= scale_factor
        prescribed_h_sph_rad *= scale_factor

        ref_e_rad = spherical_to_cartesian(
            prescribed_e_sph_rad,
            np.full_like(radius, theta_rad),
            np.full_like(radius, phi_rad),
        )
        ref_h_rad = spherical_to_cartesian(
            prescribed_h_sph_rad,
            np.full_like(radius, theta_rad),
            np.full_like(radius, phi_rad),
        )

        region_rad = np.where(radius < radius_m, 1, 2)
        computed_radial_path = fields_dir / "comsol_radial.csv"
        prescribed_radial_path = fields_dir / "prescribed_radial.csv"
        write_field_csv(
            computed_radial_path,
            x_rad,
            y_rad,
            z_rad,
            region_rad,
            comp_e_rad,
            comp_h_rad,
        )
        write_field_csv(
            prescribed_radial_path,
            x_rad,
            y_rad,
            z_rad,
            region_rad,
            ref_e_rad,
            ref_h_rad,
        )

        plot_radial_path = plots_dir / "omega_sphere_radial.png"
        _write_omega_sphere_radial_plot(
            plot_radial_path,
            radius,
            radius_m,
            sample_theta_degrees,
            sample_phi_degrees,
            comp_e_rad,
            comp_h_rad,
            ref_e_rad,
            ref_h_rad,
        )

        # 2. 2D zx-plane extraction
        span_m = zx_span_radii * radius_m
        x_axis = np.linspace(-span_m, span_m, zx_grid_size)
        z_axis = np.linspace(-span_m, span_m, zx_grid_size)
        x_grid, z_grid = np.meshgrid(x_axis, z_axis, indexing="xy")
        x_2d = x_grid.ravel()
        y_2d = np.zeros_like(x_2d)
        z_2d = z_grid.ravel()
        coordinates_2d = np.vstack((x_2d, y_2d, z_2d))
        radius_2d = np.sqrt(x_2d**2 + y_2d**2 + z_2d**2)
        radius_safe_2d = np.where(radius_2d == 0.0, 1.0, radius_2d)
        theta_2d = np.arccos(np.clip(z_2d / radius_safe_2d, -1.0, 1.0))
        phi_2d = np.arctan2(y_2d, x_2d)

        print(
            f"Extracting 2D fields at {zx_grid_size}x{zx_grid_size} "
            f"({coordinates_2d.shape[1]} points) in the zx plane...",
            flush=True,
        )
        computed_2d = _interpolate_fields(
            model, coordinates_2d, field_expressions=field_expressions
        )
        comp_e_2d = computed_2d[:3]
        comp_h_2d = computed_2d[3:]

        prescribed_e_sph_2d, prescribed_h_sph_2d = prescribed_total_fields(
            radius_2d, theta_2d, phi_2d, case
        )
        prescribed_e_sph_2d *= scale_factor
        prescribed_h_sph_2d *= scale_factor

        ref_e_2d = spherical_to_cartesian(prescribed_e_sph_2d, theta_2d, phi_2d)
        ref_h_2d = spherical_to_cartesian(prescribed_h_sph_2d, theta_2d, phi_2d)

        region_2d = np.where(radius_2d < radius_m, 1, 2)
        computed_zx_path = fields_dir / "comsol_zx.csv"
        prescribed_zx_path = fields_dir / "prescribed_zx.csv"
        write_field_csv(
            computed_zx_path,
            x_2d,
            y_2d,
            z_2d,
            region_2d,
            comp_e_2d,
            comp_h_2d,
        )
        write_field_csv(
            prescribed_zx_path,
            x_2d,
            y_2d,
            z_2d,
            region_2d,
            ref_e_2d,
            ref_h_2d,
        )

        plot_zx_path = plots_dir / "omega_sphere_zx.png"
        _write_omega_sphere_zx_plot(
            plot_zx_path,
            x_2d,
            z_2d,
            (zx_grid_size, zx_grid_size),
            radius_m,
            comp_e_2d,
            comp_h_2d,
            ref_e_2d,
            ref_h_2d,
        )

        # 3. Compute relative L2 errors
        rad_mask = radius >= 0.2 * radius_m
        rad_int = (radius >= 0.2 * radius_m) & (radius < radius_m)
        rad_ext = radius >= radius_m

        mask_2d = radius_2d >= 0.2 * radius_m
        int_2d = (radius_2d >= 0.2 * radius_m) & (radius_2d < radius_m)
        ext_2d = radius_2d >= radius_m

        mesh = model.java.component("comp1").mesh("mesh1")

        resolved_configuration = configuration.to_dict()
        resolved_configuration["comsol"] = {
            "stage": "omega_sphere",
            "frequency_hz": frequency_hz,
            "radius_lambda0": radius_lambda0,
            "epsilon_r_interior": epsilon_r_interior,
            "air_extent_lambda0": air_extent_lambda0,
            "pml_thickness_lambda0": pml_thickness_lambda0,
            "mesh_preset": mesh_preset,
            "boundary_elements_per_wavelength": boundary_elements_per_wavelength,
            "linear_solver": linear_solver,
            "physics_interface": physics_tag,
            "source_current_moment_am": source_current_moment_am,
            "chi_em": chi_em,
            "y_ee_imag": y_ee_imag,
            "z_mm_imag": z_mm_imag,
            "sample_theta_degrees": sample_theta_degrees,
            "sample_phi_degrees": sample_phi_degrees,
            "sample_min_radii": sample_min_radii,
            "sample_max_radii": sample_max_radii,
            "sample_points": sample_points,
            "zx_grid_size": zx_grid_size,
            "zx_span_radii": zx_span_radii,
        }
        with (run_dir / "configuration.json").open("w", encoding="utf-8") as stream:
            json.dump(resolved_configuration, stream, indent=2)
            stream.write("\n")

        return {
            "frequency_hz": frequency_hz,
            "wavelength_m": wavelength,
            "radius_m": radius_m,
            "radius_lambda0": radius_lambda0,
            "epsilon_r_interior": epsilon_r_interior,
            "mesh_preset": mesh_preset,
            "mesh_elements": int(mesh.getNumElem()),
            "mesh_vertices": int(mesh.getNumVertex()),
            "physics_interface": physics_tag,
            "chi_em": chi_em,
            "y_ee_imag": y_ee_imag,
            "z_mm_imag": z_mm_imag,
            "sample_theta_degrees": sample_theta_degrees,
            "sample_phi_degrees": sample_phi_degrees,
            "relative_l2_error_electric_radial_masked": _relative_error(
                comp_e_rad[:, rad_mask], ref_e_rad[:, rad_mask]
            ),
            "relative_l2_error_magnetic_radial_masked": _relative_error(
                comp_h_rad[:, rad_mask], ref_h_rad[:, rad_mask]
            ),
            "relative_l2_error_electric_radial_interior": _relative_error(
                comp_e_rad[:, rad_int], ref_e_rad[:, rad_int]
            ),
            "relative_l2_error_electric_radial_exterior": _relative_error(
                comp_e_rad[:, rad_ext], ref_e_rad[:, rad_ext]
            ),
            "relative_l2_error_electric_zx_masked": _relative_error(
                comp_e_2d[:, mask_2d], ref_e_2d[:, mask_2d]
            ),
            "relative_l2_error_magnetic_zx_masked": _relative_error(
                comp_h_2d[:, mask_2d], ref_h_2d[:, mask_2d]
            ),
            "relative_l2_error_electric_zx_interior": _relative_error(
                comp_e_2d[:, int_2d], ref_e_2d[:, int_2d]
            ),
            "relative_l2_error_electric_zx_exterior": _relative_error(
                comp_e_2d[:, ext_2d], ref_e_2d[:, ext_2d]
            ),
            "files": {
                "configuration": "configuration.json",
                "model": "model.mph",
                "computed_radial": str(computed_radial_path.relative_to(run_dir)),
                "reference_radial": str(prescribed_radial_path.relative_to(run_dir)),
                "prescribed_radial": str(prescribed_radial_path.relative_to(run_dir)),
                "radial_plot": str(plot_radial_path.relative_to(run_dir)),
                "computed_zx": str(computed_zx_path.relative_to(run_dir)),
                "reference_zx": str(prescribed_zx_path.relative_to(run_dir)),
                "prescribed_zx": str(prescribed_zx_path.relative_to(run_dir)),
                "zx_plot": str(plot_zx_path.relative_to(run_dir)),
            },
        }
    finally:
        client.remove(model)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "homogeneous", "omega_sphere"),
        default="smoke",
    )
    parser.add_argument("--version", default="6.1")
    parser.add_argument(
        "--physics-interface",
        choices=("auto", "emw", "ewfd"),
        default="auto",
        help="Physics interface: 'auto' (ewfd for 5.x, emw for 6.x), 'emw' (RF), or 'ewfd' (Wave Optics)",
    )
    parser.add_argument("--cores", type=_positive_int, default=_default_cores())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIGURATION)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--run-name")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="Build the geometry and mesh, save the model, and exit without solving",
    )
    parser.add_argument(
        "--mesh-preset",
        type=_positive_int,
        default=4,
        help="COMSOL autoMeshSize preset (1=Extremely fine to 9=Extremely coarse; default 4=Fine)",
    )
    parser.add_argument(
        "--boundary-hmax-divisor",
        type=float,
        default=30.0,
        help="Divisor N for sphere boundary element sizing lambda0/N (default: 30.0)",
    )
    parser.add_argument(
        "--outer-radius",
        type=float,
        default=1.8,
        help="Outer sphere radius in free-space wavelengths (homogeneous stage, default 1.8)",
    )
    parser.add_argument(
        "--air-extent",
        type=float,
        default=2.0,
        help="Extent b of surrounding air box in free-space wavelengths (omega_sphere stage, default 2.0)",
    )
    parser.add_argument(
        "--pml-thickness",
        type=float,
        default=0.25,
        help="PML layer thickness in free-space wavelengths (omega_sphere stage, default 0.25)",
    )
    parser.add_argument(
        "--radial-theta",
        type=float,
        default=45.0,
        help="Observation polar angle theta in degrees for radial cut (default: 45.0)",
    )
    parser.add_argument(
        "--radial-phi",
        type=float,
        default=0.0,
        help="Observation azimuthal angle phi in degrees for radial cut (default: 0.0)",
    )
    parser.add_argument(
        "--radial-points",
        type=_positive_int,
        default=101,
        help="Number of radial cut sample points (default: 101)",
    )
    parser.add_argument(
        "--radial-min-radii",
        type=float,
        default=0.05,
        help="Minimum radial cut radius in sphere radii a (default: 0.05)",
    )
    parser.add_argument(
        "--radial-max-radii",
        type=float,
        default=2.0,
        help="Maximum radial cut radius in sphere radii a (default: 2.0)",
    )
    parser.add_argument(
        "--zx-grid-size",
        type=_positive_int,
        default=80,
        help="Grid points per axis for 2D zx plane (must be even; default 80)",
    )
    parser.add_argument(
        "--zx-span-radii",
        type=float,
        default=2.0,
        help="Half-span of 2D zx plane in sphere radii a (default: 2.0)",
    )
    parser.add_argument(
        "--xem",
        type=float,
        default=0.38123,
        help="GSTC magnetoelectric parameter Xem (default: 0.38123)",
    )
    parser.add_argument(
        "--yee",
        type=float,
        default=-0.0036679,
        help="GSTC electric admittance Yee imaginary part in S (default: -0.0036679)",
    )
    parser.add_argument(
        "--zmm",
        type=float,
        default=-184.0435,
        help="GSTC magnetic impedance Zmm imaginary part in ohm (default: -184.0435)",
    )
    parser.add_argument(
        "--source-moment",
        type=float,
        default=1.0,
        help="Point dipole current moment I*Delta_z in A*m (default: 1.0)",
    )
    parser.add_argument(
        "--linear-solver",
        choices=("mumps", "pardiso", "direct", "iterative", "bicgstab", "gmres"),
        default="mumps",
        help="Linear solver for the stationary study step (default: mumps)",
    )
    parser.add_argument(
        "--allow-outside-slurm",
        action="store_true",
        help="Allow a local workstation run without SLURM_JOB_ID.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    _patch_mph_for_legacy_comsol(arguments.version)
    _require_compute_context(arguments.allow_outside_slurm)
    run_name = arguments.run_name or arguments.stage
    run_dir = arguments.run_dir or default_run_directory(
        arguments.case_id, "comsol", run_name
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "backend": "comsol",
        "stage": arguments.stage,
        "status": "running",
        "case_id": arguments.case_id,
        "run_name": run_name,
        "started_at_utc": started_at.isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "mph_version": mph.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "detected_comsol_backends": _detected_backends(),
    }

    runtime_parent = os.environ.get("TMPDIR")
    try:
        with tempfile.TemporaryDirectory(
            prefix="sie_mom-comsol-", dir=runtime_parent
        ) as temporary_directory:
            server, client = _start_comsol(
                version=arguments.version,
                cores=arguments.cores,
                runtime_dir=Path(temporary_directory),
            )
            try:
                manifest.update(
                    {
                        "comsol_version": arguments.version,
                        "cores": arguments.cores,
                        "licensed_modules": client.modules(),
                    }
                )
                if arguments.stage == "smoke":
                    manifest.update(run_smoke_test(client))
                elif arguments.stage == "homogeneous":
                    manifest.update(
                        run_homogeneous_validation(
                            client,
                            configuration_path=arguments.config,
                            run_dir=run_dir,
                            outer_radius_lambda0=arguments.outer_radius,
                            mesh_preset=arguments.mesh_preset,
                            zx_grid_size=arguments.zx_grid_size,
                        )
                    )
                elif arguments.stage == "omega_sphere":
                    physics_tag = (
                        ("ewfd" if str(arguments.version).startswith("5.") else "emw")
                        if arguments.physics_interface == "auto"
                        else arguments.physics_interface
                    )
                    manifest.update(
                        run_omega_sphere_simulation(
                            client,
                            configuration_path=arguments.config,
                            run_dir=run_dir,
                            air_extent_lambda0=arguments.air_extent,
                            pml_thickness_lambda0=arguments.pml_thickness,
                            mesh_preset=arguments.mesh_preset,
                            boundary_elements_per_wavelength=arguments.boundary_hmax_divisor,
                            sample_theta_degrees=arguments.radial_theta,
                            sample_phi_degrees=arguments.radial_phi,
                            sample_min_radii=arguments.radial_min_radii,
                            sample_max_radii=arguments.radial_max_radii,
                            sample_points=arguments.radial_points,
                            zx_grid_size=arguments.zx_grid_size,
                            zx_span_radii=arguments.zx_span_radii,
                            source_current_moment_am=arguments.source_moment,
                            chi_em=arguments.xem,
                            y_ee_imag=arguments.yee,
                            z_mm_imag=arguments.zmm,
                            linear_solver=arguments.linear_solver,
                            mesh_only=arguments.mesh_only,
                            physics_tag=physics_tag,
                        )
                    )
            finally:
                _stop_comsol(server, client)
        manifest["status"] = "complete"
        return_code = 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        return_code = 1
    finally:
        finished_at = datetime.now(timezone.utc)
        manifest["finished_at_utc"] = finished_at.isoformat()
        manifest["elapsed_seconds"] = (finished_at - started_at).total_seconds()
        manifest_path = run_dir / "run.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        print(json.dumps(manifest, indent=2))
        print(f"COMSOL run manifest: {manifest_path}")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
