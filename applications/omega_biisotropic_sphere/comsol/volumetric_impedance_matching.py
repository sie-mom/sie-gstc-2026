"""COMSOL validation driver for the omega-biisotropic sphere example."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import mph
import numpy as np
from jpype.types import JArray, JDouble, JInt, JString
from scipy.constants import c, epsilon_0, mu_0


APPLICATION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APPLICATION_DIR))

from dipole_impedance_matching import (  # noqa: E402
    prescribed_total_fields,
    spherical_to_cartesian,
)
try:  # Support both direct script execution and package imports in tests.
    from .surface_profiles import (
        compile_scalar_omega_cap_profile,
        spherical_cap_geometry,
    )
except ImportError:  # pragma: no cover - exercised by direct COMSOL execution
    from surface_profiles import (  # type: ignore
        compile_scalar_omega_cap_profile,
        spherical_cap_geometry,
    )
from workflow import (  # noqa: E402
    CapPerturbationConfiguration,
    default_run_directory,
    load_workflow_configuration,
    write_field_csv,
)


DEFAULT_CONFIGURATION = (
    APPLICATION_DIR / "configurations" / "budhu_10ghz_eps7.json"
)
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


def _java_string_tensor(values: Sequence[Sequence[str]]) -> Any:
    """Flatten tensor rows into COMSOL's row-major Java ``String[]`` form."""
    return JArray(JString)([value for row in values for value in row])


def _tangential_relative_tensor(
    alpha: str, nx: str, ny: str, nz: str
) -> list[list[str]]:
    """Return ``I + alpha*(I - n*n^T)`` as Cartesian expressions."""
    return [
        [f"1+{alpha}*(1-{nx}^2)", f"-{alpha}*{nx}*{ny}", f"-{alpha}*{nx}*{nz}"],
        [f"-{alpha}*{ny}*{nx}", f"1+{alpha}*(1-{ny}^2)", f"-{alpha}*{ny}*{nz}"],
        [f"-{alpha}*{nz}*{nx}", f"-{alpha}*{nz}*{ny}", f"1+{alpha}*(1-{nz}^2)"],
    ]


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _cap_scale(value: str) -> float:
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise argparse.ArgumentTypeError("cap scale must lie in (0, 1]")
    return result


def _mesh_element_count(mesh: Any, element_type: str) -> int:
    """Return the number of mesh elements of one COMSOL element type."""
    try:
        return len(mesh.getElemEntity(element_type))
    except IndexError:
        return 0


def _resolved_global_hmin_divisor(
    *,
    global_hmax_divisor: float,
    interior_hmax_divisor: float | None,
    outer_shell_band_hmax_divisor: float | None,
    shell_mesh: str,
    shell_tangential_divisor: float,
    inner_shell_band_hmax_divisor: float | None = None,
    source_ball_hmax_divisor: float | None = None,
    cap_edge_hmax_divisor: float | None = None,
) -> float:
    """Choose a permissive global minimum size for all requested refinements."""
    divisors = [30.0, 5.0 * global_hmax_divisor]
    if interior_hmax_divisor is not None:
        divisors.append(5.0 * interior_hmax_divisor)
    if inner_shell_band_hmax_divisor is not None:
        divisors.append(5.0 * inner_shell_band_hmax_divisor)
    if source_ball_hmax_divisor is not None:
        divisors.append(5.0 * source_ball_hmax_divisor)
    if outer_shell_band_hmax_divisor is not None:
        divisors.append(5.0 * outer_shell_band_hmax_divisor)
    if cap_edge_hmax_divisor is not None:
        divisors.append(5.0 * cap_edge_hmax_divisor)
    if shell_mesh == "swept":
        divisors.append(5.0 * shell_tangential_divisor)
    return max(divisors)


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
    # mesh.run()

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
    shell_outer_radius: float,
    theta_deg: float,
    phi_deg: float,
    computed_electric: np.ndarray,
    computed_magnetic: np.ndarray,
    reference_electric: np.ndarray,
    reference_magnetic: np.ndarray,
    reference_label: str = "Analytical Prescribed",
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
        axis.plot(normalized_r, ref_vals, "k-", linewidth=2.0, label=reference_label)
        axis.plot(normalized_r, comp_vals, "r--", linewidth=1.8, label="COMSOL")
        axis.axvspan(
            1.0,
            shell_outer_radius / sphere_radius,
            color="tab:blue",
            alpha=0.12,
            label="Volumetric GSTC shell (excluded from error)",
        )
        axis.axvline(1.0, color="tab:blue", linestyle=":")
        axis.axvline(
            shell_outer_radius / sphere_radius,
            color="tab:blue",
            linestyle=":",
        )
        if log_scale:
            axis.set_yscale("log")
        axis.set_xlabel(r"$r/a$")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
        axis.legend()

    axes[0].set_title(
        rf"Omega Sphere Radial Cut ($\theta = {theta_deg:.1f}^\circ, \phi = {phi_deg:.1f}^\circ$)"
    )
    if not is_near_pole:
        shell_thickness = shell_outer_radius - sphere_radius
        away_from_shell = (radius < sphere_radius - shell_thickness) | (
            radius > shell_outer_radius + shell_thickness
        )
        linear_values = np.concatenate(
            (
                np.real(comp_e_x[away_from_shell]) / 1e5,
                np.real(ref_e_x[away_from_shell]) / 1e5,
            )
        )
        finite_linear_values = linear_values[np.isfinite(linear_values)]
        if finite_linear_values.size > 0:
            linear_limit = 1.05 * float(np.max(np.abs(finite_linear_values)))
            if linear_limit > 0.0:
                axes[2].set_ylim(-linear_limit, linear_limit)
                axes[2].text(
                    0.99,
                    0.02,
                    "y limits exclude a one-thickness shell buffer",
                    transform=axes[2].transAxes,
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    fontsize=8,
                )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _write_omega_sphere_zx_plot(
    path: Path,
    x: np.ndarray,
    z: np.ndarray,
    grid_shape: tuple[int, int],
    sphere_radius: float,
    shell_outer_radius: float,
    computed_electric: np.ndarray,
    computed_magnetic: np.ndarray,
    reference_electric: np.ndarray,
    reference_magnetic: np.ndarray,
    reference_label: str = "Analytical Prescribed",
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
    row_labels = (reference_label, "COMSOL")
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
            ax.add_patch(
                plt.Circle(
                    (0.0, 0.0),
                    shell_outer_radius / sphere_radius,
                    fill=False,
                    color="white" if cmap == "viridis" else "black",
                    linestyle=":",
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
    cap_perturbation: CapPerturbationConfiguration | None = None,
    air_extent_lambda0: float = 2.0,
    tau_thickness_lambda0: float = 0.01,
    pml_thickness_lambda0: float = 0.25,
    mesh_preset: int = 4,
    shell_mesh: str = "tetra",
    global_hmax_divisor: float = 6.0,
    interior_hmax_divisor: float | None = None,
    source_ball_radius_lambda0: float | None = None,
    source_ball_hmax_divisor: float | None = None,
    inner_shell_band_width_lambda0: float | None = None,
    inner_shell_band_hmax_divisor: float | None = None,
    outer_shell_band_width_lambda0: float | None = None,
    outer_shell_band_hmax_divisor: float | None = None,
    cap_edge_hmax_divisor: float | None = None,
    shell_tangential_divisor: float = 20.0,
    shell_radial_layers: int = 2,
    discretization: str = "linear",
    linear_solver: str = "iterative",
    physics_tag: str = "emw",
) -> mph.Model:
    """Build the omega-biisotropic sphere model with GSTC boundary and PML block."""
    if shell_mesh not in {"tetra", "swept"}:
        raise ValueError(f"Unsupported shell mesh strategy: {shell_mesh!r}")
    if global_hmax_divisor <= 0.0:
        raise ValueError("global_hmax_divisor must be positive")
    if interior_hmax_divisor is not None and interior_hmax_divisor <= 0.0:
        raise ValueError("interior_hmax_divisor must be positive when provided")
    if (source_ball_radius_lambda0 is None) != (
        source_ball_hmax_divisor is None
    ):
        raise ValueError(
            "source-refinement ball radius and hmax divisor must be provided together"
        )
    if source_ball_radius_lambda0 is not None and source_ball_radius_lambda0 <= 0.0:
        raise ValueError("source_ball_radius_lambda0 must be positive")
    if source_ball_hmax_divisor is not None and source_ball_hmax_divisor <= 0.0:
        raise ValueError("source_ball_hmax_divisor must be positive")
    if (inner_shell_band_width_lambda0 is None) != (
        inner_shell_band_hmax_divisor is None
    ):
        raise ValueError(
            "inner-shell band width and hmax divisor must be provided together"
        )
    if (
        inner_shell_band_width_lambda0 is not None
        and inner_shell_band_width_lambda0 <= 0.0
    ):
        raise ValueError("inner_shell_band_width_lambda0 must be positive")
    if (
        inner_shell_band_hmax_divisor is not None
        and inner_shell_band_hmax_divisor <= 0.0
    ):
        raise ValueError("inner_shell_band_hmax_divisor must be positive")
    if (outer_shell_band_width_lambda0 is None) != (
        outer_shell_band_hmax_divisor is None
    ):
        raise ValueError(
            "outer-shell band width and hmax divisor must be provided together"
        )
    if (
        outer_shell_band_width_lambda0 is not None
        and outer_shell_band_width_lambda0 <= 0.0
    ):
        raise ValueError("outer_shell_band_width_lambda0 must be positive")
    if (
        outer_shell_band_hmax_divisor is not None
        and outer_shell_band_hmax_divisor <= 0.0
    ):
        raise ValueError("outer_shell_band_hmax_divisor must be positive")
    if cap_edge_hmax_divisor is not None and cap_edge_hmax_divisor <= 0.0:
        raise ValueError("cap_edge_hmax_divisor must be positive")
    if cap_edge_hmax_divisor is not None and cap_perturbation is None:
        raise ValueError(
            "cap-edge refinement requires a configuration with cap_perturbation"
        )
    if cap_edge_hmax_divisor is not None and shell_mesh != "swept":
        raise ValueError("cap-edge refinement requires --shell-mesh swept")
    if (
        cap_edge_hmax_divisor is not None
        and cap_perturbation is not None
        and cap_perturbation.angular_radius_degrees >= 90.0
    ):
        raise ValueError(
            "cap-edge refinement currently requires a cap angular radius below "
            "90 degrees"
        )
    if shell_tangential_divisor <= 0.0:
        raise ValueError("shell_tangential_divisor must be positive")
    if shell_radial_layers <= 0:
        raise ValueError("shell_radial_layers must be positive")

    model = client.create("omega_sphere_physical")
    java = model.java
    java.label("Omega-biisotropic sphere GSTC simulation")

    wavelength = c / frequency_hz
    radius_m = radius_lambda0 * wavelength
    air_extent_m = air_extent_lambda0 * wavelength
    inner_shell_band_enabled = inner_shell_band_width_lambda0 is not None
    source_ball_enabled = source_ball_radius_lambda0 is not None
    outer_shell_band_enabled = outer_shell_band_width_lambda0 is not None
    cap_edge_refinement_enabled = cap_edge_hmax_divisor is not None
    if (
        inner_shell_band_width_lambda0 is not None
        and inner_shell_band_width_lambda0 >= radius_lambda0
    ):
        raise ValueError(
            "The inner-shell refinement band must be thinner than the "
            f"sphere radius ({radius_lambda0:.6g} lambda0)"
        )
    source_parent_radius_lambda0 = radius_lambda0 - (
        inner_shell_band_width_lambda0 or 0.0
    )
    if (
        source_ball_radius_lambda0 is not None
        and source_ball_radius_lambda0 >= source_parent_radius_lambda0
    ):
        raise ValueError(
            "The source-refinement ball must fit strictly inside the dielectric "
            "bulk: "
            f"ball radius={source_ball_radius_lambda0:.6g} lambda0, "
            f"available radius={source_parent_radius_lambda0:.6g} lambda0"
        )
    if outer_shell_band_enabled:
        outer_band_radius_lambda0 = (
            radius_lambda0
            + tau_thickness_lambda0
            + outer_shell_band_width_lambda0
        )
        if outer_band_radius_lambda0 >= air_extent_lambda0:
            raise ValueError(
                "The outer-shell refinement band must end strictly inside the "
                "non-PML air box: "
                f"band outer radius={outer_band_radius_lambda0:.6g} lambda0, "
                f"air extent={air_extent_lambda0:.6g} lambda0"
            )

    # Parameters
    java.param().set("f0", f"{frequency_hz:.17g}[Hz]")
    java.param().set("lambda0", "c_const/f0")
    java.param().set("k0", "2*pi/lambda0")
    java.param().set("a", f"{radius_lambda0:.17g}*lambda0")
    java.param().set("tau", f"{tau_thickness_lambda0:.17g}*lambda0")
    java.param().set("b", f"{air_extent_lambda0:.17g}*lambda0")
    java.param().set("t_pml", f"{pml_thickness_lambda0:.17g}*lambda0")
    if inner_shell_band_width_lambda0 is not None:
        java.param().set(
            "w_in", f"{inner_shell_band_width_lambda0:.17g}*lambda0"
        )
    if source_ball_radius_lambda0 is not None:
        java.param().set(
            "r_src", f"{source_ball_radius_lambda0:.17g}*lambda0"
        )
    if outer_shell_band_width_lambda0 is not None:
        java.param().set(
            "w_out", f"{outer_shell_band_width_lambda0:.17g}*lambda0"
        )
    java.param().set("er", f"{epsilon_r_interior:.17g}")
    java.param().set("pI", f"{source_current_moment_am:.17g}[A*m]")
    java.param().set("Xem", f"{chi_em:.17g}")
    java.param().set("Yee", f"-j*{abs(y_ee_imag):.17g}[S]")
    java.param().set("Zmm", f"-j*{abs(z_mm_imag):.17g}[ohm]")

    component = java.component().create("comp1", True)
    geometry = component.geom().create("geom1", 3)
    geometry.lengthUnit("m")

    # 1. Interior sphere
    geometry.create("sph1", "Sphere")
    sph1 = geometry.feature("sph1")
    sph1.label("Interior domain")
    sph1.set("r", "a")
    sph1.set("pos", _java_double_array((0.0, 0.0, 0.0)))
    sph1.set("selresult", True)
    sph1.set("selresultshow", "all")

    # Optional conforming dielectric annulus immediately inside the GSTC
    # shell.  It remains the same physical material as the core and exists
    # only to support a finer local tetrahedral mesh near the sheet.
    inner_shell_band_selection: str | None = None
    interior_bulk_feature = "sph1"
    interior_bulk_selection = "geom1_sph1_dom"
    shell_core_input: str | list[str] = "sph1"
    if inner_shell_band_enabled:
        geometry.create("sph_inner_bulk", "Sphere")
        inner_bulk_sphere = geometry.feature("sph_inner_bulk")
        inner_bulk_sphere.label("Inner bulk refinement radius")
        inner_bulk_sphere.set("r", "a - w_in")
        inner_bulk_sphere.set("pos", _java_double_array((0.0, 0.0, 0.0)))
        inner_bulk_sphere.set("selresult", True)
        inner_bulk_sphere.set("selresultshow", "all")

        geometry.create("dif_inner_band", "Difference")
        inner_band_difference = geometry.feature("dif_inner_band")
        inner_band_difference.label("Interior near-shell dielectric band")
        inner_band_difference.selection("input").set("sph1")
        inner_band_difference.selection("input2").set("sph_inner_bulk")
        inner_band_difference.set("keepsubtract", True)
        inner_band_difference.set("selresult", True)
        inner_band_difference.set("selresultshow", "all")

        inner_shell_band_selection = "geom1_dif_inner_band_dom"
        interior_bulk_feature = "sph_inner_bulk"
        interior_bulk_selection = "geom1_sph_inner_bulk_dom"
        shell_core_input = ["dif_inner_band", "sph_inner_bulk"]

    # Optional conforming ball around the singular point source. The ball and
    # surrounding dielectric have identical material properties; the internal
    # boundary exists only to localize the tetrahedral size control.
    source_ball_selection: str | None = None
    if source_ball_enabled:
        geometry.create("sph_source", "Sphere")
        source_sphere = geometry.feature("sph_source")
        source_sphere.label("Point-source refinement ball")
        source_sphere.set("r", "r_src")
        source_sphere.set("pos", _java_double_array((0.0, 0.0, 0.0)))
        source_sphere.set("selresult", True)
        source_sphere.set("selresultshow", "all")

        geometry.create("dif_source_bulk", "Difference")
        source_bulk_difference = geometry.feature("dif_source_bulk")
        source_bulk_difference.label("Dielectric bulk outside source ball")
        source_bulk_difference.selection("input").set(interior_bulk_feature)
        source_bulk_difference.selection("input2").set("sph_source")
        source_bulk_difference.set("keepsubtract", True)
        source_bulk_difference.set("selresult", True)
        source_bulk_difference.set("selresultshow", "all")

        interior_bulk_selection = "geom1_dif_source_bulk_dom"
        source_ball_selection = "geom1_sph_source_dom"
        shell_core_input = ["dif_source_bulk", "sph_source"]
        if inner_shell_band_enabled:
            shell_core_input.insert(0, "dif_inner_band")

    # 1b. Metasurface shell
    geometry.create("sph2", "Sphere")
    sph2 = geometry.feature("sph2")
    sph2.label("Omega metasurface shell")
    sph2.set("r", "a + tau")
    sph2.set("pos", _java_double_array((0.0, 0.0, 0.0)))
    sph2.set("selresult", True)
    sph2.set("selresultshow", "all")

    # Optional conforming air annulus immediately outside the GSTC shell.
    # Construct it before consuming sph2 in the shell Difference operation.
    # This is a mesh-control partition only; it retains the surrounding-air
    # material and the same electromagnetic wave equation as the bulk air.
    outer_shell_band_selection: str | None = None
    if outer_shell_band_enabled:
        geometry.create("sph_outer_band", "Sphere")
        outer_band_sphere = geometry.feature("sph_outer_band")
        outer_band_sphere.label("Outer near-shell refinement radius")
        outer_band_sphere.set("r", "a + tau + w_out")
        outer_band_sphere.set("pos", _java_double_array((0.0, 0.0, 0.0)))
        outer_band_sphere.set("selresult", True)
        outer_band_sphere.set("selresultshow", "all")

        geometry.create("dif_outer_band", "Difference")
        outer_band_difference = geometry.feature("dif_outer_band")
        outer_band_difference.label("Exterior near-shell air band")
        outer_band_difference.selection("input").set("sph_outer_band")
        outer_band_difference.selection("input2").set("sph2")
        outer_band_difference.set("keepsubtract", True)
        outer_band_difference.set("selresult", True)
        outer_band_difference.set("selresultshow", "all")
        outer_shell_band_selection = "geom1_dif_outer_band_dom"

    shell_geometry_feature = "sph2"
    cap_shell_selection: str | None = None
    cap_background_shell_selection: str | None = None
    if shell_mesh == "swept":
        # Build the shell as its own geometry object so only it is partitioned.
        # Keep the dielectric core objects as separate domains.
        geometry.create("dif_shell", "Difference")
        dif_shell = geometry.feature("dif_shell")
        dif_shell.label("Volumetric shell (outer sphere minus core)")
        dif_shell.selection("input").set("sph2")
        dif_shell.selection("input2").set(shell_core_input)
        dif_shell.set("keepsubtract", True)
        dif_shell.set("selresult", True)
        dif_shell.set("selresultshow", "all")

        partition_input = "dif_shell"
        if cap_edge_refinement_enabled:
            assert cap_perturbation is not None
            cap_center, _ = spherical_cap_geometry(cap_perturbation)
            cap_angle = cap_perturbation.angular_radius_degrees
            cone_height = (
                2.0
                * (air_extent_lambda0 + pml_thickness_lambda0)
                * wavelength
            )
            cone_radius = cone_height * np.tan(np.deg2rad(cap_angle))

            # A cone with its apex at the origin intersects both spherical
            # shell faces at the analytical cap rim.  Splitting the shell
            # makes that discontinuity a real geometry edge, instead of
            # allowing the coefficient jump to cut through source triangles.
            geometry.create("cone_cap", "Cone")
            cap_cone = geometry.feature("cone_cap")
            cap_cone.label("Exact spherical-cap partition cone")
            cap_cone.set("r", JDouble(cone_radius))
            cap_cone.set("h", JDouble(cone_height))
            cap_cone.set("ang", JDouble(cap_angle))
            cap_cone.set(
                "pos",
                _java_double_array(
                    tuple(cone_height * component for component in cap_center)
                ),
            )
            cap_cone.set(
                "axis",
                _java_double_array(tuple(-component for component in cap_center)),
            )

            geometry.create("int_shell_cap", "Intersection")
            cap_intersection = geometry.feature("int_shell_cap")
            cap_intersection.label("Volumetric shell inside cap")
            cap_intersection.selection("input").set(["dif_shell", "cone_cap"])
            cap_intersection.set("keep", True)
            cap_intersection.set("selresult", True)
            cap_intersection.set("selresultshow", "all")

            geometry.create("dif_shell_cap_background", "Difference")
            cap_background = geometry.feature("dif_shell_cap_background")
            cap_background.label("Volumetric shell outside cap")
            cap_background.selection("input").set("dif_shell")
            cap_background.selection("input2").set("cone_cap")
            cap_background.set("selresult", True)
            cap_background.set("selresultshow", "all")

            partition_input = ["int_shell_cap", "dif_shell_cap_background"]
            cap_shell_selection = "geom1_int_shell_cap_dom"
            cap_background_shell_selection = (
                "geom1_dif_shell_cap_background_dom"
            )

        # COMSOL 6.1 cannot sweep a closed hollow shell because its inner and
        # outer spherical boundaries are disconnected face components. Three
        # coordinate work planes split the shell into eight sweepable octants.
        for plane_tag, plane_name, partition_tag in (
            ("wp_shell_x", "yz", "par_shell_x"),
            ("wp_shell_y", "zx", "par_shell_y"),
            ("wp_shell_z", "xy", "par_shell_z"),
        ):
            geometry.create(plane_tag, "WorkPlane")
            work_plane = geometry.feature(plane_tag)
            work_plane.label(f"Shell partition plane {plane_name}")
            work_plane.set("planetype", "quick")
            work_plane.set("quickplane", plane_name)

            geometry.create(partition_tag, "Partition")
            partition = geometry.feature(partition_tag)
            partition.label(f"Partition shell with {plane_name} plane")
            partition.selection("input").set(partition_input)
            partition.set("partitionwith", "workplane")
            partition.set("workplane", plane_tag)
            partition.set("selresult", True)
            partition.set("selresultshow", "all")
            partition_input = partition_tag

        shell_geometry_feature = partition_input

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

    # 2-3. Core and shell selections. Swept mode uses selections generated by
    # the geometry features, so entity renumbering cannot invalidate them.
    shell_outer_faces_selection: str | None = None
    shell_inner_faces_selection: str | None = None
    outer_shell_faces: list[int] = []
    inner_shell_faces: list[int] = []
    cap_outer_rim_edges: list[int] = []
    inner_shell_band_domains: list[int] = []
    source_ball_domains: list[int] = []
    interior_is_partitioned = inner_shell_band_enabled or source_ball_enabled
    if interior_is_partitioned:
        inner_bulk_domains = [
            int(domain)
            for domain in component.selection(interior_bulk_selection).entities(
                JInt(3)
            )
        ]
        if inner_shell_band_enabled:
            assert inner_shell_band_selection is not None
            inner_shell_band_domains = [
                int(domain)
                for domain in component.selection(
                    inner_shell_band_selection
                ).entities(JInt(3))
            ]
        if source_ball_enabled:
            assert source_ball_selection is not None
            source_ball_domains = [
                int(domain)
                for domain in component.selection(source_ball_selection).entities(
                    JInt(3)
                )
            ]
        sphere_domains = sorted(
            set(inner_bulk_domains)
            | set(inner_shell_band_domains)
            | set(source_ball_domains)
        )
        if (
            not inner_bulk_domains
            or (inner_shell_band_enabled and not inner_shell_band_domains)
            or (source_ball_enabled and not source_ball_domains)
        ):
            raise RuntimeError(
                "Partitioned dielectric geometry selections are incomplete: "
                f"source ball={source_ball_domains}, bulk={inner_bulk_domains}, "
                f"near-shell band={inner_shell_band_domains}"
            )
        component.selection().create("sphere_domains", "Explicit")
        sphere_domain_selection_node = component.selection("sphere_domains")
        sphere_domain_selection_node.label("Complete dielectric interior")
        sphere_domain_selection_node.set(JArray(JInt)(sphere_domains))
        sphere_domain_selection = "sphere_domains"

    if shell_mesh == "swept":
        if not interior_is_partitioned:
            sphere_domain_selection = "geom1_sph1_dom"
        sphere_shell_selection = f"geom1_{shell_geometry_feature}_dom"
        shell_outer_faces_selection = "geom1_sph2_bnd"
        shell_inner_faces_selection = "geom1_sph1_bnd"

        if not interior_is_partitioned:
            sphere_domains = [
                int(d)
                for d in component.selection(sphere_domain_selection).entities(JInt(3))
            ]
        shell_domains = [
            int(d)
            for d in component.selection(sphere_shell_selection).entities(JInt(3))
        ]
        outer_shell_faces = [
            int(b)
            for b in component.selection(shell_outer_faces_selection).entities(JInt(2))
        ]
        inner_shell_faces = [
            int(b)
            for b in component.selection(shell_inner_faces_selection).entities(JInt(2))
        ]
        if not cap_edge_refinement_enabled and len(shell_domains) != 8:
            raise RuntimeError(
                "Expected the swept shell to contain 8 octant domains; "
                f"selection {sphere_shell_selection!r} contains {shell_domains}"
            )
        if not sphere_domains or not outer_shell_faces or not inner_shell_faces:
            raise RuntimeError(
                "Swept-shell geometry selections are incomplete: "
                f"core={sphere_domains}, outer={outer_shell_faces}, "
                f"inner={inner_shell_faces}"
            )
        if set(outer_shell_faces) & set(inner_shell_faces):
            raise RuntimeError("Inner and outer swept-shell face selections overlap")

        if cap_edge_refinement_enabled:
            assert cap_shell_selection is not None
            assert cap_background_shell_selection is not None
            cap_domains = {
                int(domain)
                for domain in component.selection(cap_shell_selection).entities(
                    JInt(3)
                )
            }
            background_domains = {
                int(domain)
                for domain in component.selection(
                    cap_background_shell_selection
                ).entities(JInt(3))
            }
            shell_domain_set = set(shell_domains)
            if (
                not cap_domains
                or not background_domains
                or cap_domains & background_domains
                or cap_domains | background_domains != shell_domain_set
            ):
                raise RuntimeError(
                    "Cap partition did not produce a disjoint cover of the shell: "
                    f"cap={sorted(cap_domains)}, "
                    f"background={sorted(background_domains)}, "
                    f"shell={shell_domains}"
                )

            def adjacent_entities(
                from_dimension: int,
                to_dimension: int,
                entities: Sequence[int],
            ) -> set[int]:
                adjacent: set[int] = set()
                for entity in entities:
                    adjacent.update(
                        int(item)
                        for item in geometry.getAdj(
                            JInt(from_dimension),
                            JInt(to_dimension),
                            JInt(entity),
                        )
                    )
                return adjacent

            cap_boundaries = adjacent_entities(3, 2, sorted(cap_domains))
            background_boundaries = adjacent_entities(
                3, 2, sorted(background_domains)
            )
            cap_interface_faces = cap_boundaries & background_boundaries
            cap_outer_faces = cap_boundaries & set(outer_shell_faces)
            interface_edges = adjacent_entities(
                2, 1, sorted(cap_interface_faces)
            )
            cap_outer_edges = adjacent_entities(2, 1, sorted(cap_outer_faces))
            cap_outer_rim_edges = sorted(interface_edges & cap_outer_edges)
            if not cap_interface_faces or not cap_outer_rim_edges:
                raise RuntimeError(
                    "Could not resolve the exact cap interface and its outer "
                    "spherical rim: "
                    f"interface faces={sorted(cap_interface_faces)}, "
                    f"outer cap faces={sorted(cap_outer_faces)}, "
                    f"rim edges={cap_outer_rim_edges}"
                )
            print(
                "Exact cap-edge partition: "
                f"cap domains={sorted(cap_domains)}, "
                f"background shell domains={sorted(background_domains)}, "
                f"interface faces={sorted(cap_interface_faces)}, "
                f"outer rim edges={cap_outer_rim_edges}",
                flush=True,
            )
        print(
            "Swept shell selections: "
            f"domains={shell_domains}, outer faces={outer_shell_faces}, "
            f"inner faces={inner_shell_faces}",
            flush=True,
        )
    else:
        if not interior_is_partitioned:
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
            sphere_domain_selection = "sphere_domain"

        # Target a point strictly inside the shell: (a + 0.5*tau, 0, 0).
        tau_m = tau_thickness_lambda0 * wavelength
        component.selection().create("sphere_shell_ball", "Ball")
        sphere_shell_ball = component.selection("sphere_shell_ball")
        sphere_shell_ball.label("Metasurface sphere shell probe")
        sphere_shell_ball.set("entitydim", JInt(3))
        sphere_shell_ball.set("r", JDouble(0.25 * tau_m))
        sphere_shell_ball.set("posx", JDouble(radius_m + 0.5 * tau_m))
        sphere_shell_ball.set("posy", JDouble(0.0))
        sphere_shell_ball.set("posz", JDouble(0.0))

        shell_domains = [int(d) for d in sphere_shell_ball.entities(JInt(3))]
        if not shell_domains:
            raise RuntimeError(
                "Could not find shell domain via probe at "
                f"x={radius_m + 0.5 * tau_m}"
            )
        if len(shell_domains) != 1:
            raise RuntimeError(
                f"Expected one unpartitioned tetrahedral shell domain, got {shell_domains}"
            )

        component.selection().create("sphere_shell", "Explicit")
        sphere_shell = component.selection("sphere_shell")
        sphere_shell.label("Metasurface shell explicit domain")
        sphere_shell.set(shell_domains)
        sphere_shell_selection = "sphere_shell"
        print(
            f"Tetrahedral shell selections: core={sphere_domains}, "
            f"shell={shell_domains}",
            flush=True,
        )

    if inner_shell_band_enabled:
        print(
            "Interior near-shell refinement band: "
            f"domains={inner_shell_band_domains}, width="
            f"{inner_shell_band_width_lambda0:.6g} lambda0, hmax=lambda0/"
            f"{inner_shell_band_hmax_divisor:.6g}",
            flush=True,
        )

    if source_ball_enabled:
        print(
            "Point-source refinement ball: "
            f"domains={source_ball_domains}, radius="
            f"{source_ball_radius_lambda0:.6g} lambda0, hmax=lambda0/"
            f"{source_ball_hmax_divisor:.6g}",
            flush=True,
        )

    outer_shell_band_domains: list[int] = []
    if outer_shell_band_selection is not None:
        outer_shell_band_domains = [
            int(domain)
            for domain in component.selection(outer_shell_band_selection).entities(
                JInt(3)
            )
        ]
        if not outer_shell_band_domains:
            raise RuntimeError(
                "The exterior near-shell refinement band selection is empty"
            )
        print(
            "Exterior near-shell refinement band: "
            f"domains={outer_shell_band_domains}, width="
            f"{outer_shell_band_width_lambda0:.6g} lambda0, hmax=lambda0/"
            f"{outer_shell_band_hmax_divisor:.6g}",
            flush=True,
        )

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
    mat2.selection().named(sphere_domain_selection)
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
    if discretization == "linear":
        try:
            phys.prop("ShapeProperty").set("order_electricfield", 1)
        except Exception:
            try:
                phys.prop("ShapeProperty").set("order_electricfield", "1")
            except Exception:
                pass
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

    # Volumetric Shell Wave Equation (weq2)
    phys.create("weq2", "WaveEquationElectric", 3)
    phys.feature("weq2").label("GSTC Volumetric Shell")
    phys.feature("weq2").selection().named(sphere_shell_selection)
    if physics_tag == "ewfd":
        try:
            phys.feature("weq2").set("DisplacementFieldModel", "RelativePermittivity")
        except Exception:
            pass

    # Outward radial unit normal on a sphere centered at (0,0,0).  The shell
    # does not contain the origin, so these expressions are nonsingular on the
    # domains where the volumetric constitutive law is active.
    r_sph = "sqrt(x^2 + y^2 + z^2)"
    nx = f"(x/{r_sph})"
    ny = f"(y/{r_sph})"
    nz = f"(z/{r_sph})"

    yee_expression = "Yee"
    zmm_expression = "Zmm"
    xem_expression = "Xem"
    if cap_perturbation is not None:
        profile = compile_scalar_omega_cap_profile(
            cap_perturbation,
            nx=nx,
            ny=ny,
            nz=nz,
        )
        component.variable().create("var_gstc_profile")
        profile_variables = component.variable("var_gstc_profile")
        profile_variables.label("Spatial GSTC surface profile")
        profile_variables.selection().named(sphere_shell_selection)
        for name, expression in profile.variables.items():
            profile_variables.set(name, expression)
        yee_expression = "gstc_Yee"
        zmm_expression = "gstc_Zmm"
        xem_expression = "gstc_Xem"

    # The direct sheet blocks Yee*I_t and Zmm*I_t are isotropic only within
    # the tangent plane.  Embed I_t in global Cartesian coordinates with
    # P_t = I - n*n^T, leaving the background radial eigenvalue equal to one:
    #   eps_rs = I + Yee/(j*omega*eps0*tau) * P_t
    #   mu_rs  = I + Zmm/(j*omega*mu0*tau)  * P_t
    alpha_e = (
        f"({yee_expression}/(j*{physics_tag}.omega*epsilon0_const*tau))"
    )
    alpha_m = f"({zmm_expression}/(j*{physics_tag}.omega*mu0_const*tau))"

    phys.feature("weq2").set("epsilonr_mat", "userdef")
    phys.feature("weq2").set(
        "epsilonr",
        _java_string_tensor(_tangential_relative_tensor(alpha_e, nx, ny, nz)),
    )

    phys.feature("weq2").set("mur_mat", "userdef")
    phys.feature("weq2").set(
        "mur", _java_string_tensor(_tangential_relative_tensor(alpha_m, nx, ny, nz))
    )

    # Native COMSOL Equation View Overrides on weq2 via featureInfo("info")
    info = phys.feature("weq2").featureInfo("info")

    # Reciprocal Omega Metasurface Constitutive Cross-Coupling:
    # J_s = Yee * E_t - Xem * (n x H_t)
    # M_s = Zmm * H_t - Xem * (n x E_t)
    #
    # Across a volumetric thin shell of thickness tau:
    # J_v = J_s / tau = (Yee / tau)*E_t - (Xem / tau)*(n x H_t)
    # M_v = M_s / tau = (Zmm / tau)*H_t - (Xem / tau)*(n x E_t)
    #
    # In Ampere's Law: J_v = i*omega*P_v
    # => P_cross = - (Xem / (j*omega*tau)) * (n x H)
    #
    # In Faraday's Law: -j*omega*B = -j*omega*mu0*mur*H - M_v
    # => B = mu0*mur*H - (Xem / (j*omega*tau)) * (n x E)
    # => H_cross = + (Xem / (j*omega*tau*mu0*mur)) * (n x E)
    # => dHdt_cross = j*omega*H_cross = + (Xem / (tau*mu0*mur)) * (n x E)

    # Cross product components: (n x H)
    nxH_x = f"({ny}*{physics_tag}.Hz - {nz}*{physics_tag}.Hy)"
    nxH_y = f"({nz}*{physics_tag}.Hx - {nx}*{physics_tag}.Hz)"
    nxH_z = f"({nx}*{physics_tag}.Hy - {ny}*{physics_tag}.Hx)"

    # Cross product components: (n x E)
    nxE_x = f"({ny}*{physics_tag}.Ez - {nz}*{physics_tag}.Ey)"
    nxE_y = f"({nz}*{physics_tag}.Ex - {nx}*{physics_tag}.Ez)"
    nxE_z = f"({nx}*{physics_tag}.Ey - {ny}*{physics_tag}.Ex)"

    # Prefactors
    pref_P = f"-({xem_expression} / (j*{physics_tag}.omega*tau))"
    pref_H = f"({xem_expression} / (j*{physics_tag}.omega*tau*mu0_const))"
    pref_dHdt = f"({xem_expression} / (tau*mu0_const))"

    # Polarization P overrides
    info.set(f"{physics_tag}.Px", 0, [f"epsilon0_const*({physics_tag}.epsilonrxx*{physics_tag}.Ex+{physics_tag}.epsilonrxy*{physics_tag}.Ey+{physics_tag}.epsilonrxz*{physics_tag}.Ez-{physics_tag}.Ex) + {pref_P}*{nxH_x}"])
    info.set(f"{physics_tag}.Py", 0, [f"epsilon0_const*({physics_tag}.epsilonryx*{physics_tag}.Ex+{physics_tag}.epsilonryy*{physics_tag}.Ey+{physics_tag}.epsilonryz*{physics_tag}.Ez-{physics_tag}.Ey) + {pref_P}*{nxH_y}"])
    info.set(f"{physics_tag}.Pz", 0, [f"epsilon0_const*({physics_tag}.epsilonrzx*{physics_tag}.Ex+{physics_tag}.epsilonrzy*{physics_tag}.Ey+{physics_tag}.epsilonrzz*{physics_tag}.Ez-{physics_tag}.Ez) + {pref_P}*{nxH_z}"])

    # Magnetic field H overrides (adding H_cross)
    Hx_cross = f"{pref_H} * ({physics_tag}.murinvxx*{nxE_x} + {physics_tag}.murinvxy*{nxE_y} + {physics_tag}.murinvxz*{nxE_z})"
    Hy_cross = f"{pref_H} * ({physics_tag}.murinvyx*{nxE_x} + {physics_tag}.murinvyy*{nxE_y} + {physics_tag}.murinvyz*{nxE_z})"
    Hz_cross = f"{pref_H} * ({physics_tag}.murinvzx*{nxE_x} + {physics_tag}.murinvzy*{nxE_y} + {physics_tag}.murinvzz*{nxE_z})"

    info.set(f"{physics_tag}.Hx", 0, [f"({physics_tag}.murinvxx*{physics_tag}.Bx+{physics_tag}.murinvxy*{physics_tag}.By+{physics_tag}.murinvxz*{physics_tag}.Bz)/mu0_const + {Hx_cross}"])
    info.set(f"{physics_tag}.Hy", 0, [f"({physics_tag}.murinvyx*{physics_tag}.Bx+{physics_tag}.murinvyy*{physics_tag}.By+{physics_tag}.murinvyz*{physics_tag}.Bz)/mu0_const + {Hy_cross}"])
    info.set(f"{physics_tag}.Hz", 0, [f"({physics_tag}.murinvzx*{physics_tag}.Bx+{physics_tag}.murinvzy*{physics_tag}.By+{physics_tag}.murinvzz*{physics_tag}.Bz)/mu0_const + {Hz_cross}"])

    # Time derivative of H overrides (crucial for weak form)
    dHdt_cross_x = f"{pref_dHdt} * ({physics_tag}.murinvxx*{nxE_x} + {physics_tag}.murinvxy*{nxE_y} + {physics_tag}.murinvxz*{nxE_z})"
    dHdt_cross_y = f"{pref_dHdt} * ({physics_tag}.murinvyx*{nxE_x} + {physics_tag}.murinvyy*{nxE_y} + {physics_tag}.murinvyz*{nxE_z})"
    dHdt_cross_z = f"{pref_dHdt} * ({physics_tag}.murinvzx*{nxE_x} + {physics_tag}.murinvzy*{nxE_y} + {physics_tag}.murinvzz*{nxE_z})"

    info.set(f"{physics_tag}.dHdtx", 0, [f"({physics_tag}.murinvxx*{physics_tag}.dBdtx+{physics_tag}.murinvxy*{physics_tag}.dBdty+{physics_tag}.murinvxz*{physics_tag}.dBdtz)/mu0_const + {dHdt_cross_x}"])
    info.set(f"{physics_tag}.dHdty", 0, [f"({physics_tag}.murinvyx*{physics_tag}.dBdtx+{physics_tag}.murinvyy*{physics_tag}.dBdty+{physics_tag}.murinvyz*{physics_tag}.dBdtz)/mu0_const + {dHdt_cross_y}"])
    info.set(f"{physics_tag}.dHdtz", 0, [f"({physics_tag}.murinvzx*{physics_tag}.dBdtx+{physics_tag}.murinvzy*{physics_tag}.dBdty+{physics_tag}.murinvzz*{physics_tag}.dBdtz)/mu0_const + {dHdt_cross_z}"])

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
    # To avoid COMSOL evaluating physics properties and crashing during sequence generation,
    # we build a fully explicit manual mesh sequence.
    component.mesh().create("mesh1")
    mesh = component.mesh("mesh1")

    # Create manual size node
    mesh.create("size1", "Size")
    mesh.feature("size1").set("custom", True)
    mesh.feature("size1").set(
        "hmax", f"lambda0/{global_hmax_divisor:.17g}"
    )
    global_hmin_divisor = _resolved_global_hmin_divisor(
        global_hmax_divisor=global_hmax_divisor,
        interior_hmax_divisor=interior_hmax_divisor,
        source_ball_hmax_divisor=source_ball_hmax_divisor,
        inner_shell_band_hmax_divisor=inner_shell_band_hmax_divisor,
        outer_shell_band_hmax_divisor=outer_shell_band_hmax_divisor,
        shell_mesh=shell_mesh,
        shell_tangential_divisor=shell_tangential_divisor,
        cap_edge_hmax_divisor=cap_edge_hmax_divisor,
    )
    mesh.feature("size1").set(
        "hmin", f"lambda0/{global_hmin_divisor:.17g}"
    )

    if interior_hmax_divisor is not None:
        mesh.create("size_core", "Size")
        core_size = mesh.feature("size_core")
        core_size.label("Dielectric interior local-wavelength refinement")
        core_size.selection().geom("geom1", JInt(3))
        core_size.selection().named(sphere_domain_selection)
        core_size.set("custom", True)
        core_size.set("hmaxactive", True)
        core_size.set("hmax", f"lambda0/{interior_hmax_divisor:.17g}")
        core_size.set("hminactive", True)
        core_size.set("hmin", f"lambda0/{5.0 * interior_hmax_divisor:.17g}")

    if source_ball_selection is not None:
        assert source_ball_hmax_divisor is not None
        mesh.create("size_source_ball", "Size")
        source_ball_size = mesh.feature("size_source_ball")
        source_ball_size.label("Point-source refinement ball")
        source_ball_size.selection().geom("geom1", JInt(3))
        source_ball_size.selection().named(source_ball_selection)
        source_ball_size.set("custom", True)
        source_ball_size.set("hmaxactive", True)
        source_ball_size.set(
            "hmax", f"lambda0/{source_ball_hmax_divisor:.17g}"
        )
        source_ball_size.set("hminactive", True)
        source_ball_size.set(
            "hmin", f"lambda0/{5.0 * source_ball_hmax_divisor:.17g}"
        )

    if inner_shell_band_selection is not None:
        assert inner_shell_band_hmax_divisor is not None
        mesh.create("size_inner_band", "Size")
        inner_band_size = mesh.feature("size_inner_band")
        inner_band_size.label("Interior near-shell dielectric refinement")
        inner_band_size.selection().geom("geom1", JInt(3))
        inner_band_size.selection().named(inner_shell_band_selection)
        inner_band_size.set("custom", True)
        inner_band_size.set("hmaxactive", True)
        inner_band_size.set(
            "hmax", f"lambda0/{inner_shell_band_hmax_divisor:.17g}"
        )
        inner_band_size.set("hminactive", True)
        inner_band_size.set(
            "hmin", f"lambda0/{5.0 * inner_shell_band_hmax_divisor:.17g}"
        )

    if outer_shell_band_selection is not None:
        assert outer_shell_band_hmax_divisor is not None
        mesh.create("size_outer_band", "Size")
        outer_band_size = mesh.feature("size_outer_band")
        outer_band_size.label("Exterior near-shell air refinement")
        outer_band_size.selection().geom("geom1", JInt(3))
        outer_band_size.selection().named(outer_shell_band_selection)
        outer_band_size.set("custom", True)
        outer_band_size.set("hmaxactive", True)
        outer_band_size.set(
            "hmax", f"lambda0/{outer_shell_band_hmax_divisor:.17g}"
        )
        outer_band_size.set("hminactive", True)
        outer_band_size.set(
            "hmin", f"lambda0/{5.0 * outer_shell_band_hmax_divisor:.17g}"
        )

    if cap_edge_refinement_enabled:
        assert cap_edge_hmax_divisor is not None
        if not cap_outer_rim_edges:
            raise RuntimeError("The cap-edge refinement selection is empty")
        mesh.create("size_cap_rim", "Size")
        cap_rim_size = mesh.feature("size_cap_rim")
        cap_rim_size.label("Exact cap outer-rim refinement")
        cap_rim_size.selection().geom("geom1", JInt(1))
        cap_rim_size.selection().set(JArray(JInt)(cap_outer_rim_edges))
        cap_rim_size.set("custom", True)
        cap_rim_size.set("hmaxactive", True)
        cap_rim_size.set("hmax", f"lambda0/{cap_edge_hmax_divisor:.17g}")
        cap_rim_size.set("hminactive", True)
        cap_rim_size.set(
            "hmin", f"lambda0/{5.0 * cap_edge_hmax_divisor:.17g}"
        )

    if shell_mesh == "tetra":
        mesh.create("size_shell", "Size")
        size_sh = mesh.feature("size_shell")
        size_sh.label("Metasurface volumetric shell refinement")
        size_sh.selection().geom("geom1", JInt(3))
        size_sh.selection().named(sphere_shell_selection)
        size_sh.set("custom", True)
        size_sh.set("hmaxactive", True)
        size_sh.set("hmax", "tau")
        size_sh.set("hminactive", True)
        size_sh.set("hmin", "tau/5")
    else:
        assert shell_outer_faces_selection is not None
        assert shell_inner_faces_selection is not None

        # Mesh the larger, outer spherical patches first. Sweeping inward maps
        # this mesh onto slightly smaller inner patches and keeps the maximum
        # tangential edge size conservative on both shell interfaces.
        mesh.create("ftri_shell", "FreeTri")
        free_tri = mesh.feature("ftri_shell")
        free_tri.label("Outer shell source-face triangles")
        free_tri.selection().named(shell_outer_faces_selection)
        free_tri.create("size_shell_tan", "Size")
        tangential_size = free_tri.feature("size_shell_tan")
        tangential_size.label("Shell tangential resolution")
        tangential_size.selection().named(shell_outer_faces_selection)
        tangential_size.set("custom", True)
        tangential_size.set("hmaxactive", True)
        tangential_size.set(
            "hmax", f"lambda0/{shell_tangential_divisor:.17g}"
        )
        tangential_size.set("hminactive", True)
        tangential_size.set(
            "hmin", f"lambda0/{5.0 * shell_tangential_divisor:.17g}"
        )

        mesh.create("swe_shell", "Sweep")
        swept = mesh.feature("swe_shell")
        swept.label("Swept volumetric shell refinement")
        swept.selection().named(sphere_shell_selection)
        swept.selection("sourceface").named(shell_outer_faces_selection)
        swept.selection("targetface").named(shell_inner_faces_selection)
        swept.set("facemethod", "tri")
        swept.set("sweeppath", "straight")
        swept.set("targetmesh", "auto")
        swept.create("dist_shell", "Distribution")
        distribution = swept.feature("dist_shell")
        distribution.label("Uniform radial shell layers")
        distribution.selection().named(sphere_shell_selection)
        distribution.set("type", "predefined")
        distribution.set("elemcount", JInt(shell_radial_layers))
        distribution.set("elemratio", JDouble(1.0))
        distribution.set("method", "arithmetic")

    # Mesh every domain not already handled by the swept-shell operations.
    mesh.create("ftet1", "FreeTet")

    mesh.run()

    element_count = int(mesh.getNumElem())
    tetrahedron_count = _mesh_element_count(mesh, "tet")
    prism_count = _mesh_element_count(mesh, "prism")
    print(
        "Omega sphere mesh: "
        f"{element_count} total elements "
        f"({tetrahedron_count} tetrahedra, {prism_count} prisms)",
        flush=True,
    )
    if element_count == 0:
        raise RuntimeError("Omega sphere mesh is empty; refusing to solve")
    if shell_mesh == "swept":
        triangle_entities = [int(entity) for entity in mesh.getElemEntity("tri")]
        outer_shell_face_set = set(outer_shell_faces)
        source_triangle_count = sum(
            entity in outer_shell_face_set for entity in triangle_entities
        )
        expected_prism_count = source_triangle_count * shell_radial_layers
        if prism_count != expected_prism_count:
            raise RuntimeError(
                "Swept-shell prism count does not match the source-face mesh and "
                f"radial layers: prisms={prism_count}, source triangles="
                f"{source_triangle_count}, radial layers={shell_radial_layers}"
            )
        prism_domains = {
            int(entity) for entity in mesh.getElemEntity("prism")
        }
        if prism_domains != set(shell_domains):
            raise RuntimeError(
                "Prisms do not occupy exactly the swept shell domains: "
                f"prism domains={sorted(prism_domains)}, shell domains={shell_domains}"
            )
        print(
            "Swept shell verified: "
            f"{source_triangle_count} source triangles x "
            f"{shell_radial_layers} radial layers = {prism_count} prisms",
            flush=True,
        )

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
    tau_thickness_lambda0: float = 0.01,
    pml_thickness_lambda0: float = 0.25,
    mesh_preset: int = 4,
    shell_mesh: str = "tetra",
    global_hmax_divisor: float = 6.0,
    interior_hmax_divisor: float | None = None,
    source_ball_radius_lambda0: float | None = None,
    source_ball_hmax_divisor: float | None = None,
    inner_shell_band_width_lambda0: float | None = None,
    inner_shell_band_hmax_divisor: float | None = None,
    outer_shell_band_width_lambda0: float | None = None,
    outer_shell_band_hmax_divisor: float | None = None,
    cap_edge_hmax_divisor: float | None = None,
    shell_tangential_divisor: float = 20.0,
    shell_radial_layers: int = 2,
    discretization: str = "linear",
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
    cap_parameter_scale: float | None = None,
    mesh_only: bool = False,
    physics_tag: str = "emw",
) -> dict[str, Any]:
    """Solve the physical omega-biisotropic sphere model and validate against prescribed theory."""
    configuration = load_workflow_configuration(configuration_path)
    if cap_parameter_scale is not None:
        if configuration.cap_perturbation is None:
            raise ValueError(
                "--cap-scale requires a configuration with cap_perturbation"
            )
        configuration = replace(
            configuration,
            cap_perturbation=replace(
                configuration.cap_perturbation,
                parameter_scale=cap_parameter_scale,
            ),
        )
    case = configuration.case
    cap_perturbation = configuration.cap_perturbation
    cap_is_active = (
        cap_perturbation is not None
        and cap_perturbation.parameter_scale != 1.0
    )
    plot_reference_label = (
        "Unperturbed reference" if cap_is_active else "Analytical Prescribed"
    )
    surface_profile_metadata = (
        compile_scalar_omega_cap_profile(
            cap_perturbation,
            nx="nx",
            ny="ny",
            nz="nz",
        ).metadata()
        if cap_perturbation is not None
        else {"type": "uniform"}
    )
    cap_metadata = configuration.to_dict().get("cap_perturbation")
    frequency_hz = case.frequency
    wavelength = c / frequency_hz
    radius_m = case.radius
    shell_outer_radius_m = radius_m + tau_thickness_lambda0 * wavelength
    inner_shell_band_inner_radius_m = (
        radius_m - inner_shell_band_width_lambda0 * wavelength
        if inner_shell_band_width_lambda0 is not None
        else None
    )
    source_ball_radius_m = (
        source_ball_radius_lambda0 * wavelength
        if source_ball_radius_lambda0 is not None
        else None
    )
    outer_shell_band_outer_radius_m = (
        shell_outer_radius_m + outer_shell_band_width_lambda0 * wavelength
        if outer_shell_band_width_lambda0 is not None
        else None
    )
    air_extent_m = air_extent_lambda0 * wavelength
    radius_lambda0 = radius_m / wavelength
    epsilon_r_interior = case.epsilon_interior / epsilon_0
    global_hmin_divisor = _resolved_global_hmin_divisor(
        global_hmax_divisor=global_hmax_divisor,
        interior_hmax_divisor=interior_hmax_divisor,
        source_ball_hmax_divisor=source_ball_hmax_divisor,
        inner_shell_band_hmax_divisor=inner_shell_band_hmax_divisor,
        outer_shell_band_hmax_divisor=outer_shell_band_hmax_divisor,
        shell_mesh=shell_mesh,
        shell_tangential_divisor=shell_tangential_divisor,
        cap_edge_hmax_divisor=cap_edge_hmax_divisor,
    )

    theta_rad = np.deg2rad(sample_theta_degrees)
    phi_rad = np.deg2rad(sample_phi_degrees)
    radial_direction = np.array(
        [
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad),
        ]
    )
    maximum_radial_coordinate = (
        sample_max_radii * radius_m * float(np.max(np.abs(radial_direction)))
    )
    zx_half_span_m = zx_span_radii * radius_m
    tolerance = 1e-12 * wavelength
    if not mesh_only and maximum_radial_coordinate > air_extent_m + tolerance:
        raise ValueError(
            "Radial extraction enters the PML. Reduce --radial-max-radii so "
            "every coordinate remains within the non-PML air box: "
            f"requested maximum coordinate={maximum_radial_coordinate / wavelength:.6g} "
            f"lambda0, air extent={air_extent_lambda0:.6g} lambda0"
        )
    if not mesh_only and zx_half_span_m > air_extent_m + tolerance:
        raise ValueError(
            "ZX extraction enters the PML. Reduce --zx-span-radii so its "
            "half-span remains within the non-PML air box: "
            f"requested half-span={zx_half_span_m / wavelength:.6g} lambda0, "
            f"air extent={air_extent_lambda0:.6g} lambda0"
        )

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
        cap_perturbation=cap_perturbation,
        air_extent_lambda0=air_extent_lambda0,
        tau_thickness_lambda0=tau_thickness_lambda0,
        pml_thickness_lambda0=pml_thickness_lambda0,
        mesh_preset=mesh_preset,
        shell_mesh=shell_mesh,
        global_hmax_divisor=global_hmax_divisor,
        interior_hmax_divisor=interior_hmax_divisor,
        source_ball_radius_lambda0=source_ball_radius_lambda0,
        source_ball_hmax_divisor=source_ball_hmax_divisor,
        inner_shell_band_width_lambda0=inner_shell_band_width_lambda0,
        inner_shell_band_hmax_divisor=inner_shell_band_hmax_divisor,
        outer_shell_band_width_lambda0=outer_shell_band_width_lambda0,
        outer_shell_band_hmax_divisor=outer_shell_band_hmax_divisor,
        cap_edge_hmax_divisor=cap_edge_hmax_divisor,
        shell_tangential_divisor=shell_tangential_divisor,
        shell_radial_layers=shell_radial_layers,
        linear_solver=linear_solver,
        physics_tag=physics_tag,
        discretization=discretization,
    )

    try:
        if mesh_only:
            print("Mesh-only requested. Saving model and skipping solve.", flush=True)
            model_path = run_dir / "model.mph"
            model.save(model_path)
            mesh = model.java.component("comp1").mesh("mesh1")
            return {
                "mesh_elements": int(mesh.getNumElem()),
                "mesh_tetrahedra": _mesh_element_count(mesh, "tet"),
                "mesh_prisms": _mesh_element_count(mesh, "prism"),
                "mesh_vertices": int(mesh.getNumVertex()),
                "status": "mesh_only",
                "physics_interface": physics_tag,
                "shell_mesh": shell_mesh,
                "shell_diagonal_embedding": "tangential_projector",
                "gstc_surface_profile": surface_profile_metadata,
                "reference_label": configuration.reference_label,
                **(
                    {"cap_perturbation": cap_metadata}
                    if cap_metadata is not None
                    else {}
                ),
                "global_hmax_divisor": global_hmax_divisor,
                "interior_hmax_divisor": interior_hmax_divisor,
                "source_ball_radius_lambda0": source_ball_radius_lambda0,
                "source_ball_radius_m": source_ball_radius_m,
                "source_ball_hmax_divisor": source_ball_hmax_divisor,
                "inner_shell_band_width_lambda0": inner_shell_band_width_lambda0,
                "inner_shell_band_hmax_divisor": inner_shell_band_hmax_divisor,
                "inner_shell_band_inner_radius_m": inner_shell_band_inner_radius_m,
                "outer_shell_band_width_lambda0": outer_shell_band_width_lambda0,
                "outer_shell_band_hmax_divisor": outer_shell_band_hmax_divisor,
                "outer_shell_band_outer_radius_m": outer_shell_band_outer_radius_m,
                "cap_edge_geometry_aligned": cap_edge_hmax_divisor is not None,
                "cap_edge_hmax_divisor": cap_edge_hmax_divisor,
                "global_hmin_divisor": global_hmin_divisor,
                "shell_tangential_divisor": shell_tangential_divisor,
                "shell_radial_layers": shell_radial_layers,
                "tau_thickness_lambda0": tau_thickness_lambda0,
                "files": {"model": "model.mph"},
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

        region_rad = np.where(
            radius < radius_m,
            1,
            np.where(radius <= shell_outer_radius_m, 3, 2),
        )
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
            shell_outer_radius_m,
            sample_theta_degrees,
            sample_phi_degrees,
            comp_e_rad,
            comp_h_rad,
            ref_e_rad,
            ref_h_rad,
            reference_label=plot_reference_label,
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

        region_2d = np.where(
            radius_2d < radius_m,
            1,
            np.where(radius_2d <= shell_outer_radius_m, 3, 2),
        )
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
            shell_outer_radius_m,
            comp_e_2d,
            comp_h_2d,
            ref_e_2d,
            ref_h_2d,
            reference_label=plot_reference_label,
        )

        # 3. Compute relative L2 errors.  Preserve the historical metrics that
        # exclude only the shell itself, and add less interface-sensitive
        # metrics that also exclude one shell thickness on either side.
        rad_int = (radius >= 0.2 * radius_m) & (radius < radius_m)
        rad_ext = radius > shell_outer_radius_m
        rad_mask = rad_int | rad_ext

        int_2d = (radius_2d >= 0.2 * radius_m) & (radius_2d < radius_m)
        ext_2d = radius_2d > shell_outer_radius_m
        mask_2d = int_2d | ext_2d

        shell_buffer_m = shell_outer_radius_m - radius_m
        rad_int_buffered = (radius >= 0.2 * radius_m) & (
            radius < radius_m - shell_buffer_m
        )
        rad_ext_buffered = radius > shell_outer_radius_m + shell_buffer_m
        rad_mask_buffered = rad_int_buffered | rad_ext_buffered

        int_2d_buffered = (radius_2d >= 0.2 * radius_m) & (
            radius_2d < radius_m - shell_buffer_m
        )
        ext_2d_buffered = radius_2d > shell_outer_radius_m + shell_buffer_m
        mask_2d_buffered = int_2d_buffered | ext_2d_buffered

        comparison_values = {
            "electric_radial_masked": _relative_error(
                comp_e_rad[:, rad_mask], ref_e_rad[:, rad_mask]
            ),
            "magnetic_radial_masked": _relative_error(
                comp_h_rad[:, rad_mask], ref_h_rad[:, rad_mask]
            ),
            "electric_radial_interior": _relative_error(
                comp_e_rad[:, rad_int], ref_e_rad[:, rad_int]
            ),
            "electric_radial_exterior": _relative_error(
                comp_e_rad[:, rad_ext], ref_e_rad[:, rad_ext]
            ),
            "electric_zx_masked": _relative_error(
                comp_e_2d[:, mask_2d], ref_e_2d[:, mask_2d]
            ),
            "magnetic_zx_masked": _relative_error(
                comp_h_2d[:, mask_2d], ref_h_2d[:, mask_2d]
            ),
            "electric_zx_interior": _relative_error(
                comp_e_2d[:, int_2d], ref_e_2d[:, int_2d]
            ),
            "electric_zx_exterior": _relative_error(
                comp_e_2d[:, ext_2d], ref_e_2d[:, ext_2d]
            ),
            "electric_radial_buffered": _relative_error(
                comp_e_rad[:, rad_mask_buffered], ref_e_rad[:, rad_mask_buffered]
            ),
            "magnetic_radial_buffered": _relative_error(
                comp_h_rad[:, rad_mask_buffered], ref_h_rad[:, rad_mask_buffered]
            ),
            "electric_radial_interior_buffered": _relative_error(
                comp_e_rad[:, rad_int_buffered], ref_e_rad[:, rad_int_buffered]
            ),
            "electric_radial_exterior_buffered": _relative_error(
                comp_e_rad[:, rad_ext_buffered], ref_e_rad[:, rad_ext_buffered]
            ),
            "electric_zx_buffered": _relative_error(
                comp_e_2d[:, mask_2d_buffered], ref_e_2d[:, mask_2d_buffered]
            ),
            "magnetic_zx_buffered": _relative_error(
                comp_h_2d[:, mask_2d_buffered], ref_h_2d[:, mask_2d_buffered]
            ),
            "electric_zx_interior_buffered": _relative_error(
                comp_e_2d[:, int_2d_buffered], ref_e_2d[:, int_2d_buffered]
            ),
            "electric_zx_exterior_buffered": _relative_error(
                comp_e_2d[:, ext_2d_buffered], ref_e_2d[:, ext_2d_buffered]
            ),
        }
        comparison_kind = "change" if cap_is_active else "error"
        comparison_metrics = {
            f"relative_l2_{comparison_kind}_{name}": value
            for name, value in comparison_values.items()
        }

        mesh = model.java.component("comp1").mesh("mesh1")

        resolved_configuration = configuration.to_dict()
        resolved_configuration["comsol"] = {
            "stage": "omega_sphere",
            "frequency_hz": frequency_hz,
            "radius_lambda0": radius_lambda0,
            "shell_outer_radius_lambda0": shell_outer_radius_m / wavelength,
            "epsilon_r_interior": epsilon_r_interior,
            "air_extent_lambda0": air_extent_lambda0,
            "pml_thickness_lambda0": pml_thickness_lambda0,
            "mesh_preset": mesh_preset,
            "shell_mesh": shell_mesh,
            "shell_diagonal_embedding": "tangential_projector",
            "gstc_surface_profile": surface_profile_metadata,
            "reference_label": configuration.reference_label,
            **(
                {"cap_perturbation": cap_metadata}
                if cap_metadata is not None
                else {}
            ),
            "global_hmax_divisor": global_hmax_divisor,
            "interior_hmax_divisor": interior_hmax_divisor,
            "source_ball_radius_lambda0": source_ball_radius_lambda0,
            "source_ball_radius_m": source_ball_radius_m,
            "source_ball_hmax_divisor": source_ball_hmax_divisor,
            "inner_shell_band_width_lambda0": inner_shell_band_width_lambda0,
            "inner_shell_band_hmax_divisor": inner_shell_band_hmax_divisor,
            "inner_shell_band_inner_radius_m": inner_shell_band_inner_radius_m,
            "outer_shell_band_width_lambda0": outer_shell_band_width_lambda0,
            "outer_shell_band_hmax_divisor": outer_shell_band_hmax_divisor,
            "outer_shell_band_outer_radius_m": outer_shell_band_outer_radius_m,
            "cap_edge_geometry_aligned": cap_edge_hmax_divisor is not None,
            "cap_edge_hmax_divisor": cap_edge_hmax_divisor,
            "global_hmin_divisor": global_hmin_divisor,
            "shell_tangential_divisor": shell_tangential_divisor,
            "shell_radial_layers": shell_radial_layers,
            "tau_thickness_lambda0": tau_thickness_lambda0,
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
            "comparison_excludes_volumetric_shell": True,
            "comparison_shell_buffer_thicknesses": 1.0,
            "comparison_shell_buffer_m": shell_buffer_m,
            "reference_is_exact_solution": not cap_is_active,
            "comparison_metric_meaning": (
                "change_from_unperturbed_reference"
                if cap_is_active
                else "error_against_prescribed_solution"
            ),
        }
        with (run_dir / "configuration.json").open("w", encoding="utf-8") as stream:
            json.dump(resolved_configuration, stream, indent=2)
            stream.write("\n")

        return {
            "frequency_hz": frequency_hz,
            "wavelength_m": wavelength,
            "radius_m": radius_m,
            "radius_lambda0": radius_lambda0,
            "shell_outer_radius_m": shell_outer_radius_m,
            "epsilon_r_interior": epsilon_r_interior,
            "mesh_preset": mesh_preset,
            "shell_mesh": shell_mesh,
            "shell_diagonal_embedding": "tangential_projector",
            "gstc_surface_profile": surface_profile_metadata,
            "reference_label": configuration.reference_label,
            **(
                {"cap_perturbation": cap_metadata}
                if cap_metadata is not None
                else {}
            ),
            "global_hmax_divisor": global_hmax_divisor,
            "interior_hmax_divisor": interior_hmax_divisor,
            "source_ball_radius_lambda0": source_ball_radius_lambda0,
            "source_ball_radius_m": source_ball_radius_m,
            "source_ball_hmax_divisor": source_ball_hmax_divisor,
            "inner_shell_band_width_lambda0": inner_shell_band_width_lambda0,
            "inner_shell_band_hmax_divisor": inner_shell_band_hmax_divisor,
            "inner_shell_band_inner_radius_m": inner_shell_band_inner_radius_m,
            "outer_shell_band_width_lambda0": outer_shell_band_width_lambda0,
            "outer_shell_band_hmax_divisor": outer_shell_band_hmax_divisor,
            "outer_shell_band_outer_radius_m": outer_shell_band_outer_radius_m,
            "cap_edge_geometry_aligned": cap_edge_hmax_divisor is not None,
            "cap_edge_hmax_divisor": cap_edge_hmax_divisor,
            "global_hmin_divisor": global_hmin_divisor,
            "shell_tangential_divisor": shell_tangential_divisor,
            "shell_radial_layers": shell_radial_layers,
            "tau_thickness_lambda0": tau_thickness_lambda0,
            "mesh_elements": int(mesh.getNumElem()),
            "mesh_tetrahedra": _mesh_element_count(mesh, "tet"),
            "mesh_prisms": _mesh_element_count(mesh, "prism"),
            "mesh_vertices": int(mesh.getNumVertex()),
            "physics_interface": physics_tag,
            "chi_em": chi_em,
            "y_ee_imag": y_ee_imag,
            "z_mm_imag": z_mm_imag,
            "sample_theta_degrees": sample_theta_degrees,
            "sample_phi_degrees": sample_phi_degrees,
            "comparison_excludes_volumetric_shell": True,
            "comparison_shell_buffer_thicknesses": 1.0,
            "comparison_shell_buffer_m": shell_buffer_m,
            "reference_is_exact_solution": not cap_is_active,
            "comparison_metric_meaning": (
                "change_from_unperturbed_reference"
                if cap_is_active
                else "error_against_prescribed_solution"
            ),
            **comparison_metrics,
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
    parser.add_argument(
        "--discretization",
        choices=("linear", "quadratic"),
        default="linear",
        help="Element discretization order for electric field (default: linear)",
    )
    parser.add_argument("--cores", type=_positive_int, default=_default_cores())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIGURATION)
    parser.add_argument(
        "--case-id",
        default=None,
        help="Result case ID (default: case_id from --config)",
    )
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
        "--shell-mesh",
        choices=("tetra", "swept"),
        default="tetra",
        help="Volumetric-shell mesh strategy (default: tetra)",
    )
    parser.add_argument(
        "--global-hmax-divisor",
        type=_positive_float,
        default=6.0,
        help=(
            "Divisor N for the background/exterior maximum size lambda0/N; "
            "also used in the interior unless --interior-hmax-divisor is set "
            "(default: 6)"
        ),
    )
    parser.add_argument(
        "--interior-hmax-divisor",
        type=_positive_float,
        default=None,
        help=(
            "Optional divisor N for a dielectric-interior maximum size "
            "lambda0/N (default: inherit --global-hmax-divisor)"
        ),
    )
    parser.add_argument(
        "--source-ball-radius",
        type=_positive_float,
        default=None,
        help=(
            "Radius of an optional conforming refinement ball around the "
            "point dipole, in lambda0"
        ),
    )
    parser.add_argument(
        "--source-ball-hmax-divisor",
        type=_positive_float,
        default=None,
        help=(
            "Divisor N for maximum tetrahedron size lambda0/N in the "
            "optional point-source refinement ball"
        ),
    )
    parser.add_argument(
        "--inner-shell-band-width",
        type=_positive_float,
        default=None,
        help=(
            "Width of an optional conforming dielectric refinement band "
            "immediately inside the GSTC shell, in lambda0"
        ),
    )
    parser.add_argument(
        "--inner-shell-band-hmax-divisor",
        type=_positive_float,
        default=None,
        help=(
            "Divisor N for maximum tetrahedron size lambda0/N in the "
            "optional inner-shell dielectric band"
        ),
    )
    parser.add_argument(
        "--outer-shell-band-width",
        type=_positive_float,
        default=None,
        help=(
            "Width of an optional conforming air refinement band immediately "
            "outside the GSTC shell, in lambda0"
        ),
    )
    parser.add_argument(
        "--outer-shell-band-hmax-divisor",
        type=_positive_float,
        default=None,
        help=(
            "Divisor N for maximum tetrahedron size lambda0/N in the optional "
            "outer-shell air band"
        ),
    )
    parser.add_argument(
        "--cap-edge-hmax-divisor",
        type=_positive_float,
        default=None,
        help=(
            "Optional divisor N for lambda0/N refinement on an exact "
            "geometry-aligned spherical-cap rim; requires cap_perturbation "
            "and --shell-mesh swept"
        ),
    )
    parser.add_argument(
        "--shell-tangential-divisor",
        type=_positive_float,
        default=20.0,
        help="Divisor N for swept-shell tangential size lambda0/N (default: 20)",
    )
    parser.add_argument(
        "--shell-radial-layers",
        type=_positive_int,
        default=2,
        help="Number of uniform prism layers through a swept shell (default: 2)",
    )
    parser.add_argument(
        "--boundary-hmax-divisor",
        type=_positive_float,
        default=None,
        help=(
            "Deprecated alias for --shell-tangential-divisor; retained for "
            "older commands"
        ),
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
        "--tau-thickness",
        type=float,
        default=0.01,
        help="Thickness tau of the volumetric metasurface shell in free-space wavelengths (default 0.01)",
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
        "--cap-scale",
        type=_cap_scale,
        default=None,
        help=(
            "Override cap_perturbation.parameter_scale from --config; useful "
            "for a scale=1 spatial-profile regression"
        ),
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
    arguments = parser.parse_args(argv)
    if (arguments.source_ball_radius is None) != (
        arguments.source_ball_hmax_divisor is None
    ):
        parser.error(
            "--source-ball-radius and --source-ball-hmax-divisor must be "
            "provided together"
        )
    if (arguments.inner_shell_band_width is None) != (
        arguments.inner_shell_band_hmax_divisor is None
    ):
        parser.error(
            "--inner-shell-band-width and --inner-shell-band-hmax-divisor "
            "must be provided together"
        )
    if (arguments.outer_shell_band_width is None) != (
        arguments.outer_shell_band_hmax_divisor is None
    ):
        parser.error(
            "--outer-shell-band-width and --outer-shell-band-hmax-divisor "
            "must be provided together"
        )
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    selected_configuration = load_workflow_configuration(arguments.config)
    case_id = arguments.case_id or selected_configuration.case_id
    if arguments.boundary_hmax_divisor is not None:
        print(
            "WARNING: --boundary-hmax-divisor is deprecated; using its value "
            "as --shell-tangential-divisor.",
            flush=True,
        )
        arguments.shell_tangential_divisor = arguments.boundary_hmax_divisor
    _patch_mph_for_legacy_comsol(arguments.version)
    _require_compute_context(arguments.allow_outside_slurm)
    run_name = arguments.run_name or arguments.stage
    run_dir = arguments.run_dir or default_run_directory(
        case_id, "comsol", run_name
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "backend": "comsol",
        "stage": arguments.stage,
        "status": "running",
        "case_id": case_id,
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
                            tau_thickness_lambda0=arguments.tau_thickness,
                            pml_thickness_lambda0=arguments.pml_thickness,
                            mesh_preset=arguments.mesh_preset,
                            shell_mesh=arguments.shell_mesh,
                            global_hmax_divisor=arguments.global_hmax_divisor,
                            interior_hmax_divisor=arguments.interior_hmax_divisor,
                            source_ball_radius_lambda0=(
                                arguments.source_ball_radius
                            ),
                            source_ball_hmax_divisor=(
                                arguments.source_ball_hmax_divisor
                            ),
                            inner_shell_band_width_lambda0=(
                                arguments.inner_shell_band_width
                            ),
                            inner_shell_band_hmax_divisor=(
                                arguments.inner_shell_band_hmax_divisor
                            ),
                            outer_shell_band_width_lambda0=(
                                arguments.outer_shell_band_width
                            ),
                            outer_shell_band_hmax_divisor=(
                                arguments.outer_shell_band_hmax_divisor
                            ),
                            cap_edge_hmax_divisor=(
                                arguments.cap_edge_hmax_divisor
                            ),
                            shell_tangential_divisor=arguments.shell_tangential_divisor,
                            shell_radial_layers=arguments.shell_radial_layers,
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
                            cap_parameter_scale=arguments.cap_scale,
                            linear_solver=arguments.linear_solver,
                            discretization=arguments.discretization,
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
