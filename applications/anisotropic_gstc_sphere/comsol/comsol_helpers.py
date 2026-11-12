#!/usr/bin/env python3
"""COMSOL 5.4 / 6.1 session, geometry, meshing, and physics helpers for Case B validation.

Separates low-level MPh/JPype boilerplate, concentric geometry construction, meshing,
and Equation View physics overrides from the main workflow script so each phase remains
concise, modular, and easy to follow.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mph
import numpy as np
from jpype import JArray, JDouble, JInt, JString

if __package__:
    from .interpolation import ETA_0, PARAMETER_NAMES
else:
    from interpolation import ETA_0, PARAMETER_NAMES

# Reuse the compatibility JARs directory from omega_biisotropic_sphere/comsol/jars
REPO_ROOT = Path(__file__).resolve().parents[3]
COMPAT_JARS_DIR = (
    REPO_ROOT / "applications" / "omega_biisotropic_sphere" / "comsol" / "jars"
)


# =============================================================================
# 1. Java & COMSOL 5.4 / 6.1 Session Helpers
# =============================================================================


def java_double_array(values: Sequence[float]) -> Any:
    """Convert numeric values to an unambiguous Java ``double[]``."""
    return JArray(JDouble)(values)


def java_string_tensor(values: Sequence[Sequence[str]]) -> Any:
    """Flatten 3x3 tensor rows into COMSOL's row-major Java ``String[]`` form."""
    return JArray(JString)([value for row in values for value in row])


def find_modern_jvm() -> str | None:
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


def patch_mph_for_legacy_comsol(version: str) -> None:
    """Allow modern JPype (requiring Java >= 9) to run with legacy COMSOL (5.x).

    COMSOL 5.4 bundled Java 8. Modern JPype (>=1.5) strictly requires Java >= 9.
    We route MPh to the system's modern JVM (Java 11/17/21) and pass reflection
    options + JAXB compatibility JARs so COMSOL's Java API client runs cleanly.
    """
    if not str(version).startswith("5."):
        return
    try:
        import jpype

        modern_jvm = find_modern_jvm()
        if not modern_jvm:
            print("[COMSOL 5.4 patch] Warning: No modern JVM found on system.", flush=True)
            return

        try:
            backends = mph.discovery.find_backends()
            for backend in backends:
                if str(backend.get("major", "")) <= "5":
                    backend["jvm"] = Path(modern_jvm)
        except Exception:
            pass

        orig_backend = mph.discovery.backend

        def _patched_backend(v: str | None = None):
            backend = orig_backend(v)
            if str(backend.get("major", "")) <= "5":
                backend["jvm"] = Path(modern_jvm)
            return backend

        mph.discovery.backend = _patched_backend

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

            compat_jars = (
                sorted(COMPAT_JARS_DIR.glob("*.jar"))
                if COMPAT_JARS_DIR.is_dir()
                else []
            )
            orig_classpath = kwargs.get("classpath")
            if orig_classpath is not None:
                cp_list = (
                    list(orig_classpath)
                    if isinstance(orig_classpath, (list, tuple))
                    else [orig_classpath]
                )
                for jar_file in compat_jars:
                    if str(jar_file) not in cp_list:
                        cp_list.append(str(jar_file))
                kwargs["classpath"] = cp_list
            elif compat_jars:
                kwargs["classpath"] = [str(j) for j in compat_jars]

            if compat_jars:
                print(
                    "[COMSOL 5.4 patch] Injected Java EE compatibility jars: "
                    f"{[j.name for j in compat_jars]}",
                    flush=True,
                )
            return orig_startJVM(*new_args, **kwargs)

        jpype.startJVM = _patched_startJVM
        print(f"[COMSOL 5.4 patch] Redirected JVM to modern runtime: {modern_jvm}", flush=True)
    except Exception as exc:
        print(f"[COMSOL 5.4 patch] Warning: failed to apply JVM patch: {exc}", flush=True)


def start_comsol(
    *, version: str, cores: int, runtime_dir: Path
) -> tuple[mph.Server, mph.Client]:
    """Start an isolated COMSOL server and connect an MPh client."""
    patch_mph_for_legacy_comsol(version)

    server_kwargs: dict[str, Any] = {
        "cores": cores,
        "version": version,
    }

    # Isolated Eclipse runtime arguments are only needed for legacy COMSOL 5.x on Linux
    if platform.system() != "Windows" and str(version).startswith("5."):
        configuration_dir = runtime_dir / "configuration"
        temporary_dir = runtime_dir / "tmp"
        configuration_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir.mkdir(parents=True, exist_ok=True)
        server_kwargs["arguments"] = [
            "-configuration",
            str(configuration_dir),
            "-tmpdir",
            str(temporary_dir),
        ]
        server_kwargs["port"] = 0

    server = mph.Server(**server_kwargs)
    try:
        client = mph.Client(cores=cores, version=version, port=server.port)
        try:
            client.java.setDefaultGeometryKernel("comsol")
        except Exception:
            pass
    except Exception:
        server.stop()
        raise
    return server, client


def stop_comsol(server: mph.Server, client: mph.Client) -> None:
    """Cleanly disconnect the MPh client and stop the COMSOL server."""
    if client.port:
        client.disconnect()
    if server.running():
        server.stop()


# =============================================================================
# 2. Phase 3: Concentric Spherical Shell Geometry & Selections
# =============================================================================
#
# Radial Cross-Section of Concentric Regions & Meshing Strategy:
#
#   r = 0          r_src        a - w_in         a          a + tau    a + tau + w_out     b        b + t_pml
#   +----------------+--------------+------------+-------------+--------------+------------+------------+
#   |  Dipole Source |  Dielectric  | Inner Near |    GSTC     |  Outer Near  |  Exterior  |  Cartesian |
#   |      Ball      |     Bulk     | Shell Band |   Shell     |  Shell Band  |  Air Box   |  PML Layer |
#   |   (optional)   |  (eps_r,int) | (optional) | (thickness  |  (optional)  | (eps_r=1)  | (26 doms)  |
#   |                |              |            |    tau)     |              |            |            |
#   +----------------+--------------+------------+-------------+--------------+------------+------------+
#   |<-------------- sphere_domains ------------>|<-- shell -->|<------- exterior_air_domains --------->|
#
#   Meshing per region:
#   - Source Ball      : FreeTet (hmax = lambda0 / source_ball_hmax_divisor)
#   - Dielectric Bulk  : FreeTet (hmax = lambda0 / interior_hmax_divisor)
#   - Inner Shell Band : FreeTet (hmax = lambda0 / inner_shell_band_hmax_divisor)
#   - GSTC Shell       :
#       * Swept mode   : FreeTri on outer boundary (r = a + tau, hmax = lambda0 / shell_tangential_divisor)
#                        + radial Sweep inward to r = a across 8 octants with N_layers prisms
#       * Tetra mode   : FreeTet (hmax = tau, hmin = tau / 5)
#   - Outer Shell Band : FreeTet (hmax = lambda0 / outer_shell_band_hmax_divisor)
#   - Air Box & PML    : FreeTet (hmax = lambda0 / global_hmax_divisor)


@dataclass(frozen=True)
class GeometrySelections:
    """Named COMSOL selections and entity IDs produced by Phase 3 geometry setup."""

    sphere_domain_selection: str
    sphere_shell_selection: str
    shell_outer_faces_selection: str | None
    shell_inner_faces_selection: str | None
    inner_shell_band_selection: str | None
    outer_shell_band_selection: str | None
    source_ball_selection: str | None
    sphere_domains: list[int]
    shell_domains: list[int]
    exterior_air_domains: list[int]
    outer_shell_faces: list[int]
    inner_shell_faces: list[int]
    inner_shell_band_domains: list[int]
    outer_shell_band_domains: list[int]
    source_ball_domains: list[int]
    pml_domains: list[int]
    source_points: list[int]


def setup_spherical_shell_geometry(
    component: Any,
    *,
    wavelength_m: float,
    radius_m: float,
    tau_m: float,
    air_extent_m: float,
    inner_band_width_m: float | None = None,
    outer_band_width_m: float | None = None,
    source_ball_radius_m: float | None = None,
    shell_mesh: str = "swept",
    source_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GeometrySelections:
    """Build concentric spheres, optional refinement bands, octant shell partitions, and PML box.

    Units & Dimensions Convention:
      All spatial dimensions passed to this function are in physical SI **meters [m]**:
        - ``wavelength_m``: Vacuum wavelength lambda0 [m]
        - ``radius_m``: Metasurface core radius a [m]
        - ``tau_m``: Physical shell thickness tau [m]
        - ``air_extent_m``: Radius/half-width of air domain b [m]
        - ``inner_band_width_m``: Radial width of inner refinement band w_in [m]
        - ``outer_band_width_m``: Radial width of outer refinement band w_out [m]
        - ``source_ball_radius_m``: Radius of dipole source refinement ball r_src [m]
        - ``source_position_m``: Cartesian coordinates (x, y, z) of source [m]

      (Note: Mesh sizing divisors such as ``global_hmax_divisor`` or ``inner_shell_band_hmax_divisor``
      are dimensionless factors defining element size relative to wavelength: hmax = lambda / divisor).

    Uses standard primitives and Ball/Difference selections so the geometry
    construction is 100% compatible with both COMSOL 5.4 and COMSOL 6.1.
    Expects model parameters ``a``, ``tau``, ``b``, ``t_pml`` (and optionally ``w_in``,
    ``w_out``, ``r_src``) to be defined on ``model.java.param()``.
    """
    geometry = component.geom().create("geom1", 3)
    geometry.lengthUnit("m")

    # 1. Metasurface shell (a < r < a + tau)
    geometry.create("sph2", "Sphere")
    sph2 = geometry.feature("sph2")
    sph2.label("Bianisotropic metasurface shell outer sphere")
    sph2.set("r", "a + tau")
    sph2.set("pos", java_double_array((0.0, 0.0, 0.0)))
    sph2.set("selresult", True)
    sph2.set("selresultshow", "all")

    if shell_mesh == "swept":
        # Subtract a sacrificial core sphere so only the hollow shell is partitioned
        # into 8 sweepable octants by the 3 coordinate planes.
        geometry.create("sph_core_cut", "Sphere")
        sph_cut = geometry.feature("sph_core_cut")
        sph_cut.label("Shell inner boundary cut sphere")
        sph_cut.set("r", "a")
        sph_cut.set("pos", java_double_array((0.0, 0.0, 0.0)))
        sph_cut.set("selresult", True)
        sph_cut.set("selresultshow", "all")

        geometry.create("dif_shell", "Difference")
        dif_shell = geometry.feature("dif_shell")
        dif_shell.label("Volumetric shell (outer sphere minus core)")
        dif_shell.selection("input").set("sph2")
        dif_shell.selection("input2").set("sph_core_cut")
        dif_shell.set("selresult", True)
        dif_shell.set("selresultshow", "all")

        partition_input = "dif_shell"
        for plane_tag, plane_name, partition_tag in (
            ("wp_shell_x", "yz", "par_shell_x"),
            ("wp_shell_y", "zx", "par_shell_y"),
            ("wp_shell_z", "xy", "par_shell_z"),
        ):
            geometry.create(plane_tag, "WorkPlane")
            wp = geometry.feature(plane_tag)
            wp.label(f"Shell partition plane {plane_name}")
            wp.set("planetype", "quick")
            wp.set("quickplane", plane_name)

            geometry.create(partition_tag, "Partition")
            part = geometry.feature(partition_tag)
            part.label(f"Partition shell with {plane_name} plane")
            part.selection("input").set(partition_input)
            part.set("partitionwith", "workplane")
            part.set("workplane", plane_tag)
            part.set("selresult", True)
            part.set("selresultshow", "all")
            partition_input = partition_tag

        shell_geometry_feature = partition_input
        shell_inner_faces_selection: str | None = "geom1_sph_core_cut_bnd"
    else:
        shell_geometry_feature = "sph2"
        shell_inner_faces_selection = "geom1_sph1_bnd"

    # 2. Interior dielectric sphere (r = a) and concentric refinement spheres
    geometry.create("sph1", "Sphere")
    sph1 = geometry.feature("sph1")
    sph1.label("Interior dielectric sphere")
    sph1.set("r", "a")
    sph1.set("pos", java_double_array((0.0, 0.0, 0.0)))
    sph1.set("selresult", True)
    sph1.set("selresultshow", "all")

    if inner_band_width_m is not None:
        geometry.create("sph_inner_bulk", "Sphere")
        inner_bulk = geometry.feature("sph_inner_bulk")
        inner_bulk.label("Inner bulk refinement radius")
        inner_bulk.set("r", "a - w_in")
        inner_bulk.set("pos", java_double_array((0.0, 0.0, 0.0)))
        inner_bulk.set("selresult", True)
        inner_bulk.set("selresultshow", "all")

    if source_ball_radius_m is not None:
        geometry.create("sph_source", "Sphere")
        sph_src = geometry.feature("sph_source")
        sph_src.label("Point-source refinement ball")
        sph_src.set("r", "r_src")
        sph_src.set("pos", java_double_array(source_position_m))
        sph_src.set("selresult", True)
        sph_src.set("selresultshow", "all")

    if outer_band_width_m is not None:
        geometry.create("sph_outer_band", "Sphere")
        sph_out = geometry.feature("sph_outer_band")
        sph_out.label("Outer near-shell refinement radius")
        sph_out.set("r", "a + tau + w_out")
        sph_out.set("pos", java_double_array((0.0, 0.0, 0.0)))
        sph_out.set("selresult", True)
        sph_out.set("selresultshow", "all")

    # 3. Outer air box with Cartesian PML layer
    geometry.create("blk1", "Block")
    blk1 = geometry.feature("blk1")
    blk1.label("Outer air box with PML layer")
    blk1.set("size", ["2*(b + t_pml)", "2*(b + t_pml)", "2*(b + t_pml)"])
    blk1.set("base", "center")
    blk1.set("pos", java_double_array((0.0, 0.0, 0.0)))
    blk1.set("layername", ["pml"])
    blk1.setIndex("layer", "t_pml", 0)
    for side in ("layerleft", "layerright", "layerbottom", "layertop", "layerfront", "layerback"):
        blk1.set(side, True)

    # 4. Dipole source point
    geometry.create("pt1", "Point")
    pt1 = geometry.feature("pt1")
    pt1.label("Dipole source point")
    pt1.set("p", java_double_array(source_position_m))

    geometry.run("fin")

    # --- Build Selections ---
    tol = 1e-5 * wavelength_m

    # 1. Dipole point selection
    component.selection().create("dipole_origin", "Ball")
    origin = component.selection("dipole_origin")
    origin.label("Dipole source point selection")
    origin.set("entitydim", JInt(0))
    origin.set("r", JDouble(1e-6 * wavelength_m))
    origin.set("posx", JDouble(source_position_m[0]))
    origin.set("posy", JDouble(source_position_m[1]))
    origin.set("posz", JDouble(source_position_m[2]))
    source_points = [int(p) for p in origin.entities(JInt(0))]
    if len(source_points) != 1:
        raise RuntimeError(f"Expected 1 dipole point at {source_position_m}, got {source_points}")

    # 2. Complete dielectric interior (r <= a) and concentric refinement selections
    component.selection().create("sphere_domains", "Ball")
    ball_core = component.selection("sphere_domains")
    ball_core.label("Complete dielectric interior")
    ball_core.set("entitydim", JInt(3))
    ball_core.set("condition", "inside")
    ball_core.set("r", JDouble(radius_m + tol))
    sphere_domain_selection = "sphere_domains"
    sphere_domains = [int(d) for d in ball_core.entities(JInt(3))]
    if not sphere_domains:
        raise RuntimeError("No interior dielectric domains found inside r <= a")

    inner_shell_band_selection: str | None = None
    inner_shell_band_domains: list[int] = []
    if inner_band_width_m is not None:
        component.selection().create("ball_inner_bulk", "Ball")
        b_in = component.selection("ball_inner_bulk")
        b_in.set("entitydim", JInt(3))
        b_in.set("condition", "inside")
        b_in.set("r", JDouble(radius_m - inner_band_width_m + tol))

        component.selection().create("inner_shell_band", "Difference")
        diff_in = component.selection("inner_shell_band")
        diff_in.label("Interior near-shell dielectric refinement band")
        diff_in.set("entitydim", JInt(3))
        diff_in.set("add", ["sphere_domains"])
        diff_in.set("subtract", ["ball_inner_bulk"])
        inner_shell_band_selection = "inner_shell_band"
        inner_shell_band_domains = [int(d) for d in diff_in.entities(JInt(3))]

    source_ball_selection: str | None = None
    source_ball_domains: list[int] = []
    if source_ball_radius_m is not None:
        component.selection().create("source_ball", "Ball")
        b_src = component.selection("source_ball")
        b_src.label("Point-source refinement ball")
        b_src.set("entitydim", JInt(3))
        b_src.set("condition", "inside")
        b_src.set("posx", JDouble(source_position_m[0]))
        b_src.set("posy", JDouble(source_position_m[1]))
        b_src.set("posz", JDouble(source_position_m[2]))
        b_src.set("r", JDouble(source_ball_radius_m + tol))
        source_ball_selection = "source_ball"
        source_ball_domains = [int(d) for d in b_src.entities(JInt(3))]

    # 3. Shell domains and outer/inner shell boundaries
    shell_outer_faces_selection = "geom1_sph2_bnd"
    outer_shell_faces = [
        int(b) for b in component.selection(shell_outer_faces_selection).entities(JInt(2))
    ]
    inner_shell_faces = [
        int(b) for b in component.selection(shell_inner_faces_selection).entities(JInt(2))
    ]

    if shell_mesh == "swept":
        sphere_shell_selection = f"geom1_{shell_geometry_feature}_dom"
        shell_domains = [
            int(d) for d in component.selection(sphere_shell_selection).entities(JInt(3))
        ]
        if len(shell_domains) != 8:
            raise RuntimeError(
                f"Expected 8 octant domains in swept shell, got {shell_domains}"
            )
    else:
        component.selection().create("sphere_shell_ball", "Ball")
        probe = component.selection("sphere_shell_ball")
        probe.label("Metasurface shell probe")
        probe.set("entitydim", JInt(3))
        probe.set("r", JDouble(0.25 * tau_m))
        probe.set("posx", JDouble(radius_m + 0.5 * tau_m))
        probe.set("posy", JDouble(0.0))
        probe.set("posz", JDouble(0.0))
        shell_domains = [int(d) for d in probe.entities(JInt(3))]
        if len(shell_domains) != 1:
            raise RuntimeError(f"Expected 1 tetrahedral shell domain, got {shell_domains}")
        component.selection().create("sphere_shell", "Explicit")
        shell_sel = component.selection("sphere_shell")
        shell_sel.label("Metasurface shell domain")
        shell_sel.set(JArray(JInt)(shell_domains))
        sphere_shell_selection = "sphere_shell"

    outer_shell_band_selection: str | None = None
    outer_shell_band_domains: list[int] = []
    if outer_band_width_m is not None:
        component.selection().create("ball_shell_outer", "Ball")
        b_sh = component.selection("ball_shell_outer")
        b_sh.set("entitydim", JInt(3))
        b_sh.set("condition", "inside")
        b_sh.set("r", JDouble(radius_m + tau_m + tol))

        component.selection().create("ball_outer_band", "Ball")
        b_out = component.selection("ball_outer_band")
        b_out.set("entitydim", JInt(3))
        b_out.set("condition", "inside")
        b_out.set("r", JDouble(radius_m + tau_m + outer_band_width_m + tol))

        component.selection().create("outer_shell_band", "Difference")
        diff_out = component.selection("outer_shell_band")
        diff_out.label("Exterior near-shell air refinement band")
        diff_out.set("entitydim", JInt(3))
        diff_out.set("add", ["ball_outer_band"])
        diff_out.set("subtract", ["ball_shell_outer"])
        outer_shell_band_selection = "outer_shell_band"
        outer_shell_band_domains = [int(d) for d in diff_out.entities(JInt(3))]

    # 4. PML domains (26 domains around the inner air box) & exterior air domains
    component.selection().create("inner_box", "Box")
    inner_box = component.selection("inner_box")
    inner_box.label("Inner box excluding PML")
    inner_box.set("entitydim", JInt(3))
    inner_box.set("condition", "inside")
    box_tol = air_extent_m + 1e-4 * wavelength_m
    for axis in ("x", "y", "z"):
        inner_box.set(f"{axis}min", JDouble(-box_tol))
        inner_box.set(f"{axis}max", JDouble(box_tol))
    inner_box_domains = [int(d) for d in inner_box.entities(JInt(3))]

    exterior_air_domains = sorted(
        set(inner_box_domains) - set(sphere_domains) - set(shell_domains)
    )
    if not exterior_air_domains:
        raise RuntimeError("No exterior air domains found between shell and PML")

    component.selection().create("pml_domains", "Complement")
    pml_sel = component.selection("pml_domains")
    pml_sel.label("PML layer domains")
    pml_sel.set("entitydim", JInt(3))
    pml_sel.set("input", ["inner_box"])
    pml_domains = [int(d) for d in pml_sel.entities(JInt(3))]
    if len(pml_domains) != 26:
        raise RuntimeError(f"Expected 26 PML domains, got {len(pml_domains)}: {pml_domains}")

    return GeometrySelections(
        sphere_domain_selection=sphere_domain_selection,
        sphere_shell_selection=sphere_shell_selection,
        shell_outer_faces_selection=shell_outer_faces_selection,
        shell_inner_faces_selection=shell_inner_faces_selection,
        inner_shell_band_selection=inner_shell_band_selection,
        outer_shell_band_selection=outer_shell_band_selection,
        source_ball_selection=source_ball_selection,
        sphere_domains=sphere_domains,
        shell_domains=shell_domains,
        exterior_air_domains=exterior_air_domains,
        outer_shell_faces=outer_shell_faces,
        inner_shell_faces=inner_shell_faces,
        inner_shell_band_domains=inner_shell_band_domains,
        outer_shell_band_domains=outer_shell_band_domains,
        source_ball_domains=source_ball_domains,
        pml_domains=pml_domains,
        source_points=source_points,
    )


@dataclass(frozen=True)
class SheetGeometrySelections:
    """Named COMSOL selections and entity IDs for Case A zero-thickness sheet geometry."""

    sphere_domain_selection: str
    sheet_boundary_selection: str
    far_field_surface_selection: str
    inner_sheet_band_selection: str | None
    outer_sheet_band_selection: str | None
    source_ball_selection: str | None
    sphere_domains: list[int]
    exterior_air_domains: list[int]
    far_field_air_domains: list[int]
    sheet_faces: list[int]
    far_field_faces: list[int]
    inner_sheet_band_domains: list[int]
    outer_sheet_band_domains: list[int]
    source_ball_domains: list[int]
    pml_domains: list[int]
    source_points: list[int]


def setup_spherical_sheet_geometry(
    component: Any,
    *,
    wavelength_m: float,
    radius_m: float,
    air_extent_m: float,
    far_field_offset_m: float = 0.0,
    inner_band_width_m: float | None = None,
    outer_band_width_m: float | None = None,
    source_ball_radius_m: float | None = None,
    source_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SheetGeometrySelections:
    """Build Case A zero-thickness metasurface sphere (r = a), optional Huygens sphere, and PML box.

    Expects model parameters ``a``, ``b``, ``t_pml`` (and optionally ``w_ff``, ``w_in``,
    ``w_out``, ``r_src``) to be defined on ``model.java.param()``.
    """
    geometry = component.geom().create("geom1", 3)
    geometry.lengthUnit("m")

    # 1. Zero-thickness metasurface sphere at r = a
    geometry.create("sph1", "Sphere")
    sph1 = geometry.feature("sph1")
    sph1.label("Bianisotropic zero-thickness metasurface sphere (r = a)")
    sph1.set("r", "a")
    sph1.set("pos", java_double_array((0.0, 0.0, 0.0)))
    sph1.set("selresult", True)
    sph1.set("selresultshow", "all")

    # 2. Optional concentric refinement spheres & detached Huygens far-field sphere
    if inner_band_width_m is not None and inner_band_width_m > 0.0:
        geometry.create("sph_inner_bulk", "Sphere")
        inner_bulk = geometry.feature("sph_inner_bulk")
        inner_bulk.label("Inner bulk refinement radius")
        inner_bulk.set("r", "a - w_in")
        inner_bulk.set("pos", java_double_array((0.0, 0.0, 0.0)))
        inner_bulk.set("selresult", True)
        inner_bulk.set("selresultshow", "all")

    if source_ball_radius_m is not None and source_ball_radius_m > 0.0:
        geometry.create("sph_source", "Sphere")
        sph_src = geometry.feature("sph_source")
        sph_src.label("Point-source refinement ball")
        sph_src.set("r", "r_src")
        sph_src.set("pos", java_double_array(source_position_m))
        sph_src.set("selresult", True)
        sph_src.set("selresultshow", "all")

    if outer_band_width_m is not None and outer_band_width_m > 0.0:
        geometry.create("sph_outer_band", "Sphere")
        sph_out = geometry.feature("sph_outer_band")
        sph_out.label("Outer near-sheet air refinement radius")
        sph_out.set("r", "a + w_out")
        sph_out.set("pos", java_double_array((0.0, 0.0, 0.0)))
        sph_out.set("selresult", True)
        sph_out.set("selresultshow", "all")

    ff_sphere_tag: str | None = None
    ff_radius_m: float = radius_m
    if far_field_offset_m > 0.0:
        if (
            outer_band_width_m is not None
            and outer_band_width_m > 0.0
            and abs(far_field_offset_m - outer_band_width_m) < 1e-6 * wavelength_m
        ):
            ff_sphere_tag = "sph_outer_band"
            ff_radius_m = radius_m + outer_band_width_m
        else:
            geometry.create("sph_ff", "Sphere")
            sph_ff = geometry.feature("sph_ff")
            sph_ff.label("Exterior Huygens Stratton-Chu far-field sphere")
            sph_ff.set("r", "a + w_ff")
            sph_ff.set("pos", java_double_array((0.0, 0.0, 0.0)))
            sph_ff.set("selresult", True)
            sph_ff.set("selresultshow", "all")
            ff_sphere_tag = "sph_ff"
            ff_radius_m = radius_m + far_field_offset_m

    # 3. Outer air box with Cartesian PML layer
    geometry.create("blk1", "Block")
    blk1 = geometry.feature("blk1")
    blk1.label("Outer air box with PML layer")
    blk1.set("size", ["2*(b + t_pml)", "2*(b + t_pml)", "2*(b + t_pml)"])
    blk1.set("base", "center")
    blk1.set("pos", java_double_array((0.0, 0.0, 0.0)))
    blk1.set("layername", ["pml"])
    blk1.setIndex("layer", "t_pml", 0)
    for side in ("layerleft", "layerright", "layerbottom", "layertop", "layerfront", "layerback"):
        blk1.set(side, True)

    # 4. Dipole source point
    geometry.create("pt1", "Point")
    pt1 = geometry.feature("pt1")
    pt1.label("Dipole source point")
    pt1.set("p", java_double_array(source_position_m))

    geometry.run("fin")

    # --- Build Selections ---
    tol = 1e-5 * wavelength_m

    # 1. Dipole point selection
    component.selection().create("dipole_origin", "Ball")
    origin = component.selection("dipole_origin")
    origin.label("Dipole source point selection")
    origin.set("entitydim", JInt(0))
    origin.set("r", JDouble(1e-6 * wavelength_m))
    origin.set("posx", JDouble(source_position_m[0]))
    origin.set("posy", JDouble(source_position_m[1]))
    origin.set("posz", JDouble(source_position_m[2]))
    source_points = [int(p) for p in origin.entities(JInt(0))]
    if len(source_points) != 1:
        raise RuntimeError(f"Expected 1 dipole point at {source_position_m}, got {source_points}")

    # 2. Complete dielectric interior (r <= a)
    component.selection().create("sphere_domains", "Ball")
    ball_core = component.selection("sphere_domains")
    ball_core.label("Complete dielectric interior")
    ball_core.set("entitydim", JInt(3))
    ball_core.set("condition", "inside")
    ball_core.set("r", JDouble(radius_m + tol))
    sphere_domain_selection = "sphere_domains"
    sphere_domains = [int(d) for d in ball_core.entities(JInt(3))]
    if not sphere_domains:
        raise RuntimeError("No interior dielectric domains found inside r <= a")

    inner_sheet_band_selection: str | None = None
    inner_sheet_band_domains: list[int] = []
    if inner_band_width_m is not None and inner_band_width_m > 0.0:
        component.selection().create("ball_inner_bulk", "Ball")
        b_in = component.selection("ball_inner_bulk")
        b_in.set("entitydim", JInt(3))
        b_in.set("condition", "inside")
        b_in.set("r", JDouble(radius_m - inner_band_width_m + tol))

        component.selection().create("inner_sheet_band", "Difference")
        diff_in = component.selection("inner_sheet_band")
        diff_in.label("Interior near-sheet dielectric refinement band")
        diff_in.set("entitydim", JInt(3))
        diff_in.set("add", ["sphere_domains"])
        diff_in.set("subtract", ["ball_inner_bulk"])
        inner_sheet_band_selection = "inner_sheet_band"
        inner_sheet_band_domains = [int(d) for d in diff_in.entities(JInt(3))]

    source_ball_selection: str | None = None
    source_ball_domains: list[int] = []
    if source_ball_radius_m is not None and source_ball_radius_m > 0.0:
        component.selection().create("source_ball", "Ball")
        b_src = component.selection("source_ball")
        b_src.label("Point-source refinement ball")
        b_src.set("entitydim", JInt(3))
        b_src.set("condition", "inside")
        b_src.set("posx", JDouble(source_position_m[0]))
        b_src.set("posy", JDouble(source_position_m[1]))
        b_src.set("posz", JDouble(source_position_m[2]))
        b_src.set("r", JDouble(source_ball_radius_m + tol))
        source_ball_selection = "source_ball"
        source_ball_domains = [int(d) for d in b_src.entities(JInt(3))]

    # 3. Metasurface zero-thickness boundary (r = a)
    sheet_boundary_selection = "geom1_sph1_bnd"
    sheet_faces = [int(b) for b in component.selection(sheet_boundary_selection).entities(JInt(2))]
    if not sheet_faces:
        raise RuntimeError("No metasurface boundary faces found on geom1_sph1_bnd")

    outer_sheet_band_selection: str | None = None
    outer_sheet_band_domains: list[int] = []
    if outer_band_width_m is not None and outer_band_width_m > 0.0:
        component.selection().create("ball_outer_band", "Ball")
        b_out = component.selection("ball_outer_band")
        b_out.set("entitydim", JInt(3))
        b_out.set("condition", "inside")
        b_out.set("r", JDouble(radius_m + outer_band_width_m + tol))

        component.selection().create("outer_sheet_band", "Difference")
        diff_out = component.selection("outer_sheet_band")
        diff_out.label("Exterior near-sheet air refinement band")
        diff_out.set("entitydim", JInt(3))
        diff_out.set("add", ["ball_outer_band"])
        diff_out.set("subtract", ["sphere_domains"])
        outer_sheet_band_selection = "outer_sheet_band"
        outer_sheet_band_domains = [int(d) for d in diff_out.entities(JInt(3))]

    # 4. PML domains (26 domains around the inner air box) & exterior air domains
    component.selection().create("inner_box", "Box")
    inner_box = component.selection("inner_box")
    inner_box.label("Inner box excluding PML")
    inner_box.set("entitydim", JInt(3))
    inner_box.set("condition", "inside")
    box_tol = air_extent_m + 1e-4 * wavelength_m
    for axis in ("x", "y", "z"):
        inner_box.set(f"{axis}min", JDouble(-box_tol))
        inner_box.set(f"{axis}max", JDouble(box_tol))
    inner_box_domains = [int(d) for d in inner_box.entities(JInt(3))]

    exterior_air_domains = sorted(set(inner_box_domains) - set(sphere_domains))
    if not exterior_air_domains:
        raise RuntimeError("No exterior air domains found between metasurface sphere and PML")

    if ff_sphere_tag is not None:
        far_field_surface_selection = f"geom1_{ff_sphere_tag}_bnd"
        far_field_faces = [
            int(b) for b in component.selection(far_field_surface_selection).entities(JInt(2))
        ]
        component.selection().create("inside_ff_sphere", "Ball")
        inside_ff = component.selection("inside_ff_sphere")
        inside_ff.set("entitydim", JInt(3))
        inside_ff.set("condition", "inside")
        inside_ff.set("r", JDouble(ff_radius_m + tol))
        inside_ff_domains = [int(d) for d in inside_ff.entities(JInt(3))]
        far_field_air_domains = sorted(set(inner_box_domains) - set(inside_ff_domains))
        if not far_field_air_domains:
            raise RuntimeError(
                f"No exterior air domains found outside Huygens sphere r_ff={ff_radius_m} m"
            )
    else:
        far_field_surface_selection = sheet_boundary_selection
        far_field_faces = list(sheet_faces)
        far_field_air_domains = list(exterior_air_domains)

    component.selection().create("pml_domains", "Complement")
    pml_sel = component.selection("pml_domains")
    pml_sel.label("PML layer domains")
    pml_sel.set("entitydim", JInt(3))
    pml_sel.set("input", ["inner_box"])
    pml_domains = [int(d) for d in pml_sel.entities(JInt(3))]
    if len(pml_domains) != 26:
        raise RuntimeError(f"Expected 26 PML domains, got {len(pml_domains)}: {pml_domains}")

    return SheetGeometrySelections(
        sphere_domain_selection=sphere_domain_selection,
        sheet_boundary_selection=sheet_boundary_selection,
        far_field_surface_selection=far_field_surface_selection,
        inner_sheet_band_selection=inner_sheet_band_selection,
        outer_sheet_band_selection=outer_sheet_band_selection,
        source_ball_selection=source_ball_selection,
        sphere_domains=sphere_domains,
        exterior_air_domains=exterior_air_domains,
        far_field_air_domains=far_field_air_domains,
        sheet_faces=sheet_faces,
        far_field_faces=far_field_faces,
        inner_sheet_band_domains=inner_sheet_band_domains,
        outer_sheet_band_domains=outer_sheet_band_domains,
        source_ball_domains=source_ball_domains,
        pml_domains=pml_domains,
        source_points=source_points,
    )


# =============================================================================
# 3. Phase 4: Swept & Tetrahedral Shell Meshing
# =============================================================================


@dataclass(frozen=True)
class MeshStatistics:
    """Element counts and verification diagnostics produced by Phase 4 meshing."""

    total_elements: int
    tetrahedra: int
    prisms: int
    source_triangles: int
    global_hmin_divisor: float


def mesh_element_count(mesh: Any, element_type: str) -> int:
    """Return the number of mesh elements of one COMSOL element type."""
    try:
        return len(mesh.getElemEntity(element_type))
    except Exception:
        return 0


def resolved_global_hmin_divisor(
    *,
    global_hmax_divisor: float,
    interior_hmax_divisor: float | None = None,
    source_ball_hmax_divisor: float | None = None,
    inner_shell_band_hmax_divisor: float | None = None,
    outer_shell_band_hmax_divisor: float | None = None,
    shell_mesh: str = "swept",
    shell_tangential_divisor: float = 20.0,
) -> float:
    """Choose a permissive global minimum size for all requested refinements."""
    divisors = [30.0, 5.0 * global_hmax_divisor]
    for opt_div in (
        interior_hmax_divisor,
        source_ball_hmax_divisor,
        inner_shell_band_hmax_divisor,
        outer_shell_band_hmax_divisor,
    ):
        if opt_div is not None:
            divisors.append(5.0 * opt_div)
    if shell_mesh == "swept":
        divisors.append(5.0 * shell_tangential_divisor)
    return max(divisors)


def setup_spherical_shell_mesh(
    component: Any,
    selections: GeometrySelections,
    *,
    shell_mesh: str = "swept",
    global_hmax_divisor: float = 6.0,
    interior_hmax_divisor: float | None = None,
    source_ball_hmax_divisor: float | None = None,
    inner_shell_band_hmax_divisor: float | None = None,
    outer_shell_band_hmax_divisor: float | None = None,
    shell_tangential_divisor: float = 20.0,
    shell_radial_layers: int = 2,
) -> MeshStatistics:
    """Build and execute the manual COMSOL mesh sequence for the spherical shell.

    Units & Dimensions Convention:
      All mesh sizing divisors passed to this function are dimensionless factors
      specifying the maximum element size hmax relative to wavelength:
        - Exterior/Global: hmax = lambda0 / global_hmax_divisor
        - Interior Dielectric Core: hmax = lambda_interior / interior_hmax_divisor
        - Shell Tangential: hmax_tan = lambda0 / shell_tangential_divisor
        - Source Ball: hmax = lambda0 / source_ball_hmax_divisor
        - Inner/Outer Bands: hmax = lambda / band_hmax_divisor
      ``shell_radial_layers`` is the integer count of prism element layers across the shell thickness tau.
    """
    component.mesh().create("mesh1")
    mesh = component.mesh("mesh1")

    # 1. Global size
    mesh.create("size1", "Size")
    mesh.feature("size1").set("custom", True)
    mesh.feature("size1").set("hmax", f"lambda0/{global_hmax_divisor:.17g}")
    global_hmin_div = resolved_global_hmin_divisor(
        global_hmax_divisor=global_hmax_divisor,
        interior_hmax_divisor=interior_hmax_divisor,
        source_ball_hmax_divisor=source_ball_hmax_divisor,
        inner_shell_band_hmax_divisor=inner_shell_band_hmax_divisor,
        outer_shell_band_hmax_divisor=outer_shell_band_hmax_divisor,
        shell_mesh=shell_mesh,
        shell_tangential_divisor=shell_tangential_divisor,
    )
    mesh.feature("size1").set("hmin", f"lambda0/{global_hmin_div:.17g}")

    # 2. Optional concentric domain size refinements
    def _add_domain_size(tag: str, label: str, selection_name: str, divisor: float) -> None:
        mesh.create(tag, "Size")
        sz = mesh.feature(tag)
        sz.label(label)
        sz.selection().geom("geom1", JInt(3))
        sz.selection().named(selection_name)
        sz.set("custom", True)
        sz.set("hmaxactive", True)
        sz.set("hmax", f"lambda0/{divisor:.17g}")
        sz.set("hminactive", True)
        sz.set("hmin", f"lambda0/{5.0 * divisor:.17g}")

    if interior_hmax_divisor is not None:
        _add_domain_size(
            "size_core",
            "Dielectric interior refinement",
            selections.sphere_domain_selection,
            interior_hmax_divisor,
        )
    if selections.source_ball_selection is not None and source_ball_hmax_divisor is not None:
        _add_domain_size(
            "size_source_ball",
            "Point-source refinement ball",
            selections.source_ball_selection,
            source_ball_hmax_divisor,
        )
    if (
        selections.inner_shell_band_selection is not None
        and inner_shell_band_hmax_divisor is not None
    ):
        _add_domain_size(
            "size_inner_band",
            "Interior near-shell dielectric refinement",
            selections.inner_shell_band_selection,
            inner_shell_band_hmax_divisor,
        )
    if (
        selections.outer_shell_band_selection is not None
        and outer_shell_band_hmax_divisor is not None
    ):
        _add_domain_size(
            "size_outer_band",
            "Exterior near-shell air refinement",
            selections.outer_shell_band_selection,
            outer_shell_band_hmax_divisor,
        )

    # 3. Volumetric shell mesh (swept prisms or tetrahedral)
    if shell_mesh == "tetra":
        mesh.create("size_shell", "Size")
        size_sh = mesh.feature("size_shell")
        size_sh.label("Metasurface volumetric shell refinement")
        size_sh.selection().geom("geom1", JInt(3))
        size_sh.selection().named(selections.sphere_shell_selection)
        size_sh.set("custom", True)
        size_sh.set("hmaxactive", True)
        size_sh.set("hmax", "tau")
        size_sh.set("hminactive", True)
        size_sh.set("hmin", "tau/5")
    else:
        assert selections.shell_outer_faces_selection is not None
        assert selections.shell_inner_faces_selection is not None

        mesh.create("ftri_shell", "FreeTri")
        free_tri = mesh.feature("ftri_shell")
        free_tri.label("Outer shell source-face triangles")
        free_tri.selection().named(selections.shell_outer_faces_selection)
        free_tri.create("size_shell_tan", "Size")
        tan_size = free_tri.feature("size_shell_tan")
        tan_size.label("Shell tangential resolution")
        tan_size.selection().named(selections.shell_outer_faces_selection)
        tan_size.set("custom", True)
        tan_size.set("hmaxactive", True)
        tan_size.set("hmax", f"lambda0/{shell_tangential_divisor:.17g}")
        tan_size.set("hminactive", True)
        tan_size.set("hmin", f"lambda0/{5.0 * shell_tangential_divisor:.17g}")

        mesh.create("swe_shell", "Sweep")
        swept = mesh.feature("swe_shell")
        swept.label("Swept volumetric shell refinement")
        swept.selection().named(selections.sphere_shell_selection)
        swept.selection("sourceface").named(selections.shell_outer_faces_selection)
        swept.selection("targetface").named(selections.shell_inner_faces_selection)
        swept.set("facemethod", "tri")
        swept.set("sweeppath", "straight")
        swept.set("targetmesh", "auto")
        swept.create("dist_shell", "Distribution")
        dist = swept.feature("dist_shell")
        dist.label("Uniform radial shell layers")
        dist.selection().named(selections.sphere_shell_selection)
        dist.set("type", "predefined")
        dist.set("elemcount", JInt(shell_radial_layers))
        dist.set("elemratio", JDouble(1.0))
        dist.set("method", "arithmetic")

    # 4. Tetrahedral mesh on all remaining domains
    mesh.create("ftet1", "FreeTet")
    mesh.run()

    element_count = int(mesh.getNumElem())
    tetrahedron_count = mesh_element_count(mesh, "tet")
    prism_count = mesh_element_count(mesh, "prism")
    source_triangle_count = 0

    if element_count == 0:
        raise RuntimeError("Generated COMSOL mesh is empty")

    if shell_mesh == "swept":
        triangle_entities = [int(entity) for entity in mesh.getElemEntity("tri")]
        outer_face_set = set(selections.outer_shell_faces)
        source_triangle_count = sum(entity in outer_face_set for entity in triangle_entities)
        expected_prism_count = source_triangle_count * shell_radial_layers
        if prism_count != expected_prism_count:
            raise RuntimeError(
                f"Swept-shell prism count ({prism_count}) != source triangles "
                f"({source_triangle_count}) x layers ({shell_radial_layers})"
            )
        prism_domains = {int(entity) for entity in mesh.getElemEntity("prism")}
        if prism_domains != set(selections.shell_domains):
            raise RuntimeError(
                f"Prisms do not occupy exactly the swept shell domains: "
                f"{sorted(prism_domains)} vs {selections.shell_domains}"
            )

    return MeshStatistics(
        total_elements=element_count,
        tetrahedra=tetrahedron_count,
        prisms=prism_count,
        source_triangles=source_triangle_count,
        global_hmin_divisor=global_hmin_div,
    )


def setup_spherical_sheet_mesh(
    component: Any,
    selections: SheetGeometrySelections,
    *,
    global_hmax_divisor: float = 6.0,
    interior_hmax_divisor: float | None = None,
    source_ball_hmax_divisor: float | None = None,
    inner_sheet_band_hmax_divisor: float | None = None,
    outer_sheet_band_hmax_divisor: float | None = None,
    sheet_tangential_divisor: float = 25.0,
    far_field_surface_divisor: float | None = None,
) -> MeshStatistics:
    """Build and execute the manual COMSOL mesh sequence for Case A zero-thickness sheet.

    Meshes the zero-thickness metasurface boundary (r = a) explicitly with ``FreeTri``
    at ``hmax = lambda0 / sheet_tangential_divisor``, optionally refines the detached
    Huygens far-field sphere, and meshes all 3D bulk/PML domains with conformal ``FreeTet``.
    """
    component.mesh().create("mesh1")
    mesh = component.mesh("mesh1")

    # 1. Global size
    mesh.create("size1", "Size")
    mesh.feature("size1").set("custom", True)
    mesh.feature("size1").set("hmax", f"lambda0/{global_hmax_divisor:.17g}")
    global_hmin_div = resolved_global_hmin_divisor(
        global_hmax_divisor=global_hmax_divisor,
        interior_hmax_divisor=interior_hmax_divisor,
        source_ball_hmax_divisor=source_ball_hmax_divisor,
        inner_shell_band_hmax_divisor=inner_sheet_band_hmax_divisor,
        outer_shell_band_hmax_divisor=outer_sheet_band_hmax_divisor,
        shell_mesh="swept",
        shell_tangential_divisor=max(
            sheet_tangential_divisor,
            far_field_surface_divisor if far_field_surface_divisor is not None else 0.0,
        ),
    )
    mesh.feature("size1").set("hmin", f"lambda0/{global_hmin_div:.17g}")

    # 2. Optional concentric domain size refinements
    def _add_domain_size(tag: str, label: str, selection_name: str, divisor: float) -> None:
        mesh.create(tag, "Size")
        sz = mesh.feature(tag)
        sz.label(label)
        sz.selection().geom("geom1", JInt(3))
        sz.selection().named(selection_name)
        sz.set("custom", True)
        sz.set("hmaxactive", True)
        sz.set("hmax", f"lambda0/{divisor:.17g}")
        sz.set("hminactive", True)
        sz.set("hmin", f"lambda0/{5.0 * divisor:.17g}")

    if interior_hmax_divisor is not None:
        _add_domain_size(
            "size_core",
            "Dielectric interior refinement",
            selections.sphere_domain_selection,
            interior_hmax_divisor,
        )
    if selections.source_ball_selection is not None and source_ball_hmax_divisor is not None:
        _add_domain_size(
            "size_source_ball",
            "Point-source refinement ball",
            selections.source_ball_selection,
            source_ball_hmax_divisor,
        )
    if (
        selections.inner_sheet_band_selection is not None
        and inner_sheet_band_hmax_divisor is not None
    ):
        _add_domain_size(
            "size_inner_band",
            "Interior near-sheet dielectric refinement",
            selections.inner_sheet_band_selection,
            inner_sheet_band_hmax_divisor,
        )
    if (
        selections.outer_sheet_band_selection is not None
        and outer_sheet_band_hmax_divisor is not None
    ):
        _add_domain_size(
            "size_outer_band",
            "Exterior near-sheet air refinement",
            selections.outer_sheet_band_selection,
            outer_sheet_band_hmax_divisor,
        )

    # 3. Optional Huygens far-field sphere surface size
    if (
        far_field_surface_divisor is not None
        and selections.far_field_surface_selection != selections.sheet_boundary_selection
    ):
        mesh.create("size_ff_surface", "Size")
        sz_ff = mesh.feature("size_ff_surface")
        sz_ff.label("Huygens far-field sphere surface resolution")
        sz_ff.selection().geom("geom1", JInt(2))
        sz_ff.selection().named(selections.far_field_surface_selection)
        sz_ff.set("custom", True)
        sz_ff.set("hmaxactive", True)
        sz_ff.set("hmax", f"lambda0/{far_field_surface_divisor:.17g}")
        sz_ff.set("hminactive", True)
        sz_ff.set("hmin", f"lambda0/{5.0 * far_field_surface_divisor:.17g}")

    # 4. Metasurface zero-thickness boundary triangulation (r = a)
    mesh.create("ftri_sheet", "FreeTri")
    free_tri = mesh.feature("ftri_sheet")
    free_tri.label("Metasurface zero-thickness boundary triangles (r = a)")
    free_tri.selection().named(selections.sheet_boundary_selection)
    free_tri.create("size_sheet_tan", "Size")
    tan_size = free_tri.feature("size_sheet_tan")
    tan_size.label("Sheet tangential resolution")
    tan_size.selection().named(selections.sheet_boundary_selection)
    tan_size.set("custom", True)
    tan_size.set("hmaxactive", True)
    tan_size.set("hmax", f"lambda0/{sheet_tangential_divisor:.17g}")
    tan_size.set("hminactive", True)
    tan_size.set("hmin", f"lambda0/{5.0 * sheet_tangential_divisor:.17g}")

    # 5. Tetrahedral mesh on all 3D domains
    mesh.create("ftet1", "FreeTet")
    mesh.run()

    element_count = int(mesh.getNumElem())
    tetrahedron_count = mesh_element_count(mesh, "tet")
    prism_count = mesh_element_count(mesh, "prism")

    if element_count == 0:
        raise RuntimeError("Generated COMSOL sheet mesh is empty")

    triangle_entities = [int(entity) for entity in mesh.getElemEntity("tri")]
    sheet_face_set = set(selections.sheet_faces)
    sheet_triangle_count = sum(entity in sheet_face_set for entity in triangle_entities)

    return MeshStatistics(
        total_elements=element_count,
        tetrahedra=tetrahedron_count,
        prisms=prism_count,
        source_triangles=sheet_triangle_count,
        global_hmin_divisor=global_hmin_div,
    )


# =============================================================================
# 4. Phase 5: Physics, 3D Constitutive Tensor Rotation, Equation View Overrides
#             & Far-Field Setup
# =============================================================================
#
# Physical & Mathematical Bridge (exp(+j*omega*t) convention):
# ------------------------------------------------------------
# 1. Local 10-Parameter Representation in (u, v) = (theta_hat, phi_hat):
#    The physical surface map [J_s; M_s] = [[Y, chi]; [gamma, Z]] [E_t; H_t]
#    is parameterized on each face by 10 real dimensionless numbers
#    theta = (y1, y2, y3, c1, c2, c3, c4, z1, z2, z3):
#
#      Y_loc = (j / eta0) * [[y1, y3],     Z_loc = (j * eta0) * [[z1, z3],
#                            [y3, y2]]                           [z3, z2]]
#
#      chi_loc = [[c1, c2],                gamma_loc = -chi_loc^T = [[-c1, -c3],
#                 [c3, c4]]                                          [-c2, -c4]]
#
# 2. Local Tangent Unit Vectors at Every Quadrature Point (x, y, z):
#    Using spherical angles theta_sph = acos(z / r) in [0, pi] and
#    phi_sph = atan2(y, x) in [-pi, pi]:
#
#      u = theta_hat = [ cos(theta_sph)*cos(phi_sph),  cos(theta_sph)*sin(phi_sph), -sin(theta_sph) ]^T
#      v = phi_hat   = [ -sin(phi_sph),                cos(phi_sph),                 0              ]^T
#
# 3. Local-to-Global 3D Rotation of Relative Permittivity & Permeability:
#    Dividing by (j * omega * eps0 * tau) and (j * omega * mu0 * tau) cancels
#    the imaginary unit j, yielding purely REAL, SYMMETRIC 3x3 relative tensors
#    with radial eigenvalue 1:
#
#      eps_r,s = I_3D + (1 / (omega * eps0 * eta0 * tau))
#                       * [ y1*(u u^T) + y3*(u v^T + v u^T) + y2*(v v^T) ]
#
#      mu_r,s  = I_3D + (eta0 / (omega * mu0 * tau))
#                       * [ z1*(u u^T) + z3*(u v^T + v u^T) + z2*(v v^T) ]
#
# 4. Magneto-Electric Cross-Coupling & COMSOL Equation View Overrides (weq2):
#    Instead of forming 9 Cartesian components for chi_3D and gamma_3D, we project
#    E and H onto the local tangent basis (u, v), apply the 2x2 coupling matrices,
#    and rotate the resulting sheet currents back to 3D Cartesian coordinates:
#
#      E_th = u . E,   E_ph = v . E
#      H_th = u . H,   H_ph = v . H
#
#      J_s^(m) = chi_3D * H   = u * (c1*H_th + c2*H_ph) + v * (c3*H_th + c4*H_ph)
#      M_s^(e) = gamma_3D * E = -u * (c1*E_th + c3*E_ph) - v * (c2*E_th + c4*E_ph)
#
#    Across the thin volumetric shell of thickness tau:
#    * In Ampere-Maxwell's Law (curl H = j*omega*D = j*omega*(eps0*E + P)):
#        P_cross = J_s^(m) / (j * omega * tau)
#      -> Override {phys}.Px, {phys}.Py, {phys}.Pz by adding P_cross_{x,y,z}.
#
#    * In Faraday's Law (curl E = -j*omega*B, with B = mu0*mu_r,s*H + M_s^(e)/(j*omega*tau)):
#      Inverting for H in terms of COMSOL's state variable B and E gives:
#        H_cross    = - (1 / (j * omega * tau * mu0)) * mu_r,s^(-1) * M_s^(e)
#        dHdt_cross = j * omega * H_cross
#                   = - (1 / (tau * mu0)) * mu_r,s^(-1) * M_s^(e)
#      -> Override {phys}.Hx, {phys}.Hy, {phys}.Hz by adding H_cross_{x,y,z}, and
#      -> Override {phys}.dHdtx, {phys}.dHdty, {phys}.dHdtz by adding dHdt_cross_{x,y,z}.


def _symmetric_tangent_3d_tensor(
    scale_expr: str,
    t_uu: str,
    t_uv: str,
    t_vv: str,
) -> list[list[str]]:
    """Construct the 3x3 string matrix I_3D + scale * [t_uu*(u u^T) + t_uv*(u v^T + v u^T) + t_vv*(v v^T)]."""
    axes = ("x", "y", "z")
    rows: list[list[str]] = []
    for i, ax_i in enumerate(axes):
        row: list[str] = []
        for j, ax_j in enumerate(axes):
            diag = "1 + " if i == j else ""
            ui, uj = f"gstc_u{ax_i}", f"gstc_u{ax_j}"
            vi, vj = f"gstc_v{ax_i}", f"gstc_v{ax_j}"
            term = (
                f"{diag}{scale_expr}*("
                f"{t_uu}*{ui}*{uj} + {t_uv}*({ui}*{vj} + {vi}*{uj}) + {t_vv}*{vi}*{vj}"
                f")"
            )
            row.append(term)
        rows.append(row)
    return rows


def setup_bianisotropic_physics(
    model_java: Any,
    component: Any,
    selections: GeometrySelections,
    *,
    grid_files: Mapping[str, Path],
    version: str,
    physics_interface: str = "auto",
    discretization: str = "linear",
    dipole_type: str = "electric",
    dipole_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    enable_far_field: bool = True,
) -> str:
    """Configure materials, 2D interpolation functions, physics, Equation View overrides, and Far-Field.

    Automatically handles licensing differences between COMSOL versions:
      - COMSOL 5.4 / 6.4 typically provide Wave Optics (`ewfd`, `ElectromagneticWavesFrequencyDomain`)
      - COMSOL 6.1 typically provides RF Module (`emw`, `ElectromagneticWaves`)

    Returns the resolved physics tag (`'ewfd'` or `'emw'`).
    """
    # -------------------------------------------------------------------------
    # Step 5.1: Register the 10 2D (theta, phi) Interpolation Functions
    # -------------------------------------------------------------------------
    for name in PARAMETER_NAMES:
        if name not in grid_files:
            raise KeyError(f"Missing COMSOL grid file for parameter '{name}'")
        func_tag = f"int_{name}"
        func = model_java.func().create(func_tag, "Interpolation")
        func.set("source", "file")
        func.set("filename", grid_files[name].resolve().as_posix())
        func.importData()
        func.set("interp", "linear")
        func.set("extrap", "const")

    # -------------------------------------------------------------------------
    # Step 5.2: PML Coordinate System & Domain Materials (Air + Dielectric Core)
    # -------------------------------------------------------------------------
    component.coordSystem().create("pml1", "PML")
    component.coordSystem("pml1").label("Cartesian PML Coordinate System")
    component.coordSystem("pml1").selection().named("pml_domains")

    # Material 1: Exterior Air / Vacuum (applied to all domains, overridden in core)
    component.material().create("mat1", "Common")
    mat1 = component.material("mat1")
    mat1.label("Exterior Medium (eps_r = er_ext, mu_r = mur_ext)")
    mat1.selection().all()
    mat1.propertyGroup("def").set("relpermittivity", ["er_ext"])
    mat1.propertyGroup("def").set("relpermeability", ["mur_ext"])
    mat1.propertyGroup("def").set("electricconductivity", ["0[S/m]"])
    try:
        mat1.propertyGroup().create("RefractiveIndex", "Refractive index")
        mat1.propertyGroup("RefractiveIndex").set("n", ["sqrt(er_ext*mur_ext)"])
        mat1.propertyGroup("RefractiveIndex").set("ki", ["0"])
    except Exception:
        pass

    # Material 2: Interior Dielectric Sphere (r <= a)
    component.material().create("mat2", "Common")
    mat2 = component.material("mat2")
    mat2.label("Interior Dielectric Core (eps_r = er_int, mu_r = mur_int)")
    mat2.selection().named(selections.sphere_domain_selection)
    mat2.propertyGroup("def").set("relpermittivity", ["er_int"])
    mat2.propertyGroup("def").set("relpermeability", ["mur_int"])
    mat2.propertyGroup("def").set("electricconductivity", ["0[S/m]"])
    try:
        mat2.propertyGroup().create("RefractiveIndex", "Refractive index")
        mat2.propertyGroup("RefractiveIndex").set("n", ["sqrt(er_int*mur_int)"])
        mat2.propertyGroup("RefractiveIndex").set("ki", ["0"])
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Step 5.3: Create Electromagnetic Waves Physics Interface (emw or ewfd)
    # -------------------------------------------------------------------------
    # Note: COMSOL 6.1 uses RF ('emw'), whereas COMSOL 5.4 / 6.4 use Wave Optics ('ewfd').
    # When physics_interface == 'auto', we try the version-preferred interface first
    # and automatically fall back if a LicenseException occurs.
    if physics_interface == "auto":
        candidates = (
            [("ewfd", "ElectromagneticWavesFrequencyDomain"), ("emw", "ElectromagneticWaves")]
            if str(version).startswith("5.") or str(version).startswith("6.4")
            else [("emw", "ElectromagneticWaves"), ("ewfd", "ElectromagneticWavesFrequencyDomain")]
        )
    elif physics_interface == "ewfd":
        candidates = [("ewfd", "ElectromagneticWavesFrequencyDomain")]
    elif physics_interface == "emw":
        candidates = [("emw", "ElectromagneticWaves")]
    else:
        raise ValueError(f"Unsupported physics_interface: {physics_interface!r}")

    physics_tag: str | None = None
    last_error: Exception | None = None
    for tag, phys_type in candidates:
        try:
            component.physics().create(tag, phys_type, "geom1")
            physics_tag = tag
            print(f"[Physics] Activated COMSOL interface '{tag}' ({phys_type})", flush=True)
            break
        except Exception as exc:
            last_error = exc
            continue
    if physics_tag is None:
        raise RuntimeError(
            f"Could not create COMSOL Electromagnetic Waves interface (tried {candidates})"
        ) from last_error

    phys = component.physics(physics_tag)
    phys.selection().all()

    if discretization == "linear":
        for val in (1, "1"):
            try:
                phys.prop("ShapeProperty").set("order_electricfield", val)
                break
            except Exception:
                continue

    # Wave Optics (ewfd) defaults to RefractiveIndex model; switch wee1 to RelativePermittivity
    if physics_tag == "ewfd":
        try:
            phys.feature("wee1").set("DisplacementFieldModel", "RelativePermittivity")
        except Exception:
            pass

    # Exterior Scattering Boundary Condition on outer PML faces
    phys.create("sctr1", "Scattering", 2)
    phys.feature("sctr1").label("Exterior Scattering Boundary")
    phys.feature("sctr1").selection().all()

    # -------------------------------------------------------------------------
    # Step 5.4: Point Dipole Excitation (Electric or Magnetic)
    # -------------------------------------------------------------------------
    if dipole_type == "electric":
        for epd_type in ("ElectricPointDipole", "PointDipole", "PointDipoleSource"):
            try:
                phys.create("src1", epd_type, 0)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"Failed to create ElectricPointDipole in '{physics_tag}'")
        src = phys.feature("src1")
        src.label("Electric Point Dipole Source")
        src.selection().named("dipole_origin")
        src.set("normpI", "pI")
        src.set("enpI", java_double_array(dipole_direction))
    elif dipole_type == "magnetic":
        for mpd_type in ("MagneticPointDipole", "PointMagneticDipole"):
            try:
                phys.create("src1", mpd_type, 0)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"Failed to create MagneticPointDipole in '{physics_tag}'")
        src = phys.feature("src1")
        src.label("Magnetic Point Dipole Source")
        src.selection().named("dipole_origin")
        src.set("normm", "mI")
        src.set("enm", java_double_array(dipole_direction))
    else:
        raise ValueError(f"Unsupported dipole_type: {dipole_type!r}")

    # -------------------------------------------------------------------------
    # Step 5.5: Shell Local Spherical Basis & Interpolated Parameter Variables
    # -------------------------------------------------------------------------
    component.variable().create("var_gstc_shell")
    shell_vars = component.variable("var_gstc_shell")
    shell_vars.label("Bianisotropic GSTC Shell Basis, Parameters, and Sheet Currents")
    shell_vars.selection().named(selections.sphere_shell_selection)

    # Spherical coordinates on the shell (r in [a, a + tau] is strictly away from 0)
    shell_vars.set("gstc_r", "sqrt(x^2 + y^2 + z^2)")
    shell_vars.set("gstc_theta", "acos(z / gstc_r)")
    shell_vars.set("gstc_phi", "atan2(y, x)")

    # Local spherical tangent unit vectors u = theta_hat, v = phi_hat
    shell_vars.set("gstc_ux", "cos(gstc_theta)*cos(gstc_phi)")
    shell_vars.set("gstc_uy", "cos(gstc_theta)*sin(gstc_phi)")
    shell_vars.set("gstc_uz", "-sin(gstc_theta)")
    shell_vars.set("gstc_vx", "-sin(gstc_phi)")
    shell_vars.set("gstc_vy", "cos(gstc_phi)")
    shell_vars.set("gstc_vz", "0")

    # Evaluate the 10 dimensionless interpolated surface parameters at (gstc_theta, gstc_phi)
    for name in PARAMETER_NAMES:
        shell_vars.set(f"gstc_{name}", f"int_{name}(gstc_theta, gstc_phi)")

    # Project 3D Cartesian fields (E, H) onto local tangent basis (u = theta_hat, v = phi_hat)
    shell_vars.set(
        "gstc_Eth",
        f"(gstc_ux*{physics_tag}.Ex + gstc_uy*{physics_tag}.Ey + gstc_uz*{physics_tag}.Ez)",
    )
    shell_vars.set(
        "gstc_Eph",
        f"(gstc_vx*{physics_tag}.Ex + gstc_vy*{physics_tag}.Ey + gstc_vz*{physics_tag}.Ez)",
    )
    shell_vars.set(
        "gstc_Hth",
        f"(gstc_ux*{physics_tag}.Hx + gstc_uy*{physics_tag}.Hy + gstc_uz*{physics_tag}.Hz)",
    )
    shell_vars.set(
        "gstc_Hph",
        f"(gstc_vx*{physics_tag}.Hx + gstc_vy*{physics_tag}.Hy + gstc_vz*{physics_tag}.Hz)",
    )

    # Magneto-electric electric sheet current J_s^(m) = chi_loc * H_t in (u, v) and 3D Cartesian:
    #   [J_sm,theta; J_sm,phi] = [[c1, c2], [c3, c4]] [H_th; H_ph]
    shell_vars.set("gstc_Jsm_th", "(gstc_c1*gstc_Hth + gstc_c2*gstc_Hph)")
    shell_vars.set("gstc_Jsm_ph", "(gstc_c3*gstc_Hth + gstc_c4*gstc_Hph)")
    for ax in ("x", "y", "z"):
        shell_vars.set(
            f"gstc_Jsm_{ax}",
            f"(gstc_u{ax}*gstc_Jsm_th + gstc_v{ax}*gstc_Jsm_ph)",
        )

    # Electro-magnetic magnetic sheet current M_s^(e) = gamma_loc * E_t = -chi_loc^T * E_t:
    #   [M_se,theta; M_se,phi] = [[-c1, -c3], [-c2, -c4]] [E_th; E_ph]
    shell_vars.set("gstc_Mse_th", "-(gstc_c1*gstc_Eth + gstc_c3*gstc_Eph)")
    shell_vars.set("gstc_Mse_ph", "-(gstc_c2*gstc_Eth + gstc_c4*gstc_Eph)")
    for ax in ("x", "y", "z"):
        shell_vars.set(
            f"gstc_Mse_{ax}",
            f"(gstc_u{ax}*gstc_Mse_th + gstc_v{ax}*gstc_Mse_ph)",
        )

    # -------------------------------------------------------------------------
    # Step 5.6: Volumetric Shell Wave Equation (weq2) & 3D Tensors eps_r,s, mu_r,s
    # -------------------------------------------------------------------------
    phys.create("weq2", "WaveEquationElectric", 3)
    weq2 = phys.feature("weq2")
    weq2.label("Bianisotropic GSTC Volumetric Shell")
    weq2.selection().named(selections.sphere_shell_selection)
    if physics_tag == "ewfd":
        try:
            weq2.set("DisplacementFieldModel", "RelativePermittivity")
        except Exception:
            pass

    # Scale factors converting dimensionless (y1,y2,y3) and (z1,z2,z3) to relative tensors:
    #   Y_loc = (j / eta0) * [[y1, y3], [y3, y2]]
    #   eps_r,s = I_3D + Y_3D / (j * omega * eps0 * tau)
    #           = I_3D + (1 / (omega * eps0 * eta0 * tau)) * [y1*uu^T + y3*(uv^T+vu^T) + y2*vv^T]
    #
    #   Z_loc = (j * eta0) * [[z1, z3], [z3, z2]]
    #   mu_r,s  = I_3D + Z_3D / (j * omega * mu0 * tau)
    #           = I_3D + (eta0 / (omega * mu0 * tau)) * [z1*uu^T + z3*(uv^T+vu^T) + z2*vv^T]
    scale_eps = f"(1 / ({physics_tag}.omega * epsilon0_const * eta0 * tau))"
    scale_mu = f"(eta0 / ({physics_tag}.omega * mu0_const * tau))"

    weq2.set("epsilonr_mat", "userdef")
    weq2.set(
        "epsilonr",
        java_string_tensor(
            _symmetric_tangent_3d_tensor(scale_eps, "gstc_y1", "gstc_y3", "gstc_y2")
        ),
    )

    weq2.set("mur_mat", "userdef")
    weq2.set(
        "mur",
        java_string_tensor(
            _symmetric_tangent_3d_tensor(scale_mu, "gstc_z1", "gstc_z3", "gstc_z2")
        ),
    )

    # -------------------------------------------------------------------------
    # Step 5.7: COMSOL Equation View Overrides on weq2 (featureInfo("info"))
    # -------------------------------------------------------------------------
    # Here we inject the bianisotropic magneto-electric cross-coupling into COMSOL's
    # weak-form curl-curl wave equation on the shell domain:
    #
    #   1. Electric Polarization Vector {phys}.Px, {phys}.Py, {phys}.Pz:
    #      Default COMSOL expression:
    #        P_i = epsilon0_const * (sum_j epsilonr_{ij} * E_j - E_i)
    #      Modified bianisotropic expression:
    #        P_i = epsilon0_const * (sum_j epsilonr_{ij} * E_j - E_i)
    #              + gstc_Jsm_i / (j * {phys}.omega * tau)
    #
    #   2. Magnetic Field Vector {phys}.Hx, {phys}.Hy, {phys}.Hz:
    #      Default COMSOL expression:
    #        H_i = (sum_j murinv_{ij} * B_j) / mu0_const
    #      Modified bianisotropic expression (since B = mu0*mur*H + M_s^(e)/(j*omega*tau)):
    #        H_i = (sum_j murinv_{ij} * B_j) / mu0_const
    #              - (1 / (j * {phys}.omega * tau * mu0_const)) * (sum_j murinv_{ij} * gstc_Mse_j)
    #
    #   3. Time Derivative of Magnetic Field {phys}.dHdtx, {phys}.dHdty, {phys}.dHdtz:
    #      Default COMSOL expression (used directly in the weak form):
    #        dHdt_i = (sum_j murinv_{ij} * dBdt_j) / mu0_const
    #      Modified bianisotropic expression (multiplying H_cross by j*omega):
    #        dHdt_i = (sum_j murinv_{ij} * dBdt_j) / mu0_const
    #                 - (1 / (tau * mu0_const)) * (sum_j murinv_{ij} * gstc_Mse_j)
    info = weq2.featureInfo("info")

    for ax in ("x", "y", "z"):
        # Standard anisotropic electric polarization P_e,i
        p_aniso = (
            f"epsilon0_const*("
            f"{physics_tag}.epsilonr{ax}x*{physics_tag}.Ex + "
            f"{physics_tag}.epsilonr{ax}y*{physics_tag}.Ey + "
            f"{physics_tag}.epsilonr{ax}z*{physics_tag}.Ez - "
            f"{physics_tag}.E{ax})"
        )
        # Cross-coupling electric polarization P_cross,i = J_s^(m)_i / (j * omega * tau)
        p_cross = f"(gstc_Jsm_{ax} / (j * {physics_tag}.omega * tau))"
        info.set(f"{physics_tag}.P{ax}", 0, [f"{p_aniso} + {p_cross}"])

        # Apply inverse permeability tensor murinv_{ij} to magnetic sheet current M_s^(e)_j
        murinv_Mse = (
            f"({physics_tag}.murinv{ax}x*gstc_Mse_x + "
            f"{physics_tag}.murinv{ax}y*gstc_Mse_y + "
            f"{physics_tag}.murinv{ax}z*gstc_Mse_z)"
        )

        # Standard anisotropic magnetic field H_m,i = (murinv * B)_i / mu0
        h_aniso = (
            f"({physics_tag}.murinv{ax}x*{physics_tag}.Bx + "
            f"{physics_tag}.murinv{ax}y*{physics_tag}.By + "
            f"{physics_tag}.murinv{ax}z*{physics_tag}.Bz) / mu0_const"
        )
        # Cross-coupling magnetic field H_cross,i = - (murinv * M_s^(e))_i / (j * omega * tau * mu0)
        h_cross = f"-({murinv_Mse} / (j * {physics_tag}.omega * tau * mu0_const))"
        info.set(f"{physics_tag}.H{ax}", 0, [f"{h_aniso} + ({h_cross})"])

        # Standard anisotropic dH/dt_i = (murinv * dB/dt)_i / mu0
        dhdt_aniso = (
            f"({physics_tag}.murinv{ax}x*{physics_tag}.dBdtx + "
            f"{physics_tag}.murinv{ax}y*{physics_tag}.dBdty + "
            f"{physics_tag}.murinv{ax}z*{physics_tag}.dBdtz) / mu0_const"
        )
        # Cross-coupling dH/dt_cross,i = j*omega*H_cross,i = - (murinv * M_s^(e))_i / (tau * mu0)
        dhdt_cross = f"-({murinv_Mse} / (tau * mu0_const))"
        info.set(f"{physics_tag}.dHdt{ax}", 0, [f"{dhdt_aniso} + ({dhdt_cross})"])

    # -------------------------------------------------------------------------
    # Step 5.8: Stratton-Chu Far-Field Calculation on Outer Shell Boundary (r = a + tau)
    # -------------------------------------------------------------------------
    if enable_far_field and selections.shell_outer_faces_selection is not None:
        ffd = phys.create("ffd1", "FarFieldDomain", 3)
        ffd.label("Exterior Homogeneous Air Far-Field Domain")
        ffd.selection().set(JArray(JInt)(selections.exterior_air_domains))
        ffc = ffd.feature("ffc1")
        ffc.label("Outer Shell Stratton-Chu Far-Field Surface (r = a + tau)")
        ffc.selection().named(selections.shell_outer_faces_selection)

    return physics_tag


# =============================================================================
# Explicit Physics Setup Wrappers (COMSOL 5.4 vs. COMSOL 6.1)
# =============================================================================

def setup_physics_comsol_54(
    model_java: Any,
    component: Any,
    selections: GeometrySelections,
    *,
    grid_files: Mapping[str, Path],
    discretization: str = "linear",
    dipole_type: str = "electric",
    dipole_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    enable_far_field: bool = True,
) -> str:
    """Configure bianisotropic physics specifically for COMSOL 5.4 (Wave Optics ewfd).

    In COMSOL 5.4 desktop environments without the RF Module license, the Wave Optics module
    interface ('ewfd', ElectromagneticWavesFrequencyDomain) is used.
    By default, Wave Optics exposes refractive index ('n', 'k') rather than relative
    permittivity/permeability ('epsilonr', 'mur').
    This function automatically configures:
      1. DisplacementFieldModel = 'RelativePermittivity' on both domain features.
      2. RefractiveIndex property group on materials to satisfy Wave Optics requirements.
      3. Equation View overrides on weq2 for bianisotropic magneto-electric coupling.
      4. Stratton-Chu FarFieldDomain (ffd1) on outer shell boundary (r = a + tau).
    """
    return setup_bianisotropic_physics(
        model_java=model_java,
        component=component,
        selections=selections,
        grid_files=grid_files,
        version="5.4",
        physics_interface="ewfd",
        discretization=discretization,
        dipole_type=dipole_type,
        dipole_direction=dipole_direction,
        enable_far_field=enable_far_field,
    )


def setup_physics_comsol_61(
    model_java: Any,
    component: Any,
    selections: GeometrySelections,
    *,
    grid_files: Mapping[str, Path],
    discretization: str = "linear",
    dipole_type: str = "electric",
    dipole_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    enable_far_field: bool = True,
) -> str:
    """Configure bianisotropic physics specifically for COMSOL 6.1 (RF Module emw).

    In COMSOL 6.1 cluster/workstation environments with the RF Module license,
    the RF interface ('emw', ElectromagneticWaves) is used.
    This function automatically configures:
      1. RF Module frequency-domain interface emw.
      2. Anisotropic relative permittivity and permeability tensors.
      3. Equation View overrides on weq2 for bianisotropic magneto-electric coupling.
      4. Stratton-Chu FarFieldDomain (ffd1) on outer shell boundary (r = a + tau).
    """
    return setup_bianisotropic_physics(
        model_java=model_java,
        component=component,
        selections=selections,
        grid_files=grid_files,
        version="6.1",
        physics_interface="emw",
        discretization=discretization,
        dipole_type=dipole_type,
        dipole_direction=dipole_direction,
        enable_far_field=enable_far_field,
    )


# =============================================================================
# Case A: Zero-Thickness Sheet GSTC Physics (scu1 + smcd1 + up/down Traces)
# =============================================================================


@dataclass(frozen=True)
class DipoleExcitationCase:
    """Single central point-dipole excitation case for single- or multi-RHS solves."""

    case_id: str
    dipole_type: str
    dipole_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    effective_moment_amplitude: complex = 1.0 + 0.0j
    raw_excitations: list[dict[str, Any]] | None = None


def _create_point_dipole_feature(
    phys: Any,
    *,
    physics_tag: str,
    feature_tag: str,
    label: str,
    dipole_type: str,
    dipole_direction: tuple[float, float, float],
    weight_expr: str | None = None,
) -> None:
    """Create an ElectricPointDipole or MagneticPointDipole feature on ``dipole_origin``."""
    if dipole_type == "electric":
        for epd_type in ("ElectricPointDipole", "PointDipole", "PointDipoleSource"):
            try:
                phys.create(feature_tag, epd_type, 0)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"Failed to create ElectricPointDipole in '{physics_tag}'")
        src = phys.feature(feature_tag)
        src.label(label)
        src.selection().named("dipole_origin")
        src.set("normpI", f"pI*({weight_expr})" if weight_expr else "pI")
        src.set("enpI", java_double_array(dipole_direction))
    elif dipole_type == "magnetic":
        for mpd_type in ("MagneticPointDipole", "PointMagneticDipole"):
            try:
                phys.create(feature_tag, mpd_type, 0)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"Failed to create MagneticPointDipole in '{physics_tag}'")
        src = phys.feature(feature_tag)
        src.label(label)
        src.selection().named("dipole_origin")
        src.set("normm", f"mI*({weight_expr})" if weight_expr else "mI")
        src.set("enm", java_double_array(dipole_direction))
    else:
        raise ValueError(f"Unsupported dipole_type: {dipole_type!r}")


def setup_bianisotropic_sheet_physics(
    model_java: Any,
    component: Any,
    selections: SheetGeometrySelections,
    *,
    grid_files: Mapping[str, Path],
    version: str,
    physics_interface: str = "auto",
    discretization: str = "linear",
    dipole_type: str = "electric",
    dipole_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    excitation_cases: Sequence[DipoleExcitationCase] | None = None,
    enable_far_field: bool = True,
) -> str:
    """Configure Case A zero-thickness bianisotropic GSTC sheet physics on r = a.

    Enforces the exact two-sided GSTC jump conditions across the interior spherical
    boundary ``selections.sheet_boundary_selection`` (r = a) using COMSOL's broken-H(curl)
    trace operators ``up(...)`` and ``down(...)``:
      1. ``SurfaceCurrent`` (``scu1``): weak boundary load enforcing
         r_hat x (H_ext - H_int) = J_s(E_av, H_av).
      2. ``SurfaceMagneticCurrentDensity`` (``smcd1``): nodal tangential E-jump constraint
         enforcing E_t^+ - E_t^- - n_d x J_ms(E_av, H_av) = 0, i.e.,
         -r_hat x (E_ext - E_int) = M_s(E_av, H_av).

    Uses the analytic outward spherical orthonormal tangent frame (u = theta_hat, v = phi_hat)
    defined directly from (x, y, z) on r = a so that u x v = +r_hat everywhere, independent
    of COMSOL's internal patch normal orientation.
    """
    # -------------------------------------------------------------------------
    # Step 1: Register the 10 2D (theta, phi) Interpolation Functions
    # -------------------------------------------------------------------------
    for name in PARAMETER_NAMES:
        if name not in grid_files:
            raise KeyError(f"Missing COMSOL grid file for parameter '{name}'")
        func_tag = f"int_{name}"
        func = model_java.func().create(func_tag, "Interpolation")
        func.set("source", "file")
        func.set("filename", grid_files[name].resolve().as_posix())
        func.importData()
        func.set("interp", "linear")
        func.set("extrap", "const")

    # -------------------------------------------------------------------------
    # Step 2: PML Coordinate System & Bulk Domain Materials (Air + Core)
    # -------------------------------------------------------------------------
    component.coordSystem().create("pml1", "PML")
    component.coordSystem("pml1").label("Cartesian PML Coordinate System")
    component.coordSystem("pml1").selection().named("pml_domains")

    component.material().create("mat1", "Common")
    mat1 = component.material("mat1")
    mat1.label("Exterior Medium (eps_r = er_ext, mu_r = mur_ext)")
    mat1.selection().all()
    mat1.propertyGroup("def").set("relpermittivity", ["er_ext"])
    mat1.propertyGroup("def").set("relpermeability", ["mur_ext"])
    mat1.propertyGroup("def").set("electricconductivity", ["0[S/m]"])
    try:
        mat1.propertyGroup().create("RefractiveIndex", "Refractive index")
        mat1.propertyGroup("RefractiveIndex").set("n", ["sqrt(er_ext*mur_ext)"])
        mat1.propertyGroup("RefractiveIndex").set("ki", ["0"])
    except Exception:
        pass

    component.material().create("mat2", "Common")
    mat2 = component.material("mat2")
    mat2.label("Interior Dielectric Core (eps_r = er_int, mu_r = mur_int)")
    mat2.selection().named(selections.sphere_domain_selection)
    mat2.propertyGroup("def").set("relpermittivity", ["er_int"])
    mat2.propertyGroup("def").set("relpermeability", ["mur_int"])
    mat2.propertyGroup("def").set("electricconductivity", ["0[S/m]"])
    try:
        mat2.propertyGroup().create("RefractiveIndex", "Refractive index")
        mat2.propertyGroup("RefractiveIndex").set("n", ["sqrt(er_int*mur_int)"])
        mat2.propertyGroup("RefractiveIndex").set("ki", ["0"])
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Step 3: Create Electromagnetic Waves Physics Interface (emw or ewfd)
    # -------------------------------------------------------------------------
    if physics_interface == "auto":
        candidates = (
            [("ewfd", "ElectromagneticWavesFrequencyDomain"), ("emw", "ElectromagneticWaves")]
            if str(version).startswith("5.") or str(version).startswith("6.4")
            else [("emw", "ElectromagneticWaves"), ("ewfd", "ElectromagneticWavesFrequencyDomain")]
        )
    elif physics_interface == "ewfd":
        candidates = [("ewfd", "ElectromagneticWavesFrequencyDomain")]
    elif physics_interface == "emw":
        candidates = [("emw", "ElectromagneticWaves")]
    else:
        raise ValueError(f"Unsupported physics_interface: {physics_interface!r}")

    physics_tag: str | None = None
    last_error: Exception | None = None
    for tag, phys_type in candidates:
        try:
            component.physics().create(tag, phys_type, "geom1")
            physics_tag = tag
            print(f"[Physics] Activated COMSOL interface '{tag}' ({phys_type})", flush=True)
            break
        except Exception as exc:
            last_error = exc
            continue
    if physics_tag is None:
        raise RuntimeError(
            f"Could not create COMSOL Electromagnetic Waves interface (tried {candidates})"
        ) from last_error

    phys = component.physics(physics_tag)
    phys.selection().all()

    if discretization == "linear":
        for val in (1, "1"):
            try:
                phys.prop("ShapeProperty").set("order_electricfield", val)
                break
            except Exception:
                continue

    if physics_tag == "ewfd":
        try:
            phys.feature("wee1").set("DisplacementFieldModel", "RelativePermittivity")
        except Exception:
            pass

    phys.create("sctr1", "Scattering", 2)
    phys.feature("sctr1").label("Exterior Scattering Boundary")
    phys.feature("sctr1").selection().all()

    # -------------------------------------------------------------------------
    # Step 4: Point Dipole Excitation (Single-RHS or Parametric Multi-RHS)
    # -------------------------------------------------------------------------
    if excitation_cases is not None and len(excitation_cases) > 1:
        model_java.param().set("case_idx", "1")
        for idx, exc_case in enumerate(excitation_cases, start=1):
            weight_param = f"w_case_{idx}"
            model_java.param().set(weight_param, f"(case_idx=={idx})")
            _create_point_dipole_feature(
                phys,
                physics_tag=physics_tag,
                feature_tag=f"src{idx}",
                label=f"{exc_case.dipole_type.capitalize()} Point Dipole ({exc_case.case_id})",
                dipole_type=exc_case.dipole_type,
                dipole_direction=exc_case.dipole_direction,
                weight_expr=weight_param,
            )
    elif excitation_cases is not None and len(excitation_cases) == 1:
        exc_case = excitation_cases[0]
        _create_point_dipole_feature(
            phys,
            physics_tag=physics_tag,
            feature_tag="src1",
            label=f"{exc_case.dipole_type.capitalize()} Point Dipole Source",
            dipole_type=exc_case.dipole_type,
            dipole_direction=exc_case.dipole_direction,
        )
    else:
        _create_point_dipole_feature(
            phys,
            physics_tag=physics_tag,
            feature_tag="src1",
            label=f"{dipole_type.capitalize()} Point Dipole Source",
            dipole_type=dipole_type,
            dipole_direction=dipole_direction,
        )

    # -------------------------------------------------------------------------
    # Step 5: Boundary Variables on Zero-Thickness Sheet (r = a)
    # -------------------------------------------------------------------------
    component.variable().create("var_gstc_sheet")
    sheet_vars = component.variable("var_gstc_sheet")
    sheet_vars.label("Bianisotropic GSTC Zero-Thickness Sheet Basis & Currents (Case A)")
    sheet_vars.selection().geom("geom1", JInt(2))
    sheet_vars.selection().named(selections.sheet_boundary_selection)

    # Numerically pole-safe spherical angles in [0, pi] and [-pi, pi]
    sheet_vars.set("gstc_rho", "sqrt(x^2 + y^2)")
    sheet_vars.set("gstc_theta", "atan2(gstc_rho, z)")
    sheet_vars.set("gstc_phi", "atan2(y, x)")

    # Outward spherical orthonormal tangent basis: u = theta_hat, v = phi_hat (u x v = +r_hat)
    sheet_vars.set("gstc_ux", "cos(gstc_theta)*cos(gstc_phi)")
    sheet_vars.set("gstc_uy", "cos(gstc_theta)*sin(gstc_phi)")
    sheet_vars.set("gstc_uz", "-sin(gstc_theta)")
    sheet_vars.set("gstc_vx", "-sin(gstc_phi)")
    sheet_vars.set("gstc_vy", "cos(gstc_phi)")
    sheet_vars.set("gstc_vz", "0")

    # Evaluate the 10 dimensionless interpolated surface parameters at (gstc_theta, gstc_phi)
    for name in PARAMETER_NAMES:
        sheet_vars.set(f"gstc_{name}", f"int_{name}(gstc_theta, gstc_phi)")

    # Two-sided trace averages across the zero-thickness sheet r = a
    for ax in ("x", "y", "z"):
        sheet_vars.set(
            f"gstc_Eav{ax}",
            f"0.5*(up({physics_tag}.E{ax}) + down({physics_tag}.E{ax}))",
        )
        sheet_vars.set(
            f"gstc_Hav{ax}",
            f"0.5*(up({physics_tag}.H{ax}) + down({physics_tag}.H{ax}))",
        )

    # Project averaged fields onto (u = theta_hat, v = phi_hat)
    sheet_vars.set(
        "gstc_Eth",
        "(gstc_ux*gstc_Eavx + gstc_uy*gstc_Eavy + gstc_uz*gstc_Eavz)",
    )
    sheet_vars.set(
        "gstc_Eph",
        "(gstc_vx*gstc_Eavx + gstc_vy*gstc_Eavy + gstc_vz*gstc_Eavz)",
    )
    sheet_vars.set(
        "gstc_Hth",
        "(gstc_ux*gstc_Havx + gstc_uy*gstc_Havy + gstc_uz*gstc_Havz)",
    )
    sheet_vars.set(
        "gstc_Hph",
        "(gstc_vx*gstc_Havx + gstc_vy*gstc_Havy + gstc_vz*gstc_Havz)",
    )

    # Full 10-parameter bianisotropic sheet currents in (theta_hat, phi_hat):
    #   Y_loc = (j/eta0)*[[y1, y3], [y3, y2]],   chi_loc   = [[c1, c2], [c3, c4]]
    #   Z_loc = (j*eta0)*[[z1, z3], [z3, z2]],   gamma_loc = -chi_loc^T = [[-c1, -c3], [-c2, -c4]]
    sheet_vars.set(
        "gstc_Js_th",
        "((j / eta0)*(gstc_y1*gstc_Eth + gstc_y3*gstc_Eph) + (gstc_c1*gstc_Hth + gstc_c2*gstc_Hph))",
    )
    sheet_vars.set(
        "gstc_Js_ph",
        "((j / eta0)*(gstc_y3*gstc_Eth + gstc_y2*gstc_Eph) + (gstc_c3*gstc_Hth + gstc_c4*gstc_Hph))",
    )
    sheet_vars.set(
        "gstc_Ms_th",
        "(-(gstc_c1*gstc_Eth + gstc_c3*gstc_Eph) + (j * eta0)*(gstc_z1*gstc_Hth + gstc_z3*gstc_Hph))",
    )
    sheet_vars.set(
        "gstc_Ms_ph",
        "(-(gstc_c2*gstc_Eth + gstc_c4*gstc_Eph) + (j * eta0)*(gstc_z3*gstc_Hth + gstc_z2*gstc_Hph))",
    )

    # Lift local sheet currents back to 3D Cartesian coordinates:
    for ax in ("x", "y", "z"):
        sheet_vars.set(
            f"gstc_Js_{ax}",
            f"(gstc_u{ax}*gstc_Js_th + gstc_v{ax}*gstc_Js_ph)",
        )
        sheet_vars.set(
            f"gstc_Ms_{ax}",
            f"(gstc_u{ax}*gstc_Ms_th + gstc_v{ax}*gstc_Ms_ph)",
        )

    # -------------------------------------------------------------------------
    # Step 6: Electric & Magnetic Sheet Current Boundary Features on r = a
    # -------------------------------------------------------------------------
    scu1 = phys.create("scu1", "SurfaceCurrent", 2)
    scu1.label("GSTC Electric Sheet Current Density (Js)")
    scu1.selection().named(selections.sheet_boundary_selection)
    scu1.set("Js0", ["gstc_Js_x", "gstc_Js_y", "gstc_Js_z"])

    smcd1 = phys.create("smcd1", "SurfaceMagneticCurrentDensity", 2)
    smcd1.label("GSTC Magnetic Sheet Current Density (Ms)")
    smcd1.selection().named(selections.sheet_boundary_selection)
    smcd1.set("Jms0", ["gstc_Ms_x", "gstc_Ms_y", "gstc_Ms_z"])

    # -------------------------------------------------------------------------
    # Step 7: Stratton-Chu Far-Field Calculation
    # -------------------------------------------------------------------------
    if enable_far_field:
        ffd = phys.create("ffd1", "FarFieldDomain", 3)
        ffd.label("Exterior Homogeneous Air Far-Field Domain")
        ffd.selection().set(JArray(JInt)(selections.far_field_air_domains))
        ffc = ffd.feature("ffc1")
        ffc.label("Exterior Huygens Stratton-Chu Far-Field Surface")
        ffc.selection().named(selections.far_field_surface_selection)

    return physics_tag


def setup_sheet_physics_comsol_54(
    model_java: Any,
    component: Any,
    selections: SheetGeometrySelections,
    *,
    grid_files: Mapping[str, Path],
    discretization: str = "linear",
    dipole_type: str = "electric",
    dipole_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    excitation_cases: Sequence[DipoleExcitationCase] | None = None,
    enable_far_field: bool = True,
) -> str:
    """Configure Case A zero-thickness sheet physics for COMSOL 5.4 (auto-detecting ewfd/emw)."""
    return setup_bianisotropic_sheet_physics(
        model_java=model_java,
        component=component,
        selections=selections,
        grid_files=grid_files,
        version="5.4",
        physics_interface="auto",
        discretization=discretization,
        dipole_type=dipole_type,
        dipole_direction=dipole_direction,
        excitation_cases=excitation_cases,
        enable_far_field=enable_far_field,
    )


def setup_sheet_physics_comsol_61(
    model_java: Any,
    component: Any,
    selections: SheetGeometrySelections,
    *,
    grid_files: Mapping[str, Path],
    discretization: str = "linear",
    dipole_type: str = "electric",
    dipole_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    excitation_cases: Sequence[DipoleExcitationCase] | None = None,
    enable_far_field: bool = True,
) -> str:
    """Configure Case A zero-thickness sheet physics for COMSOL 6.1 (auto-detecting emw/ewfd)."""
    return setup_bianisotropic_sheet_physics(
        model_java=model_java,
        component=component,
        selections=selections,
        grid_files=grid_files,
        version="6.1",
        physics_interface="auto",
        discretization=discretization,
        dipole_type=dipole_type,
        dipole_direction=dipole_direction,
        excitation_cases=excitation_cases,
        enable_far_field=enable_far_field,
    )


# =============================================================================
# Phase 6: Study & Linear Solver Configuration
# =============================================================================

def setup_study_and_solver(
    model_java: Any,
    *,
    study_tag: str = "std1",
    solver_tag: str = "sol1",
    freq_expr: str = "f0",
    linear_solver: str = "pardiso",
    aux_param_name: str | None = None,
    aux_param_values: Sequence[int | float | str] | None = None,
) -> str:
    """Configure frequency-domain study and direct linear solver (Pardiso/MUMPS).

    Uses COMSOL's built-in showAutoSequences to generate complete frequency-domain
    solution variables, then overrides the linear solver of the stationary study step
    to use a robust direct solver (Pardiso by default) suitable for high-contrast
    and bianisotropic wave equations.

    When ``aux_param_name`` and ``aux_param_values`` are supplied (e.g. ``case_idx``
    across ``[1, 2]`` for multi-excitation solves where only the RHS load varies),
    configures an auxiliary sweep inside the frequency study step so MUMPS/Pardiso
    performs a single LU factorization and reuses it across all subsequent cases.

    Args:
        model_java: COMSOL Java model object.
        study_tag: Study identifier (default: 'std1').
        solver_tag: Solver identifier (default: 'sol1').
        freq_expr: Frequency parameter expression (default: 'f0').
        linear_solver: Direct solver engine ('pardiso', 'mumps', or 'spooles').
        aux_param_name: Optional auxiliary sweep parameter name (e.g. 'case_idx').
        aux_param_values: Optional sequence of parameter values for the auxiliary sweep.

    Returns:
        Study tag string.
    """
    std = model_java.study().create(study_tag)
    freq_feat = std.create("freq", "Frequency")
    freq_feat.set("plist", freq_expr)
    if aux_param_name is not None and aux_param_values:
        plist_str = " ".join(str(val) for val in aux_param_values)
        freq_feat.set("useparam", True)
        freq_feat.set("pname_aux", JArray(JString)([aux_param_name]))
        freq_feat.set("plistarr_aux", JArray(JString)([plist_str]))
        freq_feat.set("punit_aux", JArray(JString)([""]))

    # Generate complete solver variable sequence
    std.showAutoSequences("sol")

    # Configure direct linear solver on the stationary feature
    sol = model_java.sol(solver_tag)
    s1 = sol.feature("s1")
    s1.feature("aDef").set("complexfun", True)

    # Create or configure direct linear solver node and ensure Iterative ('i1') is disabled
    s1_tags = [str(t) for t in s1.feature().tags()]
    if "d1" in s1_tags:
        d1 = s1.feature("d1")
    else:
        d1 = s1.create("d1", "Direct")
    d1.active(True)
    d1.set("linsolver", linear_solver)
    try:
        # Keep MUMPS in-core whenever RAM permits to avoid slow disk OOC spillover
        if linear_solver == "mumps":
            d1.set("ooc", "off")
    except Exception:
        pass

    for tag in list(s1.feature().tags()):
        tag_str = str(tag)
        if tag_str.startswith("i"):
            try:
                s1.feature(tag_str).active(False)
            except Exception:
                pass

    for fc_tag in ("fc1", "fcDef"):
        if fc_tag in s1_tags:
            try:
                s1.feature(fc_tag).active(True)
                s1.feature(fc_tag).set("linsolver", "d1")
            except Exception:
                pass

    return study_tag


# =============================================================================
# Phase 7: Post-Processing & Data Extraction
# =============================================================================

def compute_dipole_normalization(
    f0_hz: float,
    er_interior: float = 7.0,
    mur_interior: float = 1.0,
    dipole_type: str = "electric",
    effective_moment_amplitude: complex = 1.0 + 0.0j,
) -> complex:
    """Compute physical dipole normalization scale factor between COMSOL and SIE.

    In the analytical dipole field routines in `sie_mom.excitation.fields`:
      - For ``dipole_type == "electric"``, ``evaluate_dipole_fields`` defines
        electric dipole excitation with p_eff = 1, omitting 1 / (4 * pi * epsilon_1).
        COMSOL's ``ElectricPointDipole`` uses current moment I * dl = 1 A*m,
        so p = (I * dl) / (j * omega), giving:
          unit_scale = -1j * eta_interior / (4.0 * pi * k_interior)
      - For ``dipole_type == "magnetic"``, ``evaluate_dipole_fields`` applies
        duality (E_m = -eta_1 * H_e, H_m = E_e / eta_1). COMSOL's
        ``MagneticPointDipole`` uses magnetic moment m = 1 A*m^2 under exp(+j*omega*t),
        giving:
          unit_scale = +eta_interior / (4.0 * pi)

    Dividing COMSOL's extracted fields (E, H, E_far) by ``unit_scale / effective_moment_amplitude``
    aligns COMSOL's numerical values exactly to the SIE-BEM benchmark fields.
    """
    from scipy.constants import c

    amp = complex(effective_moment_amplitude)
    if amp == 0:
        raise ValueError("effective_moment_amplitude must be nonzero")

    omega = 2.0 * np.pi * f0_hz
    k_interior = omega * np.sqrt(er_interior * mur_interior) / c
    eta_interior = ETA_0 * np.sqrt(mur_interior / er_interior)

    if dipole_type == "electric":
        unit_scale = -1j * eta_interior / (4.0 * np.pi * k_interior)
    elif dipole_type == "magnetic":
        unit_scale = complex(eta_interior / (4.0 * np.pi))
    else:
        raise ValueError(f"Unsupported dipole_type: {dipole_type!r}")

    return unit_scale / amp


def extract_near_field_radial(
    model: Any,
    physics_tag: str,
    *,
    r_min: float = 0.05,
    r_max: float = 2.5,
    num_points: int = 500,
    theta_rad: float = np.pi / 4.0,
    phi_rad: float = 0.0,
    a_radius: float = 2.0,
    tau_m: float = 0.03,
    dipole_scale_factor: complex | None = None,
    solnum: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract physical near-field along a radial line cut from COMSOL solution.

    Evaluates Ex, Ey, Ez, Hx, Hy, Hz using COMSOL's numerical Interp feature,
    applies dipole normalization, and tags the region (1: interior, 3: shell, 2: exterior).

    Returns dictionary matching the columns of sie_physical_radial.csv.
    """
    r_vals = np.linspace(r_min, r_max, num_points)
    if tau_m <= 0.0:
        # Avoid evaluating Interp strictly on the Case A discontinuous slit boundary r = a
        slit_mask = np.abs(r_vals - a_radius) < 1e-7 * a_radius
        if np.any(slit_mask):
            r_vals = r_vals.copy()
            r_vals[slit_mask] = a_radius + 1e-6 * a_radius
    sin_th = np.sin(theta_rad)
    cos_th = np.cos(theta_rad)
    cos_ph = np.cos(phi_rad)
    sin_ph = np.sin(phi_rad)

    x_vals = r_vals * sin_th * cos_ph
    y_vals = r_vals * sin_th * sin_ph
    z_vals = r_vals * cos_th

    coords_array = np.vstack([x_vals, y_vals, z_vals])  # (3, N)

    java = model.java
    num = java.result().numerical()
    interp_tag = "interp_rad"
    if interp_tag in list(num.tags()):
        num.remove(interp_tag)

    interp = num.create(interp_tag, "Interp")
    interp.set("data", "dset1")
    if solnum is not None:
        interp.set("solnum", JArray(JInt)([int(solnum)]))
    interp.selection().geom("geom1", JInt(3))
    interp.selection().all()
    interp.setInterpolationCoordinates(JArray(JDouble, 2)(coords_array.tolist()))

    field_names = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    interp.set("expr", [f"{physics_tag}.{f}" for f in field_names])

    real_data = np.asarray(interp.getData(), dtype=float).reshape(len(field_names), -1)
    imag_data = np.asarray(interp.getImagData(), dtype=float).reshape(len(field_names), -1)
    num.remove(interp_tag)

    fields_raw = real_data + 1j * imag_data  # (6, N)

    # Dipole scaling
    if dipole_scale_factor is not None and abs(dipole_scale_factor) > 0:
        fields_norm = fields_raw / dipole_scale_factor
    else:
        fields_norm = fields_raw

    # Determine region code: 1: interior (r < a), 3: shell (a <= r <= a + tau), 2: exterior (r > a + tau)
    regions = np.zeros(num_points, dtype=int)
    for idx, r in enumerate(r_vals):
        if r < a_radius:
            regions[idx] = 1
        elif r > a_radius + tau_m:
            regions[idx] = 2
        else:
            regions[idx] = 3

    return {
        "r_m": r_vals,
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "region": regions,
        "Ex": fields_norm[0],
        "Ey": fields_norm[1],
        "Ez": fields_norm[2],
        "Hx": fields_norm[3],
        "Hy": fields_norm[4],
        "Hz": fields_norm[5],
    }


def extract_near_field_zx(
    model: Any,
    physics_tag: str,
    *,
    x_min: float = -2.5,
    x_max: float = 2.5,
    z_min: float = -2.5,
    z_max: float = 2.5,
    nx: int = 151,
    nz: int = 151,
    y_val: float = 0.0,
    a_radius: float = 2.0,
    tau_m: float = 0.03,
    dipole_scale_factor: complex | None = None,
    batch_size: int = 10000,
    solnum: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract physical near-field on a 2D grid in the z-x plane (y = y_val).

    Evaluates Ex, Ey, Ez, Hx, Hy, Hz using COMSOL's numerical Interp feature
    in memory-friendly coordinate batches, applies dipole normalization, and tags regions.

    Returns dictionary matching the columns of sie_physical_zx.csv.
    """
    x_line = np.linspace(x_min, x_max, nx)
    z_line = np.linspace(z_min, z_max, nz)
    X, Z = np.meshgrid(x_line, z_line, indexing="ij")
    x_flat = X.ravel()
    z_flat = Z.ravel()
    y_flat = np.full_like(x_flat, y_val)
    interp_x = x_flat
    interp_y = y_flat
    interp_z = z_flat
    if tau_m <= 0.0:
        r_3d = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
        slit_mask = np.abs(r_3d - a_radius) < 1e-7 * a_radius
        if np.any(slit_mask):
            scale_out = (a_radius + 1e-6 * a_radius) / np.maximum(r_3d[slit_mask], 1e-30)
            interp_x = x_flat.copy()
            interp_y = y_flat.copy()
            interp_z = z_flat.copy()
            interp_x[slit_mask] *= scale_out
            interp_y[slit_mask] *= scale_out
            interp_z[slit_mask] *= scale_out
    total_pts = len(x_flat)

    java = model.java
    num = java.result().numerical()
    interp_tag = "interp_zx"
    if interp_tag in list(num.tags()):
        num.remove(interp_tag)

    interp = num.create(interp_tag, "Interp")
    interp.set("data", "dset1")
    if solnum is not None:
        interp.set("solnum", JArray(JInt)([int(solnum)]))
    interp.selection().geom("geom1", JInt(3))
    interp.selection().all()

    field_names = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    interp.set("expr", [f"{physics_tag}.{f}" for f in field_names])

    all_fields = np.zeros((6, total_pts), dtype=complex)

    # Batch evaluation to avoid Java heap issues on large 2D grids
    for start_idx in range(0, total_pts, batch_size):
        end_idx = min(start_idx + batch_size, total_pts)
        batch_coords = np.vstack(
            [
                interp_x[start_idx:end_idx],
                interp_y[start_idx:end_idx],
                interp_z[start_idx:end_idx],
            ]
        )
        interp.setInterpolationCoordinates(JArray(JDouble, 2)(batch_coords.tolist()))
        re_batch = np.asarray(interp.getData(), dtype=float).reshape(len(field_names), -1)
        im_batch = np.asarray(interp.getImagData(), dtype=float).reshape(len(field_names), -1)
        all_fields[:, start_idx:end_idx] = re_batch + 1j * im_batch

    num.remove(interp_tag)

    # Dipole scaling
    if dipole_scale_factor is not None and abs(dipole_scale_factor) > 0:
        all_fields /= dipole_scale_factor

    # Determine regions
    r_flat = np.hypot(x_flat, z_flat)
    regions = np.zeros(total_pts, dtype=int)
    regions[r_flat < a_radius] = 1
    if tau_m <= 0.0:
        regions[r_flat >= a_radius] = 2
    else:
        regions[r_flat > a_radius + tau_m] = 2
        regions[(r_flat >= a_radius) & (r_flat <= a_radius + tau_m)] = 3

    return {
        "x": x_flat,
        "y": y_flat,
        "z": z_flat,
        "region": regions,
        "Ex": all_fields[0],
        "Ey": all_fields[1],
        "Ez": all_fields[2],
        "Hx": all_fields[3],
        "Hy": all_fields[4],
        "Hz": all_fields[5],
        "nx": nx,
        "nz": nz,
    }


def extract_far_field_cuts(
    model: Any,
    physics_tag: str,
    *,
    num_points: int = 361,
    dipole_scale_factor: complex | None = None,
    p_rad_grid_theta: int = 45,
    p_rad_grid_phi: int = 90,
    solnum: int | None = None,
) -> dict[str, Any]:
    """Extract Stratton-Chu principal far-field cuts and directivity matching plot_far_field.py.

    Evaluates E-plane (z-x plane), H-plane (y-z plane), and XY-plane (x-y plane) cuts
    at num_points uniform angles in [0, 2*pi].
    Computes total radiated power P_rad via 2D spherical quadrature of far-field intensity,
    then computes co-pol, cross-pol, and directivity in dBi.

    Returns dictionary matching compute_principal_cuts in plot_far_field.py.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=True)
    angles_deg = np.degrees(angles)

    # 1. Directions and polarizations
    # E-plane (z-x): r_hat = [sin(alpha), 0, cos(alpha)]
    r_hat_e = np.column_stack([np.sin(angles), np.zeros_like(angles), np.cos(angles)])
    co_hat_e = np.column_stack([np.cos(angles), np.zeros_like(angles), -np.sin(angles)])
    cr_hat_e = np.column_stack([np.zeros_like(angles), np.ones_like(angles), np.zeros_like(angles)])

    # H-plane (y-z): r_hat = [0, sin(alpha), cos(alpha)]
    r_hat_h = np.column_stack([np.zeros_like(angles), np.sin(angles), np.cos(angles)])
    co_hat_h = np.column_stack([np.zeros_like(angles), np.cos(angles), -np.sin(angles)])
    cr_hat_h = np.column_stack([-np.ones_like(angles), np.zeros_like(angles), np.zeros_like(angles)])

    # XY-plane (x-y): r_hat = [cos(alpha), sin(alpha), 0]
    r_hat_xy = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    co_hat_xy = np.column_stack([-np.sin(angles), np.cos(angles), np.zeros_like(angles)])
    cr_hat_xy = np.column_stack([np.zeros_like(angles), np.zeros_like(angles), np.ones_like(angles)])

    def _eval_efar_directions(r_hat_array: np.ndarray) -> np.ndarray:
        """Evaluate Efarx, Efary, Efarz for an array of (N, 3) unit directions."""
        expr_list = []
        for r in r_hat_array:
            expr_list.extend([
                f"Efarx({r[0]:.7f}, {r[1]:.7f}, {r[2]:.7f})",
                f"Efary({r[0]:.7f}, {r[1]:.7f}, {r[2]:.7f})",
                f"Efarz({r[0]:.7f}, {r[1]:.7f}, {r[2]:.7f})",
            ])
        eval_kwargs = {"inner": [int(solnum)]} if solnum is not None else {}
        raw_vals = model.evaluate(expr_list, **eval_kwargs)
        efar = np.array(raw_vals, dtype=complex).reshape(len(r_hat_array), 3)
        if dipole_scale_factor is not None and abs(dipole_scale_factor) > 0:
            efar /= dipole_scale_factor
        return efar

    # 2. Evaluate principal cuts
    efar_e = _eval_efar_directions(r_hat_e)
    efar_h = _eval_efar_directions(r_hat_h)
    efar_xy = _eval_efar_directions(r_hat_xy)

    # 3. Compute P_rad by integrating far-field intensity over spherical grid
    th_vals = np.linspace(0.0, np.pi, p_rad_grid_theta)
    ph_vals = np.linspace(0.0, 2.0 * np.pi, p_rad_grid_phi, endpoint=False)
    th_g, ph_g = np.meshgrid(th_vals, ph_vals, indexing="ij")
    r_hat_sph = np.column_stack([
        np.sin(th_g).ravel() * np.cos(ph_g).ravel(),
        np.sin(th_g).ravel() * np.sin(ph_g).ravel(),
        np.cos(th_g).ravel(),
    ])
    efar_sph = _eval_efar_directions(r_hat_sph)
    efar_mag2_grid = np.sum(np.abs(efar_sph) ** 2, axis=1).reshape(p_rad_grid_theta, p_rad_grid_phi)
    u_grid = efar_mag2_grid / (2.0 * ETA_0)

    dth = th_vals[1] - th_vals[0]
    dph = ph_vals[1] - ph_vals[0]
    w_th = np.ones(p_rad_grid_theta)
    w_th[0] = 0.5
    w_th[-1] = 0.5
    integrand = u_grid * np.sin(th_vals)[:, None]
    p_rad = float(np.sum(integrand * w_th[:, None]) * dth * dph)

    # 4. Helper for cut metrics
    def _build_cut_dict(efar_cut: np.ndarray, co_vecs: np.ndarray, cr_vecs: np.ndarray) -> dict[str, Any]:
        f_co = np.sum(efar_cut * co_vecs, axis=1)
        f_cr = np.sum(efar_cut * cr_vecs, axis=1)
        co_pol = np.abs(f_co)
        cross_pol = np.abs(f_cr)
        total_field = np.sqrt(co_pol**2 + cross_pol**2)
        u_cut = (total_field**2) / (2.0 * ETA_0)
        d_lin = (4.0 * np.pi * u_cut) / max(p_rad, 1e-12)
        d_dbi = 10.0 * np.log10(np.maximum(d_lin, 1e-12))
        return {
            "co_pol": co_pol,
            "cross_pol": cross_pol,
            "total_field": total_field,
            "directivity_dbi": d_dbi,
        }

    cut_e = _build_cut_dict(efar_e, co_hat_e, cr_hat_e)
    cut_h = _build_cut_dict(efar_h, co_hat_h, cr_hat_h)
    cut_xy = _build_cut_dict(efar_xy, co_hat_xy, cr_hat_xy)

    return {
        "angles_deg": angles_deg,
        "p_rad_w": p_rad,
        "E_plane": cut_e,
        "H_plane": cut_h,
        "XY_plane": cut_xy,
    }


def save_physical_radial_csv(data: dict[str, np.ndarray], output_path: Path) -> None:
    """Save physical radial near-field data to CSV matching sie_physical_radial.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_pts = len(data["x"])
    rows = []
    for i in range(n_pts):
        rows.append([
            i,
            data["x"][i],
            data["y"][i],
            data["z"][i],
            data["region"][i],
            data["Ex"][i].real,
            data["Ex"][i].imag,
            data["Ey"][i].real,
            data["Ey"][i].imag,
            data["Ez"][i].real,
            data["Ez"][i].imag,
            data["Hx"][i].real,
            data["Hx"][i].imag,
            data["Hy"][i].real,
            data["Hy"][i].imag,
            data["Hz"][i].real,
            data["Hz"][i].imag,
        ])
    header = "point_index,x,y,z,region,Ex_real,Ex_imag,Ey_real,Ey_imag,Ez_real,Ez_imag,Hx_real,Hx_imag,Hy_real,Hy_imag,Hz_real,Hz_imag"
    np.savetxt(output_path, rows, delimiter=",", header=header, comments="", fmt="%.16e")


def save_physical_zx_csv(data: dict[str, np.ndarray], output_path: Path) -> None:
    """Save physical zx near-field data to CSV matching sie_physical_zx.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_pts = len(data["x"])
    rows = []
    for i in range(n_pts):
        rows.append([
            i,
            data["x"][i],
            data["y"][i],
            data["z"][i],
            data["region"][i],
            data["Ex"][i].real,
            data["Ex"][i].imag,
            data["Ey"][i].real,
            data["Ey"][i].imag,
            data["Ez"][i].real,
            data["Ez"][i].imag,
            data["Hx"][i].real,
            data["Hx"][i].imag,
            data["Hy"][i].real,
            data["Hy"][i].imag,
            data["Hz"][i].real,
            data["Hz"][i].imag,
        ])
    header = "point_index,x,y,z,region,Ex_real,Ex_imag,Ey_real,Ey_imag,Ez_real,Ez_imag,Hx_real,Hx_imag,Hy_real,Hy_imag,Hz_real,Hz_imag"
    np.savetxt(output_path, rows, delimiter=",", header=header, comments="", fmt="%.16e")


def save_far_field_cuts_csv(cuts: dict[str, Any], output_path: Path) -> None:
    """Save far-field principal cuts to CSV matching plot_far_field.py:export_cuts_csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    angles_deg = cuts["angles_deg"]
    e_cut = cuts["E_plane"]
    h_cut = cuts["H_plane"]
    xy_cut = cuts["XY_plane"]

    rows = []
    for idx, deg in enumerate(angles_deg):
        rows.append([
            deg,
            e_cut["co_pol"][idx],
            e_cut["cross_pol"][idx],
            e_cut["directivity_dbi"][idx],
            h_cut["co_pol"][idx],
            h_cut["cross_pol"][idx],
            h_cut["directivity_dbi"][idx],
            xy_cut["co_pol"][idx],
            xy_cut["cross_pol"][idx],
            xy_cut["directivity_dbi"][idx],
        ])
    header = (
        "angle_deg,E_plane_copol_V,E_plane_xpol_V,E_plane_directivity_dbi,"
        "H_plane_copol_V,H_plane_xpol_V,H_plane_directivity_dbi,"
        "XY_plane_copol_V,XY_plane_xpol_V,XY_plane_directivity_dbi"
    )
    np.savetxt(output_path, rows, delimiter=",", header=header, comments="", fmt="%.6e")


