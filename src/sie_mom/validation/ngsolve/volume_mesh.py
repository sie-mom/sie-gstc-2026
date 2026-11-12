"""Volume meshing for independent NGSolve validation.

The analytic sphere path preserves the original benchmark. The shared-interface
path tetrahedralizes around an existing, unmodified planar triangle mesh.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict

import ngsolve as ng
import numpy as np
from netgen.meshing import Element2D, FaceDescriptor, Mesh, MeshingStep, MeshPoint
from netgen.meshing import Pnt as MeshPnt
from netgen.occ import Glue, OCCGeometry, Pnt, Sphere

from sie_mom.mesh.obj_mesh import validate_triangle_mesh

from .model import k0_of, medium
from .tensor_transfer import match_interface


def make_sphere_mesh(
    config,
    *,
    ppw=3.0,
    curve_order=3,
    air_lambda=0.95,
    pml_lambda=0.25,
    pml_strength=3.0,
    interior_ppw=None,
):
    """New curved tetrahedral mesh; radial PML, with negative imaginary stretch."""
    k0 = k0_of(config)
    radius = k0 * config.case.radius_m
    air_radius = 2 * np.pi * air_lambda
    if air_radius <= radius:
        raise ValueError("air extent must be greater than the sheet radius")
    interior_ppw = ppw if interior_ppw is None else interior_ppw
    if (
        not np.isfinite([ppw, interior_ppw, air_lambda, pml_lambda, pml_strength]).all()
        or ppw <= 0
        or interior_ppw <= 0
        or curve_order < 1
        or pml_lambda < 0
        or pml_strength <= 0
    ):
        raise ValueError("invalid mesh/PML settings")
    h_inner = 2 * np.pi / (interior_ppw * np.sqrt(np.prod(medium(config, "interior"))))
    h_outer = 2 * np.pi / (ppw * np.sqrt(np.prod(medium(config, "exterior"))))
    core = Sphere(Pnt(0, 0, 0), radius)
    core.faces.name = "sheet"
    core.faces.maxh = min(h_inner, h_outer)
    core.mat("core")
    core.maxh = h_inner
    air_ball = Sphere(Pnt(0, 0, 0), air_radius)
    air_ball.faces.name = "pml_start" if pml_lambda else "outer"
    air = (air_ball - core).mat("air")
    air.maxh = h_outer
    pieces = [core, air]
    if pml_lambda:
        outer = Sphere(Pnt(0, 0, 0), air_radius + 2 * np.pi * pml_lambda)
        outer.faces.name = "outer"
        layer = (outer - air_ball).mat("pml")
        layer.maxh = h_outer
        pieces.append(layer)
    mesh = ng.Mesh(OCCGeometry(Glue(pieces)).GenerateMesh(maxh=h_outer))
    mesh.Curve(curve_order)
    if pml_lambda:
        mesh.SetPML(
            ng.pml.Radial(origin=(0, 0, 0), rad=air_radius, alpha=-1j * pml_strength), "pml"
        )
    return mesh


def _points(mesh):
    return np.asarray([tuple(p.p) for p in mesh.Points()])


def _insert_surface(destination, vertices, triangles, *, bc, inner, outer, name):
    descriptor = destination.Add(FaceDescriptor(bc=bc, domin=inner, domout=outer, surfnr=bc))
    destination.SetBCName(bc - 1, name)
    ids = [destination.Add(MeshPoint(MeshPnt(*p))) for p in vertices]
    for face in triangles:
        destination.Add(Element2D(descriptor, [ids[int(v)] for v in face]))
    return descriptor


def make_shared_mesh(
    config, source, *, ppw=4, interior_ppw=None, air_lambda=1.5, pml_lambda=0.25, pml_strength=3
):
    """Fill three domains around the unchanged planar SIE interface.

    The spherical air/PML enclosing surfaces are independently meshed. Interface
    triangles are imported without surface optimization, refinement or curving.
    The returned diagnostics verify two-sided adjacency and polyhedral volume.
    Size controls are regional wavelengths, not estimates of FEM accuracy.
    """
    start = time.perf_counter()
    interior_ppw = ppw if interior_ppw is None else interior_ppw
    values = [ppw, interior_ppw, air_lambda, pml_strength]
    if not np.isfinite(values + [pml_lambda]).all() or min(values) <= 0 or pml_lambda < 0:
        raise ValueError("mesh/PML settings must be positive; PML thickness may be zero")
    report = validate_triangle_mesh(source)
    report.require_valid()
    k0 = k0_of(config)
    xyz = source.vertices * k0
    air_radius = 2 * np.pi * air_lambda
    if air_radius <= np.linalg.norm(xyz, axis=1).max():
        raise ValueError("air radius must enclose every SIE interface vertex")
    h_inner = 2 * np.pi / (interior_ppw * np.sqrt(np.prod(medium(config, "interior"))))
    h_outer = 2 * np.pi / (ppw * np.sqrt(np.prod(medium(config, "exterior"))))
    combined = Mesh(dim=3)
    _insert_surface(combined, xyz, source.faces, bc=1, inner=1, outer=2, name="sheet")
    surfaces = [(2, air_radius, 2, 0, "outer")]
    if pml_lambda:
        surfaces = [
            (2, air_radius, 2, 3, "pml_start"),
            (3, air_radius + 2 * np.pi * pml_lambda, 3, 0, "outer"),
        ]
    for bc, radius, inner, outer, name in surfaces:
        surface = OCCGeometry(Sphere(Pnt(0, 0, 0), radius)).GenerateMesh(
            maxh=h_outer, perfstepsend=MeshingStep.MESHSURFACE
        )
        triangles = np.asarray([[v.nr - 1 for v in e.vertices] for e in surface.Elements2D()])
        _insert_surface(
            combined, _points(surface), triangles, bc=bc, inner=inner, outer=outer, name=name
        )
    materials = ("core", "air", "pml") if pml_lambda else ("core", "air")
    for i, name in enumerate(materials, 1):
        combined.SetMaterial(i, name)
    combined.SetMaxHDomain([h_inner] + [h_outer] * (len(materials) - 1))
    combined.CalcLocalH(0.3, 1)
    combined.GenerateVolumeMesh(maxh=h_outer, grading=0.3, optsteps3d=3)
    mesh = ng.Mesh(combined)
    diagnostics = _check_shared_mesh(mesh, source, k0, air_lambda, pml_lambda)
    if pml_lambda:
        mesh.SetPML(
            ng.pml.Radial(origin=(0, 0, 0), rad=air_radius, alpha=-1j * pml_strength), "pml"
        )
    diagnostics["seconds"] = time.perf_counter() - start
    return mesh, diagnostics


def _check_shared_mesh(mesh, source, k0, air_lambda, pml_lambda):
    """Check the imported interface, domain adjacency and enclosing surfaces."""
    report = validate_triangle_mesh(source)
    report.require_valid()
    combined = mesh.ngmesh
    expected_materials = ("core", "air", "pml") if pml_lambda else ("core", "air")
    if mesh.GetMaterials() != expected_materials:
        raise ValueError("saved mesh materials disagree with core/air/PML settings")
    _, _, error = match_interface(mesh, source, k0)
    all_xyz = _points(combined)
    sheet_faces = {
        tuple(sorted(v.nr + 1 for v in e.vertices))
        for e in mesh.Elements(ng.BND)
        if e.mat == "sheet"
    }
    adjacent = defaultdict(list)
    counts = Counter()
    core_volume = 0.0
    for element in combined.Elements3D():
        vertices = tuple(v.nr for v in element.vertices)
        if len(vertices) != 4:
            raise ValueError("shared-interface validation expects tetrahedral volume elements")
        counts[element.index] += 1
        a, b, c, d = all_xyz[np.asarray(vertices) - 1]
        volume = abs(np.dot(b - a, np.cross(c - a, d - a))) / 6
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("volume mesh contains a degenerate tetrahedron")
        if element.index == 1:
            core_volume += volume
        for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key = tuple(sorted((vertices[i], vertices[j], vertices[k])))
            if key in sheet_faces:
                adjacent[key].append(element.index)
    if set(adjacent) != sheet_faces or any(
        sorted(domains) != [1, 2] for domains in adjacent.values()
    ):
        raise ValueError(
            "every GSTC triangle must separate exactly one core and one air tetrahedron"
        )
    volume_error = abs(core_volume / k0**3 / report.signed_volume - 1)
    if volume_error > 1e-10:
        raise ValueError("core tetrahedra do not preserve the SIE polyhedral volume")
    radii = {"outer": 2 * np.pi * (air_lambda + pml_lambda)}
    if pml_lambda:
        radii["pml_start"] = 2 * np.pi * air_lambda
    shell_counts = Counter()
    for element in mesh.Elements(ng.BND):
        if element.mat in radii:
            shell_counts[element.mat] += 1
            radius = radii[element.mat]
            norms = [np.linalg.norm(mesh[v].point) for v in element.vertices]
            if not np.allclose(norms, radius, rtol=1e-12, atol=1e-12):
                raise ValueError("saved mesh shell radii disagree with air/PML settings")
    if set(shell_counts) != set(radii) or any(
        counts[i] == 0 for i in range(1, len(expected_materials) + 1)
    ):
        raise ValueError("mesh is missing a required material or enclosing surface")
    return {
        "source_face_count": len(source.faces),
        "interface_vertex_error_m": error,
        "all_interface_faces_separate_core_and_air": True,
        "core_volume_relative_error": volume_error,
        "tetrahedra": {name: counts[i] for i, name in enumerate(mesh.GetMaterials(), 1)},
        "outer_surface_triangles": dict(shell_counts),
        "curve_order": 1,
    }


def load_shared_mesh(config, source, path, *, air_lambda, pml_lambda, pml_strength):
    """Reuse saved tetrahedra, recheck geometry and restore the radial PML.

    Netgen .vol files store the mesh, not NGSolve's PML transformation. Reapply
    the original negative imaginary stretch before assembling the new space.
    """
    start = time.perf_counter()
    if (
        not np.isfinite([air_lambda, pml_lambda, pml_strength]).all()
        or air_lambda <= 0
        or pml_lambda < 0
        or pml_strength <= 0
    ):
        raise ValueError("invalid saved mesh/PML settings")
    mesh = ng.Mesh(str(path))
    diagnostics = _check_shared_mesh(mesh, source, k0_of(config), air_lambda, pml_lambda)
    if pml_lambda:
        mesh.SetPML(
            ng.pml.Radial(origin=(0, 0, 0), rad=2 * np.pi * air_lambda, alpha=-1j * pml_strength),
            "pml",
        )
    diagnostics["seconds"] = time.perf_counter() - start
    return mesh, diagnostics
