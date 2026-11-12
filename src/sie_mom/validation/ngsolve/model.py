"""Two-trace frequency-domain Maxwell FEM with a general GSTC interface.

Coordinates are X=k0*x; fields are e=E/k0**3, h=eta0*H/k0**3, with exp(+i*omega*t).
The caller supplies the interface frame and two-sided admittance coefficient.
Configuration objects expose physical media via .case and dipoles via .fields;
this backend does not load application configurations or choose geometry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import ngsolve as ng
import numpy as np
from scipy.constants import epsilon_0, mu_0

ETA0 = float(np.sqrt(mu_0 / epsilon_0))
R = np.array([[0.0, -1.0], [1.0, 0.0]])


def normalized_map(physical_map):
    """[eta0*Js; Ms] = L [E; eta0*H], with physical input [Y,chi;gamma,Z]."""
    return (
        np.asarray(physical_map)
        * np.array([ETA0, ETA0, 1, 1])[:, None]
        * np.array([1, 1, 1 / ETA0, 1 / ETA0])[None, :]
    )


def sheet_admittance(surface_map):
    """Return B with [h_minus;h_plus]=B[e_minus;e_plus].

    R*Delta(h)=Y*e_av+chi*h_av; -R*Delta(e)=gamma*e_av+Z*h_av.
    This formulation requires invertible Z; never regularize it silently.
    """
    surface_map = np.asarray(surface_map, dtype=complex)
    if surface_map.shape != (4, 4) or not np.isfinite(surface_map).all():
        raise ValueError("surface map must be a finite complex 4x4 matrix")
    y, chi = surface_map[:2, :2], surface_map[:2, 2:]
    gamma, z = surface_map[2:, :2], surface_map[2:, 2:]
    if np.linalg.cond(z) > 1e10:
        raise ValueError("Z is singular/ill-conditioned; this E-only formulation is unsuitable")
    average = np.hstack((np.eye(2), np.eye(2))) / 2
    jump = np.hstack((-np.eye(2), np.eye(2)))
    h_average = np.linalg.solve(z, -R @ jump - gamma @ average)
    h_jump = -R @ (y @ average + chi @ h_average)
    return np.vstack((h_average - h_jump / 2, h_average + h_jump / 2))


def k0_of(config):
    return 2 * np.pi * config.case.frequency_hz * np.sqrt(epsilon_0 * mu_0)


def direct_solver(requested="auto"):
    """Choose an available general sparse LU, with no silent fallback on failure."""
    runtime_pardiso = getattr(ng, "_mkl_pardiso", None)
    available = {
        "pardiso": (
            ng.config.USE_MKL
            or ng.config.USE_PARDISO
            or getattr(runtime_pardiso, "_pardiso", None) is not None
        ),
        "umfpack": ng.config.USE_UMFPACK,
    }
    if requested == "auto":
        for name, enabled in available.items():
            if enabled:
                return name
        raise RuntimeError("NGSolve was built without PARDISO or UMFPACK")
    if not available.get(requested, False):
        raise ValueError(f"{requested} is unavailable in this NGSolve build")
    return requested


def medium(config, region):
    return (
        getattr(config.case, f"relative_permittivity_{region}"),
        getattr(config.case, f"relative_permeability_{region}"),
    )


def dipole_cf(config, field_id, region):
    """Analytic fields with exactly the repository's effective-moment convention."""
    definition = config.fields[field_id]
    if getattr(definition, "dipole_type", None) not in ("electric", "magnetic"):
        raise ValueError("this example supports dipole sources/targets only")
    k0 = k0_of(config)
    displacement = ng.CF(
        tuple(p - k0 * s for p, s in zip((ng.x, ng.y, ng.z), definition.source_position))
    )
    radius = ng.sqrt(displacement * displacement)
    n = displacement / radius
    moment = ng.CF(tuple(definition.effective_moment))
    eps, mu = medium(config, region)
    k = np.sqrt(eps * mu)
    eta_rel = np.sqrt(mu / eps)
    phase = ng.exp(-1j * k * radius)
    e = phase * (
        k**2 / radius * (moment - n * (n * moment))
        + (1 / radius**3 + 1j * k / radius**2) * (3 * n * (n * moment) - moment)
    )
    h = (k**2 / radius - 1j * k / radius**2) * phase / eta_rel * ng.Cross(n, moment)
    if definition.dipole_type == "magnetic":
        return -eta_rel * h, e / eta_rel
    return e, h


def incident_cf(config, case_id):
    cases = {case.id: case for case in config.simulation.cases}
    if case_id not in cases:
        raise ValueError(f"unknown case {case_id!r}; choose from {list(cases)}")
    e, h = ng.CF((0j, 0j, 0j)), ng.CF((0j, 0j, 0j))
    for excitation in cases[case_id].excitations:
        if excitation.region != "interior":
            raise ValueError("this example supports interior excitations only")
        definition = config.fields[excitation.field.field_id]
        if getattr(definition, "dipole_type", None) not in ("electric", "magnetic"):
            raise ValueError("FEM excitation requires an electric or magnetic dipole")
        ei, hi = dipole_cf(config, excitation.field.field_id, "interior")
        e, h = e + ei, h + hi
    return e, h


def local(vector, frame):
    return ng.CF(tuple(vector * axis for axis in frame))


def rotate(vector):
    return ng.CF((-vector[1], vector[0]))


def split(vector):
    return ng.CF((vector[0], vector[1])), ng.CF((vector[2], vector[3]))


@dataclass
class System:
    mesh: object
    space: object
    matrix: object
    b: object
    frame: tuple
    test: tuple
    config: object
    assembly_seconds: float

    def fields_from_e(self, e_minus, e_plus):
        em, ep = local(e_minus, self.frame), local(e_plus, self.frame)
        return split(self.b * ng.CF((em[0], em[1], ep[0], ep[1])))

    def rhs(self, e_incident, h_incident, extra_h=None):
        """Optional additive sheet h traces are for manufactured-solution tests."""
        hm, hp = self.fields_from_e(e_incident, ng.CF((0, 0, 0)))
        hm = hm - local(h_incident, self.frame)
        if extra_h is not None:
            hm, hp = hm + extra_h[0], hp + extra_h[1]
        vm, vp = (local(v.Trace(), self.frame) for v in self.test)
        rhs = ng.LinearForm(self.space)
        rhs += (1j * (rotate(hm) * vm - rotate(hp) * vp)) * ng.ds("sheet", bonus_intorder=4)
        rhs.Assemble()
        return rhs


def make_space(mesh, order=2):
    """Reconstruct the same two-trace space for assembly or checkpoint loading."""
    # Product of subdomain spaces duplicates the sheet's tangential E unknowns.
    core = ng.HCurl(mesh, order=order, complex=True, definedon="core")
    exterior = ng.HCurl(mesh, order=order, complex=True, definedon="air|pml", dirichlet="outer")
    return ng.FESpace([core, exterior])


def assemble(config, mesh, *, frame, admittance, order=2):
    start = time.perf_counter()
    space = make_space(mesh, order)
    trial, test = space.TnT()
    a = ng.BilinearForm(space, symmetric=False)
    for u, v, region, name in zip(trial, test, ("interior", "exterior"), ("core", "air|pml")):
        eps, mu = medium(config, region)
        a += (ng.curl(u) * ng.curl(v) / mu - eps * u * v) * ng.dx(name)
    system = System(mesh, space, a, admittance, frame, test, config, 0)
    hm, hp = system.fields_from_e(*(u.Trace() for u in trial))
    vm, vp = (local(v.Trace(), frame) for v in test)
    a += (-1j * (rotate(hm) * vm - rotate(hp) * vp)) * ng.ds("sheet", bonus_intorder=4)
    a.Assemble()
    system.assembly_seconds = time.perf_counter() - start
    return system


def total_fields(system, solution, e_incident, h_incident):
    inner, outer = solution.components
    mu_inner = medium(system.config, "interior")[1]
    mu_outer = medium(system.config, "exterior")[1]
    return (
        (inner + e_incident, 1j / mu_inner * ng.curl(inner) + h_incident),
        (outer, 1j / mu_outer * ng.curl(outer)),
    )


def evaluate(cf, mesh, points):
    points = np.asarray(points)
    values = np.asarray(cf(mesh(*points.T)))
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite sampled fields")
    return values


def residual(system, solution, rhs):
    vector = rhs.vec.CreateVector()
    vector.data = system.matrix.mat * solution.vec - rhs.vec
    free = np.array(list(system.space.FreeDofs()), dtype=bool)
    numerator = np.linalg.norm(vector.FV().NumPy()[free])
    denominator = np.linalg.norm(rhs.vec.FV().NumPy()[free])
    return float(numerator / max(denominator, np.finfo(float).tiny))
