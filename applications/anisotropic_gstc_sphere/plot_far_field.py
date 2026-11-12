"""Compute and plot far-field radiation patterns and directivity from SIE currents.

Loads the solved RWG surface currents (J_ext, M_ext) on the outer sphere boundary,
computes the Schelkunoff radiation integrals (N, L), and evaluates the 3D
directivity D(theta, phi) and principal far-field cuts (E-plane, H-plane, XY-plane).
Overlays the exact target dipole far-field pattern for direct synthesis validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sie_mom.mesh.obj_mesh import TriangleMesh, load_triangle_obj

if __package__:
    from .prescribed_fields import evaluate_prescribed_fields
    from .workflow import (
        FieldReference,
        WorkflowConfiguration,
        load_workflow_configuration,
    )
else:
    from prescribed_fields import evaluate_prescribed_fields
    from workflow import (
        FieldReference,
        WorkflowConfiguration,
        load_workflow_configuration,
    )


SPEED_OF_LIGHT = 299792458.0
VACUUM_PERMITTIVITY = 8.8541878188e-12
VACUUM_PERMEABILITY = 1.25663706127e-6
VACUUM_IMPEDANCE = np.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)


def _build_edge_mapping(mesh: TriangleMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct deterministic edge indexing and geometry matching ObjMeshLoader.

    Returns:
        faces_edges: (N_faces, 3) int array of edge indices.
        signed_lengths: (N_faces, 3) float array of signed edge lengths.
        free_vertices: (N_faces, 3, 3) float array of free vertex positions.
    """
    faces = mesh.faces
    vertices = mesh.vertices
    edge_map: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    faces_edges = np.zeros((len(faces), 3), dtype=int)

    for f_idx, face in enumerate(faces):
        for le in range(3):
            v1, v2 = int(face[le]), int(face[(le + 1) % 3])
            edge_key = (min(v1, v2), max(v1, v2))
            if edge_key not in edge_map:
                edge_map[edge_key] = len(edges)
                edges.append(edge_key)
            faces_edges[f_idx, le] = edge_map[edge_key]

    v_a_idx = (0, 1, 2)
    v_b_idx = (1, 2, 0)
    v_free_idx = (2, 0, 1)

    free_vertices = np.zeros((len(faces), 3, 3), dtype=np.float64)
    signed_lengths = np.zeros((len(faces), 3), dtype=np.float64)

    for le in range(3):
        ia, ib, ifree = v_a_idx[le], v_b_idx[le], v_free_idx[le]
        va = vertices[faces[:, ia]]
        vb = vertices[faces[:, ib]]
        free_vertices[:, le, :] = vertices[faces[:, ifree]]
        length = np.linalg.norm(vb - va, axis=1)
        sign = np.where(faces[:, ia] > faces[:, ib], 1.0, -1.0)
        signed_lengths[:, le] = sign * length

    return faces_edges, signed_lengths, free_vertices


def _dunavant_quadrature(rule: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return reference triangle quadrature nodes (xi, eta) and weights summing to 1."""
    if rule <= 3:
        # 3-point symmetric rule (degree 2)
        nodes = np.array([[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]])
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        return nodes, weights

    # 7-point symmetric Dunavant rule (degree 5)
    a1 = (6.0 - np.sqrt(15.0)) / 21.0
    w1 = (155.0 - np.sqrt(15.0)) / 1200.0
    a2 = (6.0 + np.sqrt(15.0)) / 21.0
    w2 = (155.0 + np.sqrt(15.0)) / 1200.0
    w0 = 0.225

    nodes = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0],
            [a1, a1],
            [1.0 - 2.0 * a1, a1],
            [a1, 1.0 - 2.0 * a1],
            [a2, a2],
            [1.0 - 2.0 * a2, a2],
            [a2, 1.0 - 2.0 * a2],
        ]
    )
    weights = np.array([w0, w1, w1, w1, w2, w2, w2])
    return nodes, weights


class MetasurfaceFarFieldEvaluator:
    """Evaluates far-field radiation vectors and patterns from solved RWG surface currents."""

    def __init__(
        self,
        mesh: TriangleMesh,
        currents_path: Path,
        frequency_hz: float,
        quadrature_rule: int = 7,
    ) -> None:
        self.mesh = mesh
        self.frequency_hz = frequency_hz
        self.omega = 2.0 * np.pi * frequency_hz
        self.k0 = self.omega / SPEED_OF_LIGHT
        self.eta0 = VACUUM_IMPEDANCE

        faces_edges, signed_lengths, free_vertices = _build_edge_mapping(mesh)
        nodes, weights = _dunavant_quadrature(quadrature_rule)

        num_faces = len(mesh.faces)
        num_quad = len(weights)
        quad_points = np.zeros((num_faces, num_quad, 3), dtype=np.float64)

        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]

        for q in range(num_quad):
            xi, eta = nodes[q]
            zeta = 1.0 - xi - eta
            quad_points[:, q, :] = v0 * zeta + v1 * xi + v2 * eta

        # Read unscaled physical currents [J_int, M_int, J_ext, M_ext]
        num_edges = np.max(faces_edges) + 1
        currents_raw = np.fromfile(currents_path, dtype=np.complex128)
        expected_size = 4 * num_edges
        if currents_raw.size != expected_size:
            msg = (
                f"currents file {currents_path} has {currents_raw.size} elements; "
                f"expected {expected_size}"
            )
            raise ValueError(msg)

        j_ext = currents_raw[2 * num_edges : 3 * num_edges]
        m_ext = currents_raw[3 * num_edges : 4 * num_edges]

        # Precompute weighted basis vector contributions at each quadrature point
        c_j = np.zeros((num_faces, num_quad, 3), dtype=np.complex128)
        c_m = np.zeros((num_faces, num_quad, 3), dtype=np.complex128)

        for le in range(3):
            e_idx = faces_edges[:, le]
            je = j_ext[e_idx]
            me = m_ext[e_idx]
            le_signed = signed_lengths[:, le]
            rf = free_vertices[:, le, :]

            coeff_j = (0.5 * le_signed * je)[:, None]
            coeff_m = (0.5 * le_signed * me)[:, None]

            for q in range(num_quad):
                rho = quad_points[:, q, :] - rf
                w = weights[q]
                c_j[:, q, :] += w * coeff_j * rho
                c_m[:, q, :] += w * coeff_m * rho

        self.all_rq = quad_points.reshape(-1, 3)
        self.all_cj = c_j.reshape(-1, 3)
        self.all_cm = c_m.reshape(-1, 3)

    def evaluate_directions(
        self, r_hat: np.ndarray, theta_hat: np.ndarray, phi_hat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute F_theta, F_phi, and radiation intensity U for an array of observation directions.

        Args:
            r_hat: (N, 3) array of radial unit vectors.
            theta_hat: (N, 3) array of theta unit vectors.
            phi_hat: (N, 3) array of phi unit vectors.

        Returns:
            f_theta: (N,) complex far-field theta component.
            f_phi: (N,) complex far-field phi component.
            intensity: (N,) real radiation intensity U(theta, phi) in W/sr.
        """
        # Phase matrix: exp(+j * k0 * r_hat . r_q) of shape (N, M_quad)
        phase_matrix = np.exp(1j * self.k0 * np.dot(r_hat, self.all_rq.T))
        n_vectors = np.dot(phase_matrix, self.all_cj)  # (N, 3)
        l_vectors = np.dot(phase_matrix, self.all_cm)  # (N, 3)

        n_theta = np.sum(n_vectors * theta_hat, axis=1)
        n_phi = np.sum(n_vectors * phi_hat, axis=1)
        l_theta = np.sum(l_vectors * theta_hat, axis=1)
        l_phi = np.sum(l_vectors * phi_hat, axis=1)

        # Standard Schelkunoff far-field formula (Balanis eq. 12-10):
        # E_far = (exp(-j k0 r) / r) * [ F_theta theta_hat + F_phi phi_hat ]
        coeff = -1j * self.k0 / (4.0 * np.pi)
        f_theta = coeff * (self.eta0 * n_theta + l_phi)
        f_phi = coeff * (self.eta0 * n_phi - l_theta)

        intensity = (np.abs(f_theta) ** 2 + np.abs(f_phi) ** 2) / (2.0 * self.eta0)
        return f_theta, f_phi, intensity

    def compute_total_power_and_directivity(
        self, num_theta: int = 91, num_phi: int = 181
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Compute total radiated power P_rad (W) and 3D directivity D(theta, phi).

        Returns:
            p_rad: Total radiated power in Watts.
            d_max: Maximum numerical directivity (dimensionless).
            d_max_dbi: Maximum numerical directivity in dBi.
            theta_grid: (num_theta,) array.
            phi_grid: (num_phi,) array.
            directivity_grid: (num_theta, num_phi) array of directivity.
        """
        theta_vals = np.linspace(0.0, np.pi, num_theta)
        phi_vals = np.linspace(0.0, 2.0 * np.pi, num_phi, endpoint=False)
        th_grid, ph_grid = np.meshgrid(theta_vals, phi_vals, indexing="ij")

        sin_th = np.sin(th_grid).ravel()
        cos_th = np.cos(th_grid).ravel()
        sin_ph = np.sin(ph_grid).ravel()
        cos_ph = np.cos(ph_grid).ravel()

        r_hat = np.column_stack([sin_th * cos_ph, sin_th * sin_ph, cos_th])
        theta_hat = np.column_stack([cos_th * cos_ph, cos_th * sin_ph, -sin_th])
        phi_hat = np.column_stack([-sin_ph, cos_ph, np.zeros_like(sin_ph)])

        _, _, intensity = self.evaluate_directions(r_hat, theta_hat, phi_hat)
        u_grid = intensity.reshape(num_theta, num_phi)

        dtheta = theta_vals[1] - theta_vals[0]
        dphi = phi_vals[1] - phi_vals[0]

        weights_th = np.ones(num_theta)
        weights_th[0] = 0.5
        weights_th[-1] = 0.5

        integrand = u_grid * np.sin(theta_vals)[:, None]
        p_rad = float(np.sum(integrand * weights_th[:, None]) * dtheta * dphi)

        directivity_grid = 4.0 * np.pi * u_grid / p_rad
        d_max = float(np.max(directivity_grid))
        d_max_dbi = float(10.0 * np.log10(max(d_max, 1e-12)))

        return p_rad, d_max, d_max_dbi, theta_vals, phi_vals, directivity_grid


def compute_prescribed_target_total_power(
    configuration: WorkflowConfiguration,
    reference: FieldReference | None,
    num_theta: int = 91,
    num_phi: int = 181,
    r_far: float = 1e6,
) -> float:
    """Compute total radiated power P_rad of the exterior prescribed target field."""
    if reference is None:
        return 0.0

    theta_vals = np.linspace(0.0, np.pi, num_theta)
    phi_vals = np.linspace(0.0, 2.0 * np.pi, num_phi, endpoint=False)
    th_grid, ph_grid = np.meshgrid(theta_vals, phi_vals, indexing="ij")

    radius = np.full(th_grid.size, r_far)
    e_sph, _ = evaluate_prescribed_fields(
        reference,
        "exterior",
        configuration,
        radius=radius,
        theta=th_grid.ravel(),
        phi=ph_grid.ravel(),
    )

    eth = e_sph[1].reshape(num_theta, num_phi)
    eph = e_sph[2].reshape(num_theta, num_phi)
    eta0 = configuration.case.eta_exterior

    u_grid = (r_far**2 / (2.0 * eta0)) * (np.abs(eth) ** 2 + np.abs(eph) ** 2)

    dtheta = theta_vals[1] - theta_vals[0]
    dphi = phi_vals[1] - phi_vals[0]
    weights_th = np.ones(num_theta)
    weights_th[0] = 0.5
    weights_th[-1] = 0.5

    integrand = u_grid * np.sin(theta_vals)[:, None]
    p_rad = float(np.sum(integrand * weights_th[:, None]) * dtheta * dphi)
    return p_rad


def compute_prescribed_target_far_field(
    configuration: WorkflowConfiguration,
    reference: FieldReference | None,
    r_hat: np.ndarray,
    p_rad_target: float,
    r_far: float = 1e6,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exterior prescribed target far field along arbitrary directions.

    Uses evaluate_prescribed_fields directly from the application's prescribed field module,
    providing complete generality for any configured target field definition.

    Args:
        configuration: Full workflow configuration object.
        reference: FieldReference to the exterior target field (or None).
        r_hat: (N, 3) observation direction unit vectors.
        p_rad_target: Total radiated power of the target field in Watts.
        r_far: Far-field radius for numerical evaluation in meters.

    Returns:
        f_mag: (N,) target field amplitude r|E| in Volts.
        directivity_dbi: (N,) target directivity in dBi.
    """
    if reference is None or p_rad_target <= 0.0:
        zeros = np.zeros(len(r_hat), dtype=float)
        return zeros, np.full(len(r_hat), -120.0, dtype=float)

    # Convert r_hat unit vectors to spherical coordinates (theta, phi)
    theta = np.arccos(np.clip(r_hat[:, 2], -1.0, 1.0))
    phi = np.arctan2(r_hat[:, 1], r_hat[:, 0])
    phi = np.where(np.hypot(r_hat[:, 0], r_hat[:, 1]) == 0.0, 0.0, phi)

    radius = np.full_like(theta, r_far)
    electric_sph, _ = evaluate_prescribed_fields(
        reference,
        "exterior",
        configuration,
        radius=radius,
        theta=theta,
        phi=phi,
    )

    # Transverse spherical electric components (E_theta, E_phi) in the far field
    eth = electric_sph[1]
    eph = electric_sph[2]

    f_mag = r_far * np.sqrt(np.abs(eth) ** 2 + np.abs(eph) ** 2)

    eta0 = configuration.case.eta_exterior
    u_intensity = (f_mag**2) / (2.0 * eta0)
    directivity_lin = (4.0 * np.pi * u_intensity) / max(p_rad_target, 1e-12)
    directivity_dbi = 10.0 * np.log10(np.maximum(directivity_lin, 1e-12))

    return f_mag, directivity_dbi


def compute_principal_cuts(
    evaluator: MetasurfaceFarFieldEvaluator,
    p_rad: float,
    num_points: int = 361,
) -> dict[str, Any]:
    """Compute 2D continuous 360-degree cuts in the E-plane, H-plane, and XY-plane."""
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=True)
    angles_deg = np.degrees(angles)

    cuts: dict[str, Any] = {}

    # 1. E-plane: z-x plane (phi = 0 for alpha in [0, pi], phi = pi for alpha in [pi, 2pi])
    # alpha is measured from +z axis towards +x axis:
    # r_hat = [sin(alpha), 0, cos(alpha)]
    r_hat_e = np.column_stack([np.sin(angles), np.zeros_like(angles), np.cos(angles)])
    # In z-x plane, co-pol unit vector is theta_hat (parallel to x at broadside +z):
    # theta_hat = [cos(alpha), 0, -sin(alpha)]
    theta_hat_e = np.column_stack([np.cos(angles), np.zeros_like(angles), -np.sin(angles)])
    # cross-pol unit vector is y_hat:
    phi_hat_e = np.column_stack(
        [np.zeros_like(angles), np.ones_like(angles), np.zeros_like(angles)]
    )

    f_th_e, f_ph_e, u_e = evaluator.evaluate_directions(r_hat_e, theta_hat_e, phi_hat_e)
    d_e = 4.0 * np.pi * u_e / p_rad
    d_e_dbi = 10.0 * np.log10(np.maximum(d_e, 1e-12))
    co_pol_e = np.abs(f_th_e)
    cross_pol_e = np.abs(f_ph_e)

    cuts["E_plane"] = {
        "angles_deg": angles_deg,
        "angles_rad": angles,
        "r_hat": r_hat_e,
        "co_hat": theta_hat_e,
        "cross_hat": phi_hat_e,
        "co_pol": co_pol_e,
        "cross_pol": cross_pol_e,
        "total_field": np.sqrt(co_pol_e**2 + cross_pol_e**2),
        "directivity_dbi": d_e_dbi,
        "description": "E-plane (z-x plane): alpha=0 is +z, alpha=90 is +x",
    }

    # 2. H-plane: y-z plane (perpendicular to x-dipole)
    # beta is measured from +z axis towards +y axis:
    # r_hat = [0, sin(beta), cos(beta)]
    r_hat_h = np.column_stack([np.zeros_like(angles), np.sin(angles), np.cos(angles)])
    # In y-z plane, co-pol is along x-axis (+x):
    co_hat_h = np.column_stack([np.ones_like(angles), np.zeros_like(angles), np.zeros_like(angles)])
    # cross-pol is tangential in y-z plane:
    cross_hat_h = np.column_stack([np.zeros_like(angles), np.cos(angles), -np.sin(angles)])

    f_co_h, f_cr_h, u_h = evaluator.evaluate_directions(r_hat_h, co_hat_h, cross_hat_h)
    d_h = 4.0 * np.pi * u_h / p_rad
    d_h_dbi = 10.0 * np.log10(np.maximum(d_h, 1e-12))
    co_pol_h = np.abs(f_co_h)
    cross_pol_h = np.abs(f_cr_h)

    cuts["H_plane"] = {
        "angles_deg": angles_deg,
        "angles_rad": angles,
        "r_hat": r_hat_h,
        "co_hat": co_hat_h,
        "cross_hat": cross_hat_h,
        "co_pol": co_pol_h,
        "cross_pol": cross_pol_h,
        "total_field": np.sqrt(co_pol_h**2 + cross_pol_h**2),
        "directivity_dbi": d_h_dbi,
        "description": "H-plane (y-z plane): beta=0 is +z, beta=90 is +y",
    }

    # 3. XY-plane: equatorial plane (theta = 90 deg)
    # phi measured from +x towards +y:
    # r_hat = [cos(phi), sin(phi), 0]
    r_hat_xy = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    # co-pol is phi_hat: [-sin(phi), cos(phi), 0]
    phi_hat_xy = np.column_stack([-np.sin(angles), np.cos(angles), np.zeros_like(angles)])
    # cross-pol is z_hat: [0, 0, 1]
    z_hat_xy = np.column_stack([np.zeros_like(angles), np.zeros_like(angles), np.ones_like(angles)])

    f_ph_xy, f_z_xy, u_xy = evaluator.evaluate_directions(r_hat_xy, phi_hat_xy, z_hat_xy)
    d_xy = 4.0 * np.pi * u_xy / p_rad
    d_xy_dbi = 10.0 * np.log10(np.maximum(d_xy, 1e-12))
    co_pol_xy = np.abs(f_ph_xy)
    cross_pol_xy = np.abs(f_z_xy)

    cuts["XY_plane"] = {
        "angles_deg": angles_deg,
        "angles_rad": angles,
        "r_hat": r_hat_xy,
        "co_hat": phi_hat_xy,
        "cross_hat": z_hat_xy,
        "co_pol": co_pol_xy,
        "cross_pol": cross_pol_xy,
        "total_field": np.sqrt(co_pol_xy**2 + cross_pol_xy**2),
        "directivity_dbi": d_xy_dbi,
        "description": "XY-plane (equatorial): phi=0 is +x, phi=90 is +y",
    }

    return cuts


def plot_polar_pattern(
    cut_data: dict[str, Any],
    target_data: tuple[np.ndarray, np.ndarray] | None,
    title: str,
    output_path: Path,
    use_dbi: bool = True,
    dbi_min: float = -30.0,
) -> None:
    """Generate a high-quality polar radiation pattern diagram."""
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    angles = cut_data["angles_rad"]
    if use_dbi:
        values_co = np.maximum(cut_data["directivity_dbi"], dbi_min)
        max_val = max(5.0, float(np.max(values_co)) + 2.0)
        ax.set_ylim(dbi_min, max_val)
    else:
        values_co = cut_data["co_pol"]

    # Plot metasurface computed co-pol
    ax.plot(angles, values_co, label="Metasurface Co-pol", color="#007a87", linewidth=2.2)

    # Plot cross-pol if significant
    if not use_dbi:
        ax.plot(
            angles,
            cut_data["cross_pol"],
            label="Metasurface Cross-pol",
            color="#d95f02",
            linewidth=1.4,
            linestyle="--",
        )

    # Overlay target field if available
    if target_data is not None:
        target_field, target_dbi = target_data
        target_vals = np.maximum(target_dbi, dbi_min) if use_dbi else target_field
        ax.plot(
            angles,
            target_vals,
            label="Prescribed Target",
            color="#c44838",
            linewidth=1.8,
            linestyle=":",
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=18, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_rectangular_cuts(
    cuts: dict[str, Any],
    targets: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    output_path: Path,
    dbi_min: float = -30.0,
) -> None:
    """Plot E-plane and H-plane cuts in rectangular dB format side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    plane_keys = [("E_plane", "E-Plane Cut (z-x plane)"), ("H_plane", "H-Plane Cut (y-z plane)")]

    for ax, (key, plane_name) in zip(axes, plane_keys):
        data = cuts[key]
        angles_deg = data["angles_deg"]
        dbi_vals = np.maximum(data["directivity_dbi"], dbi_min)

        ax.plot(
            angles_deg, dbi_vals, label="Metasurface Directivity", color="#007a87", linewidth=2.2
        )

        if key in targets:
            _, target_dbi = targets[key]
            ax.plot(
                angles_deg,
                np.maximum(target_dbi, dbi_min),
                label="Prescribed Target",
                color="#c44838",
                linewidth=1.8,
                linestyle="--",
            )

        ax.set_title(plane_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Angle [degrees]", fontsize=10)
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_ylim(dbi_min, 6.0)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True)

    axes[0].set_ylabel("Directivity [dBi]", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def export_cuts_csv(cuts: dict[str, Any], output_path: Path) -> None:
    """Save angle-by-angle sampled field cuts to CSV."""
    angles_deg = cuts["E_plane"]["angles_deg"]
    rows = []
    header = [
        "angle_deg",
        "E_plane_copol_V",
        "E_plane_xpol_V",
        "E_plane_directivity_dbi",
        "H_plane_copol_V",
        "H_plane_xpol_V",
        "H_plane_directivity_dbi",
        "XY_plane_copol_V",
        "XY_plane_xpol_V",
        "XY_plane_directivity_dbi",
    ]

    for idx, deg in enumerate(angles_deg):
        rows.append(
            [
                deg,
                cuts["E_plane"]["co_pol"][idx],
                cuts["E_plane"]["cross_pol"][idx],
                cuts["E_plane"]["directivity_dbi"][idx],
                cuts["H_plane"]["co_pol"][idx],
                cuts["H_plane"]["cross_pol"][idx],
                cuts["H_plane"]["directivity_dbi"][idx],
                cuts["XY_plane"]["co_pol"][idx],
                cuts["XY_plane"]["cross_pol"][idx],
                cuts["XY_plane"]["directivity_dbi"][idx],
            ]
        )

    table = np.array(rows)
    np.savetxt(output_path, table, delimiter=",", header=",".join(header), comments="", fmt="%.6e")


def process_case_far_field(
    run_dir: Path,
    case_id: str,
    mesh: TriangleMesh,
    workflow_config: WorkflowConfiguration,
    output_base_dir: Path | None = None,
    num_points: int = 361,
) -> dict[str, Any]:
    """Evaluate far-field radiation patterns and plots for one simulation case."""
    case_dir = run_dir / "cases" / case_id
    currents_path = case_dir / "currents.bin"
    if not currents_path.is_file():
        raise FileNotFoundError(f"missing surface currents: {currents_path}")

    freq_hz = float(workflow_config.case.frequency_hz)
    evaluator = MetasurfaceFarFieldEvaluator(mesh, currents_path, freq_hz)

    print(f"[{case_id}] Integrating 3D radiation over sphere...", flush=True)
    p_rad, d_max, d_max_dbi, _, _, _ = evaluator.compute_total_power_and_directivity()
    print(
        f"[{case_id}] P_rad = {p_rad:.4f} W, D_max = {d_max:.4f} ({d_max_dbi:.2f} dBi)", flush=True
    )

    print(f"[{case_id}] Computing principal plane cuts (E-plane, H-plane, XY-plane)...", flush=True)
    cuts = compute_principal_cuts(evaluator, p_rad, num_points=num_points)

    # Determine target prescribed field if defined in configuration
    targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    target_info = None

    this_case = next((c for c in workflow_config.simulation.cases if c.id == case_id), None)
    target_ref: FieldReference | None = None
    if this_case and this_case.excitations:
        inc_field_id = this_case.excitations[0].field.field_id
        for trans in workflow_config.transformations:
            if trans.gstc.interior.field_id == inc_field_id:
                target_ref = trans.gstc.exterior
                break
    if target_ref is None:
        for trans in workflow_config.transformations:
            if trans.id == case_id:
                target_ref = trans.gstc.exterior
                break

    if target_ref is not None:
        print(
            f"[{case_id}] Evaluating exterior prescribed target field: '{target_ref.field_id}'...",
            flush=True,
        )
        p_rad_target = compute_prescribed_target_total_power(workflow_config, target_ref)
        target_field_def = workflow_config.fields.get(target_ref.field_id)
        field_type_name = (
            type(target_field_def).__name__ if target_field_def is not None else "unknown"
        )

        for plane in ("E_plane", "H_plane", "XY_plane"):
            r_hat_cut = cuts[plane]["r_hat"]
            f_mag_target, d_dbi_target = compute_prescribed_target_far_field(
                workflow_config,
                target_ref,
                r_hat_cut,
                p_rad_target,
            )
            targets[plane] = (f_mag_target, d_dbi_target)

        target_d_max = max(float(np.max(targets[p][1])) for p in targets)
        print(
            f"[{case_id}] Target prescribed field: "
            f"P_rad = {p_rad_target:.4f} W, D_max = {target_d_max:.2f} dBi",
            flush=True,
        )

        target_info = {
            "field_id": target_ref.field_id,
            "field_type": field_type_name,
            "target_radiated_power_w": p_rad_target,
            "target_directivity_max_dbi": target_d_max,
        }

    # Setup output directory
    out_dir = (output_base_dir or (case_dir / "far_field")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    export_cuts_csv(cuts, out_dir / "far_field_cuts.csv")

    # Generate plots
    plot_polar_pattern(
        cuts["E_plane"],
        targets.get("E_plane"),
        f"E-Plane Directivity [dBi] - {case_id}",
        out_dir / "far_field_e_plane_polar.png",
        use_dbi=True,
    )
    plot_polar_pattern(
        cuts["H_plane"],
        targets.get("H_plane"),
        f"H-Plane Directivity [dBi] - {case_id}",
        out_dir / "far_field_h_plane_polar.png",
        use_dbi=True,
    )
    plot_polar_pattern(
        cuts["XY_plane"],
        targets.get("XY_plane"),
        f"XY-Plane Directivity [dBi] - {case_id}",
        out_dir / "far_field_xy_plane_polar.png",
        use_dbi=True,
    )

    plot_rectangular_cuts(
        cuts,
        targets,
        f"Principal Plane Radiation Directivity - {case_id}",
        out_dir / "far_field_cuts_rectangular.png",
    )

    # Save metrics JSON
    metrics = {
        "case_id": case_id,
        "frequency_hz": freq_hz,
        "total_radiated_power_w": p_rad,
        "directivity_max_linear": d_max,
        "directivity_max_dbi": d_max_dbi,
        "target_info": target_info,
        "e_plane_max_dbi": float(np.max(cuts["E_plane"]["directivity_dbi"])),
        "h_plane_max_dbi": float(np.max(cuts["H_plane"]["directivity_dbi"])),
        "xy_plane_max_dbi": float(np.max(cuts["XY_plane"]["directivity_dbi"])),
    }
    with (out_dir / "far_field_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{case_id}] Saved far-field plots and metrics to: {out_dir}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the simulation run directory (e.g. simulation_iter80)",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Path to sphere.obj (defaults to mesh referenced in configuration)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON configuration (defaults to <run-dir>/configuration.json)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Specific case IDs to evaluate (default: all cases found in configuration)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory for far-field plots",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=361,
        help="Number of angular points for 2D pattern cuts (default: 361)",
    )

    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    # Locate configuration
    config_path = args.config
    if config_path is None:
        candidates = [
            run_dir / "configuration.json",
            run_dir / "run.json",
            run_dir.parent / "experiment.json",
        ]
        for cand in candidates:
            if cand.is_file():
                config_path = cand
                break
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"cannot locate configuration JSON in {run_dir}")

    workflow_config = load_workflow_configuration(config_path)

    # Locate mesh
    mesh_path = args.mesh
    if mesh_path is None:
        if workflow_config.mesh_path and Path(workflow_config.mesh_path).is_file():
            mesh_path = Path(workflow_config.mesh_path).resolve()
        elif workflow_config.sie and workflow_config.sie.mesh and workflow_config.sie.mesh.path:
            configured_mesh = workflow_config.sie.mesh.path
            for cand in [
                Path(configured_mesh),
                config_path.parent / configured_mesh,
                Path("applications/anisotropic_gstc_sphere") / configured_mesh,
            ]:
                if cand.is_file():
                    mesh_path = cand.resolve()
                    break
    if mesh_path is None or not mesh_path.is_file():
        fallback = Path(
            "applications/anisotropic_gstc_sphere/meshes/reflectionless_100mhz_radius2m_ppw10/sphere.obj"
        )
        if fallback.is_file():
            mesh_path = fallback.resolve()
        else:
            raise FileNotFoundError("cannot locate sphere mesh for far-field evaluation")

    print(f"Loading mesh: {mesh_path}")
    mesh = load_triangle_obj(mesh_path)

    # Determine cases
    case_ids = args.cases
    if not case_ids:
        case_ids = [c.id for c in workflow_config.simulation.cases]
    if not case_ids:
        # Fallback to scanning cases directory
        cases_dir = run_dir / "cases"
        if cases_dir.is_dir():
            case_ids = [
                d.name for d in cases_dir.iterdir() if d.is_dir() and (d / "currents.bin").is_file()
            ]

    print(f"Evaluating far field for cases: {case_ids}")
    all_metrics = {}
    for case_id in case_ids:
        metrics = process_case_far_field(
            run_dir,
            case_id,
            mesh,
            workflow_config,
            output_base_dir=(args.output_dir / case_id if args.output_dir else None),
            num_points=args.num_points,
        )
        all_metrics[case_id] = metrics

    print("\n=========================================================================")
    print("                      FAR-FIELD EVALUATION COMPLETE                      ")
    print("=========================================================================")
    for cid, met in all_metrics.items():
        prad = met["total_radiated_power_w"]
        dmax = met["directivity_max_dbi"]
        print(f"Case: {cid:22s} | P_rad = {prad:8.3f} W | D_max = {dmax:5.2f} dBi")
    print("=========================================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
