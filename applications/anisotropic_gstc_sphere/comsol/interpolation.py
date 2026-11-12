#!/usr/bin/env python3
"""Interpolation and surface parameter utilities for COMSOL Case B validation.

Provides:
- Loading of GSTC 10-parameter vectors theta, 32-complex face tensors, or face_parameters.csv
- Centroid extraction and spherical coordinate mapping using validated TriangleMesh
- Adaptive angular grid resolution calculation based on face count N_f
- 2D (theta, phi) scattered-to-regular interpolation with periodic wrapping in phi
- Export of COMSOL-native 2D grid interpolation files
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.interpolate import griddata

from sie_mom.mesh.obj_mesh import load_triangle_obj

# Physical vacuum characteristic impedance [Ohm]
VACUUM_PERMITTIVITY: float = 8.8541878188e-12
VACUUM_PERMEABILITY: float = 1.25663706127e-6
ETA_0: float = math.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)

PARAMETER_NAMES: List[str] = [
    "y1", "y2", "y3",
    "c1", "c2", "c3", "c4",
    "z1", "z2", "z3",
]


def extract_thetas_from_tensors(records: np.ndarray, eta0: float = ETA_0) -> np.ndarray:
    """Extract normalized parameters theta (N_faces, 10) from physical tensors (N_faces, 32).

    Parameters:
      p = 0..2 : y1, y2, y3 (dimensionless electric admittance, scaled by eta0)
      p = 3..6 : c1, c2, c3, c4 (real dimensionless magneto-electric coupling)
      p = 7..9 : z1, z2, z3 (dimensionless magnetic impedance, scaled by 1/eta0)
    """
    num_faces = records.shape[0]
    thetas = np.zeros((num_faces, 10), dtype=np.float64)
    for m in range(num_faces):
        L = records[m, :16].reshape(4, 4)
        thetas[m, 0] = L[0, 0].imag * eta0
        thetas[m, 1] = L[1, 1].imag * eta0
        thetas[m, 2] = L[0, 1].imag * eta0
        thetas[m, 3] = L[0, 2].real
        thetas[m, 4] = L[0, 3].real
        thetas[m, 5] = L[1, 2].real
        thetas[m, 6] = L[1, 3].real
        thetas[m, 7] = L[2, 2].imag / eta0
        thetas[m, 8] = L[3, 3].imag / eta0
        thetas[m, 9] = L[2, 3].imag / eta0
    return thetas


def load_face_parameters(
    source_path: Path,
    mesh_path: Path | None = None,
    eta0: float = ETA_0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load surface parameters and spherical coordinates from .bin or .csv.

    Parameters:
      source_path: Path to thetas.bin, face_tensors.bin, or face_parameters.csv
      mesh_path: Path to sphere OBJ mesh (required for binary files)
      eta0: Vacuum reference impedance [Ohm]

    Returns:
      thetas: (N_faces, 10) float64 array of dimensionless parameters
      theta: (N_faces,) float64 array of polar angles in radians [0, pi]
      phi: (N_faces,) float64 array of azimuthal angles in radians [-pi, pi]
      centroids: (N_faces, 3) float64 array of face centroids in Cartesian coordinates [m]
    """
    source_path = Path(source_path)

    if source_path.suffix.lower() == ".csv":
        with source_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        num_faces = len(rows)
        thetas = np.zeros((num_faces, 10), dtype=np.float64)
        centroids = np.zeros((num_faces, 3), dtype=np.float64)
        theta_deg = np.zeros(num_faces, dtype=np.float64)
        phi_deg = np.zeros(num_faces, dtype=np.float64)

        for i, row in enumerate(rows):
            centroids[i, 0] = float(row["x_m"])
            centroids[i, 1] = float(row["y_m"])
            centroids[i, 2] = float(row["z_m"])
            theta_deg[i] = float(row["theta_deg"])
            phi_deg[i] = float(row["phi_deg"])

            # Read the 10 parameters
            thetas[i, 0] = float(row.get("theta_0_y1", 0.0))
            thetas[i, 1] = float(row.get("theta_1_y2", 0.0))
            thetas[i, 2] = float(row.get("theta_2_y3", 0.0))
            thetas[i, 3] = float(row.get("theta_3_c1", 0.0))
            thetas[i, 4] = float(row.get("theta_4_c2", 0.0))
            thetas[i, 5] = float(row.get("theta_5_c3", 0.0))
            thetas[i, 6] = float(row.get("theta_6_c4", 0.0))
            thetas[i, 7] = float(row.get("theta_7_z1", 0.0))
            thetas[i, 8] = float(row.get("theta_8_z2", 0.0))
            thetas[i, 9] = float(row.get("theta_9_z3", 0.0))

        theta = np.deg2rad(theta_deg)
        # Wrap phi into [-pi, pi]
        phi = np.arctan2(np.sin(np.deg2rad(phi_deg)), np.cos(np.deg2rad(phi_deg)))
        return thetas, theta, phi, centroids

    # Binary file loading
    if mesh_path is None:
        raise ValueError(f"mesh_path is required to load coordinates for binary file {source_path}")

    mesh = load_triangle_obj(mesh_path)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    r = np.linalg.norm(centroids, axis=1)
    if np.any(r == 0.0):
        raise ValueError("Mesh has zero-radius face centroids")
    theta = np.arccos(np.clip(centroids[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(centroids[:, 1], centroids[:, 0])

    file_bytes = source_path.stat().st_size
    num_faces = len(centroids)

    # Check whether it is thetas (N * 10 * 8 bytes) or tensors (complex128 or complex64)
    if file_bytes == num_faces * 10 * 8:
        thetas = np.fromfile(source_path, dtype=np.float64).reshape(num_faces, 10)
    elif file_bytes == num_faces * 32 * 16:
        records = np.fromfile(source_path, dtype=np.complex128).reshape(num_faces, 32)
        thetas = extract_thetas_from_tensors(records, eta0=eta0)
    elif file_bytes == num_faces * 32 * 8:
        records = np.fromfile(source_path, dtype=np.complex64).reshape(num_faces, 32).astype(np.complex128)
        thetas = extract_thetas_from_tensors(records, eta0=eta0)
    elif file_bytes == num_faces * 16 * 16:
        records = np.fromfile(source_path, dtype=np.complex128).reshape(num_faces, 16)
        thetas = extract_thetas_from_tensors(records, eta0=eta0)
    elif file_bytes == num_faces * 16 * 8:
        records = np.fromfile(source_path, dtype=np.complex64).reshape(num_faces, 16).astype(np.complex128)
        thetas = extract_thetas_from_tensors(records, eta0=eta0)
    else:
        raise ValueError(
            f"File size {file_bytes} bytes for {source_path} does not match expected size for "
            f"{num_faces} faces: expected {num_faces * 10 * 8} (thetas.bin), "
            f"{num_faces * 32 * 16} / {num_faces * 32 * 8} (face_tensors.bin, f64/f32)"
        )

    return thetas, theta, phi, centroids


def compute_adaptive_grid_resolution(num_faces: int, oversample_factor: float = 2.0) -> Tuple[int, int]:
    """Compute (N_theta, N_phi) regular grid resolution dynamically from mesh face count.

    Calculates:
      Average face solid angle: Omega_face = 4*pi / N_f
      Face angular width: Delta_theta_face = sqrt(Omega_face)
      Nyquist step: Delta_theta_grid = Delta_theta_face / oversample_factor
      N_theta = max(19, ceil(pi / Delta_theta_grid) + 1)
      N_phi = 2 * (N_theta - 1) + 1
    """
    omega_face = 4.0 * math.pi / max(1, num_faces)
    delta_face = math.sqrt(omega_face)
    delta_grid = delta_face / max(1.0, oversample_factor)

    n_theta = max(19, int(math.ceil(math.pi / delta_grid)) + 1)
    # Ensure n_theta is odd so an exact equator point theta = pi/2 exists
    if n_theta % 2 == 0:
        n_theta += 1
    n_phi = 2 * (n_theta - 1) + 1
    return n_theta, n_phi


def interpolate_thetas_to_grid(
    theta: np.ndarray,
    phi: np.ndarray,
    thetas: np.ndarray,
    n_theta: int = 181,
    n_phi: int = 361,
    method: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate unstructured face parameters to a regular (theta, phi) grid with periodic wrapping.

    Parameters:
      theta: (N_faces,) polar angles in radians [0, pi]
      phi: (N_faces,) azimuthal angles in radians [-pi, pi]
      thetas: (N_faces, 10) parameter values
      n_theta: Number of grid points along theta
      n_phi: Number of grid points along phi
      method: Interpolation method for griddata ('cubic', 'linear', or 'nearest')

    Returns:
      grid_theta: (n_theta,) 1D array of polar angles in radians [0, pi]
      grid_phi: (n_phi,) 1D array of azimuthal angles in radians [-pi, pi]
      grid_thetas: (n_phi, n_theta, 10) 3D array of interpolated values.
                   Row index corresponds to phi, column index to theta.
    """
    grid_theta = np.linspace(0.0, np.pi, n_theta)
    grid_phi = np.linspace(-np.pi, np.pi, n_phi)
    mesh_theta, mesh_phi = np.meshgrid(grid_theta, grid_phi)

    # Periodic padding across phi = -pi and +pi by +/- 2*pi
    two_pi = 2.0 * np.pi
    phi_pad = np.concatenate([phi - two_pi, phi, phi + two_pi])
    theta_pad = np.concatenate([theta, theta, theta])

    grid_thetas = np.zeros((n_phi, n_theta, 10), dtype=np.float64)

    for p in range(10):
        val = thetas[:, p]
        val_pad = np.concatenate([val, val, val])

        # High-order interpolation with fallback for pole extremities
        val_grid = griddata((theta_pad, phi_pad), val_pad, (mesh_theta, mesh_phi), method=method)
        nan_mask = np.isnan(val_grid)
        if np.any(nan_mask):
            val_linear = griddata((theta_pad, phi_pad), val_pad, (mesh_theta, mesh_phi), method="linear")
            val_grid[nan_mask] = val_linear[nan_mask]
            nan_mask = np.isnan(val_grid)
            if np.any(nan_mask):
                val_nearest = griddata((theta_pad, phi_pad), val_pad, (mesh_theta, mesh_phi), method="nearest")
                val_grid[nan_mask] = val_nearest[nan_mask]
        grid_thetas[:, :, p] = val_grid

    return grid_theta, grid_phi, grid_thetas


def export_comsol_grid_files(
    output_dir: Path,
    grid_theta: np.ndarray,
    grid_phi: np.ndarray,
    grid_thetas: np.ndarray,
    file_prefix: str = "interp",
) -> Dict[str, Path]:
    """Export 10 COMSOL-native 2D grid files for Interpolation features.

    File format:
      % Grid
      theta_0 theta_1 ... theta_{N-1}
      phi_0 phi_1 ... phi_{M-1}
      % Data
      val(th_0, ph_0) val(th_1, ph_0) ... val(th_{N-1}, ph_0)
      val(th_0, ph_1) val(th_1, ph_1) ... val(th_{N-1}, ph_1)
      ...

    Parameters:
      output_dir: Directory where grid text files will be saved
      grid_theta: (n_theta,) polar angles in radians [0, pi]
      grid_phi: (n_phi,) azimuthal angles in radians [-pi, pi]
      grid_thetas: (n_phi, n_theta, 10) interpolated parameters
      file_prefix: Prefix for filenames (default: 'interp')

    Returns:
      Dict mapping parameter name ("y1", ..., "z3") to Path of the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta_str = " ".join(f"{th:.8e}" for th in grid_theta)
    phi_str = " ".join(f"{ph:.8e}" for ph in grid_phi)
    grid_header = f"% Grid\n{theta_str}\n{phi_str}\n% Data\n"

    paths: Dict[str, Path] = {}
    for p, name in enumerate(PARAMETER_NAMES):
        file_path = output_dir / f"{file_prefix}_{name}.txt"
        data_block = grid_thetas[:, :, p]

        lines = [grid_header]
        for row in range(len(grid_phi)):
            row_str = " ".join(f"{val:.8e}" for val in data_block[row, :])
            lines.append(f"{row_str}\n")

        file_path.write_text("".join(lines), encoding="ascii")
        paths[name] = file_path

    return paths

