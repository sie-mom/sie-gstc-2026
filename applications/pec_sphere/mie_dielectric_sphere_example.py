import os

import numpy as np
import pandas as pd

try:
    from scattnlay import fieldnlay
except ImportError:
    print("Error: 'scattnlay' package not found.")
    print("Please install it using: pip install python-scattnlay")
    raise SystemExit(1)

def _build_observation_grid(radius, grid_res):
    grid_span = 4.0 * radius
    grid_coords = np.linspace(-grid_span, grid_span, grid_res)
    z_pts, x_pts = np.meshgrid(grid_coords, grid_coords)
    y_pts = np.zeros_like(x_pts)
    return x_pts, y_pts, z_pts, grid_span


def _save_fields_csv(output_filename, x_pts, z_pts, e_field, h_field):
    e_magnitude = np.linalg.norm(e_field, axis=1)
    h_magnitude = np.linalg.norm(h_field, axis=1)

    df = pd.DataFrame({
        "x": x_pts.ravel(),
        "z": z_pts.ravel(),
        "Ex_real": e_field[:, 0].real,
        "Ex_imag": e_field[:, 0].imag,
        "Ey_real": e_field[:, 1].real,
        "Ey_imag": e_field[:, 1].imag,
        "Ez_real": e_field[:, 2].real,
        "Ez_imag": e_field[:, 2].imag,
        "E_magnitude": e_magnitude,
        "Hx_real": h_field[:, 0].real,
        "Hx_imag": h_field[:, 0].imag,
        "Hy_real": h_field[:, 1].real,
        "Hy_imag": h_field[:, 1].imag,
        "Hz_real": h_field[:, 2].real,
        "Hz_imag": h_field[:, 2].imag,
        "H_magnitude": h_magnitude,
    })

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    print(f"Saving total field data to '{output_filename}'...")
    df.to_csv(output_filename, index=False)
    print("Data saved successfully.")


def _run_fieldnlay_sphere(eps_r1, mu_r1, eps_r2, mu_r2, radius, lambda0, x_pts, y_pts, z_pts, pec_layer_index):
    n_med = np.sqrt(eps_r2 * mu_r2)
    n_sph = np.sqrt(eps_r1 * mu_r1)

    x = np.array([2 * np.pi * n_med * radius / lambda0])
    m = np.array([n_sph / n_med])

    scaling_factor = (2 * np.pi * n_med) / lambda0
    pts_scaled = np.column_stack([
        x_pts.ravel() * scaling_factor,
        y_pts.ravel() * scaling_factor,
        z_pts.ravel() * scaling_factor,
    ]).astype(float)

    xp = pts_scaled[:, 0]
    yp = pts_scaled[:, 1]
    zp = pts_scaled[:, 2]

    terms, e_field, h_field = fieldnlay(x, m, xp, yp, zp, pl=pec_layer_index)
    return terms, e_field, h_field, x[0], m[0]


def calculate_mie_field_pec_limit(pec_imag=10000.0, output_basename="eh_total_zx_plane_mie_limit.csv"):
    """PEC approximation using high-loss dielectric limit: eps_r1 = 1 + i*pec_imag."""
    print("--- Mie Theory Calculation (scattnlay PEC-limit approximation) ---")

    # --- Physical Parameters (from C++ example) ---
    c0 = 299792458.0
    freq = 1e9  # 1 GHz
    lambda0 = c0 / freq
    radius = 2.0 * lambda0

    eps_r1 = 1.0 + 1j * pec_imag
    mu_r1 = 1.0
    eps_r2 = 1.0  # Air outside
    mu_r2 = 1.0

    grid_res = 400
    x_pts, y_pts, z_pts, grid_span = _build_observation_grid(radius, grid_res)

    terms, e_field, h_field, size_param, rel_index = _run_fieldnlay_sphere(
        eps_r1=eps_r1,
        mu_r1=mu_r1,
        eps_r2=eps_r2,
        mu_r2=mu_r2,
        radius=radius,
        lambda0=lambda0,
        x_pts=x_pts,
        y_pts=y_pts,
        z_pts=z_pts,
        pec_layer_index=-1,
    )

    print(f"eps_r1: {eps_r1}")
    print(f"Wavelength (lambda0): {lambda0:.6f} m")
    print(f"Sphere radius (a): {radius:.6f} m")
    print(f"Size parameter (x): {size_param:.6f}")
    print(f"Relative refractive index (m): {rel_index}")
    print(f"Terms used: {terms}")
    print(f"\nObservation grid: {grid_res} x {grid_res} points, spanning [{-grid_span:.2f}, {grid_span:.2f}] m")

    print("Calculation complete.")

    data_directory = "./applications/pec_sphere/data"
    output_filename = os.path.join(data_directory, output_basename)
    _save_fields_csv(output_filename, x_pts, z_pts, e_field, h_field)


def calculate_mie_field_pec_exact(output_basename="eh_total_zx_plane_mie.csv"):
    """Exact PEC using scattnlay PEC layer selector (pl=0 for single-layer sphere)."""
    print("--- Mie Theory Calculation (scattnlay EXACT PEC using pl=0) ---")

    c0 = 299792458.0
    freq = 1e9
    lambda0 = c0 / freq
    radius = 2.0 * lambda0

    # m is not physically used for a PEC layer, but fieldnlay expects x,m arrays.
    eps_r1 = 1.0
    mu_r1 = 1.0
    eps_r2 = 1.0
    mu_r2 = 1.0

    grid_res = 400
    x_pts, y_pts, z_pts, grid_span = _build_observation_grid(radius, grid_res)

    terms, e_field, h_field, size_param, rel_index = _run_fieldnlay_sphere(
        eps_r1=eps_r1,
        mu_r1=mu_r1,
        eps_r2=eps_r2,
        mu_r2=mu_r2,
        radius=radius,
        lambda0=lambda0,
        x_pts=x_pts,
        y_pts=y_pts,
        z_pts=z_pts,
        pec_layer_index=0,
    )

    print(f"Wavelength (lambda0): {lambda0:.6f} m")
    print(f"Sphere radius (a): {radius:.6f} m")
    print(f"Size parameter (x): {size_param:.6f}")
    print(f"Reference m passed to fieldnlay: {rel_index}")
    print(f"Terms used: {terms}")
    print(f"\nObservation grid: {grid_res} x {grid_res} points, spanning [{-grid_span:.2f}, {grid_span:.2f}] m")
    print("Calculation complete.")

    data_directory = "./applications/pec_sphere/data"
    output_filename = os.path.join(data_directory, output_basename)
    _save_fields_csv(output_filename, x_pts, z_pts, e_field, h_field)


def main():
    # Adjustable PEC-limit surrogate (kept from the original workflow).
    pec_limit_imag = 10000.0
    calculate_mie_field_pec_limit(
        pec_imag=pec_limit_imag,
        output_basename="eh_total_zx_plane_mie_limit.csv",
    )

    # Exact PEC reference (recommended for final comparison with EFIE PEC solver).
    calculate_mie_field_pec_exact(
        output_basename="eh_total_zx_plane_mie.csv",
    )

if __name__ == "__main__":
    main()