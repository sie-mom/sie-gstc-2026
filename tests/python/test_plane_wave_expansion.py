import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from spherical_expansion.svwf_sphere import get_M_N, compute_fields
from spherical_expansion.solver import evaluate_incident_plane_wave
from spherical_expansion.utils import plot_comparison_2d


def test_svwf_round_trip_expansion():
    """
    Tests the correctness of the SVWF basis and the least-squares solution by
    performing a "round-trip" expansion.

    1. A known "target" field is created from a predefined set of coefficients
       and outgoing SVWFs (kind=3, Hankel functions).
    2. The tangential components of this target field are sampled on the surface
       of a sphere. This becomes the RHS vector 'V' of the linear system.
    3. A system matrix 'Z' is constructed using the same outgoing SVWF basis
       functions evaluated at the sample points.
    4. The system Z*x = V is solved for the unknown coefficients 'x' using
       least squares.
    5. The solved coefficients are compared against the original predefined
       coefficients to verify the accuracy of the process.
    6. The field is reconstructed using the solved coefficients and compared
       to the original target field for visual confirmation.
    """
    # --- 1. Parameters ---
    c0 = 299792458.0
    freq = 1e9
    lambda0 = c0 / freq
    radius = 1.5 * lambda0
    k = 2 * np.pi / lambda0
    eta = 377.0

    # Expansion and sampling parameters
    n_max = 15
    n_theta = 2 * n_max + 1  # Use a sufficient number of points for oversampling
    n_phi = 2 * n_theta

    # --- 2. Define Modes and Unknowns ---
    modes = []
    for n in range(1, n_max + 1):
        for m in range(-n, n + 1):
            modes.append((m, n))

    num_modes = len(modes)
    num_unknowns = 2 * num_modes  # c_nm and d_nm per mode

    col_map = {}
    idx = 0
    for m, n in modes:
        col_map[('c', (m, n))] = idx
        col_map[('d', (m, n))] = idx + 1
        idx += 2

    # --- 3. Create Target Field from Known Coefficients ---
    theta_points = np.linspace(0.001, np.pi - 0.001, n_theta)
    phi_points = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Define a simple, known set of coefficients for a few modes
    c_coeffs_known = {mn: 0.0 for mn in modes}
    d_coeffs_known = {mn: 0.0 for mn in modes}
    c_coeffs_known[(1, 1)] = 1.0 + 0.5j
    d_coeffs_known[(1, 1)] = 0.5 - 0.2j
    c_coeffs_known[(-1, 2)] = -0.3 + 0.8j
    d_coeffs_known[(-1, 2)] = 0.1 + 0.1j

    # Compute the target E-field on the sphere surface using these coefficients
    # We use kind=3 (outgoing waves) as our basis.
    E_target, _ = compute_fields(
        k, radius, theta_points, phi_points,
        c_coeffs_known, d_coeffs_known,
        kind=3, eta=eta, grid=True
    )
    E_target_th, E_target_ph = E_target[1], E_target[2]
    
    # Flatten for the linear system
    V = np.concatenate([E_target_th.flatten(), E_target_ph.flatten()])

    # --- 4. Build System Matrix Z ---
    num_points = n_theta * n_phi
    num_equations = 2 * num_points  # E_theta and E_phi components
    Z = np.zeros((num_equations, num_unknowns), dtype=complex)

    for m, n in modes:
        # Get outgoing SVWFs (kind=3) to build the system matrix
        M, N = get_M_N(n, m, k, radius, theta_points, phi_points, kind=3, grid=True)

        # Tangential components (M_theta, M_phi, N_theta, N_phi)
        M_th, M_ph = M[1].flatten(), M[2].flatten()
        N_th, N_ph = N[1].flatten(), N[2].flatten()

        # Column for c_nm (associated with M)
        idx_c = col_map[('c', (m, n))]
        Z[:num_points, idx_c] = M_th
        Z[num_points:, idx_c] = M_ph

        # Column for d_nm (associated with N)
        idx_d = col_map[('d', (m, n))]
        Z[:num_points, idx_d] = N_th
        Z[num_points:, idx_d] = N_ph

    # --- 5. Solve for Coefficients ---
    # Apply integration weights for a more accurate surface integral representation
    weights_theta = np.sqrt(np.sin(theta_points))
    weights_grid = np.tile(weights_theta[:, None], (1, n_phi)).flatten()
    weights = np.concatenate([weights_grid, weights_grid])

    Z_w = Z * weights[:, None]
    V_w = V * weights

    # --- Column-normalize the matrix for better conditioning ---
    col_norms = np.linalg.norm(Z_w, axis=0)
    col_norms[col_norms == 0] = 1.0 # Avoid division by zero for empty columns
    Z_scaled = Z_w / col_norms[None, :]

    # Solve using least squares
    x_scaled, residuals, rank, s = np.linalg.lstsq(Z_scaled, V_w, rcond=1e-12)

    # Un-scale the solution vector
    x_solved = x_scaled / col_norms

    rel_residual = np.linalg.norm(Z_w @ x_solved - V_w) / np.linalg.norm(V_w) if np.linalg.norm(V_w) > 0 else 0.0
    print(f"Relative residual of least-squares fit: {rel_residual:.3e}")

    # Unpack solved coefficients
    c_coeffs = {mn: x_solved[col_map[('c', mn)]] for mn in modes}
    d_coeffs = {mn: x_solved[col_map[('d', mn)]] for mn in modes}

    # --- 6. Reconstruct and Compare Field ---
    # Define a 1D circle for verification, avoiding the origin.
    # The circle is in the XY plane (theta = pi/2) with a radius different from the sampling sphere.
    num_obs_points = 401
    obs_radius = 0.7 * radius
    
    R_obs = np.full(num_obs_points, obs_radius)
    Theta_obs = np.full(num_obs_points, np.pi / 2.0)
    Phi_obs = np.linspace(0, 2 * np.pi, num_obs_points, endpoint=False)

    # Use the solved coefficients to reconstruct the field
    E_reconstructed, _ = compute_fields(
        k, R_obs, Theta_obs, Phi_obs, c_coeffs, d_coeffs, kind=3, eta=eta, grid=False
    )

    # We can compare any component. Let's use the magnitude for simplicity.
    E_recon_mag = np.sqrt(np.abs(E_reconstructed[0])**2 + np.abs(E_reconstructed[1])**2 + np.abs(E_reconstructed[2])**2)

    # Calculate the original target field on the same grid for comparison
    E_target_grid, _ = compute_fields(
        k, R_obs, Theta_obs, Phi_obs, c_coeffs_known, d_coeffs_known, kind=3, eta=eta, grid=False
    )
    E_target_mag = np.sqrt(np.abs(E_target_grid[0])**2 + np.abs(E_target_grid[1])**2 + np.abs(E_target_grid[2])**2)

    # --- 7. Verification ---
    # Calculate the relative error between the reconstructed and original fields
    error_norm = np.linalg.norm(E_recon_mag - E_target_mag)
    signal_norm = np.linalg.norm(E_target_mag)
    rel_error = error_norm / signal_norm

    print(f"Relative error of reconstructed field magnitude: {rel_error:.3e}")

    # Save a plot for visual inspection (optional)
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots'))
    os.makedirs(plots_dir, exist_ok=True)
    # plot_comparison_1d(
    #     Phi_obs, np.real(E_recon_mag), np.real(E_target_mag),
    #     label1="Reconstructed |E|", label2="Target |E|",
    #     xlabel="Phi (radians)",
    #     title="SVWF Round-Trip Verification (Circular Cut)",
    #     save_path=os.path.join(plots_dir, "svwf_round_trip_test_1d.png")
    # )

    assert rel_error < 1e-9, "Relative error in SVWF round-trip expansion is too high."



if __name__ == "__main__":
    test_svwf_round_trip_expansion()
    # test_plane_wave_expansion_revisited()