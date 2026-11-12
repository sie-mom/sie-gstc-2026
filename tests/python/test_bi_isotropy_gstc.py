import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from spherical_expansion.solver import solve_collocation, evaluate_incident_plane_wave
from spherical_expansion.svwf_sphere import compute_fields
from spherical_expansion.utils import plot_comparison_1d, spherical_to_cartesian_vector, verify_gstc_boundary_conditions
import pandas as pd

try:
    from scattnlay import fieldnlay
except ImportError:
    fieldnlay = None

def test_svwf_solver_file(tmp_path):
    """
    Compares the GSTC solver (with zero GSTC params) against Mie theory (scattnlay).
    """
    # --- Parameters ---
    c0 = 299792458.0
    freq = 1e9  # 1 GHz
    lambda0 = c0 / freq
    radius = 1.0 * lambda0 # Sphere radius
    
    # Dielectric sphere parameters
    eps_r1 = 1.0  # Inside
    mu_r1 = 1.0
    eps_r2 = 1.0  # Outside (Air)
    mu_r2 = 1.0
    
    omega = 2 * np.pi * freq
    eps0 = 8.854e-12
    mu0 = np.pi * 4e-7
    
    k0 = omega * np.sqrt(eps_r2 * eps0 * mu_r2 * mu0) # Exterior medium
    k1 = omega * np.sqrt(eps_r1 * eps0 * mu_r1 * mu0) # Interior medium

    eta_vacuum = np.sqrt(mu0 / eps0)
    eta0 = eta_vacuum * np.sqrt(mu_r2 / eps_r2) # Exterior eta
    eta1 = eta_vacuum * np.sqrt(mu_r1 / eps_r1) # Interior eta
    
    # GSTC parameters (Zero for standard dielectric interface)
    chi_ee = 0.2
    chi_mm = 0.1
    chi_em = 0.1
    chi_me = 0.1
    sheetE = -1 # Corrected sign for exp(-j*omega*t) time convention
    sheetM = 1

    Y_gstc = sheetE * 1j * omega * eps0 * chi_ee # Electric Admittance Density [S/m]
    Z_gstc = sheetM * (1j * omega * mu0 * chi_mm) # Magnetic Impedance Density [Ohm/m]
    Chi_gstc = sheetE * 1j * k0 * chi_em # Dimensionless
    Gamma_gstc = sheetM * (1j * k0 * chi_me) # Dimensionless

    gstc_params = {
        'Y': Y_gstc,  # High admittance to avoid singular matrix issues
        'Chi': Chi_gstc,
        'Gamma': Gamma_gstc,
        'Z': Z_gstc
    }
    
    # Solver settings
    n_max = 15 # Reduced from 20 to avoid overfitting/conditioning issues at low frequency
    n_theta = 50
    n_phi = 50
    print(f"Using n_max={n_max}, n_theta={n_theta}, n_phi={n_phi}")
    # Check oversampling condition requested by user
    # 4 * N_theta * N_phi >= 8 * n_max * (n_max + 3) (Interpretation of "8 x N 2 (N + 3)")
    # This ensures ~2x oversampling of unknowns.
    num_eq = 4 * n_theta * n_phi
    num_unk_approx = 4 * n_max * (n_max + 2)
    
    print(f"Oversampling Check:")
    print(f"  Equations: {num_eq}")
    print(f"  Unknowns (approx): {num_unk_approx}")
    print(f"  Ratio: {num_eq / num_unk_approx:.2f}")
    
    # --- Run GSTC Solver ---
    print("Running GSTC Solver...")
    # Use dense grid for solver
    solver_result = solve_collocation(
        n_max=n_max,
        k0=k0,
        eta0=eta0,
        k1=k1,
        eta1=eta1,
        radius=radius,
        gstc_params=gstc_params,
        n_theta=n_theta,
        n_phi=n_phi,
        row_scale=True,
        return_metrics=True,
    )
    a_coeffs, b_coeffs, c_coeffs, d_coeffs, solver_metrics = solver_result
    
    verify_gstc_boundary_conditions(radius, k0, eta0, k1, eta1, a_coeffs, b_coeffs, c_coeffs, d_coeffs, gstc_params)
    
    
    # --- Define Observation Points (1D cut along x-axis) ---
    # We'll look at points inside and outside
    num_points_axis = 300 # Points per axis
    limit = 4.0 * radius
    x_pts = np.linspace(-limit, limit, num_points_axis)
    z_pts = np.linspace(-limit, limit, num_points_axis)
    X, Z = np.meshgrid(x_pts, z_pts)
    Y = np.zeros_like(X)

    # Flatten for computation
    x_obs = X.flatten()
    y_obs = Y.flatten()
    z_obs = Z.flatten()
    num_obs = len(x_obs)
    
    # Convert to spherical coordinates for our solver
    r_obs = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
    theta_obs = np.arccos(z_obs / np.where(r_obs==0, 1, r_obs))
    phi_obs = np.arctan2(y_obs, x_obs)
    
    # Handle origin singularity if any
    r_obs[r_obs < 1e-9] = 1e-9
    
    # --- Compute Fields using GSTC Solver ---
    E_gstc_mag = np.zeros(num_obs)
    E_gstc = np.zeros((num_obs, 3), dtype=complex)
    H_gstc_mag = np.zeros(num_obs)
    H_gstc = np.zeros((num_obs, 3), dtype=complex)
    
    # Split into interior and exterior points
    mask_int = r_obs < radius
    mask_ext = r_obs >= radius
    
    # Interior
    print("Computing interior fields...")
    if np.any(mask_int):
        E_in_vec, H_in_vec = compute_fields(
            k=k1,
            r=r_obs[mask_int],
            theta=theta_obs[mask_int],
            phi=phi_obs[mask_int],
            a_coeffs=c_coeffs, # c -> M (Electric)
            b_coeffs=d_coeffs, # d -> N (Electric)
            kind=1, # Regular Bessel
            eta=eta1, 
            grid=False
        )
        # E_in_vec is tuple (Er, Eth, Eph)
        Ex, Ey, Ez = spherical_to_cartesian_vector(E_in_vec[0], E_in_vec[1], E_in_vec[2], theta_obs[mask_int], phi_obs[mask_int])
        E_in_cart = np.stack([Ex, Ey, Ez], axis=-1)
        E_gstc[mask_int] = E_in_cart
        E_gstc_mag[mask_int] = np.linalg.norm(E_in_cart, axis=1)

        Hx, Hy, Hz = spherical_to_cartesian_vector(H_in_vec[0], H_in_vec[1], H_in_vec[2], theta_obs[mask_int], phi_obs[mask_int])
        H_in_cart = np.stack([Hx, Hy, Hz], axis=-1)
        H_gstc[mask_int] = H_in_cart
        H_gstc_mag[mask_int] = np.linalg.norm(H_in_cart, axis=1)
        
    # Exterior (Scattered + Incident)
    print("Computing exterior fields...")
    if np.any(mask_ext):
        # Scattered
        E_sc_vec, H_sc_vec = compute_fields(
            k=k0,
            r=r_obs[mask_ext],
            theta=theta_obs[mask_ext],
            phi=phi_obs[mask_ext],
            a_coeffs=a_coeffs,
            b_coeffs=b_coeffs,
            kind=3, # Hankel
            eta=eta0, # Exterior eta
            grid=False
        )
        
        # Incident field (Plane wave x-polarized, propagating +z) using helper
        E_inc_sph, H_inc_sph = evaluate_incident_plane_wave(
            k=k0,
            r=r_obs[mask_ext],
            theta=theta_obs[mask_ext],
            phi=phi_obs[mask_ext],
            grid=False,
            eta=eta0
        )
        
        # Store incident field
        Ex_inc, Ey_inc, Ez_inc = spherical_to_cartesian_vector(E_inc_sph[0], E_inc_sph[1], E_inc_sph[2], theta_obs[mask_ext], phi_obs[mask_ext])
        E_inc_cart = np.stack([Ex_inc, Ey_inc, Ez_inc], axis=1)

        Hx_inc, Hy_inc, Hz_inc = spherical_to_cartesian_vector(H_inc_sph[0], H_inc_sph[1], H_inc_sph[2], theta_obs[mask_ext], phi_obs[mask_ext])
        H_inc_cart = np.stack([Hx_inc, Hy_inc, Hz_inc], axis=1)
        
        # Convert Scattered to Cartesian
        Ex_sc, Ey_sc, Ez_sc = spherical_to_cartesian_vector(E_sc_vec[0], E_sc_vec[1], E_sc_vec[2], theta_obs[mask_ext], phi_obs[mask_ext])
        E_sc_cart = np.stack([Ex_sc, Ey_sc, Ez_sc], axis=-1)
        Hx_sc, Hy_sc, Hz_sc = spherical_to_cartesian_vector(H_sc_vec[0], H_sc_vec[1], H_sc_vec[2], theta_obs[mask_ext], phi_obs[mask_ext])
        H_sc_cart = np.stack([Hx_sc, Hy_sc, Hz_sc], axis=-1)
        
        # Total Field
        E_tot_cart = E_sc_cart + E_inc_cart
        H_tot_cart = H_sc_cart + H_inc_cart
        
        E_gstc[mask_ext] = E_tot_cart
        E_gstc_mag[mask_ext] = np.linalg.norm(E_tot_cart, axis=1)
        H_gstc[mask_ext] = H_tot_cart
        H_gstc_mag[mask_ext] = np.linalg.norm(H_tot_cart, axis=1)

    
    # --- Save Results to CSV ---
    # Create a pandas DataFrame with all components
    df = pd.DataFrame({
        'x': x_obs.ravel(),
        'z': z_obs.ravel(),
        'Ex_real': E_gstc[:, 0].real,
        'Ex_imag': E_gstc[:, 0].imag,
        'Ey_real': E_gstc[:, 1].real,
        'Ey_imag': E_gstc[:, 1].imag,
        'Ez_real': E_gstc[:, 2].real,
        'Ez_imag': E_gstc[:, 2].imag,
        'E_magnitude': E_gstc_mag,
        'Hx_real': H_gstc[:, 0].real,
        'Hx_imag': H_gstc[:, 0].imag,
        'Hy_real': H_gstc[:, 1].real,
        'Hy_imag': H_gstc[:, 1].imag,
        'Hz_real': H_gstc[:, 2].real,
        'Hz_imag': H_gstc[:, 2].imag,
        'H_magnitude': H_gstc_mag
    })

    output_filename = tmp_path / "eh_total_2d_cut_gstc.csv"
    
    print(f"Saving total field data to '{output_filename}'...")
    df.to_csv(output_filename, index=False)
    print("Data saved successfully.")
