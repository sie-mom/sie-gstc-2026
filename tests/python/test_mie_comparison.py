import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from spherical_expansion.solver import solve_collocation, evaluate_incident_plane_wave
from spherical_expansion.svwf_sphere import compute_fields
from spherical_expansion.utils import plot_comparison_1d, spherical_to_cartesian_vector

try:
    from scattnlay import fieldnlay
except ImportError:
    fieldnlay = None

@pytest.mark.skipif(fieldnlay is None, reason="scattnlay not installed")
def test_mie_comparison():
    """
    Compares the GSTC solver (with zero GSTC params) against Mie theory (scattnlay).
    """
    # --- Parameters ---
    c0 = 299792458.0
    freq = 1e9  # 1 GHz
    lambda0 = c0 / freq
    radius = 1.0 * lambda0 # Sphere radius
    
    # Dielectric sphere parameters
    eps_r1 = 2.2  # Inside
    mu_r1 = 1.0
    eps_r2 = 1.0  # Outside (Air)
    mu_r2 = 1.0
    
    k0 = 2 * np.pi * freq / c0 * np.sqrt(eps_r2 * mu_r2) # Exterior medium
    k1 = 2 * np.pi * freq / c0 * np.sqrt(eps_r1 * mu_r1) # Interior medium

    eps0 = 8.854e-12
    mu0 = np.pi * 4e-7
    eta_vacuum = np.sqrt(mu0 / eps0)
    eta0 = eta_vacuum * np.sqrt(mu_r2 / eps_r2) # Exterior eta
    eta1 = eta_vacuum * np.sqrt(mu_r1 / eps_r1) # Interior eta
    
    # GSTC parameters (Zero for standard dielectric interface)
    gstc_params = {
        'Y': 0.0,
        'Chi': 0.0,
        'Gamma': 0.0,
        'Z': 0.0
    }
    
    # Solver settings
    n_max = 15 # Reduced from 20 to avoid overfitting/conditioning issues at low frequency
    n_theta = 40
    n_phi = 40
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
    
    # Debug: Print max coefficient magnitudes
    max_a = max([abs(v) for v in a_coeffs.values()]) if a_coeffs else 0
    max_b = max([abs(v) for v in b_coeffs.values()]) if b_coeffs else 0
    max_c = max([abs(v) for v in c_coeffs.values()]) if c_coeffs else 0
    max_d = max([abs(v) for v in d_coeffs.values()]) if d_coeffs else 0
    
    print(f"Max coeff magnitudes: a={max_a:.4e}, b={max_b:.4e}, c={max_c:.4e}, d={max_d:.4e}")
    print(
        "Solver residuals: scaled {:.3e} (rel {:.3e}), orig {:.3e} (rel {:.3e})".format(
            solver_metrics['residual_scaled_norm'],
            solver_metrics['residual_scaled_rel'],
            solver_metrics['residual_norm'],
            solver_metrics['residual_rel'],
        )
    )
    if 'bc_H_rms' in solver_metrics:
        print(
            "BC residuals: H_rms {:.3e} (rel {:.3e}), E_rms {:.3e} (rel {:.3e})".format(
                solver_metrics['bc_H_rms'],
                solver_metrics['bc_H_rel_rms'],
                solver_metrics['bc_E_rms'],
                solver_metrics['bc_E_rel_rms'],
            )
        )
    
    # --- Define Observation Points (1D cut along x-axis) ---
    # We'll look at points inside and outside
    num_obs = 100
    x_obs = np.linspace(-2*radius, 2*radius, num_obs)
    y_obs = np.zeros_like(x_obs)
    z_obs = np.zeros_like(x_obs)
    
    # Convert to spherical coordinates for our solver
    r_obs = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
    theta_obs = np.arccos(z_obs / np.where(r_obs==0, 1, r_obs))
    phi_obs = np.arctan2(y_obs, x_obs)
    
    # Handle origin singularity if any
    r_obs[r_obs < 1e-9] = 1e-9
    
    # --- Compute Fields using GSTC Solver ---
    E_gstc_mag = np.zeros(num_obs)
    E_gstc = np.zeros((num_obs, 3), dtype=complex)
    E_inc_vec = np.zeros((num_obs, 3), dtype=complex)
    
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
        Ax, Ay, Az = spherical_to_cartesian_vector(E_in_vec[0], E_in_vec[1], E_in_vec[2], theta_obs[mask_int], phi_obs[mask_int])
        E_in_cart = np.stack([Ax, Ay, Az], axis=-1)
        E_gstc[mask_int] = E_in_cart
        E_gstc_mag[mask_int] = np.linalg.norm(E_in_cart, axis=1)
        
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
        
        # Incident field (Plane wave x-polarized, propagating +z)
        E0 = 1.0
        phase = np.exp(1j * k0 * r_obs[mask_ext] * np.cos(theta_obs[mask_ext]))
        
        # Cartesian Incident Field
        Ex_inc = E0 * phase
        Ey_inc = np.zeros_like(phase)
        Ez_inc = np.zeros_like(phase)
        
        # Store incident field
        E_inc_vec[mask_ext] = np.stack([Ex_inc, Ey_inc, Ez_inc], axis=1)
        
        # Convert Scattered to Cartesian
        Ax_sc, Ay_sc, Az_sc = spherical_to_cartesian_vector(E_sc_vec[0], E_sc_vec[1], E_sc_vec[2], theta_obs[mask_ext], phi_obs[mask_ext])
        E_sc_cart = np.stack([Ax_sc, Ay_sc, Az_sc], axis=-1)
        
        # Total Field
        E_tot_cart = E_sc_cart + E_inc_vec[mask_ext]
        
        E_gstc[mask_ext] = E_tot_cart
        E_gstc_mag[mask_ext] = np.linalg.norm(E_tot_cart, axis=1)

    # --- Run Mie (scattnlay) ---
    print("Running scattnlay...")
    n_med = np.sqrt(eps_r2 * mu_r2)
    n_sph = np.sqrt(eps_r1 * mu_r1)
    
    x_size = np.array([2 * np.pi * n_med * radius / lambda0])
    m_idx = np.array([n_sph / n_med])
    
    scaling_factor = (2 * np.pi * n_med) / lambda0
    xp = x_obs * scaling_factor
    yp = y_obs * scaling_factor
    zp = z_obs * scaling_factor
    
    terms, E_mie, H_mie = fieldnlay(x_size, m_idx, xp, yp, zp)
    
    # E_mie is (N, 3) array of (Ex, Ey, Ez)
    E_mie_mag = np.linalg.norm(E_mie, axis=1)
    
    # --- Compare ---
    print("Comparing results...")
    
    # Plot
    plot_comparison_1d(
        x_obs, 
        E_gstc_mag, 
        E_mie_mag, 
        label1="GSTC (Zero Params)", 
        label2="Mie (scattnlay)",
        xlabel="x (m)",
        ylabel="|E|",
        title="Comparison of Electric Field Magnitude",
        save_path="./tests/python/plots/comparison_mie_solver.png"
    )
    
    # Calculate error
    # Ignore points exactly on boundary where discontinuity might cause issues in sampling
    mask_safe = np.abs(r_obs - radius) > 0.05 * radius
    
    # Separate errors
    mask_safe_int = mask_safe & (r_obs < radius)
    mask_safe_ext = mask_safe & (r_obs > radius)
    
    # Debug: Print field values at a few points
    print("\n--- Debug Field Values ---")
    indices = [0, len(r_obs)//2, -1] # Center (approx), Surface/Interface, Far
    for idx in indices:
        r_val = r_obs[idx]
        th_val = theta_obs[idx]
        ph_val = phi_obs[idx]
        print(f"Point {idx}: r={r_val:.3e}, theta={th_val:.3f}, phi={ph_val:.3f}")
        print(f"  GSTC E: {E_gstc[idx]}")
        print(f"  Mie  E: {E_mie[idx]}")
        print(f"  Inc  E: {E_inc_vec[idx]}")
        print(f"  Diff E: {np.linalg.norm(E_gstc[idx] - E_mie[idx])}")
        
    # Compute error metrics
    rel_error_int = np.linalg.norm(E_gstc_mag[mask_safe_int] - E_mie_mag[mask_safe_int]) / np.linalg.norm(E_mie_mag[mask_safe_int])
    rel_error_ext = np.linalg.norm(E_gstc_mag[mask_safe_ext] - E_mie_mag[mask_safe_ext]) / np.linalg.norm(E_mie_mag[mask_safe_ext])
    
    rel_error = np.linalg.norm(E_gstc_mag[mask_safe] - E_mie_mag[mask_safe]) / np.linalg.norm(E_mie_mag[mask_safe])
    
    print(f"Relative Error (Total): {rel_error:.4e}")
    print(f"Relative Error (Interior): {rel_error_int:.4e}")
    print(f"Relative Error (Exterior): {rel_error_ext:.4e}")
    
    assert rel_error < 0.05, f"Relative error {rel_error} is too high!"

if __name__ == "__main__":
    test_mie_comparison()
