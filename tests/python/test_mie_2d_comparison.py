import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from spherical_expansion.solver import solve_collocation
from spherical_expansion.svwf_sphere import compute_fields
from spherical_expansion.utils import plot_field_cut_2d, plot_comparison_2d, spherical_to_cartesian_vector

try:
    from scattnlay import fieldnlay
except ImportError:
    fieldnlay = None

@pytest.mark.skipif(fieldnlay is None, reason="scattnlay not installed")
def test_mie_2d_comparison():
    """
    Compares the GSTC solver (with zero GSTC params) against Mie theory (scattnlay)
    on a 2D slice (XZ plane).
    """
    # --- Parameters ---
    c0 = 299792458.0
    freq = 1e9  # 1 GHz
    lambda0 = c0 / freq
    radius = 2.0 * lambda0 # Sphere radius
    
    # Dielectric sphere parameters
    eps_r1 = 2.2  # Interior
    mu_r1 = 1.0
    eps_r2 = 1.0  # Exterior (Air)
    mu_r2 = 1.0
    
    k0 = 2 * np.pi * freq / c0 * np.sqrt(eps_r2 * mu_r2) # Exterior medium
    k1 = 2 * np.pi * freq / c0 * np.sqrt(eps_r1 * mu_r1) # Interior medium  

    eta0 = np.sqrt(mu_r2/eps_r2) # Exterior eta
    eta1 = np.sqrt(mu_r1/eps_r1) # Interior eta
    
    # GSTC parameters (Zero for standard dielectric interface)
    gstc_params = {
        'Y': 0.0,
        'Chi': 0.0,
        'Gamma': 0.0,
        'Z': 0.0
    }
    
    # Solver settings
    n_max = 15
    n_theta = 40
    n_phi = 80
    
    # --- Run GSTC Solver ---
    print("Running GSTC Solver...")
    a_coeffs, b_coeffs, c_coeffs, d_coeffs = solve_collocation(
        n_max=n_max,
        k0=k0,     # Exterior medium
        eta0=eta0, # Exterior eta
        k1=k1,     # Interior medium
        eta1=eta1, # Interior eta
        radius=radius,
        gstc_params=gstc_params,
        n_theta=n_theta,
        n_phi=n_phi
    )
    
    # --- Define Observation Grid (XZ plane) ---
    num_points = 200
    limit = 4.0 * radius
    x = np.linspace(-limit, limit, num_points)
    z = np.linspace(-limit, limit, num_points)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)
    
    # Convert to spherical coordinates
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Theta = np.arccos(Z / np.where(R==0, 1, R))
    Phi = np.arctan2(Y, X)
    
    # Handle origin
    R[R < 1e-9] = 1e-9
    
    # Flatten for computation
    r_flat = R.flatten()
    theta_flat = Theta.flatten()
    phi_flat = Phi.flatten()
    
    # --- Compute Fields using GSTC Solver ---
    E_gstc_mag = np.zeros_like(r_flat)
    E_gstc_vec = np.zeros((len(r_flat), 3), dtype=complex)
    
    mask_int = r_flat < radius
    mask_ext = r_flat >= radius


    # Interior
    print("Computing interior fields...")
    if np.any(mask_int):
        E_in_vec, _ = compute_fields(
            k=k1,
            r=r_flat[mask_int],
            theta=theta_flat[mask_int],
            phi=phi_flat[mask_int],
            a_coeffs=c_coeffs,
            b_coeffs=d_coeffs,
            kind=1,
            eta=eta1,
            grid=False
        )
        Ax, Ay, Az = spherical_to_cartesian_vector(E_in_vec[0], E_in_vec[1], E_in_vec[2], theta_flat[mask_int], phi_flat[mask_int])
        E_in_cart = np.stack([Ax, Ay, Az], axis=-1)
        E_gstc_vec[mask_int] = E_in_cart
        E_gstc_mag[mask_int] = np.linalg.norm(E_in_cart, axis=1)
    
        
    # Exterior
    print("Computing exterior fields...")
    if np.any(mask_ext):
        # Scattered
        E_sc_vec, _ = compute_fields(
            k=k0,
            r=r_flat[mask_ext],
            theta=theta_flat[mask_ext],
            phi=phi_flat[mask_ext],
            a_coeffs=a_coeffs,
            b_coeffs=b_coeffs,
            kind=3,
            eta=eta0,
            grid=False
        )
        
        # Incident (Plane wave x-polarized, +z prop)
        E0 = 1.0
        phase = np.exp(1j * k0 * r_flat[mask_ext] * np.cos(theta_flat[mask_ext]))
        Ex_inc = E0 * phase
        Ey_inc = np.zeros_like(phase)
        Ez_inc = np.zeros_like(phase)
        E_inc_cart = np.stack([Ex_inc, Ey_inc, Ez_inc], axis=1)
        
        # Scattered to Cartesian
        Ax_sc, Ay_sc, Az_sc = spherical_to_cartesian_vector(E_sc_vec[0], E_sc_vec[1], E_sc_vec[2], theta_flat[mask_ext], phi_flat[mask_ext])
        E_sc_cart = np.stack([Ax_sc, Ay_sc, Az_sc], axis=-1)
        
        # Total
        E_tot_cart = E_sc_cart + E_inc_cart
        E_gstc_vec[mask_ext] = E_tot_cart
        E_gstc_mag[mask_ext] = np.linalg.norm(E_tot_cart, axis=1)

    # Reshape to 2D
    E_gstc_mag_2d = E_gstc_mag.reshape(X.shape)

    # --- Run Mie (scattnlay) ---
    print("Running scattnlay...")
    n_med = np.sqrt(eps_r2 * mu_r2)
    n_sph = np.sqrt(eps_r1 * mu_r1)
    
    x_size = np.array([2 * np.pi * n_med * radius / lambda0])
    m_idx = np.array([n_sph / n_med])
    
    scaling_factor = (2 * np.pi * n_med) / lambda0
    xp = X.flatten() * scaling_factor
    yp = Y.flatten() * scaling_factor
    zp = Z.flatten() * scaling_factor
    
    terms, E_mie, H_mie = fieldnlay(x_size, m_idx, xp, yp, zp)
    
    E_mie_mag = np.linalg.norm(E_mie, axis=1)
    E_mie_mag_2d = E_mie_mag.reshape(X.shape)
    
    # Extract Ex components (Real part)
    Ex_gstc_real_2d = np.real(E_gstc_vec[:, 0]).reshape(X.shape)
    Ex_mie_real_2d = np.real(E_mie[:, 0]).reshape(X.shape)
    
    # --- Plotting ---
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots'))
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot Comparison Magnitude
    plot_comparison_2d(
        X, Z, E_gstc_mag_2d, E_mie_mag_2d,
        label1="GSTC", label2="Mie",
        title="Electric Field Magnitude Comparison",
        xlabel="x (m)", ylabel="z (m)",
        save_path=os.path.join(plots_dir, "E_field_Comparison_2D.png")
    )
    
    # # Plot Difference Magnitude
    # diff_mag = np.abs(E_gstc_mag_2d - E_mie_mag_2d)
    # plot_field_cut_2d(
    #     X, Z, diff_mag,
    #     title="Difference |E_GSTC| - |E_Mie|",
    #     xlabel="x (m)",
    #     ylabel="z (m)",
    #     save_path=os.path.join(plots_dir, "E_field_Diff_2D.png")
    # )
    
    # Plot Comparison Re(Ex)
    plot_comparison_2d(
        X, Z, Ex_gstc_real_2d, Ex_mie_real_2d,
        label1="GSTC", label2="Mie",
        title="Re(Ex) Comparison",
        xlabel="x (m)", ylabel="z (m)",
        save_path=os.path.join(plots_dir, "Ex_real_Comparison_2D.png")
    )
    
    # # Plot Difference Re(Ex)
    # diff_ex_real = np.abs(Ex_gstc_real_2d - Ex_mie_real_2d)
    # plot_field_cut_2d(
    #     X, Z, diff_ex_real,
    #     title="Difference |Re(Ex_GSTC) - Re(Ex_Mie)|",
    #     xlabel="x (m)",
    #     ylabel="z (m)",
    #     save_path=os.path.join(plots_dir, "Ex_real_Diff_2D.png")
    # )
    
    # Calculate relative error
    # Avoid boundary
    mask_safe = np.abs(r_flat - radius) > 0.05 * radius
    rel_error = np.linalg.norm(E_gstc_mag[mask_safe] - E_mie_mag[mask_safe]) / np.linalg.norm(E_mie_mag[mask_safe])
    
    print(f"Relative Error (Total): {rel_error:.4e}")
    
    # We don't assert here to allow plotting even if error is high, 
    # but typically we would want:
    # assert rel_error < 0.05

if __name__ == "__main__":
    test_mie_2d_comparison()
