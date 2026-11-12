import os
import matplotlib.pyplot as plt
import math, cmath
from spherical_expansion.utils import spherical_to_cartesian_vector
import numpy as np
from typing import List, Tuple

try:
    from scipy.special import sph_harm_y, spherical_jn
    _HAVE_SPH_HARM_Y = True
except Exception:
    from scipy.special import sph_harm as _sph_harm, spherical_jn
    _HAVE_SPH_HARM_Y = False

def Y_nm(n: int, m: int, theta: float, phi: float) -> complex:
    if _HAVE_SPH_HARM_Y:
        return sph_harm_y(n, m, theta, phi)
    else:
        return _sph_harm(m, n, phi, theta)

def spherical_jn_all(nmax: int, x: float) -> np.ndarray:
    return np.array([spherical_jn(n, x) for n in range(nmax+1)], dtype=complex)

def psi_prime_from_j(n: int, x: float, jvals: np.ndarray) -> complex:
    if n == 0:
        return math.cos(x)
    return x*jvals[n-1] - n*jvals[n]

def X_nm_components(n: int, m: int, theta: float, phi: float, dtheta: float = 1e-6) -> Tuple[complex, complex]:
    # Xθ =  i/√(n(n+1)) * (1/sinθ) ∂φY,  Xφ = -i/√(n(n+1)) ∂θY  (degree-first (n,m))
    if n == 0: return 0.0j, 0.0j
    Y = Y_nm(n, m, theta, phi)
    dY_dphi = 1j*m*Y
    th1 = np.minimum(np.maximum(theta - dtheta, 1e-8), math.pi-1e-8)
    th2 = np.minimum(np.maximum(theta + dtheta, 1e-8), math.pi-1e-8)
    Y1 = Y_nm(n, m, th1, phi); Y2 = Y_nm(n, m, th2, phi)
    dY_dtheta = (Y2 - Y1) / (th2 - th1)
    st = np.maximum(1e-14, np.sin(theta))
    fac = 1.0 / math.sqrt(n*(n+1))
    Xth =  1j*fac*(dY_dphi/st)
    Xph = -1j*fac*dY_dtheta
    return Xth, Xph

def M_nm_components(n: int, m: int, k: float, r: float, theta: float, phi: float, jvals: np.ndarray = None) -> Tuple[complex, complex, complex]:
    if jvals is None: jvals = spherical_jn_all(n, k*r)
    Xth, Xph = X_nm_components(n, m, theta, phi)
    jl = jvals[n]
    Er = 0.0j; Eth = jl*Xth; Eph = jl*Xph
    return Er, Eth, Eph

def N_nm_components(n: int, m: int, k: float, r: float, theta: float, phi: float, jvals: np.ndarray = None, h: float = 1e-6) -> Tuple[complex, complex, complex]:
    # N = (1/k) ∇×M with M_r=0, M_θ=j_l Xθ, M_φ=j_l Xφ
    if jvals is None: jvals = spherical_jn_all(n, k*r)
    Xth, Xph = X_nm_components(n, m, theta, phi)
    jl = jvals[n]; x = k*r
    st = np.maximum(1e-12, np.sin(theta))
    # curl_r numerically (angular)
    th1 = np.minimum(np.maximum(theta - h, 1e-6), math.pi-1e-6)
    th2 = np.minimum(np.maximum(theta + h, 1e-6), math.pi-1e-6)
    Xth1, Xph1 = X_nm_components(n, m, th1, phi); Xth2, Xph2 = X_nm_components(n, m, th2, phi)
    Aphi1 = jvals[n]*Xph1; Aphi2 = jvals[n]*Xph2
    # Vectorized guard against division by zero in the numerical derivative.
    delta_th = th2 - th1
    # Use np.divide with a `where` clause to avoid 0/0 warnings.
    # Where delta_th is near zero, the numerator is also zero. The limit is finite.
    numerator = np.sin(th2)*Aphi2 - np.sin(th1)*Aphi1
    d_sinAphi_dth = np.divide(numerator, delta_th, out=np.zeros_like(numerator), where=np.abs(delta_th) > 1e-15)
    hphi = 1e-6
    Xth_p, _ = X_nm_components(n, m, theta, phi + hphi)
    Xth_m, _ = X_nm_components(n, m, theta, phi - hphi)
    Ath_p = jvals[n]*Xth_p; Ath_m = jvals[n]*Xth_m
    dAth_dphi = (Ath_p - Ath_m)/(2*hphi)
    # The 1/sin(theta) term is numerically unstable at the poles. Use np.divide to handle safely.
    # The numerator should also go to zero, so the limit is finite.
    # Also handle r=0 in the denominator.
    numerator_curl_r = d_sinAphi_dth - dAth_dphi
    curl_r = np.divide(numerator_curl_r, (r*st), out=np.zeros_like(numerator_curl_r), where=(r > 1e-9) & (st > 1e-9))
    # tangential via analytic radial derivative: ∂/∂r (r j_l) = ψ'_l(x)
    psi_p = psi_prime_from_j(n, x, jvals)
    curl_th = -np.divide(1.0, r, out=np.zeros_like(r), where=r>1e-9) * psi_p * Xph
    curl_ph =  np.divide(1.0, r, out=np.zeros_like(r), where=r>1e-9) * psi_p * Xth
    Er = curl_r / k; Eth = curl_th / k; Eph = curl_ph / k
    return Er, Eth, Eph

def project_coeffs_x(nmax: int, k: float, r_fit: float = 1.0, n_th: int = 40, n_ph: int = 80):
    """
    Numerically compute coefficients {c_j} such that
      x̂ e^{ikz} ≈ Σ_j c_j * B_j(r,θ,φ)
    where B_j runs over {M_{nm}^{(1)}, N_{nm}^{(1)}} with 1<=n<=nmax, -n<=m<=n.
    We perform a weighted least-squares on the sphere r=r_fit with rows scaled by sqrt(sinθ).
    Returns (coeffs, basis), where basis is a list of ("M"/"N", n, m).
    """
    thetas = np.linspace(1e-3, math.pi-1e-3, n_th)
    phis   = np.linspace(0, 2*math.pi, n_ph, endpoint=False)
    basis = [(kind, n, m) for n in range(1, nmax+1) for m in range(-n, n+1) for kind in ("M","N")]
    rows = []; rhs = []
    jvals = spherical_jn_all(nmax, k*r_fit)
    for th in thetas:
        w = math.sqrt(math.sin(th)) # sphere geometric area compensation
        for ph in phis:
            Ex_t, Ey_t, Ez_t = cmath.exp(1j*k*r_fit*math.cos(th)), 0.0j, 0.0j
            cols = []
            for kind, n, m in basis:
                if kind == "M":
                    Er, Eth, Eph = M_nm_components(n, m, k, r_fit, th, ph, jvals=jvals)
                else:
                    Er, Eth, Eph = N_nm_components(n, m, k, r_fit, th, ph, jvals=jvals)
                ex, ey, ez = spherical_to_cartesian_vector(Er, Eth, Eph, th, ph)
                cols.append((ex, ey, ez))
            rows.append([w*c[0] for c in cols]); rhs.append(w*Ex_t)
            rows.append([w*c[1] for c in cols]); rhs.append(w*Ey_t)
            rows.append([w*c[2] for c in cols]); rhs.append(w*Ez_t)
    A = np.array(rows, dtype=complex); f = np.array(rhs, dtype=complex)
    c, *_ = np.linalg.lstsq(A, f, rcond=1e-10)
    return c, basis

def project_coeffs_x_vectorized(nmax: int, k: float, r_fit: float = 1.0, n_th: int = 40, n_ph: int = 80):
    """
    Vectorized version of project_coeffs_x.
    Numerically compute coefficients {c_j} such that
      x̂ e^{ikz} ≈ Σ_j c_j * B_j(r,θ,φ)
    where B_j runs over {M_{nm}^{(1)}, N_{nm}^{(1)}} with 1<=n<=nmax, -n<=m<=n.
    We perform a weighted least-squares on the sphere r=r_fit with rows scaled by sqrt(sinθ).
    Returns (coeffs, basis), where basis is a list of ("M"/"N", n, m).
    """
    thetas = np.linspace(1e-3, math.pi-1e-3, n_th)
    phis   = np.linspace(0, 2*math.pi, n_ph, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')

    basis = [(kind, n, m) for n in range(1, nmax+1) for m in range(-n, n+1) for kind in ("M","N")]
    num_basis = len(basis)
    num_points = n_th * n_ph

    # --- Assemble RHS vector 'f' ---
    Ex_t = np.exp(1j*k*r_fit*np.cos(TH)).flatten()
    Ey_t = np.zeros(num_points, dtype=complex)
    Ez_t = np.zeros(num_points, dtype=complex)
    f = np.concatenate([Ex_t, Ey_t, Ez_t])

    # --- Assemble Matrix 'A' ---
    A = np.zeros((3 * num_points, num_basis), dtype=complex)
    jvals = spherical_jn_all(nmax, k*r_fit)

    for j, (kind, n, m) in enumerate(basis):
        # Evaluate basis functions over the entire grid
        if kind == "M":
            Er, Eth, Eph = M_nm_components(n, m, k, r_fit, TH, PH, jvals=jvals)
        else:
            Er, Eth, Eph = N_nm_components(n, m, k, r_fit, TH, PH, jvals=jvals)
        
        # Convert to Cartesian over the grid
        Ex, Ey, Ez = spherical_to_cartesian_vector(Er, Eth, Eph, TH, PH)
        
        # Place flattened results into the correct column of A
        A[0*num_points : 1*num_points, j] = Ex.flatten()
        A[1*num_points : 2*num_points, j] = Ey.flatten()
        A[2*num_points : 3*num_points, j] = Ez.flatten()

    # Apply geometric weighting to rows
    w = np.sqrt(np.sin(TH)).flatten()
    W = np.concatenate([w, w, w])
    A = A * W[:, None]
    f = f * W

    c, *_ = np.linalg.lstsq(A, f, rcond=1e-10)
    return c, basis

def evaluate_from_coeffs(c: np.ndarray, basis: list, k: float, r: float, theta: float, phi: float, nmax: int):
    jvals = spherical_jn_all(nmax, k*r)
    Er = Eth = Eph = 0.0j
    for idx, (kind, n, m) in enumerate(basis):
        if kind == "M":
            ErB, EthB, EphB = M_nm_components(n, m, k, r, theta, phi, jvals=jvals)
        else:
            ErB, EthB, EphB = N_nm_components(n, m, k, r, theta, phi, jvals=jvals)
        Er += c[idx]*ErB; Eth += c[idx]*EthB; Eph += c[idx]*EphB
    return spherical_to_cartesian_vector(Er, Eth, Eph, theta, phi)

def evaluate_from_coeffs_vectorized(c: np.ndarray, basis: list, k: float, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, nmax: int):
    """Vectorized version of evaluate_from_coeffs."""
    shape = r.shape
    Er_total = np.zeros(shape, dtype=complex)
    Eth_total = np.zeros(shape, dtype=complex)
    Eph_total = np.zeros(shape, dtype=complex)

    for idx, (kind, n, m) in enumerate(basis):
        # jvals needs to be calculated for each r, which is now an array.
        jvals = spherical_jn_all(nmax, k * r)
        if kind == "M":
            ErB, EthB, EphB = M_nm_components(n, m, k, r, theta, phi, jvals=jvals)
        else:
            ErB, EthB, EphB = N_nm_components(n, m, k, r, theta, phi, jvals=jvals)
        Er_total += c[idx]*ErB; Eth_total += c[idx]*EthB; Eph_total += c[idx]*EphB
    return spherical_to_cartesian_vector(Er_total, Eth_total, Eph_total, theta, phi)

def max_rel_error_x(nmax: int = 6, k: float = 1.0, r_fit: float = 1.0) -> float:
    c, basis = project_coeffs_x_vectorized(nmax, k, r_fit=r_fit, n_th=50, n_ph=50)
    worst = 0.0
    for r in (0.7, 1.0, 1.3):
        for th in (0.4, 1.0, 1.5):
            for ph in (0.3, 1.2, 2.2):
                Ex, Ey, Ez = evaluate_from_coeffs(c, basis, k, r, th, ph, nmax)
                Ex_true = cmath.exp(1j*k*r*math.cos(th))
                err = np.linalg.norm([Ex-Ex_true, Ey, Ez])
                worst = max(worst, float(err/max(1e-14, abs(Ex_true))))
    return worst

if __name__ == "__main__":
    for nmax in (4, 6, 8):
        print(f"nmax={nmax}: max_rel_err_x = {max_rel_error_x(nmax):.3e}")

    # --- Plot the reconstructed pattern on a grid ---

    # Define a 2D grid for visualization (XZ plane)
    grid_res = 101
    limit = 2.0  # Plot a region larger than the fitting sphere (r=1.0)
    x_coords = np.linspace(-limit, limit, grid_res)
    z_coords = np.linspace(-limit, limit, grid_res)
    X, Z = np.meshgrid(x_coords, z_coords)
    Y = np.zeros_like(X)

    # Prepare arrays to store the field values
    Ex_rec_grid = np.zeros_like(X, dtype=complex)
    Ex_target_grid = np.zeros_like(X, dtype=complex)

    # Parameters from the test
    k = 1.0

    nmax = 6
    c, basis = project_coeffs_x_vectorized(nmax, k, r_fit=1.0, n_th=50, n_ph=50)   
    
    # --- Vectorized field evaluation on the grid ---
    print("Evaluating fields on grid...")
    # Calculate target field over the whole grid at once
    Ex_target_grid = np.exp(1j * k * Z)

    # Convert Cartesian grid to spherical coordinates
    R_grid = np.sqrt(X**2 + Y**2 + Z**2)
    Theta_grid = np.arccos(np.divide(Z, R_grid, out=np.zeros_like(Z), where=R_grid!=0))
    Phi_grid = np.arctan2(Y, X)

    # Reconstruct the field over the entire grid
    Ex_rec, Ey_rec, Ez_rec = evaluate_from_coeffs_vectorized(
        c, basis, k, R_grid, Theta_grid, Phi_grid, nmax
    )
    Ex_rec_grid = Ex_rec


    # Define the output directory and ensure it exists
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "stratton_reconstruction.png")

    # Plot the comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    vmin, vmax = -1.1, 1.1
    axes[0].pcolormesh(X, Z, np.real(Ex_rec_grid), vmin=vmin, vmax=vmax, cmap='viridis', shading='auto')
    axes[0].set_title("Reconstructed Re(Ex)")
    axes[1].pcolormesh(X, Z, np.real(Ex_target_grid), vmin=vmin, vmax=vmax, cmap='viridis', shading='auto')
    axes[1].set_title("Target Re(Ex)")
    plt.savefig(save_path)
    plt.close(fig) # Close the figure to free up memory
