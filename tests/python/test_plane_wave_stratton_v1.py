import numpy as np
import math
import cmath
import os
import scipy.special as sps
import pandas as pd
import matplotlib.pyplot as plt

# ----- Helpers: spherical Bessel j_l and psi' -----
def spherical_jn_all(lmax, x):
    vals = np.array([sps.spherical_jn(l, x) for l in range(lmax+1)], dtype=float)
    return vals

def psi_prime_from_j(l, x, jvals):
    # psi_l = x j_l(x); psi_l' = x j_{l-1}(x) - l j_l(x) (valid for l>=0 with j_{-1}=cos(x)/x; SciPy can provide via recurrence by hand)
    if l == 0:
        return math.cos(x)
    jlm1 = jvals[l-1]
    jl = jvals[l]
    return x*jlm1 - l*jl

# ----- Y_lm and angular derivatives -----
def Ylm(l, m, theta, phi):
    """Spherical harmonic with SciPy's (theta, phi) convention."""
    # Legacy scipy.special.sph_harm(m, l, phi, theta) is deprecated in 1.15.
    # The replacement sph_harm_y uses the mathematically standard ordering
    # Y_l^m(theta, phi) = sph_harm_y(l, m, theta, phi).
    return sps.sph_harm_y(l, m, theta, phi)

def dY_dphi(l, m, theta, phi, Y=None):
    if Y is None:
        Y = Ylm(l, m, theta, phi)
    return 1j * m * Y

def dY_dtheta_numeric(l, m, theta, phi, h=1e-6):
    # central difference on theta in [0, pi]
    th1 = min(max(theta - h, 1e-9), math.pi-1e-9)
    th2 = min(max(theta + h, 1e-9), math.pi-1e-9)
    Y1 = Ylm(l, m, th1, phi)
    Y2 = Ylm(l, m, th2, phi)
    return (Y2 - Y1)/(th2 - th1)

# ----- X_lm components (tangential) -----
def Xlm_components(l, m, theta, phi):
    # X_theta = + i / sqrt(l(l+1)) * (1/sinθ) * ∂Y/∂φ
    # X_phi   = - i / sqrt(l(l+1)) * ∂Y/∂θ
    if l == 0:
        return 0.0+0.0j, 0.0+0.0j
    Y = Ylm(l, m, theta, phi)
    dYphi = dY_dphi(l, m, theta, phi, Y)
    dYth = dY_dtheta_numeric(l, m, theta, phi)
    st = math.sin(theta)
    if st == 0:
        st = 1e-12
    fac = 1.0/ math.sqrt(l*(l+1))
    Xth = 1j * fac * (dYphi / st)
    Xph = -1j * fac * dYth
    return Xth, Xph

# ----- Build M_lm and N_lm at (r, theta, phi) -----
def Mlm_components(l, m, k, r, theta, phi, jvals=None):
    if jvals is None:
        jvals = spherical_jn_all(l, k*r)
    Xth, Xph = Xlm_components(l, m, theta, phi)
    jl = jvals[l]
    Er = 0.0+0.0j
    Eth = jl * Xth
    Eph = jl * Xph
    return Er, Eth, Eph

# def curl_tangential(ath, aph, r, theta, phi):
#     # A has only theta, phi components
#     st = math.sin(theta); ct = math.cos(theta)
#     # numerical angular derivatives of A components
#     h = 1e-6
#     def Ath(th): return ath(th)
#     def Aph(th): return aph(th)
#     # d/dθ terms
#     th1 = min(max(theta - h, 1e-6), math.pi-1e-6)
#     th2 = min(max(theta + h, 1e-6), math.pi-1e-6)
#     d_sinAph_dtheta = (math.sin(th2)*aph(th2) - math.sin(th1)*aph(th1))/(th2 - th1)
#     dAth_dphi = (aph(theta + 0j*0) * 0)  # placeholder, we will do φ-derivative analytically below if needed
#     # We'll approximate ∂Aθ/∂φ numerically too
#     hphi = 1e-6
#     def Ath_phi(ph): 
#         return ath(theta, phi=ph)
#     dAth_dphi_num = (ath(theta, phi+hphi) - ath(theta, phi-hphi))/(2*hphi)
#     # radial component
#     curl_r = (d_sinAph_dtheta - dAth_dphi_num) / (r*st)
#     # θ and φ components require ∂/∂r of r*Aφ and r*Aθ
#     # We'll pass in closures for Aθ(r) and Aφ(r) below, so this function will not be used.
#     return curl_r

def Nlm_components(l, m, k, r, theta, phi, jvals=None):
    # Using analytic expressions for curl of M when A_r = 0:
    # (∇×M)_r = (1/(r sinθ)) [ ∂(sinθ A_φ)/∂θ - ∂A_θ/∂φ ]
    # (∇×M)_θ = -(1/r) ∂(r A_φ)/∂r
    # (∇×M)_φ =  (1/r) ∂(r A_θ)/∂r
    if jvals is None:
        jvals = spherical_jn_all(l, k*r)
    Xth, Xph = Xlm_components(l, m, theta, phi)
    jl = jvals[l]
    # Define Aθ, Aφ
    Aθ = jl * Xth
    Aφ = jl * Xph
    st = math.sin(theta)
    # angular derivatives
    # ∂(sinθ Aφ)/∂θ
    h = 1e-6
    th1 = min(max(theta - h, 1e-6), math.pi-1e-6)
    th2 = min(max(theta + h, 1e-6), math.pi-1e-6)
    Xth1, Xph1 = Xlm_components(l, m, th1, phi)
    Xth2, Xph2 = Xlm_components(l, m, th2, phi)
    d_sinAφ_dθ = (math.sin(th2)*(jl*Xph2) - math.sin(th1)*(jl*Xph1)) / (th2 - th1)
    # ∂Aθ/∂φ
    hphi = 1e-6
    Xth_p = Xlm_components(l, m, theta, phi+hphi)[0]
    Xth_m = Xlm_components(l, m, theta, phi-hphi)[0]
    dAθ_dφ = jl * (Xth_p - Xth_m) / (2*hphi)
    curl_r = (d_sinAφ_dθ - dAθ_dφ) / (r*st)
    # radial derivative via psi'
    x = k*r
    jvals_r = spherical_jn_all(l, x)  # same as jvals
    psi_p = psi_prime_from_j(l, x, jvals_r)
    curl_θ = -(1/r) * psi_p * Xph
    curl_φ =  (1/r) * psi_p * Xth
    # N = (1/k) curl M
    Er = curl_r / k
    Eth = curl_θ / k
    Eph = curl_φ / k
    return Er, Eth, Eph

# Convert spherical to Cartesian (reuse earlier function)
def sph_to_cart(Er, Etheta, Ephi, theta, phi):
    st, ct = math.sin(theta), math.cos(theta)
    sp, cp = math.sin(phi), math.cos(phi)
    rx, ry, rz = st*cp, st*sp, ct
    tx, ty, tz = ct*cp, ct*sp, -st
    px, py, pz = -sp, cp, 0.0
    Ex = Er*rx + Etheta*tx + Ephi*px
    Ey = Er*ry + Etheta*ty + Ephi*py
    Ez = Er*rz + Etheta*tz + Ephi*pz
    return Ex, Ey, Ez

# Build basis and project numerically to test completeness
def project_plane_wave_coefficients(lmax=4, k=1.0, r=1.0, n_theta=41, n_phi=82):
    thetas = np.linspace(1e-3, math.pi-1e-3, n_theta)
    phis = np.linspace(0, 2*math.pi, n_phi, endpoint=False)
    # Evaluate target E = xhat * e^{ikz}
    # Build basis vectors B_j = M_lm and N_lm flattened in Cartesian components
    basis = []
    labels = []
    for l in range(1, lmax+1):
        for m in range(-l, l+1):
            basis.append(("M", l, m))
            basis.append(("N", l, m))
    nb = len(basis)
    # Assemble least-squares system A c ≈ f, where rows are sampling points*components
    rows = []
    rhs = []
    for th in thetas:
        for ph in phis:
            z = r*math.cos(th)
            phase = cmath.exp(1j*k*z)
            Ex_target, Ey_target, Ez_target = phase, 0.0+0.0j, 0.0+0.0j
            # Build one row-block per component
            # Evaluate all basis at this (th,ph)
            jvals = spherical_jn_all(lmax, k*r)
            cart_cols = []
            for kind,l,m in basis:
                if kind=="M":
                    Er, Eth, Eph = Mlm_components(l, m, k, r, th, ph, jvals=jvals)
                else:
                    Er, Eth, Eph = Nlm_components(l, m, k, r, th, ph, jvals=jvals)
                Ex_b, Ey_b, Ez_b = sph_to_cart(Er, Eth, Eph, th, ph)
                cart_cols.append((Ex_b, Ey_b, Ez_b))
            # Append rows for x,y,z components
            rows.append([col[0] for col in cart_cols])
            rhs.append(Ex_target)
            rows.append([col[1] for col in cart_cols])
            rhs.append(Ey_target)
            rows.append([col[2] for col in cart_cols])
            rhs.append(Ez_target)
    A = np.array(rows, dtype=complex)
    f = np.array(rhs, dtype=complex)
    # Solve least squares
    c, *_ = np.linalg.lstsq(A, f, rcond=1e-6)
    # Reconstruct error on a validation grid
    max_err = 0.0
    for th in np.linspace(0.2, 2.9, 10):
        for ph in np.linspace(0.1, 6.1, 10):
            z = r*math.cos(th)
            phase = cmath.exp(1j*k*z)
            Ex_target, Ey_target, Ez_target = phase, 0.0+0.0j, 0.0+0.0j
            # reconstruct
            Ex_rec=0; Ey_rec=0; Ez_rec=0
            jvals = spherical_jn_all(lmax, k*r)
            idx = 0
            for kind,l,m in basis:
                if kind=="M":
                    Er, Eth, Eph = Mlm_components(l, m, k, r, th, ph, jvals=jvals)
                else:
                    Er, Eth, Eph = Nlm_components(l, m, k, r, th, ph, jvals=jvals)
                Ex_b, Ey_b, Ez_b = sph_to_cart(Er, Eth, Eph, th, ph)
                Ex_rec += c[idx]*Ex_b; Ey_rec += c[idx]*Ey_b; Ez_rec += c[idx]*Ez_b
                idx += 1
            err = np.linalg.norm([Ex_rec-Ex_target, Ey_rec-Ey_target, Ez_rec-Ez_target])
            max_err = max(max_err, float(err))
    return max_err, c, basis

def main():
    """Run the legacy reconstruction visualization outside Pytest."""

    lmax = 6
    max_err, coeffs, basis = project_plane_wave_coefficients(
        lmax=lmax, k=1.0, r=1.0, n_theta=25, n_phi=50
    )
    print(f"Max reconstruction error (lmax={lmax}):", max_err)

    grid_res = 101
    limit = 2.0
    x_coords = np.linspace(-limit, limit, grid_res)
    z_coords = np.linspace(-limit, limit, grid_res)
    X, Z = np.meshgrid(x_coords, z_coords)
    Y = np.zeros_like(X)
    Ex_rec_grid = np.zeros_like(X, dtype=complex)
    Ex_target_grid = np.zeros_like(X, dtype=complex)
    k = 1.0

    for i in range(grid_res):
        for j in range(grid_res):
            x_pt, y_pt, z_pt = X[i, j], Y[i, j], Z[i, j]
            r_pt = np.sqrt(x_pt**2 + y_pt**2 + z_pt**2)
            if r_pt == 0:
                continue
            theta_pt = np.arccos(z_pt / r_pt)
            phi_pt = np.arctan2(y_pt, x_pt)
            Ex_rec = 0.0j
            jvals = spherical_jn_all(lmax, k * r_pt)
            for idx, (kind, l, m) in enumerate(basis):
                if kind == "M":
                    Er, Eth, Eph = Mlm_components(
                        l, m, k, r_pt, theta_pt, phi_pt, jvals=jvals
                    )
                else:
                    Er, Eth, Eph = Nlm_components(
                        l, m, k, r_pt, theta_pt, phi_pt, jvals=jvals
                    )
                Ex_b, _, _ = sph_to_cart(Er, Eth, Eph, theta_pt, phi_pt)
                Ex_rec += coeffs[idx] * Ex_b

            Ex_rec_grid[i, j] = Ex_rec
            Ex_target_grid[i, j] = cmath.exp(1j * k * z_pt)

    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "stratton_reconstruction.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    vmin, vmax = -1.1, 1.1
    axes[0].pcolormesh(
        X,
        Z,
        np.real(Ex_rec_grid),
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        shading="auto",
    )
    axes[0].set_title("Reconstructed Re(Ex)")
    axes[1].pcolormesh(
        X,
        Z,
        np.real(Ex_target_grid),
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        shading="auto",
    )
    axes[1].set_title("Target Re(Ex)")
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
