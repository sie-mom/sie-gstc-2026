import math
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Union

import numpy as np
from scipy.special import sph_harm_y, spherical_jn, spherical_yn


_FIELD_WORKER_CONTEXT = None


def Y_nm(n: int, m: int, theta: float, phi: float) -> complex:
    """Wrapper for SciPy's spherical harmonic Y_n^m(theta, phi)."""
    return sph_harm_y(n, m, theta, phi)


def _x_nm_components_with_harmonic(
    n: int,
    m: int,
    theta: float,
    phi: float,
    dtheta: float = 1e-6,
):
    """Return ``X_theta``, ``X_phi``, and the already evaluated ``Y_nm``."""
    if n == 0:
        zeros = np.zeros_like(np.asarray(theta), dtype=complex)
        return zeros, zeros, zeros
    Y = Y_nm(n, m, theta, phi)
    dY_dphi = 1j * m * Y
    th1 = np.minimum(np.maximum(theta - dtheta, 1e-8), math.pi - 1e-8)
    th2 = np.minimum(np.maximum(theta + dtheta, 1e-8), math.pi - 1e-8)
    Y1 = Y_nm(n, m, th1, phi)
    Y2 = Y_nm(n, m, th2, phi)
    dY_dtheta = np.divide(
        Y2 - Y1,
        th2 - th1,
        out=np.zeros_like(Y),
        where=np.abs(th2 - th1) > 1e-15,
    )

    st = np.sin(theta)
    fac = 1.0 / math.sqrt(n * (n + 1))
    Xth = np.divide(
        1j * fac * dY_dphi,
        st,
        out=np.zeros_like(Y),
        where=st > 1e-9,
    )
    Xph = -1j * fac * dY_dtheta
    return Xth, Xph, Y


def X_nm_components(n: int, m: int, theta: float, phi: float, dtheta: float = 1e-6) -> Tuple[complex, complex]:
    r"""
    Calculates the tangential components of the vector spherical harmonic X_nm.
    This is the core, normalized angular basis function.

    .. math::
       X_\theta = \frac{i}{\sqrt{n(n+1)}} \frac{1}{\sin\theta} \frac{\partial Y}{\partial \phi} \\
       X_\phi = \frac{-i}{\sqrt{n(n+1)}} \frac{\partial Y}{\partial \theta}
    """
    Xth, Xph, _ = _x_nm_components_with_harmonic(
        n, m, theta, phi, dtheta=dtheta
    )
    return Xth, Xph


def get_radial_functions(n_max: int, rho: np.ndarray, kind: Union[str, int]):
    """
    Computes the radial functions z_n(rho) and [rho*z_n(rho)]' for n=0..n_max.
    """
    # Create a working copy to handle arrays with zeros
    rho_safe = np.copy(rho)
    zero_mask = (rho == 0)
    np.place(rho_safe, zero_mask, 1e-30) # Replace 0 with a small number to avoid warnings

    # Get all z_n and their derivatives at once
    if kind in ('j', 1):
        z_n_vals = np.array([spherical_jn(n, rho) for n in range(n_max + 1)], dtype=complex)
        z_n_prime_vals = np.array([spherical_jn(n, rho_safe, derivative=True) for n in range(n_max + 1)], dtype=complex)
    elif kind in ('y', 2):
        z_n_vals = np.array([spherical_yn(n, rho_safe) for n in range(n_max + 1)], dtype=complex)
        z_n_prime_vals = np.array([spherical_yn(n, rho_safe, derivative=True) for n in range(n_max + 1)], dtype=complex)
    elif kind in ('h1', 3):
        jn = np.array([spherical_jn(n, rho) for n in range(n_max + 1)], dtype=complex)
        yn = np.array([spherical_yn(n, rho_safe) for n in range(n_max + 1)], dtype=complex)
        jn_p = np.array([spherical_jn(n, rho_safe, derivative=True) for n in range(n_max + 1)], dtype=complex)
        yn_p = np.array([spherical_yn(n, rho_safe, derivative=True) for n in range(n_max + 1)], dtype=complex)
        z_n_vals = jn + 1j * yn
        z_n_prime_vals = jn_p + 1j * yn_p
    else:
        raise ValueError("kind must be one of 'j', 'y', 'h1' or 1, 2, 3")

    # ψ'_n(ρ) = d/dρ (ρ * z_n(ρ)) = z_n(ρ) + ρ * z_n'(ρ)
    psi_prime_vals = z_n_vals + rho * z_n_prime_vals
    return z_n_vals, psi_prime_vals


# def get_M_N(n: int, m: int, k: float, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, kind: Union[str, int] = 'j', grid: bool = True):
#     """
#     Calculates the Spherical Vector Wave Function pair (M_nm, N_nm).
#     This formulation is based on the curl relationship N = (1/k) * curl(M)
#     and is inherently normalized if Y_nm is normalized.
#     """
#     if grid:
#         theta, phi = np.meshgrid(theta, phi, sparse=True, indexing='ij')

#     # --- Angular Part ---
#     Xth, Xph = X_nm_components(n, m, theta, phi)

#     # --- Radial Part ---
#     rho = k * r
#     z_n_vals, psi_prime_vals = get_radial_functions(n, rho, kind)
#     zn = z_n_vals[n]
#     psi_prime = psi_prime_vals[n]

#     # --- M_nm components ---
#     M_r = np.zeros_like(Xth)
#     M_theta = zn * Xth
#     M_phi = zn * Xph
#     M = (M_r, M_theta, M_phi)

#     # --- N_nm components (from curl of M) ---
#     # (∇×M)_r = (1/(r sinθ)) [ ∂(sinθ M_φ)/∂θ - ∂M_θ/∂φ ]
#     # (∇×M)_θ = -(1/r) ∂(r M_φ)/∂r
#     # (∇×M)_φ =  (1/r) ∂(r M_θ)/∂r
#     st = np.sin(theta)
#     h = 1e-6 # Finite difference step

#     # Radial component (curl_r)
#     th1 = np.minimum(np.maximum(theta - h, 1e-6), math.pi-1e-6)
#     th2 = np.minimum(np.maximum(theta + h, 1e-6), math.pi-1e-6)
#     Xth1, Xph1 = X_nm_components(n, m, th1, phi); Xth2, Xph2 = X_nm_components(n, m, th2, phi)
#     Mphi1 = zn * Xph1; Mphi2 = zn * Xph2
#     numerator = np.sin(th2)*Mphi2 - np.sin(th1)*Mphi1
#     d_sinMphi_dth = np.divide(numerator, th2 - th1, out=np.zeros_like(numerator), where=np.abs(th2-th1) > 1e-15)
    
#     hphi = 1e-6
#     Xth_p, _ = X_nm_components(n, m, theta, phi + hphi)
#     Xth_m, _ = X_nm_components(n, m, theta, phi - hphi)
#     Mth_p = zn * Xth_p; Mth_m = zn * Xth_m
#     dMth_dphi = (Mth_p - Mth_m)/(2*hphi)

#     numerator_curl_r = d_sinMphi_dth - dMth_dphi
#     curl_r = np.divide(numerator_curl_r, (r*st), out=np.zeros_like(numerator_curl_r), where=(r > 1e-9) & (st > 1e-9))

#     # Tangential components
#     curl_th = -np.divide(psi_prime, r, out=np.zeros_like(r, dtype=complex), where=r>1e-9) * Xph
#     curl_ph =  np.divide(psi_prime, r, out=np.zeros_like(r, dtype=complex), where=r>1e-9) * Xth

#     N_r = curl_r / k
#     N_theta = curl_th / k
#     N_phi = curl_ph / k
#     N = (N_r, N_theta, N_phi)

#     return M, N

def get_M_N(
    n: int,
    m: int,
    k: float,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    kind: Union[str, int] = 'j',
    grid: bool = True,
    radial_functions=None,
):
    """
    Calculates the Spherical Vector Wave Function pair (M_nm, N_nm).
    Uses the analytical formulation for N.
    
    Corrections applied:
    1. Tangential signs match the standard Curl derivation (N_theta ~ -X_phi).
    2. Radial component includes '1j' because X_nm is defined with '1j'.

    Args:
        n: Degree of the spherical harmonic.
        m: Order of the spherical harmonic.
        k: Wavenumber.
        r: Radial coordinates (numpy array).
        theta: Polar angles in radians (numpy array).
        phi: Azimuthal angles in radians (numpy array).
        kind: Type of radial function ('j', 'y', 'h1' or 1, 2, 3).
        grid: If True, computes on a meshgrid of (theta, phi). If False, computes element-wise.

    Returns:
        A tuple (M, N) containing the vector wave functions.
        M: A tuple (M_r, M_theta, M_phi) of numpy arrays (complex).
        N: A tuple (N_r, N_theta, N_phi) of numpy arrays (complex).
    """
    if grid:
        theta, phi = np.meshgrid(theta, phi, sparse=True, indexing='ij')

    # --- Angular Part ---
    # Xth, Xph include 1/sqrt(n(n+1)) and the factor '1j'
    Xth, Xph, Y = _x_nm_components_with_harmonic(n, m, theta, phi)

    # --- Radial Part ---
    rho = k * r
    if radial_functions is None:
        z_n_vals, psi_prime_vals = get_radial_functions(n, rho, kind)
    else:
        z_n_vals, psi_prime_vals = radial_functions
        if z_n_vals.shape[0] <= n or psi_prime_vals.shape[0] <= n:
            raise ValueError("radial_functions do not include requested degree")
    zn = z_n_vals[n]
    psi_prime = psi_prime_vals[n]

    # --- M_nm components ---
    M_r = np.zeros_like(Xth)
    M_theta = zn * Xth
    M_phi = zn * Xph
    M = (M_r, M_theta, M_phi)

    # --- N_nm components (Analytical) ---
    rho_safe = np.copy(rho)
    np.place(rho_safe, rho_safe == 0, 1e-30)
    inv_rho = 1.0 / rho_safe

    # 1. Radial Component of N
    # Analytical Curl of M yields a factor of 1j relative to Y_nm
    # because M contains X_nm which contains 1j.
    if n == 0:
        factor_r = 0.0
    else:
        factor_r = math.sqrt(n * (n + 1)) * inv_rho
    
    # FIXED: Added 1j to match the phase of the tangential components
    N_r = 1j * factor_r * zn * Y

    # 2. Tangential Components of N
    # N = (1/k) * curl(M)
    # N_theta = -(1/rho) d/drho(rho M_phi) = -(psi_prime/rho) * X_phi
    # N_phi   = +(1/rho) d/drho(rho M_theta) = +(psi_prime/rho) * X_theta
    
    factor_tan = psi_prime * inv_rho
    
    # FIXED: Signs restored to (- , +) pattern matching standard curl
    N_theta = -factor_tan * Xph
    N_phi   =  factor_tan * Xth

    N = (N_r, N_theta, N_phi)

    return M, N

def compute_fields(
    k: float,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    a_coeffs: dict,
    b_coeffs: dict,
    kind: Union[str, int] = 'j',
    eta: float = 1.0,
    grid: bool = False,
):
    """
    Computes the total Electric and Magnetic fields from the SVWF expansion coefficients.
    
    E = Sum [ a_mn * M_mn + b_mn * N_mn ]
    H = (1 / (i * eta)) * Sum [ a_mn * N_mn + b_mn * M_mn ]

    Args:
        k: Wavenumber.
        r, theta, phi: Coordinates.
        a_coeffs: Dictionary of coefficients for M_mn (Electric).
        b_coeffs: Dictionary of coefficients for N_mn (Electric).
        kind: Bessel function kind ('j', 'y', 'h1').
        eta: Impedance of the medium.
        grid: Whether to compute on a meshgrid (True) or element-wise (False).

    Returns:
        Tuple (E, H), where E and H are tuples (r, theta, phi) components.
        Each component is a numpy array.
    """
    # Determine shape of output
    if grid:
        shape = (len(theta), len(phi))
    else:
        shape = r.shape
        
    E_r = np.zeros(shape, dtype=complex)
    E_theta = np.zeros(shape, dtype=complex)
    E_phi = np.zeros(shape, dtype=complex)
    
    H_r = np.zeros(shape, dtype=complex)
    H_theta = np.zeros(shape, dtype=complex)
    H_phi = np.zeros(shape, dtype=complex)
    
    # Iterate over all modes present in either coefficient dictionary. Radial
    # functions depend on degree and observation radius, not on azimuthal order;
    # compute all degrees once rather than recomputing 0..n for every (m, n).
    all_modes = set(a_coeffs.keys()) | set(b_coeffs.keys())
    maximum_degree = max((n for _, n in all_modes), default=0)
    radial_functions = get_radial_functions(maximum_degree, k * r, kind)

    for (m, n) in all_modes:
        a_val = a_coeffs.get((m, n), 0.0)
        b_val = b_coeffs.get((m, n), 0.0)
        
        # Skip modes that have no contribution
        if a_val == 0.0 and b_val == 0.0:
            continue

        # Get SVWFs
        M, N = get_M_N(
            n,
            m,
            k,
            r,
            theta,
            phi,
            kind=kind,
            grid=grid,
            radial_functions=radial_functions,
        )
        
        # E field accumulation
        E_r += a_val * M[0] + b_val * N[0]
        E_theta += a_val * M[1] + b_val * N[1]
        E_phi += a_val * M[2] + b_val * N[2]
        
        # H field accumulation
        # H = (1/i*eta) * (a*N + b*M)
        factor = 1 / (1j * eta)
        H_r += factor * (a_val * N[0] + b_val * M[0])
        H_theta += factor * (a_val * N[1] + b_val * M[1])
        H_phi += factor * (a_val * N[2] + b_val * M[2])
        
    return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)


def _initialize_field_worker(k, a_coeffs, b_coeffs, kind, eta):
    """Initialize immutable SVWF data once in each field worker process."""

    global _FIELD_WORKER_CONTEXT
    _FIELD_WORKER_CONTEXT = (k, a_coeffs, b_coeffs, kind, eta)


def _compute_field_chunk(coordinates):
    """Evaluate one ordered observation-point chunk in a worker process."""

    if _FIELD_WORKER_CONTEXT is None:  # pragma: no cover - executor contract
        raise RuntimeError("field worker was not initialized")
    k, a_coeffs, b_coeffs, kind, eta = _FIELD_WORKER_CONTEXT
    r, theta, phi = coordinates
    return compute_fields(
        k,
        r,
        theta,
        phi,
        a_coeffs,
        b_coeffs,
        kind=kind,
        eta=eta,
        grid=False,
    )


def compute_fields_chunked(
    k: float,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    a_coeffs: dict,
    b_coeffs: dict,
    kind: Union[str, int] = 'j',
    eta: float = 1.0,
    workers: int = 1,
    chunk_size: int | None = None,
):
    """Evaluate element-wise fields in deterministic process-based chunks.

    ``workers=1`` delegates directly to :func:`compute_fields`. With multiple
    workers, coefficient dictionaries are initialized once per process and
    only coordinate chunks are submitted for each task.
    """

    if workers < 1:
        raise ValueError("workers must be at least one")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be at least one")
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    if r.ndim != 1:
        raise ValueError("chunked field evaluation requires one-dimensional arrays")
    if workers == 1 or r.size == 0:
        return compute_fields(
            k,
            r,
            theta,
            phi,
            a_coeffs,
            b_coeffs,
            kind=kind,
            eta=eta,
            grid=False,
        )

    resolved_chunk_size = chunk_size or max(
        1, math.ceil(r.size / (4 * workers))
    )
    chunks = [
        (r[start:stop], theta[start:stop], phi[start:stop])
        for start in range(0, r.size, resolved_chunk_size)
        for stop in (min(start + resolved_chunk_size, r.size),)
    ]
    with ProcessPoolExecutor(
        max_workers=min(workers, len(chunks)),
        initializer=_initialize_field_worker,
        initargs=(k, a_coeffs, b_coeffs, kind, eta),
    ) as executor:
        results = list(executor.map(_compute_field_chunk, chunks))

    electric = tuple(
        np.concatenate([result[0][component] for result in results])
        for component in range(3)
    )
    magnetic = tuple(
        np.concatenate([result[1][component] for result in results])
        for component in range(3)
    )
    return electric, magnetic


def get_M(n: int, m: int, k: float, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, kind: Union[str, int] = 'j', grid: bool = True):
    """
    Calculates the Spherical Vector Wave Function M_nm.

    This is a convenience wrapper around the more efficient get_M_N.

    Args:
        n, m, k, r, theta, phi, kind: See get_M_N docstring.

    Returns:
        A tuple (M_r, M_theta, M_phi) of the vector components.
    """
    M, _ = get_M_N(n, m, k, r, theta, phi, kind=kind, grid=grid)
    return M


def get_N(n: int, m: int, k: float, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, kind: Union[str, int] = 'j', grid: bool = True):
    """
    Calculates the Spherical Vector Wave Function N_nm.

    This is a convenience wrapper around the more efficient get_M_N.

    Args:
        n, m, k, r, theta, phi, kind: See get_M_N docstring.

    Returns:
        A tuple (N_r, N_theta, N_phi) of the vector components.
    """
    _, N = get_M_N(n, m, k, r, theta, phi, kind=kind, grid=grid)
    return N
