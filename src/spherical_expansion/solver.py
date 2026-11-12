from time import perf_counter

import numpy as np
from scipy.linalg import lstsq as scipy_lstsq

from sie_mom.excitation import evaluate_plane_wave_fields

from .svwf_sphere import get_M_N, get_radial_functions


def evaluate_incident_plane_wave(
    k: float,
    r: float,
    theta: np.ndarray,
    phi: np.ndarray,
    E0: float = 1.0,
    eta: float = 1.0,
    grid: bool = True,
):
    """
    Evaluates the incident plane wave fields (E_inc, H_inc) at the given coordinates.
    Assumes x-polarized wave propagating in +z direction.

    E_inc = E0 * exp(ikz) * x_hat
    H_inc = (E0/eta) * exp(ikz) * y_hat

    Args:
        k: Wavenumber.
        r: Radial distance.
        theta: Polar angles (1D array).
        phi: Azimuthal angles (1D array).
        E0: Amplitude.
        grid: If True, treat theta and phi as axes of a grid. If False, treat as a list of points.

    Returns:
        A tuple containing two tuples: `(E_inc, H_inc)`.
        `E_inc` is `(E_r, E_theta, E_phi)`.
        `H_inc` is `(H_r, H_theta, H_phi)`.
    """
    electric, magnetic = evaluate_plane_wave_fields(
        k,
        r,
        theta,
        phi,
        eta,
        electric_field=(E0, 0.0, 0.0),
        propagation_direction=(0.0, 0.0, 1.0),
        grid=grid,
    )
    return tuple(electric), tuple(magnetic)


def solve_collocation(
    n_max: int,
    k0: float,
    eta0: float,
    k1: float,
    eta1: float,
    radius: float,
    gstc_params: dict,
    n_theta: int = None,
    n_phi: int = None,
    row_scale: bool = False,
    return_metrics: bool = False,
):
    """
    Solves for scattering from a GSTC-coated sphere using collocation.

    This function builds and solves the linear system Z*x = V.

    Args:
        n_max: Truncation limit for the SVWF expansion.
        k0: Wavenumber of the exterior medium.
        k1: Wavenumber of the interior medium.
        radius: Radius of the sphere 'a'.
        gstc_params: Dictionary containing GSTC tensors 'Y', 'Chi', 'Gamma', 'Z'.
                     Each value can be a scalar (isotropic/uniform) or a 2x2 matrix.
                     For now, we assume uniform tensors.
                     If a parameter is missing, it defaults to 0.0.
        n_theta: Number of points in theta direction (optional).
        n_phi: Number of points in phi direction (optional).

    Returns:
        A tuple of dictionaries (a_coeffs, b_coeffs, c_coeffs, d_coeffs)
        containing the solved scattering and interior field coefficients.
    """
    # --- 1. Define Unknowns and Modes ---
    # 'modes' is a list of tuples (m, n) representing the spherical harmonic indices.
    #   m: order (-n to n)
    #   n: degree (1 to n_max)
    # The list is ordered primarily by n, then by m.
    modes = []
    for n in range(1, n_max + 1):
        for m in range(-n, n + 1):
            modes.append((m, n))

    num_modes = len(modes)
    # 4 unknowns per mode (a, b, c, d)
    num_unknowns = 4 * num_modes
    
    # Map (coeff_type, m, n) to column index
    col_map = {}
    idx = 0
    for m, n in modes:
        for c_type in ['a', 'b', 'c', 'd']:
            # The unknown vector x is organized as blocks of 4 coefficients per mode:
            # x = [a_mode1, b_mode1, c_mode1, d_mode1, a_mode2, ...]^T
            col_map[(c_type, (m, n))] = idx
            idx += 1

    # --- 2. Define Collocation Grid ---
    # We need 2 equations per point (J and M boundary conditions).
    # Each equation has theta and phi components. So 4 equations per point.
    # We need 4 * num_modes equations total.
    # So we need num_points = num_modes.
    
    # Simple grid strategy:
    # N_theta * N_phi = num_modes approx.
    # Let's try to match it exactly or use a least-squares if overdetermined.
    # For this implementation, we'll use the user's suggestion but ensure we have enough points.
    # N_theta = n_max + 1, N_phi = 2*n_max + 1 -> Total ~ 2*n_max^2.
    # Num modes = n_max(n_max+2) ~ n_max^2.
    # So we have roughly 2x oversampling, which is good (least squares).
    
    if n_theta is None:
        N_theta = n_max + 1
    else:
        N_theta = n_theta
        
    if n_phi is None:
        N_phi = 2 * n_max + 1
    else:
        N_phi = n_phi
    
    # Shift theta slightly to avoid poles (0 and pi) where fields might be singular or zero
    theta_points = np.linspace(0.001, np.pi-0.001, N_theta)
    phi_points = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    num_points = N_theta * N_phi
    num_equations = 4 * num_points # 2 vector equations (J and M) -> 4 scalar eq per point
    
    # --- 3. Initialize System ---
    # We will solve Z * x = V using least squares if overdetermined
    Z = np.zeros((num_equations, num_unknowns), dtype=complex)
    V = np.zeros(num_equations, dtype=complex)
    
    # --- 4. Pre-calculate Incident Field (RHS) ---
    # E_inc, H_inc at all points
    # Shapes: (N_theta, N_phi)
    # The function returns nested tuples: ((Er, Eth, Eph), (Hr, Hth, Hph))
    E_inc_sph, H_inc_sph = evaluate_incident_plane_wave(k0, radius, theta_points, phi_points, E0=1.0, eta=eta0, grid=True)
    E_inc_th, E_inc_ph = E_inc_sph[1], E_inc_sph[2]
    H_inc_th, H_inc_ph = H_inc_sph[1], H_inc_sph[2]
     
    
    # Flatten arrays to loop over points easily
    E_inc_th = E_inc_th.flatten()
    E_inc_ph = E_inc_ph.flatten()
    H_inc_th = H_inc_th.flatten()
    H_inc_ph = H_inc_ph.flatten()
    
    # --- 5. Prepare GSTC Tensors ---
    # Assuming uniform tensors for now. 
    # If they are scalars, make them 2x2 identity scaled.
    # Tensors are 2x2 matrices acting on [theta, phi] components.
    def get_tensor(name):
        val = gstc_params.get(name, 0.0)
        if np.isscalar(val):
            return val * np.eye(2)
        return np.array(val)

    Y_tensor   = get_tensor('Y')
    Chi_tensor = get_tensor('Chi')
    Gam_tensor = get_tensor('Gamma')
    Z_tensor   = get_tensor('Z')
    
    # --- 6. Build Matrix Z and Vector V ---
    
    # Pre-calculate SVWFs for all modes at all points
    # This is more efficient than calling get_M_N inside the loop
    # We need M and N for k0 (exterior) and k1 (interior)
    
    # Dictionaries to store wave functions: keys (m,n), values: (M_th, M_ph, N_th, N_ph) arrays
    # Exterior (k0, outgoing/Hankel for scattered) -> kind=3
    # Interior (k1, regular/Bessel) -> kind=1
    
    # Note: get_M_N returns (M_r, M_th, M_ph), we only need th and ph.
    # Also get_M_N does meshgrid internally.
    
    svwf_ext = {}
    svwf_int = {}
    
    for m, n in modes:
        # Exterior (Scattered): Hankel (kind=3)
        # get_M_N returns M=(Mr, Mth, Mph) and N=(Nr, Nth, Nph).
        # We store only tangential components flattened: (M_theta, M_phi, N_theta, N_phi).
        # M_ext[1] is M_theta, M_ext[2] is M_phi.
        M_ext, N_ext = get_M_N(n, m, k0, radius, theta_points, phi_points, kind=3)
        svwf_ext[(m, n)] = (M_ext[1].flatten(), M_ext[2].flatten(), N_ext[1].flatten(), N_ext[2].flatten())
        
        # Interior: Bessel (kind=1)
        # Format stored in svwf_int is also (M_theta, M_phi, N_theta, N_phi).
        M_int, N_int = get_M_N(n, m, k1, radius, theta_points, phi_points, kind=1)
        svwf_int[(m, n)] = (M_int[1].flatten(), M_int[2].flatten(), N_int[1].flatten(), N_int[2].flatten())

    # --- Vectorized Construction of Z and V ---
    
    # Prepare vectorized incident fields
    # Ei, Hi shape: (2, num_points)
    # Row 0: Theta components, Row 1: Phi components.
    # This specific layout (2, N) allows for vectorized matrix multiplication with the GSTC tensors 
    # (like Y_tensor, Z_tensor), which are 2x2 matrices acting on the tangential vector space [theta, phi]^T.
    Ei = np.array([E_inc_th, E_inc_ph])
    Hi = np.array([H_inc_th, H_inc_ph])
    
    # Rotation operator R = n x . 
    # R([A_th, A_ph]) = [-A_ph, A_th]
    # Derivation: n = r_hat.
    # r_hat x (Ath * th_hat + Aph * ph_hat) 
    # = Ath * (r_hat x th_hat) + Aph * (r_hat x ph_hat)
    # = Ath * ph_hat + Aph * (-th_hat) = -Aph * th_hat + Ath * ph_hat
    def rot_vec(vec):
        # vec shape (2, N)
        return np.array([-vec[1], vec[0]])
        
    # --- Fill RHS (Vector V) ---
    # Eq 1 (J): -(R - Chi/2)H_inc + (Y/2)E_inc
    rhs_J = -(rot_vec(Hi) - (Chi_tensor @ Hi) / 2.0) + (Y_tensor @ Ei) / 2.0
    
    # Eq 2 (M): -(R + Gam/2)E_inc - (Z/2)H_inc
    rhs_M = -(rot_vec(Ei) + (Gam_tensor @ Ei) / 2.0) - (Z_tensor @ Hi) / 2.0
    
    # --- Fill V (interleaved) ---
    V[0::4] = rhs_J[0]
    V[1::4] = rhs_J[1]
    V[2::4] = rhs_M[0]
    V[3::4] = rhs_M[1]
    
    print(f"DEBUG: Max V magnitude: {np.max(np.abs(V))}")
    print(f"DEBUG: Mean V magnitude: {np.mean(np.abs(V))}")
    
    # --- Apply Integration Weights (Weighted Least Squares) ---
    # We want to minimize integral over sphere surface, not sum over points.
    # Weights w_i ~ sqrt(sin(theta_i))
    # This corresponds to minimizing the surface integral of the squared error.
    # dA = R^2 sin(theta) dtheta dphi.
    # Sum |err|^2 * w^2 ~ Integral |err|^2 dA
    # So w ~ sqrt(sin(theta))
    
    weights_theta = np.sqrt(np.sin(theta_points))
    # Broadcast to (N_theta, N_phi) and flatten
    weights_grid = np.tile(weights_theta[:, None], (1, N_phi)).flatten()
    # Repeat for 4 equations per point
    weights = np.repeat(weights_grid, 4)
    
    # Apply weights to V (and later to Z)
    # We are solving W * Z * x = W * V
    # Effectively multiplying rows.
    # V = V * weights # Moved to after loop to allow unit balancing first
    
    # --- Fill LHS (Matrix Z) ---
    eta1_rel = k0 / k1
    
    for m, n in modes:
        # Exterior SVWFs (Scattered)
        # Shapes: (num_points,)
        M_sc_th, M_sc_ph, N_sc_th, N_sc_ph = svwf_ext[(m, n)]
        vec_M_sc = np.array([M_sc_th, M_sc_ph]) # (2, N)
        vec_N_sc = np.array([N_sc_th, N_sc_ph]) # (2, N)
        
        # Interior SVWFs
        M_in_th, M_in_ph, N_in_th, N_in_ph = svwf_int[(m, n)]
        vec_M_in = np.array([M_in_th, M_in_ph]) # (2, N)
        vec_N_in = np.array([N_in_th, N_in_ph]) # (2, N)
        
        # --- Define Basis Vectors for Unknowns ---
        # E_scat = a*M + b*N
        # H_scat = (1/i*eta0)*(a*N + b*M) -> (1/1j)*(a*N + b*M)
        
        # Coeff a: E -> M_sc, H -> (1/1j* eta0) * N_sc
        E_a = vec_M_sc
        H_a = (1/(1j * eta0)) * vec_N_sc
        
        # Coeff b: E -> N_sc, H -> (1/1j* eta0) * M_sc
        E_b = vec_N_sc
        H_b = (1/(1j * eta0)) * vec_M_sc
        
        # E_int = c*M + d*N
        # H_int = (1/i*eta1)*(c*N + d*M) -> (1/(1j*eta1))*(c*N + d*M)
        
        # Coeff c: E -> M_in, H -> (1/(1j*eta1))*N_in
        E_c = vec_M_in
        H_c = (1/(1j * eta1)) * vec_N_in
        
        # Coeff d: E -> N_in, H -> (1/(1j*eta1))*M_in
        E_d = vec_N_in
        H_d = (1/(1j * eta1)) * vec_M_in
        
        # --- Apply GSTC Operators ---
        # Helper to compute columns
        def compute_cols(E, H, is_interior=False):
            if not is_interior:
                # Terms for scattered fields (a, b coefficients)
                # J eq: (R - Chi/2)H_sc - (Y/2)E_sc
                col_J = (rot_vec(H) - (Chi_tensor @ H) / 2.0) - (Y_tensor @ E) / 2.0
                # M eq: (R + Gam/2)E_sc + (Z/2)H_sc
                col_M = (rot_vec(E) + (Gam_tensor @ E) / 2.0) + (Z_tensor @ H) / 2.0
            else:
                # Terms for interior fields (c, d coefficients)
                # J eq: -(R + Chi/2)H_int - (Y/2)E_int
                col_J = -(rot_vec(H) + (Chi_tensor @ H) / 2.0) - (Y_tensor @ E) / 2.0
                # M eq: -(R - Gam/2)E_int + (Z/2)H_int
                col_M = -(rot_vec(E) - (Gam_tensor @ E) / 2.0) + (Z_tensor @ H) / 2.0
            return col_J, col_M

        # Calculate columns
        col_J_a, col_M_a = compute_cols(E_a, H_a, is_interior=False)
        col_J_b, col_M_b = compute_cols(E_b, H_b, is_interior=False)
        col_J_c, col_M_c = compute_cols(E_c, H_c, is_interior=True)
        col_J_d, col_M_d = compute_cols(E_d, H_d, is_interior=True)
        
        # --- Fill Z Matrix ---
        # a
        idx_a = col_map[('a', (m, n))]
        Z[0::4, idx_a] = col_J_a[0]
        Z[1::4, idx_a] = col_J_a[1]
        Z[2::4, idx_a] = col_M_a[0]
        Z[3::4, idx_a] = col_M_a[1]
        
        # b
        idx_b = col_map[('b', (m, n))]
        Z[0::4, idx_b] = col_J_b[0]
        Z[1::4, idx_b] = col_J_b[1]
        Z[2::4, idx_b] = col_M_b[0]
        Z[3::4, idx_b] = col_M_b[1]
        
        # c
        idx_c = col_map[('c', (m, n))]
        Z[0::4, idx_c] = col_J_c[0]
        Z[1::4, idx_c] = col_J_c[1]
        Z[2::4, idx_c] = col_M_c[0]
        Z[3::4, idx_c] = col_M_c[1]
        
        # d
        idx_d = col_map[('d', (m, n))]
        Z[0::4, idx_d] = col_J_d[0]
        Z[1::4, idx_d] = col_J_d[1]
        Z[2::4, idx_d] = col_M_d[0]
        Z[3::4, idx_d] = col_M_d[1]

    # --- Unit Balancing of Equations ---
    # To improve matrix condition, we make both GSTC equations have the same units (E-field).
    # The J-equation (rows 0,1) has units of H-field (A/m).
    # The M-equation (rows 2,3) has units of E-field (V/m).
    # We multiply the J-equation by eta0 to give it units of E-field.
    V[0::4] *= eta0
    V[1::4] *= eta0
    Z[0::4, :] *= eta0
    Z[1::4, :] *= eta0


    # Apply weights to Z and V (Weighted Least Squares)
    # This completes the weighting
    V = V * weights
    Z = Z * weights[:, None]

    print(f"DEBUG: Max Z magnitude (weighted): {np.max(np.abs(Z))}")
    print(f"DEBUG: Mean Z magnitude (weighted): {np.mean(np.abs(Z))}")

    # Prepare working copies
    Z_work = Z.copy()
    V_work = V.copy()
    
    # Note: Row scaling would undo the integration weights if done by simple normalization.
    # We skip row scaling here to preserve the geometric weighting.
    # If unit balancing is needed, it should be done by scaling equations by constants, not by their norm.
    
    # --- 7. Solve ---
    # RMS Magnitude Normalization (Column Scaling)
    
    col_norms = np.linalg.norm(Z_work, axis=0)
    
    min_norm = np.min(col_norms)
    max_norm = np.max(col_norms)
    print(f"DEBUG: Column norms range: {min_norm:.3e} to {max_norm:.3e}")
    
    # --- SAFETY CHECK 1: Check for NaNs before scaling ---
    if np.any(np.isnan(Z_work)) or np.any(np.isnan(V_work)):
        print("ERROR: NaNs detected in Z_work or V_work matrix construction!")
        # Find where
        if np.any(np.isnan(V_work)): print("  -> NaNs found in RHS Vector V")
        if np.any(np.isnan(Z_work)): print("  -> NaNs found in System Matrix Z")
        raise ValueError("NaNs detected in linear system construction.")

    # --- SAFETY CHECK 2: Protect against Zero/Tiny Columns ---
    # If a column is effectively zero (e.g. < 1e-30), dividing by it creates Inf.
    # We set the norm to 1.0 to avoid division by zero. 
    # (The column remains zero, which is fine for lstsq).
    
    safe_norms = col_norms.copy()
    # Replace tiny norms with 1.0 to prevent overflow
    safe_norms[safe_norms < 1e-30] = 1.0
    
    # Scale Z
    Z_scaled = Z_work / safe_norms[None, :]
    
    # --- SAFETY CHECK 3: Check for NaNs/Infs AFTER scaling ---
    if np.any(np.isnan(Z_scaled)) or np.any(np.isinf(Z_scaled)):
        print("ERROR: NaNs or Infs generated during matrix scaling.")
        print(f"  Min Safe Norm: {np.min(safe_norms)}")
        raise ValueError("Matrix scaling failed (likely division by zero column).")

    # Least squares solution
    # We use SciPy's 'gelsy' driver (QR with pivoting).
    # It is strictly more robust than Numpy's 'gelsd' (SVD) for rank-deficient 
    # systems (like the k0=k1 transparent sphere case).
    try:
        x_scaled, residuals, rank, s = scipy_lstsq(
            Z_scaled, 
            V_work, 
            cond=1e-14,    # 'cond' replaces 'rcond' in scipy
            lapack_driver='gelsy'
        )
    except Exception as e:
        print(f"CRITICAL ERROR in scipy.linalg.lstsq: {e}")
        # Determine if matrix is singular or has bad values
        print(f"  Z_scaled stats -> Max: {np.max(np.abs(Z_scaled))}, NaNs: {np.any(np.isnan(Z_scaled))}, Infs: {np.any(np.isinf(Z_scaled))}")
        raise e

    # Residual diagnostics (scaled system)
    res_scaled = V_work - Z_scaled @ x_scaled
    res_scaled_norm = np.linalg.norm(res_scaled)
    V_work_norm = np.linalg.norm(V_work) + 1e-16
    # V_norm is the norm of the weighted V
    V_norm = V_work_norm
    
    print(
        "DEBUG: Residual ||Z_scaled x - V|| = {:.3e} (rel {:.3e})".format(
            res_scaled_norm,
            res_scaled_norm / V_work_norm,
        )
    )
    
    # Recover true coefficients
    x = x_scaled / col_norms
    
    # Calculate original residual (weighted)
    res_orig = V_work - Z_work @ x
    res_orig_norm = np.linalg.norm(res_orig)
    print(
        "DEBUG: Residual ||Z x - V|| (weighted) = {:.3e} (rel {:.3e})".format(
            res_orig_norm,
            res_orig_norm / V_norm,
        )
    )

        
    
    # --- 8. Unpack Coefficients ---
    a_coeffs = {}
    b_coeffs = {}
    c_coeffs = {}
    d_coeffs = {}
    
    for m, n in modes:
        a_coeffs[(m, n)] = x[col_map[('a', (m, n))]]
        b_coeffs[(m, n)] = x[col_map[('b', (m, n))]]
        c_coeffs[(m, n)] = x[col_map[('c', (m, n))]]
        d_coeffs[(m, n)] = x[col_map[('d', (m, n))]]


    metrics = {
        'row_scale': row_scale,
        'residual_scaled_norm': res_scaled_norm,
        'residual_scaled_rel': res_scaled_norm / V_work_norm,
        'residual_norm': res_orig_norm,
        'residual_rel': res_orig_norm / V_norm,
    }

    if return_metrics:
        return (a_coeffs, b_coeffs, c_coeffs, d_coeffs, metrics)
    
    return a_coeffs, b_coeffs, c_coeffs, d_coeffs

def solve_collocation_general(
    n_max: int,
    k0: float,
    eta0: float,
    k1: float,
    eta1: float,
    radius: float,
    gstc_params: dict,
    n_theta: int = None,
    n_phi: int = None,
    row_scale: bool = False,
    return_metrics: bool = False,
    incident_field_func = None,
    is_interior_source: bool = False,
):
    """
    Solves for scattering from a GSTC-coated sphere using collocation, supporting
    NON-UNIFORM (spatially varying) and ANISOTROPIC GSTC parameters.

    Args:
        n_max, k0, eta0, k1, eta1, radius: Same as solve_collocation.
        gstc_params: Dictionary containing 'Y', 'Chi', 'Gamma', 'Z'.
                     Values can be:

                     - Scalar: converted to uniform isotropic tensor.
                     - 2x2 Array: converted to uniform anisotropic tensor.
                     - Callable f(theta, phi): must return shape (2, 2, N) or (2, 2).
                       If it returns (2, 2), it's broadcasted.
                       theta/phi args are 1D arrays of size N (flattened grid).
        n_theta, n_phi: Grid dimensions.
        incident_field_func: Optional callable. func(theta_flat, phi_flat) -> 
                             (E_th, E_ph, H_th, H_ph) flattened arrays of size N.
        is_interior_source: Set to True if the incident field originates from inside the sphere.

    Returns:
        Same as solve_collocation.
    """
    workflow_started = perf_counter()
    # --- 1. Define Unknowns and Modes ---
    modes = []
    for n in range(1, n_max + 1):
        for m in range(-n, n + 1):
            modes.append((m, n))

    num_modes = len(modes)
    num_unknowns = 4 * num_modes
    
    col_map = {}
    idx = 0
    for m, n in modes:
        for c_type in ['a', 'b', 'c', 'd']:
            col_map[(c_type, (m, n))] = idx
            idx += 1

    # --- 2. Define Collocation Grid ---
    if n_theta is None: N_theta = n_max + 1
    else: N_theta = n_theta
        
    if n_phi is None: N_phi = 2 * n_max + 1
    else: N_phi = n_phi
    
    theta_points_1d = np.linspace(0.001, np.pi-0.001, N_theta)
    phi_points_1d = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    # Create Full Grid for Tensor Evaluation
    # We need flattened arrays of size N to pass to user functions
    theta_grid, phi_grid = np.meshgrid(theta_points_1d, phi_points_1d, indexing='ij')
    
    # Flatten everything
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    num_points = N_theta * N_phi
    num_equations = 4 * num_points

    # --- 3. Initialize System ---
    Z = np.zeros((num_equations, num_unknowns), dtype=complex)
    V = np.zeros(num_equations, dtype=complex)
    
    # --- 4. Pre-calculate Incident Field (RHS) ---
    if incident_field_func is None:
        # Default to exterior plane wave
        E_inc_sph, H_inc_sph = evaluate_incident_plane_wave(k0, radius, theta_points_1d, phi_points_1d, E0=1.0, eta=eta0, grid=True)
        
        # Flatten: Shape (N,)
        E_inc_th = E_inc_sph[1].flatten()
        E_inc_ph = E_inc_sph[2].flatten()
        H_inc_th = H_inc_sph[1].flatten()
        H_inc_ph = H_inc_sph[2].flatten()
    else:
        # Expected to return exactly the 4 flattened tangential components
        E_inc_th, E_inc_ph, H_inc_th, H_inc_ph = incident_field_func(theta_flat, phi_flat)
    
    # --- 5. Prepare General GSTC Tensors ---
    # Target Shape: (2, 2, num_points)
    # Indices: (Out-Component, In-Component, Point-Index)
    # 0 -> theta, 1 -> phi
    
    def prepare_tensor(name):
        val = gstc_params.get(name, 0.0)
        
        # Case A: Callable (Function of position)
        if callable(val):
            # Evaluate function
            # Expected signature: func(theta_array, phi_array) -> returns (2, 2, N)
            tensor = val(theta_flat, phi_flat)
            
            # Validation
            if tensor.shape == (2, 2):
                # User returned a constant matrix from a function? Broadcast it.
                tensor = np.tile(tensor[:, :, None], (1, 1, num_points))
            elif tensor.shape != (2, 2, num_points):
                raise ValueError(f"GSTC Param '{name}' callable returned shape {tensor.shape}, expected (2, 2, {num_points})")
            return tensor

        # Case B: Scalar (Uniform Isotropic)
        if np.isscalar(val):
            # Create (2, 2) identity * val, then tile
            base = val * np.eye(2)
            return np.tile(base[:, :, None], (1, 1, num_points))
            
        # Case C: Array (Uniform Anisotropic or Pre-computed Non-uniform)
        arr = np.array(val)
        if arr.shape == (2, 2):
            return np.tile(arr[:, :, None], (1, 1, num_points))
        elif arr.shape == (2, 2, num_points):
            return arr
        else:
            raise ValueError(f"GSTC Param '{name}' has invalid shape {arr.shape}")

    Y_tensor   = prepare_tensor('Y')
    Chi_tensor = prepare_tensor('Chi')
    Gam_tensor = prepare_tensor('Gamma')
    Z_tensor   = prepare_tensor('Z')
    setup_seconds = perf_counter() - workflow_started
    
    # --- 6. Build Matrix Z and Vector V ---
    
    # Same SVWF pre-calc as standard solver
    basis_started = perf_counter()
    svwf_ext = {}
    svwf_int = {}
    radial_ext = get_radial_functions(n_max, k0 * radius, kind=3)
    radial_int = get_radial_functions(n_max, k1 * radius, kind=1)
    
    for m, n in modes:
        M_ext, N_ext = get_M_N(
            n,
            m,
            k0,
            radius,
            theta_points_1d,
            phi_points_1d,
            kind=3,
            radial_functions=radial_ext,
        )
        svwf_ext[(m, n)] = (M_ext[1].flatten(), M_ext[2].flatten(), N_ext[1].flatten(), N_ext[2].flatten())
        
        M_int, N_int = get_M_N(
            n,
            m,
            k1,
            radius,
            theta_points_1d,
            phi_points_1d,
            kind=1,
            radial_functions=radial_int,
        )
        svwf_int[(m, n)] = (M_int[1].flatten(), M_int[2].flatten(), N_int[1].flatten(), N_int[2].flatten())
    basis_seconds = perf_counter() - basis_started

    # --- Vectorized Construction ---
    assembly_started = perf_counter()
    
    # Field Vectors: Shape (2, N)
    Ei = np.array([E_inc_th, E_inc_ph])
    Hi = np.array([H_inc_th, H_inc_ph])
    
    def rot_vec(vec):
        # n x v same as before
        return np.array([-vec[1], vec[0]])
        
    def apply_tensor(T, vec):
        """
        Computes T . vec per point using Einstein Summation.
        T: (2, 2, N) -> indices i,j,n
        vec: (2, N)  -> indices j,n
        Result: (2, N) -> indices i,n
        """
        return np.einsum('ijn,jn->in', T, vec)
        
    def compute_cols_gen(E, H, is_interior=False):
        # Calculate terms using einsum 'apply_tensor'
        term_rot_H = rot_vec(H)
        term_rot_E = rot_vec(E)
        term_Chi_H = apply_tensor(Chi_tensor, H)
        term_Y_E   = apply_tensor(Y_tensor, E)
        term_Gam_E = apply_tensor(Gam_tensor, E)
        term_Z_H   = apply_tensor(Z_tensor, H)

        if not is_interior:
            col_J = (term_rot_H - term_Chi_H / 2.0) - term_Y_E / 2.0
            col_M = (term_rot_E + term_Gam_E / 2.0) + term_Z_H / 2.0
        else:
            col_J = -(term_rot_H + term_Chi_H / 2.0) - term_Y_E / 2.0
            col_M = -(term_rot_E - term_Gam_E / 2.0) + term_Z_H / 2.0
        return col_J, col_M

    # --- Fill RHS (Vector V) ---
    # The RHS is exactly the negative of the corresponding operator evaluated at the incident fields
    col_J_inc, col_M_inc = compute_cols_gen(Ei, Hi, is_interior=is_interior_source)
    rhs_J = -col_J_inc
    rhs_M = -col_M_inc
    
    V[0::4] = rhs_J[0]
    V[1::4] = rhs_J[1]
    V[2::4] = rhs_M[0]
    V[3::4] = rhs_M[1]
    
    # --- Weights ---
    weights_theta = np.sqrt(np.sin(theta_points_1d))
    weights_grid = np.tile(weights_theta[:, None], (1, N_phi)).flatten()
    weights = np.repeat(weights_grid, 4)
    
    # --- Fill LHS (Matrix Z) ---
    for m, n in modes:
        M_sc_th, M_sc_ph, N_sc_th, N_sc_ph = svwf_ext[(m, n)]
        vec_M_sc = np.array([M_sc_th, M_sc_ph]) # (2, N)
        vec_N_sc = np.array([N_sc_th, N_sc_ph]) # (2, N)
        
        M_in_th, M_in_ph, N_in_th, N_in_ph = svwf_int[(m, n)]
        vec_M_in = np.array([M_in_th, M_in_ph]) # (2, N)
        vec_N_in = np.array([N_in_th, N_in_ph]) # (2, N)
        
        # Define Basis Fields
        E_a, H_a = vec_M_sc, (1/(1j * eta0)) * vec_N_sc
        E_b, H_b = vec_N_sc, (1/(1j * eta0)) * vec_M_sc
        E_c, H_c = vec_M_in, (1/(1j * eta1)) * vec_N_in
        E_d, H_d = vec_N_in, (1/(1j * eta1)) * vec_M_in
        
        col_J_a, col_M_a = compute_cols_gen(E_a, H_a, False)
        col_J_b, col_M_b = compute_cols_gen(E_b, H_b, False)
        col_J_c, col_M_c = compute_cols_gen(E_c, H_c, True)
        col_J_d, col_M_d = compute_cols_gen(E_d, H_d, True)
        
        # Fill Z
        idx_a = col_map[('a', (m, n))]
        Z[0::4, idx_a] = col_J_a[0]
        Z[1::4, idx_a] = col_J_a[1]
        Z[2::4, idx_a] = col_M_a[0]
        Z[3::4, idx_a] = col_M_a[1]
        
        idx_b = col_map[('b', (m, n))]
        Z[0::4, idx_b] = col_J_b[0]
        Z[1::4, idx_b] = col_J_b[1]
        Z[2::4, idx_b] = col_M_b[0]
        Z[3::4, idx_b] = col_M_b[1]
        
        idx_c = col_map[('c', (m, n))]
        Z[0::4, idx_c] = col_J_c[0]
        Z[1::4, idx_c] = col_J_c[1]
        Z[2::4, idx_c] = col_M_c[0]
        Z[3::4, idx_c] = col_M_c[1]
        
        idx_d = col_map[('d', (m, n))]
        Z[0::4, idx_d] = col_J_d[0]
        Z[1::4, idx_d] = col_J_d[1]
        Z[2::4, idx_d] = col_M_d[0]
        Z[3::4, idx_d] = col_M_d[1]
    matrix_assembly_seconds = perf_counter() - assembly_started

    # --- Unit Balancing & Weighting (Identical to Standard) ---
    scaling_started = perf_counter()
    V[0::4] *= eta0
    V[1::4] *= eta0
    Z[0::4, :] *= eta0
    Z[1::4, :] *= eta0

    np.multiply(V, weights, out=V)
    np.multiply(Z, weights[:, None], out=Z)

    # --- Solving (Identical to Standard) ---
    col_norms = np.linalg.norm(Z, axis=0)
    safe_norms = col_norms.copy()
    safe_norms[safe_norms < 1e-30] = 1.0
    np.divide(Z, safe_norms[None, :], out=Z)
    scaling_seconds = perf_counter() - scaling_started

    least_squares_started = perf_counter()
    try:
        x_scaled, residuals, rank, s = scipy_lstsq(
            Z,
            V,
            cond=1e-14,
            lapack_driver='gelsy',
        )
    except Exception as e:
        print(f"Error in solve_collocation_general: {e}")
        raise e
    least_squares_seconds = perf_counter() - least_squares_started

    # Residuals
    residual_started = perf_counter()
    residual = V - Z @ x_scaled
    residual_norm = np.linalg.norm(residual)
    V_norm = np.linalg.norm(V) + 1e-16

    # This is also the physical residual: (A D^-1) x_scaled == A x.
    x = x_scaled / safe_norms
    residual_seconds = perf_counter() - residual_started

    # Output
    a_coeffs, b_coeffs, c_coeffs, d_coeffs = {}, {}, {}, {}
    for m, n in modes:
        a_coeffs[(m, n)] = x[col_map[('a', (m, n))]]
        b_coeffs[(m, n)] = x[col_map[('b', (m, n))]]
        c_coeffs[(m, n)] = x[col_map[('c', (m, n))]]
        d_coeffs[(m, n)] = x[col_map[('d', (m, n))]]

    metrics = {
        'row_scale': row_scale,
        'residual_scaled_norm': residual_norm,
        'residual_scaled_rel': residual_norm / V_norm,
        'residual_norm': residual_norm,
        'residual_rel': residual_norm / V_norm,
        'setup_seconds': setup_seconds,
        'basis_seconds': basis_seconds,
        'matrix_assembly_seconds': matrix_assembly_seconds,
        'scaling_seconds': scaling_seconds,
        'least_squares_seconds': least_squares_seconds,
        'residual_seconds': residual_seconds,
        'total_seconds': perf_counter() - workflow_started,
        'equation_count': num_equations,
        'unknown_count': num_unknowns,
        'matrix_bytes': Z.nbytes,
        'full_size_matrix_arrays': 1,
    }

    if return_metrics:
        return (a_coeffs, b_coeffs, c_coeffs, d_coeffs, metrics)
    
    return a_coeffs, b_coeffs, c_coeffs, d_coeffs
