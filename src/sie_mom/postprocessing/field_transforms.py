"""Field transformations and spherical projections for SIE-MoM postprocessing."""

from __future__ import annotations

from typing import Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def cartesian_to_spherical(
    field: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
) -> NDArray[np.complex128]:
    """Project a Cartesian vector array onto the spherical (r, theta, phi) basis.

    Parameters
    ----------
    field:
        Cartesian components (A_x, A_y, A_z). Can be component-first shape
        (3, ...) or row-wise shape (..., 3).
    theta, phi:
        Spherical polar and azimuthal angles in radians. Broadcast-compatible with
        the sample dimensions of field.

    Returns
    -------
    NDArray[np.complex128]:
        Component-first array (A_r, A_theta, A_phi) of shape (3, ...).
    """
    field_arr = np.asarray(field, dtype=complex)
    if field_arr.shape[0] == 3:
        ax, ay, az = field_arr[0], field_arr[1], field_arr[2]
    elif field_arr.shape[-1] == 3:
        ax, ay, az = field_arr[..., 0], field_arr[..., 1], field_arr[..., 2]
    else:
        raise ValueError("field must have 3 components either first or last")

    th = np.asarray(theta, dtype=float)
    ph = np.asarray(phi, dtype=float)
    st, ct = np.sin(th), np.cos(th)
    sp, cp = np.sin(ph), np.cos(ph)

    ar = ax * st * cp + ay * st * sp + az * ct
    atheta = ax * ct * cp + ay * ct * sp - az * st
    aphi = -ax * sp + ay * cp
    return np.stack((ar, atheta, aphi), axis=0)


def spherical_to_cartesian(
    field: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
) -> NDArray[np.complex128]:
    """Convert spherical (r, theta, phi) vector components to Cartesian (x, y, z).

    Parameters
    ----------
    field:
        Spherical components (A_r, A_theta, A_phi). Can be component-first (3, ...)
        or row-wise (..., 3).
    theta, phi:
        Spherical angles in radians.

    Returns
    -------
    NDArray[np.complex128]:
        Component-first array (A_x, A_y, A_z) of shape (3, ...).
    """
    field_arr = np.asarray(field, dtype=complex)
    if field_arr.shape[0] == 3:
        ar, atheta, aphi = field_arr[0], field_arr[1], field_arr[2]
    elif field_arr.shape[-1] == 3:
        ar, atheta, aphi = field_arr[..., 0], field_arr[..., 1], field_arr[..., 2]
    else:
        raise ValueError("field must have 3 components either first or last")

    th = np.asarray(theta, dtype=float)
    ph = np.asarray(phi, dtype=float)
    st, ct = np.sin(th), np.cos(th)
    sp, cp = np.sin(ph), np.cos(ph)

    ax = ar * st * cp + atheta * ct * cp - aphi * sp
    ay = ar * st * sp + atheta * ct * sp + aphi * cp
    az = ar * ct - atheta * st
    return np.stack((ax, ay, az), axis=0)


def complex_vector(data: dict[str, np.ndarray], field: str) -> NDArray[np.complex128]:
    """Stack Cartesian real and imaginary column components into (N, 3) complex vectors.

    Supports both 'Ex_real'/'Ex_imag' and 'E_x_re'/'E_x_im' column formats.
    """
    vectors = []
    for axis in ("x", "y", "z"):
        if f"{field}{axis}_real" in data:
            re = data[f"{field}{axis}_real"]
            im = data[f"{field}{axis}_imag"]
        elif f"{field}_{axis}_re" in data:
            re = data[f"{field}_{axis}_re"]
            im = data[f"{field}_{axis}_im"]
        else:
            raise KeyError(f"no columns found for field '{field}' axis '{axis}'")
        vectors.append(re + 1j * im)
    return np.column_stack(vectors)


def spherical_coordinates_from_cartesian(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute (r, theta, phi) from Cartesian coordinates with safe origin handling."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    radii = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
    cosine = np.divide(z_arr, radii, out=np.ones_like(radii), where=radii > 0)
    theta = np.arccos(np.clip(cosine, -1.0, 1.0))
    phi = np.arctan2(y_arr, x_arr)
    # Set phi to 0 on the z-axis where x=0 and y=0
    rho = np.hypot(x_arr, y_arr)
    phi = np.where(rho == 0.0, 0.0, phi)
    return radii, theta, phi


def dominant_tangential_components(
    data: dict[str, np.ndarray],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Extract E_theta and H_phi from a field dictionary."""
    radii, theta, phi = spherical_coordinates_from_cartesian(
        data["x"], data["y"], data["z"]
    )
    e_cart = complex_vector(data, "E")
    h_cart = complex_vector(data, "H")

    e_sph = cartesian_to_spherical(e_cart.T, theta, phi)
    h_sph = cartesian_to_spherical(h_cart.T, theta, phi)

    e_theta = e_sph[1]
    h_phi = h_sph[2]
    return e_theta, h_phi


def spherical_components(
    data: dict[str, np.ndarray],
    field: str,
    *,
    mask_origin: bool = False,
) -> NDArray[np.complex128]:
    """Project Cartesian field data onto (r, theta, phi) returning shape (N, 3)."""
    radii, theta, phi = spherical_coordinates_from_cartesian(
        data["x"], data["y"], data["z"]
    )
    if np.any(radii == 0) and not mask_origin:
        raise ValueError("spherical components are undefined at the sphere center")
    cart = complex_vector(data, field)
    sph = cartesian_to_spherical(cart.T, theta, phi).T  # (N, 3)
    if mask_origin:
        sph[radii == 0] = np.nan
    return sph


def to_decibels(
    values: ArrayLike,
    *,
    floor_relative_db: float = 80.0,
    reference: float | None = None,
) -> NDArray[np.float64]:
    """Convert field values to decibels 20*log10(|val| / ref).

    If reference is None, uses max(|values|). Clamps at floor_relative_db below peak.
    """
    arr = np.abs(np.asarray(values))
    finite = arr[np.isfinite(arr)]
    if not finite.size or np.max(finite) == 0.0:
        return np.full_like(arr, -floor_relative_db, dtype=float)

    ref = float(np.max(finite)) if reference is None else float(reference)
    min_val = ref * (10.0 ** (-floor_relative_db / 20.0))
    clipped = np.maximum(arr, min_val)
    return 20.0 * np.log10(clipped / ref)


def zx_grid(
    data: dict[str, np.ndarray],
    grid_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Validate a complete ZX grid and return unique axes and coordinate order."""
    x, z = np.unique(data["x"]), np.unique(data["z"])
    if len(x) != grid_size or len(z) != grid_size or len(data["x"]) != grid_size**2:
        raise ValueError("ZX field coordinates do not match the saved grid size")
    if "y" in data and not np.all(data["y"] == 0):
        raise ValueError("ZX field samples must lie in the y=0 plane")
    order = np.lexsort((data["x"], data["z"]))
    expected_x, expected_z = np.meshgrid(x, z)
    if not (
        np.array_equal(data["x"][order], expected_x.ravel())
        and np.array_equal(data["z"][order], expected_z.ravel())
    ):
        raise ValueError("ZX field coordinates contain duplicate or missing grid points")
    return x, z, order


def budhu_moment_scale(manifest: dict) -> complex:
    r"""Compute conversion scale from effective moment to physical current moment I Delta z = 1 A m.

    This function provides the exact analytical conversion between the
    dimensionless effective source amplitude (:math:`p_{\mathrm{eff}} = 1.0`)
    used internally by the integral equation solvers and the physical electric
    current moment (:math:`I\Delta z = 1\text{ A}\cdot\text{m}`) used by COMSOL
    and Budhu & Dault (2026).

    Mathematical Derivation
    -----------------------
    In the SIE/SVWF source model (:func:`sie_mom.excitation.evaluate_dipole_fields`),
    the effective moment is defined as :math:`p_{\mathrm{eff}} = p_{\mathrm{phys}} / (4\pi\varepsilon)`,
    omitting the :math:`1/(4\pi\varepsilon)` factor.
    Under the :math:`e^{+j\omega t}` convention, an electric current dipole of
    moment :math:`I\Delta z` corresponds to a physical charge dipole moment
    :math:`p_{\mathrm{phys}} = I\Delta z / (j\omega) = -j I\Delta z / \omega`.
    Using the medium identity :math:`\omega\varepsilon = k / \eta`:

    .. math::

        p_{\mathrm{eff}} = \frac{-j I\Delta z}{4\pi\omega\varepsilon} = \frac{-j\eta}{4\pi k} (I\Delta z).

    Therefore, to convert fields computed with an arbitrary effective moment
    :math:`p_{\mathrm{eff}}` to the physical fields of a :math:`1\text{ A}\cdot\text{m}`
    current dipole, the required scale factor is:

    .. math::

        \text{scale} = \frac{-j\,\eta_{\mathrm{int}}}{4\pi k_{\mathrm{int}} \cdot p_{\mathrm{eff}}}.

    Solver Normalization Roles in Publication Comparisons
    -----------------------------------------------------
    * **Solvers multiplied by this scale factor**:
      - Prescribed / Target fields (:func:`sie_mom.excitation.evaluate_dipole_fields`)
      - SIE-GSTC (MoM) solver (:file:`omega_sphere_sie` / :file:`anisotropic_sphere_app`)
      - SVWF collocation solver
      - PMCHWT bare dielectric sphere solver (:file:`pmchwt_dielectric_sphere_app`)
    * **Solvers NOT multiplied by this scale factor (already physical)**:
      - COMSOL Multiphysics (configured with ``ElectricPointDipole`` parameter
        ``pI = 1.0 [A*m]``, which natively solves Maxwell's equations in physical
        V/m and A/m units).
    """
    from scipy.constants import epsilon_0, mu_0

    freq = float(manifest.get("frequency_hz", manifest.get("frequency", 1e10)))
    omega = 2.0 * np.pi * freq

    interior = manifest.get("interior", {})
    eps_r = float(interior.get("epsilon_r", interior.get("relative_permittivity", 1.0)))
    mu_r = float(interior.get("mu_r", interior.get("relative_permeability", 1.0)))

    epsilon = eps_r * epsilon_0
    mu = mu_r * mu_0
    k = omega * np.sqrt(mu * epsilon)
    eta = np.sqrt(mu / epsilon)

    source = manifest.get("source", {})
    moment = complex(source.get("effective_moment_interior", 1.0))
    return -1j * eta / (4.0 * np.pi * k * moment)