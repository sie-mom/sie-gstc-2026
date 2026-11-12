import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional, Tuple
from .solver import evaluate_incident_plane_wave
from .svwf_sphere import compute_fields


def _resolve_fontsizes(
    base: Optional[float] = None,
    fontsizes: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[float]]:
    """Resolve a consistent set of font sizes.

    Args:
        base: Baseline font size applied to all plot elements unless overridden.
        fontsizes: Optional overrides. Supported keys:
            - "title", "label", "tick", "legend", "suptitle", "colorbar_label", "colorbar_tick"

    Returns:
        Dict mapping keys to resolved sizes (or None if no sizing requested).
    """
    fs: Dict[str, Optional[float]] = {
        "title": base,
        "label": base,
        "tick": base,
        "legend": base,
        "suptitle": base,
        "colorbar_label": base,
        "colorbar_tick": base,
    }

    if fontsizes:
        for k, v in fontsizes.items():
            if k in fs and v is not None:
                fs[k] = float(v)

    return fs

def plot_field_cut_2d(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    field_magnitude: np.ndarray,
    title: str = "Field Magnitude",
    xlabel: str = "x (m)",
    ylabel: str = "z (m)",
    cmap: str = "viridis",
    save_path: str = None,
    fontsize: Optional[float] = None,
    fontsizes: Optional[Dict[str, float]] = None,
):
    """
    Plots a 2D colormap of the field magnitude.
    
    Args:
        x_grid: 2D array of x coordinates.
        z_grid: 2D array of z coordinates.
        field_magnitude: 2D array of field magnitudes.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        cmap: Colormap name.
        save_path: If provided, saves the figure to this path.
    """
    fs = _resolve_fontsizes(fontsize, fontsizes)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x_grid, z_grid, field_magnitude, shading='auto', cmap=cmap)
    cbar = plt.colorbar(label='Magnitude')
    if fs["colorbar_label"] is not None:
        cbar.set_label('Magnitude', fontsize=fs["colorbar_label"])
    if fs["colorbar_tick"] is not None:
        cbar.ax.tick_params(labelsize=fs["colorbar_tick"])

    plt.title(title, fontsize=fs["title"])  # None is fine
    plt.xlabel(xlabel, fontsize=fs["label"])  # None is fine
    plt.ylabel(ylabel, fontsize=fs["label"])  # None is fine
    if fs["tick"] is not None:
        plt.xticks(fontsize=fs["tick"])
        plt.yticks(fontsize=fs["tick"])
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def spherical_to_cartesian_vector(
    Ar: np.ndarray, 
    Ath: np.ndarray, 
    Aph: np.ndarray, 
    theta: np.ndarray, 
    phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts vector components from a spherical to a Cartesian coordinate system.
    This function is vectorized to handle arrays of points.

    Args:
        Ar, Ath, Aph: Spherical vector components (radial, polar, azimuthal).
        theta, phi: Spherical coordinate angles of the points.

    Returns:
        A tuple (Ax, Ay, Az) of the Cartesian vector components.
    """
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    
    Ax = Ar * st * cp + Ath * ct * cp - Aph * sp
    Ay = Ar * st * sp + Ath * ct * sp + Aph * cp
    Az = Ar * ct - Ath * st
    
    return Ax, Ay, Az

def plot_comparison_1d(
    coords: np.ndarray,
    field1: np.ndarray,
    field2: np.ndarray,
    label1: str = "Method 1",
    label2: str = "Method 2",
    xlabel: str = "Position",
    ylabel: str = "Field Magnitude",
    title: str = "Field Comparison",
    save_path: str = None,
    fontsize: Optional[float] = None,
    fontsizes: Optional[Dict[str, float]] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """
    Plots a 1D comparison of two fields.
    
    Args:
        coords: 1D array of coordinates.
        field1: 1D array of field values (Method 1).
        field2: 1D array of field values (Method 2).
        label1: Label for Method 1.
        label2: Label for Method 2.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        save_path: If provided, saves the figure to this path.
        ymin: Minimum limit for y-axis.
        ymax: Maximum limit for y-axis.
    """
    fs = _resolve_fontsizes(fontsize, fontsizes)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(coords, field1, label=label1, linestyle='-', marker='o', markersize=3)
    ax.plot(coords, field2, label=label2, linestyle='--', marker='x', markersize=3)
    ax.legend(fontsize=fs["legend"])
    ax.set_title(title, fontsize=fs["title"])
    ax.set_xlabel(xlabel, fontsize=fs["label"])
    ax.set_ylabel(ylabel, fontsize=fs["label"])
    if fs["tick"] is not None:
        ax.tick_params(axis="x", labelsize=fs["tick"])
        ax.tick_params(axis="y", labelsize=fs["tick"])
        
    if ymin is not None and ymax is not None:
         ax.set_ylim(ymin, ymax)
    elif ymin is not None:
         ax.set_ylim(bottom=ymin)
    elif ymax is not None:
         ax.set_ylim(top=ymax)

    # For very small/large magnitudes (e.g., |H| ~ 1e-3), use scientific notation
    # so tick labels don't become long strings that push/clamp the y-label.
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

    ax.grid(True)
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    else:
        plt.show()

def plot_comparison_2d(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    field1: np.ndarray,
    field2: np.ndarray,
    label1: str = "Method 1",
    label2: str = "Method 2",
    title: str = "Field Comparison",
    xlabel: str = "x (m)",
    ylabel: str = "z (m)",
    cmap: str = "viridis",
    save_path: str = None,
    clim: tuple = None,
    fontsize: Optional[float] = None,
    fontsizes: Optional[Dict[str, float]] = None,
):
    """
    Plots two 2D fields side-by-side with a common color scale.
    """
    fs = _resolve_fontsizes(fontsize, fontsizes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Determine common color scale, allowing user override
    if clim:
        vmin, vmax = clim
    else:
        # Handle potential NaNs or Infs gracefully
        valid_min = np.nanmin([np.nanmin(field1), np.nanmin(field2)])
        valid_max = np.nanmax([np.nanmax(field1), np.nanmax(field2)])
        vmin, vmax = valid_min, valid_max
    
    # Plot 1
    im1 = axes[0].pcolormesh(x_grid, z_grid, field1, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(label1, fontsize=fs["title"])
    axes[0].set_xlabel(xlabel, fontsize=fs["label"])
    axes[0].set_ylabel(ylabel, fontsize=fs["label"])
    if fs["tick"] is not None:
        axes[0].tick_params(axis="both", labelsize=fs["tick"])
    axes[0].set_aspect('equal')
    
    # Plot 2
    im2 = axes[1].pcolormesh(x_grid, z_grid, field2, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(label2, fontsize=fs["title"])
    axes[1].set_xlabel(xlabel, fontsize=fs["label"])
    if fs["tick"] is not None:
        axes[1].tick_params(axis="both", labelsize=fs["tick"])
    axes[1].set_aspect('equal')
    
    # Colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.05, pad=0.02, label='Magnitude')
    if fs["colorbar_label"] is not None:
        cbar.set_label('Magnitude', fontsize=fs["colorbar_label"])
    if fs["colorbar_tick"] is not None:
        cbar.ax.tick_params(labelsize=fs["colorbar_tick"])
    
    plt.suptitle(title, fontsize=fs["suptitle"])  # None is fine
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def interpolate_field_on_circle(csv_file, component, radius, num_points=360):
    """
    Interpolates field data from a CSV onto a circular path.

    Args:
        csv_file (str): Path to the CSV file.
        component (str): The data component to interpolate.
        radius (float): The radius of the circular path.
        num_points (int): The number of points to interpolate along the circle.

    Returns:
        tuple: A tuple containing (angles, interpolated_field_values).
               Returns (None, None) if an error occurs.
    """
    from scipy.interpolate import griddata
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_file)
        if component not in df.columns:
            print(f"Error: Component '{component}' not found in {csv_file}")
            return None, None
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file}")
        return None, None

    # Generate points along the circumference
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_circ = radius * np.cos(angles)
    z_circ = radius * np.sin(angles)
    circ_points = np.column_stack((x_circ, z_circ))

    # Interpolate field values at those points using cubic interpolation
    interpolated_field = griddata(
        (df['x'].values, df['z'].values),
        df[component].values,
        circ_points,
        method='cubic'
    )
    
    return angles, interpolated_field

def plot_field_from_csv(
    ax,
    csv_file,
    component_to_plot,
    title,
    vmin=None,
    vmax=None,
    cmap="jet",
    fontsize: Optional[float] = None,
    fontsizes: Optional[Dict[str, float]] = None,
    add_colorbar: bool = True,
    return_mappable: bool = False,
):
    """
    Reads field data from a CSV and plots a specified component on a
    matplotlib Axes object.

    Args:
        ax: The matplotlib Axes to plot on.
        csv_file: Path to the CSV file.
        component_to_plot: The name of the column to visualize (e.g., 'E_magnitude').
        title: The title for the subplot.
        vmin (optional): Minimum value for the color bar for consistent scaling.
        vmax (optional): Maximum value for the color bar for consistent scaling.
        cmap (optional): Matplotlib colormap instance or name. Use "comsol" for the
            COMSOL-inspired diverging palette. Defaults to "jet".
    """
    import pandas as pd
    from matplotlib.ticker import ScalarFormatter

    fs = _resolve_fontsizes(fontsize, fontsizes)

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        ax.text(
            0.5,
            0.5,
            f"File not found:\n{os.path.basename(csv_file)}",
            ha='center',
            va='center',
            color='red',
            fontsize=fs["label"] or 12,
        )
        ax.set_title(title, fontsize=fs["title"])
        if return_mappable:
            return None, None, None
        return None, None
        if return_mappable:
            return df, None, None
        return df, None

    if component_to_plot not in df.columns:
        ax.text(
            0.5,
            0.5,
            f"Component '{component_to_plot}'\nnot in file.",
            ha='center',
            va='center',
            color='red',
            fontsize=fs["label"] or 12,
        )
        ax.set_title(title, fontsize=fs["title"])
        return df, None  # Return None for color limits

    num_points = len(df)
    grid_res = int(np.sqrt(num_points))
    if grid_res * grid_res != num_points:
        ax.text(
            0.5,
            0.5,
            "Data is not a square grid.",
            ha='center',
            va='center',
            color='red',
            fontsize=fs["label"],
        )
        ax.set_title(title, fontsize=fs["title"])
        if return_mappable:
            return df, None, None
        return df, None

    # Reshape the 1D data into 2D grids for plotting
    x = df['x'].values.reshape((grid_res, grid_res))
    z = df['z'].values.reshape((grid_res, grid_res))
    field_component = df[component_to_plot].values.reshape((grid_res, grid_res))

    # Use provided vmin/vmax if they are not None, otherwise use data min/max.
    # This is crucial for side-by-side comparison.
    color_min = vmin if vmin is not None else np.min(field_component)
    color_max = vmax if vmax is not None else np.max(field_component)

    if isinstance(cmap, str):
        if cmap.lower() == "comsol":
            # Optional dependency: if you have a COMSOL-inspired custom colormap,
            # expose it as get_comsol_colormap(). Otherwise, fall back safely.
            cmap_func = globals().get("get_comsol_colormap")
            cmap_to_use = cmap_func() if callable(cmap_func) else plt.get_cmap("coolwarm")
        else:
            cmap_to_use = plt.get_cmap(cmap)
    else:
        cmap_to_use = cmap

    # Use pcolormesh for plotting. It's fast and effective for grid data.
    im = ax.pcolormesh(
        x,
        z,
        field_component,
        shading='auto',
        cmap=cmap_to_use,
        vmin=color_min,
        vmax=color_max,
    )

    fig = ax.get_figure()
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, label=f'{component_to_plot}')
        if fs["colorbar_label"] is not None:
            cbar.set_label(f'{component_to_plot}', fontsize=fs["colorbar_label"])
        if fs["colorbar_tick"] is not None:
            cbar.ax.tick_params(labelsize=fs["colorbar_tick"])

        # Use scientific notation for very small/large values so colorbar tick labels
        # don't become long (e.g., 0.00xx) and push/clamp nearby labels.
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 3))
        cbar.ax.yaxis.set_major_formatter(fmt)
        # Force a redraw so the formatter and offset text are applied.
        cbar.update_ticks()

    ax.set_title(title, fontsize=fs["title"])
    ax.set_xlabel('x (m)', fontsize=fs["label"])
    ax.set_ylabel('z (m)', fontsize=fs["label"])
    if fs["tick"] is not None:
        ax.tick_params(axis="both", labelsize=fs["tick"])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Return the dataframe and the determined color limits.
    # Optionally return the artist used for color mapping so callers can create a shared colorbar.
    if return_mappable:
        return df, (color_min, color_max), im
    return df, (color_min, color_max)

def verify_gstc_boundary_conditions(
    radius: float, 
    k0: float, eta0: float, 
    k1: float, eta1: float, 
    a_coeffs: dict, b_coeffs: dict, c_coeffs: dict, d_coeffs: dict, 
    gstc_params: dict,
    tolerance: float = 0.01,
    verbose: bool = True
):
    """
    General verification of GSTC boundary conditions for any Bianisotropic sheet.
    
    Checks:
      1. n x (H_ext - H_int) = J_surf = Y*E_av + Chi*H_av
      2. -n x (E_ext - E_int) = M_surf = Z*H_av + Gamma*E_av
    
    Args:
        radius: Sphere radius.
        k0, eta0: Exterior medium parameters.
        k1, eta1: Interior medium parameters.
        coeffs: SVWF coefficients (a,b,c,d).
        gstc_params: Dictionary with keys 'Y', 'Z', 'Chi', 'Gamma'.
        tolerance: Acceptance threshold for relative error (default 1%).
        
    Returns:
        bool: True if pass, False if fail.
    """
    if verbose:
        print("\n--- Verifying GSTC Boundary Conditions ---")
    
    # 1. Select test points (Shifted off poles to avoid singularities)
    theta_test = np.linspace(0.1, np.pi - 0.1, 30)
    phi_test = np.zeros_like(theta_test) 
    
    # 2. Compute Fields EXACTLY at the boundary (r = radius)
    # Interior
    E_in, H_in = compute_fields(k1, np.full_like(theta_test, radius), theta_test, phi_test, c_coeffs, d_coeffs, kind=1, eta=eta1, grid=False)
    # Exterior Scattered
    E_sc, H_sc = compute_fields(k0, np.full_like(theta_test, radius), theta_test, phi_test, a_coeffs, b_coeffs, kind=3, eta=eta0, grid=False)
    # Exterior Incident (CRITICAL: Pass eta=eta0)
    E_inc, H_inc = evaluate_incident_plane_wave(k0, radius, theta_test, phi_test, eta=eta0, grid=False)
    
    # Total Exterior
    E_ext = [s + i for s, i in zip(E_sc, E_inc)]
    H_ext = [s + i for s, i in zip(H_sc, H_inc)]
    
    # 3. Extract Tangential Vectors (Theta, Phi)
    E_int_tan = np.array([E_in[1], E_in[2]])
    H_int_tan = np.array([H_in[1], H_in[2]])
    E_ext_tan = np.array([E_ext[1], E_ext[2]])
    H_ext_tan = np.array([H_ext[1], H_ext[2]])
    
    # 4. GSTC Terms
    E_av = (E_ext_tan + E_int_tan) / 2.0
    H_av = (H_ext_tan + H_int_tan) / 2.0
    delta_E = E_ext_tan - E_int_tan
    delta_H = H_ext_tan - H_int_tan
    
    # Rotation n x A = [-A_phi, A_theta]
    nxDeltaH = np.array([-delta_H[1], delta_H[0]])
    nxDeltaE = np.array([-delta_E[1], delta_E[0]])
    
    # Sources
    def apply_op(Op, vec):
        if np.isscalar(Op): return Op * vec
        return np.einsum('ij,jn->in', Op, vec)

    Y = gstc_params.get('Y', 0.0)
    Z = gstc_params.get('Z', 0.0)
    Chi = gstc_params.get('Chi', 0.0)
    Gam = gstc_params.get('Gamma', 0.0)
    
    Js = apply_op(Y, E_av) + apply_op(Chi, H_av)
    Ms = apply_op(Z, H_av) + apply_op(Gam, E_av)
    
    # 5. Residuals
    # Eq 1: n x DeltaH - Js = 0
    res_J_vec = nxDeltaH - Js
    # Eq 2: -n x DeltaE - Ms = 0  => n x DeltaE + Ms = 0
    res_M_vec = nxDeltaE + Ms
    
    # 6. Normalization & Metrics
    def magnitude(vec): return np.sqrt(np.abs(vec[0])**2 + np.abs(vec[1])**2)
    
    err_J = magnitude(res_J_vec)
    err_M = magnitude(res_M_vec)
    
    # Normalize by local field magnitude (plus epsilon)
    norm_H = magnitude(H_ext_tan) + 1e-15
    norm_E = magnitude(E_ext_tan) + 1e-15
    
    # Use Median instead of Mean to be robust against nodes (points where field is zero)
    rel_err_J = np.median(err_J / norm_H)
    rel_err_M = np.median(err_M / norm_E)
    
    if verbose:
        print(f"GSTC Residual (Current BC): {rel_err_J:.2e}")
        print(f"GSTC Residual (Voltage BC): {rel_err_M:.2e}")
        if rel_err_J < tolerance and rel_err_M < tolerance:
            print(">> SUCCESS: Boundary conditions satisfied.")
        else:
            print(">> WARNING: High residuals.")

    return (rel_err_J < tolerance) and (rel_err_M < tolerance)