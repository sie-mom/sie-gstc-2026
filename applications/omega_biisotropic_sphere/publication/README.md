# Publication Reproduction: Example 1 (Dipole Impedance Matching)

This directory contains the automated end-to-end workflow to reproduce the numerical simulations and publication figures for **Example 1** (Budhu et al., $10\text{ GHz}$, $\varepsilon_{r,\text{int}} = 7.0$).

---

## 1. Physical Problem Overview

* **Geometry & Material**: A dielectric sphere of radius $a = 0.7\lambda_0 \approx 20.985\text{ mm}$ with relative permittivity $\varepsilon_r = 7.0$, $\mu_r = 1.0$ at $f = 10\text{ GHz}$ located in free space ($\varepsilon_0, \mu_0$).
* **Excitation**: A centered, $z$-directed electric dipole ($p_z = 1.0$) in the interior.
* **Objective**:
  * **Without metasurface (Bare sphere)**: High dielectric contrast creates strong internal reflection and standing waves.
  * **With omega-biisotropic metasurface**: The boundary conditions eliminate reflections and match the interior dipole to a power-conserved exterior radiation field ($p_{\text{ext}} = 11.386$).

---

## 2. Prerequisites

### C++ Solvers
Ensure the C++ executables are compiled in `build/`:
```bash
cmake -B build -S .
cmake --build build --target pmchwt_dielectric_sphere_app omega_sphere_sie -j
```

### Python Environment
Activate the repository virtual environment:
```bash
source .venv/bin/activate
```

---

## 3. One-Click Reproduction

From the repository root, run:
```bash
./applications/omega_biisotropic_sphere/publication/run_example_1.sh
```

Or invoke the Python runner directly:
```bash
python applications/omega_biisotropic_sphere/publication/run_example_1.py
```

### Optional Flags
* `--plots-only`: Skip solver runs and re-generate figures from existing data:
  ```bash
  ./applications/omega_biisotropic_sphere/publication/run_example_1.sh --plots-only
  ```
* `--skip-pmchwt`: Skip the bare-sphere PMCHWT simulation.
* `--skip-sie`: Skip the SIE-GSTC omega metasurface simulation.
* `--no-residual`: Skip post-solve residual reassembly to accelerate simulation time.

---

## 4. Pipeline Steps

1. **Bare Dielectric Sphere Simulation (`pmchwt_dielectric_sphere_app`)**:
   Solves the penetrable dielectric sphere using the PMCHWT Surface Integral Equation formulation with identical RWG mesh and quadrature rules to establish the baseline response without metasurface.
   * *Output*: `applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7/pmchwt/bare_sphere/`

2. **Metasurface-Coated Sphere Simulation (`omega_sphere_sie`)**:
   Synthesizes the omega-biisotropic surface parameters ($\mathbf{Y}_{ee}, \mathbf{Y}_{mm}, \mathbf{Y}_{em}$) and solves the coupled SIE-GSTC system.
   * *Output*: `applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7/sie/least_squares/`

3. **Publication Figure Export (`plot_example_1.py`)**:
   Reads the computed CSV fields, projects to dominant spherical tangential components ($E_\theta$), applies physical Budhu moment normalization ($I\Delta z = 1\text{ A}\cdot\text{m}$), and renders publication figures in `figures/`.

---

## 5. Generated Figures

Output location: `applications/omega_biisotropic_sphere/publication/figures/`

* **Figure 1 (2D Tangential Field Maps $|E_\theta|$ in dB for $x \ge 0$)**:
  * Panel (a): Without metasurface (PMCHWT MoM).
  * Panel (b): With metasurface (SIE-GSTC MoM).
  * Format: Compact single-column side-by-side layout ($3.5\text{ in}$ width, 1:1 aspect ratio) showing the right half-plane ($x \ge 0, z \in [-2, 2]$) with shared $z$-axis and colorbar.
  * Files: `figure_1_2d_field_maps.pdf` and `figure_1_2d_field_maps.png`.

* **Figure 2 (1D Radial Cut Comparison $\operatorname{Re}(E_\theta)$ along $\theta = 45^\circ$)**:
  * Curves: Prescribed Target, SIE-GSTC (MoM), SVWF Collocation, and COMSOL (FEM).
  * Files: `figure_2_radial_cut_comparison.pdf` and `figure_2_radial_cut_comparison.png`.

