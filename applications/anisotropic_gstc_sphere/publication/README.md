# Publication Reproduction: Example 2 (Anisotropic Dipole Polarization Rotator)

This directory contains the automated end-to-end workflows to reproduce the numerical simulations and publication figures for **Example 2** (General Anisotropic GSTC Sphere, $10\text{ GHz}$, $\varepsilon_{r,\text{int}} = 7.0$):
* **Part A**: Unilateral polarization rotation ($\mathbf{p} \to \mathbf{m}$ under electric incidence).
* **Part B**: Dual polarization rotator ($\mathbf{p} \leftrightarrow \mathbf{m}$ under both electric and magnetic incidence).

---

## 1. Physical Problem Overview

### Common Geometry & Material Parameters
* **Host Geometry**: Penetrable dielectric sphere of radius $a = 0.7\lambda_0 \approx 20.985\text{ mm}$ operating at $f = 10\text{ GHz}$ ($\lambda_0 = 30.0\text{ mm}$).
* **Media**:
  - Interior ($r < a$): Dense dielectric with $\varepsilon_{r,\text{int}} = 7.0$, $\mu_{r,\text{int}} = 1.0$ ($\eta_{\text{int}} \approx 142.4\,\Omega$, $\lambda_d \approx 11.34\text{ mm}$).
  - Exterior ($r > a$): Free space with $\varepsilon_{r,\text{ext}} = 1.0$, $\mu_{r,\text{ext}} = 1.0$ ($\eta_0 \approx 376.7\,\Omega$, $\lambda_0 = 30.0\text{ mm}$).
* **Discretization**: Validated high-resolution mesh with points-per-wavelength $\text{PPW} = 6$ in dielectric ($h \approx 1.89\text{ mm}$, 3,796 triangular faces, 22,776 dense unknowns, $7.73\text{ GiB}$ VRAM).

### Example 2, Part A: Unilateral Electric-to-Magnetic Rotation
* **Excitation**: Centered, $z$-directed electric dipole ($\mathbf{p} = p_0\hat{\mathbf{z}}$, TM mode) in the interior.
* **Objective**: Transform interior TM dipole field into pure TE magnetic-dipole radiation in the exterior ($\mathbf{m} = m_0\hat{\mathbf{z}}$ with power-matched moment $m_0 = 7^{1.25} \approx 11.386$).
* **Degrees of Freedom**: 1 transformation provides 4 scalar boundary constraints on the 10 reciprocal lossless parameters (underdetermined synthesis).

### Example 2, Part B: Dual Polarization Rotator (Complementary State)
* **Transformations Prescribed Simultaneously**:
  1. `electric_incidence`: Interior electric dipole $\mathbf{p} = 1.0\hat{\mathbf{z}} \to$ exterior magnetic dipole $\mathbf{m} = 11.386\hat{\mathbf{z}}$ ($\text{TM} \to \text{TE}$).
  2. `magnetic_incidence`: Interior magnetic dipole $\mathbf{m} = 1.0\hat{\mathbf{z}} \to$ exterior electric dipole $\mathbf{p} = 11.386\hat{\mathbf{z}}$ ($\text{TE} \to \text{TM}$).
* **Physical Duality**: Under magnetic incidence, the roles of $\theta$ and $\phi$ reverse cleanly: the interior contains a pure TE mode ($E_\phi \neq 0, E_\theta = 0$), and the exterior radiation donut is pure TM ($E_\theta \neq 0, E_\phi = 0$).
* **Degrees of Freedom**: 2 transformations provide $4 + 4 = 8$ scalar boundary constraints on the 10 reciprocal lossless parameters. A single, passive-lossless metasurface achieves both states simultaneously.

---

## 2. Prerequisites

### C++ Solver
Ensure the spherical anisotropic GSTC executable is compiled in `build/`:
```bash
cmake -B build -S .
cmake --build build --target anisotropic_sphere_app -j
```

### Python Environment
Activate the repository virtual environment:
```bash
source .venv/bin/activate
```

---

## 3. Reproduction Commands

### Part A: Unilateral Polarization Rotator
To run the complete Part A reproduction pipeline (C++ SIE + SVWF modal benchmark + figures + surface parameters):
```bash
./applications/anisotropic_gstc_sphere/publication/run_example_2_part_a.sh
```
Or via Python:
```bash
.venv/bin/python applications/anisotropic_gstc_sphere/publication/run_example_2_part_a.py
```

### Part B: Dual Polarization Rotator
To run the complete Part B reproduction pipeline (C++ SIE dual batch solve + SVWF modal benchmarks for both cases + all single and combined figures + Part B surface parameters):
```bash
./applications/anisotropic_gstc_sphere/publication/run_example_2_part_b.sh
```
Or via Python:
```bash
.venv/bin/python applications/anisotropic_gstc_sphere/publication/run_example_2_part_b.py
```

> **Note on Caching**: If simulation results already exist in `results/...`, the pipeline skips re-running the C++ solver to save time. To force a clean re-solve from scratch, pass `--force`:
> ```bash
> ./applications/anisotropic_gstc_sphere/publication/run_example_2_part_b.sh --force
> ```

---

### Generating Figures Only (from existing simulation data)

To regenerate or re-style figures without touching the solvers:
```bash
# Part A:
.venv/bin/python applications/anisotropic_gstc_sphere/publication/plot_example_2_part_a.py

# Part B (generates all individual cases and combined panels):
.venv/bin/python applications/anisotropic_gstc_sphere/publication/plot_example_2_part_b.py
```

### Command-Line Options
Both runners and plotting scripts support:
* `--vmin-db <float>`: Colorbar minimum for electric incidence in normalized dB (default: `-112.0`).
* `--vmax-db <float>`: Colorbar maximum for electric incidence in normalized dB (default: `-58.0`).
* `--mag-vmin-db <float>`: Colorbar minimum for magnetic incidence in normalized dB (default: `-75.0` for origin mode, `-30.0` for radiated mode).
* `--mag-vmax-db <float>`: Colorbar maximum for magnetic incidence in normalized dB (default: `-45.0` for origin mode, `0.0` for radiated mode).
* `--norm {origin, radiated}`: Normalization reference: `'origin'` (core singularity peak) or `'radiated'` (exterior radiated peak = 0 dB).
* `--cmap <str>`: Colormap for 2D field maps (default: `viridis`; options include `turbo`, `inferno`, `plasma`).
* `--dpi <int>`: PNG output resolution (default: `300`).
* `--force` *(runner only)*: Clean existing simulation directory and re-run C++ SIE solve from scratch.
* `--plots-only` *(runner only)*: Skip solver runs and only render the figures.
* `--skip-parameters` *(runner only)*: Skip metasurface constitutive parameter postprocessing plots.

#### Recommended Colorbar Dynamic Ranges:
1. **Electric Incidence State (Case 1)**:
   - Dynamic range: `[-112, -58] dB` ($54\text{ dB}$ span).
   - Peak radiated donut is at $-60.5\text{ dB}$ (reaching light green/yellow near the top of the colorbar).
2. **Magnetic Incidence State (Case 2)**:
   - Dynamic range: `[-100, -40] dB` ($60\text{ dB}$ span with clean tick multiples of 10) or `[-97, -43] dB` (exact $14.8\text{ dB}$ shift preserving the identical $54\text{ dB}$ span).
   - Prevents flat yellow saturation caused by the lower numerical origin peak ($1/r^2$ vs $1/r^3$ singularity).
   - Radiated donut peaks smoothly at $-45.65\text{ dB}$ with clear gradient transitions down to the deep purple suppression nulls.
3. **Combined 4-Panel Map**:
   - The runner automatically aligns the exterior radiated peaks of both states to a single shared colorbar (`[-112, -58] dB`), producing symmetrical, unsaturated field maps across all four panels.

---

## 4. Generated Figures Summary

All publication figures are saved to `applications/anisotropic_gstc_sphere/publication/figures/` in both vector `.pdf` and $300\text{ DPI}$ `.png`:

### Part A Figures
* `figure_1_part_a_2d_field_maps.pdf / .png`: 2-panel 2D map ($|E_\theta|$ TM vs. $|E_\phi|$ TE).
* `figure_2_part_a_radial_cut_transformation.pdf / .png`: 1D radial cut (Target vs. SIE vs. SVWF).
* `surface_parameters/`: Constitutive tensor maps ($\mathbf{Y}_{ee}, \mathbf{Z}_{mm}, \boldsymbol{\chi}$) and condition number $\kappa(M)$.

### Part B Figures
* `figure_1_part_b_magnetic_incidence_2d_field_maps.pdf / .png`: Individual 2D map for Case 2 ($|E_\phi|$ TE interior vs. $|E_\theta|$ TM exterior, calibrated `[-100, -40] dB`).
* `figure_1_part_b_electric_incidence_2d_field_maps.pdf / .png`: Individual 2D map for Case 1 ($|E_\theta|$ TM interior vs. $|E_\phi|$ TE exterior, `[-112, -58] dB`).
* `figure_1_part_b_combined_2d_field_maps.pdf / .png`: Comprehensive 4-panel (2x2) dual comparison showcasing both cases and full field duality.
* `figure_2_part_b_magnetic_incidence_radial_cut.pdf / .png`: Individual 1D radial cut for Case 2 properly framed to $[-30, +10]\times 10^5\text{ V/m}$.
* `figure_2_part_b_electric_incidence_radial_cut.pdf / .png`: Individual 1D radial cut for Case 1 properly framed to $[-8, +18]\times 10^5\text{ V/m}$.
* `figure_2_part_b_radial_cut_transformation.pdf / .png`: Side-by-side 2-panel 1D radial cut comparing Case 1 and Case 2 against Target, SIE, and SVWF.
* `surface_parameters_part_b/`: Dual metasurface constitutive parameter maps ($\mathbf{Y}_{ee}, \mathbf{Z}_{mm}, \boldsymbol{\chi}$) and condition number $\kappa(M)$.

* **Figure 1: 2D Field Maps ($|E_\theta|$ vs. $|E_\phi|$)**:
  * Panel (a): $|E_\theta|$ [TM mode]. Shows the active electric dipole inside ($r < a$) and extinguishment outside ($r > a$, $> 50\text{ dB}$ down).
  * Panel (b): $|E_\phi|$ [TE mode]. Shows extinguished field inside ($r < a$, $> 37\text{ dB}$ peak, $> 50\text{ dB}$ median down) and the radiating magnetic dipole donut in the exterior ($r > a$).
  * Format: Side-by-side vertical half-space ($x \ge 0, z/a \in [-2, 2]$) sharing a single colorbar ($[-112, -58]\text{ dB}$).
  * Files: `figure_1_part_a_2d_field_maps.pdf` and `figure_1_part_a_2d_field_maps.png`.

* **Figure 2: 1D Radial Cut Transformation ($\theta = 45^\circ, \phi = 0^\circ$)**:
  * Unified piecewise radial representation:
    - $r/a \le 1.0$: $\operatorname{Re}(E_\theta)$ (dominant TM mode).
    - $r/a > 1.0$: $\operatorname{Re}(E_\phi)$ (dominant TE mode).
  * Curves: Prescribed Target (black solid), SIE-GSTC MoM (orange dashed), and SVWF Collocation (green dotted).
  * Physical Scale: Scaled to Budhu dipole moment ($I\Delta z = 1\text{ A}\cdot\text{m}$, plotted as $\times 10^5\text{ V/m}$).
  * Files: `figure_2_part_a_radial_cut_transformation.pdf` and `figure_2_part_a_radial_cut_transformation.png`.

* **Metasurface Constitutive Parameter Maps (`figures/surface_parameters/`)**:
  * `Y_imaginary.pdf` / `.png`: Surface admittance $\operatorname{Im} Y_{\theta\theta}, \operatorname{Im} Y_{\theta\phi}, \operatorname{Im} Y_{\phi\phi}$ [S].
  * `Z_imaginary.pdf` / `.png`: Surface impedance $\operatorname{Im} Z_{\theta\theta}, \operatorname{Im} Z_{\theta\phi}, \operatorname{Im} Z_{\phi\phi}$ [$\Omega$].
  * `chi_real.pdf` / `.png`: Magneto-electric coupling $\operatorname{Re} \chi_{\theta\theta}, \operatorname{Re} \chi_{\theta\phi}, \operatorname{Re} \chi_{\phi\theta}, \operatorname{Re} \chi_{\phi\phi}$ [dimensionless].
  * `map_condition_number.pdf` / `.png`: Balanced condition number $\kappa(M)$ on logarithmic scale.
  * `face_parameters.csv`: Tabular per-face coordinates, angles, and tensor entries.
  * `parameter_checks.json`: Automated passivity/losslessness and condition number validation statistics.

