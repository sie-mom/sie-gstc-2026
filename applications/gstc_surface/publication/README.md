# Publication Workflow: Example 3 (Circular-Arc Spindle GSTC Metasurface)

This directory provides the high-level, cross-platform reproduction and publication plotting scripts for **Example 3**: the 50% elongated circular-arc spindle (*ogive*, length-to-diameter ratio $L/D = 1.5$) GSTC metasurface at 2.0 GHz with dual polarization transformations.

---

## 1. Physical Case Summary

* **Geometry**: Circular-arc spindle along axis $\hat{\mathbf{x}}$ with maximum radius $a = 0.1049\text{ m}$ (diameter $D = 0.2099\text{ m}$), tip-to-tip length $L = 0.3148\text{ m}$ ($L/D = 1.5$).
* **Media**: Interior dielectric $\varepsilon_{r,\text{int}} = 5.0$, $\mu_{r,\text{int}} = 1.0$ ($\lambda_d \approx 0.0671\text{ m}$); exterior free space $\varepsilon_{r,\text{ext}} = 1.0$.
* **Discretization**: Curvature-adaptive tip-graded mesh with $\text{PPW} = 5$ ($3{,}766$ triangular elements, tip size factor $0.25$).
* **Dual Transformations**:
  1. `electric_z_incidence`: Centered $z$-directed electric dipole (interior) $\to$ exterior radiation (active transverse electric component in $zx$-plane is $E_\theta$).
  2. `electric_y_incidence`: Centered $y$-directed electric dipole (interior) $\to$ exterior radiation (active transverse electric component in $zx$-plane is $E_\phi$).

---

## 2. High-Level Orchestrator (`run_example_3.py`)

A single cross-platform Python runner coordinates the end-to-end pipeline: tensor synthesis, GPU LU solves, observation grid classification, near-field integration, bare dielectric PMCHWT baseline calculation, optional NGSolve FEM validation, and publication figure generation.

### Quick Commands

#### 1. Generate Figures from Existing Data (Fastest)
Skips all numerical solves and renders all vector PDFs and 300 DPI PNGs:
```bash
python -m applications.gstc_surface.publication.run_example_3 --plots-only
```

#### 2. Incremental / Resuming Execution
Executes only missing stages, reusing existing meshes, solves, and field evaluations:
```bash
python -m applications.gstc_surface.publication.run_example_3
```

#### 3. Full End-to-End Simulation (`--force`)
Wipes previous intermediate results and recomputes tensor preparation, batch GPU LU solve, PMCHWT baseline, and field integrations:

* **Reusing existing mesh** (default behavior, preserves `e50_ppw5_tip025.obj`):
  ```bash
  python -m applications.gstc_surface.publication.run_example_3 --force
  ```

* **Regenerating surface mesh from scratch** (forces Gmsh to regenerate the curvature-adaptive `.obj`):
  ```bash
  python -m applications.gstc_surface.publication.run_example_3 --force --regenerate-mesh
  ```

#### 4. End-to-End with Independent NGSolve FEM Cross-Validation
Includes the independent Order-2 NGSolve FEM volume solver (with 16 CPU threads):

* **Cached / incremental**:
  ```bash
  python -m applications.gstc_surface.publication.run_example_3 --with-ngsolve --threads 16
  ```

* **Full rerun with existing mesh**:
  ```bash
  python -m applications.gstc_surface.publication.run_example_3 --with-ngsolve --threads 16 --force
  ```

* **Full rerun with fresh mesh regeneration**:
  ```bash
  python -m applications.gstc_surface.publication.run_example_3 --with-ngsolve --threads 16 --force --regenerate-mesh
  ```

### CLI Option Reference

| Option | Default | Description |
| :--- | :--- | :--- |
| `--plots-only` | `False` | Skip simulation stages 1–4C; generate publication figures from existing files. |
| `--force` | `False` | Force re-run of simulation stages, wiping previous intermediate directories. |
| `--use-existing-mesh` | `True` | Reuse existing `.obj` surface mesh if present instead of invoking Gmsh. |
| `--regenerate-mesh` | `False` | Force regeneration of the surface mesh via Gmsh, overwriting existing `.obj`. |
| `--with-ngsolve` | `False` | Execute independent Order-2 NGSolve FEM validation (Stage 4B). |
| `--threads N` | `16` | Number of CPU threads for OpenMP, MKL, and NGSolve parallel execution. |
| `--skip-pmchwt` | `False` | Skip Stage 4C bare dielectric baseline simulation (PMCHWT). |
| `--no-residual` | `False` | Skip algebraic residual verification during the PMCHWT baseline solve. |
| `--figures-dir PATH`| `figures/` | Destination directory for output publication figures (PDF & PNG). |

---

## 3. Dedicated Publication Plotting Scripts

### 2D Near-Field Maps and 1D Radial Cuts (Figures 1 and 2)
Generates side-by-side 2D $zx$-plane field maps in normalized dB ($[-30, +10]\text{ dB}$) comparing the bare dielectric baseline (PMCHWT, no metasurface) against the synthesized metasurface (SIE-GSTC), complete with the triangulated mesh outline, dipole indicators (vertical arrow $\uparrow$ for $z$-dipole, solid dot $\bullet$ for $y$-dipole), and shared colorbars. Also generates 1D radial line cuts ($\theta = 45^\circ, \phi = 0^\circ$) using a two-tier layout with point-by-point difference subpanels:

```bash
python -m applications.gstc_surface.publication.plot_fields_publication \
  --fields-dir applications/gstc_surface/results/ogive/e50_ppw5_tip025/run01/sie/fields/run01 \
  --pmchwt-dir applications/gstc_surface/results/ogive/e50_ppw5_tip025/run01/pmchwt \
  --ngsolve-fields-dir applications/gstc_surface/results/ogive/e50_ppw5_tip025/run01/ngsolve/order2_run01/fields/run01 \
  --output-dir applications/gstc_surface/publication/figures \
  --vmin-db -30.0 \
  --vmax-db 10.0
```

### 3D Surface Parameter Maps (Figure 3)
Renders the synthesized 10-parameter physical GSTC tensors on the triangulated spindle using a single broadside orthographic view ($+y$ view) to minimize vertical space in the journal layout:

```bash
python -m applications.gstc_surface.publication.plot_parameters_publication \
  --inputs-dir applications/gstc_surface/results/ogive/e50_ppw5_tip025/run01/inputs \
  --output-dir applications/gstc_surface/publication/figures/surface_parameters \
  --percentile 98 \
  --show-edges
```

---

## 4. Generated Publication Figures

Outputs are saved in vector PDF and 300 DPI raster PNG under `figures/`:

| File | Description | Layout |
| :--- | :--- | :--- |
| `figures/figure_1_combined_2x2_field_maps.pdf` | Dual transformation comparison: Row 1 Trans. 1 (without vs. with), Row 2 Trans. 2 (without vs. with) | 4 panels (2x2), double column |
| `figures/figure_1a_case1_side_by_side.pdf` | Transformation 1 ($z$-dipole, $|E_\theta|$): (a) Without metasurface (PMCHWT) vs. (b) With metasurface (SIE-GSTC) | 2 panels (1x2), double column |
| `figures/figure_1b_case2_side_by_side.pdf` | Transformation 2 ($y$-dipole, $|E_\phi|$): (a) Without metasurface (PMCHWT) vs. (b) With metasurface (SIE-GSTC) | 2 panels (1x2), double column |
| `figures/figure_2_radial_cut_comparison.pdf` | 1D radial cuts ($\theta=45^\circ$): Real parts $\operatorname{Re}(E_\theta)$ and $\operatorname{Re}(E_\phi)$ with point-by-point difference subpanels | 2 panels, two-tier layout |
| `figures/figure_2_radial_cut_magnitude.pdf` | Companion 1D radial cuts: Magnitudes $|E_\theta|$ and $|E_\phi|$ with difference subpanels | 2 panels, two-tier layout |
| `figures/surface_parameters/figure_3_combined_surface_parameters.pdf` | Single-row 3D parameter overview: $\operatorname{Im}(Y_{11}), \operatorname{Im}(Z_{11}), \operatorname{Re}(\chi_{12})$, residual | 4 panels, double column |
| `figures/surface_parameters/figure_3a_surface_parameters_admittance.pdf` | Electric admittance block $\operatorname{Im}(Y_{11}, Y_{12}, Y_{22})$ [S] | 3 panels, single row |
| `figures/surface_parameters/figure_3b_surface_parameters_impedance.pdf` | Magnetic impedance block $\operatorname{Im}(Z_{11}, Z_{12}, Z_{22})$ [$\Omega$] | 3 panels, single row |
| `figures/surface_parameters/figure_3c_surface_parameters_coupling.pdf` | Magnetoelectric coupling block $\operatorname{Re}(\chi_{11}, \chi_{12}, \chi_{21}, \chi_{22})$ | 4 panels (2x2 grid) |
| `figures/surface_parameters/figure_3d_surface_parameters_residual.pdf` | Per-face tensor fitting relative residual distribution | 1 panel, single column |
