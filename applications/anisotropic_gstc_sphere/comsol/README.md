# COMSOL Multiphysics Validation Suite for Bianisotropic Spherical Metasurfaces

This suite provides a modular, automated, and mathematically verified pipeline to validate the **Surface Integral Equation Boundary Element Method (SIE-BEM)** solver against **COMSOL Multiphysics 3D FEM** using the **Case B Volumetric Thin-Shell Representation** of reciprocal-lossless bianisotropic metasurfaces.

---

## 1. Overview & Dual-Version Architecture

The metasurface is modeled as a physical shell of finite thickness $\tau \ll \lambda$ located at $r \in [a, a + \tau]$, where the 2D surface sheet impedances and magneto-electric cross-couplings are embedded into effective 3D anisotropic material tensors and weak-form constitutive equations.

The pipeline natively supports both local and cluster environments:
- **COMSOL 5.4 (Desktop)**: Runs with the **Wave Optics Module** (`ewfd`, `ElectromagneticWavesFrequencyDomain`). Automatically configures `DisplacementFieldModel = "RelativePermittivity"`, material `RefractiveIndex` groups, and legacy Java EE runtime patches (`jakarta.activation`, `jakarta.xml.bind`, Java 21 `libjvm.so`).
- **COMSOL 6.1 (Cluster/Workstation)**: Runs with the **RF Module** (`emw`, `ElectromagneticWaves`).

---

## 2. Concentric Geometry & Meshing Architecture

To ensure numerical stability and avoid geometric singularity issues at the coordinate poles, the geometry employs **8-octant shell partitioning** and concentric mesh refinement bands:

```text
       Concentric Radial Cross-Section & Meshing Architecture
       ======================================================
                                                      
                  [ PML Absorbing Layer ]             r = b + t_pml
  --------------------------------------------------- 
                  [ Exterior Air Domain ]             r = b
  ................................................... 
             [ Outer Refinement Band: h_outer ]       r = a + tau + d_ext
  =================================================== 
  //////////////// [ Bianisotropic Shell ] ////////// r = a + tau
  ////////////////   Thickness tau << lam0 ////////// (Swept 2x Prisms
  ===================================================  or Tetrahedra)
             [ Inner Refinement Band: h_inner ]       r = a
  ................................................... r = a - d_int
             [ Interior Dielectric Domain ]           
  ................................................... 
             [ Source Refinement Ball: h_src ]        r = r_src_ball
  ---------------------------------------------------
                        ( x )                         r = 0 (Dipole)
```

### Meshing Strategies
1. **Swept Prism Meshing (`--shell-mesh swept`, Recommended)**:
   - Evaluates a 2D triangular surface mesh on the inner shell boundary ($r = a$) with element size $h_{\text{tangential}} \le \lambda_{\min} / 6$.
   - Sweeps $N_{\text{layers}} = 2$ prism layers radially across thickness $\tau$ through the 8 octant partitions.
   - Volume domains (interior ball, exterior air, PML) are filled with conforming tetrahedra.
   - Result: ~15,000 to 70,000 total elements, solving directly in 15–45 seconds with Pardiso.
2. **Conformal Tetrahedral Meshing (`--shell-mesh tetra`)**:
   - Generates fully tetrahedral elements across both shell and volume domains with bounded aspect ratios.

---

## 3. Physics & Equation View Weak-Form Overrides

On the thin shell domain $a \le r \le a + \tau$, the local spherical coordinate basis is defined as:
$$\mathbf{u} = \hat{\boldsymbol{\theta}}, \quad \mathbf{v} = \hat{\boldsymbol{\phi}}, \quad \mathbf{w} = \hat{\mathbf{r}}$$

The anisotropic relative permittivity $\boldsymbol{\epsilon}_{r,s}$ and permeability $\boldsymbol{\mu}_{r,s}$ tensors are rotated into global Cartesian coordinates:
$$\mathbf{R} = \begin{bmatrix} \hat{\boldsymbol{\theta}}_x & \hat{\boldsymbol{\phi}}_x & \hat{\mathbf{r}}_x \\ \hat{\boldsymbol{\theta}}_y & \hat{\boldsymbol{\phi}}_y & \hat{\mathbf{r}}_y \\ \hat{\boldsymbol{\theta}}_z & \hat{\boldsymbol{\phi}}_z & \hat{\mathbf{r}}_z \end{bmatrix}, \quad \boldsymbol{\epsilon}_{r} = \mathbf{R} \begin{bmatrix} \epsilon_{uu} & \epsilon_{uv} & 0 \\ \epsilon_{uv} & \epsilon_{vv} & 0 \\ 0 & 0 & 1 \end{bmatrix} \mathbf{R}^T$$

The bianisotropic magneto-electric cross-coupling terms are injected directly into COMSOL's weak-form curl-curl wave equation feature (`weq2`) via **Equation View overrides** (`featureInfo("info")`):

1. **Electric Polarization Vector** ($\mathbf{P}$):
   $$P_i = \epsilon_0 \left( \sum_j \epsilon_{r,ij} E_j - E_i \right) + \frac{J_{s,i}^{(m)}}{j\omega\tau}$$
2. **Magnetic Field Vector** ($\mathbf{H}$):
   $$H_i = \frac{1}{\mu_0} \sum_j \mu_{r,\text{inv},ij} B_j - \frac{1}{j\omega\tau\mu_0} \sum_j \mu_{r,\text{inv},ij} M_{s,j}^{(e)}$$
3. **Time Derivative of Magnetic Field** ($\frac{\partial\mathbf{H}}{\partial t}$):
   $$\frac{\partial H_i}{\partial t} = \frac{1}{\mu_0} \sum_j \mu_{r,\text{inv},ij} \frac{\partial B_j}{\partial t} - \frac{1}{\tau\mu_0} \sum_j \mu_{r,\text{inv},ij} M_{s,j}^{(e)}$$

### Physical Dipole Normalization
Analytical dipole fields in `sie_mom.excitation.fields` define an electric dipole with $p_{\text{eff}} = 1$ (omitting $1/(4\pi\epsilon_1)$). COMSOL's `ElectricPointDipole` uses current moment $I\Delta l = 1\text{ A}\cdot\text{m}$. The physical scaling factor is:
$$\text{scale} = \frac{-j \eta_{\text{interior}}}{4\pi k_{\text{interior}}}$$
Dividing COMSOL's extracted fields by this factor matches the SIE-BEM benchmark fields directly.

---

## 4. Usage Instructions

### Running the COMSOL Validation Simulation
```bash
# Using optimized binary parameters and workflow configuration:
python applications/anisotropic_gstc_sphere/comsol/bianisotropic_shell_validation.py \
    --thetas applications/anisotropic_gstc_sphere/results/dipole_rotation/orthogonal_90deg_er7_offset_1m_phase_aligned_smooth002_norm_400iter/optimized_thetas.bin \
    --mesh applications/anisotropic_gstc_sphere/meshes/reflectionless_100mhz_radius2m_ppw10/sphere.obj \
    --config applications/anisotropic_gstc_sphere/configurations/moment_rot/rot90_eps7.json \
    --version 5.4 \
    --shell-mesh swept \
    --output-dir applications/anisotropic_gstc_sphere/results/dipole_rotation/comsol/orthogonal_90deg_er7_offset_1m_swept
```

Example with the full set of meshing and interpolation options:
```bash
python applications/anisotropic_gstc_sphere/comsol/bianisotropic_shell_validation.py \
    --thetas applications/anisotropic_gstc_sphere/results/dipole_rotation/orthogonal_90deg_er7_offset_1m_phase_aligned_smooth002_norm_400iter/optimized_thetas.bin \
    --mesh applications/anisotropic_gstc_sphere/meshes/reflectionless_100mhz_radius2m_ppw10/sphere.obj \
    --config applications/anisotropic_gstc_sphere/configurations/moment_rot/rot90_eps7.json \
    --version 5.4 \
    --shell-mesh swept \
    --shell-layers 5 \
    --grid-oversample 5.0 \
    --grid-method cubic \
    --shell-tangential-divisor 35.0 \
    --interior-hmax-divisor 20.0 \
    --source-ball-radius 0.15 \
    --source-ball-hmax-divisor 40.0 \
    --inner-shell-band-width 0.05 \
    --inner-shell-band-hmax-divisor 30.0 \
    --outer-shell-band-width 0.05 \
    --outer-shell-band-hmax-divisor 30.0 \
    --output-dir applications/anisotropic_gstc_sphere/results/dipole_rotation/comsol/orthogonal_90deg_er7_offset_1m_swept \
    --cores 20
```

### Running via Presets in `runs/`

The `runs/` directory contains ready-to-run automation scripts for both local workstations and the TinkerCliffs cluster:

1. **Linux Workstation (Bash)**:
   ```bash
   # Run baseline or override parameters via environment variables:
   bash applications/anisotropic_gstc_sphere/comsol/runs/orthogonal_90deg_er7_swept.sh

   # With custom tag and layer override:
   TAG=layers7 SHELL_LAYERS=7 bash applications/anisotropic_gstc_sphere/comsol/runs/orthogonal_90deg_er7_swept.sh
   ```

2. **Windows Workstation / Remote Server (PowerShell)**:
   ```powershell
   # Run baseline with COMSOL 6.1 (conda activate comsol):
   .\applications\anisotropic_gstc_sphere\comsol\runs\orthogonal_90deg_er7_swept.ps1

   # With custom parameters (e.g. GLOBAL_DIV=20):
   $env:TAG = "global20"; $env:GLOBAL_DIV = "20.0"; .\applications\anisotropic_gstc_sphere\comsol\runs\orthogonal_90deg_er7_swept.ps1
   ```

3. **TinkerCliffs Cluster (Slurm Batch)**:
   ```bash
   # Submit with default settings (64 CPUs, 800G RAM, COMSOL 6.1):
   sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_orthogonal_90deg_er7_swept.sbatch

   # Or submit with COMSOL 5.4:
   COMSOL_VERSION=5.4 sbatch applications/anisotropic_gstc_sphere/comsol/runs/tinkercliffs_orthogonal_90deg_er7_swept.sbatch
   ```

### Comparing SIE-BEM vs. COMSOL FEM Results
```bash
python applications/anisotropic_gstc_sphere/comsol/compare_sie_comsol.py \
    --sie-dir applications/anisotropic_gstc_sphere/results/dipole_rotation/orthogonal_90deg_er7_offset_1m_phase_aligned_smooth002_norm_400iter/simulation_iter400/cases/electric_incidence \
    --comsol-dir applications/anisotropic_gstc_sphere/results/dipole_rotation/comsol/orthogonal_90deg_er7_offset_1m_swept \
    --output-dir applications/anisotropic_gstc_sphere/results/dipole_rotation/comsol/orthogonal_90deg_er7_offset_1m_swept/comparison
```

---

## 5. Artifacts and Outputs

The validation and comparison scripts generate standard numerical artifacts:
- `comsol_physical_radial.csv`: 1D radial cut ($E_x, E_y, E_z, H_x, H_y, H_z$) across interior, shell, and exterior.
- `comsol_physical_zx.csv`: 2D near-field map in the $zx$-plane ($y=0$).
- `comsol_far_field_cuts.csv`: 360° continuous Stratton-Chu principal cuts (E-plane, H-plane, XY-plane) and directivity in dBi.
- `comsol_metrics.json`: Mesh counts, solver time, radiated power $P_{\text{rad}}$, and directivity maxima.
- `radial_near_field_comparison.png`: 4-panel field and error profiles overlaying SIE vs. COMSOL.
- `zx_near_field_comparison.png`: 3-panel 2D near-field maps (SIE | COMSOL | Relative Difference).
- `far_field_polar_comparison.png`: 3-panel polar radiation patterns (E-plane, H-plane, XY-plane).
- `far_field_rectangular_comparison.png`: Rectangular directivity cuts overlaying SIE vs. COMSOL.
- `comparison_metrics.json`: Quantitative summary of relative errors and directivity MAE.



