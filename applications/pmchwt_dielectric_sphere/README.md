# Compare Dielectric Formulations (T-PMCHWT, N-PMCHWT, N-Müller)

*Note: Despite the directory name `pmchwt_dielectric_sphere` (a historical artifact), this application is a fully general, closed-region dielectric solver that supports T-PMCHWT, N-PMCHWT, and N-Müller formulations on any arbitrary closed mesh!*

From the repository root, run:

```bash
python3 applications/pmchwt_dielectric_sphere/compare_formulations.py
```

The script builds the PMCHWT application, runs the formulations with the same
inputs, and compares the complex E and H fields in each region. It prints the
relative L2 differences and writes each run's fields and log under
`build/pmchwt_formulation_comparison/`. If matplotlib is installed, it also
writes `cross_validation_fields.png` there.

The included `meshes/icosahedron_20_faces_r0p03m.obj` is the 20-face sphere of
radius 0.03 m used for the quick smoke run. The default case uses a 1 GHz,
unit-amplitude plane wave traveling in +z with E along x. The source is in the
exterior: **region 1 = interior, region 2 = exterior**.

This mesh checks that the comparison workflow runs; it is too coarse to establish
convergence or solver accuracy. To compare on a finer mesh, pass `--mesh PATH`.
Run `python3 applications/pmchwt_dielectric_sphere/compare_formulations.py --help`
to see the available material and sampling settings.

## Advanced Example

To run a high-resolution, full comparison between all three dielectric formulations (T-PMCHWT, N-PMCHWT, and N-Müller) using a fine mesh (e.g., 5,120 faces) and generating detailed 2D planar and 1D radial cuts, use a command like this:

```bash
python3 applications/pmchwt_dielectric_sphere/compare_formulations.py \
  --mesh applications/pmchwt_dielectric_sphere/meshes/icosahedron_r2lambda_lev4_5120f.obj \
  --output-dir applications/pmchwt_dielectric_sphere/results/l4_5120f_form_a_test \
  --frequency-hz 1e9 \
  --ntl-form line \
  --epsilon-r-interior 2.2 \
  --mu-r-interior 1.0 \
  --epsilon-r-exterior 1.0 \
  --mu-r-exterior 1.0 \
  --quadrature-rule 7 \
  --zx-grid-size 180 \
  --zx-span-radii 2.0 \
  --radial-points 200 \
  --radial-min-radii 0.05 \
  --radial-max-radii 2.0 \
  --formulations t n muller
  --singularity-threshold-lambda0 0.1
```

### Analytical Mie Series Comparison & Fast Plotting

To compare against the analytical Mie series (via `scattnlay`) or to quickly regenerate plots from existing CSV outputs without re-solving:

- **`--include-mie`**: Computes the exact analytical Mie series at the exact same spatial grid points and overlays it on the 1D radial cuts and 2D difference maps.
- **`--plot-only`**: Skips running the C++ solvers entirely; loads existing data from `--output-dir/data/` and updates the plots in seconds.

```bash
python3 applications/pmchwt_dielectric_sphere/compare_formulations.py \
  --mesh applications/pmchwt_dielectric_sphere/meshes/icosahedron_r2lambda_lev4_5120f.obj \
  --output-dir applications/pmchwt_dielectric_sphere/results/l4_5120f_form_a_test \
  --frequency-hz 1e9 \
  --epsilon-r-interior 2.2 \
  --mu-r-interior 1.0 \
  --epsilon-r-exterior 1.0 \
  --mu-r-exterior 1.0 \
  --formulations t n muller \
  --plot-only \
  --include-mie
```
*(Notice that when using `--plot-only`, you can omit all solver-specific parameters like `--ntl-form`, `--quadrature-rule`, and sampling counts, since the spatial grid is loaded directly from the existing data files!)*

