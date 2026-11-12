# Initial Linux validation — 2026-11-03

The experiment runs both saved excitation cases and passes its formulation
checks. **The 10 GHz benchmark is not yet demonstrated to be numerically
converged.** Sensitivity to spatial/PML settings still needs assessment before
using it as a final reference solution. The later parallel experiment also
established that the finite-offset GSTC residuals have a substantial analytic
baseline; their approximately 16% values are not 16% FEM boundary errors. See
`PARALLEL_VALIDATION.md`.

Repository base: `c62502793baa782bdda5d864cc50cab1a0537ca3`. The experiment was
added in its own folder; existing SIE/COMSOL source files were unchanged.

Environment: Ubuntu 24.04.5 LTS, Intel Core i9-12900KF (24 logical CPUs),
Python 3.12.3, NGSolve/Netgen 6.2.2608, NumPy 2.5.4, SciPy 1.18.1.
Runs used `--threads 4`, `OPENBLAS_NUM_THREADS=1`, and UMFPACK. These are
measurements from this machine, not predictions for the Windows machine.
Windows execution has not been tested. These initial runs used UMFPACK;
subsequent Linux PARDISO validation is recorded in `PARALLEL_VALIDATION.md`.

## Inputs and reproducibility

Used the saved `dual_pol_eps7/configuration.json` and `face_tensors.bin`, with
the configured `sphere_ppw6.obj` identifying all 3,796 tensor records. Every
run records their SHA-256 hashes. Local normalized tensor coefficients differ
by at most `1.605e-12` across faces; the normalized Z block has condition number
about 7.74. All complex tensor entries were retained.

Raw metrics, logs, field arrays, package versions and plots are retained locally
under the Git-ignored directory:

```text
applications/anisotropic_gstc_sphere/results/dipole_pol_rot/dual_pol_eps7/experiments/ngsolve/validation_20261103/
```

Subdirectories are `coarse`, `medium`, `fine`, `order3_pml025`, `order3`, and
`vtk`. Runs originally wrote to `/tmp/ngsolve-sphere-*` and were copied here;
the recorded original command paths have been preserved. `order3_pml025`
matches the final defaults. `order3` uses the larger PML described below.

Reproduce the default numerical settings from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/run.py \
  --ppw 3 --order 3 --pml-lambda .25 --pml-strength 3
```

All runs below solved **both** electric and magnetic incidence, with one LU
factorization per mesh. New timestamped output directories avoid overwrites.

## Formulation and utility checks

```bash
OPENBLAS_NUM_THREADS=1 applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  -m pytest applications/anisotropic_gstc_sphere/experiments/ngsolve/test_model.py \
  -o addopts='' -q -s
```

**7 tests passed in 1.25 seconds**, with no skipped tests. They cover both
original GSTC equations, singular-Z rejection, full-complex tensor loading and
nonconstant-tensor rejection, electric/magnetic source normalization and Maxwell
curl equations, a manufactured two-sided spherical problem, and both dipole
polarizations of the Huygens far-field integral.

For the manufactured problem, relative E L2 errors changed as follows when
increasing the element order from 1 to 2 on the same mesh:

| Region | Order 1 | Order 2 |
|---|---:|---:|
| Interior | 20.00% | 2.006% |
| Exterior | 11.09% | 2.013% |

This test uses a smaller electrical size and exact outer Dirichlet data. It
checks volume/interface implementation and convergence, not benchmark PML
accuracy. The analytic-source Huygens integrals agree with complex electric
and magnetic dipole far fields to about `3e-15` relative error.

A separate smoke run exercised `--case-id electric_incidence --ppw 1 --order 1
--vtk`. VTK and saved-result plotting completed. Ruff checks passed; Ruff
reports only the repository's existing deprecated configuration-key warning.

## Benchmark measurements

PML thickness/strength were `.25 lambda0 / 3` except the last row, which used
`.5 lambda0 / 1.5`. All used air radius `.95 lambda0` and geometry order 3.

| Run | ppw | Field order | Tetrahedra | DOFs | Assembly (s) | Shared LU (s) |
|---|---:|---:|---:|---:|---:|---:|
| coarse | 2 | 2 | 2,546 | 30,760 | 0.15 | 2.65 |
| medium | 3 | 2 | 5,614 | 66,896 | 0.30 | 12.61 |
| fine | 4 | 2 | 14,051 | 162,131 | 0.59 | 125.12 |
| order3_pml025 (default) | 3 | 3 | 5,614 | 158,816 | 0.52 | 116.44 |
| order3 (larger PML) | 3 | 3 | 5,784 | 163,515 | 0.53 | 122.87 |

Mesh generation took 0.11–0.75 seconds. Linear relative residuals were between
approximately `6e-16` and `1.4e-15`. The fast assembly demonstrates that this
implementation does not encounter the same long initial setup observed in
COMSOL for these meshes. **It is not a speed comparison at matched accuracy
or matched DOFs**: the COMSOL run had over two million volume elements.

For the default third-order run:

| Diagnostic | Electric incidence | Magnetic incidence |
|---|---:|---:|
| Exterior radial E difference from prescribed target | 2.59% | 1.57% |
| Exterior radial H difference from prescribed target | 1.74% | 2.05% |
| Interior radial E difference from prescribed target | 0.153% | 0.304% |
| Interior radial H difference from prescribed target | 0.268% | 0.160% |
| Relative difference of inner/outer power flux | 0.328% | 0.368% |
| Electric-current GSTC residual at offset `0.005a` | 16.87% | 16.13% |
| Magnetic-current GSTC residual at offset `0.005a` | 15.76% | 16.10% |

The radial metrics use complex fields, without amplitude fitting. Small
interior aggregate errors can be dominated by the strong analytic source near
the origin; they do not establish small scattered-field errors. The jump
diagnostics use independently computed volume-curl H at finite offsets, not
exact boundary traces or the reconstructed sheet admittance values.

The second-order mesh sequence was not monotone in all diagnostics. Moving
from order 2 to order 3 on the **same mesh and PML settings** improved the
exterior target comparisons and power balance substantially. Near-sheet H
and both GSTC residuals still require further study.

Changing the PML thickness/strength at order 3 changed exterior radial E by
2.96% / 5.16%, and H by 1.63% / 2.38% (electric / magnetic incidence).
The associated volume mesh also changes, so this is combined mesh/PML
sensitivity, not a pure measurement of PML reflection. That sensitivity alone
prevents a claim of percent-level convergence.

## Next accuracy checks

Increase field order and spatial resolution independently, examine all saved
GSTC offsets, and vary PML thickness/strength and air extent. Compare complex
fields and far-field amplitude/polarization with SIE or completed COMSOL data
on the same observation points. Prescribed design fields and a good linear
residual are not sufficient to establish an independent reference solution.

## 2D field-map addition

The same default benchmark was repeated in `order3_pml025_2d` to save the volume
solutions and 200x200 ZX grids. The initial run had retained only radial/far-field
samples. The repeat used the same 5,614 tetrahedra and 158,816 DOFs; the shared
LU took 117.32 seconds. Complex radial samples agree with the initial run within
`3e-14` relative difference. The initial run's arrays and plots were preserved.

Each case has 31,008 valid grid samples; the PML/outside-air corners, source
singularity and a `0.001a` neighborhood of the sheet are excluded. Sampling
took about 0.29 seconds per case. Complex total E/H retain the original field
normalization. New maps use the existing `plot_fields.py` renderer directly.

Eight computed maps are under `order3_pml025/plots/zx`, and eight comparison
maps under `order3_pml025/plots/zx_vs_prescribed`. These include E/H norms and
the spherical r/theta/phi component pairs. Plot metadata identifies the repeat
run as the source. The masks reflect the spherical physical-air region; fields
in the spherical PML are not plotted as physical near fields.

**9 tests passed** after adding independent ZX source/region/mask checks and a
mesh/solution checkpoint round-trip check. Reloading the full benchmark and
evaluating its 200x200 fields again, without assembly or a solve, reproduced
the saved maps within `5e-10` relative difference. The extra roundoff comes
from serialization of curved geometry. Ruff checks passed.

The new field plots do not change the convergence limitations reported above.
