# Interface-based PEC sphere EFIE

This application uses **region 2 as the exterior**. The mesh normal points from
region 1 to region 2. Its only unknown is the exterior electric current `J2`,
one complex coefficient per RWG edge; there is no magnetic-current unknown.

- T-EFIE tests `i omega mu2 TL2(J2) = E_inc,2` with RWG functions.
- N-EFIE tests `i omega mu2 nTL2(J2) = E_inc,2` with `f x n`.
- N-EFIE `--ntl-form surface` uses Form B; `--ntl-form line` uses Form A.
- The RHS uses the positive exterior incident electric field in both cases.
  The PMCHWT region-2 minus sign does not apply to this one-region equation.
- The existing combined quadrature routines also calculate TK/nTK internally.
  The PEC matrix assembles only TL or nTL; TK/nTK is not part of either equation.

The source is currently a unit x-polarized plane wave traveling in +z in the
exterior medium. The comparison runner writes data/t and either data/n_surface
or data/n_line. Each directory contains current and field CSVs, plus a run log.
The plots directory contains a radial cut and three ZX field comparisons.

For field evaluation, the app passes physical J2 and an all-zero M vector to
the shared surface-current evaluator. Its output is the positive integral term
Q, so the field plotted at every sample, inside and outside, is
F_incident - Q. Interior samples are evaluated, not assigned zero; their
remaining field shows how well the solved current cancels the incident field.
The region column uses the mean mesh-vertex radius only as a label. The
evaluator uses ordinary triangle quadrature without a near-singular correction,
so values close to the surface need caution. These plots compare two numerical
solutions; they are not a Mie-series validation.

For example, from the repository root:

```bash
python3 applications/pec_sphere/compare_efie_formulations.py \
  --mesh applications/pmchwt_dielectric_sphere/meshes/icosahedron_20_faces_r0p03m.obj \
  --output-dir applications/pec_sphere/results/icosahedron_t_vs_n_surface \
  --frequency-hz 1e9 --quadrature-rule 7 --ntl-form surface
```

To compare T-EFIE with N-MFIE:

```bash
python3 applications/pec_sphere/compare_efie_formulations.py \
  --mesh applications/pec_sphere/meshes/gmsh_sphere_approx5120f/sphere.obj \
  --output-dir applications/pec_sphere/results/gmsh_sphere_5120f_t_vs_n_mfie_200mhz \
  --frequency-hz 2e8 \
  --quadrature-rule 7 \
  --compare-with n-mfie \
  --radial-points 200 \
  --radial-min-radii 0.05 \
  --radial-max-radii 2.0 \
  --zx-grid-size 180 \
  --singularity-threshold-lambda0 0.05
```

To generate fields and plots from an existing comparison without solving
again, repeat its mesh, material, frequency, quadrature rule, and output
directory with --reuse-currents:

~~~bash
python3 applications/pec_sphere/compare_efie_formulations.py \
  --mesh applications/pmchwt_dielectric_sphere/meshes/sphere_r2lambda_lev3_1280f.obj \
  --output-dir applications/pec_sphere/results/sphere_lev3_1280f_t_vs_n_surface \
  --frequency-hz 1e9 --quadrature-rule 7 --ntl-form surface --reuse-currents
~~~

This preserves currents.csv and run.log; field output and plots are refreshed.

**Current diagnostic caveat:** the shared nTL routines currently omit the
scalar-gradient contribution for exact self-face pairs. That experiment is
recorded in commit `9e79ec6`; N-EFIE results therefore do not yet validate
an unmodified nTL operator. The T-EFIE path does not use that nTL shortcut.
