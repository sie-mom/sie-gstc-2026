# Lossless Interior-Field Feedback

This independent experiment keeps both exterior transformation targets fixed,
extracts a shared reciprocal-lossless surface map, solves the original incident
excitations, and feeds the computed interior total fields back into extraction.
It does not change the application workflow, JSON schema, or core solver.

## Run

```bash
cmake -S . -B build
cmake --build build --target anisotropic_sphere_app anisotropic_surface_sampler -j2
.venv/bin/python applications/anisotropic_gstc_sphere/experiments/iterate_lossless.py \
  --config applications/anisotropic_gstc_sphere/configurations/moment_rot/rot90_eps7.json
```

Results go into a new timestamped directory beneath
`results/<case_id>/experiments/lossless_feedback/`. Optional `--output-dir` selects
a new or empty destination. Existing runs and meshes are not modified.

## Edit The Experiment

The settings near the top of `iterate_lossless.py` are deliberately ordinary
Python constants:

- `ITERATIONS = 5`: number of extraction/solve steps, including the initial one.
- `OFFSET_M = 0.01`: inward/outward sampling distance in metres.
- `DAMPING = 1.0`: direct interior replacement; try `0.5` for smaller updates.
- `STATE_CASES`: explicit transformation-to-simulation-case pairs. Adjust these
  IDs when using another JSON; the script does not infer matching from names.

Physical settings, field definitions, excitations, mesh, quadrature, and matrix
assembly threshold come from the JSON. Both simulation excitations remain fixed.
The existing C++ app currently requires real incident amplitudes; feedback fields
remain complex, and the script never converts them to magnitudes or rescales them.

## Numerical Steps

`initialize_states()` evaluates the interior and exterior prescribed fields.
`extract_and_export()` forms jumps/averages and fits one shared lossless map,
then writes that map and its full inverse. `solve_currents()` runs the unchanged
batch app with grid evaluation disabled, sharing one LU across the incident cases.
`sample_total_fields()` calls the separate C++ sampler and constructs total fields
as **incident minus integral contribution**. `update_interior_states()` updates
interior E/H together, leaving all exterior target arrays unchanged.

Each iteration needs a new matrix assembly/LU because its surface map changes.

## Sampling Convention

The production preparation code uses the ideal sphere radius in each centroid
direction, with no deliberate two-sided offset. This experiment instead uses the
actual flat-face centroid `c` as the nominal boundary point. Initial fields and
fixed exterior extraction targets are evaluated there. The sampler evaluates
interior integrals at `c - d*c/|c|` and exterior integrals at `c + d*c/|c|`.
Both remain in the same spherical `(r, theta, phi)` basis and OBJ face order;
only theta/phi components enter extraction. The offset follows the radial normal,
not the flat triangle normal.

Interior feedback is an **approximate boundary trace**: the first iteration uses
centroid fields and later ones use inward-offset samples. Check sensitivity to
`d` and mesh refinement before interpreting improvement as convergence. Exterior
error compares computed and prescribed complex fields at the same outward-offset
points. The existing field kernel already uses singularity extraction for nearby
faces, with a fixed threshold of `0.1` times the regional wavelength; this is
independent of the JSON's matrix assembly threshold.

## Saved Data

- `experiment.json`: configuration snapshot, offset, damping, and state/case order.
- `geometry_and_targets.npz`: centroids, angles, areas, and unchanged initial targets.
- `history.csv`: per-case exterior complex-field error, interior update size,
  and worst-face extraction residual for every step.
- `iteration_*/face_tensors.bin`: shared map and inverse in the normal solver layout.
- `iteration_*/extraction.npz`: rank, conditioning, fitting and inversion residuals.
- `iteration_*/fields.npz`: interior fields used for extraction, computed regional
  total fields, and exterior targets at the comparison points, in `(state, 3, face)`
  order. Components are radial, theta, phi.
- `iteration_*/cases/<case>/`: physical currents, solver metrics, sampled integral
  CSVs, and sampling log. `solver.log` records the batch command and its output.

There is no automatic convergence decision or optimizer. Small interior updates
alone do not imply a successful exterior match.

## Plot A Saved Iteration

Prepare the radial and ZX grids from saved currents, then use the unchanged
application plotter. This performs field evaluation only: no parameter extraction,
matrix assembly, LU factorization, or repeat iteration. Both cases are prepared
using the configuration snapshot in the parent `experiment.json`, not the original
configuration file. Keep the original mesh unchanged.

```bash
cmake --build build --target anisotropic_surface_sampler -j2
ITERATION_DIR=applications/anisotropic_gstc_sphere/results/dipole_rotation/legacy_er1/analytical_baseline_lossless/experiments/lossless_feedback/20261013T200018228997Z/iteration_004
.venv/bin/python applications/anisotropic_gstc_sphere/experiments/prepare_plot_fields.py \
  --iteration-dir "$ITERATION_DIR"
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
  --run-dir "$ITERATION_DIR" \
  --case electric_incidence \
  --reference-transformation electric_z_to_x \
  --line-scale db --map-scale db \
  --map-color-limits percentile --map-percentiles 2 95
```

Select another `iteration_*` directory to inspect a later step. For the magnetic
case, use `--case magnetic_incidence --reference-transformation magnetic_z_to_x`.
Omit both options to plot all cases without prescribed comparisons. The existing
plotter's basis, component, and scale options work as in the normal workflow.

Post-processing adds `configuration.json`, `run.json`, and `plot_points.bin` to
the iteration. Each case gets `plot_samples/*_integrals.csv`, `plot_sampling.log`,
and `fields/sie_physical_{radial,zx}.csv`; the plotter writes its figures to
`cases/<case>/plots/`. Total fields are **incident minus Q**, with the interior
representation selected for `r<a` and the exterior one for `r>=a`. No amplitudes
are normalized. Repeating preparation replaces only these post-processing files;
currents, tensors, feedback samples, `fields.npz`, and experiment history are kept.

`--reference-transformation` compares against the **original prescribed fields**.
Its interior curve is the initial ideal target, not the updated interior feedback
state. The exterior reference remains the experiment's fixed target field.
