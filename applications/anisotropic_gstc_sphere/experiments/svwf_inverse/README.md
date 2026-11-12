# SVWF Inverse Synthesis

This independent prototype fits a shared reciprocal-passive-lossless surface
and regular interior fields to prescribed exterior total fields. Physical
settings, field definitions, transformation targets, and regional incident
excitations come from the existing application JSON. Synthesis does not require
the SIE workflow. Optional SIE validation supplies the saved material through
the shared workflow's Python evaluator interface; the configuration schema,
C++ solver, and `spherical_expansion` implementation are unchanged.

## Run

From the repository root:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/experiments/svwf_inverse/inverse_synthesis.py \
  --config applications/anisotropic_gstc_sphere/configurations/crosspolarization_100mhz_radius2m_ppw10_reciprocal_lossless.json
```

The script uses only CPU/SciPy and does not require a mesh or CUDA executable.
Results go to a fresh timestamped directory under
`results/<case_id>/experiments/svwf_inverse/`. Optional `--output-dir` selects
a new or empty directory; existing results are not overwritten. To solve a
previously synthesized surface without optimizing again, use the forward-only
mode described below.

## Edit Settings

The constants near the top of `inverse_synthesis.py` are the experimental inputs:

- `STATE_CASES`: explicit transformation-to-simulation-case pairs. Every
  transformation and case must appear once; matching is not inferred from names.
- `N_MAX = 6`: SVWF truncation; exterior coefficients are `a,b`, interior `c,d`.
- `N_THETA`, `N_PHI`: inverse collocation and forward-solver grid sizes.
- `ANGULAR_DEGREE`: real spherical-harmonic degree of the surface profiles.
- `MAX_NFEV`: nonlinear residual-evaluation budget, not a convergence claim.

`ANGULAR_DEGREE = 0` gives ten constant real coefficients in the spherical
theta/phi frame. Higher degrees allow azimuthal variation without independent
unknowns at every point. Degree L uses `(L+1)^2` real profiles for each of the
ten material parameters. Phi-dependent profiles remove the constant-model
symmetry restriction, but do not guarantee a realizable or accurate target.

The routine has no special branch for cloaking or crosspolarization. To use
another configuration, change `STATE_CASES` to its explicit state/case mapping.
The prescribed interior fields are used only for initialization; incident
fields stay fixed, while the optimized interior total fields may differ.

## Numerical Steps

1. `prepare_problem()` evaluates regional incident fields and exterior targets
   on the ideal sphere, with no offset. Exterior scattered targets are total
   targets minus exterior incidence; with only interior incidence they equal
   the exterior total targets. `fit_trace()` fits both tangential E and H using
   outgoing SVWFs. Its `a,b` coefficients are subsequently fixed.
2. `surface_parameters()` parameterizes symmetric imaginary `Y,Z` and real
   `chi,gamma`, with `gamma = -chi.T`. One surface is shared by all states.
   The initial guess projects the existing local lossless extraction onto
   the chosen angular basis and fits the initial regular interior response.
3. `InverseProblem` constructs interior total fields as unchanged interior
   incidence plus regular `c,d`. It minimizes `J - Y E_avg - chi H_avg` and
   `M - gamma E_avg - Z H_avg`. Unknown material coefficients multiply unknown
   interior fields, so this is a bilinear, nonlinear least-squares problem.
4. `run_experiment()` uses SciPy trust-region least squares with an analytic
   Jacobian. Complex unknowns/residuals are packed as separate real/imaginary
   parts; fields are never converted to magnitudes for fitting. Quadrature
   and impedance weighting balance residual units. Column/state normalization
   is numerical only; physical source amplitudes and targets are not rescaled.
5. `forward_validation()` uses the unchanged `solve_collocation_general()`
   to solve for all `a,b,c,d` with the fitted surface. Simultaneous interior
   and exterior incidence is handled by superposition of two regional solves.
   Field errors and local/global power differences are evaluated on a distinct,
   finer spherical sampling grid.

Inverse samples use Gauss-Legendre quadrature in cos(theta) and uniform phi.
The forward solver retains its own existing uniform-theta grid. Quadrature
weights saved by this experiment are solid-angle weights; multiply by radius
squared to integrate power in watts. All ten material rows are dimensionless
internally: electric admittance is divided by the reference impedance, magnetic
impedance multiplied by it, and coupling is dimensionless.

SVWF computations and coefficient archives use `exp(-j omega t)`. Configured
fields are conjugated into this convention. Saved field arrays and physical
surface blocks use the application's `exp(+j omega t)` convention. Conjugate
saved tensor blocks before using them directly with the SVWF forward solver.

## Saved Data

- `configuration.json`: unchanged physical configuration snapshot.
- `experiment.json`: settings, state/case order, component labels, and conventions.
- `history.csv`: objective values at residual evaluations, including rejected
  optimizer trials; rows are not necessarily accepted iterations.
- `surface_parameters.npz`: `Y,Chi,Gamma,Z` with shape `(2,2,point)`, spherical
  angles, solid-angle weights, and real `angular_coefficients` of shape `(10,profile)`.
- `synthesis.npz`: initial/final optimization vectors and candidate interior,
  fitted exterior, original targets, incidents, and weighted GSTC residuals.
  Field arrays have shape `(state,4,point)`, ordered `E_theta,E_phi,H_theta,H_phi`.
- `cases/<case>/inverse_coefficients.npz`: fixed target `a,b` and optimized `c,d`.
- `cases/<case>/forward_coefficients.npz`: independently solved `a,b,c,d`.
  Both coefficient archives use the existing save/load API in
  `spherical_expansion.coefficients`.
- `cases/<case>/validation_fields.npz`: forward interior/exterior total fields
  and original exterior targets on the validation grid, shape `(4,point)`.
- `metrics.json`: optimizer termination, expansion error, GSTC residuals,
  actual forward exterior-field error, and local/global power differences.

`status: completed` means the script finished, not that the target was achieved.
Inspect `optimizer_success`, GSTC residuals, and especially the independent
forward error. There is no automatic acceptance threshold, global-optimum
guarantee or regularization penalty. Synthesis itself does not export SIE inputs;
use the separate validation step below. Increase
SVWF/angular orders and sampling to test stability before trusting a design.

## Plot Saved Results

`prepare_plot_fields.py` reconstructs total E/H on the radial and ZX grids from
the saved configuration. It uses the existing SVWF field and coefficient APIs,
not a new optimization or forward solve. Default `--solution forward` selects
the independently solved physical response; `--solution inverse` selects the
fitted candidate with prescribed exterior `a,b`. Only the forward response tests
whether the synthesized surface produces the requested transformation.

```bash
EXPERIMENT_DIR=applications/anisotropic_gstc_sphere/results/crosspolarization_100mhz_radius2m_ppw10_lossless/experiments/svwf_inverse/20261013T221521559929Z
.venv/bin/python applications/anisotropic_gstc_sphere/experiments/svwf_inverse/prepare_plot_fields.py \
  --experiment-dir "$EXPERIMENT_DIR"
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
  --run-dir "$EXPERIMENT_DIR/plot_data/forward" \
  --case electric_incidence --reference-transformation electric_z_to_x \
  --line-scale db --map-scale db --map-color-limits percentile --map-percentiles 2 95
```

Use `--case magnetic_incidence --reference-transformation magnetic_z_to_x` for
the second transformation. Existing spherical/Cartesian, component, scale,
real/imaginary, phase, and percentile options are supported. The standard output
includes radial component magnitudes and real/imaginary curves, plus ZX maps.
For explicitly selected real parts:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
  --run-dir "$EXPERIMENT_DIR/plot_data/forward" \
  --case electric_incidence --reference-transformation electric_z_to_x \
  --line-components E_theta H_phi --line-part real --line-scale linear \
  --map-components E_theta H_phi --map-part real --map-scale linear
```

To inspect the inverse candidate separately:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/experiments/svwf_inverse/prepare_plot_fields.py \
  --experiment-dir "$EXPERIMENT_DIR" --solution inverse
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
  --run-dir "$EXPERIMENT_DIR/plot_data/inverse" \
  --case electric_incidence --reference-transformation electric_z_to_x \
  --line-scale db --map-scale db --map-percentiles 2 95
```

Each dataset has its own configuration snapshot, `run.json`, and
`cases/<case>/fields/svwf_{radial,zx}.csv`. Figures go under
`plot_data/<solution>/cases/<case>/plots/`. Repeating preparation replaces only
post-processing data; original coefficients, metrics, and synthesis arrays are
unchanged. There is no amplitude normalization. The interior reference curve
is the original prescription, not an enforced optimization target; differences
there may represent allowed reflections. The exterior reference remains the
desired total field. An inverse-candidate exterior match is imposed, not an
independent validation. Compare complex components as well as magnitudes.

## Inspect Surface Parameters

Plot and check a saved surface without running synthesis or any solver:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/experiments/svwf_inverse/plot_surface_parameters.py \
  --experiment-dir "$EXPERIMENT_DIR"
```

Set `EXPERIMENT_DIR` to a saved experiment directory as in the plotting example
above. Only `experiment.json` and `surface_parameters.npz` are read. Outputs go
to `parameter_checks/`; repeating the command replaces these inspection outputs
but does not modify synthesis inputs or results.

- `Y_imaginary.png`: the three independent imaginary electric-admittance entries, in S.
- `Z_imaginary.png`: the three independent imaginary magnetic-impedance entries, in ohms.
- `chi_real.png`: the four real coupling entries; the reciprocal partner is `gamma = -chi.T`.
- `map_condition_number.png`: local impedance-balanced map conditioning on a log color scale.
- `checks.json`: finite-data, reciprocity, forbidden real/imaginary parts,
  skew-Hermitian losslessness, conditioning percentiles, and worst-location diagnostics.

Parameter maps show signed linear values versus theta and phi, with symmetric
color limits about zero; no magnitudes or dB conversion are used. The periodic
phi seam is closed for display only. All checks use the saved samples, not a
denser angular reconstruction, and finite-data/grid errors are rejected before
output is written. Material-constraint violations are reported, not projected away.

The dimensionless map used for conditioning is `[[eta*Y, chi], [gamma, Z/eta]]`,
with `eta` taken from the saved reference impedance. Reciprocity and losslessness
flags use an absolute tolerance of `1e-10` in these balanced units. Numerically
singular samples are counted separately and colored pink; infinite condition
percentiles are stored as JSON `null`. A singular map can still satisfy the
material constraints. These checks do not establish an exact field match,
spatial convergence, physical realizability, or the condition number of the
SVWF solver matrix.

## Forward-Only SVWF Check

Use the existing `inverse_synthesis.py` entry point to solve a saved material
at another forward resolution, without extraction, target fitting, or nonlinear
optimization. The saved angular coefficients, angular degree, reference
impedance, media, and incident excitations stay unchanged. The linear solver
computes all exterior `a,b` and interior `c,d`; prescribed exterior fields enter
error reporting only and are not imposed on the forward solution.

With `EXPERIMENT_DIR` pointing to the original saved inverse run:

```bash
FORWARD_DIR="$EXPERIMENT_DIR/forward_validation/nmax10_theta36_phi72"
OPENBLAS_NUM_THREADS=1 .venv/bin/python applications/anisotropic_gstc_sphere/experiments/svwf_inverse/inverse_synthesis.py \
  --forward-only --experiment-dir "$EXPERIMENT_DIR" \
  --n-max 10 --n-theta 36 --n-phi 72 --output-dir "$FORWARD_DIR"
```

These are example refinement settings, not a convergence guarantee. Omitted
resolution options use the original run's saved settings, not the current
constants near the top of the script. Grids must satisfy the same basic
minimum sampling checks as synthesis. Increasing field truncation does not
change the material's angular degree. Physical amplitudes are not normalized.

Outputs go into a new directory. Without `--output-dir`, the default is
`$EXPERIMENT_DIR/forward_validation/<timestamp>/`. Existing nonempty output
directories are rejected. Each run contains `configuration.json`,
`experiment.json`, `metrics.json`, and per-case `forward_coefficients.npz`
and `validation_fields.npz`. Its manifest identifies it as a forward-only
calculation, records the chosen resolution and source hashes, and has no
optimizer results or inverse-candidate archives. Original runs are unchanged.

Prepare and plot those new coefficients using the existing commands:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/experiments/svwf_inverse/prepare_plot_fields.py \
  --experiment-dir "$FORWARD_DIR"
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
  --run-dir "$FORWARD_DIR/plot_data/forward" \
  --case electric_incidence --reference-transformation electric_z_to_x \
  --line-scale db --map-scale db --map-color-limits percentile --map-percentiles 2 95
```

Use the magnetic case/reference for the other state. `--solution inverse` is
not available for a forward-only result. The same `solve_forward_response()`
also performs the original post-optimization validation; `forward_validation()`
is its adapter for an in-memory inverse problem. Both SIE validation and this
mode share `load_parameter_evaluator()` in `inverse_synthesis.py`.

## Validate With SIE

`run_sie_validation.py` loads a selected run's `configuration.json`,
`experiment.json`, and `surface_parameters.npz`. It reuses `angular_basis()`
and `surface_parameters()` to reconstruct the saved physical material at the
mesh-face centroid directions, not the current experiment settings. It does
not interpolate sampled tensors, fit fields again, optimize, or rerun SVWF.

First check preparation without executing C++:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/experiments/svwf_inverse/run_sie_validation.py \
  --experiment-dir "$EXPERIMENT_DIR" --prepare-only \
  --run-dir "$EXPERIMENT_DIR/sie_preparation_check"
```

Then run the existing SIE batch app, with unchanged saved incident excitations:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/experiments/svwf_inverse/run_sie_validation.py \
  --experiment-dir "$EXPERIMENT_DIR"
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
  --run-dir "$EXPERIMENT_DIR/sie_validation" \
  --case electric_incidence --reference-transformation electric_z_to_x \
  --line-scale db --map-scale db --map-color-limits percentile --map-percentiles 2 95
```

The adapter only loads the material and calls `run_sie_workflow()` with a
`parameter_evaluator(theta, phi)`. The shared workflow loads the mesh once,
validates it, exports maps/inverses, creates manifests, runs C++, and checks
outputs. It reuses the configured mesh if present, otherwise generates it.
No new JSON option is needed. Saved tensors already use the SIE `+j` convention;
they are not conjugated or amplitude-normalized.

Default output is `sie_validation/` inside the selected experiment, with the
usual SIE layout. `parameter_source` records the source directory, degree,
reference impedance, and input SHA-256 hashes. Numerical map conditioning and
inversion residuals are checked at the actual mesh samples, independently of
the saved rectangular-grid inspection. Singular maps are rejected without a
pseudoinverse. Material checks use a per-face relative tolerance of `1e-10` in
impedance-balanced units; both inverse products must have Frobenius residual
at most `1e-8`. Neither check establishes discretization convergence.

Existing output directories are protected; use `--run-dir` for another run.
A `--prepare-only` directory cannot subsequently be reused for a full solve;
the example deliberately uses a separate preparation directory. Original
synthesis artifacts are not modified.

Compare these SIE plots with `plot_data/forward`, prepared above, as well as
the prescribed exterior targets. The forward SVWF response tests the same
material; the inverse candidate has imposed exterior coefficients. Original
interior prescriptions are reference curves, not enforced reflectionless
targets. Choose the matching saved run for the intended media and incident
amplitudes, rather than editing its snapshot for validation. Mesh sizing still
uses the existing free-space convention; assess interior-wavelength resolution
separately for higher permittivity.

References: [SciPy nonlinear least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
and the repository's `docs/source/sphere_svwf.rst`.
