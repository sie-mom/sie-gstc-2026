# Anisotropic GSTC sphere

This application prescribes interior and exterior fields from a named catalog
of plane waves and dipoles. The supplied cloaking configurations use zero
interior fields and unchanged exterior plane waves for two states: x and y polarization
propagating along +z. Python extracts surface parameter tensors; the C++
application calls the shared spherical GSTC SIE solver.

## Shared Extraction Routines

The reusable mathematics lives in
[`src/sie_mom/gstc/surface_parameters.py`](../../src/sie_mom/gstc/surface_parameters.py).
It accepts any number of prescribed boundary states in an oriented
orthonormal tangent basis, independently of geometry or source type.

- `extract_reciprocal_complex_least_squares`: symmetric complex self-blocks
  and `Y_me = -Y_em.T`, with no passivity or losslessness constraint.
- `extract_reciprocal_lossless_least_squares`: the map
  `[[j B, K], [-K.T, j X]]`, with real symmetric `B` and `X` and real `K`.
  Its self-blocks are imaginary and its coupling blocks are real, enforcing
  reciprocal passive-lossless behavior at the specified frequency.

Both routines support general tensors or diagonal self-blocks with zero
coupling. Their inputs are `BoundaryQuantities`, formed from prescribed
tangential interior/exterior fields with `form_boundary_quantities`.
All phasors use `exp(+j omega t)`.

The reference impedance is supplied explicitly. Magnetic-current residuals
are divided by this impedance to give both current residual blocks units of
A/m. SVD determines the rank and fit. Rank-deficient cases retain the
minimum-norm physical coefficients used by the legacy complex extractor;
no column normalization changes that choice.

## Application Organization

```text
anisotropic_gstc_sphere/
    configurations/<case>_<method>.json
    workflow.py                # Configuration, validation, and paths
    prescribed_fields.py       # Prescribed interior/exterior total E/H
    prepare_sie_inputs.py      # Tangential projection, extraction, and export
    sie_gstc_sphere.py         # Mesh/preparation/C++ pipeline coordinator
    anisotropic_sphere_app.cpp # C++ batch solver interface
    plot_fields.py            # Saved physical-field plots per simulation case
    plot_surface_parameters.py # Surface parameter maps and condition numbers
    meshes/<case_id>/sphere.obj
    publication/               # Publication figures and one-click reproduction runners
        README.md
        plot_example_2_part_a.py
        run_example_2_part_a.py
        run_example_2_part_a.sh
        figures/
    results/<case_id>/sie/<run_name>/
        configuration.json
        run.json
        sie_inputs.json
        face_tensors.bin
        batch_metrics.json
        cases/<simulation_case_id>/
            solver_metrics.json
            currents.bin
            fields/
            plots/
```

A `PrescribedFieldState` contains four total-field arrays for one transformation:
interior E/H and exterior E/H, with spherical components `(r, theta, phi)`.
`prescribed_fields.py` evaluates those fields; it does not compute their jumps
or averages, select a tensor model, or read the simulation excitations.
`prepare_sie_inputs.py` selects the tangential components and calls the shared
`form_boundary_quantities` routine to produce a `BoundaryQuantities` object
for that same transformation:

```text
E_avg = (E_exterior + E_interior) / 2
H_avg = (H_exterior + H_interior) / 2
J_s = n x (H_exterior - H_interior)
M_s = -n x (E_exterior - E_interior)

[J_s; M_s] = Lambda [E_avg; H_avg]
```

These fields are tangential, and `n` points from interior to exterior. The
currents are the known left-hand side of this GSTC map; the averages multiply
the unknown surface parameter map `Lambda` on the right. Preparation passes
these quantities to the reusable extraction routines and exports their result.
The two data models represent the same state at different preparation stages.
Adding a transformation means configuring its interior/exterior field references, not
adding a special case inside an extractor. Each region's medium determines the
wavenumber and impedance used to evaluate its prescribed fields.

The spherical convention is unchanged: evaluate at the sphere radius in face
centroid directions, use the `(theta, phi)` tangent basis with outward radial
normal, and preserve OBJ face ordering. The mesh must be a healthy, closed,
outward-oriented sphere centered at the origin, with the configured radius.
Vertex-derived face frames are a later, separate migration.

### Supplying Computed Surface Parameters

The Python functions `run_sie_workflow()` and `export_sie_inputs()` accept an
optional `parameter_evaluator(theta, phi)`. It returns `SurfaceParameters` with
four physical blocks of shape `(2, 2, face_count)`, in `exp(+j omega t)` and
the same outward theta/phi frame. Preparation calls it once, after validating
the mesh, instead of evaluating prescribed fields and running local extraction.
Without it, all existing commands retain their extraction behavior.

Both paths use the same material checks, full-map inversion, binary export,
and C++ batch execution. Checks enforce the configured reciprocity, losslessness,
and tensor structure; singular maps and inaccurate inverses are rejected, never
replaced with pseudoinverses. Map conditioning is distinct from extraction rank.

An optional `parameter_source` dictionary records provenance in `run.json` and
`sie_inputs.json`. Supplied maps have `extraction_method: null` in those manifests,
because no local extraction was performed; the configuration snapshot is unchanged.
No configuration-schema option or C++ change is required. The experimental
[SVWF validation adapter](experiments/svwf_inverse/README.md#validate-with-sie)
uses this interface to evaluate a saved angular surface directly on mesh faces.

## Configurations And Runs

Only one configuration schema is supported: `schema_version: 2`.
The Python API uses `WorkflowConfiguration` and `load_workflow_configuration`,
without version suffixes or a legacy compatibility loader. The `schema_version`
keys in run, input, and C++ metrics artifacts version those separate formats;
they are not configuration schema selectors.

The required sections are `fields`, `transformations`, and `simulation`.
Physical, sampling, and SIE settings retain their documented defaults when
omitted. Unknown keys, missing references, duplicate JSON keys, duplicate IDs,
and invalid numeric values are rejected before running the pipeline.

- `fields` defines region-independent Cartesian parameters. Plane waves use
  `electric_amplitude` (E0) and real `propagation_direction`; dipoles use
  `dipole_type`, `effective_moment`, and real `source_position` in meters.
  Complex amplitudes and moments require `{"real": [x, y, z], "imag": [x, y, z]}`.
  The effective dipole moment follows the shared evaluator's normalization,
  not a physical SI moment. There is no additional scale multiplier.
- Each transformation has an `id` and a `gstc` object. Both `interior` and
  `exterior` must be present: a field ID selects prescribed total E/H, while
  explicit `null` selects zero E/H. All transformations constrain one surface
  parameter map. Neither region is intrinsically zero; nonzero fields can be
  prescribed on both sides, with no automatic power adjustment.
- `simulation.cases` defines independent incident excitations, not selections
  of GSTC transformations. Each case has an `id` and one or two `excitations`,
  each containing `field` and `region`, with at most one entry per region.
  An empty cases array is allowed for tensor preparation only.

For example, the following sections prescribe dipole polarization conversion
and independently select interior incidence (not a validated physical design):

```json
{
  "schema_version": 2,
  "fields": {
    "dipole_z": {
      "type": "dipole", "dipole_type": "electric",
      "effective_moment": {"real": [0, 0, 1], "imag": [0, 0, 0]},
      "source_position": [0, 0, 0]
    },
    "dipole_x": {
      "type": "dipole", "dipole_type": "electric",
      "effective_moment": {"real": [1, 0, 0], "imag": [0.5, 0, 0]},
      "source_position": [0, 0, 0]
    }
  },
  "transformations": [
    {"id": "z_to_x", "gstc": {"interior": "dipole_z", "exterior": "dipole_x"}},
    {"id": "x_to_z", "gstc": {"interior": "dipole_x", "exterior": "dipole_z"}}
  ],
  "simulation": {
    "cases": [
      {"id": "dipole_incidence", "excitations": [{"field": "dipole_z", "region": "interior"}]}
    ]
  }
}
```

Preparation evaluates complex plane waves and electric/magnetic dipoles in
either GSTC region, retaining the spherical basis and `exp(+j omega t)`
convention. Plane-wave spatial phase is `exp(-j k.r)` without conjugating E0;
dipoles retain the shared evaluator's outgoing `exp(-j kR)` phase.

The C++ app solves all `simulation.cases` with one matrix assembly and LU
factorization through `solve_gstc_dense_batch` (using `--basis spherical` by
default, or `--basis face` for the intrinsic vertex-ordered triangle-edge frame). Each case supplies
one or two incident sources, at most one per region. Electric/magnetic dipoles
support arbitrary real effective-moment vectors and source positions; plane
waves support arbitrary real transverse E0 vectors and propagation directions.
Each source uses its assigned region's medium for both RHS assembly and field
reconstruction. Preparation still requires an invertible full surface map.

The shared C++ excitation types currently require real amplitude components
and nonzero incident amplitudes. Complex incident sources and empty simulation
case lists fail before creating run artifacts; `--prepare-only` still permits
their tensor preparation. Complex GSTC target fields remain supported.

Python reads and validates JSON, then passes source parameters to one C++
command: `--case ID`, followed by `--dipole REGION TYPE PX PY PZ X Y Z` or
`--plane-wave REGION EX EY EZ DX DY DZ`. Repeat `--case` for each independent
RHS; two source options with different regions under one case form one combined
RHS. `TYPE` is `electric` or `magnetic`. The app's `--output-dir` is the shared
run root. These options replace the old single-source amplitude CLI; the JSON
configuration schema is unchanged. The omega app and single-case core solver
are unchanged.

Two cloaking JSON configurations describe the same physical case with different
extraction constraints:

- [Reciprocal complex](configurations/cloaking_100mhz_radius1m_ppw8_reciprocal_complex.json)
- [Reciprocal passive-lossless](configurations/cloaking_100mhz_radius1m_ppw8_reciprocal_lossless.json)

Both retain the previous defaults: radius 1 m, frequency 100 MHz, vacuum on
both sides, unit-amplitude x/y states, 8 points per free-space wavelength,
quadrature rule 17, dense LU with eta0 scaling, and 100-point ZX axes and radial
cut. Mesh sizing retains the previous `lambda0 = 3e8 / frequency_hz` convention.

From the repository root, generate the mesh, prepare tensors, and run C++:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/sie_gstc_sphere.py \
    --config applications/anisotropic_gstc_sphere/configurations/cloaking_100mhz_radius1m_ppw8_reciprocal_complex.json
```

The default results directory is:

```text
results/cloaking_100mhz_radius1m_ppw8/sie/reciprocal_complex_general/
```

The lossless configuration instead uses `sie/reciprocal_lossless_general/`.
Default run names combine the extraction method and tensor structure. Use
`--run-name <name>` for another run under the same case, or `--run-dir <path>`
for a custom destination. These options are mutually exclusive. Run directories
must be new or empty; existing results are never overwritten.

Relative `sie.mesh.path` values are resolved from this application directory.
CLI configuration and custom run paths are resolved from the working directory.
Meshes are stored separately from results and can be shared by both methods.
An existing mesh is never regenerated silently: pass `--use-existing-mesh` to
validate and reuse it. A missing solver executable is reported before meshing;
build it with `cmake --build build --target anisotropic_sphere_app -j2`.

For preparation only, without invoking C++, use:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/sie_gstc_sphere.py \
    --config applications/anisotropic_gstc_sphere/configurations/cloaking_100mhz_radius1m_ppw8_reciprocal_lossless.json \
    --use-existing-mesh \
    --prepare-only
```

Alternatively, prepare inputs on an explicitly supplied existing mesh:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/prepare_sie_inputs.py \
    --config applications/anisotropic_gstc_sphere/configurations/cloaking_100mhz_radius1m_ppw8_reciprocal_complex.json \
    --mesh path/to/sphere.obj \
    --run-dir path/to/new_run
```

The overridden mesh path is saved in the configuration snapshot. `workflow.py`
is now a configuration library, not the pipeline command. Numerical and physical
settings normally belong in JSON; the coordinator's CLI selects files and run
behavior. `--singularity-threshold-lambda0 VALUE` optionally overrides the JSON
threshold for one run. It must be positive and finite; the effective value is
saved in the run metadata and configuration snapshot without modifying the
source configuration file.

## Dipole Displacement And Attenuation Test

[Reciprocal complex dipole transformations](configurations/dipole_shift/refl_complex.json)
defines a 2 m sphere at 100 MHz, with vacuum in both regions:

| Transformation | Prescribed Interior Total Field | Prescribed Exterior Total Field |
| --- | --- | --- |
| `electric_displacement` | Electric z dipole at (0, 0, 0.5) m, effective moment = 1 | Electric z dipole at the origin, effective moment = 1 |
| `magnetic_attenuation` | Magnetic z dipole at the origin, effective moment = 1 | Magnetic z dipole at the origin, effective moment = 0.5 |

Each interior total field equals its intended incident field, prescribing zero
interior reflection. The simulation section records two independent interior
excitation cases using those same field definitions. Run both with a shared LU:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/sie_gstc_sphere.py \
    --config applications/anisotropic_gstc_sphere/configurations/dipole_shift/refl_complex.json
```

This generates the configured mesh on the first run. For later runs,
add `--use-existing-mesh` and select a fresh `--run-name`. Add `--prepare-only`
to extract tensors without solving. Spherical evaluation
points remain face-centroid directions projected onto the sphere radius, not
the literal planar triangle centroids.

The JSON uses normalized effective moments, not physical SI moments. Each
transformation is scaled independently, with the same factor applied to its
interior/exterior E and H. This preserves the exact GSTC constraints, the E/H
ratio within each field, and the exterior/interior magnetic amplitude ratio
of 0.5. It does not make electric and magnetic physical SI moments equal or
guarantee well-conditioned extraction. For inconsistent targets, independent
state scaling can change the least-squares compromise.

The regression verifies prescribed nonzero interior fields, electric source
displacement, half-amplitude exterior magnetic fields, both GSTC residuals,
reciprocity, binary export, and matrix inversion. On its eight-face test mesh,
the complex extraction has rank 7 with ten independent complex coefficients,
leaving three complex free coefficients; the exporter uses the extractor's
minimum-norm solution. Sixteen tensor entries with ten independent coefficients
does not mean ten unconstrained degrees of freedom remain after prescribing
the transformations.

An opt-in CUDA smoke test also checks the batch handoff, per-case outputs,
two-region current/field superposition, and single-case batch agreement on an
eight-face mesh. This is not evidence that an SIE solution realizes the intended
reflectionless transformations. The selected
complex method does not enforce passivity or losslessness. No passive-lossless
version of this test is introduced in this checkpoint.

## Saved Artifacts And Diagnostics

Every coordinated run saves the resolved configuration in `configuration.json`
and its status in `run.json`: `preparing`, `prepared`, `running`, `completed`, or
`failed`. A prepared run has not executed the solver. A completed run has
produced the required solver outputs, but is not a claim of physical accuracy.
Run manifest schema 2 records shared input paths at the root and per-case
excitations, status, command, and output paths in `simulation.cases`. Artifact
paths in `files` are relative to the run directory. Case outputs live in
`cases/<id>/`; a preparation-only run leaves each case `pending` with a null command. Executed
cases become `running`, then `completed` or `failed`. The metadata also records
the spherical basis, phasor convention, sampling, and mesh SHA-256 fingerprint.

`batch_metrics.json` records shared device preparation, matrix assembly, LU
factorization, multi-RHS solve, and residual matrix reassembly timings. Its
`solve_seconds` excludes factorization. Each `cases/<id>/solver_metrics.json`
records that case's RHS assembly, residual, and physical current maxima, with
a link to shared metrics relative to its case directory. Shared timings are
not duplicated as per-case solve costs. Residual evaluation reassembles the
original matrix once in the same storage after all solutions are computed.

The app tests can be run after building the executable:

```bash
SIE_MOM_TEST_CUDA_APP=1 .venv/bin/python -m pytest tests/python/test_anisotropic_gstc_app.py
```

Input preparation writes `face_tensors.bin` and `sie_inputs.json`. The binary
contains each face's physical fields-to-currents map followed by its full
inverse, row-major little-endian complex128. The JSON records the configuration,
mesh identity, extraction method, weighting, and diagnostics for every state.
The current C++ interface requires an invertible complete surface map; singular
maps fail preparation rather than being replaced with a different material.

The current prescribed cloaking target is not generally compatible with a
lossless sheet. On the existing 320-face mesh, the constrained relative fitting
residual is approximately 1. The lossless configuration exposes this mismatch;
it is not a validated lossless cloak. Selecting the lossless material model
does not make an incompatible field transformation lossless.

A small fitting residual does not establish uniqueness or SIE field accuracy.
In particular, the two-state general complex extraction cannot identify all
ten complex coefficients uniquely. These diagnostics are distinct from the
C++ linear-system residual and target-versus-computed field errors.

## Plotting Saved Fields

Plot all simulation cases from a saved run, independently of the solver:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir applications/anisotropic_gstc_sphere/results/refl_complex/sie/reciprocal_complex_general
```

Add `--case electric_incidence` to select one simulation case. `--results-dir`
is an alias for `--run-dir`; both now select the shared run root, not a case
directory. The plotter reads `run.json` and its saved `configuration.json`,
without consulting the original configuration or regenerating fields.

`--reference-transformation` is optional. Without it, plotting still works:
radial figures contain only computed curves, and maps contain only computed
panels. With it, radial figures also overlay prescribed curves and maps add
prescribed panels. The default component grouping and ten-figure count are the
same in either mode. Omitting both `--case` and `--reference-transformation`
plots all completed simulation cases without prescribed comparisons.

By default, each selected completed case produces ten figures in its recorded
`plots/` directory:

- `radial_magnitudes.png`: physical E/H vector magnitudes along the saved ray.
- `radial_electric_components.png`: real/imaginary E in `(r, theta, phi)`.
- `radial_magnetic_components.png`: real/imaginary H in `(r, theta, phi)`.
- `radial_spherical_E_r_H_r_magnitude.png`,
  `radial_spherical_E_theta_H_theta_magnitude.png`, and
  `radial_spherical_E_phi_H_phi_magnitude.png`: three separate E/H component-pair figures.
- `zx_magnitudes.png`: physical E/H magnitudes in the ZX plane.
- `zx_spherical_E_r_H_r_magnitude.png`,
  `zx_spherical_E_theta_H_theta_magnitude.png`, and
  `zx_spherical_E_phi_H_phi_magnitude.png`: three separate component-pair maps.

Coordinates are normalized by the saved sphere radius; the boundary is marked
at `r/a = 1`. Radial curves are drawn separately in the two regions, without
connecting across the sheet. Spherical components use the sphere-centered
frame, not a source-centered frame. Dipole incident-source positions lying in
the ZX plane are marked; off-plane sources are not projected into that plane.
Electric and magnetic fields retain V/m and A/m units without amplitude
normalization. Only the piecewise physical fields are used, never the algebraic
sum of interior/exterior representations.

Magnitude axes and color scales default to logarithmic, not dB. Choose
`--line-scale linear|log|db` and `--map-scale linear|log|db` independently.
For example, add `--line-scale linear --map-scale db` for linear radial
magnitudes and dB maps. The existing `--magnitude-scale` sets both unless
overridden by these individual options; component real/imaginary plots remain
linear. `log` retains SI amplitudes on a logarithmic axis; `db` converts the
amplitudes to dB and plots those numbers on a linear axis.

dB displays use `20*log10(|E| / (1 V/m))` and
`20*log10(|H| / (1 A/m))`, with the reference stated in each label. This changes
only the display, not field amplitudes, effective moments, or saved data. No
normalization by a field maximum is applied. Zero values are masked on log and
dB displays; entirely zero fields use a linear fallback for log plots or a
zero-field annotation for dB plots (`-infinity` dB). No artificial dB floor is
applied.

Map color limits default to `--map-color-limits percentile`, matching the omega
plotter's 2nd-98th percentile limits. Computed and prescribed panels share the
same limits for each quantity. Values outside these limits saturate the colors;
colorbar extensions indicate clipping. Field amplitudes and saved data remain
unchanged, and radial curves are never clipped by this option. Use
`--map-color-limits full` to show the full finite range, including source peaks.
Use `--map-percentiles LOW HIGH` to adjust percentile limits; the default is
`2 98`, and valid bounds satisfy `0 <= LOW < HIGH <= 100`. For example,
`--map-color-limits percentile --map-percentiles 2 95` lowers the upper color
limit to the 95th percentile. `2 80` or `2 70` saturates more high-field samples
to reveal weaker-field structure. These are percentiles of the finite displayed
samples, not percentages of the maximum amplitude. Computed and prescribed
samples are pooled for each quantity to keep their color limits identical.
`--map-percentiles` does not affect radial curves, is ignored in `full` mode,
and leaves phase-map limits fixed at -180 to 180 degrees.
Both policies work with `linear`, `log`, or `db` magnitude displays; signed maps
retain symmetric limits and phase maps keep their fixed degree range. Constant
fields receive a nondegenerate range, including plane-wave magnitudes.
Repeating the command replaces these PNGs but leaves solver artifacts intact.
Cases without saved physical fields, malformed CSVs, and inconsistent metadata
fail explicitly rather than silently producing partial selections.

### Components And Prescribed Comparisons

Use `--line-basis spherical|cartesian` and `--map-basis spherical|cartesian`
independently (both default to spherical). These select the field-component
basis, not the sampling plane: ZX coordinates remain `x/a, z/a`.
`--line-components` and `--map-components` select one or more quantities:
`E`, `H` for vector norms; `E_r E_theta E_phi H_r H_theta H_phi` in the spherical
basis; or `E_x E_y E_z H_x H_y H_z` in the Cartesian basis. Norms are basis
independent. With no `--line-components` filter, all three components of each
field are plotted automatically in three E/H component-pair figures, using
`--line-scale`. With no `--map-components` filter, the same grouping is used
for ZX maps with `--map-scale`. Cartesian bases instead produce the pairs
`(E_x, H_x)`, `(E_y, H_y)`, and `(E_z, H_z)`; their component-magnitude filenames
start with `radial_cartesian_` or `zx_cartesian_`. Vector-norm plots are retained,
as are linear real/imaginary diagnostics. An explicit line selection replaces
the six default radial figures with one selected-quantity figure; an explicit
map selection replaces the four default ZX figures with one selected-quantity
figure. Cartesian diagnostic filenames end in `_components_cartesian.png`.
Exactly zero prescribed components are labeled on log/dB magnitude plots,
where a zero-amplitude reference cannot be drawn as a finite curve.

For the PPW10 electric-displacement example, compare all simulated and
prescribed component magnitudes without choosing individual components or
rescaling amplitudes. Both component bases default to spherical:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir applications/anisotropic_gstc_sphere/results/refl_complex/sie/reciprocal_complex_general \
    --case electric_incidence \
    --reference-transformation electric_displacement \
    --line-scale db --map-scale db \
    --map-color-limits percentile
```

`--reference-transformation` requires an explicit `--case`; no association is
inferred from case names or ordering. The reference is evaluated from the
configuration snapshot on the exact saved observation coordinates, using each
point's saved region and its corresponding medium. Only that region's
prescribed total field is evaluated. Radial figures overlay black solid
prescribed curves with colored dotted computed curves and sparse open markers:
circles for interior samples, triangles for exterior samples. The regions stay
separate, and markers remain visible when curves overlap. Maps place computed
and prescribed results side by side with identical color limits for each
quantity. Source markers on prescribed panels identify the regional field
definitions, including equivalent exterior dipole sources.

Selected-component filenames include the view, basis, quantities, and field
part, for example `zx_spherical_E_theta_H_phi_magnitude.png`. Comparison
filenames additionally end in `_vs_<transformation_id>.png`; existing
computed-only figures are not overwritten by comparison figures.
The script prints exactly the files generated by that command. Older figures
with different component selections or reference IDs remain in the directory
and do not inherit new scale settings. An optional filter such as
`--line-components E_theta H_phi --map-components E_theta H_phi` produces only
those quantities instead of the default full set.

Plot styling is centralized in `_radial_curve` and `REGION_STYLES` /
`REGION_MARKERS` in `plot_fields.py`. `_display_magnitude` controls the display
conversion; `_plot_radial` and the map-selection loop in `plot_fields` control
the default grouping. None of these changes the saved fields.

`--line-part` and `--map-part` independently select `magnitude` (default),
`real`, `imag`, or `phase`. Signed real/imaginary values and phase require
explicit vector-component selections and a `linear` display scale. Phase is
shown in degrees, with zero-amplitude phases masked; maps use a cyclic palette.
Signed maps use symmetric color limits and a diverging palette. Singular
prescribed dipole-source samples and undefined spherical components at the
sphere center are masked, not replaced by finite fields.

### Command Variants

Run these examples from the repository root. Set the saved run once; change
`ppw10` to another completed run when needed:

```bash
RUN_DIR=applications/anisotropic_gstc_sphere/results/refl_complex/sie/reciprocal_complex_general
```

Computed-only electric case, with dB magnitudes in both views:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$RUN_DIR" --case electric_incidence \
    --line-scale db --map-scale db --map-color-limits percentile
```

All cases, computed only, with logarithmic magnitudes:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$RUN_DIR" --line-scale log --map-scale log
```

Magnetic case compared with its prescribed attenuation transformation:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$RUN_DIR" --case magnetic_incidence \
    --reference-transformation magnetic_attenuation \
    --line-scale db --map-scale db --map-color-limits percentile
```

All Cartesian components, with linear radial magnitudes and dB maps:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$RUN_DIR" --case electric_incidence \
    --reference-transformation electric_displacement \
    --line-basis cartesian --map-basis cartesian \
    --line-scale linear --map-scale db --map-color-limits percentile
```

The scale and color-limit flags can be changed independently in any magnitude
command above:

| Variant | Flags |
| --- | --- |
| Linear amplitudes in both views | `--line-scale linear --map-scale linear` |
| Logarithmic amplitudes in both views | `--line-scale log --map-scale log` |
| dB magnitudes in both views | `--line-scale db --map-scale db` |
| Logarithmic radial magnitudes, dB maps | `--line-scale log --map-scale db` |
| Source-peak-resistant map limits (default) | `--map-color-limits percentile` |
| Lower the upper color limit to the 95th percentile | `--map-color-limits percentile --map-percentiles 2 95` |
| Stronger saturation of high-field samples | `--map-color-limits percentile --map-percentiles 2 80` |
| Saturate the highest 30% of finite displayed samples | `--map-color-limits percentile --map-percentiles 2 70` |
| Full finite map range, including source peaks | `--map-color-limits full` |
| Spherical radial components, Cartesian maps | `--line-basis spherical --map-basis cartesian` |

For example, keep dB magnitudes and reduce the upper map limit to the 95th
percentile while comparing the electric case:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$RUN_DIR" --case electric_incidence \
    --reference-transformation electric_displacement \
    --line-scale db --map-scale db \
    --map-color-limits percentile --map-percentiles 2 95
```

For a focused comparison, select only `E_theta` and `H_phi`. This produces one
radial figure and one map figure instead of the ten default figures:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$RUN_DIR" --case electric_incidence \
    --reference-transformation electric_displacement \
    --line-components E_theta H_phi --map-components E_theta H_phi \
    --line-scale db --map-scale db
```

For signed diagnostics, explicitly select components and use linear scales.
This example plots real radial components and imaginary map components:

```bash
.venv/bin/python applications/anisotropic_gstc_sphere/plot_fields.py \
    --run-dir "$RUN_DIR" --case electric_incidence \
    --reference-transformation electric_displacement \
    --line-components E_theta H_phi --map-components E_theta H_phi \
    --line-part real --map-part imag --line-scale linear --map-scale linear
```

Replace `--line-part real --map-part imag` with
`--line-part phase --map-part phase` in that command for phase plots in degrees.
Keep both scales `linear`; phase and signed components cannot use `log` or `db`.
All comparison commands also work without `--reference-transformation`, in
which case they produce computed-only figures. Changing scales replaces only
the matching filenames; older figures for other selections remain untouched.

The familiar dipole radiation lobes describe angular variation, whereas these
maps show spatial fields, including radial decay and near-field terms. A
shifted interior source also removes symmetry about the sphere's equatorial
plane; component selection alone does not guarantee two prominent oval lobes.

Complex-field error metrics and comparisons between runs remain later stages.

---

## Optional NGSolve validation

The sphere FEM runner is
[`validate_ngsolve.py`](validate_ngsolve.py), using the reusable backend in
[`src/sie_mom/validation/ngsolve/`](../../src/sie_mom/validation/ngsolve/).
The backend separates the Maxwell/GSTC formulation, volume meshing and tensor
transfer from application configuration and output handling.

NGSolve is optional and can share the root environment. From the repository
root, after preparing the existing dual-polarization sphere inputs:

```bash
python -m pip install -e ".[ngsolve]"
python -m applications.anisotropic_gstc_sphere.validate_ngsolve --help
python -m applications.anisotropic_gstc_sphere.validate_ngsolve \
  --order 3 --ppw 3 --threads 4 \
  --output-dir applications/anisotropic_gstc_sphere/results/dipole_pol_rot/dual_pol_eps7/ngsolve/run01
```

Choose a fresh output directory. The default input configuration and tensors
are the saved `results/dipole_pol_rot/dual_pol_eps7/` files; use `--config` and
`--tensors` for another compatible input. The sphere adapter currently requires
constant tensors in the spherical basis and centered dipoles. It shares one
factorization across the configured source cases.

The [original experiment documentation](experiments/ngsolve/README.md) explains
the formulation, numerical controls, checkpoint sampling and plots. Its old
`run.py`, `model.py` and `sample_zx.py` interfaces remain compatible; the new
runner does not depend on experiment code. Environments are reused without
being moved, and no additional environment is required.

For varying face tensors on a non-spherical object, use the
[general-surface runner](../gstc_surface/README.md#ngsolve-with-the-same-sie-interface).
That path preserves the SIE interface mesh and face frames. Its order-1–3
ogive comparisons are recorded there, including the unresolved near-sheet
differences; a completed run alone does not establish converged field accuracy.

## Publication Figures & End-to-End Reproduction

Automated end-to-end reproduction scripts and IEEE publication figure generators are located in [`publication/`](publication/):

* **Example 2 Part A (Electric to Magnetic Polarization Rotator)**:
  - Bash runner: `./applications/anisotropic_gstc_sphere/publication/run_example_2_part_a.sh`
  - Python runner: `.venv/bin/python applications/anisotropic_gstc_sphere/publication/run_example_2_part_a.py`
  - Plotting only: `.venv/bin/python applications/anisotropic_gstc_sphere/publication/plot_example_2_part_a.py`
  - Detailed documentation, options, and figure descriptions are in [`publication/README.md`](publication/README.md).

