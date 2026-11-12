# GSTC surface examples

Application workspace for non-spherical GSTC examples and the paper's Example 3.
The original case is the EMCC-style tangent cone-sphere, with a lossless dielectric
interior and GSTCs covering the whole surface. The prescribed transformation
is dual-polarization impedance matching: electric `z -> z` and `y -> y`, with
both moments perpendicular to the body's axis of revolution, the `x` axis.
The [mild circular ogive](#mild-circular-ogive) provides a gentler non-spherical
case with the same matching transformations and centered dipoles.

The organizational reference is the complete
[`anisotropic_gstc_sphere`](../anisotropic_gstc_sphere/) application: configuration,
mesh preparation, tensor synthesis, SIE solution, independent validation and
publication figures. Phase 1 provides CPU-only mesh generation, prescribed-field
evaluation and joint tensor preparation. Phase 2 provides a solve-only batch
coordinator, independent observation preparation, saved-current field evaluation
and diagnostic plotting. FEM validation and large-mesh convergence remain later
checkpoints.

## Original cone-sphere case

[`configs/cone_sphere/ppw6.json`](configs/cone_sphere/ppw6.json) records the original physical
and mesh parameters, source fields, prescribed transformations and tensor
constraints. It follows the spherical application's version-2 structure, adding
`geometry`, moving shape dimensions out of `physical`, and using sampling
distances in metres. [`workflow.py`](workflow.py) loads and validates this
configuration for preparation, solving and field sampling. The existing
spherical application's loader does not accept this configuration.

Set the single `sie.mesh.points_per_wavelength` value manually when changing
resolution; the starting value is `6.0`, and `8.0` is the planned finer check.
This specifies the target surface-edge length relative to the shortest material
wavelength. `sie.mesh.path` names the canonical generated
mesh. The full surface is covered by GSTCs,
and `sie.basis=face` selects its intrinsic tangent frame.

| Parameter | Value |
| --- | --- |
| Frequency | 2 GHz |
| Spherical-end radius | 0.0748538 m (2.947 inches) |
| Cone half-angle | 7 degrees |
| Overall length | Approximately 0.689 m, derived from radius and half-angle |
| Maximum diameter | 0.1497076 m |
| Interior relative permittivity / permeability | 5 / 1, lossless |
| Exterior relative permittivity / permeability | 1 / 1 |
| Axis of revolution | Cartesian `x`, with apex toward `+x` |
| Spherical-end center | `(0, 0, 0)` m |
| Dipole positions | Both at `(0.10, 0, 0)` m, excited separately |
| Prescribed transformations | Electric `z -> z` and electric `y -> y` |
| Extraction | General reciprocal-lossless, all four tangential tensor blocks |
| GSTC coverage | Entire closed surface, including spherical end and cone |
| Tensor frame | Intrinsic oriented face tangents |
| Surface mesh setting | 6 elements per interior dielectric wavelength; manually adjustable |

The physical dimensions follow the cone-sphere in Gibson's Figure 7.16(c).
The EMCC reference target is PEC; this dielectric GSTC case is an adaptation.
The benchmark family is documented on
[Gibson's benchmark-data page](https://lucernhammer.tripointindustries.com/benchmark_data/EMCC/3d_shapes/3d_shapes.html).
At 2 GHz the chosen shape is approximately 4.60 free-space wavelengths long
and 1.00 wavelength in maximum diameter.

Radius and cone half-angle define the tangent geometry. With the spherical
center at zero on an axial coordinate, the junction is at `R*sin(alpha)`, its
radius is `R*cos(alpha)`, and the sharp apex is at `R/sin(alpha)`. The resulting
overall length is about 0.689067 m. The book's rounded axial dimensions sum to
0.689026 m; use the tangent construction consistently rather than enforcing
both rounded lengths as additional independent constraints. This shape differs
from the existing generator's equatorial hemisphere-to-cone join.

### Prescribed fields

The two electric dipoles are co-located on the axis, inside the conical section.
At `x=0.10` m the cone's cross-section radius is approximately `0.06314` m.
The dipoles define two independent excitation cases sharing one synthesized
sheet; they are not two simultaneously driven sources in each case. An exterior
field definition is the desired continuation of a dipole at the same position,
not an additional physical source in the exterior region.

Interior effective moments have unit amplitude along `z` or `y`. The exterior
moments have the same orientation and phase, with amplitude
`(k_in/k_out)^2 * sqrt(eta_out/eta_in) = 5^(5/4) = 7.476743906106103`.
This uses the existing effective-moment convention, not SI electric dipole
moments, and matches the individual prescribed normal real-power fluxes.
Fit both transformations together using `reciprocal_lossless` extraction with
`tensor_structure=general`; inspect aggregate/per-state residuals, rank and
conditioning before treating the prescribed pair as exactly realizable.

The `zx` observation plane (`y=0`) contains the body's longitudinal profile.
The `z` dipole lies in that plane, while the `y` dipole is perpendicular to it,
giving two transverse polarizations in the same geometric view. The draft
`sampling` section retains the sphere configuration's grid count, radial count
and angular settings. Rectangular `zx_x_limits_m` and `zx_z_limits_m` bounds
cover the full body with surrounding air. `radial_min_m` and `radial_max_m`
specify distances from the global origin along the configured ray; the starting
45-degree cut crosses the conical surface obliquely near the spherical end.
Spherical field components retain the
standard Cartesian `z` polar axis, independently of the body's `x` axis of
revolution.

Preliminary Gmsh 4.15.2 surface-only sizing, using
`h_max=lambda_interior/ppw` and `h_min=0.8*h_max`, gave:

| Interior mesh sizing | Surface triangles | RWG edges | Complex unknowns | Matrix + estimated LU workspace |
| --- | ---: | ---: | ---: | ---: |
| 6 elements per wavelength | 3,556 | 5,334 | 21,336 | 13.6 GiB |
| 8 elements per wavelength | 6,252 | 9,378 | 37,512 | 41.9 GiB |

These are **mesh outputs, not prescribed triangle counts**. Phase 1 generated
the canonical 6-ppw mesh at `meshes/cone_sphere/surface.obj`; the 8-ppw row is
still the earlier planning estimate. With four current blocks `(J1, M1, J2, M2)`, the system dimension is
`N=4*edge_count`. The double-complex matrix requires `16*N*N` bytes; the table
adds an estimated equal-sized LU workspace. This is a planning estimate for an
A100 80 GB, excluding other allocations and later local refinement. Confirm
the actual mesh counts, available GPU memory and CUDA workspace size before a
solve. Neither mesh level has been demonstrated converged for this GSTC case.

## Mild circular ogive

[`configs/ogive/ppw7.json`](configs/ogive/ppw7.json) selects the gentle circular
ogive and the canonical [`meshes/ogive/ppw7.obj`](meshes/ogive/ppw7.obj). It keeps
2 GHz, interior er=5, exterior air, separate z-to-z / y-to-y matching, general
reciprocal-lossless extraction and the face basis. Both physical electric
dipoles are at `(0,0,0)` m. The prescribed exterior fields remain targets.

| Parameter | Value |
| --- | --- |
| Maximum transverse radius | 0.1049273603 m = 0.7 lambda0 |
| Maximum diameter | 0.2098547206 m = 1.4 lambda0 |
| Left / right axial half-length | 0.12591283236 m = 0.84 lambda0, each |
| Total length | 0.25182566472 m = 1.68 lambda0 |
| Axis / maximum-radius plane center | +x / `(0,0,0)` m |
| Tip half-angle relative to axis | 79.611 degrees, each |
| Surface triangles / RWG edges | 4,146 / 6,219 |
| Complex unknowns | 24,876 |
| Matrix plus estimated equal-sized LU workspace | 18.44 GiB |

The transverse radius matches the earlier sphere's **radius**, not its diameter.
Each axial end is extended by 20%. A circular ogive has pointed ends even for
this mild extension. The canonical mesh uses a uniform 7-ppw target; optional
tip grading is described below. The global ZX sampling origin and both source positions coincide,
and the square sampling window extends equally in x and z.

The reusable [`CircularOgive`](../../src/sie_mom/mesh/circular_ogive.py) geometry
accepts `max_radius_m`, `left_length_m`, `right_length_m`, `center_m` and
`axis_direction`. The center is the center of the maximum-radius plane; for an
asymmetric case it is not the midpoint between tips. Half-lengths must be at
least the maximum radius. Equal half-lengths give a symmetric ogive; setting
both equal to the radius gives a sphere. These are independent dimensions;
tip angles are derived, not additional configuration inputs.

For one half-length a and transverse radius b, the generating circle has
radius `(a*a+b*b)/(2*b)`. The two arcs meet tangentially at the maximum-radius
plane. [`CircularOgiveGenerator`](../../src/sie_mom/mesh/mesh_generators.py)
revolves them with Gmsh. Surface validation uses each geometry's profile,
bounds and normal directions. Tensor extraction uses actual triangle frames.
Point classification continues to use the saved numerical mesh. The C++
solver, field evaluator and plotting routines are shared with the cone case.

### Optional tip grading and the local 4-ppw comparison

[`configs/ogive/ppw4_grad.json`](configs/ogive/ppw4_grad.json) adds an optional
block under `sie.mesh`:

```json
{
  "path": "meshes/ogive/ppw4_grad.obj",
  "points_per_wavelength": 4.0,
  "tip_grading": {
    "size_factor": 0.5,
    "region_fraction": 0.25
  }
}
```

`size_factor` sets `h_tip / h_body`. `region_fraction` sets the transition
length divided by that side's axial half-length. Both must lie in `(0,1]`.
If the block is absent, the existing uniform target is retained. Grading is
currently supported for circular ogives. Each asymmetric half uses its own
length, with the configured center and axis used for spatial projection.

At axial distance s from an end, use `t=clip(s/(region_fraction*a),0,1)` and
`h=h_tip+(h_body-h_tip)*(3*t*t-2*t*t*t)`. This is the agreed cubic smoothstep,
implemented as Gmsh MathEval fields for the two ends, combined by their minimum.
The global minimum size is lowered to permit the local target; boundary-size
extension is disabled for graded generation so the finest curve sizes do not
spread across the whole CAD patch. The triangulation is regenerated and is
not a nested refinement of the old mesh. All per-face tensors are re-extracted.

At 4 ppw, the body target is 16.76 mm, the tip target 8.38 mm and the transition
length 31.48 mm on each side. The original uniform run under
`results/ogive/smoke01/` was preserved; graded preparation, solves, fields and
plots are under `results/ogive/grade01/`.

| Metric | Uniform 4 ppw | Graded 4 ppw |
| --- | ---: | ---: |
| Surface triangles / complex unknowns | 1,420 / 8,520 | 1,582 / 9,492 |
| Median actual edge length nearest the tips | 15.94 mm | 8.61 mm |
| Median actual edge length in the middle | 16.14 mm | 16.26 mm |
| Area-weighted joint fitting residual | 0.1657% | 0.1479% |
| Maximum joint fitting residual | 1.0717% | 0.7276% |
| Matrix plus equal-LU estimate | 2.16 GiB | 2.69 GiB |

The graded two-case GPU solve passed on the RTX 3060, with algebraic residuals
3.28481e-13 and 1.87116e-13. Both cases completed 40,000-point ZX and 300-point
radial field evaluations and six plots each, using the same plotting settings.
All mesh topology/orientation, finite-data and tensor inverse checks passed;
the original prescribed-field fitting warning remains.

The field results are mixed. Exterior complex-vector L2 errors on the radial
cut changed from 5.83% / 3.48% to 4.08% / 1.60% for z-case E/H, and from
2.68% / 2.54% to 3.34% / 1.89% for y-case E/H. On the ZX grid, z-case E/H changed
from 11.78% / 27.54% to 15.41% / 22.97%, and y-case E/H changed from 5.04% / 8.60%
to 10.62% / 14.81%. Refining the tips did not uniformly improve field accuracy.

The grids have identical coordinates. There are 42 mesh-label differences on
the ZX grid and none on the radial cut. An additional comparison restricted
to points labeled exterior by both meshes retains the same mixed trend;
classification differences do not explain it. All points were evaluated, with
no near-surface exclusion or change to integration. The graded mesh remains
a coarse diagnostic, not a converged solution.

The regression suite passed 78 checks, with three opt-in GPU tests skipped by
the default suite settings. The real graded two-dipole GPU solve and field
workflow described above executed separately. New coverage includes factor
validation, backward-compatible configuration loading, asymmetric/rotated
grading and actual tip/body edge sizes. Ruff and whitespace checks passed.

For a fresh repeat, use the same stages as below with
`--config applications/gstc_surface/configs/ogive/ppw4_grad.json` and a fresh
run root such as `results/ogive/grade02/`. Use `--use-existing-mesh` to retain
the tested canonical graded mesh. To change grading or ppw, choose a new mesh
path and generate it before preparing its tensors.

### Run the 7-ppw ogive

Use the root Python environment and existing double-precision CUDA build.
The same `mesh`, `sampling` and `plot` dependencies described below apply.
Use a GPU with sufficient memory; 18.44 GiB is a planning estimate excluding
other allocations. The local 12-GB GPU was tested with a separate 4-ppw mesh.

Run each stage individually from the repository root, stopping on failure.
The canonical mesh is provided, so preparation uses `--use-existing-mesh`.
Each stage requires a fresh output directory. To repeat the entire workflow,
change the outer `run01` to `run02`. Both dipoles run by default.

```powershell
$gstcApp = "applications/gstc_surface"
$gstcRun = "$gstcApp/results/ogive/ppw7/run01"

python -m applications.gstc_surface.prepare_sie_inputs `
  --config "$gstcApp/configs/ogive/ppw7.json" `
  --use-existing-mesh --run-dir "$gstcRun/inputs"

python -m applications.gstc_surface.sie_gstc_surface `
  --inputs-dir "$gstcRun/inputs" --run-dir "$gstcRun/sie"

python -m applications.gstc_surface.observations --solve-dir "$gstcRun/sie"

python -m applications.gstc_surface.evaluate_fields `
  --solve-dir "$gstcRun/sie" `
  --observations-dir "$gstcRun/sie/observations/run01"

python -m applications.gstc_surface.plot_fields `
  --fields-dir "$gstcRun/sie/fields/run01" `
  --output-dir "$gstcRun/sie/fields/run01/plots/run01" `
  --map-reference-region exterior --map-reference-percentile 100 `
  --vmin-db -30 --vmax-db 10
```

Linux Bash:

```bash
source .venv/bin/activate
gstc_app=applications/gstc_surface
gstc_run="$gstc_app/results/ogive/ppw7/run01"

python -m applications.gstc_surface.prepare_sie_inputs \
  --config "$gstc_app/configs/ogive/ppw7.json" \
  --use-existing-mesh --run-dir "$gstc_run/inputs"

python -m applications.gstc_surface.sie_gstc_surface \
  --inputs-dir "$gstc_run/inputs" --run-dir "$gstc_run/sie"

python -m applications.gstc_surface.observations --solve-dir "$gstc_run/sie"

python -m applications.gstc_surface.evaluate_fields \
  --solve-dir "$gstc_run/sie" \
  --observations-dir "$gstc_run/sie/observations/run01"

python -m applications.gstc_surface.plot_fields \
  --fields-dir "$gstc_run/sie/fields/run01" \
  --output-dir "$gstc_run/sie/fields/run01/plots/run01" \
  --map-reference-region exterior --map-reference-percentile 100 \
  --vmin-db -30 --vmax-db 10
```

Plots are saved separately for `electric_z_incidence/` and
`electric_y_incidence/`, beneath the selected fields directory's
`plots/run01/`. Both 2D maps and 1D spherical-component plots are produced.

### Ogive verification, 2026-11-05

Linux preparation of the canonical 7-ppw mesh reproduced the prototype's mesh
ordering, fitted maps and joint residuals exactly. Its area-weighted joint fit
is 0.08443%, maximum joint fit 0.46792%, and maximum per-state fits 3.156% / 4.546%.
The original fitting warning remains. The prepared outputs are under
`results/ogive/ppw7/inputs/`; the documented `run01` path above is fresh.

A separate 4-ppw mesh completed the entire workflow on the local RTX 3060
(12 GB), using the existing Release/double-precision CUDA executables:

| Local workflow check | Result |
| --- | --- |
| Surface triangles / RWG edges / complex unknowns | 1,420 / 2,130 / 8,520 |
| Batched z / y algebraic residuals | 2.40809e-13 / 1.60607e-13 |
| Matrix assembly / factorization / two-RHS solve | 2.063 s / 9.697 s / 0.012 s |
| ZX / radial observations, per case | 40,000 / 300 |
| Plots | Six per case, including 2D maps and 1D curves |

Actual mesh labels were used throughout; there were 57 analytic/mesh label
differences on the ZX grid and one on the radial cut. No near-surface exclusion
was introduced. All evaluated fields were finite. Plots are under
`results/ogive/smoke01/sie/fields/run01/plots/run01/`, in both case directories.

The coarse run is an execution check, not a converged accuracy result. For
example, raw exterior complex-vector L2 errors on the ZX grid are 11.78% / 27.54%
for E/H in the z case and 5.04% / 8.60% in the y case. Exterior radial-cut errors
are 5.83% / 3.48% and 2.68% / 2.54%, respectively. These are unweighted sample
errors over each saved grid, including points near the surface. Display limits
do not change them. The 7-ppw solve remains for a larger GPU, and native Windows
execution of the ogive case remains pending.

The selected regression suite passed 65 checks. Its three opt-in GPU field
checks were initially skipped, then enabled separately and all passed. Coverage
includes the existing cone workflow and new ogive rotation/translation,
asymmetric meshing, sphere limit, geometric validation, tensor export and
two-case command. Ruff and whitespace checks passed. No C++ solver, CUDA kernel,
field evaluator or plotting implementation changed.

## 10-degree / 7-ppw comparison case

[`configs/cone_sphere/a10_ppw7.json`](configs/cone_sphere/a10_ppw7.json) retains
the original frequency, media, dipoles, transformations and numerical settings.
It increases the cone half-angle to 10 degrees and the spherical-end radius to
`0.1019515805286003` m, preserving the original overall length
`0.689067333186545` m. The sphere center remains at zero; both dipoles remain
at `(0.1, 0, 0)` m. The original 7-degree configurations are preserved.

The canonical mesh is
[`meshes/cone_sphere/a10_ppw7.obj`](meshes/cone_sphere/a10_ppw7.obj), copied
without modification from the Linux preparation experiment. Reuse this same
mesh on Windows to compare the same discretization; regenerating with Gmsh
can produce a different triangulation. Its SHA-256 is
`1058266edc0079481c5948359965cf6d1424ebd0f51f94b1cd312ddcb8849c7c`.

| Preparation metric | Value |
| --- | ---: |
| Surface triangles / RWG edges | 6,816 / 10,224 |
| Complex unknowns, four current blocks per edge | 40,896 |
| Median / P95 joint fit residual | 0.112% / 0.802% |
| Area-weighted / maximum joint fit residual | 0.476% / 2.690% |
| Maximum per-state fit residual, z / y | 10.23% / 13.68% |
| Maximum balanced inverse residual | 5.69e-12 |
| Dense complex128 matrix | 24.92 GiB |
| Matrix plus estimated equal-sized LU workspace | 49.84 GiB |

All fits have rank 10. These are CPU preparation diagnostics, not solved-field
accuracy results. The fitting warning remains. The memory estimate excludes
other allocations and is not a measured GPU peak; this case is intended for
the larger GPU server, not the local 12-GB GPU. NGSolve validation remains a
later checkpoint.

On Linux (2026-11-05), preparing this configuration with the canonical mesh
reproduced the experiment's diagnostics and tensor binary exactly. The solve
dry run validated both cases without launching CUDA. Local preparation outputs
are under `results/cone_sphere/a10_ppw7/check01/inputs/`; the `run01` path below
is reserved for the subsequent full run.

Run the following stages individually from the repository root, stopping if
a stage fails. Use the root Python environment (the NGSolve environment is
not needed). The existing double-precision CUDA build must provide
`anisotropic_sphere_app` and `evaluate_surface_fields`; if already configured,
build them with:

```console
cmake --build build --target anisotropic_sphere_app evaluate_surface_fields --parallel 4
```

The root environment needs the `mesh`, `sampling` and `plot` extras. Install
missing dependencies with `python -m pip install -e ".[mesh,sampling,plot]"`.
Outputs are Git-ignored. Each stage requires a fresh output directory; change
the outer `run01` to `run02` when repeating the full workflow.

### Windows PowerShell

Activate your existing `comsol` environment and use the existing CUDA build.
The first command prepares tensors from the committed mesh; it does not
regenerate it. No `--case` filter is used, so both dipoles are solved and plotted.

```powershell
$gstcApp = "applications/gstc_surface"
$gstcRun = "$gstcApp/results/cone_sphere/a10_ppw7/run01"

# 1. Prepare tensors and inspect fitting/inverse diagnostics.
python -m applications.gstc_surface.prepare_sie_inputs `
  --config "$gstcApp/configs/cone_sphere/a10_ppw7.json" `
  --use-existing-mesh `
  --run-dir "$gstcRun/inputs"

Get-Content "$gstcRun/inputs/diagnostics.json"
nvidia-smi --query-gpu=name,memory.free --format=csv

# 2. Solve both transformations using one matrix factorization.
python -m applications.gstc_surface.sie_gstc_surface `
  --inputs-dir "$gstcRun/inputs" `
  --run-dir "$gstcRun/sie"

# 3. Generate and classify the ZX and radial observation points.
python -m applications.gstc_surface.observations `
  --solve-dir "$gstcRun/sie"

# 4. Evaluate fields from saved currents.
python -m applications.gstc_surface.evaluate_fields `
  --solve-dir "$gstcRun/sie" `
  --observations-dir "$gstcRun/sie/observations/run01"

# 5. Plot both cases: 2D maps and 1D spherical-component curves.
python -m applications.gstc_surface.plot_fields `
  --fields-dir "$gstcRun/sie/fields/run01" `
  --output-dir "$gstcRun/sie/fields/run01/plots/run01" `
  --map-reference-region exterior `
  --map-reference-percentile 100 `
  --vmin-db -30 --vmax-db 10
```

### Linux Bash

Use an appropriately sized GPU for stages 2 and 4.

```bash
source .venv/bin/activate
gstc_app=applications/gstc_surface
gstc_run="$gstc_app/results/cone_sphere/a10_ppw7/run01"

python -m applications.gstc_surface.prepare_sie_inputs \
  --config "$gstc_app/configs/cone_sphere/a10_ppw7.json" \
  --use-existing-mesh --run-dir "$gstc_run/inputs"

cat "$gstc_run/inputs/diagnostics.json"
nvidia-smi --query-gpu=name,memory.free --format=csv

python -m applications.gstc_surface.sie_gstc_surface \
  --inputs-dir "$gstc_run/inputs" --run-dir "$gstc_run/sie"

python -m applications.gstc_surface.observations --solve-dir "$gstc_run/sie"

python -m applications.gstc_surface.evaluate_fields \
  --solve-dir "$gstc_run/sie" \
  --observations-dir "$gstc_run/sie/observations/run01"

python -m applications.gstc_surface.plot_fields \
  --fields-dir "$gstc_run/sie/fields/run01" \
  --output-dir "$gstc_run/sie/fields/run01/plots/run01" \
  --map-reference-region exterior --map-reference-percentile 100 \
  --vmin-db -30 --vmax-db 10
```

Plots are written under
`results/cone_sphere/a10_ppw7/run01/sie/fields/run01/plots/run01/`, in separate
`electric_z_incidence/` and `electric_y_incidence/` directories. The `[-30,10]`
dB range is a display choice; it does not change saved fields or error metrics.
The radial and ZX sampling settings are unchanged from the earlier cases.

## Phase 1: geometry and tensor preparation

From the repository root, with the root Python environment active and the
`mesh` extra installed, use:

```console
python -m applications.gstc_surface.prepare_sie_inputs --use-existing-mesh --run-dir applications/gstc_surface/results/cone_sphere/prep03
```

For a configuration whose mesh does not exist, omit `--use-existing-mesh` to
generate it. Both existing mesh files and existing run directories are protected
from overwriting. To try 8 ppw, change the single
`sie.mesh.points_per_wavelength` value and choose a new `sie.mesh.path`, then run
without `--use-existing-mesh`. Reusing a mesh does not regenerate or refine it.

The preparation path is:

1. Generate the tangent geometry, compact unused CAD nodes and orient triangles
   outward before creating the canonical OBJ. Validate closedness, topology,
   geometry and winding; subsequent reads preserve its face/vertex ordering.
2. Evaluate all four Cartesian fields for each transformation at face centroids
   using the existing sphere application's Cartesian field evaluator.
3. Project onto each triangle's actual `(t1, t2)` frame and call the existing
   simultaneous reciprocal extractor.
4. Validate the physical maps and export each map and its full inverse using
   the shared [`tensor_io.py`](../../src/sie_mom/gstc/tensor_io.py) helper. The
   inverse is computed in impedance-balanced units, then converted back to
   physical units; the existing `1e-8` two-sided inverse tolerance is retained.

### Lossless-map inversion check, 2026-11-05

For reciprocal-lossless maps, the balanced matrix `B` is skew-Hermitian, so
`i*B` is Hermitian. Tensor export now solves `(i*B) X = i*I` using SciPy's
Hermitian solver. Arithmetic and records remain complex128; fitted maps,
physical-unit conversions and the `1e-8` two-sided acceptance check are
preserved. General reciprocal-complex maps retain the general inversion path.

This addresses a sensitive inverse at face 442 of the 3,558-face Windows mesh.
The previous inverse failed at `1.46e-8` on Windows and `4.17e-8` when that
mesh was prepared on Linux. Full preparation/export with the new solver passed
on Linux for both the 3,556-face Linux mesh (maximum inverse residual
`2.79e-11`) and the Windows-generated mesh (`5.90e-11`). Their fitted maps
were preserved exactly. The local GSTC fitting warning remains separate from
this inverse check.

The focused regression suite passed 294 tests, including the face-442 map,
general complex inversion, singular-map rejection and losslessness validation.
**Execution of the updated code on Windows remains pending.** Use the existing
Windows mesh with `--use-existing-mesh` and a fresh preparation output directory
for that check; retain the previous diagnostics for comparison.

Each run writes:

| File | Contents |
| --- | --- |
| `configuration.json` | Input configuration snapshot |
| `sie_inputs.json` | Mesh hash/counts, versions, geometry, export status and warnings |
| `diagnostics.json` | Joint/per-state fitting, power balance, rank, conditioning and inverse checks |
| `face_diagnostics.csv` | Face-indexed coordinates, residuals, conditioning and prescribed normal fluxes |
| `surface_samples.npz` | Cartesian fields, actual triangle frames, maps, residuals and smooth-normal control |
| `face_tensors.bin` | Existing SIE layout: 32 little-endian complex128 values per face, map then inverse |

If full-map inversion fails, diagnostics and samples are retained with status
`diagnostics_only`, and the command exits with an error. A successful export
with a fitting warning means the fitted sheet satisfies the material constraints
but does not reproduce every prescribed field exactly on the faceted surface.
The `1e-8` fitting-warning threshold is diagnostic, not an accuracy guarantee.

### First checkpoint, 2026-11-04

The completed preparation run is `results/cone_sphere/prep02/`. The earlier
`prep01/` retains the failed physical-unit inversion attempt for comparison.
Python validation and the existing C++ `mesh_obj_loader_tool` both accepted the
canonical mesh: 1,780 vertices, 3,556 triangles, 5,334 edges, Euler characteristic
2, no holes or non-manifold edges, and consistent outward winding.

| Diagnostic | Result |
| --- | ---: |
| Local coefficient rank | 10 at every face |
| Median joint relative fitting residual | 0.2793% |
| 95th percentile joint relative fitting residual | 2.5946% |
| Maximum joint relative fitting residual | 12.2889% |
| Area-weighted global relative fitting residual | 1.8397% |
| Maximum per-state residual, `z -> z` / `y -> y` | 29.4199% / 35.3731% |
| Individual prescribed normal-power relative L2 defects | Below `3e-16` |
| Smooth-normal control: median / maximum fitting residual | `1.99e-13` / `2.49e-9` |
| Maximum balanced-map condition number | `4.06e5` |
| Maximum coefficient-matrix condition number, physical coefficient units | `1.10e9` |
| Maximum two-sided balanced inverse residual | `8.98e-9` |

The smooth-normal control evaluates the same Cartesian fields at the same
centroids but projects them onto the analytic surface tangent planes. It is
**diagnostic only**; exported tensors use the actual flat-triangle frames.
The much smaller control residual points to faceting as the source of the
joint-fit discrepancy. Individual power balance alone does not remove that
discrepancy: the rotating coherent combination of the two states has a normal
power defect on the faceted interface (relative L2 about 0.84%).

The worst joint residual occurs at face 2218 (zero-based), near
`(0.50330, 0.00547, -0.01111)` m. Inspect the per-face data and compare mesh
resolution before treating the prescribed targets as accurately realized.
These are weighted local GSTC fitting diagnostics, not solved field-error or
scattering-error estimates. No CUDA solve or NGSolve validation has run in this
phase. The current export status is `prepared_with_warnings`.

### Local mesh issue and deferred follow-up, 2026-11-04

**Working decision:** retain the current 6-ppw mesh and synthesized tensors as
the first diagnostic baseline, proceed to SIE solutions and field comparisons,
and assess the practical effect before changing the mesh or extraction method.
Keep the fitting warning attached to these results; this is not yet a
converged or validated example.

Face 2218 is a local geometric approximation concern. Let `R_hat` point from
the source to the face centroid. The analytic cone gives
`n_analytic dot R_hat = 0.15227`, while this flat triangle gives
`n_face dot R_hat = -0.00112`: the planar facet is nearly grazing and slightly
back-facing relative to the source, despite having outward, consistent mesh
winding. It is 11.09 cm from the tip (16.1% of the total axial length).
Its edges are comparable to the local cone radius. This points to insufficient
local geometric resolution, rather than a defect in `compute_face_frames`.

The three edge-neighbor residuals are only 0.047%, 0.078% and 0.206%, compared
with 12.289% on face 2218. Separate CPU-only mesh studies reduced the maximum
joint residual from 12.289% at 6 ppw to 5.647%, 4.017% and 1.727% at 8, 12 and
16 ppw; the area-weighted residual decreased as well. All four meshes passed
the existing Python and C++ topology/orientation checks. These studies support
the geometric explanation but do not establish solved-field accuracy. The
12/16-ppw meshes are diagnostic cases, not dense A100 mesh recommendations.

Detailed local artifacts, including face locations, neighboring-face data and
refinement controls, are beneath `results/cone_sphere/normal_check/`
(Git-ignored). The retained
[diagnostic report](results/cone_sphere/normal_check/README.md) and
[coordinate/source-location figure](results/cone_sphere/normal_check/face_location_xyz.png)
provide the local investigation context.

Deferred work:

- [ ] Inspect the first SIE field comparisons, including the neighborhood of
  face 2218, before deciding whether local mesh changes are needed.
- [ ] If needed, improve local curvature resolution and compare meshes within
  the GPU memory budget. Check both global field metrics and local fitting
  residuals; the latter alone are not solved-field errors.
- [ ] **Column equilibration is deferred.** The diagnostic scaled solve gives
  the same 12.289% residual. A future implementation must restore physical
  coefficient units, distinguish scaled from unscaled conditioning, and
  preserve the current minimum-norm selection for non-unique fits.

## Local surface-parameter plots

[`plot_surface_parameters.py`](plot_surface_parameters.py) reads existing
prepared inputs and colors the actual triangles with their local physical
GSTC coefficients. It does not regenerate meshes, refit tensors or run a solver.
The geometry-independent renderer lives in
[`sie_mom.postprocessing.mesh_plotting`](../../src/sie_mom/postprocessing/mesh_plotting.py);
mesh generation remains in `sie_mom.mesh`.

For the 50%-elongated, graded 5-ppw ogive, run from the repository root:

```bash
gstc_app=applications/gstc_surface
gstc_run="$gstc_app/results/ogive/e50_ppw5_tip025/run01"

python -m applications.gstc_surface.plot_surface_parameters \
  --inputs-dir "$gstc_run/inputs" \
  --output-dir "$gstc_run/inputs/plots/surface/run01" \
  --percentile 98 \
  --show-edges
```

Use a fresh output directory; existing figures are preserved. Omit
`--show-edges` to display colors without triangle outlines. The same CLI works
with the other prepared cone/ogive cases and requires no geometry-specific
equations. If the canonical mesh has changed or the inputs were moved, supply
`--mesh PATH` to the exact original mesh or its saved solve snapshot. Mesh bytes
must match the preparation SHA-256; face ordering, local frames and exported
map values are checked against `surface_samples.npz` before plotting.

The four figures use opposite orthographic views from -y and +y; the x-axis
appears reversed in the +y view, as indicated by its coordinate ticks:

| File | Face values |
| --- | --- |
| `electric.png` | Im(Y): independent 11, 12, 22 entries, in S |
| `magnetic.png` | Im(Z): independent 11, 12, 22 entries, in Ohm |
| `coupling.png` | Re(chi): 11, 12, 21, 22 entries, dimensionless |
| `fit_residual.png` | Saved joint relative extraction residual, in percent |

Plot labels use the literature notation `Y = Y_ee`, `chi = Y_em`,
`gamma = Y_me = -chi.T` and `Z = Y_mm` for the forward map. Values are read
directly from the physical `face_tensors.bin` maps, without conversion to the
normalized optimization coordinates. Labels show only physical tensor
components. The magnetic block `Z` is part of the forward map, not its full
inverse.

These ten real coefficients describe the reciprocal-lossless maps; the other
coupling block follows `gamma = -chi.T`. Reciprocal-complex preparations are
rejected rather than silently discarding their additional coefficients.
Indices 1 and 2 denote the existing face-edge tangent directions. There is no
basis rotation, vertex averaging or angular interpolation. Consequently,
component patterns can reflect changes in tangent directions between faces;
they should not be interpreted as coordinate-independent material contrasts.
Lighting and shading are disabled: each face interior is colored only by its
saved scalar value and the color scale. Optional triangle outlines and their
screen sampling can affect perceived contrast, but do not modify face values.

Each physical block shares one signed, zero-centered linear color scale across
its components and both views. Its limit is the chosen percentile of the
absolute values pooled across the independent coefficients and all faces,
with equal weighting per coefficient/face. The residual map uses a nonnegative
scale. Colorbar extensions indicate saturation. `plots.json` records limits,
clipped-face counts, true extrema and their original zero-based face indices,
along with input hashes and preparation warnings. Use `--percentile 100` in
another output directory to show the full range. Only display colors saturate;
the saved tensor values and residuals remain unchanged.

## Phase 2: solve-only checkpoint

[`sie_gstc_surface.py`](sie_gstc_surface.py) consumes a completed preparation
directory; it does not generate a mesh, synthesize tensors, or evaluate fields.
The preparation's `configuration.json` is authoritative. Changing the live
configuration does not silently change an already prepared case.

Before execution, the coordinator checks the configuration snapshot, mesh
SHA-256, topology/counts, tensor binary layout and finite coefficients. It copies
the prepared artifacts and the exact mesh into a fresh run directory. This
preserves the mesh associated with saved currents when the canonical mesh is
later refined. Existing preparation and solve directories are never overwritten.

From the repository root, using the root virtual environment:

```bash
python -m applications.gstc_surface.sie_gstc_surface --inputs-dir applications/gstc_surface/results/cone_sphere/prep02 --run-dir applications/gstc_surface/results/cone_sphere/sie/run01 --dry-run
```

`--dry-run` checks the inputs and prints the command without creating files or
requiring a GPU. Remove `--dry-run` to execute on a GPU with sufficient memory.
Both modes print the assembly threshold actually passed to C++, in vacuum
wavelengths and metres, together with the touching-triangle coverage count:

```text
Using assembly singularity threshold: 0.157750205 lambda0 (0.0236461609 m)
  Configured minimum: 0.1 lambda0; touching triangle pairs covered: 9543/9543
```

These example values are for the graded 4-ppw ogive. The coordinator treats
`sie.singularity_threshold_lambda0` as a minimum: it measures the largest
centroid distance among all distinct triangles sharing a vertex, including
edge neighbors, and uses the larger of the configured distance and 1.05 times
that maximum. Coverage comes from measuring every touching pair; the 5% factor
provides numerical padding. It checks the strict kernel comparison after float
conversion for both single- and double-precision geometry arithmetic.

The selected value and coverage diagnostics are saved under
`solver.singularity_threshold` in `run.json`. Prepared configurations and tensors
are unchanged, so existing inputs can be reused in a fresh solve directory.
This policy covers touching pairs in matrix assembly; it does not establish
quadrature convergence or change the independent field-evaluation threshold.
The earlier `grade01` run used the fixed 0.1-lambda0 threshold before this helper
was introduced.

The starting 6-ppw mesh needs approximately 13.6 GiB for the matrix plus estimated
LU workspace, before other allocations; retain the A100 80 GB as the accuracy-run
target. Use a separate coarse configuration and mesh to smoke-test the workflow
on a smaller GPU. The actual LU workspace is determined by cuSolver.

The coordinator invokes the existing `anisotropic_sphere_app` executable with
`--basis face --no-fields`. Its sphere-specific observation generator is not
called. Both physical dipole excitations are passed in one batch, sharing one LU
factorization. Algebraic residual verification reassembles the matrix after the
solve when `sie.compute_residual` is enabled. Preparation warnings remain visible
and are retained in `run.json`; they are distinct from algebraic residuals.

The default executable is under
`build/applications/anisotropic_gstc_sphere/` (`anisotropic_sphere_app.exe` on
Windows). Select another build with `--executable PATH`. Build it with:

```bash
cmake --build build --target anisotropic_sphere_app --parallel 4
```

Each solve directory contains:

```text
run.json                       # Command, hashes, warnings and execution status
configuration.json             # Unmodified preparation snapshot
sie_inputs.json                # Unmodified preparation metadata
surface.obj                    # Exact mesh snapshot used by the solve
face_tensors.bin                # Exact prepared tensor records
solver.log                     # C++ stdout and stderr
batch_metrics.json             # Shared assembly/LU/solve timings and dimensions
cases/electric_z_incidence/
    currents.bin               # Physical complex128 J1, M1, J2, M2 blocks
    solver_metrics.json        # Per-case RHS/residual diagnostics
cases/electric_y_incidence/
    currents.bin
    solver_metrics.json
```

The declared preparation diagnostics are copied alongside these files. The
original paths inside `sie_inputs.json` are preparation provenance; use
`run.json`'s file paths for the solve snapshot. The original configuration may
request field evaluation, but this coordinator always records and applies the
solve-only override `evaluate_fields=false`.

After a successful process, the coordinator validates case IDs, mesh dimensions,
physical current layout/size, finite coefficients and requested residual outputs.
It records status `solved` only after those checks pass. Failures retain the
input snapshot, log and a `failed` manifest. A solved status is not an accuracy
validation of the prescribed transformation.

### Coarse workflow check, 2026-11-04

A separate 2-ppw configuration was prepared under
`results/cone_sphere/smoke01/`, retaining the same geometry, 2 GHz frequency,
media, tensors' extraction constraints and two physical dipole cases. Only
`sie.mesh.points_per_wavelength` and `sie.mesh.path` were changed in that local
configuration. The canonical 6-ppw mesh and configuration were preserved.

The mesh files have distinct roles:

| Mesh | Triangles | Location relative to this application |
| --- | --- | --- |
| Canonical 6-ppw baseline | 3,556 | [`meshes/cone_sphere/surface.obj`](meshes/cone_sphere/surface.obj) |
| Generated 2-ppw coarse mesh | 422 | [`results/cone_sphere/smoke01/surface.obj`](results/cone_sphere/smoke01/surface.obj) |
| Exact coarse mesh snapshot used by the solve | 422 | [`results/cone_sphere/smoke01/sie/surface.obj`](results/cone_sphere/smoke01/sie/surface.obj) |

The two coarse OBJ files are byte-identical. They live under Git-ignored
`results/`, alongside the local coarse configuration, rather than replacing the
tracked baseline mesh. Use the solve's snapshot when inspecting its fields.

The actual CUDA run used an RTX 3060 (12 GB), driver 580.178.04, CUDA Toolkit
13.0, and the Release/double-precision build. GPU access required execution
outside the agent's sandbox. Results are in `smoke01/sie/`:

| Check | Result |
| --- | --- |
| Surface triangles / RWG edges | 422 / 633 |
| Complex unknowns, four blocks per edge | 2,532 |
| Dense matrix storage | 102,576,384 bytes, approximately 0.096 GiB |
| Cases / LU factorizations | 2 / 1 |
| Matrix assembly | 0.210 s |
| LU factorization | 0.317 s |
| Shared two-RHS solve | 0.00306 s |
| Residual matrix reassembly | 0.187 s |
| Scaled relative algebraic residual, electric z | `1.51713e-13` |
| Scaled relative algebraic residual, electric y | `1.27744e-13` |

The run command was:

```bash
python -m applications.gstc_surface.sie_gstc_surface --inputs-dir applications/gstc_surface/results/cone_sphere/smoke01/inputs --run-dir applications/gstc_surface/results/cone_sphere/smoke01/sie
```

For a new coarse check, copy the main configuration into a fresh results folder,
set `sie.mesh.points_per_wavelength=2.0` and give `sie.mesh.path` a separate mesh
path under that folder, relative to this application. Run `prepare_sie_inputs`
with that configuration and a fresh preparation directory, then pass it as
`--inputs-dir` to the solve coordinator. Use a fresh solve directory each time.

Both current files and their metadata passed output validation. Preparation
retained its flat-triangle fitting warning (maximum joint fit residual 13.406%
on this coarse mesh). This verifies workflow execution, not transformation
accuracy. No general-surface fields or plots were produced at this checkpoint.

The focused Python regression suite passed 301 checks. Its two opt-in sphere
CUDA integration tests were then enabled and passed separately, exercising the
existing batch workflow with residual/field evaluation enabled and disabled.
New coordinator tests, Ruff checks and `git diff --check` passed. Windows and the
large-mesh accuracy run remain to be tested on the selected server.

## Phase 2: observation preparation and field evaluation

Observation preparation and field evaluation are separate from synthesis and
solving. [`observations.py`](observations.py) generates the configured ZX grid
and radial ray and classifies points against the exact saved mesh.
[`evaluate_fields.py`](evaluate_fields.py) then invokes the dedicated
[`evaluate_surface_fields.cpp`](evaluate_surface_fields.cpp) tool on the saved
physical currents. Neither stage assembles a matrix or repeats a solve.

Install the optional classification dependencies in the root environment:

```bash
python -m pip install -e ".[sampling]"
cmake --build build --target evaluate_surface_fields --parallel 4
```

The shared [`MeshPointClassifier`](../../src/sie_mom/mesh/point_classification.py)
uses Trimesh with `process=False` and `validate=False`, preserving vertices,
face order and winding. The default backend is Trimesh's existing NumPy/rtree
ray classifier. No new ray-intersection algorithm is implemented here. Embree
is optional: install `embreex` and select `--backend embree` explicitly if
needed. Missing Embree does not silently change a requested backend. See
[Trimesh's dependency documentation](https://trimesh.org/install.html).

For the existing coarse solve, run these commands from the repository root:

```bash
python -m applications.gstc_surface.observations --solve-dir applications/gstc_surface/results/cone_sphere/smoke01/sie
python -m applications.gstc_surface.evaluate_fields --solve-dir applications/gstc_surface/results/cone_sphere/smoke01/sie --observations-dir applications/gstc_surface/results/cone_sphere/smoke01/sie/observations/run01
```

These commands were executed for the check below; use a fresh `--output-dir`
when repeating either stage. Observation preparation is CPU-only. Field
evaluation requires the CUDA device. The root virtual environment is sufficient;
the NGSolve environment is not used.

Observation preparation defaults to `observations/run01/` beneath the solve
directory and writes:

```text
observations.json              # Mesh identity, backend, grid settings and hashes
observations_zx.csv            # point_index,x,y,z,region; metres, region 1/2
observations_radial.csv
```

Use `--sampling-config PATH` to select only the sampling section of another
configuration for a zoom or different cut, keeping the original solve and its
physics fixed. Classification happens once per grid and is reused by both
dipoles. Analytic `geometry.contains()` is retained for reference
comparisons, and its disagreement count is recorded without changing mesh-based
labels. No near-surface distance exclusion mask is applied. Trimesh does not
define containment exactly on a triangle; deliberately on-surface evaluation
points must be assigned their intended regional label separately.

Field evaluation defaults to `fields/run01/` beneath the solve directory:

```text
fields.json                   # Commands, current/output hashes, timings and status
configuration.json            # Original solve configuration
observations.json             # Observation metadata and copied input CSVs
observations_zx.csv
observations_radial.csv
electric_z_incidence/
    sie_physical_zx.csv        # Cartesian complex E and H; shared CSV schema
    sie_physical_radial.csv
    zx.log
    radial.log
electric_y_incidence/
    sie_physical_zx.csv
    sie_physical_radial.csv
    zx.log
    radial.log
```

`--case ID` evaluates one case, and `--executable PATH` selects another evaluator
build, including `.exe` on Windows. A field stage checks the saved mesh,
configuration, preparation metadata, currents and observation hashes before
execution, then checks finite output fields and preservation of observation
indices, coordinates and region labels. Existing output directories are refused.
The original solve manifest and current files remain unchanged.

The C++ evaluator consumes supplied labels; it has no cone/sphere classifier or
tensor extraction. Region 1 selects physical `J1,M1` and the interior medium;
region 2 selects `J2,M2` and the exterior medium. The existing source and field
libraries form both E and H as `F_total = F_incident(region) - Q(region)`. Only
physical simulation excitations are added. Prescribed exterior fields remain
targets, not extra incident sources.

Integration independently selects ordinary quadrature or regular plus singular
integration per observation/triangle pair. The existing kernel threshold is
fixed at 0.1 times the wavelength in that region. The configuration's
`sie.singularity_threshold_lambda0` controls matrix assembly, not this field
kernel. No kernel behavior was changed for this checkpoint.

A local diagnostic on the current 200-by-200 ZX grid found 11,660 points inside
the analytic geometry versus 11,576 inside the 3,556-face mesh: 84 region labels
differ. At face 2218, a point 0.5 mm outward from its centroid is outside the mesh
but inside the analytic body. The geometric classification and CUDA kernel's
distance-based singularity-extraction selection have independent roles. The
Trimesh classifier reproduced those 11,576 interior points in 3.46 s.

### Saved-current field check, 2026-11-04

The 422-face coarse mesh was evaluated on the same RTX 3060, without a new
factorization. Trimesh 5.2.0 and rtree 1.4.1 were installed in the root environment.
The 40,000 ZX samples classified in 0.743 s and the 300 radial samples in 0.007 s.
There were 464 analytic/mesh label differences on the coarse ZX grid and four
on the radial ray; the field evaluator used the mesh labels throughout.

| Case / grid | Samples | Evaluation process and CSV validation |
| --- | --- | --- |
| Electric z / ZX | 40,000 | 1.56 s |
| Electric z / radial | 300 | 0.262 s |
| Electric y / ZX | 40,000 | 1.33 s |
| Electric y / radial | 300 | 0.276 s |

All fields were finite and all coordinates, indices and region labels matched
their observation inputs. The broader Python suite passed 322 checks. Five
opt-in GPU tests were initially skipped by its default configuration, then
enabled separately and all passed: electric/magnetic incident-field recovery
with zero currents, regional current-block selection and field linearity, and
the two existing sphere application integration checks. Ruff and whitespace
checks passed. The coarse run verifies execution and data conventions; it is
not a large-mesh accuracy check. Diagnostic plots are available as described
below.

## Saved-field diagnostic plots

[`plot_fields.py`](plot_fields.py) reads the completed field stage and produces
six figures per case: ZX component comparisons for E and H, radial magnitude
comparisons for E and H, and radial real/imaginary component comparisons for E
and H. Each shows SIE and the selected prescribed transformation. It performs no
solve or field integration.

Reuse is concentrated in [`src/sie_mom/postprocessing`](../../src/sie_mom/postprocessing/):
`load_field_csv`, `complex_vector`, `spherical_components`,
`spherical_coordinates_from_cartesian`, `zx_grid`, and `to_decibels` handle
data loading, field assembly, projection, grid ordering and display conversion.
The existing Cartesian prescribed-field evaluator supplies targets. This
application adds the mesh outline, case/reference selection and figure layout.

Current readable coarse-mesh figures are in:

```text
results/cone_sphere/smoke01/sie/fields/run01/plots/run03/
    plots.json
    electric_z_incidence/
        zx_E_spherical_db.png
        zx_H_spherical_db.png
        radial_E_spherical_db.png
        radial_H_spherical_db.png
        radial_E_spherical_complex.png
        radial_H_spherical_complex.png
    electric_y_incidence/
        ...same six filenames...
```

The command used was:

```bash
python -m applications.gstc_surface.plot_fields --fields-dir applications/gstc_surface/results/cone_sphere/smoke01/sie/fields/run01 --output-dir applications/gstc_surface/results/cone_sphere/smoke01/sie/fields/run01/plots/run03 --map-reference-region exterior --map-reference-percentile 100 --vmin-db -50 --vmax-db 0
```

Use a fresh `--output-dir` (for example `plots/run04`) when rerendering. The
default for a new field run is `plots/run01`. Install the root environment's
`plot` and `sampling` extras if needed: `python -m pip install -e ".[plot,sampling]"`.

The default basis is spherical about the **global origin `(0,0,0)`**, with the
standard **z polar axis**. It is not centered on the displaced dipoles and does
not change the GSTC face basis. Distances on radial plots are in metres. Use
`--basis cartesian` to display x/y/z components instead. Both cases use the
configured global-origin radial ray, initially theta=45 degrees and phi=0.

For ZX maps, a common color scale is shared by all three components and both
columns for each field. The default reference is the **98th percentile of the
prescribed vector magnitude over all samples**; the default dB color range is
`[-80, 0]`. The command above instead sets **0 dB to the prescribed exterior
vector peak** and uses `[-50, 0]` for a readable exterior angular pattern.
Stronger interior values saturate; colorbar arrows indicate saturation. This
applies the normalization/range approach from the sphere publication figures,
without copying their source- and geometry-dependent numerical limits. For
these saved electric cases, the exterior E peak is about 86 dB (z case) and
80 dB (y case) below the all-sample source peak.

`--map-reference-percentile` selects a percentile in `(0,100]` of the chosen
reference region; use `100` for its peak or `98` to reduce the effect of
isolated maxima. `--map-reference-region all` includes both media;
`exterior` uses the saved exterior labels. `--vmin-db` and `--vmax-db` set the
color limits independently of the reference. A deeper requested minimum also
lowers the map's numerical zero floor, so a publication-style minimum such as
`-112` is not accidentally clamped at `-80`. `--floor-db` sets the default
range and the radial zero floor. `--scale linear` displays physical magnitudes
with the chosen reference as the upper color limit; omit the dB-limit flags
in that mode. No component is normalized to its own peak, so suppressed and
zero-prescribed components remain comparable.

For quick color-only diagnostics, `[-25, 0]` dB gives stronger E-theta
contrast, while `[-30, 10]` dB reduces interior H-phi saturation and reveals
its two lobes more clearly. Lowering the maximum to `-10` increases yellow
saturation. These are display choices, not accuracy improvements. The current
wide, vertically narrow window and the displacement between the dipoles and
the global spherical origin limit comparison with the sphere's publication
figures; a source-centered component view and taller sampling window remain
future diagnostic work.

Radial magnitude plots use the prescribed vector peak as a common reference
across components. Real and imaginary curves retain V/m or A/m. Regional curves
are broken at region transitions; dotted boundary markers show the midpoint of
the neighboring samples' crossing bracket. ZX outlines are the saved numerical
mesh's intersection with `y=0`, and black crosses show physical dipole positions.
Mesh SHA-256 checking prevents mixing a coarse field run with the 6-ppw outline.

References are matched through physical excitation field IDs, independently of
case/transformation ordering. If that match is ambiguous, select `--case ID`
and `--reference-transformation ID`. The existing mesh region labels are used
for targets too; plotting does not reclassify against the analytic surface.

`plots.json` records settings, display references, figure paths and separate
interior/exterior complex-vector L2 comparisons. Errors use raw complex fields
with no phase alignment or amplitude fitting. They are unweighted sample
comparisons, distinct from algebraic residuals and power integrals. Color limits
and display floors do not change those errors or the physical field CSVs.
The 422-face results remain coarse workflow diagnostics pending a larger-mesh
accuracy study.

## Directory organization

```text
gstc_surface/
    README.md
    workflow.py               # Configuration adapter
    artifacts.py              # Read-only prepared-input and observation checks
    geometry.py               # Canonical surface generation and validation
    prepare_sie_inputs.py     # CPU-only synthesis and tensor export
    sie_gstc_surface.py        # Solve prepared inputs in one batch; no fields
    validate_ngsolve.py        # Independent FEM with the same interface triangles
    observations.py            # CPU mesh classification and reusable point grids
    evaluate_fields.py         # Saved-current field orchestration
    evaluate_surface_fields.cpp # Geometry-independent E/H evaluator
    plot_fields.py             # Shared postprocessing with surface-specific plots
    configs/                  # Physical cases and numerical settings
        cone_sphere/          # Cone-sphere configurations
        ogive/                # Circular-arc spindle (ogive) configurations
    meshes/                   # Canonical surface meshes and CAD geometry
    publication/              # Example 3 reproduction and figure scripts
        README.md
        figures/              # Selected publication figures
    results/                  # Generated runs and diagnostic plots; Git-ignored
```

Use one application for both geometry variants. Distinct configurations select
the geometry and transformation; a separate application or validator copy is
not needed for each case.

Use descriptive names of moderate length, such as `cone_sphere.json`,
`surface.obj`, `fields_radial.npz` and `metrics.json`. A future run path could be
`results/cone_sphere/ngsolve/run01/`. These names are illustrative, not existing
cases or a finalized output schema. Detailed dimensions, frequency, media,
mesh settings, source definitions and provenance belong in configuration/run
metadata. Add identifying information to filenames only when useful. Names
must be usable on both Linux and Windows.

Track configurations, workflow scripts, documentation and canonical input
meshes. Keep generated tensors, currents, volume meshes, solutions and
diagnostic plots beneath `results/`. Selected paper figures belong in
`publication/figures/` when the example is ready.

## Reuse and scope

Reuse the existing general-surface mathematics:

- [`mesh_generators.py`](../../src/sie_mom/mesh/mesh_generators.py): cone-sphere
  geometry generation, including a separate tangent-cone-sphere generator.
- [`tangent_cone_sphere.py`](../../src/sie_mom/mesh/tangent_cone_sphere.py):
  shared analytic dimensions, inside checks and normal directions.
- [`face_frames.py`](../../src/sie_mom/mesh/face_frames.py): intrinsic tangent
  frames, vector projection and tensor lifting.
- [`surface_parameters.py`](../../src/sie_mom/gstc/surface_parameters.py):
  extraction of the four physical tangential tensor blocks.
- [`tensor_io.py`](../../src/sie_mom/gstc/tensor_io.py): shared physical map
  validation, inversion and SIE record packing; also used by sphere preparation.
- [`gstc_tensor_utils.cuh`](../../include/sie_mom/cuda/gstc_tensor_utils.cuh):
  matching face-frame conventions and Cartesian tensor lifting in C++/CUDA.
- [`gstc_sie.hpp`](../../include/sie_mom/solvers/gstc_sie.hpp): shared GSTC SIE
  solver interface.

Preserve the surface mesh's face indexing and vertex order when associating
per-face tensors with intrinsic frames. Adapt the spherical application's
geometry checks, region classification, field sampling and boundary outlines
for the chosen surface.

For this phase, the configuration adapter reuses the sphere application's field
definitions/parsers and SIE setting types, and preparation imports its Cartesian
prescribed-field evaluator. Those shared configuration dependencies can be
extracted when the common runner interface is established; the extraction
mathematics and CUDA solver have not been duplicated.

NGSolve uses the optional shared backend in
[`src/sie_mom/validation/ngsolve`](../../src/sie_mom/validation/ngsolve/).
`model.py` supplies the Maxwell/GSTC weak form and source/field conventions;
`volume_mesh.py` supplies analytic-sphere and imported-interface volume meshes;
`tensor_transfer.py` supplies frames and exact facewise tensor correspondence.
These modules do not import application configurations. The sphere application
remains the regression baseline and its old experiment entry points continue
to work.

## NGSolve with the same SIE interface

The runner consumes prepared inputs. Every physical tensor and intrinsic frame
is associated with its original SIE triangle. Netgen meshes the core, air and
PML around this unchanged interface. Coordinate matching checks a bijection
between FEM boundary elements and original SIE faces, independent of numbering.
The runner also checks that every GSTC triangle separates core and air, and
that the tetrahedra preserve the SIE polyhedral volume. It does not curve or
refine the sheet. Higher field order does not change its planar geometry.

The original t1/t2 vectors and the derived two-sided sheet admittance are native
constant surface coefficient fields. Every quadrature point on a face receives
the same saved physical tensor through its original frame. There is no spatial
interpolation or Python callback during assembly. All four original blocks are
retained. The current E-only formulation requires each magnetic block Z to be
invertible and rejects condition numbers above 1e10; it never substitutes a
pseudoinverse or regularization.

NGSolve is optional. If desired, install into the root virtual environment:

```bash
python -m pip install -e ".[ngsolve]"
```

No new environment is required. The existing sphere experiment environment
can also run the new module using `PYTHONPATH=src`; the refactoring validation
used that environment without changing any packages.

For a fresh order-1 workflow check, using the existing SIE observation grids
and field data:

```bash
gstc_app=applications/gstc_surface
gstc_run="$gstc_app/results/ogive/e50_ppw5_tip025/run01"

python -m applications.gstc_surface.validate_ngsolve \
  --inputs-dir "$gstc_run/inputs" \
  --observations-dir "$gstc_run/sie/observations/run01" \
  --sie-fields-dir "$gstc_run/sie/fields/run01" \
  --output-dir "$gstc_run/ngsolve/order1_run03" \
  --order 1 --ppw 4 --threads 4
```

The output directory must be fresh. Omit the two observation/comparison options
for a solve/checkpoint-only run. The default air radius is 1.8 lambda0 so the
entire current ZX grid fits in physical air; the PML thickness is 0.25 lambda0.
All requested points must lie in core or physical air. The runner verifies
saved region labels against FEM materials and raises on disagreement or PML
points; it does not silently reclassify or discard samples. Mesh, field-order
and PML choices are numerical controls requiring convergence assessment.

For a field-order comparison, add `--mesh-run-dir` pointing to a completed FEM
run and choose a fresh output directory, for example:

```bash
python -m applications.gstc_surface.validate_ngsolve \
  --inputs-dir "$gstc_run/inputs" \
  --observations-dir "$gstc_run/sie/observations/run01" \
  --sie-fields-dir "$gstc_run/sie/fields/run01" \
  --mesh-run-dir "$gstc_run/ngsolve/order2_run01" \
  --output-dir "$gstc_run/ngsolve/order3_run02" \
  --order 3 --ppw 4 --threads 4
```

Reuse requires identical configuration, interface and mesh/PML settings. The
runner checks interface correspondence, core/air adjacency, core volume and
enclosing radii again, copies the saved volume mesh unchanged and records its
SHA-256. It restores the original radial PML transformation, which is not
stored in Netgen's volume file. Only the field space changes in this comparison.

Each batch shares one factorization across both dipole cases. Outputs include
the exact input snapshots, `mesh.vol.gz`, `interface.npz` (FEM-to-SIE face IDs),
per-case `solution.sol`, `metrics.json`, and physical field CSVs under
`fields/run01/`. Comparison metrics use unweighted complex vector L2 errors
against SIE at the same saved coordinates and regions, without phase/amplitude
fitting. Interior total-field errors can be dominated by the analytic source.

Field CSV references accept both Windows `\\` and Linux `/` separators when
reading copied runs. New field manifests use `/` on both platforms. The FEM
runner checks all requested SIE comparison files and their hashes before
meshing or factorization; copying a run does not require editing its manifest.

The existing field plotter can display FEM fields against prescribed targets:

```bash
python -m applications.gstc_surface.plot_fields \
  --fields-dir "$gstc_run/ngsolve/order1_run03/fields/run01" \
  --map-reference-region exterior --map-reference-percentile 100 \
  --vmin-db -30 --vmax-db 10
```

It reads `solver_label` from the field manifest; old field runs default to
"SIE" and FEM runs display "NGSolve". Neither plotting nor sampling reruns the
CUDA solver. A completed batch and small algebraic residual establish workflow
execution, not field accuracy. The initial order-1 ogive comparison shows large
exterior differences and must not be presented as converged FEM validation.

The 2026-11-05 Linux checkpoint tested orders 1–3 on the identical saved volume
mesh (45,135 tetrahedra). Shared PARDISO factorization took 6.23 s, 84.42 s and
480.69 s respectively. Order 3 has 1,153,652 free DOFs. Both dipoles are
centered electric sources at 2 GHz, with interior relative permittivity 5 and
air outside. All 3,766 original interface triangles, physical tensors and
observation coordinates are preserved across orders.

Exterior complex-vector L2 differences relative to the saved SIE fields:

| Case / samples | E: order 2 → 3 | H: order 2 → 3 |
| --- | ---: | ---: |
| z dipole, radial | 8.76% → **8.13%** | 10.13% → **12.79%** |
| y dipole, radial | 11.31% → **7.39%** | 10.58% → **5.64%** |
| z dipole, ZX | 29.79% → **60.77%** | 76.62% → **124.45%** |
| y dipole, ZX | 24.38% → **38.77%** | 52.93% → **73.94%** |

Each case includes all 40,000 ZX points and 300 radial points, with separate
interior/exterior comparisons and no point exclusions or phase/amplitude fit.
Radial agreement mostly improves, while full ZX agreement worsens. The largest
1% of exterior point differences account for 80.4–84.5% of the squared
discrepancy, with the worst points near the sheet. All batches had algebraic
residuals below 5e-15. This checkpoint establishes a reusable comparison
workflow. Field convergence remains unverified; near-sheet discrepancies,
volume-mesh and PML sensitivity require investigation. The detailed
[checkpoint report and figure locations](results/ogive/e50_ppw5_tip025/run01/ngsolve/README.md)
are under ignored results. Native Windows execution of this refactor has not
been tested.

The current milestone includes geometry and tensor preparation, a working SIE
solve, draft Example 3 figures and an independent FEM comparison through the
shared backend. Further accuracy studies can follow separately.
