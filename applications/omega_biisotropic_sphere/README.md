# Omega-biisotropic sphere application

This directory contains the spherical reciprocal omega-sheet impedance-
matching example. It compares two parameter-extraction routes for a
prescribed field transformation:

- the projected analytical formulas for the lossless parameters
  `Y = j B`, `Z = j X`, and real `chi`;
- the simultaneous real least-squares system assembled from the same GSTCs.

The implemented case follows a centered, z-directed electric dipole at
10 GHz. The sphere has radius `a = 0.7 lambda_0`, relative permittivity 7
inside, and free space outside. The exterior dipole amplitude is selected so
that the normal time-average real power is conserved pointwise.

`svwf_dipole_impedance_matching.py` extracts the sheet parameters through
`sie_mom.gstc.omega_biisotropic` and passes the resulting tensor blocks to the
SVWF collocation solver. It reports the prescribed power mismatch, solver
residual, interior reflection, and exterior transmission error.

## Complete example workflow

Install the optional plotting dependency once with

```bash
python -m pip install -e '.[plot]'
```

Generate the default constrained real least-squares solution and its sampled
field data with

```bash
python applications/omega_biisotropic_sphere/svwf_dipole_impedance_matching.py \
    --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json
```

Then render the two comparison figures from the saved data:

```bash
python applications/omega_biisotropic_sphere/plot_dipole_impedance_matching.py \
    --run-dir applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7/svwf/least_squares
```

The default result path is organized as
`results/<case_id>/<backend>/<run_name>`. The configuration therefore writes
this run beneath `results/budhu_10ghz_eps7/svwf/least_squares`. An explicit
`--run-dir` can still be used for a temporary or specially named run.

To compare the projected analytical extraction, generate a separate run with

```bash
python applications/omega_biisotropic_sphere/svwf_dipole_impedance_matching.py \
    --extraction-method projected \
    --run-dir applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7/svwf/projected
```

For a spatial extent close to Fig. 3 of the paper, use

```bash
python applications/omega_biisotropic_sphere/svwf_dipole_impedance_matching.py \
    --extraction-method projected \
    --run-dir applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7/svwf/projected \
    --zx-span-radii 3.0 \
    --radial-max-radii 4.2
```

Each run is self-contained:

```text
results/budhu_10ghz_eps7/svwf/least_squares/
    configuration.json
    run.json
    svwf_coefficients.npz
    fields/
        prescribed_zx.csv
        svwf_zx.csv
        prescribed_radial.csv
        svwf_radial.csv
    plots/
        fields_zx.png
        radial_cut.png
        real_ex_zx.png
        real_ex_radial_cut.png
```

The CSV files use Cartesian coordinates and Cartesian complex field
components, split into real and imaginary columns. They are the canonical data
exchange format for this example so that a future C++ solver can write the same
schema and reuse the plotting script. All saved phasors use the
`exp(+j omega t)` convention. `configuration.json` records the complete,
resolved input used for the run, including any command-line overrides.
`run.json` records the physical parameters, diagnostics, and relative
data-file paths.

The saved coefficients can be evaluated on a different observation grid
without assembling or solving the collocation system again. The source run
supplies the physical and SVWF configuration; sampling options may be changed:

```bash
python applications/omega_biisotropic_sphere/svwf_dipole_impedance_matching.py \
    --evaluate-from-run applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7/svwf/least_squares \
    --run-dir applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7/svwf/fields_300 \
    --zx-grid-size 300 \
    --radial-points 501
```

The output manifest labels this as `workflow_mode: field_evaluation` and
records the source coefficient archive. A distinct `--run-dir` is required so
the solved run cannot be overwritten accidentally.

Field sampling is serial by default. Add `--field-workers N` to split the
observation points into deterministic process-based chunks. By default each
worker receives approximately four chunks; `--field-chunk-size POINTS` can
override that policy for benchmarking. Both settings and the parent/worker
peak-memory measurements are recorded in `run.json`.

The checked-in configuration under `configurations/` can be copied to define a
new material or frequency without duplicating the application. Give the copy a
new `case_id`, change `physical.frequency_hz` or
`physical.interior.epsilon_r`, and pass it with `--config`. Individual values
can also be overridden for a temporary run, for example

```bash
python applications/omega_biisotropic_sphere/svwf_dipole_impedance_matching.py \
    --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json \
    --frequency-hz 12e9 \
    --epsilon-r-interior 4.0 \
    --run-dir applications/omega_biisotropic_sphere/results/12ghz_eps4/svwf/least_squares
```

The first two figures compare the dominant tangential components. The
`real_ex` figures reproduce the Cartesian quantity used in the paper:
`Re(E_x)` in the ZX plane and along the radial cut at 45 degrees. For those
figures, the plotting routine converts the arbitrary saved effective-moment
normalization to the paper's physical interior current moment
`I Delta z = 1 A m`.

Reusable analytical fields and GSTC algebra live in the installed Python
package under `src/sie_mom`; the application file contains only the physical
case and the SVWF validation flow.

## Small-cap perturbation

The same two coordinators also run a localized sheet defect. The configuration
`configurations/budhu_10ghz_eps7_cap5.json` keeps the dipole, bulk media, radius,
and observation cuts of the matching example. It adds:

```json
"cap_perturbation": {
  "theta_degrees": 90.0,
  "phi_degrees": 0.0,
  "angular_radius_degrees": 5.0,
  "parameter_scale": 0.1
}
```

This is a **5-degree angular radius** (10-degree diameter) centered on the
positive x axis. After either projected or least-squares extraction,
`dipole_impedance_matching.apply_cap_perturbation()` multiplies `B`, `X`, and
`chi` by the real scale at points inside the cap. It preserves the reciprocal
lossless structure. Removing the section, or setting the scale to 1, recovers
the unperturbed sheet. The allowed scale is `0 < parameter_scale <= 1`:
the SIE export needs an invertible forward matrix.

SVWF evaluates this mask at its angular collocation points. SIE evaluates the
same mask at OBJ face-centroid directions, then rebuilds **both** the forward
matrix and its inverse in `face_tensors.bin`. The SIE input manifest records
the affected face indices; SVWF reports the affected collocation-point count.
An active cap containing no sampled points is rejected. The original mesh
can be reused because only its sheet tensors change.

Run SVWF and plot its saved fields:

```bash
.venv/bin/python applications/omega_biisotropic_sphere/svwf_dipole_impedance_matching.py \
    --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7_cap5.json
.venv/bin/python applications/omega_biisotropic_sphere/plot_dipole_impedance_matching.py \
    --run-dir applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7_cap5/svwf/least_squares
```

Run SIE with the existing sphere mesh and include the SVWF fields in its plots:

```bash
.venv/bin/python applications/omega_biisotropic_sphere/sie_dipole_impedance_matching.py \
    --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7_cap5.json \
    --use-existing-mesh \
    --svwf-run applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7_cap5/svwf/least_squares
```

Both commands save under `results/budhu_10ghz_eps7_cap5/<backend>/least_squares`.
Change `extraction_method` in the shared configuration to `projected` to use
that extraction for both solvers; the mask is identical. Do not reuse tensors
from a different cap configuration: `--reuse-inputs` checks this explicitly.

The saved analytical fields are labeled **Unperturbed reference**. Departures
from them are expected for the modified sheet, so the cap run reports an
exterior change from reference and an interior reflected-field norm, rather
than treating those differences as solver errors. Baseline extraction and
prescribed-power diagnostics still describe the original transformation.

The cap configuration starts with `n_max=8` and a `73 x 144` angular grid,
approximately 2.5-degree spacing. This samples the small cap but does not
establish modal or mesh convergence. These first 2D/radial plots are an
exploratory experiment; interpreting oscillations as surface waves requires
further checks. Only the sheet is perturbed; the bulk dielectric remains
homogeneous, so this does not model the full air cylinder in COMSOL.

#### TinkerCliffs PPW-9 and PPW-10 SIE sweep

The PPW-9 and PPW-10 cap configurations can be run as a serialized Slurm
array on the TinkerCliffs H200 queue:

```bash
cd /home/germanf/repo/sie_mom
sbatch applications/omega_biisotropic_sphere/tinkercliffs_sie_cap_ppw9_10.sbatch
```

The array indices are the PPW values and `%1` permits only one of them to run
at a time. Results remain separate under
`results/budhu_10ghz_eps7_cap5_ppw{9,10}/sie/least_squares`. The job records
the host, Git commit, GPU, and Slurm identifiers in each run's
`slurm_environment.txt`. It also configures a dedicated
`build-tinkercliffs-sm90` executable instead of reusing an SM86 binary built
for the Ampere A10.

Submit only one resolution when desired:

```bash
sbatch --array=9 applications/omega_biisotropic_sphere/tinkercliffs_sie_cap_ppw9_10.sbatch
sbatch --array=10 applications/omega_biisotropic_sphere/tinkercliffs_sie_cap_ppw9_10.sbatch
```

A completed result is skipped by default. To deliberately replace it, add
`--export=ALL,FORCE_RERUN=1`. If the repository is not located at the default
TinkerCliffs path, set it at submission time with
`--export=ALL,SIE_MOM_REPO=/path/to/sie_mom`.

#### PPW-8 mesh with cap-local refinement

`configurations/budhu_10ghz_eps7_cap5_ppw8_capref32.json` retains PPW 8 over
the sphere while using PPW 32 around the perturbed cap. The physical cap still
ends at 5 degrees. The fully refined mesh extends through 5.5 degrees, giving
the material discontinuity a 0.5-degree fine-mesh buffer, and then grades
smoothly back to PPW 8 by 10 degrees. Existing uniform-PPW configurations do
not contain `sie.mesh.cap_refinement` and retain their original behavior.

Generate and inspect only the mesh with:

```bash
.venv/bin/python applications/omega_biisotropic_sphere/generate_sphere_mesh.py \
  --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7_cap5_ppw8_capref32.json
```

The first generated mesh had 7,398 faces and 11,097 edges, corresponding to a
44,388-unknown dense system and a 29.36-GiB complex-double matrix before solver
workspace. The centroid mask selected 198 cap faces. Their combined area was
99.21 percent of the analytical 5-degree cap area, equivalent to a 4.980-degree
cap. This is substantially more faithful than uniform PPW 8, which selected
only 14 faces and produced 111.81 percent of the analytical cap area.

The follow-up configuration `configurations/budhu_10ghz_eps7_cap2p5_ppw5.json`
reduces the cap radius to 2.5 degrees, increases the SVWF grid to `145 x 288`
(approximately 1.25-degree spacing), and increases `n_max` to 20. It requests a separate
SIE mesh at 5 points per shortest material wavelength, leaving the original
mesh available. Pass this file to the same commands above; its results are
under `results/budhu_10ghz_eps7_cap2p5_ppw5/`.

For now, this configuration is used only with SVWF. Its 5-PPW SIE mesh has no
face centroid inside the 2.5-degree cap, so tensor preparation stops rather
than silently treating the sheet as unperturbed. A future SIE run will keep
the sphere at 5 PPW globally and add localized Gmsh refinement around the cap,
providing several cap faces without the memory cost of global refinement.

### Publication-focused validation plan

The oscillations currently visible near the perturbed sheet are **candidate
near-surface waves**, not yet a verified surface-wave result. The two solvers
have independent numerical limitations, so the validation should proceed in
the following order.

1. **Make SIE field evaluation reliable near the sheet.** The present field
   evaluator applies one fixed triangle quadrature rule at every observation
   distance. Add distance-aware near-singular integration, singularity
   extraction, and the correct one-sided limit when a field is requested on
   the surface. Until that is complete, evaluate at explicitly recorded
   offsets on both sides of the sheet rather than at `r = a`. Validate the
   evaluator by showing that the extinction error remains bounded and
   decreases under quadrature and mesh refinement.
2. **Separate SVWF modal convergence from collocation convergence.** First
   vary `n_max` on one fixed angular grid. Then hold `n_max` fixed and refine
   the angular grid. Report both the fitted-system residual and a GSTC
   residual on an independent validation grid, separating the cap, a narrow
   annulus around its edge, and the remainder of the sphere instead of
   reporting only one global norm. The 2.5-degree cap currently has 13
   collocation points and about four grid intervals across its diameter. That
   is sufficient to introduce a perturbation, but not to establish convergence
   of the sharp cap boundary. Any nonuniform or locally refined collocation
   grid must use corresponding surface-area weights; otherwise the cap is
   artificially over-weighted in the least-squares norm.
3. **Resolve the cap locally in the SIE mesh.** Retain the affordable global
   5-PPW sphere and add localized Gmsh refinement around the cap. Perform a
   short convergence sequence based on the number and size of faces inside
   the cap. This also reduces the staircase error introduced when a circular
   cap is represented by a face-centroid mask. Local cap refinement resolves
   the excitation, but it does not by itself guarantee propagation accuracy:
   if the measured tangential wavelength is shorter than the bulk wavelength,
   the mesh along the propagation path must also resolve that shorter scale.
4. **Compare physical diagnostics only after both numerical checks pass.** Use
   the same cap and phasor convention in both solvers and plot the
   baseline-subtracted field (perturbed minus unperturbed) at matching sample
   points. Angular complex-field traces at several offsets `a +/- delta`,
   together with normal decay and tangential Poynting flux, are needed to
   distinguish a surface wave from Gibbs ringing, mesh-scale oscillations, or
   field-quadrature error. Extract the dominant angular order or tangential
   wavenumber and compare it with the surface-mode spectrum predicted by the
   uniform omega sheet. The final publication comparison can then include
   SVWF, SIE, and COMSOL as three independent formulations.

Dense LU solves the discretized SIE system to its reported residual; it does
not make the continuous SIE result exact. Likewise, SVWF modes satisfy
Maxwell's equations in each homogeneous region, while the truncated expansion
satisfies the spatially varying GSTCs in a weighted least-squares sense. Their
different error mechanisms make agreement after convergence substantially
more informative than agreement in a single plot.

## Generate the SIE mesh

The SIE validation uses a Gmsh-generated triangular sphere exported as OBJ.
The mesh path and its points-per-wavelength setting belong to the shared case
configuration under `sie.mesh`; no mesh choice is compiled into the future SIE
driver. Relative mesh paths are interpreted from this application directory.

Install the optional mesh dependency once with

```bash
python -m pip install -e '.[mesh]'
```

Then generate the mesh described by the checked-in configuration:

```bash
python applications/omega_biisotropic_sphere/generate_sphere_mesh.py \
    --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json
```

This produces
`applications/omega_biisotropic_sphere/meshes/budhu_10ghz_eps7/sphere.obj`.
Generated meshes are ignored by Git. The target edge length is

```text
h = min(lambda_interior, lambda_exterior) / points_per_wavelength.
```

The default of four points per shortest wavelength is an initial coarse mesh
for the dense SIE validation, not a convergence claim. For a reproducible
denser trial, copy the configuration and change both `points_per_wavelength`
and the mesh `path`; `--ppw` and `--output` are available only as temporary
command-line overrides. After generation, the script reports the triangle/edge
counts and an estimate of the dense complex-double system matrix memory. If the C++
`mesh_obj_loader_tool` target has been built, the script also runs its existing
closed-surface topology checks; otherwise it explains how to enable that
check. The reusable C++ `ObjMeshLoader` remains the single mesh-loading path,
so a mesh factory is not needed at this stage.

## Prepare the SIE inputs

The normal preparation workflow is a single command:

```bash
python applications/omega_biisotropic_sphere/prepare_sie_inputs.py \
    --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json
```

It calls `generate_sphere_mesh.generate_mesh()`, validates the resulting OBJ
without changing its vertex order, face order, or winding, evaluates the
prescribed fields at the physical sphere radius using each planar face-centroid
direction, and extracts the physical `exp(+j omega t)` sheet parameters. It
then writes one 32-value complex64 tensor record per OBJ face: the row-major
4-by-4 fields-to-currents map followed by its full row-major inverse.

The default output directory is
`results/<case_id>/sie/<extraction_method>/` and contains only:

```text
face_tensors.bin
sie_inputs.json
```

The JSON manifest records the mesh hash and unchanged face ordering, phasor
convention, spherical local basis, binary layout, extracted parameter means
and spreads, and numerical residuals. To repeat only the validation and tensor
export while developing, reuse the already generated configured mesh with:

```bash
python applications/omega_biisotropic_sphere/prepare_sie_inputs.py \
    --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json \
    --use-existing-mesh
```

## Complete SIE validation

The SIE workflow uses the same checked-in configuration as the SVWF workflow.
Build its thin executable with:

```bash
cmake -S . -B build
cmake --build build --target omega_sphere_sie -j2
```

For the first complete field run, reuse the mesh and tensors prepared above:

```bash
python applications/omega_biisotropic_sphere/sie_dipole_impedance_matching.py \
    --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json \
    --reuse-inputs
```

Omit `--reuse-inputs` to run mesh generation, validation, and tensor extraction
before the solve. Add `--no-residual` for a faster exploratory run that skips
the post-solve residual reassembly. The optional `--svwf-run PATH` adds an
already sampled SVWF result to the comparison plots; it is not used by the SIE
solve. The two runs must use the same observation coordinates.

The C++ executable writes `currents.bin` and `solver_metrics.json`. The current
file is little-endian interleaved float64 and contains physical RWG
coefficients in `[J1, M1, J2, M2]` block order. The solver uses
`[J1, M1/eta0, J2, M2/eta0]` internally and removes the numerical scaling
before export.

For every ZX and radial observation point, the executable evaluates both
equivalent-current representations over the complete sampling domain:

```text
interior representation = incident dipole - L1(J1, M1)
exterior representation = L2(J2, M2)
```

The physical field is written by piecewise selection: the interior
representation for `r < a` and the exterior representation for `r > a`. The
plotter forms their algebraic sum only as a labeled extinction diagnostic. It
is not silently used as the physical field because any failed extinction would
contaminate the sum with a continuation evaluated in the wrong medium.

The resulting SIE run contains:

```text
results/budhu_10ghz_eps7/sie/least_squares/
    configuration.json
    run.json
    face_tensors.bin
    sie_inputs.json
    currents.bin
    solver_metrics.json
    fields/
        prescribed_zx.csv
        prescribed_radial.csv
        sie_interior_representation_zx.csv
        sie_exterior_representation_zx.csv
        sie_physical_zx.csv
        sie_interior_representation_radial.csv
        sie_exterior_representation_radial.csv
        sie_physical_radial.csv
    plots/
        fields_zx.png
        radial_cut.png
        real_ex_zx.png
        real_ex_radial_cut.png
        sie_representations_zx.png
        sie_representations_radial.png
```

The standard plots compare the piecewise physical SIE field with the prescribed
field and, when requested, the SVWF result. The two `sie_representations` plots
show the interior representation, exterior representation, and their
diagnostic sum so that an extinction failure remains visible.

## COMSOL validation

COMSOL automation lives in `comsol/`; generated models, fields, manifests, and
logs live under `results/<case_id>/comsol/<run_name>/` and are ignored by Git.
Install the optional Python interface with

```bash
python -m pip install -e '.[comsol]'
```

The first checkpoint is a small environment smoke test. It starts COMSOL,
queries the licensed products, creates and removes an empty model, and shuts
the server down without solving. Submit it from the repository root on
TinkerCliffs with

```bash
COMSOL_STAGE=smoke COMSOL_RUN_NAME=smoke \
    sbatch applications/omega_biisotropic_sphere/comsol/tinkercliffs.sbatch
```

The job uses eight CPU cores and 32 GB for at most 15 minutes. Its manifest and
full job log are written beneath
`results/budhu_10ghz_eps7/comsol/smoke/`. The driver refuses to start COMSOL
outside a Slurm allocation unless `--allow-outside-slurm` is supplied for a
workstation run.

The next stage validates a centered z-directed electric-current dipole in a
homogeneous free-space sphere. It samples the complex electric and magnetic
fields along a 45-degree radial cut and compares them with the analytical
dipole evaluator. Submit it with

```bash
COMSOL_STAGE=homogeneous COMSOL_RUN_NAME=homogeneous \
    sbatch applications/omega_biisotropic_sphere/comsol/tinkercliffs.sbatch
```

This first radiation model uses a spherical scattering boundary at
`1.8 lambda_0` and COMSOL's fine physics-controlled mesh (preset 4). These are baseline
settings for source validation, not convergence settings.

The homogeneous model's point source uses the named
`dipole_origin` selection: a box of half-width `1e-6 lambda_0` around the
origin, restricted to geometric points. The driver checks that exactly one
point is selected before meshing and solving, and records its ID in the log
and successful run manifest. Do not use all points for the dipole; that also
selects geometric vertices on the outer sphere. Earlier runs using that
selection are not valid single-dipole references.

After solving, the driver saves `model.mph`, extracts the complex fields into
`fields/comsol_radial.csv` and `fields/analytical_radial.csv`, and writes
`plots/homogeneous_radial.png`. The plot compares tangential field magnitudes;
the manifest's relative errors compare the full complex vectors. A completed
run means these steps succeeded, not that mesh convergence has been established.

The batch script defaults to the full `omega_sphere` stage, using a dielectric
sphere, GSTC surface currents, and an air block surrounded by PML layers:

```bash
sbatch applications/omega_biisotropic_sphere/comsol/tinkercliffs.sbatch
```

Unless `COMSOL_RUN_NAME` is supplied, each submission writes to
`results/budhu_10ghz_eps7/comsol/omega_sphere-<job_id>/` to preserve previous
results. The default mesh preset is 4, the sphere boundary maximum element size
is `lambda_0/30`, and the air block half-width is `2 lambda_0`. Override these
with `COMSOL_MESH_PRESET`, `COMSOL_BOUNDARY_DIVISOR`, and `COMSOL_AIR_EXTENT`.
Slurm resources can be overridden with `sbatch --mem=... --time=...`.

Successful sphere runs save `model.mph`, the computed and prescribed radial
and zx-plane CSV files under `fields/`, comparison images under `plots/`, and
errors in `run.json`. Check both Slurm's exit status and the manifest's `status`.
The PML selection must contain all 26 surrounding domains; the driver rejects
an incomplete selection before solving.

The mesh sequence is generated from the physics before adding the boundary
size override. Adding only a `Size` feature to a new mesh disables automatic
generation and leaves no operation to create elements; this caused the earlier
`Undefined post expression` failure during field extraction. Empty meshes now
fail before the study runs. New models use the built-in COMSOL geometry kernel
to avoid requiring CAD Import for these primitive geometries.

The MPh driver starts `mphserver`, which uses regular COMSOL and RF licenses
even when launched by Slurm. If the log reports license error `-4` (licensed
number of users already reached), release an idle interactive session or wait
for a seat before resubmitting. COMSOL's separate batch licenses apply to its
native batch workflow; see the [COMSOL license documentation](https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_installation.02.030.html).

### Current volumetric-shell checkpoint and next mesh study

The COMSOL 6.1 run
`omega_case19_projector_shell90_glob20_int35_6_1` is the current reproducible
checkpoint for the reciprocal omega-biisotropic volume model. It used

- a `0.01 lambda_0` shell thickness;
- a swept shell with five radial prism layers and tangential
  `hmax = lambda_0/90`;
- `hmax = lambda_0/35` in the dielectric interior;
- a global background `hmax = lambda_0/20`, superseded by the finer local
  controls above; and
- the tangential-projector embedding of the diagonal electric and magnetic
  shell response.

The mesh contained 4,643,559 elements: 4,017,699 tetrahedra and 625,860
prisms. The solve and field extraction completed in about 869 seconds on the
recorded 192-core COMSOL 6.1 server run. The errors that exclude the volume
shell plus one shell thickness on both sides were

| Comparison | Electric relative L2 error | Magnetic relative L2 error |
| --- | ---: | ---: |
| 45-degree radial cut | 0.130 | 0.125 |
| ZX plane | 0.125 | 0.112 |

The corresponding masks that exclude only the shell gave 0.211/0.250 on the
radial cut and 0.271/0.199 in the ZX plane. Interior/exterior electric errors
also fell from 0.147/0.408 to 0.122/0.131 when the one-thickness buffer was
applied in the ZX plane. This sensitivity, together with the plots, localizes
much of the remaining discrepancy near the thin-volume interfaces. It does
not establish monotonic convergence with tangential shell refinement. In
fact, the shell-only-excluded errors are larger than the recorded
`lambda_0/75` Case 18b values (0.117/0.100 on the radial cut and 0.171/0.107
in the ZX plane). Case 18b predates the buffered metrics and must be
reprocessed or rerun to determine whether this difference is confined to the
interface neighborhood or also occurs in the bulk field.

The next COMSOL mesh experiment should introduce concentric refinement zones
while leaving the physical media unchanged:

1. Add a small, parameterized source-refinement ball `0 < r < r_src` around
   the point dipole. This can improve the sampled near field away from the
   origin, but the ideal point-dipole field is singular at `r = 0`; ordinary
   refinement cannot make the field at the source finite. A rigorous source
   study should also consider singular-field subtraction or a regularized
   finite source, and all error norms must retain a documented source
   exclusion radius.
2. Retain a resolved dielectric-interior region from `r_src` to the inner GSTC
   radius. Its initial maximum size can remain `lambda_0/35`.
3. Keep the GSTC volume itself as a separate swept-prism shell. Its radial and
   tangential convergence studies should be varied independently.
4. Add a conforming exterior near-shell refinement band from the outer GSTC
   radius to `r = r_shell,out + w_near`, with a maximum element size comparable
   to the dielectric interior and a controlled transition to the global air
   mesh. Start by testing `w_near` rather than fixing it at `0.2 lambda_0`.
   With the present `0.95 lambda_0` air-box half-width, a `0.2 lambda_0` band
   reaches approximately `0.91 lambda_0` and leaves only `0.04 lambda_0`
   before the PML along the coordinate axes.
5. Use the coarser global size only beyond the near-shell band and before the
   PML. Internal partitions used for mesh control must retain identical
   material properties and a conforming mesh so that they do not introduce a
   physical interface.

Change one refinement control at a time and record element count, degrees of
freedom, memory, wall time, solver residual, and field errors. Diagnostic
buffers should also become a short, predetermined sequence--for example
`1, 2, 5, 10` shell thicknesses--rather than selecting one buffer after seeing
the result. Preserve the shell-only-excluded and one-thickness metrics as
well.

The eventual publication comparison should evaluate COMSOL, SIE, SVWF, and
the prescribed analytical field on identical coordinates, source exclusions,
and shell-buffer masks. This separates bulk cross-solver agreement from the
distinct near-source, thin-volume-interface, and SIE near-singular evaluation
errors.

### Spatially varying sheet-parameter contract

The nonuniform spherical-cap example already establishes most of the
application-level contract. `apply_cap_perturbation()` evaluates the cap mask
from the dot product of unit radial directions and scales all three reciprocal
omega parameters pointwise. SVWF samples this profile at its collocation
points. SIE samples it at OBJ face centroids, constructs a separate forward
and inverse 4-by-4 GSTC map for every face, and writes the resulting 32 complex
values per face to `face_tensors.bin`. The C++/CUDA solver already accepts
these per-face records. The `cap_perturbation` entry in the SIE run manifest
records the experiment; it is not the location where the material variation
is applied.

The COMSOL volumetric-shell implementation now compiles the shared spherical
cap into component variables scoped to the shell. For the sphere it defines

```text
n = (x, y, z)/sqrt(x^2 + y^2 + z^2)
c = unit vector toward the cap center
q = n dot c
s(q) = cap_scale when q >= cos(cap_angular_radius), otherwise 1

Yee_local = s(q) Yee_baseline
Zmm_local = s(q) Zmm_baseline
Xem_local = s(q) Xem_baseline
```

Using the direction `n`, rather than the raw radius, makes the prescribed
surface value constant along each radial line through the finite-thickness
shell. COMSOL will evaluate these coordinate-dependent coefficients at its
element integration points while assembling the frequency-domain matrix. They
are not Python callbacks executed during every linear-solver iteration, and a
coordinate-only profile keeps the problem linear.

This first extension reproduces the existing cap and prepares for more general
nonuniform **scalar reciprocal omega** profiles. Although the values vary over
the sphere, `Yee_local I_t` and `Zmm_local I_t` remain isotropic within each
local tangent plane. A general nonuniform anisotropic sheet requires four
local 2-by-2 tensor blocks, a documented tangential basis, and a pointwise
local-to-global transformation before the Cartesian COMSOL constitutive
tensors are assembled. Functions written directly in `(theta, phi)` must also
handle the polar coordinate singularity and the azimuthal branch cut; using
`(n_x, n_y, n_z)` as the profile inputs avoids those coordinate artifacts for
a sphere. A future polarization-conversion case--for example, a z-directed
dipole field inside transformed into an x- or y-directed dipole field
outside--belongs to this full-tensor path and will also need its prescribed
target fields supplied to the parameter-extraction stage.

The remaining implementation checkpoints are:

1. Generalize the shared evaluator to smooth analytic profiles and tabulated
   surface data. SIE may continue with a piecewise-constant per-face
   approximation initially; convergence then requires mesh refinement or
   evaluation at face quadrature points.
2. Extend the schema from the three scalar omega parameters to full local
   tangential tensor blocks, and test the common basis transformation
   independently before comparing solver fields.

A sharp cap edge is a discontinuous material coefficient. COMSOL must either
align/refine the shell mesh around that edge or use a controlled smoothing
width and study its convergence. The current SIE face-centroid mask is also a
piecewise-constant approximation of the circular edge, so matching the
mathematical profile does not remove the need for backend-specific spatial
resolution studies.

The initial COMSOL cap study used the established `lambda_0/75` tangential
shell mesh. First, a scale-one regression exercised the spatial-profile path
while retaining the uniform physical sheet:

```powershell
python applications\omega_biisotropic_sphere\comsol\volumetric_impedance_matching.py --stage omega_sphere --version 6.1 --config applications\omega_biisotropic_sphere\configurations\budhu_10ghz_eps7_cap5.json --run-name omega_cap5_scale1_regression_shell75_6_1 --air-extent 0.95 --shell-mesh swept --global-hmax-divisor 20 --interior-hmax-divisor 35 --shell-tangential-divisor 75 --shell-radial-layers 5 --tau-thickness 0.01 --cap-scale 1 --linear-solver mumps --discretization linear --radial-theta 45 --radial-phi 0 --radial-points 501 --radial-max-radii 1.9 --zx-grid-size 300 --zx-span-radii 1.3 --allow-outside-slurm --cores 192
```

Then remove the override to run the configured 5-degree cap with scale 0.1:

```powershell
python applications\omega_biisotropic_sphere\comsol\volumetric_impedance_matching.py --stage omega_sphere --version 6.1 --config applications\omega_biisotropic_sphere\configurations\budhu_10ghz_eps7_cap5.json --run-name omega_cap5_scale0p1_shell75_6_1 --air-extent 0.95 --shell-mesh swept --global-hmax-divisor 20 --interior-hmax-divisor 35 --shell-tangential-divisor 75 --shell-radial-layers 5 --tau-thickness 0.01 --linear-solver mumps --discretization linear --radial-theta 45 --radial-phi 0 --radial-points 501 --radial-max-radii 1.9 --zx-grid-size 300 --zx-span-radii 1.3 --allow-outside-slurm --cores 192
```

For the active cap there is no closed-form cap solution in this workflow. Its
plots therefore label the prescribed uniform-sheet field as the *unperturbed
reference*, and its manifest reports `relative_l2_change_*` rather than
mislabeling those differences as numerical errors.

A subsequent `lambda_0/90` cap run increased the shell mesh from 434,360 to
625,860 prisms and visually reduced some near-sheet fluctuation, although the
changes from the unperturbed reference were not monotonic across all field
cuts. Those `relative_l2_change_*` values are not cap-solution errors. The
first radial-band experiment nevertheless holds the shell at `lambda_0/90` so
that only the exterior volume mesh changes relative to that completed run.

The first radial-refinement experiment adds a conforming air band immediately
outside the volumetric shell. It is a mesh partition only: its material and
wave equation are identical to the surrounding air. Start with a width of
`5*tau = 0.05 lambda_0` and `hmax = lambda_0/35`, while retaining the
`lambda_0/90` cap-shell mesh:

```powershell
python applications\omega_biisotropic_sphere\comsol\volumetric_impedance_matching.py --stage omega_sphere --version 6.1 --config applications\omega_biisotropic_sphere\configurations\budhu_10ghz_eps7_cap5.json --run-name omega_cap5_shell90_outerband5tau_h35_6_1 --air-extent 0.95 --shell-mesh swept --global-hmax-divisor 20 --interior-hmax-divisor 35 --outer-shell-band-width 0.05 --outer-shell-band-hmax-divisor 35 --shell-tangential-divisor 90 --shell-radial-layers 5 --tau-thickness 0.01 --linear-solver mumps --discretization linear --radial-theta 45 --radial-phi 0 --radial-points 501 --radial-max-radii 1.9 --zx-grid-size 300 --zx-span-radii 1.3 --allow-outside-slurm --cores 192
```

Both band options must be supplied together. Omitting both preserves the
previous geometry exactly. The band must end strictly inside the non-PML air
box; for this command its outer radius is `0.76 lambda_0`, leaving
`0.19 lambda_0` before the PML along each coordinate axis.

The whole-shell `lambda_0/120` experiment also refines the cap, but it does
not force the discontinuous cap boundary to follow element edges. The next
experiment therefore targets the cap rim rather than increasing the uniform
shell cost again. Supplying `--cap-edge-hmax-divisor N` constructs a cone with
its apex at the sphere center and its axis and semiangle taken from
`cap_perturbation`. Its intersection with both spherical shell faces is the
exact analytical cap circle. COMSOL splits the swept shell at that cone, and
the source-face part of the rim receives `hmax = lambda_0/N`; the swept mesh
then carries the conforming partition through the shell thickness.

Start with the established `lambda_0/90` background shell, the existing
five-tau outer band, and a localized `lambda_0/240` cap rim:

```powershell
python applications\omega_biisotropic_sphere\comsol\volumetric_impedance_matching.py --stage omega_sphere --version 6.1 --config applications\omega_biisotropic_sphere\configurations\budhu_10ghz_eps7_cap5.json --run-name omega_cap5_shell90_caprim240_outerband5tau_h35_6_1 --air-extent 0.95 --shell-mesh swept --global-hmax-divisor 20 --interior-hmax-divisor 35 --outer-shell-band-width 0.05 --outer-shell-band-hmax-divisor 35 --cap-edge-hmax-divisor 240 --shell-tangential-divisor 90 --shell-radial-layers 5 --tau-thickness 0.01 --linear-solver mumps --discretization linear --radial-theta 45 --radial-phi 0 --radial-points 501 --radial-max-radii 1.9 --zx-grid-size 300 --zx-span-radii 1.3 --allow-outside-slurm --cores 192
```

This option is deliberately opt-in and requires both a cap configuration and
the swept-shell strategy. Omitting it leaves the earlier 8-octant shell
topology and all previous `/75`, `/90`, `/120`, and outer-band commands
available unchanged. With it enabled for the present +x-centered cap, the
shell contains 4 cap sectors plus the original 8 background sectors. Entity
numbers are discovered from propagated geometry selections and adjacency;
none of the cap domains, faces, or rim edges are hardcoded.

The aligned-rim run made the candidate surface-wave pattern clearer without
a comparable increase in shell elements, so the next experiment targets the
volume fields immediately adjacent to the sheet. The optional inner band
partitions the dielectric into an unchanged inner bulk and a concentric
near-sheet annulus; both retain `epsilon_r = 7`. Together with the existing
air band, use a symmetric five-tau neighborhood with
`hmax = lambda_0/60`, while retaining the `/90` shell and `/240` cap rim:

```powershell
python applications\omega_biisotropic_sphere\comsol\volumetric_impedance_matching.py --stage omega_sphere --version 6.1 --config applications\omega_biisotropic_sphere\configurations\budhu_10ghz_eps7_cap5.json --run-name omega_cap5_shell90_caprim240_symmetricbands5tau_h60_6_1 --air-extent 0.95 --shell-mesh swept --global-hmax-divisor 20 --interior-hmax-divisor 35 --inner-shell-band-width 0.05 --inner-shell-band-hmax-divisor 60 --outer-shell-band-width 0.05 --outer-shell-band-hmax-divisor 60 --cap-edge-hmax-divisor 240 --shell-tangential-divisor 90 --shell-radial-layers 5 --tau-thickness 0.01 --linear-solver mumps --discretization linear --radial-theta 45 --radial-phi 0 --radial-points 501 --radial-max-radii 1.9 --zx-grid-size 300 --zx-span-radii 1.3 --allow-outside-slurm --cores 192
```

Both inner-band options must be supplied together. Omitting them preserves
the previous single-domain dielectric interior. The `/35` interior control
still applies to the rest of the dielectric; the more restrictive `/60`
control supersedes it only in the inner annulus. The two radial partitions
are mesh controls, not additional physical interfaces: material properties
and governing equations remain continuous across their outer boundaries.

The highest-resolution completed cap case subsequently used `/120` over the
whole swept shell and widened both `/60` near-sheet bands to `0.1 lambda_0`.
To isolate the axial contamination from the singular COMSOL point source,
repeat that case with one additional conforming dielectric ball of radius
`0.15 lambda_0` and `hmax = lambda_0/120`:

```powershell
python applications\omega_biisotropic_sphere\comsol\volumetric_impedance_matching.py --stage omega_sphere --version 6.1 --config applications\omega_biisotropic_sphere\configurations\budhu_10ghz_eps7_cap5.json --run-name omega_cap5_shell120_caprim240_symmetricbands10tau_h60_sourceball0p15_h120_6_1 --air-extent 0.95 --shell-mesh swept --global-hmax-divisor 20 --interior-hmax-divisor 35 --source-ball-radius 0.15 --source-ball-hmax-divisor 120 --inner-shell-band-width 0.1 --inner-shell-band-hmax-divisor 60 --outer-shell-band-width 0.1 --outer-shell-band-hmax-divisor 60 --cap-edge-hmax-divisor 240 --shell-tangential-divisor 120 --shell-radial-layers 5 --tau-thickness 0.01 --linear-solver mumps --discretization linear --radial-theta 45 --radial-phi 0 --radial-points 501 --radial-max-radii 1.9 --zx-grid-size 300 --zx-span-radii 1.3 --allow-outside-slurm --cores 192
```

Both source-ball options must be supplied together. Omitting them reproduces
the previous geometry. The ball and the surrounding core both retain
`epsilon_r = 7`; its internal spherical boundary is solely a mesh-control
partition. This refinement does not regularize the ideal point dipole or make
the field at `r = 0` finite. Its primary diagnostic is whether the spurious
transverse field along the dipole axis decreases at fixed nonzero radii.

A local COMSOL 6.1 mesh-only check completed successfully with 20 cores in
84 seconds. It produced 10,266,018 elements: 9,153,288 tetrahedra and
1,112,730 swept prisms. Relative to the preceding Windows mesh, the total
element count increased by 4.76% and the tetrahedron count by 5.38%. COMSOL
resolved the source ball as one dielectric domain and verified
`222,546 source triangles * 5 layers = 1,112,730 prisms`. The small prism-count
difference from the preceding Windows run reflects a complete remesh on a
different platform; the swept-shell consistency check still passed.

## Deferred improvements

- Replace the sphere-specific `(theta_hat, phi_hat)` tensor contract with a
  deterministic per-triangle tangential basis. The Python extraction/export
  and CUDA tensor rotation must adopt the same basis definition together so
  that general curved surfaces can be supported without changing tensor
  components implicitly.
- Revisit the CUDA source organization after the reusable solver interface is
  stable. Header-based `.cuh` implementations are convenient for templates,
  inline device utilities, and the current kernels, but compiled non-template
  orchestration may be moved into `.cu` sources to reduce implementation
  exposure and repeated compilation. This cleanup is not required for the
  present sphere validation.
- Add singularity extraction or another validated near-singular treatment to
  the field-evaluation operator before sampling on or very close to the mesh.
  The present sampling points do not lie exactly on the nominal spherical
  boundary, but points close to it can still expose ordinary-quadrature error.
- Generalize the application-level field composition for an impressed source
  in the exterior region. The reusable current-field evaluator is
  medium-indexed, but this first example deliberately fixes the centered dipole
  in region 1.

## Intended dependency direction

```text
applications/omega_biisotropic_sphere
    -> sie_mom.excitation
    -> sie_mom.gstc.omega_biisotropic
    -> COMSOL API through the optional MPh interface
    -> sie_mom::solvers::solve_gstc_dense
    -> sie_mom::solvers::evaluate_surface_integral_fields
    -> spherical_expansion.solver
```
