# NGSolve spherical GSTC experiment

Independent, frequency-domain Maxwell FEM for the existing **10 GHz,
radius 0.7 lambda0, interior epsilon_r=7** dual-polarization sphere. It reads the
saved configuration and full-precision `face_tensors.bin`; it does not rerun
extraction or modify the SIE/COMSOL workflows. Both centered electric and magnetic
dipole cases are supported, with one matrix factorization shared across their RHSs.

This is an experimental validator. Completion and a small linear residual do
**not** establish spatial accuracy. Check `VALIDATION.md` for measured results
and outstanding convergence work.

The reusable FEM implementation now lives in
[`src/sie_mom/validation/ngsolve`](../../../../src/sie_mom/validation/ngsolve/).
The sphere runner is
[`applications.anisotropic_gstc_sphere.validate_ngsolve`](../../validate_ngsolve.py).
The local `model.py` and `run.py` preserve the original commands and imports;
they no longer duplicate the FEM formulation. Existing plots and checkpoint
utilities remain available here during the transition.

NGSolve is an optional project extra. To use the root virtual environment,
install it there with `python -m pip install -e ".[ngsolve]"`, then run
`python -m applications.anisotropic_gstc_sphere.validate_ngsolve`.
A separate environment is optional; none is created by the runners. The
existing experiment environment can still run these modules with
`PYTHONPATH=src` from the repository root on Linux, without installing packages.

## Install and run

Run from the repository root. The isolated environment is ignored by Git.
No COMSOL license, CUDA, or compiled SIE executable is required.

Linux:

```bash
python3 -m venv applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python -m pip install \
  -r applications/anisotropic_gstc_sphere/experiments/ngsolve/requirements.txt
OPENBLAS_NUM_THREADS=1 applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/run.py
```

Windows PowerShell, separate from the COMSOL environment:

```powershell
py -3.12 -m venv applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv
$python = ".\applications\anisotropic_gstc_sphere\experiments\ngsolve\.venv\Scripts\python.exe"
& $python -m pip install -r applications/anisotropic_gstc_sphere/experiments/ngsolve/requirements.txt
$env:OPENBLAS_NUM_THREADS = "1"
& $python applications/anisotropic_gstc_sphere/experiments/ngsolve/run.py
```

NGSolve is pinned to the tested version. Windows installation/execution still
needs validation. `--inverse auto` selects PARDISO if available through either
the compiled build or the runtime MKL loader, otherwise UMFPACK. An unavailable
or failing backend produces an error; it does not quietly change the solver.
For the tested runtime PARDISO wrapper, `--threads` sets both NGSolve task threads
and the local MKL PARDISO thread limit. UMFPACK/BLAS threading differs.

Useful options:

```text
--case-id electric_incidence       # default: all configured cases
--ppw 3 --order 3                  # default: tested initial benchmark
--ppw 2 --order 2                  # faster smoke run; less accurate
--ppw 3 --order 2                  # lower-order comparison
--ppw 4 --order 2                  # independent mesh refinement
--ppw 3 --interior-ppw 4 --order 3 # refine core/sheet without global refinement
--air-lambda .95                   # radius of the physical exterior region
--pml-lambda .25 --pml-strength 3   # radial PML thickness and stretch
--pml-lambda .5 --pml-strength 1.5 # useful independent truncation check
--threads 4 --inverse auto
--output-dir /path/to/new/run      # refuses to overwrite nonempty directories
--vtk                             # optional ParaView field output
```

The default inputs are:

```text
applications/anisotropic_gstc_sphere/results/dipole_pol_rot/dual_pol_eps7/configuration.json
applications/anisotropic_gstc_sphere/results/dipole_pol_rot/dual_pol_eps7/face_tensors.bin
```

These generated files must already exist. Override them with `--config` and
`--tensors`. This initial example deliberately rejects nonconstant local tensors,
non-spherical tensor bases, non-dipole fields, off-center sources, exterior
excitations, and singular/ill-conditioned magnetic impedance blocks. The saved
dual-polarization case meets those requirements. Constancy refers to the local
theta/phi coefficients: the Cartesian tensor rotates around the sphere.

## Formulation

Time dependence is `exp(+i omega t)`, with outgoing phase `exp(-i k r)`.
The finite-element coordinates are `X=k0*x`, and fields are scaled as
`e=E/k0^3`, `h=eta0*H/k0^3`. This common amplitude scaling preserves the
repository's effective dipole moment, complex phase, and electric/magnetic duality.

The saved physical surface matrix `[Y, chi; gamma, Z]` is converted to
`L=[eta0*Y, chi; gamma, Z/eta0]`. For the outward normal and right-handed local
theta/phi frame, let `R=[[0,-1],[1,0]]`. The original GSTCs are

```text
R (h+ - h-) = L_ee e_av + L_eh h_av
-R (e+ - e-) = L_he e_av + L_hh h_av
```

We solve the second equation locally for `h_av`, then the first for `h+ - h-`.
This produces a 4x4 admittance `B`, with `[h-; h+]=B[e-; e+]`. The only local
inverse is of `L_hh`. Its condition number is recorded; singular cases require
a different formulation rather than a numerical regularization.

Two separate `HCurl` spaces, restricted to the core and exterior, duplicate
the tangential electric unknowns at the sheet. A single continuous `HCurl`
space would suppress the magnetic-current jump. On a curved mesh, the spherical
frame is projected onto the geometric tangent plane and oriented by the outward
core normal. The geometric approximation is controlled by `--curve-order`.

With the 3x2 frame matrix `T=[t_theta, t_phi]`, local components are `T^T E`.
Each original 2x2 block would lift to Cartesian coordinates as `T A T^T`.
The implementation performs that mapping implicitly by projecting trial and
test traces into the local frame; it does not assemble four explicit 3x3
tensors. Both magnetoelectric blocks are retained. The name `sheet_admittance`
refers to the derived two-sided 4x4 field map `B`, not an electric-only sheet.

The loader reads all 16 complex entries of the original four-block surface map
per OBJ face. The other 16 entries, containing a stored inverse, are unused.
It verifies that these local maps are constant within tolerance, then uses one
representative map. There is no spatial interpolation in this sphere prototype.
The frame still varies over the surface, so the corresponding Cartesian tensor
is spatially varying even when its local coefficients are constant.

Each region contributes

```text
integral [mu_r^-1 curl(e) . curl(v) - epsilon_r e . v] dX
```

and the interface contributes

```text
-i integral [(R h-) . v- - (R h+) . v+] dS
```

after substituting `B`. The implementation uses bilinear products with real
finite-element basis functions; no conjugation of the unknown is introduced.
There is no global null-space construction or multiplier elimination step.

The interior unknown is the **regular scattered correction**. The total
interior field is the analytic dipole plus that correction. The incident fields
enter only the sheet RHS, including their magnetic trace. This avoids discretizing
the singular point source or changing its physical amplitude. The exterior
unknown is its total field. Magnetic fields used in diagnostics come from
`h=i curl(e)/mu_r`, plus the analytic incident field inside.

## Geometry and numerical controls

A new curved tetrahedral mesh follows the ideal sphere; the OBJ identifies
the saved face records, but is not used as the FEM volume mesh. The physical
frequency, radius, media, tensors, and excitation moments come from the saved
configuration. `ppw` is an element-size control using each region's wavelength,
not a promise of accuracy or a direct equivalent of COMSOL's mesh divisors.

The default outer air radius and PML thickness match the reported COMSOL lengths.
**The PML is spherical, whereas the COMSOL setup uses a surrounding box.** A
negative imaginary radial stretch absorbs this time convention. Tangential E
is set to zero at the far end of the PML. PML mesh resolution, strength, thickness,
and air extent all need convergence checks. Changing these numerical controls
does not change the physical sphere.

## Non-spherical interfaces

The shared backend now also supports prepared non-spherical surfaces through
[`applications.gstc_surface.validate_ngsolve`](../../../gstc_surface/validate_ngsolve.py).
See the [general-surface workflow and measured ogive results](../../../gstc_surface/README.md#ngsolve-with-the-same-sie-interface).

That runner imports the original SIE triangles as an unchanged planar interface
and generates tetrahedra in the core, air and PML around it. Each face retains
its original physical tensor and intrinsic frame, with exact correspondence
checked independently of FEM numbering. The ogive checkpoint exercised spatially
varying tensors at field orders 1–3. It establishes a practical comparison
workflow; the recorded near-sheet discrepancies and convergence limits remain.
No nonmatching-mesh tensor interpolation is implemented.

The sphere adapter deliberately retains its curved analytic geometry,
spherical tensor basis, constant local tensors and centered-source restrictions
for regression compatibility. Those restrictions do not apply to the shared
weak form. Both paths require invertible magnetic blocks for the current
E-only formulation. They include the tangential-polarization GSTCs used by SIE;
independent normal surface polarizations would require additional terms.

## Output and interpretation

Default output is a new timestamped directory under
`results/dipole_pol_rot/dual_pol_eps7/experiments/ngsolve/`.

- `metrics.json`: input hashes, package versions, physical parameters, mesh DOFs,
  separate mesh/assembly/factorization/RHS/backsolve timings, actual backend,
  linear residuals, both GSTC probe residuals, power flux and target comparisons.
- `configuration.json`, `sheet.npz`: configuration snapshot, normalized sheet
  matrix and locally derived admittance.
- `<case>/radial_fields.npz`: complex Cartesian total E/H in the repository's
  original normalization, positions and interior mask. Includes prescribed
  design fields when the case has a unique matching transformation. The radial
  line uses the configured angles and sampling count, but stops before the PML
  and omits samples too close to the interface.
- `<case>/zx_fields.npz`: complex Cartesian total E/H on the configured square
  ZX grid, with a validity mask. Its half-span is `min(zx_span_radii*a, air_radius)`,
  matching the COMSOL extraction rule. Points outside the physical air sphere,
  within `0.001a` of the sheet, and at the source singularity are masked as NaN.
- `mesh.vol.gz`, `<case>/solution.sol`: curved mesh and both FE solution
  components for later sampling without assembly or another solve. Added after
  the initial validation runs; older runs do not contain these checkpoints.
- `<case>/far_field_phi0.npz`: complex phi=0 far-field cut from a Huygens integral
  at `r=a+0.1 lambda0` (or a smaller radius if necessary). With dimensionless
  distance `R=k0*r`, `e ~ E_far_normalized exp(-i*k_rel*R)/R`; multiply the saved
  amplitude by `k0^2` for the coefficient of `exp(-i*k*r)/r` in original E units.
- Optional VTK: real/imaginary interior **correction** and exterior total E,
  in the dimensionless field normalization. The analytic incident field is not
  included in the interior VTK arrays.

Plot a saved run without another solve:

```bash
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/plot.py /path/to/run
```

This creates a new `plots/` directory with radial magnitude comparisons and
far-field polarization cuts. It refuses to overwrite an existing plot directory.
For readability, radial plots floor magnitudes at 1e-10 of each panel's peak.
The NPZ files retain the unmodified complex fields for phase-sensitive comparisons.

To reproduce the shared SIE/COMSOL **1D spherical-component plots**, including
real and imaginary parts of all six components and the prescribed fields:

```bash
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/plot_radial.py \
  /path/to/run --compare-target
```

This creates `plots_radial_spherical/<case>/` with six figures per excitation:
vector magnitudes, the three E/H component-magnitude pairs, and separate
three-by-two real/imaginary component grids for E and H. Magnitudes default to
dB re 1 SI unit (`--scale linear` or `--scale log` also work); real/imaginary
parts remain linear in V/m and A/m. These line plots use no percentiles or
amplitude rescaling. They preserve the saved radial range, split curves at the
sheet, and mark `r/a=1`. No new solve or environment activation is needed.
Use `--output-dir` to select a new directory; existing plots are preserved.

To **overlay SIE, NGSolve and prescribed radial fields**, with dedicated
suppression panels:

```bash
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/plot_radial_sie.py \
  /path/to/checkpointed/run --floor-db -100
```

The default SIE input is `results/dipole_pol_rot/dual_pol_eps7`; change it with
`--sie-dir`. This reads `cases/<case>/fields/sie_physical_radial.csv`, checks
matching configuration/tensor hashes, and evaluates the saved NGSolve mesh and
solution **at the SIE coordinates**. No interpolation, assembly, factorization,
solve, or fitted amplitude/phase scaling is used. Points in the PML and within
`0.001a` of the sheet are excluded. The physical air limit includes the same
`0.02 lambda0` buffer as the original radial sampler.

New figures go to `plots_radial_sie/<case>/`: vector and component magnitudes,
real/imaginary component grids, and `radial_suppression_sie_ngsolve.png`.
Magnitude overlays use one prescribed exterior vector-peak reference for E and
one for H, shared by all three curves and all components on the plotted ray.
Their lower display limit defaults to -100 dB. Exact-zero targets are -infinity
dB mathematically; roundoff and smaller values are drawn at the labeled floor.
The source field can exceed 0 dB and remains visible.

Suppression panels instead show
`20 log10(|F_unwanted(r)| / ||F_prescribed(r)||)`, separately for E/H and
interior/exterior. A prescribed component is classified as zero if its ratio
to the local prescribed vector norm stays below `1e-12` throughout that region.
The nonzero vector denominator avoids dividing by a zero prescribed component.
More negative levels mean less leakage. This is a radial field-amplitude
comparison, not integrated power conversion efficiency.

`comparison_metrics.json` records per-component and combined unwanted-vector
RMS/peak leakage, complex-vector errors, exact sample ranges, references and
source hashes. Statistics use unmodified ratios, never the plotting floor.
The default common scoring interval is `0.2 <= r/a <= 1.3`, excluding
`|r/a-1| < 0.05`; near-sheet samples remain visible in the plots. All compared
runs use the same scoring points, so enlarging the air does not change the
score's range. `--metrics-range` changes the interval. The NPZ alongside each
case preserves all three complex vectors at the common coordinates.
## Matching 2D ZX maps

For surface/volume element counts and unknowns across SIE, NGSolve and the
reported COMSOL run, see [MESH_COMPARISON.md](MESH_COMPARISON.md).

For a **quick computed/prescribed comparison**, prefer a common 60 dB window
instead of percentiles on weak components:

```bash
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/plot_quick_compare.py \
  /path/to/run --output-dir /path/to/new/quick_comparison
```

This plots `20*log10(|component|/reference)` over `[-60,0] dB`. The reference
is the prescribed exterior vector-magnitude peak, separately for E and H.
Each reference is shared across computed/prescribed panels and all components
of that field. Computed fields are never independently normalized to match the
target. Values above 0 dB are saturated (including the strong interior source);
very small and exactly zero values receive the bottom color rather than
stretching the colorbar to numerical roundoff. Spatially excluded samples stay
blank. Raw complex fields are unchanged. `--dynamic-range-db` changes the span.
The exact numerical references and limits are recorded in `plot_settings.json`.

These fixed bounds follow the publication README's approach, but use an
exterior reference appropriate for comparisons rather than its origin-peak
reference and case-specific limits. Original 2–98% percentile plots are retained.

`plot_zx.py` calls the existing renderer in
`applications/anisotropic_gstc_sphere/plot_fields.py`. It uses the same spherical
or Cartesian components, `x/a` and `z/a` axes, sheet outline, color maps,
linear/log/dB scales, percentile limits, and optional prescribed comparisons.
It reads the NGSolve NPZ samples directly; it does not pretend the run is an SIE
or COMSOL manifest. `plot_far_field.py` is the application's SIE-current-based
far-field workflow; it is not needed for near-field ZX maps.

From the repository root, substitute the path to a new run containing
`<case>/zx_fields.npz`:

```bash
NGSOLVE_RUN=/path/to/ngsolve/run
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/plot_zx.py "$NGSOLVE_RUN"
```

This creates `plots_zx/<case>/` with vector magnitudes and each spherical
component pair, using the shared plotter's default percentile range (2–98%)
and dB display without amplitude normalization. Useful variants:

```bash
# Computed and prescribed fields side by side, sharing color limits per quantity.
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/plot_zx.py "$NGSOLVE_RUN" \
  --compare-target --output-dir "$NGSOLVE_RUN/plots_zx_vs_prescribed"

# Real Cartesian components; imag and phase are also supported.
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/plot_zx.py "$NGSOLVE_RUN" \
  --basis cartesian --part real --scale linear \
  --output-dir "$NGSOLVE_RUN/plots_zx_cartesian_real"
```

Unlike the COMSOL box, the NGSolve physical air region is spherical. The square
grid's corners therefore have no physical air solution and remain blank. The
PML-transformed fields are excluded. The narrow sheet exclusion avoids selecting
an ambiguous trace on the approximate curved interface. Masked values are never
replaced with zero or extrapolated. Both computed and prescribed panels use the
same sampling mask.

Increase sampling density from a checkpoint without repeating the solve:

```bash
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/sample_zx.py "$NGSOLVE_RUN" \
  --grid-size 400 --output-dir "$NGSOLVE_RUN/zx_400"
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/plot_zx.py "$NGSOLVE_RUN" \
  --data-dir "$NGSOLVE_RUN/zx_400" --output-dir "$NGSOLVE_RUN/plots_zx_400"
```

This increases plot sampling density, not FEM resolution. Checkpoint loading
requires the same NGSolve version. Output and plot directories must be new or
empty. The original `validation_20261103/order3_pml025` had no checkpoints or
ZX arrays, so its 2D maps were generated by repeating the same numerical setup
in `order3_pml025_2d`. The new plot metadata identifies that source run.

GSTC diagnostics use independently evaluated E and volume-curl H at radial
offsets `0.02a`, `0.01a`, and `0.005a`. These are **finite-offset probes**, not
exact boundary traces. They include offset, geometry, and discretization error;
they are not computed by substituting the sheet law back into itself. Compare
all offsets under mesh/order refinement. The power flux uses normalized fields
and dimensionless surface area, and measures the difference between closed
surfaces inside and outside the sheet. For this lossless case, improvement
toward equal outward flux is a useful independent diagnostic.

Target discrepancies are not automatically FEM errors: in general a synthesized
least-squares sheet may not reproduce its prescribed fields exactly. A small
radial-line error also says little about unmeasured angles or near-sheet H.
For validation, compare complex fields, far-field polarization and amplitude,
power, both jump residuals, and refinement sensitivity together.

## Optional multithreaded PARDISO experiment

The Linux test successfully enabled MKL PARDISO at runtime, without compiling
NGSolve or writing C++. This does not establish that every platform/wheel or
cluster module configuration will work. Install the optional runtime into the
experiment environment:

```bash
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python -m pip install \
  -r applications/anisotropic_gstc_sphere/experiments/ngsolve/requirements-parallel.txt
```

This does not replace NumPy's OpenBLAS backend. Use explicit thread limits and
record the backend. The tested larger-domain command, from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_DYNAMIC=FALSE \
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/run.py \
  --inverse pardiso --pardiso-ordering metis --threads 8 \
  --ppw 3 --interior-ppw 4 --order 3 \
  --air-lambda 1.25 --pml-lambda .5 --pml-strength 1.5
```

Omit `--interior-ppw` to recover the original core mesh-size setting. This override
also refines the sheet and consequently influences adjacent exterior elements.
All physical parameters and synthesized tensors remain those in the saved input.

`--pardiso-ordering` selects `minimum_degree`, `metis` (runner default), or
`parallel_metis`. Ordering can affect fill, time, memory and numerical behavior;
keep it fixed when measuring thread scaling. To repeat a fixed-matrix benchmark
using an existing checkpoint:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_DYNAMIC=FALSE \
applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  applications/anisotropic_gstc_sphere/experiments/ngsolve/benchmark_threads.py \
  "$NGSOLVE_RUN" --threads 1 4 8 16 --ordering metis \
  --output-dir /path/to/new/thread_benchmark
```

The script reassembles once on the saved curved mesh, restores its PML transform,
then reuses that exact matrix/RHS for every thread count. It records a matrix
fingerprint, wall and process CPU times, linear residuals and coefficient
differences from the saved solution. Small differences from the original solve
can include curved-mesh serialization error. Avoid concurrent solves during
timing. CPU seconds divided by wall seconds indicate actual parallel CPU work;
a configured thread limit alone does not establish utilization.

`compare_common_points.py` evaluates checkpointed refinements on the same baseline
radial coordinates, so increasing the air extent does not change the comparison
points. Interior comparisons omit `r<0.2a` to reduce domination by the source
singularity. Use its `--help` for arguments.

See `PARALLEL_VALIDATION.md` for measured scaling and refinement results.

## Checks

```bash
OPENBLAS_NUM_THREADS=1 applications/anisotropic_gstc_sphere/experiments/ngsolve/.venv/bin/python \
  -m pytest applications/anisotropic_gstc_sphere/experiments/ngsolve/test_model.py -q -s
```

Tests check both original GSTCs for arbitrary complex traces, reject singular Z,
compare both dipole types with the existing field routines, verify both Maxwell
curl equations, verify complex Huygens far-field amplitude/polarization, and
check convergence of a manufactured two-sided spherical Maxwell solution.
The manufactured test uses a smaller electrical size and an exact outer
boundary value, isolating the volume/interface formulation from PML error.
It is not a substitute for convergence of the 10 GHz benchmark.

References: [NGSolve installation](https://ngsolve.org/installation.html),
[subdomain spaces](https://docu.ngsolve.org/latest/i-tutorials/unit-1.5-subdomains/subdomains.html),
[H(curl) spaces](https://docu.ngsolve.org/latest/i-tutorials/unit-2.3-hcurlhdiv/hcurlhdiv.html),
[PML transformations](https://docu.ngsolve.org/v6.2.2104/i-tutorials/unit-1.7-helmholtz/pml.html).
