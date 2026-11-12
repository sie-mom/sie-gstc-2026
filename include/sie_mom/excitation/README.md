# Shared excitation

The application prepares one source and passes it to both the solver and
incident-field evaluation:

```cpp
#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/solvers/pmchwt_dielectric_solver.hpp"

sie_mom::solvers::PmchwtOptions options;
options.frequency_hz = 1e9;
options.interior = {2.5, 1.0};
options.exterior = {1.0, 1.0};

sie_mom::excitation::DipoleExcitation dipole;
dipole.effective_moment = make_vec3(0, 0, 1);
dipole.position = make_vec3(0, 0, 0);
dipole.region = sie_mom::excitation::SourceRegion::interior;

const auto source = sie_mom::excitation::prepare_source(
    dipole, options.frequency_hz, options.interior, options.exterior,
    options.vacuum_permittivity, options.vacuum_permeability
);
const auto solution = sie_mom::solvers::solve_pmchwt_dense(mesh, options, source);
const auto incident = sie_mom::excitation::evaluate_incident_fields(points, source);
```

Use `PlaneWaveExcitation` instead of `DipoleExcitation` to change source type.
Plane waves specify a real electric-amplitude vector and a propagation direction;
preparation normalizes the direction and checks that the polarization is transverse.
The default plane-wave region is exterior; the default dipole region is interior.

## Where the logic lives

- `src/sie_mom/excitation/fields.py`: analytical Python dipole and plane-wave
  evaluators, exposed through `sie_mom.excitation`. See the
  [Python excitation README](../../../src/sie_mom/excitation/README.md) for
  component ordering and the preserved spatial-phase conventions.
- `excitation.hpp`: source settings, the prepared source, and the public API.
- `source_functors.cuh`: existing electric/magnetic dipole and plane-wave formulas,
  retaining their `sie_mom::cuda` namespace and `exp(+j omega t)` convention.
- `src/sie_mom/excitation/excitation.cu`: validation, binding the source region's
  medium and frequency, and GPU incident-field evaluation.
- `sie_mom/fields.hpp`: material and field-sample types shared with surface-current
  evaluation; these types do not belong to either solver.

The prepared source owns a copy of its settings. Later edits to the original
settings do not affect it. Both solvers reject a prepared source if its frequency,
source-medium parameters, or vacuum constants differ from their options.

The host visitor dispatches to a concrete CUDA functor. To add a source, add its
settings to `SourceSpec`, its functor to `PreparedExcitation::Source`, and its
validation/construction overload to `make_functor` in `excitation.cu`. The solver
and incident-field launch code need no source-specific changes.

## Regional fields

`evaluate_incident_fields` evaluates the homogeneous source field at all supplied
points. It does not determine which region contains a point. In regional total
fields, include this incident contribution only in `source.region()`.

Surface-current evaluation remains in `solvers/surface_current_fields.hpp`.
For each region its convention is `F_total = F_incident - Q[J, M]`.
GSTC returns both regional current pairs. PMCHWT returns the interior-oriented
pair `[J1, M1]`; use `[-J1, -M1]` for the exterior representation. Its two-block
RHS is the tested interior incident field minus the tested exterior incident field.

All repository includes now use `sie_mom/excitation/source_functors.cuh` in place
of the previous CUDA-directory header. There is no compatibility header.
