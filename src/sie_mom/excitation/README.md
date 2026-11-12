# Shared Excitation

This directory groups the Python and C++/CUDA excitation implementations.
They are separate language implementations; placing them together does not
change the C++ build or create a Python binding to CUDA.

| File | Responsibility |
| --- | --- |
| [fields.py](fields.py) | Analytical electric/magnetic dipole and plane-wave E/H fields in Python. |
| [__init__.py](__init__.py) | Public Python imports from `sie_mom.excitation`. |
| [excitation.cu](excitation.cu) | C++ source validation, medium binding, and GPU incident-field evaluation. |

The C++ declarations and field functors live under
[`include/sie_mom/excitation`](../../../include/sie_mom/excitation/README.md).

## Python API

```python
from sie_mom.excitation import evaluate_dipole_fields, evaluate_plane_wave_fields
```

Both functions return `(E, H)` with component-first shape `(3, ...)`, ordered
as `(r, theta, phi)`. Observation coordinates are spherical; dipole positions,
effective moments, plane-wave electric amplitudes, and propagation directions
are specified in global Cartesian coordinates. Callers supply the homogeneous
medium's wavenumber `k` and impedance `eta`.

The relocation preserves the existing formulas and spatial-phase conventions:

- `evaluate_dipole_fields` uses `exp(-j k R)` and accepts electric or magnetic
  dipoles. An electric dipole's `effective_moment` is `p / (4 pi epsilon)`,
  not its physical SI moment. The magnetic branch uses the documented duality
  convention.
- `evaluate_plane_wave_fields` uses `exp(+j k.r)`. Applications that require
  the opposite spatial phase retain their existing conversion; this module
  move does not change or harmonize that convention.

These evaluators generate analytical fields. They do not choose GSTC
transformations, impose surface parameter constraints, select incident regions,
or decide whether the interior or exterior prescribed field is zero. Those
choices belong to the application and its configuration. There is no field
superposition or new configuration schema introduced by this relocation.

All in-repository Python imports now use `sie_mom.excitation`; the former
top-level `sie_mom.excitation_sources` module has been moved, not retained as
a compatibility wrapper.
