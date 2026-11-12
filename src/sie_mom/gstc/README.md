# Local GSTC Surface Parameter Extraction

This directory contains reusable, pointwise GSTC synthesis algorithms: they
extract surface parameter tensors from prescribed boundary fields. They do not
choose a field transformation, generate sources, construct surface coordinates,
or solve the global scattering problem.

The extraction layer can be reused with spherical tangent frames or
vertex-derived face frames, provided they satisfy the same oriented orthonormal
frame convention. This does not yet make the application or C++ solver
independent of spherical geometry.

## Modules

| Module | Responsibility |
| --- | --- |
| [surface_parameters.py](surface_parameters.py) | General reciprocal complex and reciprocal passive-lossless tensor extraction, with rank and fitting diagnostics. |
| [omega_biisotropic.py](omega_biisotropic.py) | Reciprocal lossless omega-sheet specialization, plus boundary-quantity and power helpers currently shared by both extractors. |
| [__init__.py](__init__.py) | Public imports for the extraction layer. |

The general extractor's import of `BoundaryQuantities` from
`omega_biisotropic.py` shares a data container; it does not impose an omega
material model or a spherical coordinate system.

## Local Frame Contract

At each surface sample, use real unit vectors `t1`, `t2`, and `n` satisfying:

```text
t1 . t2 = t1 . n = t2 . n = 0
t1 x t2 = n
n points from the interior region to the exterior region
```

The normal is outward for a closed surface. Tangential components of a global
Cartesian vector `F` are `F_t = [t1 . F, t2 . F]`. In this frame:

```text
n x [F_1, F_2] = [-F_2, F_1]
```

Frames may vary from sample to sample. At any one sample, all prescribed states,
electric and magnetic fields, currents, and tensor blocks must use the same
frame. The functions accept components, not frame vectors, so they cannot check
orthonormality, handedness, or the normal direction themselves.

Nonorthogonal or non-unit coordinate bases are not supported directly. Their
metric factors and component transformations must be handled before calling
these routines. Reversing the normal or swapping tangent axes without updating
the convention changes the GSTC signs.

## Boundary Quantities And Material Models

All routines use the `exp(+j omega t)` phasor convention. With the jump defined
as exterior minus interior, `form_boundary_quantities` forms:

```text
E_avg = (E_ext + E_int) / 2
H_avg = (H_ext + H_int) / 2
J_s   =  n x (H_ext - H_int)
M_s   = -n x (E_ext - E_int)

[J_s]   [Y_ee  Y_em] [E_avg]
[M_s] = [Y_me  Y_mm] [H_avg] = Lambda [E_avg; H_avg]
```

These are physical fields-to-currents surface parameters, not polarization
susceptibilities. `Y_ee` has units of siemens, `Y_mm` has units of ohms, and
`Y_em` and `Y_me` are dimensionless. This layer implements a local tangential
sheet model; it does not include normal-polarization derivative terms or
coupling between different surface samples.

`surface_parameters.py` provides two extraction methods:

- `extract_reciprocal_complex_least_squares`: symmetric self-blocks and
  `Y_me = -Y_em.T`, without imposing passivity or losslessness.
- `extract_reciprocal_lossless_least_squares`: the same reciprocity condition,
  with `Lambda = [[j B, K], [-K.T, j X]]`, where `B` and `X` are real symmetric
  and `K` is real. The self-blocks are purely imaginary and the coupling blocks
  are purely real.

For the second model, `Lambda + Lambda.conj().T = 0`, so the sheet absorbs or
generates no time-average real power for any local field state. This is the
passive-lossless constraint at the modeled frequency, not a guarantee of
broadband realizability. An incompatible prescribed transformation can still
have a nonzero fitting residual.

`structure="general"` has ten independent complex or real coefficients,
respectively. `structure="diagonal"` has four: both self-blocks are diagonal
and both coupling blocks are zero. The omega specialization instead uses three
real parameters, with isotropic self-blocks and normal-cross coupling.

Boundary fields have component-first shape `(2, ...samples)`. General tensor
blocks have shape `(2, 2, ...samples)`; `SurfaceParameters.as_matrix()` returns
`(4, 4, ...samples)` in the component order `[t1, t2]` for each field or current.
All supplied states must have matching sample shapes.

## Reuse Versus Basis Invariance

Geometry-independent input does not mean every extraction choice is invariant
under a change of tangent axes:

- For a rotation within the same tangent plane and with the same normal, let
  `F_new = R F_old` and `Q = block_diag(R, R)`. The same physical map is
  `Lambda_new = Q Lambda_old Q.T`. Full-rank general extraction obeys this
  transformation, up to numerical tolerances.
- A diagonal anisotropic tensor is generally not diagonal after rotating the
  axes. The diagonal model therefore selects material axes and is not an
  unrestricted, basis-independent material model.
- Rank-deficient general extraction currently returns the minimum Euclidean
  norm of the independent physical coefficients, without column normalization.
  That selection is not invariant under a tangent-axis rotation: the same
  underdetermined fields can select different physical tensors in different
  frames. A basis-invariant selection rule is not implemented yet.
- Changing from a radial sphere normal to a triangle's planar normal changes
  the tangent plane itself. This is not merely a two-dimensional basis rotation.
  Fields must be projected using the intended surface normal and sampling
  geometry; an existing local tensor cannot simply be relabeled.

For the general extractors, `unique` indicates full column rank, not an exact
fit. Rank-deficient samples retain a minimum-norm fit and report infinite
condition number. Inspect the aggregate and per-state residuals separately.
Magnetic-current residuals are divided by the supplied positive
`reference_impedance`, putting both residual blocks in A/m. This weighting does
not normalize the independent coefficients. The omega routines have their own
diagnostic and residual conventions; do not compare their raw residual values
without checking that scaling.

The [general extraction tests](../../../tests/python/test_surface_parameters.py)
cover full-rank tangent-frame rotations, material constraints, and non-unique
fits. They do not assert rotation invariance of rank-deficient minimum-norm
solutions.

## Planned Vertex-Derived Face Frames

For a nondegenerate triangular face with consistently outward-oriented vertex
order `(v0, v1, v2)`, a deterministic frame can be constructed as:

```text
e1 = v1 - v0
e2 = v2 - v0
n  = normalize(e1 x e2)
t1 = normalize(e1)
t2 = n x t1
```

This gives `t1 x t2 = n`. Outward orientation must be established by the mesh
layer: a vertex cross product alone does not prove that the normal is outward.
Reject degenerate faces and verify consistent winding on the closed mesh. A
radial test against the origin is not a general outward-normal test for
arbitrary closed surfaces.

The planned separation of responsibilities is:

1. A geometry/frame provider supplies sample positions and `(n, t1, t2)`.
2. Prescribed-field routines evaluate interior and exterior `E` and `H` in
   global Cartesian components. They define the transformation and sources,
   such as plane waves or sets of dipoles, independently of extraction.
3. The preparation layer projects those fields onto `(t1, t2)` and calls
   `form_boundary_quantities` for every prescribed state.
4. This directory's extractors fit the selected surface parameter model.
5. The export and C++ assembly layers use exactly the same per-face frame,
   normal convention, sample association, and component ordering.

For example, a block `Y_local` is lifted to Cartesian coordinates as
`T Y_local T.T`, with the real `3 x 2` matrix `T = [t1 t2]`. This only represents
the intended tensor when the assembler uses the frame in which it was
extracted. Choosing two independently constructed right-handed frames is not
enough. Frame vectors must eventually be exported or reproduced by an explicit
shared rule; their serialization is not implemented by the extractors.

## Local Face Coordinate System Integration

The intrinsic face-edge coordinate frame system is fully implemented end to end:

- [face_frames.py](../mesh/face_frames.py) computes intrinsic orthogonal frames
  `t1 = normalize(v1 - v0)`, `n = normalize((v1 - v0) x (v2 - v0))`,
  `t2 = n x t1` directly from ordered face vertices `(v0, v1, v2)`, together with
  Cartesian-to-frame field projection (`project_to_face_frame`) and 2D-to-3D tensor
  lifting (`lift_2x2_to_3x3`).
- [excitation/fields.py](../excitation/fields.py) exposes Cartesian field
  evaluators (`evaluate_plane_wave_cartesian`, `evaluate_dipole_fields_cartesian`).
- [prepare_sie_inputs.py](../../../applications/anisotropic_gstc_sphere/prepare_sie_inputs.py)
  supports both `--basis spherical` and `--basis face`. When `basis == "face"`,
  it evaluates Cartesian prescribed fields at face centroids, projects them onto
  `(t1, t2)`, and exports face tensors in the intrinsic local frame.
- [GSTC assembly and matvec kernels](../../../include/sie_mom/cuda/sie_kernels_assembly.cuh)
  support `GstcBasisType::face` and `GstcBasisType::spherical` via `get_local_basis_vectors`
  in [gstc_tensor_utils.cuh](../../../include/sie_mom/cuda/gstc_tensor_utils.cuh).
- The core solver is geometry-agnostic: [gstc_sie.hpp](../../../include/sie_mom/solvers/gstc_sie.hpp)
  and [gstc_sie.cu](../../../src/sie_mom/solvers/gstc_sie.cu).
- Forward executables like [anisotropic_sphere_app.cpp](../../../applications/anisotropic_gstc_sphere/anisotropic_sphere_app.cpp)
  accept `--basis {spherical,face}` (defaulting to `spherical` for backward compatibility).
