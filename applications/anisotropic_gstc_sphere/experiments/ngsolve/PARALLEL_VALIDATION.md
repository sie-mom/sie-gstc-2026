# Linux threading and geometry experiment — 2026-11-03

Multithreaded PARDISO worked without compiling NGSolve or writing C++. Refining
the core/sheet improved the sampled exterior field comparisons. Enlarging the
air/PML region improved power balance but did not uniformly improve those field
comparisons. This is an exploratory test, not a completed convergence study.

The previous approximately 16% finite-offset GSTC diagnostic must **not** be
interpreted as a 16% FEM boundary error: analytic target fields produce a similar
value at the same offsets. Details below.

## Environment and implementation

Ubuntu 24.04.5 LTS, Intel i9-12900KF, 24 logical CPUs, Python 3.12.3,
NGSolve 6.2.2608. Optional packages installed only in the experiment's `.venv`:
MKL 2026.2.0, Intel OpenMP 2026.2.0 and threadpoolctl 3.7.0, plus their runtime
dependencies. The repository root environment was not changed.

The wheel reports `USE_MKL=False`, `USE_PARDISO=False`, and `USE_UMFPACK=True`.
Those compile-time flags alone do not describe runtime capabilities. Installing
MKL enabled the wheel's Python PARDISO loader. The experiment's earlier solver
selection guard incorrectly rejected this possibility and has been corrected.
Explicit unavailable-backend requests still fail rather than silently falling
back. An added regression test covers runtime availability with both compile-time
flags disabled.

The installed [NGSolve runtime wrapper](https://github.com/NGSolve/ngsolve/blob/v6.2.2608/python/solvers/mkl_pardiso.py)
loads MKL from the pip distribution, supports complex matrices, and temporarily
sets the MKL thread limit from NGSolve's thread count while suspending its task
manager. This behavior was verified with a complex sparse probe solve and runtime
thread-pool inspection. NumPy and NGSolve's OpenBLAS libraries remained limited
to one thread; MKL/Intel OpenMP supplied the PARDISO parallel work.

Windows, native MUMPS integration, and TinkerCliffs were not tested. A generic
claim that loading a cluster MKL module automatically enables this particular
Python wheel is not established; its runtime loader checks pip-package metadata.

## Fixed-matrix thread scaling

`benchmark_threads.py` loaded the original curved mesh checkpoint, restored the
PML transform, assembled once, and reused that matrix and both RHSs throughout
each benchmark. Both ordering studies produced the identical matrix SHA-256:

```text
6852911b253fcee23def6320349a04d87c0e66665841a0694052f83b2aa7d0c1
```

There were 158,816 total DOFs. The reported time includes sparse analysis and
factorization, excluding assembly and the subsequent two backsolves.

| Threads | Minimum-degree ordering (s) | METIS ordering (s) |
|---:|---:|---:|
| 1 | 93.95 | 33.60 |
| 4 | 27.41 | 10.27 |
| 8 | 18.77 | 7.00 |
| 16 | 22.67 | 7.55 |

A repeated one-thread minimum-degree run took 94.08 seconds. The CPU-seconds /
wall-seconds ratios were approximately 1.00, 3.93, 7.72 and 15.45 for its 1/4/8/16
thread runs; the METIS ratios were approximately 1.00, 3.76, 7.13 and 14.10.
These demonstrate actual parallel CPU work, not merely configured thread limits.
Eight threads was fastest among those tested on this heterogeneous desktop CPU.
This does not predict the best count on the Windows server.

METIS also helped substantially at the same thread count. Comparing the older
UMFPACK time of roughly 116 seconds with PARDISO/METIS at roughly 7 seconds
therefore combines backend, ordering and threading changes. It is not a pure
threading speedup. These are exploratory single-run timings, not statistical
performance guarantees.

Benchmark residuals were around `2e-15`. Coefficients differed from the saved
UMFPACK checkpoint by approximately `1e-6` or less; this comparison includes
curved-mesh serialization/reassembly differences. Fresh baseline field samples
matched the earlier run to numerical precision. Tiny linear residuals establish
accuracy for the assembled linear system, not continuum electromagnetic accuracy.

## Core mesh and air/PML experiments

All runs retained the synthesized `face_tensors.bin`, 10 GHz frequency, radius
0.7 lambda0, interior epsilon_r=7, both original dipole cases, field order 3 and
geometry order 3. All used PARDISO, METIS ordering and eight threads, with
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=8`, `OMP_NUM_THREADS=8`, `MKL_DYNAMIC=FALSE`.

| Run | Core ppw | Exterior ppw | Air radius / lambda0 | PML thickness / lambda0 | PML strength | DOFs | Analysis/LU (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_metis8 | 3 | 3 | 0.95 | 0.25 | 3 | 158,816 | 6.89 |
| core4_metis8 | 4 | 3 | 0.95 | 0.25 | 3 | 383,266 | 42.71 |
| core4_air125_pml05_metis8 | 4 | 3 | 1.25 | 0.5 | 1.5 | 409,109 | 57.79 |

Tetrahedra counts were 5,614 / 13,950 / 14,876, respectively. Assembly took
0.32 / 0.62 / 0.65 seconds. Factorization CPU/wall ratios were 7.03 / 7.55 / 7.66.
All six physical case solves completed, with residuals around `2e-15`–`3.3e-15`.
Each geometry reused one factorization across both excitations.

`--interior-ppw 4` reduces the core/sheet maximum element size from regional
wavelength/3 to wavelength/4. It also influences adjacent exterior elements;
it does not preserve the exterior mesh exactly. The larger-domain run changes
air extent, PML thickness, stretch and the generated mesh together. It is not
an isolated PML-reflection measurement.

`compare_common_points.py` evaluated all checkpoints on the original baseline's
radial observation points. This avoids comparing different radial intervals
when the air region grows. Exterior complex-field differences from the prescribed
targets on those **same points** were:

| Run | Electric incidence E | Electric incidence H | Magnetic incidence E | Magnetic incidence H |
|---|---:|---:|---:|---:|
| Baseline | 2.587% | 1.744% | 1.567% | 2.050% |
| Finer core/sheet | 1.481% | 1.382% | 0.921% | 1.662% |
| Finer core/sheet + larger air/PML | 1.779% | 1.308% | 1.795% | 1.746% |

Inner/outer power-flux relative imbalances were:

| Run | Electric incidence | Magnetic incidence |
|---|---:|---:|
| Baseline | 0.328% | 0.368% |
| Finer core/sheet | 0.268% | 0.279% |
| Finer core/sheet + larger air/PML | 0.0613% | 0.0527% |

Thus the larger domain improved power balance but not every field comparison.
The results support further controlled refinement, not an assumption that a
larger domain automatically makes every diagnostic better. Radial comparisons
also do not establish error at every angle or in each suppressed component.

## Finite-offset GSTC diagnostic clarification

The existing diagnostic compares fields at `a-d` and `a+d` as if they were
two traces at the same surface. Even an exact interface solution need not satisfy
that displaced-surface expression. Evaluating the analytic prescribed fields
with the same diagnostic and `d/a=0.005` gave:

| Case | Analytic electric-current residual | Analytic magnetic-current residual |
|---|---:|---:|
| Electric incidence | 15.97% | 15.45% |
| Magnetic incidence | 15.40% | 16.06% |

These values account for much of the corresponding numerical approximately 16%
measurements. We must not call those measurements 16% FEM boundary errors, and
subtracting the scalar percentages is not a valid error estimator either.
Use true one-sided boundary traces, an offset-limit study, or compare the
residual **vectors** with the analytic displaced-surface baseline. The original
probes remain saved unchanged with their finite-offset labels.

## Files, checks and reproduction

All artifacts are under the Git-ignored directory:

```text
applications/anisotropic_gstc_sphere/results/dipole_pol_rot/dual_pol_eps7/experiments/ngsolve/parallel_20261103/
```

- `threads/benchmark.json`, `threads_metis/benchmark.json`: matrix fingerprints,
  thread counts, CPU/wall times, residuals and runtime library details.
- `commands.json`: exact commands for the three physical model runs.
- `common_point_comparison.json`: comparisons on identical radial coordinates.
- `analytic_offset_diagnostics.json`: displaced-surface analytic baselines.
- Each physical run: configuration, metrics, mesh/solution checkpoints and complex
  radial/ZX/far-field samples. Finer runs also have `plots_quick_compare/`.
- `installed-packages.txt`, logs and `tests.log`: reproducibility evidence.

**10 tests passed in 2.79 seconds**, with no skips. Ruff checks passed, with only
the repository's existing configuration deprecation warning. No tracked files
outside this isolated experiment were changed. Nothing was committed.

See the main README's optional PARDISO section for installation and run commands.
The pasted advice was right that MKL PARDISO can be optional and convenient here.
Its claims of automatic exact/reference-grade physical results, universal DOF
timing thresholds, and core count being the only solver difference were too strong.
