# Mesh comparison for the dual-polarization GSTC sphere

The common physical surface is the GSTC sphere at `r=a=0.7 lambda0`, with
interior relative permittivity 7 and exterior relative permittivity 1 at 10 GHz.
Counts below are per mesh/system, shared by both incident polarizations.

| Solver / run | Triangles on GSTC sphere | Volume tetrahedra, including PML | Complex unknowns, total / free | Field approximation |
| --- | ---: | ---: | ---: | --- |
| SIE / BEM, `dual_pol_eps7` | 3,796 | 0 | 22,776 / 22,776 | RWG currents on planar triangles |
| NGSolve, `baseline_metis8` | 824 | 5,614 | 158,816 / 147,208 | HCurl order 3; curved geometry order 3 |
| NGSolve, `core4_metis8` | 1,504 | 13,950 | 383,266 / 362,172 | HCurl order 3; curved geometry order 3 |
| NGSolve, `core4_air125_pml05_metis8` | 1,470 | 14,876 | 409,109 / 382,428 | HCurl order 3; curved geometry order 3 |
| COMSOL 6.1, Windows Case A `case2` | 13,596 | 2,236,150 | Not reported | Linear, from the supplied log |

SIE's surface mesh has 1,900 vertices and 5,694 RWG edges. Its formulation
uses four coefficient blocks, `J1, M1, J2, M2`, giving
`4 * 5,694 = 22,776` complex unknowns. It needs no surrounding volume mesh or
PML. The matrix is dense; NGSolve's volume matrix is sparse. These unknown
counts therefore do not directly predict memory use or solve time.

NGSolve counts above are **unique geometric triangles on the named `sheet`
interface**, rather than every face of every tetrahedron or every mesh boundary.
The two independent tangential E traces share this geometric interface: triangles
are counted once. Order-3 fields have multiple basis functions per mesh entity;
824 NGSolve interface triangles are not equivalent to 824 lowest-order RWG
triangles. Curved geometry also represents the sphere differently from the
planar SIE mesh. Equal triangle counts would not establish equal accuracy.

NGSolve's original `validation_20261103/order3_pml025_2d` checkpoint has the same
824 sheet triangles and 5,614 tetrahedra as the fresh parallel baseline. The
earlier `order3_pml025` run has no mesh checkpoint; the repeated run's matching
settings and previously checked field agreement identify that comparison.

The larger-air/PML run regenerates the whole mesh, so its interface count is
slightly different despite the same core size limit. These meshes are not a
nested refinement sequence. Their volume breakdown is:

| NGSolve run | Core tetrahedra | Physical-air tetrahedra | PML tetrahedra | Total |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2,134 | 2,037 | 1,443 | 5,614 |
| Finer core | 8,428 | 3,761 | 1,761 | 13,950 |
| Finer core + larger air/PML | 5,797 | 7,211 | 1,868 | 14,876 |

The baseline and finer-core runs have air radius `0.95 lambda0` and PML
thickness `0.25 lambda0`. The larger run uses `1.25 lambda0` and `0.5 lambda0`.
COMSOL's pasted log uses `0.95 lambda0` and `0.25 lambda0`, but additionally
reports source-ball refinement and different volume/surface size settings.
NGSolve treats the centered singular dipole analytically and solves for a
regular interior correction. This further prevents a simple element-for-element
comparison with the COMSOL setup.

Provenance: SIE counts come from
`results/dipole_pol_rot/dual_pol_eps7/batch_metrics.json`. NGSolve boundary and
material counts were read directly from each saved `mesh.vol.gz`, with total
and free DOFs taken from its `metrics.json`. All boundary elements were checked
to have three vertices, and the `sheet` vertex triples were checked for duplicates.
The COMSOL row comes from the user-provided Windows log dated
`2026-11-02T21:29:52`; no COMSOL result directory is available in this Linux
checkout, and its DOF count cannot be inferred reliably from the element count.

The machine-readable counts, checkpoint hashes and source paths are saved in
`results/dipole_pol_rot/dual_pol_eps7/experiments/ngsolve/parallel_20261103/mesh_comparison.json`.
For an accuracy comparison, combine this table with the common-point complex
field errors in [PARALLEL_VALIDATION.md](PARALLEL_VALIDATION.md), and ultimately
check convergence for each formulation.
