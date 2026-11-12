.. _quickstart:

Quick start
===========

This guide installs SIE-MoM and runs the publication workflows from the
repository root. Python coordinates the simulations and plots; the dense
SIE systems are assembled and solved by the C++/CUDA executables.

Prerequisites
-------------

The commands below use a Linux shell and require:

* Python 3.10 or newer, with ``venv`` and ``pip``.
* CMake 3.18 or newer and a C++17 compiler compatible with the CUDA Toolkit.
* An NVIDIA GPU, a working NVIDIA driver, and the CUDA Toolkit, including
  ``nvcc``, cuBLAS, and cuSOLVER.
* The Python dependencies installed below: NumPy, SciPy, Gmsh, Matplotlib,
  pandas, trimesh, and rtree. Eigen is included in ``vendor/eigen``.

Check that the driver and compiler are available:

.. code-block:: bash

   nvidia-smi
   nvcc --version

The publication configurations use dense matrices. Examples 2 and 3 have
about 23,000 unknowns: the double-precision complex matrix alone occupies
about 8 GiB of GPU memory, with additional memory needed for the solver
workspace and other arrays. A 12 GB GPU has been used for the spherical
publication-sized system. Memory use grows quadratically with the number
of unknowns; host RAM and disk space are also needed for field samples and
generated results.

Install Python dependencies
---------------------------

From your checkout of the repository:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[mesh,plot,sampling]" "pandas>=2.2"

NumPy and SciPy are installed with the base package. The extras provide
meshing, plotting, and observation-point classification; pandas is also
used by the spherical benchmark runners.

Build the CUDA executables
--------------------------

Configure a Release build with double precision. Set the CUDA architecture
to match the GPU on which the simulations will run; ``86`` below is the
example value for an RTX 3060.

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
     -DSIE_MOM_USE_DOUBLE_PRECISION=ON -DCMAKE_CUDA_ARCHITECTURES=86
   cmake --build build --parallel 4 --target \
     pmchwt_dielectric_sphere_app omega_sphere_sie anisotropic_sphere_app \
     evaluate_surface_fields mesh_obj_loader_tool

Keep the build directory named ``build``: the publication runners look for
the executables there. CMake must report that it found the CUDA Toolkit
and enabled CUDA language support.

Run the publication examples
----------------------------

Run the following commands from the repository root with the virtual
environment active. Each runner writes numerical results under its
application's ``results/`` directory and PDF/PNG figures under its
``publication/figures/`` directory.

Example 1: Omega-biisotropic impedance matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A dielectric sphere excited by an interior electric dipole is coated with
an omega-biisotropic metasurface. The workflow compares the coated sphere
with the bare dielectric sphere.

On a fresh checkout, generate the mesh if it is absent and prepare the SVWF
benchmark required by the radial comparison plot:

.. code-block:: bash

   python -m applications.omega_biisotropic_sphere.generate_sphere_mesh \
     --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json
   python -m applications.omega_biisotropic_sphere.svwf_dipole_impedance_matching \
     --config applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7.json

Then run the publication workflow:

.. code-block:: bash

   python applications/omega_biisotropic_sphere/publication/run_example_1.py

The runner performs the PMCHWT baseline and SIE-GSTC simulations, then plots
the results. It uses the prepared SVWF data; an existing COMSOL export is
included in the comparison when available.

Detailed options: :download:`Example 1 workflow notes
<../../applications/omega_biisotropic_sphere/publication/README.md>`.

Example 2: Anisotropic polarization rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Part A transforms electric-dipole excitation into magnetic-dipole radiation.
Part B uses one metasurface for both electric-to-magnetic and
magnetic-to-electric transformations. Each part has its own runner:

.. code-block:: bash

   python applications/anisotropic_gstc_sphere/publication/run_example_2_part_a.py
   python applications/anisotropic_gstc_sphere/publication/run_example_2_part_b.py

The runners coordinate mesh preparation, SIE-GSTC solves, independent SVWF
benchmarks, field plots, and surface-parameter plots.

Detailed options: :download:`Example 2 workflow notes
<../../applications/anisotropic_gstc_sphere/publication/README.md>`.

Example 3: GSTC metasurface on a circular-arc spindle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example applies two prescribed dipole-field transformations to a
nonspherical dielectric body:

.. code-block:: bash

   python -m applications.gstc_surface.publication.run_example_3

The runner coordinates mesh and tensor preparation, the batch SIE-GSTC
solve, observation sampling, field evaluation, a bare-dielectric PMCHWT
baseline, and publication plots.

Detailed options: :download:`Example 3 workflow notes
<../../applications/gstc_surface/publication/README.md>`.

Reusing results and optional comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every publication runner accepts ``--help`` and ``--plots-only``.
``--plots-only`` requires numerical results already present on disk.
Examples 2 and 3 reuse completed stages by default; ``--force`` removes
existing stage results and recomputes them. Example 1 instead provides
``--skip-pmchwt`` and ``--skip-sie`` to reuse the corresponding results.

The SIE workflows run independently of COMSOL and NGSolve. To add Example
3's optional NGSolve comparison, install ``python -m pip install -e
".[ngsolve]"`` and run its publication command with ``--with-ngsolve``.

Where to go next
----------------

* :doc:`api_reference` documents the Python helpers and C++/CUDA interfaces
  for building your own workflows.
* :doc:`metasurface_applications` explains the application workflows.
* :doc:`mesh` describes mesh generation and validity requirements.
* :doc:`theory`, :doc:`singular`, and :doc:`matrix_assembly_v2` provide the
  derivations and implementation details for extending the solver.
