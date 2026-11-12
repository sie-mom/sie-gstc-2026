SIE-MoM
========

SIE-MoM is a research code for electromagnetic scattering and radiation from
dielectric bodies and metasurfaces described by generalized sheet transition
conditions (GSTCs). It combines a C++/CUDA surface integral equation solver
with Python workflows for mesh generation, surface-parameter synthesis,
validation, and plotting.

Developed by German Farinas Perez.
Contact: `germanf@vt.edu <mailto:germanf@vt.edu>`_.

.. admonition:: Associated publication
   :class: seealso

   **SIE-GSTC Formulation for Closed Bianisotropic Metasurfaces with
   Passive Lossless Synthesis**

   Authors: **German Farinas Perez** and **Jordan Budhu**.

   Status: *Submitted*.

The three publication workflows reproduce the paper's numerical examples:
omega-biisotropic impedance matching, anisotropic polarization rotation,
and a GSTC metasurface on a nonspherical surface.

Start with :doc:`quickstart` to install the software and reproduce an example.
Use the :doc:`api_reference` when writing your own workflows. The developer
guide records the mathematical derivations and their connections to the
implementation for readers extending or checking the solver.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Developer guide and numerical methods

   theory
   quadrature
   singular
   matrix_assembly_v2
   scatter_field
   sphere_svwf
   mesh
   metasurface_applications
   implementing_gstc_comsol_case_b
   bibliography
