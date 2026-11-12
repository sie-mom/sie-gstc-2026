.. _api_excitation_sources:

Analytical Excitation Sources API (Python)
==========================================

The analytical source routines evaluate plane-wave and point-dipole fields
independently of a particular GSTC synthesis application. Field arrays use
the component-first spherical ordering :math:`(r,\theta,\phi)`.

The public Python API is ``sie_mom.excitation``; its analytical implementation
lives in ``sie_mom.excitation.fields`` alongside the C++/CUDA implementation.

.. automodule:: sie_mom.excitation
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
