.. _api_mesh:

Mesh API (C++ and Python)
=========================

This section documents the C++ mesh classes and utilities from their Doxygen
comments, and the Python mesh generators and face-frame helpers from their
docstrings.


Data Structures
---------------

.. doxygenstruct:: sie_mom::mesh::MeshData
   :project: sie_mom

.. doxygenclass:: sie_mom::mesh::SphericalSurface
   :project: sie_mom


ObjMeshLoader
-------------

.. doxygenclass:: sie_mom::mesh::ObjMeshLoader
   :project: sie_mom


Mesh Utilities
--------------

.. doxygenfunction:: sie_mom::mesh::export_mesh_centroids
   :project: sie_mom

.. doxygenfunction:: sie_mom::mesh::export_mesh_data
   :project: sie_mom


Gmsh Mesh Generators
--------------------

.. automodule:: sie_mom.mesh.mesh_generators
   :members: BaseGmshGenerator, SphereGenerator, ConeSphereGenerator, BluntedConeSphereGenerator, CircularOgiveGenerator, CommonMeshOptions
   :show-inheritance:

.. automethod:: sie_mom.mesh.mesh_generators.BaseGmshGenerator.generate


Circular Ogive Geometry and Tip Grading
---------------------------------------

The circular-arc spindle in publication Example 3 uses
:class:`~sie_mom.mesh.mesh_generators.CircularOgiveGenerator` with
:class:`~sie_mom.mesh.circular_ogive.CircularOgive` geometry and optional
:class:`~sie_mom.mesh.circular_ogive.OgiveTipGrading` for local tip refinement.
Call the inherited :meth:`~sie_mom.mesh.mesh_generators.BaseGmshGenerator.generate`
method to generate and export the mesh.

.. automodule:: sie_mom.mesh.circular_ogive
   :members: CircularOgive, OgiveTipGrading
   :show-inheritance:


Python Mesh & Face Frames
-------------------------

.. automodule:: sie_mom.mesh.face_frames
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sie_mom.mesh.obj_mesh
   :members:
   :undoc-members:
   :show-inheritance:
