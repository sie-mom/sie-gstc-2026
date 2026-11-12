.. _quadrature:

Quadrature Rules for Numerical Integration
==========================================

In the context of the Method of Moments (MoM), we frequently encounter integrals over triangular domains. These integrals are often difficult or impossible to solve analytically. Numerical quadrature provides a powerful method for approximating these integrals. The core idea is to replace the integral of a function over a domain with a weighted sum of the function's values at a discrete set of points within that domain.

For a triangular domain :math:`T`, the approximation is:

.. math::

   \int_T f(\mathbf{r}) \, dS \approx A \sum_{i=1}^{N} w_i f(\mathbf{r}_i)

where:
- :math:`A` is the area of the triangle.
- :math:`N` is the number of quadrature points.
- :math:`\mathbf{r}_i` are the quadrature points (or nodes) within the triangle.
- :math:`w_i` are the corresponding quadrature weights.

The set of points and weights, :math:`\{(\mathbf{r}_i, w_i)\}`, is called the quadrature rule. The accuracy of the approximation depends on the number of points and their strategic placement.

The Standard Reference Triangle
-------------------------------

To standardize the process, quadrature rules are typically defined on a **standard reference triangle**. This is a simple, right-angled isosceles triangle in a 2D Cartesian plane with vertices at :math:`(0,0)`, :math:`(1,0)`, and :math:`(0,1)`.

Any physical triangle in 3D space can be mapped to this reference triangle, and vice-versa. This allows us to use a single, universal set of quadrature points for any triangle in the mesh.

Barycentric Coordinates
-----------------------

Barycentric coordinates provide a natural way to describe locations within a triangle. For a triangle with vertices :math:`\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2`, any point :math:`\mathbf{P}` inside the triangle can be expressed as a weighted average of the vertices:

.. math::

   \mathbf{P} = \alpha \mathbf{v}_0 + \beta \mathbf{v}_1 + \gamma \mathbf{v}_2

where the coordinates :math:`(\alpha, \beta, \gamma)` are the barycentric coordinates. They must satisfy:

.. math::

   \alpha + \beta + \gamma = 1
   \quad \text{and} \quad
   \alpha, \beta, \gamma \ge 0

The coordinates correspond to the fractional "area" opposite each vertex. For example, :math:`\alpha` is the ratio of the area of the sub-triangle formed by :math:`\mathbf{P}, \mathbf{v}_1, \mathbf{v}_2` to the total area of the triangle.

Dunavant Quadrature Rules
-------------------------

This project uses **Dunavant's symmetric quadrature rules**, which are specifically designed for triangles and offer a good balance of accuracy and efficiency. The rules are "symmetric" in that the set of quadrature points is invariant under permutations of the triangle's vertices.

Project Convention for Coordinates
----------------------------------

A critical implementation detail in this project is the relationship between the Cartesian coordinates of the quadrature points on the reference triangle and their barycentric coordinates.

The Python helper function :py:meth:`sie_mom.quadrature.dunavant_rules.dunavant_rule` returns a set of 2D points :math:`(x, y)` and corresponding weights. These :math:`(x, y)` coordinates are **not** arbitrary Cartesian points; they are defined on the standard reference triangle and directly correspond to two of the barycentric coordinates:

- :math:`x` corresponds to the barycentric coordinate :math:`\beta` (associated with vertex :math:`\mathbf{v}_1`).
- :math:`y` corresponds to the barycentric coordinate :math:`\gamma` (associated with vertex :math:`\mathbf{v}_2`).

The first barycentric coordinate, :math:`\alpha`, is then implicitly calculated as :math:`\alpha = 1 - x - y`.

This convention is consumed by the CUDA helper function :cpp:func:`::mapTriQuadPoint`, which maps these points from the reference element to the physical triangle in 3D space. The function signature is:

.. code-block:: c++

   __device__ inline void mapTriQuadPoint(int face_idx, float xi, float eta,
                                          const float* vertices, const int* triangles,
                                          float* r_out);

Here, the input parameters ``xi`` and ``eta`` are expected to be the :math:`\beta` and :math:`\gamma` coordinates, respectively. The mapping to the physical point :math:`\mathbf{P}` is performed as:

.. math::

   \mathbf{P} = (1 - \beta - \gamma)\mathbf{v}_0 + \beta \mathbf{v}_1 + \gamma \mathbf{v}_2

This consistent interpretation of coordinates between the Python point generator and the CUDA mapping function is essential for the correctness of the numerical integration.
