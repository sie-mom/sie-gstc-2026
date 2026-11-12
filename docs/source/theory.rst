.. _theory:

###################
Theoretical Basis
###################

Operator identities and interface equations
===========================================

General formulation of the Surface Integral Equations (SIE) representation :cite:p:`poggio1973integral,gibson2021method,kumagai1990integral,Medgyesi-Mitschang:94`.

   .. math::
      :label: eq:sie1

      -\tfrac{1}{2}\,\hat{\mathbf{n}} \times \mathbf{M}_1
      + j\omega \mu_1\, \mathcal{L}_{k_1}[\mathbf{J}_1]_{\mathrm{tan}}
      + \mathcal{K}_{k_1}[\mathbf{M}_1]_{\mathrm{tan}} = 0.

   .. math::
      :label: eq:sie2

      	\tfrac{1}{2}\,\hat{\mathbf{n}} \times \mathbf{J}_1
      + j\omega \varepsilon_1\, \mathcal{L}_{k_1}[\mathbf{M}_1]_{\mathrm{tan}}
      - \mathcal{K}_{k_1}[\mathbf{J}_1]_{\mathrm{tan}} = 0.

   .. math::
      :label: eq:sie3

      	\tfrac{1}{2}\,\hat{\mathbf{n}} \times \mathbf{M}_2
      + j\omega \mu_2\, \mathcal{L}_{k_2}[\mathbf{J}_2]_{\mathrm{tan}}
      + \mathcal{K}_{k_2}[\mathbf{M}_2]_{\mathrm{tan}} = \mathbf{E}^{\mathrm{inc}}_{\mathrm{tan}}.

   .. math::
      :label: eq:sie4

      -\tfrac{1}{2}\,\hat{\mathbf{n}} \times \mathbf{J}_2
      + j\omega \varepsilon_2\, \mathcal{L}_{k_2}[\mathbf{M}_2]_{\mathrm{tan}}
      - \mathcal{K}_{k_2}[\mathbf{J}_2]_{\mathrm{tan}} = \mathbf{H}^{\mathrm{inc}}_{\mathrm{tan}}.


Let :math:`\hat{\mathbf{n}}` be the unit normal on :math:`S` pointing from *region 1* to *region 2*.
The tangential trace is :math:`\mathbf{X}_{\mathrm{tan}} := - \hat{\mathbf{n}} \times (\hat{\mathbf{n}} \times \mathbf{X})`.

Equivalent surface currents (two-sided traces)
----------------------------------------------
Here, :math:`\mathbf{J}_i` and :math:`\mathbf{M}_i` are the equivalent electric and magnetic surface currents on the interface :math:`S` between two homogeneous regions (1 and 2) with wavenumbers :math:`k_i`, permittivities :math:`\varepsilon_i`, and permeabilities :math:`\mu_i`. 


.. math::
   :label: eq:eqcurrents

   \boxed{
   \begin{aligned}
      \mathbf{J}_1 &= -\,\hat{\mathbf{n}} \times \mathbf{H}_1, &\qquad
      \mathbf{M}_1 &= \hat{\mathbf{n}} \times \mathbf{E}_1, \\
      \mathbf{J}_2 &= \hat{\mathbf{n}} \times \mathbf{H}_2, &\qquad
      \mathbf{M}_2 &= -\,\hat{\mathbf{n}} \times \mathbf{E}_2
   \end{aligned}
   }

The operators :math:`\mathcal{L}` and :math:`\mathcal{K}` are defined as follows:

.. math::
   :label: eq:Loperator

   \mathcal{L}_{k_i}\{\mathbf{X}(\mathbf{r})\}
   = \operatorname{p.v.}\!\int_{S} \left[
       \mathbf{X}(\mathbf{r}')\,G_i
       - \frac{1}{k_i^2}\big(\nabla'_{S}\!\cdot \mathbf{X}(\mathbf{r}')\big)
         \nabla' G_i
     \right] \mathrm{d}S',

.. math::
   :label: eq:Koperator

   \mathcal{K}_{k_i}\{\mathbf{X}(\mathbf{r})\}
   = \operatorname{p.v.}\!\int_{S} \mathbf{X}(\mathbf{r}') \times \nabla' G_i \, \mathrm{d}S'.

In both definitions, the notation :math:`\operatorname{p.v.}\!\int_S` indicates the surface principal-value integral obtained by excising an infinitesimal patch around the observation point :math:`\mathbf{r}` before taking the limit as the patch radius shrinks to zero.

With wavenumber :math:`k_i`, the free-space Green's function is:

.. math::
   :label: eq:greenfunc

   G_i(\mathbf{r}, \mathbf{r}') = \frac{e^{-j k_i |\mathbf{r} - \mathbf{r}'|}}{4\pi |\mathbf{r} - \mathbf{r}'|}.

.. The CUDA implementation of this function can be found here:
.. :cpp:func:`::getGreenFunctionValue`.

A sketch of the interface :math:`S` separating two regions, with equivalent surface currents and an incident field in Region 2, is shown below.

.. _fig:geometry:

.. tikz:: A sketch of the interface S separating two regions.

   \begin{tikzpicture}[scale=1.0, >=Latex]
     % Surface patch S
     \fill[blue!10] (-2.5,-0.6) .. controls (-2.0,1.2) and (2.0,1.2) .. (2.5,-0.6)
                               .. controls (2.0,-2.0) and (-2.0,-2.0) .. (-2.5,-0.6);
     \draw[thick] (-2.5,-0.6) .. controls (-2.0,1.2) and (2.0,1.2) .. (2.5,-0.6) .. controls (2.0,-2.0) and (-2.0,-2.0) .. (-2.5,-0.6) node[left, xshift=1mm] {$S$};
     % Region labels
     \node at (0, 0.9) {Region 2 (exterior)};
     \node at (0, -0.7) {Region 1 (body)};
     % Normal vector
     \draw[very thick, ->] (0, 0.8) -- (0, 1.5) node[above] {$\hat{\mathbf{n}}$};
     % Equivalent currents
     \node at (-3.0, 0.0) {$\mathbf{J}_2, \mathbf{M}_2$};
     \node at (-1.4, -0.2) {$\mathbf{J}_1, \mathbf{M}_1$};
     % Incident field
     \draw[->, thick, red] (-2.2, 1.0) -- (-1.5, 0.5) node[midway, above left] {$\mathbf{E}^{\mathrm{inc}}, \mathbf{H}^{\mathrm{inc}}$};
   \end{tikzpicture}

Discretization into RWG basis functions
------------------------------------------
The equivalent surface currents are expanded in terms of Rao-Wilton-Glisson (RWG) basis functions :math:`\mathbf{f}_n(\mathbf{r})` defined on the triangular mesh of the surface :math:`S` :cite:p:`rao1982electromagnetic`:

.. math::
   :label: eq:basis

   \begin{aligned}
      \mathbf{J}_i(\mathbf{r}) &= \sum_{n} I_n \mathbf{f}_n(\mathbf{r}), \\
      \mathbf{M}_i(\mathbf{r}) &= \sum_{n} M_n \mathbf{f}_n(\mathbf{r}),
   \end{aligned} \qquad i = 1, 2.
   
Each basis function :math:`\mathbf{f}_n` is associated with the *n*-th interior edge of the mesh and is non-zero only on the two triangles, :math:`T_n^+` and :math:`T_n^-`, that share this edge. The function has a constant normal component across the shared edge, ensuring continuity of current, and a constant divergence on each triangle.

..
   DEBUGGING BLOCK: A minimal TikZ picture to test the rendering pipeline.
   If this works, the problem is in the RWG figure's syntax.
   If this fails, the problem is in conf.py or the LaTeX installation.

.. _fig:rwg:

.. tikz:: The Rao-Wilton-Glisson (RWG) basis function :math:`\mathbf{f}_n` on a pair of triangles.
   
   \begin{tikzpicture}[scale=1.2, >=Latex]
     % Define vertices
     \coordinate (v0) at (0,0);
     \coordinate (v1) at (4,0);
     \coordinate (v2) at (2,2.5);
     \coordinate (v3) at (2,-2.5);
     % Draw triangles and fill
     \fill[blue!10] (v0) -- (v1) -- (v2) -- cycle;
     \fill[red!10]  (v0) -- (v1) -- (v3) -- cycle;
     \draw[thick] (v0) -- (v1) -- (v2) -- cycle;
     \draw[thick] (v0) -- (v1) -- (v3) -- cycle;
     % Labels
     \node at (1.5, 1.3) {$T_n^+$};
     \node at (1.5, -1.3) {$T_n^-$};
     \node[below] at (2,0) {edge $n$ (length $l_n$)};
     \node[above] at (v2) {free vertex $\mathbf{r}_n^+$};
     \node[below] at (v3) {free vertex $\mathbf{r}_n^-$};
     % Basis function vectors
     \draw[->, very thick, blue] (2, 2.0) -- (2, 0.5) node[midway, right] {$\mathbf{f}_n(\mathbf{r})$};
     \draw[->, very thick, red] (2, -0.5) -- (2, -2.0) node[midway, right] {$\mathbf{f}_n(\mathbf{r})$};
   \end{tikzpicture}

The RWG basis function associated with edge :math:`n` is defined as:

.. math::
   :label: eq:rwgbasis

   \mathbf{f}_n(\mathbf{r}) =
   \begin{cases}
      \frac{l_n}{2 A_n^+} (\mathbf{r} - \mathbf{r}_n^+), & \mathbf{r} \in T_n^+, \\
      \frac{l_n}{2 A_n^-} (\mathbf{r}_n^- - \mathbf{r}), & \mathbf{r} \in T_n^-, \\
      0, & \text{elsewhere},
   \end{cases}

where :math:`l_n` is the length of edge :math:`n`, :math:`A_n^+` and :math:`A_n^-` are the areas of triangles :math:`T_n^+` and :math:`T_n^-`, respectively, and :math:`\mathbf{r}_n^+` and :math:`\mathbf{r}_n^-` are the position vectors of the free vertices of these triangles (the vertices not on edge :math:`n`).  


Edge Indexing Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~


There are two forms of edge indexing used in the mesh:

1. **Global indexing**: Each edge is assigned a unique global index :math:`e`.
2. **Local indexing**: Each edge is referenced by a pair :math:`(n, i)`, where :math:`n` is the face (triangle) index and :math:`i` is the local edge index within that face (:math:`i = 0, 1, 2`).

The mapping between these is not a direct formula, but rather a lookup in a mapping array:

.. math::
   :label: eq:faceedges-mapping

   e = \texttt{faceEdges}[3n + i]

where :math:`e` is the global edge index for the local edge :math:`i` of face :math:`n`. In Python, this is typically accessed as ``face_edges[n, i]`` or ``face_edges[n * 3 + i]`` if flattened. In CUDA/C++, the access is ``faceEdges[3 * n + i]``.

This mapping array is constructed during mesh preprocessing by :py:meth:`sie_mom.mesh.spherical_surface.SphericalSurfaceMesh._build_edges_and_mapping` and then exposed through the :py:attr:`sie_mom.mesh.spherical_surface.SphericalSurfaceMesh.faces_edges` attribute, ensuring that each local edge in each face is mapped to the correct global edge index for matrix assembly. The :math:`3n+i` notation (or :math:`3*n+i`) is simply the way to flatten the 2D (face, local edge) index into a 1D array index; the value at that position is the actual global edge index.

**Summary:**

- The value :math:`e` is not :math:`3n+i` itself, but the value stored in the mapping array at that position.
- This mapping is used in both the Python mesh code and the CUDA kernel (see :cpp:func:`::kernel_L_operator`) to ensure consistent assembly of the global MoM matrix.


RWG half–basis on a face
~~~~~~~~~~~~~~~~~~~~~~~~~~


**Notation policy.** :math:`\mathbf{f}` denotes the *RWG basis function (vector field)*.
Plain :math:`f_n` denotes the *triangular geometric face* number :math:`n`.

.. admonition:: Definition — half RWG on face :math:`f_n`

   .. math::
      :label: eq:rwg_half_basis

      \mathbf{f}_{n i}(\mathbf r)
      \;=\;
      \frac{1}{2\,A_n}\;e_{ni}\;\big(\mathbf{r} - \mathbf{r}_{\overline{\partial e_{ni}}}\big),
      \qquad
      \mathbf{r} \in f_n.

* :math:`A_n` — area of face :math:`f_n` (real, positive).

* :math:`e_{ni}` — the **oriented** local edge length of face :math:`f_n` with local index :math:`i`
     (orientation sign already included).

* :math:`\partial e_{ni}` — the **boundary** of the edge (the two endpoints):

   .. math:: \partial e_{ni} = \{v_a, v_b\}.

* :math:`\overline{\partial e_{ni}}` — the **complement** of :math:`\partial e_{ni}` inside the face vertex set, i.e. the **free vertex** of :math:`e_{ni}` in :math:`f_n`:

   .. math::
        \overline{\partial e_{ni}} \;=\; \{v_0,v_1,v_2\}\setminus\{v_a,v_b\} \;=\; \{v_{\text{free}}\}.

* :math:`\mathbf{r}` — position vector on the face :math:`f_n` domain.



.. admonition:: Boundary operators on a triangle :math:`f_n = [v_0,v_1,v_2]`

   **Face boundary.** (sum of its three oriented local edges)

   .. math::
      \partial f_n \;=\; e_{n0} \;+\; e_{n1} \;+\; e_{n2}.

   **Edge as an oriented pair.** With the global “min→max” canonical orientation for edges:

   .. math::
      e \;=\; \pm [v_1, v_2],
      \qquad
      \text{sign} = \begin{cases}
        +1, & \text{if } v_1 > v_2, \\
        -1, & \text{if } v_1 < v_2.
      \end{cases}

   .. admonition:: Note on Sign Convention
      :class: tip

      The absolute choice of sign is arbitrary. What is physically and mathematically crucial is that for any given global edge, the two triangles that share it (:math:`T_n^+` and :math:`T_n^-`) traverse that edge in opposite directions.

      Because all faces are oriented consistently (e.g., counter-clockwise), if one face traverses the edge as :math:`(v_1, v_2)`, the adjacent face must traverse it as :math:`(v_2, v_1)`. The sign convention above therefore guarantees that one face gets a :math:`+1` sign and the other gets a :math:`-1` for their shared local edge. This ensures the normal component of the current is continuous across the edge, which is the defining property of the RWG basis function.

      This logic is implemented in the :cpp:func:`::getEdgeInfo` helper function.

   **Edge boundary (endpoints).**

   .. math::
      \partial e \;=\; \{v_1, v_2\}.

   **Free vertex in** :math:`f_n` **(complement of the edge endpoints).**

   .. math::
      \overline{\partial e}
      \;=\;
      \{v_0, v_1, v_2\}\setminus \{v_1, v_2\}
      \;=\; \{v_0\}
      \;=\; \{v_{\text{free}}\}.



Testing of the :math:`\mathcal{L}` operator
============================================


The rule for forming the matrix corresponding to operator testing (matrix assembly) is:

.. math::
   \mathcal{TL}_{mj} = \int_{f_m} \mathbf{f}_{mj} \cdot \mathcal{L}_k[\mathbf{X}] \, dS


where the vector field :math:`\mathbf{X}` is expanded as:

.. math::
   \mathbf{X} = \sum_{i=0}^2 \alpha_{ni} \mathbf{f}_{ni} = \sum_{e=0}^2 \alpha_{e} \mathbf{f}_{e}



with the correspondence :math:`e = \mathtt{faceEdges}[3n + i]`.

We now write the structure of the matrix element for the operator :math:`\mathcal{L}` in terms of the testing and source basis functions:

.. math::
   \mathcal{TL}_{mj,ni} = \int_{f_m} \mathbf{f}_{mj} \cdot \mathcal{L}_{f_n}(k, \mathbf{f}_{ni}) \, dS

where:

- The integration is over the observation triangle (face) :math:`f_m`.
- The testing basis function is :math:`\mathbf{f}_{mj}` (local edge :math:`j` of face :math:`m`).
- The source basis function is :math:`\mathbf{f}_{ni}` (local edge :math:`i` of face :math:`n`).
- :math:`\mathcal{L}_{S_n}` denotes the action of the operator with integration over the source triangle :math:`S_n`.

Expanding the operator, we have:

.. math::
   \mathcal{TL}_{mj,ni} = \int_{f_m} \mathbf{f}_{mj} \cdot \left[ \int_{f_n} \mathbf{f}_{ni} G - \frac{1}{k^2} (\nabla'_S \cdot \mathbf{f}_{ni}) \nabla' G \, dS' \right] dS

where:

- :math:`G` is the Green's function.
- :math:`\nabla'_S` denotes the surface divergence with respect to the source coordinates :math:`\mathbf{r}'`.
- The inner integral is over the source triangle :math:`f_n`.

.. _sie_L_testing_single_pair:


**Notation.** Bold symbols are vector basis functions (e.g. :math:`\mathbf f_{mj}`, :math:`\mathbf f_{ni}`).
Plain :math:`f_m,\ f_n` denote the triangular **domains** under the integrals.
The boundary curves are :math:`\partial f_m,\ \partial f_n` with outward co-normals
:math:`\boldsymbol{\nu},\ \boldsymbol{\nu}'` (tangent to each triangle).

Start from Stratton–Chu’s :math:`\mathcal{L}` tested on :math:`f_m` against a single source triangle :math:`f_n` :cite:p:`stratton1939diffraction`:

.. math::

   \mathcal{TL}_{mj,ni}
   = \int_{f_m} \mathbf f_{mj}(\mathbf r)\cdot
     \left[
       \int_{f_n}
         \left(
           \mathbf f_{ni}(\mathbf r')\, G(\mathbf r,\mathbf r')
           -\frac{1}{k^2}\, (\nabla'_S\!\cdot \mathbf f_{ni}(\mathbf r'))\ \nabla' G(\mathbf r,\mathbf r')
         \right)\, dS'
     \right] dS.

Split into
:math:`\mathrm{Term\ 1}+\mathrm{Term\ 2}`:

.. math::

   \mathrm{Term\ 1}
   = \iint_{f_m \times f_n} \big(\mathbf f_{mj} \cdot \mathbf f_{ni}\big)\, G \, dS\, dS',
   \qquad
   \mathrm{Term\ 2}
   = -\frac{1}{k^2}\iint_{f_m\times f_n}
     (\nabla'_S\!\cdot \mathbf f_{ni})\ \big(\mathbf f_{mj}\cdot \nabla' G\big)\, dS'\, dS.

**Distributional split on** :math:`f_n`.
Extend :math:`\mathbf f_{ni}` by zero outside :math:`f_n` and use the distributional identity

:math:`\nabla'_S\!\cdot(\chi_{f_n}\mathbf f_{ni}) = (\nabla'_S\!\cdot \mathbf f_{ni})\,\chi_{f_n} + (\mathbf f_{ni}\!\cdot \boldsymbol{\nu}')\,\delta_{\partial f_n}`,

to write :math:`\mathrm{Term\ 2}=\mathrm{Term\ 2.1}+\mathrm{Term\ 2.2}` with

.. math::

   \mathrm{Term\ 2.1}
   = -\frac{1}{k^2} \int_{f_m} \oint_{\partial f_n}
     \big(\mathbf f_{ni}\!\cdot \boldsymbol{\nu}'\big)\, \big(\mathbf f_{mj}\!\cdot \nabla' G\big)\, dl'\, dS,

.. math::

   \mathrm{Term\ 2.2}
   = -\frac{1}{k^2} \iint_{f_m\times f_n}
     (\nabla'_S\!\cdot \mathbf f_{ni})\ \big(\mathbf f_{mj}\!\cdot \nabla' G\big)\, dS'\, dS.

Handle :math:`\mathrm{Term\ 2.1}` by using :math:`\nabla' G=-\nabla G` and surface integration by parts on :math:`f_m`:

.. math::

   \int_{f_m} \mathbf f_{mj}\!\cdot \nabla G\, dS
   = \oint_{\partial f_m} G\, (\mathbf f_{mj}\!\cdot \boldsymbol{\nu})\, dl
     - \int_{f_m} G\, (\nabla_S\!\cdot \mathbf f_{mj})\, dS,

which yields

.. math::

   \mathrm{Term\ 2.1}
   = \frac{1}{k^2}
     \oint_{\partial f_n}
     \left[
       \ \oint_{\partial f_m} (\mathbf f_{ni}\!\cdot \boldsymbol{\nu}')
           (\mathbf f_{mj}\!\cdot \boldsymbol{\nu})\, G\, dl
       \;-\;
       \int_{f_m} (\mathbf f_{ni}\!\cdot \boldsymbol{\nu}')
           (\nabla_S\!\cdot \mathbf f_{mj})\, G\, dS
     \right] dl'.

Handle :math:`\mathrm{Term\ 2.2}` by surface integration by parts on :math:`f_n` (no boundary term here because it was extracted in :math:`\mathrm{Term\ 2.1}`), then the surface product rule on :math:`f_m`:

.. math::

   \mathrm{Term\ 2.2}
   = \frac{1}{k^2} \iint_{f_m\times f_n}
     \mathbf f_{ni}\!\cdot \nabla'_S\!\big(\mathbf f_{mj}\!\cdot \nabla' G\big)\, dS'\, dS

.. math::

   \hspace{3.7em}
   = -\frac{1}{k^2}
     \left[
       \ \oint_{\partial f_m}\!\int_{f_n}
         (\mathbf f_{mj}\!\cdot \boldsymbol{\nu})\, \mathbf f_{ni}\!\cdot \nabla G \ dS'\, dl
       \;-\;
       \int_{f_m}\!\int_{f_n}
         (\nabla_S\!\cdot \mathbf f_{mj})\, \mathbf f_{ni}\!\cdot \nabla G \ dS'\, dS
     \right].

Finally use :math:`\mathbf f_{ni}\!\cdot \nabla G = - \mathbf f_{ni}\!\cdot \nabla' G` and integrate by parts on :math:`f_n` for the remaining surface term.

**Final expression (single testing/source triangle).**

.. math::
   :label: eq:TL_mjni_single_pair

   \begin{aligned}
   \mathcal{TL}_{mj,ni}
   &= \iint_{f_m\times f_n} \big(\mathbf f_{mj}\!\cdot \mathbf f_{ni}\big)\, G \; dS\, dS'
   \\[2pt]
   &\quad - \frac{1}{k^2}\iint_{f_m\times f_n}
           \big(\nabla_S\!\cdot \mathbf f_{mj}\big)\,
           \big(\nabla'_S\!\cdot \mathbf f_{ni}\big)\, G \; dS\, dS'
   \\[2pt]
   &\quad + \frac{1}{k^2}\oint_{\partial f_m}\!\int_{f_n}
           (\mathbf f_{mj}\!\cdot \boldsymbol{\nu})\,
           (\nabla'_S\!\cdot \mathbf f_{ni})\, G \; dS' \, dl
   \\[2pt]
   &\quad + \frac{1}{k^2}\oint_{\partial f_n}\!\int_{f_m}
           (\mathbf f_{ni}\!\cdot \boldsymbol{\nu}')\,
           (\nabla_S\!\cdot \mathbf f_{mj})\, G \; dS \, dl'
   \\[2pt]
   &\quad - \frac{1}{k^2}\oint_{\partial f_m}\!\oint_{\partial f_n}
           (\mathbf f_{mj}\!\cdot \boldsymbol{\nu})\,
           (\mathbf f_{ni}\!\cdot \boldsymbol{\nu}')\, G \; dl \, dl'.
   \end{aligned}

.. note::

   On a closed surface, the edge integrals cancel in global assembly (interior edges are traversed twice with opposite co-normals).



Global assembly of :math:`\mathcal{TL}` (face-by-face)
------------------------------------------------------


**Prerequisite.** The single testing/source pair entry :math:`\mathcal{TL}_{mj,ni}` is given in :eq:`eq:TL_mjni_single_pair`.

**Mapping.** Let :math:`\texttt{faceEdges}\in\{0,\dots,n_E\!-\!1\}^{3n_F}` map each face and its local edge
to a global edge id:
:math:`\texttt{faceEdges}[3m+j] = e`, for face :math:`m` and local edge :math:`j\in\{0,1,2\}`.

Define the indicator
:math:`\delta_{e,\,p}=\begin{cases}1,&\text{if }\texttt{faceEdges}[p]=e,\\[2pt]0,&\text{otherwise.}\end{cases}`.

**Assembly law (index-light).**

.. math::
   :label: eq:TL_global_assembly

   \mathcal{TL}_{e_1,e_2}
   \;=\;
   \sum_{m,j}\ \sum_{n,i}\;
   \delta_{e_1,\,3m+j}\;\delta_{e_2,\,3n+i}\; \mathcal{TL}_{mj,ni}.

Here the sums run over all faces :math:`m,n` and local edges :math:`j,i\in\{0,1,2\}`.
This is the theoretical statement mirrored by accumulating into the matrix entry
indexed by :math:`(\texttt{faceEdges}[3m{+}j],\,\texttt{faceEdges}[3n{+}i])`.

---

Why exactly **four** contributions for closed surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a closed triangulation, each interior global edge :math:`e` belongs to exactly two faces.
Let :math:`\star(e)=\{f^{+}(e),\,f^{-}(e)\}` be that pair (the “star” of the edge).
For :math:`m\in\star(e)` there is a unique local index :math:`j(e\!\mid m)` with
:math:`\texttt{faceEdges}[3m+j(e\!\mid m)]=e`. Likewise
:math:`i(e\!\mid n)` for :math:`n\in\star(e)`.

Then :eq:`eq:TL_global_assembly` collapses to exactly four face pairs:

.. math::
   :label: eq:TL_four_terms

   \mathcal{TL}_{e_1,e_2}
   \;=\;
   \sum_{m\in\star(e_1)}\ \sum_{n\in\star(e_2)}
   \mathcal{TL}_{\,m\,j(e_1\mid m),\; n\,i(e_2\mid n)}.

This is the face-by-face analogue of the usual :math:`T_m^{\pm}\times T_n^{\pm}` split,
without switching to an edge-by-edge assembly viewpoint.

---

Cancellation of boundary terms on closed :math:`S`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write the single-pair kernel as

.. math::

   \mathcal{TL}_{mj,ni}
   \;=\; A_{mj,ni}
     \;-\;\frac{1}{k^2} B_{mj,ni}
     \;+\;\frac{1}{k^2} C_{mj,ni}^{(\partial f_m)}
     \;+\;\frac{1}{k^2} D_{mj,ni}^{(\partial f_n)}
     \;-\;\frac{1}{k^2} E_{mj,ni}^{(\partial f_m\times\partial f_n)},

where :math:`A,B` are the surface–surface terms and :math:`C,D,E` are the line/surface and line/line
terms from :eq:`eq:TL_mjni_single_pair`.

On a closed surface:

* along the common interior edge of :math:`f^{+}(e_1)` and :math:`f^{-}(e_1)`,
  the co-normal flips and the RWG half-sign flips, hence
  :math:`C_{f^{+}j,\,ni}^{(\partial f^{+})}+C_{f^{-}j,\,ni}^{(\partial f^{-})}=0`;
  similarly the :math:`D` terms cancel around :math:`e_2`.

* the four double-edge terms cancel pairwise:
  :math:`E_{f^{+}j,\,f^{+}i}+E_{f^{+}j,\,f^{-}i}+E_{f^{-}j,\,f^{+}i}+E_{f^{-}j,\,f^{-}i}=0`.

Therefore only the surface–surface parts survive:

.. math::
   :label: eq:TL_no_boundary

   \mathcal{TL}_{e_1,e_2}
   \;=\;
   \sum_{m\in\star(e_1)}\ \sum_{n\in\star(e_2)}
   \left(\,A_{mj,ni}\;-\;\frac{1}{k^2}B_{mj,ni}\right),
   \qquad \text{(closed } S\text{)}.

.. note::

   If :math:`S` has a boundary, the corresponding edge terms remain on the physical boundary.
   In global assembly, **interior** edges still cancel because they are traversed twice with opposite co-normals and opposite RWG half-signs.


.. tikz:: Two faces sharing a common edge.
   
   \begin{tikzpicture}[
     scale=1.0,
     every node/.style={font=\small},
     face/.style={line width=0.9pt},
     shared/.style={line width=1.4pt},
     % Define a reusable 'pic' for an edge neighborhood
     pics/edge neighborhood/.style 2 args={
       code={
         % #1: edge label (e.g., e_1)
         % #2: local index label (e.g., j)
         \coordinate (L) at (-1.50, 0.00);
         \coordinate (T) at ( 0.00, 1.20);
         \coordinate (R) at ( 1.50, 0.00);
         \coordinate (B) at ( 0.00,-1.20);
         % Draw the two faces sharing the central edge
         \draw[face] (L)--(T)--(B)--cycle;
         \draw[face] (R)--(T)--(B)--cycle;
         \draw[shared] (T)--(B); % Shared edge
         % Face labels
         \node[anchor=west] at ($(R)!0.5!(T)$) {$f^{+}$};
         \node[anchor=east] at ($(L)!0.33!(T)$) {$f^{-}$};
         % Local edge markers
         \node[anchor=west] at ( 0.12, 0.10) {$#2(+)$};
         \node[anchor=east] at (-0.12, 0.10) {$#2(-)$};
         % Co-normal arrows (opposite directions)
         \draw[-{Stealth[length=2.6mm]},line width=0.9pt] ( 0.12,-0.25) -- +(0, 0.55);
         \draw[-{Stealth[length=2.6mm]},line width=0.9pt] (-0.12, 0.25) -- +(0,-0.55);
         % Edge label below
         \node at (0,-1.55) {$#1$};
       }
     }
   ]
     % Draw the two neighborhoods using the 'pic'
     \pic at (-3.0, 0.0) {edge neighborhood={e_1}{j}};
     \pic at ( 3.0, 0.0) {edge neighborhood={e_2}{i}};
 
     % Note about the four face pairs
     \node at (0,0) {$\times$ 4 face pairs};
 
     % Note about cancellation of line terms
     \node[draw, rounded corners=3pt, align=left, anchor=west] at (5.0, 0.8) {
       co-normals opposite \\ $\Rightarrow$ line terms cancel
     };
   \end{tikzpicture}


From single-pair Sourface Integral to Quadrature Dunavant rules
-------------------------------------------------------------------------------

**Starting point.** The tested single-pair entry :math:`TL_{mj,ni}` is given by :eq:`eq:TL_mjni_single_pair`.
In this section we **drop the line terms** (closed-surface assembly will cancel them) and keep only the
two **double-surface** terms.

Step 1 — Substitute half-RWG and use constant divergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Half-RWG on a face (our compact notation):

.. math::

   \mathbf f_{mj}(\mathbf r) = \frac{e_{mj}}{2A_m}\,\big(\mathbf r-\mathbf r_{\overline{\partial e_{mj}}}\big),
   \qquad
   \mathbf f_{ni}(\mathbf r') = \frac{e_{ni}}{2A_n}\,\big(\mathbf r'-\mathbf r_{\overline{\partial e_{ni}}}\big),

with constant surface divergences

.. math::

   \nabla_S\!\cdot \mathbf f_{mj} = \frac{e_{mj}}{A_m}, \qquad
   \nabla'_S\!\cdot \mathbf f_{ni} = \frac{e_{ni}}{A_n}.

Keeping only the area–area parts of :eq:`eq:TL_mjni_single_pair` gives

.. math::
   :label: eq:TL_mjni_surface_only

   \mathcal{TL}_{mj,ni}^{(\mathrm{surf})}
   \;=\;
   \iint_{f_m\times f_n} \big(\mathbf f_{mj}\!\cdot \mathbf f_{ni}\big)\, G\, dS\, dS'
   \;-\;
   \frac{1}{k^2}
   \iint_{f_m\times f_n} \big(\nabla_S\!\cdot \mathbf f_{mj}\big)\big(\nabla'_S\!\cdot \mathbf f_{ni}\big)\, G\, dS\, dS'.

Introduce
:math:`\boldsymbol\rho_{mj}(\mathbf r)=\mathbf r-\mathbf r_{\overline{\partial e_{mj}}}`,
:math:`\boldsymbol\rho_{ni}(\mathbf r')=\mathbf r'-\mathbf r_{\overline{\partial e_{ni}}}`,
and the factor
:math:`C_{mj,ni}=\dfrac{e_{mj}e_{ni}}{A_mA_n}`.
Then

.. math::
   :label: eq:TL_mjni_surface_RWG

   \mathcal{TL}_{mj,ni}^{(\mathrm{surf})}
   \;=\;
   C_{mj,ni}\Bigg[
   \frac{1}{4}\iint_{f_m\times f_n}
   \big(\boldsymbol\rho_{mj}(\mathbf r)\cdot \boldsymbol\rho_{ni}(\mathbf r')\big)\,G(\mathbf r,\mathbf r')\, dS\, dS'
   \;-\;
   \frac{1}{k^2}\iint_{f_m\times f_n} G(\mathbf r,\mathbf r')\, dS\, dS'
   \Bigg].

Step 2 — Dunavant quadrature on each triangle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\{(\alpha_p,\beta_p,\gamma_p),\,w_p\}_{p=1}^{Q_m}` be a Dunavant rule on :math:`f_m`
Let :math:`\{(\alpha_p,\beta_p,\gamma_p),\,w_p\}_{p=1}^{Q_m}` be a Dunavant rule on :math:`f_m` :cite:p:`dunavant1985quadrature`
(with :math:`\sum_p w_p=1`) and points
:math:`\mathbf r_{m,p}=\alpha_p \mathbf r_{v_{m0}}+\beta_p \mathbf r_{v_{m1}}+\gamma_p \mathbf r_{v_{m2}}`.
Similarly on :math:`f_n` we use :math:`\{(\alpha'_q,\beta'_q,\gamma'_q),\,w'_q\}_{q=1}^{Q_n}` and
:math:`\mathbf r'_{n,q}`.

The tensor-product rule gives

.. math::

   \iint_{f_m\times f_n}\Phi(\mathbf r,\mathbf r')\,dS\,dS'
   \;\approx\;
   A_mA_n \sum_{p=1}^{Q_m}\sum_{q=1}^{Q_n} w_p\,w'_q\,
   \Phi\!\big(\mathbf r_{m,p},\,\mathbf r'_{n,q}\big).

Using this in :eq:`eq:TL_mjni_surface_RWG` cancels :math:`A_mA_n` with
:math:`C_{mj,ni}=e_{mj}e_{ni}/(A_mA_n)` and yields the **double-sum** form

.. math::
   :label: eq:TL_mjni_dunavant

   \begin{aligned}
   \mathcal{TL}_{mj,ni}^{(\mathrm{surf})}
   &\approx
   e_{mj}\,e_{ni}
   \sum_{p=1}^{Q_m}\sum_{q=1}^{Q_n} w_p\,w'_q\,
   \Bigg[
     \frac{1}{4}\,
     \Big(\boldsymbol\rho_{mj}(\mathbf r_{m,p})\cdot \boldsymbol\rho_{ni}(\mathbf r'_{n,q})\Big)
     \;-\;\frac{1}{k^2}
   \Bigg]\;
   G\!\big(\mathbf r_{m,p},\,\mathbf r'_{n,q}\big).
   \end{aligned}

Here
:math:`G(\mathbf r,\mathbf r')=\dfrac{e^{-jk\|\mathbf r-\mathbf r'\|}}{4\pi \|\mathbf r-\mathbf r'\|}`.

**Implementation:** |tl-tk-quadrature| in
``include/sie_mom/cuda/dunavant_quadrature.cuh`` evaluates this TL sum and
the TK sum below using the full kernels (``is_singular=false``).

Compact “modifier x integration Kernel” (M*D) form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For later reuse, define the Green-kernel samples and the geometric modifier at the
quadrature nodes:

.. math::

   D_{pq} := G\!\big(\mathbf r_{m,p},\,\mathbf r'_{n,q}\big),\qquad
   M^{(mj,ni)}_{pq} := \frac{1}{4}\Big(\boldsymbol\rho_{mj}(\mathbf r_{m,p})\cdot
                                        \boldsymbol\rho_{ni}(\mathbf r'_{n,q})\Big)
                       - \frac{1}{k^2}.

Then :eq:`eq:TL_mjni_dunavant` can be written as

.. math::
   :label: eq:TL_kernel_modifier

   \mathcal{TL}_{mj,ni}^{(\mathrm{surf})}
   \;\approx\;
   e_{mj}\,e_{ni}\;
   \sum_{p=1}^{Q_m}\sum_{q=1}^{Q_n} w_p\,w'_q\;
   M^{(mj,ni)}_{pq}\; D_{pq}.

.. note::

   The modifier :math:`M^{(mj,ni)}_{pq}` depends on the specific half-bases
   only through the vectors :math:`\boldsymbol\rho_{mj}(\mathbf r_{m,p})`
   and :math:`\boldsymbol\rho_{ni}(\mathbf r'_{n,q})` and the wavenumber :math:`k`.
   No surface areas appear here—they canceled in the quadrature step.

.. tip::

   For the initial implementation we adopt a **single Dunavant order** applied **uniformly to all faces**,
   near or far. For example, choose the 3-point rule (order 2) or the 7-point rule (order 5), and use the
   same nodes/weights on every triangle:

   .. math::

      \{(\alpha_p,\beta_p,\gamma_p),\,w_p\}_{p=1}^{Q}, \qquad
      \sum_{p=1}^{Q} w_p = 1.

   With this choice we set :math:`Q_m = Q_n = Q`, use the same weight set on both faces, and the
   double-sum form :eq:`eq:TL_kernel_modifier` becomes

   .. math::

      \mathcal{TL}_{mj,ni}^{(\mathrm{surf})}
      \;\approx\;
      e_{mj}\,e_{ni}
      \sum_{p=1}^{Q}\sum_{q=1}^{Q} w_p\,w_q\;
      M^{(mj,ni)}_{pq}\; D_{pq}.

   This “same-order-everywhere” policy is a clean starting point and keeps the theory simple.
   Later, if tighter accuracy is needed for touching/near/self interactions, you can swap in
   higher-order rules or specialized schemes without changing the algebra above.


.. _sie_K_pair_to_dunavant:

Tested :math:`\mathcal{K}` operator (single testing/source pair) and Dunavant form
===================================================================================

**Operator.** For region :math:`i` with wavenumber :math:`k_i`,

.. math::

   \mathcal{K}_{k_i}\{\mathbf X\}(\mathbf r)
   = \mathrm{p.v.}\!\int_S \mathbf X(\mathbf r') \times \nabla' G_i(\mathbf r,\mathbf r')\, dS',
   \quad
   \nabla' G_i(\mathbf r,\mathbf r')=\Big(jk_i+\tfrac{1}{R}\Big)\,G_i\,\hat{\mathbf R},

with :math:`\mathbf R=\mathbf r-\mathbf r'`, :math:`R=\|\mathbf R\|`, :math:`\hat{\mathbf R}=\mathbf R/R`,
and :math:`G_i=e^{-jk_i R}/(4\pi R)`.

**Tested single-pair entry.**

.. math::

   TK_{mj,ni}
   :=\int_{f_m}\mathbf f_{mj}(\mathbf r)\cdot \Big[\int_{f_n} \mathbf f_{ni}(\mathbf r')\times \nabla' G_i(\mathbf r,\mathbf r')\, dS'\Big]\, dS.


Using the triple product identity
:math:`\mathbf a\cdot(\mathbf b\times\mathbf c)=(\mathbf a\times\mathbf b)\cdot \mathbf c`,

.. math::

   \boxed{\
   TK_{mj,ni}
   =\iint_{f_m\times f_n}\big(\mathbf f_{mj}(\mathbf r)\times \mathbf f_{ni}(\mathbf r')\big)
   \cdot \nabla' G_i(\mathbf r,\mathbf r')\, dS\, dS'.\
   }


Step 1 — Substitute half-RWG (surface–surface form)
---------------------------------------------------

Half-RWG in compact notation (no line terms kept here):

.. math::

   \mathbf f_{mj}(\mathbf r)=\frac{e_{mj}}{2A_m}\,\boldsymbol\rho_{mj}(\mathbf r),\qquad
   \mathbf f_{ni}(\mathbf r')=\frac{e_{ni}}{2A_n}\,\boldsymbol\rho_{ni}(\mathbf r'),

with :math:`\boldsymbol\rho_{mj}(\mathbf r)=\mathbf r-\mathbf r_{\overline{\partial e_{mj}}}` and analogously for :math:`\boldsymbol\rho_{ni}`.
Then

.. math::
   :label: eq:TK_mjni_surface_surface

   \boxed{
   TK_{mj,ni}
   =\frac{e_{mj}e_{ni}}{4A_mA_n}\,
   \iint_{f_m\times f_n}
   \Big(jk_i+\tfrac{1}{R}\Big)\,G_i(\mathbf r,\mathbf r')\;
   \big(\boldsymbol\rho_{mj}(\mathbf r)\times \boldsymbol\rho_{ni}(\mathbf r')\big)\cdot \hat{\mathbf R}\; dS\, dS'.
   }

(No distributional edge term is needed here; see the curiosity box below.)

Step 2 — Dunavant quadrature (double sum)
-----------------------------------------

Let :math:`\{(\alpha_p,\beta_p,\gamma_p),w_p\}_{p=1}^{Q_m}` be a Dunavant rule on :math:`f_m`
and :math:`\{(\alpha'_q,\beta'_q,\gamma'_q),w'_q\}_{q=1}^{Q_n}` on :math:`f_n`,
with points :math:`\mathbf r_{m,p}`, :math:`\mathbf r'_{n,q}` and :math:`\sum_p w_p=\sum_q w'_q=1`.
Applying the tensor-product rule cancels :math:`A_mA_n` with the prefactor and yields

.. math::
   :label: eq:TK_mjni_dunavant

   \begin{aligned}
   TK_{mj,ni}
   &\approx
   e_{mj}\,e_{ni}
   \sum_{p=1}^{Q_m}\sum_{q=1}^{Q_n} w_p\,w'_q\;
   \underbrace{\tfrac14\big(\boldsymbol\rho_{mj}(\mathbf r_{m,p})\times \boldsymbol\rho_{ni}(\mathbf r'_{n,q})\big)}_{\displaystyle M^{(mj,ni)}_{pq}}
   \\[-2pt]
   &\hspace{8.4em}\dot\;
   \underbrace{\Big(jk_i+\tfrac{1}{R_{pq}}\Big)\,G_i(\mathbf r_{m,p},\mathbf r'_{n,q}) \hat{\mathbf R}_{pq}}_{\displaystyle D_{pq}},
   \end{aligned}

where :math:`R_{pq}=\|\mathbf r_{m,p}-\mathbf r'_{n,q}\|`, :math:`\hat{\mathbf R}_{pq}=(\mathbf r_{m,p}-\mathbf r'_{n,q})/R_{pq}`.

**Implementation:** the TK outputs of |tl-tk-quadrature| evaluate this sum.

**Compact “modifier x integration Kernel” (M*D) form**
Define :math:`\mathbf{D_{pq}}=(jk_i+1/R_{pq})\,G_i(\mathbf r_{m,p},\mathbf r'_{n,q})\hat{\mathbf R}_{pq}`,
:math:`\mathbf{M^{(mj,ni)}_{pq}}=\tfrac14(\boldsymbol\rho_{mj}\times \boldsymbol\rho_{ni})`.
Then

.. math::
   :label: eq:TK_kernel_modifier

   TK_{mj,ni}
   \;\approx\;
   e_{mj}\,e_{ni}\sum_{p=1}^{Q_m}\sum_{q=1}^{Q_n} w_p\,w'_q\;
   \mathbf{M^{(mj,ni)}_{pq}} \cdot \mathbf{D_{pq}}.

.. admonition:: Curiosity — why no edge :math:`\delta` term here?

   For :math:`\mathcal L` we pushed a surface divergence onto the RWG and picked up
   a distributional contribution supported on :math:`\partial f_n`.
   For :math:`\mathcal K` we keep :math:`\nabla' G_i` **on the kernel** and use it explicitly,
   so no edge :math:`\delta` is introduced in the surface–surface form. If one integrates
   by parts instead, line terms appear but cancel in global assembly on closed :math:`S`
   (opposite co-normals and RWG sign flip), leading back to the same net result.

.. note::

   The usual jump term :math:`\tfrac12\,\hat{\mathbf n}\times \mathbf M` is handled
   as a separate (simple, 1-D) contribution in the implementation; it is **not** included
   in :eq:`eq:TK_mjni_dunavant`–:eq:`eq:TK_kernel_modifier`.

.. tip::

   **Self-pair vanishing (planar triangles).**
   If :math:`m=n` (both points on the same flat triangle), then
   :math:`\boldsymbol\rho_{mj}(\mathbf r)` and :math:`\boldsymbol\rho_{ni}(\mathbf r')`
   are tangential, hence their cross product is **normal** to the triangle, while
   :math:`\hat{\mathbf R}_{pq}` lies in the triangle plane (since :math:`\mathbf r-\mathbf r'`
   is tangential). Therefore

   .. math::
      (\boldsymbol\rho_{mj}\times \boldsymbol\rho_{ni})\cdot \hat{\mathbf R}_{pq}=0
      \quad\Rightarrow\quad
      TK_{mj,ni}^{\text{(surface–surface)}}=0 \ \text{when } m=n.

   In other words, the **planar self-term** of the double-layer–type surface integral
   vanishes; only the separate jump term contributes for :math:`m=n`.


Testing of the :math:`\mathcal{I}` operator
============================================

The matrix elements for the identity operator :math:`\mathcal{I}` are given by:

.. math::
   :label: eq:identity_matrix_element_reprise

   \mathcal{I}_{mj,ni} = \int_{f_m} \mathbf{f}_{mj} \cdot (\hat{\mathbf{n}} \times \mathbf{f}_{ni}) \, dS

This integral is non-zero only when the domains of the basis functions overlap. For the local matrix assembly, we consider the case where the test face :math:`m` and source face :math:`n` are the same, i.e., :math:`m=n`. In this scenario, both basis functions are defined on the same triangle :math:`f_m`.

Step 1 — Substitute half-RWG basis functions
---------------------------------------------

Substituting the definitions of the half-RWG basis functions on face :math:`f_m` (where :math:`A_m = A_n`):

.. math::

   \mathbf{f}_{mj}(\mathbf{r}) = \frac{e_{mj}}{2A_m} \boldsymbol{\rho}_{mj}(\mathbf{r})
   \qquad \text{and} \qquad
   \mathbf{f}_{ni}(\mathbf{r}) = \frac{e_{ni}}{2A_m} \boldsymbol{\rho}_{ni}(\mathbf{r})

into the integral gives:

.. math::
   :label: eq:I_mjni_substituted

   \mathcal{I}_{mj,ni} = \frac{e_{mj}e_{ni}}{4A_m^2} \int_{f_m} \boldsymbol{\rho}_{mj}(\mathbf{r}) \cdot (\hat{\mathbf{n}} \times \boldsymbol{\rho}_{ni}(\mathbf{r})) \, dS

where :math:`\hat{\mathbf{n}}` is the unit normal to face :math:`f_m`.

.. note::
   The unit normal :math:`\hat{\mathbf{n}}` is computed using the cross product of two edge vectors of the triangle, for instance, ``normalize(cross(v1 - v0, v2 - v0))``. This calculation's correctness hinges on the mesh having a consistent vertex winding order (e.g., counter-clockwise when viewed from outside the surface). This consistency ensures that all normals point outwards, from Region 1 to Region 2, aligning with the convention established in :eq:`eq:eqcurrents`.

Step 2 — Dunavant quadrature on the triangle
---------------------------------------------------

We apply a Dunavant quadrature rule to approximate the surface integral. The integral of a function :math:`g(\mathbf{r})` over the triangle :math:`f_m` is approximated as:

.. math::

   \int_{f_m} g(\mathbf{r}) \, dS \approx A_m \sum_{p=1}^{Q_m} w_p g(\mathbf{r}_{m,p})

Applying this to :eq:`eq:I_mjni_substituted`, the area :math:`A_m` from the quadrature rule cancels one of the :math:`A_m` terms in the denominator:

.. math::
   :label: eq:I_mjni_dunavant

   \mathcal{I}_{mj,ni} \approx \frac{e_{mj}e_{ni}}{4A_m} \sum_{p=1}^{Q_m} w_p \left[ \boldsymbol{\rho}_{mj}(\mathbf{r}_{m,p}) \cdot (\hat{\mathbf{n}} \times \boldsymbol{\rho}_{ni}(\mathbf{r}_{m,p})) \right]

This cross-identity expression is implemented by
:cpp:func:`::calculateCrossIdentityQuadrature`. The result is purely real,
but it is stored in a ``complex__float`` matrix to facilitate its addition
to the complex-valued TK operator blocks during the final matrix assembly.


Testing of the incident E and H fields vectors
==========================================================

The right-hand side of the SIEs in :eq:`eq:sie3` and :eq:`eq:sie4` involves testing the incident electric and magnetic fields against the basis functions. The resulting vectors are known as the excitation vectors, :math:`\mathbf{V}_E` and :math:`\mathbf{V}_H`.

For a given global edge :math:`e_m` associated with testing function :math:`\mathbf{f}_m`, the corresponding elements of the excitation vectors are:

.. math::
   :label: eq:excitation_vectors_

   V_{E,m} = \int_S \mathbf{f}_m(\mathbf{r}) \cdot \mathbf{E}^{\mathrm{inc}}_{\mathrm{tan}}(\mathbf{r}) \, dS
   \qquad \text{and} \qquad
   V_{H,m} = \int_S \mathbf{f}_m(\mathbf{r}) \cdot \mathbf{H}^{\mathrm{inc}}_{\mathrm{tan}}(\mathbf{r}) \, dS

where :math:`\mathbf{E}^{\mathrm{inc}}_{\mathrm{tan}}` and :math:`\mathbf{H}^{\mathrm{inc}}_{\mathrm{tan}}` are the tangential components of the incident fields on the surface :math:`S`.

The integral is assembled face by face. For a single face :math:`f_n` and a local edge :math:`i` (which maps to a global edge :math:`e`), its contribution to the global excitation vector is:

.. math::
   :label: eq:excitation_local

   V_{e,n,i} = \int_{f_n} \mathbf{f}_{ni}(\mathbf{r}) \cdot \mathbf{F}^{\mathrm{inc}}_{\mathrm{tan}}(\mathbf{r}) \, dS

where :math:`\mathbf{F}^{\mathrm{inc}}` represents either the incident electric or magnetic field.

Step 1 — Substitute half-RWG basis functions
---------------------------------------------

Using the definition of the half-RWG basis function from :eq:`eq:rwg_half_basis`:

.. math::

   \mathbf{f}_{ni}(\mathbf{r}) = \frac{e_{ni}}{2A_n} (\mathbf{r} - \mathbf{r}_{\overline{\partial e_{ni}}}) = \frac{e_{ni}}{2A_n} \boldsymbol{\rho}_{ni}(\mathbf{r})

the local contribution to the excitation vector becomes:

.. math::
   :label: eq:excitation_substituted

   V_{e,n,i} = \frac{e_{ni}}{2A_n} \int_{f_n} \boldsymbol{\rho}_{ni}(\mathbf{r}) \cdot \mathbf{F}^{\mathrm{inc}}_{\mathrm{tan}}(\mathbf{r}) \, dS

Step 2 — Dunavant quadrature on the triangle
---------------------------------------------

We approximate the integral using a Dunavant quadrature rule with :math:`Q_n` points :math:`\mathbf{r}_{n,p}` and weights :math:`w_p`:

.. math::

   \int_{f_n} g(\mathbf{r}) \, dS \approx A_n \sum_{p=1}^{Q_n} w_p g(\mathbf{r}_{n,p})

Applying this to :eq:`eq:excitation_substituted`, the triangle area :math:`A_n` cancels out:

.. math::
   :label: eq:excitation_dunavant

   V_{e,n,i} \approx \frac{e_{ni}}{2A_n} A_n \sum_{p=1}^{Q_n} w_p \left[ \boldsymbol{\rho}_{ni}(\mathbf{r}_{n,p}) \cdot \mathbf{F}^{\mathrm{inc}}_{\mathrm{tan}}(\mathbf{r}_{n,p}) \right]

This simplifies to the final expression used for numerical computation:

.. math::
   :label: eq:excitation_final

   \boxed{
   V_{e,n,i} \approx \frac{e_{ni}}{2} \sum_{p=1}^{Q_n} w_p \left[ \boldsymbol{\rho}_{ni}(\mathbf{r}_{n,p}) \cdot \mathbf{F}^{\mathrm{inc}}_{\mathrm{tan}}(\mathbf{r}_{n,p}) \right]
   }

The total value for the global excitation vector element :math:`V_e` is obtained by summing the contributions :math:`V_{e,n,i}` from the two triangles that share the global edge :math:`e`. This calculation uses
:cpp:func:`::calculateExcitationQuadrature`, which is parameterized by a
functor that evaluates the specific incident field (e.g., a plane wave).

Testing of the :math:`\hat{\mathbf{n}} \times \mathcal{L}` operator
===================================================================

Testing the operator :math:`\hat{\mathbf{n}} \times \mathcal{L}` with a standard RWG basis function :math:`\mathbf{f}_{mj}` is mathematically equivalent to testing the standard :math:`\mathcal{L}` operator with a rotated testing function :math:`\mathbf{t}_m = \mathbf{f}_{mj} \times \hat{\mathbf{n}}_m`:

.. math::
   n\mathcal{TL}_{mj,ni} = \int_{f_m} \mathbf{f}_{mj} \cdot (\hat{\mathbf{n}}_m \times \mathcal{L}[\mathbf{f}_{ni}]) \, dS
   = \int_{f_m} (\mathbf{f}_{mj} \times \hat{\mathbf{n}}_m) \cdot \mathcal{L}[\mathbf{f}_{ni}] \, dS.

Let :math:`\mathbf{t}_m = \mathbf{f}_{mj} \times \hat{\mathbf{n}}_m` denote this rotated testing function. Substituting the definition of the :math:`\mathcal{L}` operator, the single testing/source pair entry becomes:

.. math::
   n\mathcal{TL}_{mj,ni} = \int_{f_m} \mathbf{t}_m(\mathbf{r}) \cdot
   \left[ \int_{f_n} \left( \mathbf{f}_{ni}(\mathbf{r}') \, G(\mathbf{r}, \mathbf{r}')
   - \frac{1}{k^2} (\nabla'_S \cdot \mathbf{f}_{ni}(\mathbf{r}')) \, \nabla' G(\mathbf{r}, \mathbf{r}') \right) dS' \right] dS.

Following the same procedure as for standard :math:`\mathcal{TL}`, we can split this into two terms:

.. math::
   \mathrm{Term\ 1}
   = \iint_{f_m \times f_n} \big(\mathbf{t}_m \cdot \mathbf{f}_{ni}\big)\, G \, dS\, dS',
   \qquad
   \mathrm{Term\ 2}
   = -\frac{1}{k^2}\iint_{f_m\times f_n}
     (\nabla'_S\!\cdot \mathbf{f}_{ni})\ \big(\mathbf{t}_m\cdot \nabla' G\big)\, dS'\, dS.

**Distributional split on** :math:`f_n`.
Just as with the standard :math:`\mathcal{L}` operator, we extend :math:`\mathbf{f}_{ni}` by zero outside :math:`f_n` and use the distributional identity

:math:`\nabla'_S\!\cdot(\chi_{f_n}\mathbf{f}_{ni}) = (\nabla'_S\!\cdot \mathbf{f}_{ni})\,\chi_{f_n} + (\mathbf{f}_{ni}\!\cdot \boldsymbol{\nu}')\,\delta_{\partial f_n}`

to write :math:`\mathrm{Term\ 2} = \mathrm{Term\ 2.1} + \mathrm{Term\ 2.2}`, splitting it into a boundary contour integral on the source triangle and a double surface integral:

.. math::
   \mathrm{Term\ 2.1}
   = -\frac{1}{k^2} \int_{f_m} \oint_{\partial f_n}
     \big(\mathbf{f}_{ni}\!\cdot \boldsymbol{\nu}'\big)\, \big(\mathbf{t}_m \cdot \nabla' G\big)\, dl'\, dS,

.. math::
   \mathrm{Term\ 2.2}
   = -\frac{1}{k^2} \iint_{f_m\times f_n}
     (\nabla'_S\!\cdot \mathbf{f}_{ni})\ \big(\mathbf{t}_m \cdot \nabla' G\big)\, dS'\, dS.

At this point, we analyze the behavior of these terms during global matrix assembly. For any internal edge shared by two source triangles :math:`f_{n^+}` and :math:`f_{n^-}`, the co-normals :math:`\boldsymbol{\nu}'` point in opposite directions. Since the standard RWG basis function :math:`\mathbf{f}_{ni}` guarantees continuous normal components across edges, the sum :math:`(\mathbf{f}_{n^+i} \cdot \boldsymbol{\nu}^+) + (\mathbf{f}_{n^-i} \cdot \boldsymbol{\nu}^-)` exactly vanishes. For a closed surface, all edges are internal, so all line integrals in :math:`\mathrm{Term\ 2.1}` cancel out completely. Therefore, we can drop this term entirely for a closed surface.

Now we are left with :math:`\mathrm{Term\ 2.2}`. In the standard :math:`\mathcal{TL}` evaluation, the next step would be to apply surface integration by parts on the testing triangle :math:`f_m` to shift the gradient operator off the Green's function. Let's see what happens if we attempt this with our rotated testing function :math:`\mathbf{t}_m`.

For a **piecewise planar triangular panel** where the face normal :math:`\hat{\mathbf{n}}_m` is constant and :math:`\nabla_S \times \mathbf{f}_{mj} = 0`, the surface divergence of the rotated testing function vanishes identically:

.. math::
   \nabla_S \cdot \mathbf{t}_m = \nabla_S \cdot (\mathbf{f}_{mj} \times \hat{\mathbf{n}}_m) = \hat{\mathbf{n}}_m \cdot (\nabla_S \times \mathbf{f}_{mj}) = 0

Applying the surface divergence theorem on :math:`f_m` yields:

.. math::
   \begin{aligned}
   \int_{f_m} \mathbf{t}_m \cdot \nabla' G \, dS 
   &= - \int_{f_m} \mathbf{t}_m \cdot \nabla G \, dS \\
   &= - \int_{f_m} \big[ \nabla_S \cdot (G \mathbf{t}_m) - G (\nabla_S \cdot \mathbf{t}_m) \big] \, dS \\
   &= - \oint_{\partial f_m} G \, (\mathbf{t}_m \cdot \boldsymbol{\nu}_m) \, dl + \int_{f_m} G \underbrace{(\nabla_S \cdot \mathbf{t}_m)}_{=0} \, dS \\
   &= - \oint_{\partial f_m} G \, (\mathbf{t}_m \cdot \boldsymbol{\nu}_m) \, dl
   \end{aligned}

Notice that the inner product on the boundary is :math:`\mathbf{t}_m \cdot \boldsymbol{\nu}_m = (\mathbf{f}_{mj} \times \hat{\mathbf{n}}_m) \cdot \boldsymbol{\nu}_m = \mathbf{f}_{mj} \cdot (\hat{\mathbf{n}}_m \times \boldsymbol{\nu}_m) = \mathbf{f}_{mj} \cdot \hat{\mathbf{l}}_m`, where :math:`\hat{\mathbf{l}}_m` is the unit vector tangent to the edge of the testing face.

This reveals a critical difference: the standard RWG basis function :math:`\mathbf{f}_{mj}` does **not** enforce tangential continuity across edges. Therefore, the term :math:`\mathbf{f}_{mj} \cdot \hat{\mathbf{l}}_m` is discontinuous across adjacent triangles, meaning **these line integrals on the testing triangle do not cancel during global assembly**.

To evaluate :math:`n\mathcal{TL}`, we are faced with two computational choices:

* **Form A (Gibson's Line Representation):** Analytically transfer the testing derivative to the boundary :math:`\partial f_m` via integration by parts (:math:`\nabla G \to G`, area :math:`\to` line). This lowers the pointwise kernel singularity from :math:`O(R^{-2})` to weakly singular :math:`O(R^{-1})` along edges.
* **Form B (Double Area Integration):** Retain :math:`\nabla' G` in the double-surface integral without integration by parts.

Form A (Gibson's Line Representation)
-------------------------------------

Substituting the boundary integral back into the operator expression yields:

.. math::
   :label: eq:nTL_formA_operator

   n\mathcal{TL}_{mj,ni} = \iint_{f_m \times f_n} \big(\mathbf{t}_{mj} \cdot \mathbf{f}_{ni}\big)\, G \, dS\, dS'
   + \frac{1}{k^2} \oint_{\partial f_m} \int_{f_n} (\mathbf{t}_{mj} \cdot \boldsymbol{\nu}_m) (\nabla'_S \cdot \mathbf{f}_{ni}) \, G \, dS'\, dl

where :math:`\boldsymbol{\nu}_m` is the outward-pointing in-plane normal vector to the boundary of triangle :math:`f_m`.

To evaluate the boundary line integral numerically in a code-ready form for far interactions, we discretize the outer contour integral over the three edges :math:`e \in \{1, 2, 3\}` of the testing triangle :math:`f_m` using a 1D Gauss-Legendre quadrature rule, while the inner integral over the source triangle :math:`f_n` relies on our existing 2D Dunavant quadrature. Substituting the basis functions :math:`\mathbf{t}_{mj} = \frac{e_{mj}}{2A_m} (\boldsymbol{\rho}_{mj} \times \hat{\mathbf{n}}_m)` and :math:`\nabla'_S \cdot \mathbf{f}_{ni} = \frac{e_{ni}}{A_n}`, the discrete summation form of the line integral is:

.. math::
   :label: eq:nTL_formA_line_discrete

   \frac{e_{mj} e_{ni}}{2 A_m k^2} \sum_{e=1}^3 L_e \sum_{q=1}^{Q_{1D}} \sum_{p=1}^{Q_{2D}} w_q^{1D} w_p^{2D} \Big[ \big( \boldsymbol{\rho}_{mj}(\mathbf{r}_{eq}) \times \hat{\mathbf{n}}_m \big) \cdot \boldsymbol{\nu}_{m,e} \Big] G(\mathbf{r}_{eq}, \mathbf{r}_{np}')

Here, :math:`\boldsymbol{\nu}_{m,e}` is the outward in-plane normal for edge :math:`e`, :math:`L_e` is the length of edge :math:`e`, and :math:`\mathbf{r}_{eq}` are the 1D Gauss-Legendre quadrature points evaluated along the edge. Note that using the factor :math:`L_e` (rather than :math:`L_e/2`) assumes our 1D Gauss-Legendre quadrature rules are normalized to the reference interval :math:`[0, 1]` with weights summing to unity (:math:`\sum_{q=1}^{Q_{1D}} w_q^{1D} = 1`). Furthermore, unlike the 2D area integration where the Jacobian :math:`A_m` cancels with the :math:`1/A_m` factor in the testing basis function, the 1D line integral preserves the :math:`A_m` term in the denominator.

Combining this with the standard 2D-2D Dunavant quadrature for the first term :math:`\big(\mathbf{t}_{mj} \cdot \mathbf{f}_{ni}\big)`, the complete far-interaction discrete formula for Form A is:

.. math::
   :label: eq:nTL_formA_discrete

   n\mathcal{TL}_{mj,ni} \approx e_{mj} e_{ni} \Bigg( & \frac{1}{4} \sum_{p=1}^{Q_{2D}} \sum_{q=1}^{Q_{2D}} w_p^{2D} w_q^{2D} \Big[ \big(\boldsymbol{\rho}_{mj}(\mathbf{r}_{mp}) \times \hat{\mathbf{n}}_m \big) \cdot \boldsymbol{\rho}_{ni}(\mathbf{r}_{nq}') \Big] G(\mathbf{r}_{mp}, \mathbf{r}_{nq}') \\
   & + \frac{1}{2 A_m k^2} \sum_{e=1}^3 L_e \sum_{q=1}^{Q_{1D}} \sum_{p=1}^{Q_{2D}} w_q^{1D} w_p^{2D} \Big[ \big( \boldsymbol{\rho}_{mj}(\mathbf{r}_{eq}) \times \hat{\mathbf{n}}_m \big) \cdot \boldsymbol{\nu}_{m,e} \Big] G(\mathbf{r}_{eq}, \mathbf{r}_{np}') \Bigg)

**Implementation:** :cpp:func:`::calculateFormAQuadrature_nTL_nTK` combines
the triangle–triangle and edge–triangle quadrature in this Form A expression.

Form B (Double Area Integration)
--------------------------------

In Form B, we directly evaluate the double surface integral containing :math:`\nabla' G` without transferring the derivative to the boundary:

.. math::
   :label: eq:nTL_surface_final

   n\mathcal{TL}_{mj,ni} = \iint_{f_m \times f_n} \big(\mathbf{t}_m \cdot \mathbf{f}_{ni}\big)\, G \, dS\, dS'
   - \frac{1}{k^2} \iint_{f_m\times f_n} (\nabla'_S\!\cdot \mathbf{f}_{ni})\ \big(\mathbf{t}_m\cdot \nabla' G\big)\, dS'\, dS.

While Form B avoids the geometric tracking of boundary edges, testing the pointwise :math:`O(R^{-2})` kernel over 2D areas can lead to numerical ill-conditioning and near-field ringing on coarse meshes. For singularity extraction in Form B, the same Green's function singular split and the same underlying Wilton potential family can be reused, but the geometric linear combination required by :math:`n\mathcal{TL}` must be derived separately (as detailed in :doc:`singular`).


To compute the numerical double surface integral for :math:`n\mathcal{TL}`, we apply the Dunavant quadrature rules. Recall from our convention that :math:`\int_{f} g(\mathbf{r}) dS \approx A \sum_{p=1}^Q w_p g(\mathbf{r}_p)` where the weights sum to 1. When applying this to the double integral, the product of the differential areas :math:`dS \, dS'` is replaced by :math:`A_m A_n \, w_p w_q`.

Substituting the definitions of the basis functions :math:`\mathbf{t}_{mj} = \frac{e_{mj}}{2A_m} (\boldsymbol{\rho}_{mj} \times \hat{\mathbf{n}}_m)` and :math:`\mathbf{f}_{ni} = \frac{e_{ni}}{2A_n} \boldsymbol{\rho}_{ni}`, we get a factor of :math:`\frac{1}{4 A_m A_n}` for the first term. Multiplying by the area Jacobian :math:`A_m A_n` leaves a geometric coefficient of :math:`1/4`.

For the second term, the divergence :math:`\nabla'_S \cdot \mathbf{f}_{ni} = \frac{e_{ni}}{A_n}` lacks a factor of 1/2. Therefore, the combination :math:`(\nabla'_S \cdot \mathbf{f}_{ni}) \mathbf{t}_{mj}` carries a factor of :math:`\frac{1}{2 A_m A_n}`. Multiplying by the area Jacobian leaves a coefficient of :math:`1/2`.

This subtle difference of :math:`1/2` versus :math:`1/4` is a direct result of avoiding surface integration by parts on the testing triangle!

The fully discrete Dunavant quadrature form for the regular part of the :math:`n\mathcal{TL}` operator becomes:

.. math::
   :label: eq:nTL_dunavant

   n\mathcal{TL}_{mj,ni} \approx e_{mj}e_{ni} \sum_{p=1}^{Q_m} \sum_{q=1}^{Q_n} w_p w_q \left[ \frac{1}{4} \big( (\boldsymbol{\rho}_{mj,p} \times \hat{\mathbf{n}}_m) \cdot \boldsymbol{\rho}_{ni,q} \big) G_{pq} - \frac{1}{2k^2} \big( (\boldsymbol{\rho}_{mj,p} \times \hat{\mathbf{n}}_m) \cdot \nabla' G_{pq} \big) \right]
   
**Implementation:** :cpp:func:`::calculateDoubleQuadrature_nTL_nTK` evaluates
this Form B double-surface sum.

Testing of the :math:`\hat{\mathbf{n}} \times \mathcal{K}` operator
===================================================================

The evaluation of the tested :math:`\hat{\mathbf{n}} \times \mathcal{K}` operator is significantly simpler than the :math:`\hat{\mathbf{n}} \times \mathcal{L}` operator because the standard :math:`\mathcal{K}` formulation does not rely on surface integration by parts.

We define the single testing/source pair entry as:

.. math::
   n\mathcal{TK}_{mj,ni} = \int_{f_m} \mathbf{f}_{mj}(\mathbf{r}) \cdot \big( \hat{\mathbf{n}}_m \times \mathcal{K}[\mathbf{f}_{ni}] \big) \, dS

Using the scalar triple product identity, we can shift the cross product onto the testing function:

.. math::
   n\mathcal{TK}_{mj,ni} = \int_{f_m} \big( \mathbf{f}_{mj}(\mathbf{r}) \times \hat{\mathbf{n}}_m \big) \cdot \mathcal{K}[\mathbf{f}_{ni}] \, dS = \int_{f_m} \mathbf{t}_m(\mathbf{r}) \cdot \mathcal{K}[\mathbf{f}_{ni}] \, dS

where :math:`\mathbf{t}_m = \mathbf{f}_{mj} \times \hat{\mathbf{n}}_m` is the exact same rotated testing function used for the :math:`n\mathcal{TL}` operator.

Substituting the definition of the :math:`\mathcal{K}` operator, we obtain the double surface integral:

.. math::
   n\mathcal{TK}_{mj,ni} = \iint_{f_m \times f_n} \mathbf{t}_m(\mathbf{r}) \cdot \big( \mathbf{f}_{ni}(\mathbf{r}') \times \nabla' G(\mathbf{r}, \mathbf{r}') \big) \, dS \, dS'

Notice that the inner integral is mathematically identical to the inner integral of the standard :math:`\mathcal{TK}` operator. The only difference in the :math:`n\mathcal{TK}` operator is that the resulting vector field is dotted with :math:`\mathbf{t}_m` instead of :math:`\mathbf{f}_{mj}`. This means that the numerical quadrature and singularity extraction logic for the inner integral of :math:`\mathcal{K}` can be reused entirely without modification.


From single-pair Surface Integral to Quadrature Dunavant rules
--------------------------------------------------------------

As with the previous operators, we apply the Dunavant quadrature rules, replacing the differential area product :math:`dS \, dS'` with the area Jacobian :math:`A_m A_n \, w_p w_q`.

Substituting the definitions :math:`\mathbf{t}_{mj} = \frac{e_{mj}}{2A_m} (\boldsymbol{\rho}_{mj} \times \hat{\mathbf{n}}_m)` and :math:`\mathbf{f}_{ni} = \frac{e_{ni}}{2A_n} \boldsymbol{\rho}_{ni}`, we get a leading geometric factor of :math:`\frac{1}{4 A_m A_n}`. Canceling the area product with the Jacobian leaves a clean :math:`1/4` coefficient, which perfectly matches the scaling of the standard :math:`\mathcal{TK}` operator.

The fully discrete Dunavant quadrature form for the regular part of the :math:`n\mathcal{TK}` operator becomes:

.. math::
   :label: eq:nTK_dunavant

   n\mathcal{TK}_{mj,ni} \approx e_{mj}e_{ni} \sum_{p=1}^{Q_m} \sum_{q=1}^{Q_n} w_p w_q \left[ \frac{1}{4} \big( \boldsymbol{\rho}_{mj,p} \times \hat{\mathbf{n}}_m \big) \cdot \big( \boldsymbol{\rho}_{ni,q} \times \nabla' G_{pq} \big) \right]

**Implementation:** the nTK outputs of
:cpp:func:`::calculateDoubleQuadrature_nTL_nTK` (Form B) and
:cpp:func:`::calculateFormAQuadrature_nTL_nTK` (Form A) evaluate this sum.

.. |tl-tk-quadrature| replace:: :cpp:func:`calculateDoubleQuadrature() <void ::calculateDoubleQuadrature(const vec3_t[3], const int[3], const vec3_t[3], const int[3], const float*, const float*, int, real_t, real_t, bool, complex__float[3][3], complex__float[3][3], complex__float[3][3], complex__float[3][3])>`
