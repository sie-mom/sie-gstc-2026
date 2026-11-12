.. matrix_assembly_v2.rst

#########################
Matrix Assembly
#########################

General Boundary Integral Equation Theory (BIE or SIE)
=======================================================

We recall the general SIE formulation from equation :eq:`eq:sie1`, :eq:`eq:sie2`, :eq:`eq:sie3` and :eq:`eq:sie4` from the theory section that we repeat here for convenience and emphasizing the testing of the operators involved:

General formulation of the Tested Surface Integral Equations (SIE) representation.

   .. math::
      :label: eq:sie1_testing

      -\tfrac{1}{2}\,\mathcal{I}[\mathbf{M}_1] + j\omega \mu_1\, \mathcal{TL}_{k_1}[\mathbf{J}_1] + \mathcal{TK}_{k_1}[\mathbf{M}_1] = 0.

   .. math::
      :label: eq:sie2_testing

      \tfrac{1}{2}\,\mathcal{I}[\mathbf{J}_1] + j\omega \varepsilon_1\, \mathcal{TL}_{k_1}[\mathbf{M}_1] - \mathcal{TK}_{k_1}[\mathbf{J}_1] = 0.

   .. math::
      :label: eq:sie3_testing

      \tfrac{1}{2}\,\mathcal{I}[\mathbf{M}_2] + j\omega \mu_2\, \mathcal{TL}_{k_2}[\mathbf{J}_2] + \mathcal{TK}_{k_2}[\mathbf{M}_2] = \mathbf{V}_E.

   .. math::
      :label: eq:sie4_testing

      -\tfrac{1}{2}\,\mathcal{I}[\mathbf{J}_2] + j\omega \varepsilon_2\, \mathcal{TL}_{k_2}[\mathbf{M}_2] - \mathcal{TK}_{k_2}[\mathbf{J}_2] = \mathbf{V}_H.

where we have introduced the notation for the tested identity operator :math:`\mathcal{I}` and the excitation vectors :math:`\mathbf{V}_E` and :math:`\mathbf{V}_H`. The elements of the identity matrix :math:`\mathcal{I}_{mn}` are given by the inner product:

.. math::
   :label: eq:identity_matrix_element

   \mathcal{I}_{mj,ni} = \int_{f_m} \mathbf{f}_{mj} \cdot (\hat{\mathbf{n}} \times \mathbf{f}_{ni}) \, dS

The excitation vectors are formed by testing the incident fields:

.. math::

   (\mathbf{V}_E)_{mj} = \int_{f_m} \mathbf{f}_{mj} \cdot \mathbf{E}^{\mathrm{inc}} \, dS, \qquad (\mathbf{V}_H)_{mj} = \int_{f_m} \mathbf{f}_{mj} \cdot \mathbf{H}^{\mathrm{inc}} \, dS

To create a matrix system, we can express the four SIEs in a more compact form using discrete Calderón-like boundary operators. This approach provides greater conceptual clarity and aligns the implementation more closely with modern boundary integral equation theory.

Let :math:`\mathcal{TL}_{k}` and :math:`\mathcal{TK}_{k}` be the tested single- and double-layer operators, and let :math:`\mathcal{I}` denote the tested identity operator (or mass matrix for RWG functions). We define two 2x2 block operators, one for the interior region (1) and one for the exterior region (2).

The interior operator :math:`\mathbf{C}^{(1)}` and exterior operator :math:`\mathbf{C}^{(2)}` are defined as:

.. math::
   :label: eq:calderon_operators

   \mathbf{C}^{(1)} \equiv
   \begin{bmatrix}
      j\omega\mu_1 \mathcal{TL}_{k_1} & \mathcal{TK}_{k_1} - \tfrac{1}{2}\mathcal{I} \\
      -\mathcal{TK}_{k_1} + \tfrac{1}{2}\mathcal{I} & j\omega\varepsilon_1 \mathcal{TL}_{k_1}
   \end{bmatrix},
   \qquad
   \mathbf{C}^{(2)} \equiv
   \begin{bmatrix}
      j\omega\mu_2 \mathcal{TL}_{k_2} & \mathcal{TK}_{k_2} + \tfrac{1}{2}\mathcal{I} \\
      -\mathcal{TK}_{k_2} - \tfrac{1}{2}\mathcal{I} & j\omega\varepsilon_2 \mathcal{TL}_{k_2}
   \end{bmatrix}

With these definitions, the four original equations (:eq:`eq:sie1_testing` - :eq:`eq:sie4_testing`) can be written as two decoupled 2x2 block systems, one for each region with its own set of currents:

.. math::
   :label: eq:decoupled_systems

   \mathbf{C}^{(1)} \begin{bmatrix} \mathbf{J}_1 \\ \mathbf{M}_1 \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \end{bmatrix},
   \qquad
   \mathbf{C}^{(2)} \begin{bmatrix} \mathbf{J}_2 \\ \mathbf{M}_2 \end{bmatrix} = \begin{bmatrix} \mathbf{V}_E \\ \mathbf{V}_H \end{bmatrix}.

The first equation represents the null-field condition inside the scatterer (Region 1), while the second represents the field condition outside (Region 2).

To combine these into a single system, we can define sum and difference operators:

.. math::

   \mathbf{C}_{\text{sum}} = \mathbf{C}^{(1)} + \mathbf{C}^{(2)}, \qquad \mathbf{C}_{\text{diff}} = \mathbf{C}^{(2)} - \mathbf{C}^{(1)}.

.. note:: The explicit forms of the sum and difference operators are:

   The sum operator :math:`\mathbf{C}_{\text{sum}}` corresponds to :math:`\mathbf{C}^{(1)} + \mathbf{C}^{(2)}`:

   .. math::

      \mathbf{C}_{\text{sum}} =
      \begin{bmatrix}
         j\omega(\mu_1\mathcal{TL}_{k_1} + \mu_2\mathcal{TL}_{k_2}) & \mathcal{TK}_{k_1} + \mathcal{TK}_{k_2} \\
         -(\mathcal{TK}_{k_1} + \mathcal{TK}_{k_2}) & j\omega(\varepsilon_1\mathcal{TL}_{k_1} + \varepsilon_2\mathcal{TL}_{k_2})
      \end{bmatrix}

   The difference operator :math:`\mathbf{C}_{\text{diff}}` corresponds to :math:`\mathbf{C}^{(2)} - \mathbf{C}^{(1)}`:

   .. math::

      \mathbf{C}_{\text{diff}} =
      \begin{bmatrix}
         j\omega(\mu_2\mathcal{TL}_{k_2} - \mu_1\mathcal{TL}_{k_1}) & (\mathcal{TK}_{k_2} - \mathcal{TK}_{k_1}) + \mathcal{I} \\
         -(\mathcal{TK}_{k_2} - \mathcal{TK}_{k_1}) - \mathcal{I} & j\omega(\varepsilon_2\mathcal{TL}_{k_2} - \varepsilon_1\mathcal{TL}_{k_1})
      \end{bmatrix}


Boundary Conditions for Dielectric Interfaces (PMCHWT)
============================================================

For a passive dielectric interface, the tangential components of the total electric and magnetic fields must be continuous across the boundary :math:`S`.

.. math::
   :label: eq:bc_tangential

   \hat{\mathbf{n}} \times \mathbf{E}_1 = \hat{\mathbf{n}} \times \mathbf{E}_2, \qquad
   \hat{\mathbf{n}} \times \mathbf{H}_1 = \hat{\mathbf{n}} \times \mathbf{H}_2.

We can relate these physical fields to the equivalent surface currents defined in :eq:`eq:eqcurrents`.

From the continuity of the tangential electric field:

.. math::

   \mathbf{M}_1 = \hat{\mathbf{n}} \times \mathbf{E}_1 = \hat{\mathbf{n}} \times \mathbf{E}_2 = -(-\hat{\mathbf{n}} \times \mathbf{E}_2) = -\mathbf{M}_2.

From the continuity of the tangential magnetic field:

.. math::

   -\mathbf{J}_1 = \hat{\mathbf{n}} \times \mathbf{H}_1 = \hat{\mathbf{n}} \times \mathbf{H}_2 = \mathbf{J}_2.

This gives us the fundamental relationship between the currents on the two sides of the interface:

.. math::
   :label: eq:current_relations

   \mathbf{J}_1 = -\mathbf{J}_2, \qquad \mathbf{M}_1 = -\mathbf{M}_2.

This result is significant because it halves the number of unknown current coefficients. We can define a single set of equivalent currents, :math:`\mathbf{J} \equiv \mathbf{J}_2` and :math:`\mathbf{M} \equiv \mathbf{M}_2`, and express the currents for Region 1 in terms of them. This allows us to combine the four initial equations into a system of two coupled equations with two unknown current distributions.

Final 2x2 Block Matrix System
--------------------------------

To obtain a system free of spurious interior resonances, we combine the interior and exterior equations. As shown in the previous section, this is achieved by adding the operator equations from :eq:`eq:decoupled_systems` and applying the boundary conditions.

Before enforcing a single set of currents, this gives

.. math::
   :label: eq:pmchwt_system

   \begin{aligned}
   & j\omega \mu_1\,\mathcal{TL}_{k_1}[\mathbf{J}_1] + \big(\mathcal{TK}_{k_1} - \tfrac{1}{2}\mathcal{I}\big)[\mathbf{M}_1]
     + j\omega \mu_2\,\mathcal{TL}_{k_2}[\mathbf{J}_2] + \big(\mathcal{TK}_{k_2} + \tfrac{1}{2}\mathcal{I}\big)[\mathbf{M}_2]
     \;=\; \mathbf{V}_E, \\[4pt]
   & -\big(\mathcal{TK}_{k_1} - \tfrac{1}{2}\mathcal{I}\big)[\mathbf{J}_1] + j\omega \varepsilon_1\,\mathcal{TL}_{k_1}[\mathbf{M}_1]
     - \big(\mathcal{TK}_{k_2} + \tfrac{1}{2}\mathcal{I}\big)[\mathbf{J}_2] + j\omega \varepsilon_2\,\mathcal{TL}_{k_2}[\mathbf{M}_2]
     \;=\; \mathbf{V}_H .
   \end{aligned}

Enforce the boundary relations :math:`\mathbf{J}_1 = -\mathbf{J}_2` and :math:`\mathbf{M}_1 = -\mathbf{M}_2`,
and define :math:`\mathbf{J} \equiv \mathbf{J}_2`, :math:`\mathbf{M} \equiv \mathbf{M}_2`. The jump terms
:math:`\pm \tfrac{1}{2}\mathcal{I}` cancel, and we obtain the **PMCHWT** system

.. math::
   :label: eq:final_matrix_system

   \begin{bmatrix}
      j\omega\!\left(\mu_1 \mathcal{TL}_{k_1} + \mu_2 \mathcal{TL}_{k_2}\right)
      &
      \mathcal{TK}_{k_1} + \mathcal{TK}_{k_2}
      \\
      -\left(\mathcal{TK}_{k_1} + \mathcal{TK}_{k_2}\right)
      &
      j\omega\!\left(\varepsilon_1 \mathcal{TL}_{k_1} + \varepsilon_2 \mathcal{TL}_{k_2}\right)
   \end{bmatrix} 
   \begin{bmatrix}
      \mathbf{J} \\
      \mathbf{M}
   \end{bmatrix}
   =
   \begin{bmatrix}
      \mathbf{V}_E \\
      \mathbf{V}_H
   \end{bmatrix} .

.. note::

   - There is **no explicit** :math:`\mathcal{I}` operator in the final 2×2 system. This is a key feature for the stability of the formulation.
   - The resulting system matrix is precisely the sum operator :math:`\mathbf{C}_{\text{sum}}` defined earlier. The PMCHWT system can therefore be written concisely as :math:`\mathbf{C}_{\text{sum}} \begin{bmatrix} \mathbf{J} \\ \mathbf{M} \end{bmatrix} = \begin{bmatrix} \mathbf{V}_E \\ \mathbf{V}_H \end{bmatrix}`.


Implementation Strategy
-------------------------

This section outlines the practical steps to translate the final PMCHWT matrix equation (:eq:`eq:final_matrix_system`) into a concrete implementation. The goal is to assemble the `(2N x 2N)` impedance matrix `Z`, where `N` is the number of edges in the mesh.

**Implementation:** :cpp:func:`sie_mom::solvers::solve_dielectric_dense` selects
the formulation through :cpp:enum:`sie_mom::solvers::DielectricFormulation`.
For ``T_PMCHWT``, :cpp:func:`sie_mom::solvers::dielectric::launch_t_pmchwt_kernel`
launches :cpp:func:`sie_mom::solvers::dielectric::t_pmchwt_kernel`, which evaluates
each test/source face pair and accumulates its local RWG contributions into
the four global blocks. The dense array uses column-major storage.
The solver applies vacuum-impedance scaling internally and returns physical
currents in the interior-oriented order :math:`[\mathbf{J}_1,\mathbf{M}_1]`;
the exterior-oriented pair used above is its negative.

The rotated ``N_PMCHWT`` variant uses
:cpp:func:`sie_mom::solvers::dielectric::launch_n_pmchwt_kernel` and
:cpp:func:`sie_mom::solvers::dielectric::n_pmchwt_kernel`, with
:math:`n\mathcal{TL}` and :math:`n\mathcal{TK}` in place of the standard tested
operators. :cpp:enum:`sie_mom::solvers::NtlIntegralForm` selects the line
(Form A) or surface (Form B) evaluation of :math:`n\mathcal{TL}` described in
:doc:`theory`; singular contributions are documented in :doc:`singular`.

**Steps:Matrix Structure and Indexing**

The final system involves two unknown current vectors, :math:`\mathbf{J}` and :math:`\mathbf{M}`, each of size `N`. The global matrix `Z` is therefore of size `(2N x 2N)`. A standard memory layout is to arrange the blocks as follows:

- Rows/columns `0` to `N-1` correspond to the electric current :math:`\mathbf{J}`.
- Rows/columns `N` to `2N-1` correspond to the magnetic current :math:`\mathbf{M}`.

The four blocks of the impedance matrix are mapped into a single contiguous array:

- :math:`Z_{JJ}` populates the top-left quadrant: `Z[i, j]`
- :math:`Z_{JM}` populates the top-right quadrant: `Z[i, j + N]`
- :math:`Z_{MJ}` populates the bottom-left quadrant: `Z[i + N, j]`
- :math:`Z_{MM}` populates the bottom-right quadrant: `Z[i + N, j + N]`

where `i` and `j` are global edge indices from `0` to `N-1`.


Algorithm for PMCHWT Matrix Assembly
------------------------------------

The following pseudocode outlines the complete algorithm for assembling the `(2N x 2N)` PMCHWT impedance matrix `Z`.

.. code-block:: text
   :caption: PMCHWT Matrix Assembly Algorithm
   :name: alg:pmchwt_assembly

   // N = number of edges in the mesh
   // Initialize a (2N x 2N) complex matrix Z with zeros
   Initialize Z(2N, 2N) = 0

   for each test face m from 0 to num_faces-1:
       for each source face n from 0 to num_faces-1:
           // Efficiently compute all four 3x3 local interaction matrices
           // This is the most computationally expensive step.
           [local_TL1, local_TL2, local_TK1, local_TK2] = calculate_local_matrices(m, n, k1, k2)

           for each test edge i_local from 0 to 2:
               for each source edge j_local from 0 to 2:
                   // Get global indices for the edges
                   g_i = global_edge_index(face=m, local_edge=i_local)
                   g_j = global_edge_index(face=n, local_edge=j_local)

                   // --- Accumulate contributions into Z blocks ---

                   // Z_JJ block: j*omega*(mu1*TL1 + mu2*TL2)
                   Z[g_i, g_j] += j*omega*(mu1*local_TL1[i_local,j_local] + mu2*local_TL2[i_local,j_local])

                   // Z_JM block: (TK1 + TK2)
                   Z[g_i, g_j + N] += (local_TK1[i_local,j_local] + local_TK2[i_local,j_local])

                   // Z_MJ block: -(TK1 + TK2)
                   Z[g_i + N, g_j] -= (local_TK1[i_local,j_local] + local_TK2[i_local,j_local])

                   // Z_MM block: j*omega*(eps1*TL1 + eps2*TL2)
                   Z[g_i + N, g_j + N] += j*omega*(eps1*local_TL1[i_local,j_local] + eps2*local_TL2[i_local,j_local])

   return Z

Derivation of the n-Müller Formulation
===============================================

We begin with the four surface integral equations:

.. math::
   :label: eq:muller_sie1

   -\frac{1}{2}\hat{\mathbf{n}}\times\mathbf{M}_1 + j\omega\mu_1 \left[\mathcal{L}_1[\mathbf{J}_1]\right]_{\tan} + \left[\mathcal{K}_1[\mathbf{M}_1]\right]_{\tan} = 0 ,

.. math::
   :label: eq:muller_sie2

   \frac{1}{2}\hat{\mathbf{n}}\times\mathbf{J}_1 + j\omega\epsilon_1 \left[\mathcal{L}_1[\mathbf{M}_1]\right]_{\tan} - \left[\mathcal{K}_1[\mathbf{J}_1]\right]_{\tan} = 0 ,

.. math::
   :label: eq:muller_sie3

   \frac{1}{2}\hat{\mathbf{n}}\times\mathbf{M}_2 + j\omega\mu_2 \left[\mathcal{L}_2[\mathbf{J}_2]\right]_{\tan} + \left[\mathcal{K}_2[\mathbf{M}_2]\right]_{\tan} = \mathbf{E}_{\tan}^{\mathrm{inc}},

.. math::
   :label: eq:muller_sie4

   -\frac{1}{2}\hat{\mathbf{n}}\times\mathbf{J}_2 + j\omega\epsilon_2 \left[\mathcal{L}_2[\mathbf{M}_2]\right]_{\tan} - \left[\mathcal{K}_2[\mathbf{J}_2]\right]_{\tan} = \mathbf{H}_{\tan}^{\mathrm{inc}}.

Using the current identifications

.. math::
   \mathbf{J} = \mathbf{J}_2 = -\mathbf{J}_1, \qquad \mathbf{M} = \mathbf{M}_2 = -\mathbf{M}_1,

consider first the combination :math:`\epsilon_2 \cdot` :eq:`eq:muller_sie3` :math:`+ \, \epsilon_1 \cdot` :eq:`eq:muller_sie1`. This gives

.. math::
   j\omega\mu_2\epsilon_2 \left[\mathcal{L}_2[\mathbf{J}]\right]_{\tan} - j\omega\mu_1\epsilon_1 \left[\mathcal{L}_1[\mathbf{J}]\right]_{\tan}
   + \epsilon_2 \left[\mathcal{K}_2[\mathbf{M}]\right]_{\tan} - \epsilon_1 \left[\mathcal{K}_1[\mathbf{M}]\right]_{\tan}
   + \frac{1}{2} \left( \epsilon_2\hat{\mathbf{n}}\times\mathbf{M} + \epsilon_1\hat{\mathbf{n}}\times\mathbf{M} \right)
   = \epsilon_2\mathbf{E}_{\tan}^{\mathrm{inc}}.

Now consider :math:`\mu_2 \cdot` :eq:`eq:muller_sie4` :math:`+ \, \mu_1 \cdot` :eq:`eq:muller_sie2`. We obtain

.. math::
   j\omega\mu_2\epsilon_2 \left[\mathcal{L}_2[\mathbf{M}]\right]_{\tan} - j\omega\mu_1\epsilon_1 \left[\mathcal{L}_1[\mathbf{M}]\right]_{\tan}
   - \mu_2 \left[\mathcal{K}_2[\mathbf{J}]\right]_{\tan} + \mu_1 \left[\mathcal{K}_1[\mathbf{J}]\right]_{\tan}
   - \frac{1}{2} \left( \mu_2\hat{\mathbf{n}}\times\mathbf{J} + \mu_1\hat{\mathbf{n}}\times\mathbf{J} \right)
   = \mu_2\mathbf{H}_{\tan}^{\mathrm{inc}}.

This second combined equation is multiplied by :math:`-1`.

We then apply the :math:`\hat{\mathbf{n}}\times` operation and test with the basis/testing function :math:`\mathbf{f}_m`. Recall that for any vector field :math:`\mathbf{X}`,

.. math::
   \mathbf{X}_{\tan} = -\hat{\mathbf{n}}\times\hat{\mathbf{n}}\times\mathbf{X},

and therefore

.. math::
   \hat{\mathbf{n}}\times\mathbf{X}_{\tan} = -\hat{\mathbf{n}}\times\hat{\mathbf{n}}\times\hat{\mathbf{n}}\times\mathbf{X} = \hat{\mathbf{n}}\times\mathbf{X}.

Crucially, when :math:`\hat{\mathbf{n}}\times` is applied to the jump terms which are already of the form :math:`\hat{\mathbf{n}}\times\mathbf{M}`, we obtain:

.. math::
   \hat{\mathbf{n}}\times(\hat{\mathbf{n}}\times\mathbf{M}) = -\mathbf{M}

Testing this with :math:`\mathbf{f}_m` yields :math:`-\mathbf{G}`, where :math:`\mathbf{G}` is the Gram matrix. Hence, the complete :math:`n`-tested block system becomes

.. math::
   \left\{
   \begin{bmatrix}
   j\omega\epsilon_2\mu_2\,n\mathcal{TL}_2 - j\omega\epsilon_1\mu_1\,n\mathcal{TL}_1 & \epsilon_2\,n\mathcal{TK}_2 - \epsilon_1\,n\mathcal{TK}_1 \\
   \mu_2\,n\mathcal{TK}_2 - \mu_1\,n\mathcal{TK}_1 & -\left( j\omega\epsilon_2\mu_2\,n\mathcal{TL}_2 - j\omega\epsilon_1\mu_1\,n\mathcal{TL}_1 \right)
   \end{bmatrix}
   -
   \begin{bmatrix}
   0 & \dfrac{\epsilon_2+\epsilon_1}{2}\,\mathbf{G} \\
   \dfrac{\mu_1+\mu_2}{2}\,\mathbf{G} & 0
   \end{bmatrix}
   \right\}
   \begin{bmatrix}
   \mathbf{J}\\
   \mathbf{M}
   \end{bmatrix}
   =
   \begin{bmatrix}
   \epsilon_2\,\mathbf{V}_E\\
   -\mu_2\,\mathbf{V}_H
   \end{bmatrix}.

The excitation vectors are

.. math::
   V_E = \int \mathbf{f}_m\cdot \left( \hat{\mathbf{n}}\times \mathbf{E}^{\mathrm{inc}} \right) \,dS = \int \left( \mathbf{f}_m\times\hat{\mathbf{n}} \right) \cdot \mathbf{E}^{\mathrm{inc}} \,dS ,

and

.. math::
   V_H = \int \mathbf{f}_m\cdot \left( \hat{\mathbf{n}}\times \mathbf{H}^{\mathrm{inc}} \right) \,dS = \int \left( \mathbf{f}_m\times\hat{\mathbf{n}} \right) \cdot \mathbf{H}^{\mathrm{inc}} \,dS .

**Implementation:** The same :cpp:func:`sie_mom::solvers::solve_dielectric_dense`
entry point selects this formulation with ``DielectricFormulation::N_Muller``.
:cpp:func:`sie_mom::solvers::dielectric::launch_n_muller_kernel` launches
:cpp:func:`sie_mom::solvers::dielectric::n_muller_kernel`, which combines the
material-weighted :math:`n\mathcal{TL}` and :math:`n\mathcal{TK}` contributions
and adds the Gram-matrix jump terms on coincident faces. Both integral forms
are available through :cpp:enum:`sie_mom::solvers::NtlIntegralForm`.

GSTC Formulation for SIE (Tested Form)
===============================================

In this section we transcribe the handwritten notes from the PDF into
Sphinx/LaTeX, using the notation already introduced for the SIE operators,
currents, and GSTC blocks.

Admittance GSTC in terms of sums and differences
------------------------------------------------

We start from the admittance GSTC written in block form for the sums of currents
and the averaged tangential fields:

.. math::

   \begin{bmatrix}
     \mathbf J_1 + \mathbf J_2 \\
     \mathbf M_1 + \mathbf M_2
   \end{bmatrix}
   =
   \begin{bmatrix}
     \mathbf Y_{ee} & \mathbf Y_{em} \\
     \mathbf Y_{me} & \mathbf Y_{mm}
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf E_{\mathrm{avg},\tan} \\
     \mathbf H_{\mathrm{avg},\tan}
   \end{bmatrix}.

Using the usual tangential projection
:math:`\mathbf v_{\tan} = -\hat{\mathbf n}\times(\hat{\mathbf n}\times\mathbf v)`,
we rewrite the averaged tangential fields in terms of :math:`\mathbf E_1,\mathbf
E_2,\mathbf H_1,\mathbf H_2`:

.. math::

   \begin{bmatrix}
     \mathbf J_1 + \mathbf J_2 \\
     \mathbf M_1 + \mathbf M_2
   \end{bmatrix}
   =
   \begin{bmatrix}
     \mathbf Y_{ee} & \mathbf Y_{em} \\
     \mathbf Y_{me} & \mathbf Y_{mm}
   \end{bmatrix}
   \begin{bmatrix}
     \dfrac{-\hat{\mathbf n}\times\hat{\mathbf n}\times\mathbf E_1
            -\hat{\mathbf n}\times\hat{\mathbf n}\times\mathbf E_2}{2} \\[6pt]
     \dfrac{-\hat{\mathbf n}\times\hat{\mathbf n}\times\mathbf H_1
            -\hat{\mathbf n}\times\hat{\mathbf n}\times\mathbf H_2}{2}
   \end{bmatrix}.

Using the current–field traces

.. math::

   \mathbf J_1=-\hat{\mathbf n}\times\mathbf H_1,\quad
   \mathbf M_1=\hat{\mathbf n}\times\mathbf E_1,\quad
   \mathbf J_2=\hat{\mathbf n}\times\mathbf H_2,\quad
   \mathbf M_2=-\hat{\mathbf n}\times\mathbf E_2,

one obtains

.. math::

   \begin{bmatrix}
     \mathbf J_1 + \mathbf J_2 \\
     \mathbf M_1 + \mathbf M_2
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathbf Y_{ee} & \mathbf Y_{em} \\
     \mathbf Y_{me} & \mathbf Y_{mm}
   \end{bmatrix}
   \begin{bmatrix}
     -\hat{\mathbf n}\times\mathbf M_1 + \hat{\mathbf n}\times\mathbf M_2 \\
     \hat{\mathbf n}\times\mathbf J_1 - \hat{\mathbf n}\times\mathbf J_2
   \end{bmatrix}.

Expanding the block product explicitly:

.. math::

   \begin{bmatrix}
     \mathbf J_1 + \mathbf J_2 \\
     \mathbf M_1 + \mathbf M_2
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathbf Y_{ee}\bigl(-\hat{\mathbf n}\times\mathbf M_1+\hat{\mathbf n}\times\mathbf M_2\bigr)
     + \mathbf Y_{em}\bigl(\hat{\mathbf n}\times\mathbf J_1-\hat{\mathbf n}\times\mathbf J_2\bigr)
     \\[4pt]
     \mathbf Y_{me}\bigl(-\hat{\mathbf n}\times\mathbf M_1+\hat{\mathbf n}\times\mathbf M_2\bigr)
     + \mathbf Y_{mm}\bigl(\hat{\mathbf n}\times\mathbf J_1-\hat{\mathbf n}\times\mathbf J_2\bigr)
   \end{bmatrix}.

Now apply :math:`\hat{\mathbf n}\times(\cdot)` to both sides of the equation:

.. math::

   \begin{bmatrix}
     \hat{\mathbf n}\times\mathbf J_1 + \hat{\mathbf n}\times\mathbf J_2 \\
     \hat{\mathbf n}\times\mathbf M_1 + \hat{\mathbf n}\times\mathbf M_2
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathbf Y_{ee}^{\perp}(\mathbf M_1 - \mathbf M_2)
     + \mathbf Y_{em}^{\perp}(-\mathbf J_1 + \mathbf J_2) \\[4pt]
     \mathbf Y_{me}^{\perp}(\mathbf M_1 - \mathbf M_2)
     + \mathbf Y_{mm}^{\perp}(-\mathbf J_1 + \mathbf J_2)
   \end{bmatrix},

where the rotated admittance tensors are
:math:`\mathbf Y_{ab}^{\perp}:=-\hat{\mathbf n}\times\mathbf Y_{ab}\,\hat{\mathbf
n}\times`. This definition enforces the identity :math:`\hat{\mathbf n}\times(\mathbf Y_{ab} \mathbf v) = \mathbf Y_{ab}^{\perp} (\hat{\mathbf n}\times\mathbf v)`, allowing the rotation operator to commute with the constitutive tensor so that the GSTC relations can be expressed in terms of the projector traces. (This corresponds to :math:`\mathbf T^{\perp}=-R\mathbf T R`
with :math:`R=\hat{\mathbf n}\times(\cdot)`, absorbing the minus sign from :math:`\hat{\mathbf n}\times\hat{\mathbf n}\times = -(\cdot)_{\tan}`).
   
Next, we **test** this relation with the RWG basis. Denoting by
:math:`\mathcal{I}` the tested identity operator and by :math:`\mathcal{M}[\cdot]`
the tested map of a tangential tensor, we obtain

.. math::
   :label: I_raw

   \begin{bmatrix}
     \mathcal{I}[\mathbf J_1] + \mathcal{I}[\mathbf J_2] \\
     \mathcal{I}[\mathbf M_1] + \mathcal{I}[\mathbf M_2]
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathcal{M}[\mathbf Y_{ee}^{\perp}](\mathbf M_1 - \mathbf M_2)
     + \mathcal{M}[\mathbf Y_{em}^{\perp}](-\mathbf J_1 + \mathbf J_2) \\[6pt]
     \mathcal{M}[\mathbf Y_{me}^{\perp}](\mathbf M_1 - \mathbf M_2)
     + \mathcal{M}[\mathbf Y_{mm}^{\perp}](-\mathbf J_1 + \mathbf J_2)
   \end{bmatrix}.

.. admonition:: What is the "Mass Matrix"?
   :class: tip

   The term **standard mass matrix** refers to the matrix representation of the identity operator :math:`\mathcal{I}` after testing. Its elements are the inner products of the basis functions with each other:

   .. math::
      M_{ij} = \langle \mathbf{f}_i, \mathcal{I}[\mathbf{f}_j] \rangle = \langle \mathbf{f}_i, \mathbf{f}_j \rangle = \int_S \mathbf{f}_i \cdot \mathbf{f}_j \, dS.

   If the basis functions :math:`\{\mathbf{f}_k\}` were orthonormal, this would be the identity matrix. However, RWG basis functions are not orthogonal, so the mass matrix is sparse but not diagonal.

   **Generalized (Weighted) Mass Matrix**

   In this document, we use the notation :math:`\mathcal{M}[\mathbf{T}]` to represent a **generalized** or **weighted mass matrix**, where a material tensor :math:`\mathbf{T}` (like :math:`\mathbf{Z}_{ee}` or :math:`\mathbf{Y}_{ab}^{\perp}`) is included in the inner product:

   .. math::
      \mathcal{M}[\mathbf{T}]_{ij} = \langle \mathbf{f}_i, \mathbf{T} \mathbf{f}_j \rangle = \int_S \mathbf{f}_i \cdot (\mathbf{T} \mathbf{f}_j) \, dS.

   This is a standard construction in the Finite Element Method (FEM).

   **Why the name "mass matrix"?**
   If the basis functions :math:`\{\mathbf{f}_k\}` were orthonormal, this inner product would be the Kronecker delta :math:`\delta_{ij}`, and the mass matrix would be the identity matrix. However, the RWG basis functions are not orthogonal, so the mass matrix is a sparse (but not diagonal) matrix.
   The name is borrowed from FEM, particularly in structural mechanics. In solving dynamic equations like :math:`\mathbf{M}\ddot{\mathbf{u}} + \mathbf{K}\mathbf{u} = \mathbf{F}`, the matrix :math:`\mathbf{M}` multiplying the acceleration term arises from testing the mass density of the system and has the same mathematical form as our matrix of inner products. This terminology is widely adopted in computational electromagnetics for matrices without spatial derivatives.

   *Reference: J. M. Jin, "The Finite Element Method in Electromagnetics," 3rd ed., Wiley-IEEE Press, 2014.*
   

SIE representations for the tested identity
-------------------------------------------

From the four tested SIE equations, we can solve for
:math:`\mathcal{I}[\mathbf M_p]` and :math:`\mathcal{I}[\mathbf J_p]`
(:math:`p=1,2`) in terms of the single- and double-layer operators
and the excitation vectors:

.. math::

   \mathcal{I}[\mathbf M_1]
   = 2\Bigl\{j\omega\mu_1\,\mathcal{TL}_{k_1}[\mathbf J_1]
              + \mathcal{TK}_{k_1}[\mathbf M_1]\Bigr\},

.. math::

   \mathcal{I}[\mathbf M_2]
   = 2\Bigl\{-j\omega\mu_2\,\mathcal{TL}_{k_2}[\mathbf J_2]
             - \mathcal{TK}_{k_2}[\mathbf M_2]\Bigr\}
     + 2\,\mathbf V_E,

.. math::

   \mathcal{I}[\mathbf J_1]
   = 2\Bigl\{-j\omega\varepsilon_1\,\mathcal{TL}_{k_1}[\mathbf M_1]
             + \mathcal{TK}_{k_1}[\mathbf J_1]\Bigr\},

.. math::

   \mathcal{I}[\mathbf J_2]
   = 2\Bigl\{j\omega\varepsilon_2\,\mathcal{TL}_{k_2}[\mathbf M_2]
             - \mathcal{TK}_{k_2}[\mathbf J_2]
             - \mathbf V_H\Bigr\}.

(These four relations are exactly the SIE equations
:eq:`eq:sie1_testing`–:eq:`eq:sie4_testing` solved for the tested identity
applied to each current.)

The idea in the handwritten notes is to **substitute** these expressions into
the raw relation :eq:`I_raw` to obtain a full ``\mathbf Y``–representation in
terms of the Calderón-like operators.

Impedance GSTC for sums of currents
-----------------------------------

We now repeat the same procedure using the **impedance** GSTC:

.. math::

   \begin{bmatrix}
     \mathbf E_{\mathrm{avg},\tan} \\
     \mathbf H_{\mathrm{avg},\tan}
   \end{bmatrix}
   =
   \begin{bmatrix}
     \mathbf Z_{ee} & \mathbf Z_{em} \\
     \mathbf Z_{me} & \mathbf Z_{mm}
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf J_1 + \mathbf J_2 \\
     \mathbf M_1 + \mathbf M_2
   \end{bmatrix}.

Using again the trace identities, the combinations

.. math::

   \begin{bmatrix}
     -\hat{\mathbf n}\times\mathbf M_1 + \hat{\mathbf n}\times\mathbf M_2 \\
     \hat{\mathbf n}\times\mathbf J_1 - \hat{\mathbf n}\times\mathbf J_2
   \end{bmatrix}
   =
   2
   \begin{bmatrix}
     \mathbf E_{\mathrm{avg},\tan} \\
     \mathbf H_{\mathrm{avg},\tan}
   \end{bmatrix},

so that

.. math::

   \begin{bmatrix}
     -\hat{\mathbf n}\times\mathbf M_1 + \hat{\mathbf n}\times\mathbf M_2 \\
     \hat{\mathbf n}\times\mathbf J_1 - \hat{\mathbf n}\times\mathbf J_2
   \end{bmatrix}
   =
   2
   \begin{bmatrix}
     \mathbf Z_{ee} & \mathbf Z_{em} \\
     \mathbf Z_{me} & \mathbf Z_{mm}
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf J_1 + \mathbf J_2 \\
     \mathbf M_1 + \mathbf M_2
   \end{bmatrix}.

Testing with RWG functions gives

.. math::
   :label: II_raw

   \begin{bmatrix}
     -\mathcal{I}[\mathbf M_1] + \mathcal{I}[\mathbf M_2] \\
     \mathcal{I}[\mathbf J_1] - \mathcal{I}[\mathbf J_2]
   \end{bmatrix}
   =
   2
   \begin{bmatrix}
     \mathcal{M}[\mathbf Z_{ee}]\,(\mathbf J_1+\mathbf J_2)
     + \mathcal{M}[\mathbf Z_{em}]\,(\mathbf M_1+\mathbf M_2) \\[6pt]
     \mathcal{M}[\mathbf Z_{me}]\,(\mathbf J_1+\mathbf J_2)
     + \mathcal{M}[\mathbf Z_{mm}]\,(\mathbf M_1+\mathbf M_2)
   \end{bmatrix}.
   

Summary in compact block form
-----------------------------

Collecting the two raw tested relations, the handwritten notes summarize them as

.. math::
   :label: I

   \begin{bmatrix}
     \mathcal{I}[\mathbf J_1] + \mathcal{I}[\mathbf J_2] \\
     \mathcal{I}[\mathbf M_1] + \mathcal{I}[\mathbf M_2]
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathcal{M}[\mathbf Y_{ee}^{\perp}] & \mathcal{M}[\mathbf Y_{em}^{\perp}] \\
     \mathcal{M}[\mathbf Y_{me}^{\perp}] & \mathcal{M}[\mathbf Y_{mm}^{\perp}]
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf M_1 - \mathbf M_2 \\
     -\mathbf J_1 + \mathbf J_2
   \end{bmatrix},

and

.. math::
   :label: II

   \begin{bmatrix}
     -\mathcal{I}[\mathbf M_1] + \mathcal{I}[\mathbf M_2] \\
     \mathcal{I}[\mathbf J_1] - \mathcal{I}[\mathbf J_2]
   \end{bmatrix}
   =
   2
   \begin{bmatrix}
     \mathcal{M}[\mathbf Z_{ee}] & \mathcal{M}[\mathbf Z_{em}] \\
     \mathcal{M}[\mathbf Z_{me}] & \mathcal{M}[\mathbf Z_{mm}]
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf J_1 + \mathbf J_2 \\
     \mathbf M_1 + \mathbf M_2
   \end{bmatrix}.

Finally, as noted in the original handwritten comments, substituting the SIE
representations for :math:`\mathcal{I}[\mathbf J_p]` and
:math:`\mathcal{I}[\mathbf M_p]` into (I) and (II) yields **two coupled tested
integral equations** in terms of the Calderón-like operators and the GSTC
blocks, providing a formulation closely aligned with the general BIE theory.


Derivation of the Final Coupled Equations
-----------------------------------------

As stated, the next step is to substitute the SIE representations for the tested
identity operators into the summary equations :eq:`I` and :eq:`II`. This will
produce two coupled matrix equations, which together form the complete 4x4 block
system for the unknown currents :math:`\mathbf{J}_1, \mathbf{M}_1, \mathbf{J}_2, \mathbf{M}_2`.

First Equation (from Impedance GSTC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We start with Equation :eq:`II`, which is driven by the impedance tensor
:math:`\mathbf{Z}`:

.. math::

   \begin{bmatrix}
     -\mathcal{I}[\mathbf M_1] + \mathcal{I}[\mathbf M_2] \\
     \mathcal{I}[\mathbf J_1] - \mathcal{I}[\mathbf J_2]
   \end{bmatrix}
   =
   2
   \begin{bmatrix}
     \mathcal{M}[\mathbf Z_{ee}]\,(\mathbf J_1+\mathbf J_2)
     + \mathcal{M}[\mathbf Z_{em}]\,(\mathbf M_1+\mathbf M_2) \\[6pt]
     \mathcal{M}[\mathbf Z_{me}]\,(\mathbf J_1+\mathbf J_2)
     + \mathcal{M}[\mathbf Z_{mm}]\,(\mathbf M_1+\mathbf M_2)
   \end{bmatrix}.

We substitute the expressions for :math:`\mathcal{I}[\mathbf J_p]` and
:math:`\mathcal{I}[\mathbf M_p]` into the left-hand side. For the top row,
:math:`-\mathcal{I}[\mathbf M_1] + \mathcal{I}[\mathbf M_2]` becomes:

.. math::

   -2\Bigl\{j\omega\mu_1\,\mathcal{TL}_{k_1}[\mathbf J_1] + \mathcal{TK}_{k_1}[\mathbf M_1]\Bigr\}
   + 2\Bigl\{-j\omega\mu_2\,\mathcal{TL}_{k_2}[\mathbf J_2] - \mathcal{TK}_{k_2}[\mathbf M_2]\Bigr\} + 2\,\mathbf V_E.

For the bottom row, :math:`\mathcal{I}[\mathbf J_1] - \mathcal{I}[\mathbf J_2]` becomes:

.. math::

   2\Bigl\{-j\omega\varepsilon_1\,\mathcal{TL}_{k_1}[\mathbf M_1] + \mathcal{TK}_{k_1}[\mathbf J_1]\Bigr\}
   - 2\Bigl\{j\omega\varepsilon_2\,\mathcal{TL}_{k_2}[\mathbf M_2] - \mathcal{TK}_{k_2}[\mathbf J_2] - \mathbf V_H\Bigr\}.

We can now equate the substituted left-hand side with the right-hand side of
the original equation. Dividing the entire system by -2, moving the excitation
vectors :math:`\mathbf{V}_E` and :math:`\mathbf{V}_H` to the right, and collecting all
terms involving the unknown currents on the left yields the first of our two
final equations:

.. math::
   :label: eq:final_gstc_1

   \begin{aligned}
      & \begin{bmatrix}
        j\omega\mu_1\mathcal{TL}_{k_1} & \mathcal{TK}_{k_1} \\
        -\mathcal{TK}_{k_1} & j\omega\varepsilon_1\mathcal{TL}_{k_1}
      \end{bmatrix}
      \begin{bmatrix} \mathbf{J}_1 \\ \mathbf{M}_1 \end{bmatrix}
      +
      \begin{bmatrix}
        j\omega\mu_2\mathcal{TL}_{k_2} & \mathcal{TK}_{k_2} \\
        -\mathcal{TK}_{k_2} & j\omega\varepsilon_2\mathcal{TL}_{k_2}
      \end{bmatrix}
      \begin{bmatrix} \mathbf{J}_2 \\ \mathbf{M}_2 \end{bmatrix} \\
      & +
      \begin{bmatrix}
        \mathcal{M}[\mathbf{Z}_{ee}] & \mathcal{M}[\mathbf{Z}_{em}] \\
        \mathcal{M}[\mathbf{Z}_{me}] & \mathcal{M}[\mathbf{Z}_{mm}]
      \end{bmatrix}
      \begin{bmatrix} \mathbf{J}_1+\mathbf{J}_2 \\ \mathbf{M}_1+\mathbf{M}_2 \end{bmatrix}
      =
      \begin{bmatrix} \mathbf{V}_E \\ \mathbf{V}_H \end{bmatrix}.
   \end{aligned}

This equation connects all four unknown currents through the Calderón operators
and the impedance GSTC terms.

Second Equation (from Admittance GSTC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we follow the same procedure for Equation :eq:`I`, which is driven by the
rotated admittance tensor :math:`\mathbf{Y}^{\perp}`:

.. math::

   \begin{bmatrix}
     \mathcal{I}[\mathbf J_1] + \mathcal{I}[\mathbf J_2] \\
     \mathcal{I}[\mathbf M_1] + \mathcal{I}[\mathbf M_2]
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathcal{M}[\mathbf Y_{ee}^{\perp}] & \mathcal{M}[\mathbf Y_{em}^{\perp}] \\
     \mathcal{M}[\mathbf Y_{me}^{\perp}] & \mathcal{M}[\mathbf Y_{mm}^{\perp}]
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf M_1 - \mathbf M_2 \\
     -\mathbf J_1 + \mathbf J_2
   \end{bmatrix}.

Let's swap the rows for clarity:

.. math::
   :label: admittance_swapped

   \begin{bmatrix}
     \mathcal{I}[\mathbf M_1] + \mathcal{I}[\mathbf M_2] \\
     \mathcal{I}[\mathbf J_1] + \mathcal{I}[\mathbf J_2]
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathcal{M}[\mathbf Y_{me}^{\perp}] & \mathcal{M}[\mathbf Y_{mm}^{\perp}] \\
     \mathcal{M}[\mathbf Y_{ee}^{\perp}] & \mathcal{M}[\mathbf Y_{em}^{\perp}]
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf M_1 - \mathbf M_2 \\
     -\mathbf J_1 + \mathbf J_2
   \end{bmatrix}.

We substitute the SIE representations into the left-hand side. The sum
:math:`\mathcal{I}[\mathbf J_1] + \mathcal{I}[\mathbf J_2]` becomes:

.. math::

   2\Bigl\{-j\omega\varepsilon_1\,\mathcal{TL}_{k_1}[\mathbf M_1] + \mathcal{TK}_{k_1}[\mathbf J_1]\Bigr\}
   + 2\Bigl\{j\omega\varepsilon_2\,\mathcal{TL}_{k_2}[\mathbf M_2] - \mathcal{TK}_{k_2}[\mathbf J_2] - \mathbf V_H\Bigr\}.

And the sum :math:`\mathcal{I}[\mathbf M_1] + \mathcal{I}[\mathbf M_2]` becomes:

.. math::

   2\Bigl\{j\omega\mu_1\,\mathcal{TL}_{k_1}[\mathbf J_1] + \mathcal{TK}_{k_1}[\mathbf M_1]\Bigr\}
   + 2\Bigl\{-j\omega\mu_2\,\mathcal{TL}_{k_2}[\mathbf J_2] - \mathcal{TK}_{k_2}[\mathbf M_2]\Bigr\} + 2\,\mathbf V_E.

Now substitute these expressions into the left-hand side of :eq:`admittance_swapped`.   

.. math::

   2
   \begin{bmatrix}
     j\omega\mu_1\mathcal{TL}_{k_1}[\mathbf J_1] + \mathcal{TK}_{k_1}[\mathbf M_1] - j\omega\mu_2\mathcal{TL}_{k_2}[\mathbf J_2] - \mathcal{TK}_{k_2}[\mathbf M_2] + \mathbf V_E \\
     \mathcal{TK}_{k_1}[\mathbf J_1] - j\omega\varepsilon_1\mathcal{TL}_{k_1}[\mathbf M_1] - \mathcal{TK}_{k_2}[\mathbf J_2] + j\omega\varepsilon_2\mathcal{TL}_{k_2}[\mathbf M_2] - \mathbf V_H
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathcal{M}[\mathbf Y_{me}^{\perp}] & \mathcal{M}[\mathbf Y_{mm}^{\perp}] \\
     \mathcal{M}[\mathbf Y_{ee}^{\perp}] & \mathcal{M}[\mathbf Y_{em}^{\perp}]
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf M_1 - \mathbf M_2 \\
     -\mathbf J_1 + \mathbf J_2
   \end{bmatrix}.

To align the right-hand side with the standard difference current vector, we
rearrange the terms. This involves swapping the columns of the mass matrix and
negating the appropriate terms to match the desired vector
:math:`[\mathbf{J}_2 - \mathbf{J}_1, \mathbf{M}_2 - \mathbf{M}_1]^T`:

.. math::

   2
   \begin{bmatrix}
     j\omega\mu_1\mathcal{TL}_{k_1}[\mathbf J_1] + \mathcal{TK}_{k_1}[\mathbf M_1] - j\omega\mu_2\mathcal{TL}_{k_2}[\mathbf J_2] - \mathcal{TK}_{k_2}[\mathbf M_2] + \mathbf V_E \\
     \mathcal{TK}_{k_1}[\mathbf J_1] - j\omega\varepsilon_1\mathcal{TL}_{k_1}[\mathbf M_1] - \mathcal{TK}_{k_2}[\mathbf J_2] + j\omega\varepsilon_2\mathcal{TL}_{k_2}[\mathbf M_2] - \mathbf V_H
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathcal{M}[\mathbf Y_{mm}^{\perp}] & -\mathcal{M}[\mathbf Y_{me}^{\perp}] \\
     \mathcal{M}[\mathbf Y_{em}^{\perp}] & -\mathcal{M}[\mathbf Y_{ee}^{\perp}]
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf J_2 - \mathbf J_1 \\
     \mathbf M_2 - \mathbf M_1
   \end{bmatrix}.

Multiplying the second row by -1 to adjust signs gives:

.. math::

   2
   \begin{bmatrix}
     j\omega\mu_1\mathcal{TL}_{k_1}[\mathbf J_1] + \mathcal{TK}_{k_1}[\mathbf M_1] - j\omega\mu_2\mathcal{TL}_{k_2}[\mathbf J_2] - \mathcal{TK}_{k_2}[\mathbf M_2] + \mathbf V_E \\
     -\mathcal{TK}_{k_1}[\mathbf J_1] + j\omega\varepsilon_1\mathcal{TL}_{k_1}[\mathbf M_1] + \mathcal{TK}_{k_2}[\mathbf J_2] - j\omega\varepsilon_2\mathcal{TL}_{k_2}[\mathbf M_2] + \mathbf V_H
   \end{bmatrix}
   =
   \frac12
   \begin{bmatrix}
     \mathcal{M}[\mathbf Y_{mm}^{\perp}] & -\mathcal{M}[\mathbf Y_{me}^{\perp}] \\
     -\mathcal{M}[\mathbf Y_{em}^{\perp}] & \mathcal{M}[\mathbf Y_{ee}^{\perp}]
   \end{bmatrix}
   \begin{bmatrix}
     \mathbf J_2 - \mathbf J_1 \\
     \mathbf M_2 - \mathbf M_1
   \end{bmatrix}.


Finally, we divide the entire system by -2 and rearrange the terms, moving all
operators to the left-hand side and the excitation vectors to the right-hand
side. This gives the second final equation:

.. math::
   :label: eq:final_gstc_2

   \begin{aligned}
      &
      \begin{bmatrix}
        j\omega\mu_2\mathcal{TL}_{k_2} & \mathcal{TK}_{k_2} \\
        -\mathcal{TK}_{k_2} & j\omega\varepsilon_2\mathcal{TL}_{k_2}
      \end{bmatrix}
      \begin{bmatrix} \mathbf{J}_2 \\ \mathbf{M}_2 \end{bmatrix}
      -
      \begin{bmatrix}
        j\omega\mu_1\mathcal{TL}_{k_1} & \mathcal{TK}_{k_1} \\
        -\mathcal{TK}_{k_1} & j\omega\varepsilon_1\mathcal{TL}_{k_1}
      \end{bmatrix}
      \begin{bmatrix} \mathbf{J}_1 \\ \mathbf{M}_1 \end{bmatrix} \\
      & + \frac14
      \begin{bmatrix}
        \mathcal{M}[\mathbf{Y}_{mm}^{\perp}] & -\mathcal{M}[\mathbf{Y}_{me}^{\perp}] \\
        -\mathcal{M}[\mathbf{Y}_{em}^{\perp}] & \mathcal{M}[\mathbf{Y}_{ee}^{\perp}]
      \end{bmatrix}
      \begin{bmatrix} \mathbf{J}_2-\mathbf{J}_1 \\ \mathbf{M}_2-\mathbf{M}_1 \end{bmatrix}
      =
      \begin{bmatrix} \mathbf{V}_E \\ \mathbf{V}_H \end{bmatrix}.
   \end{aligned}




Final 4x4 Block Matrix System
-----------------------------

The two coupled equations, :eq:`eq:final_gstc_1` and :eq:`eq:final_gstc_2`, combine
into a single 4x4 block matrix system. To better reflect the implementation
strategy where non-local and local operators are assembled separately, we can
split the final system matrix :math:`\mathbf{A}` into two parts: a Calderón
operator matrix :math:`\mathbf{A_C}` and a GSTC mass-matrix :math:`\mathbf{A_M}`.

The final system takes the form:

.. math::
   (\mathbf{A_C} + \mathbf{A_M})\mathbf{x} = \mathbf{b}

**Implementation:** :cpp:func:`sie_mom::solvers::solve_gstc_dense` assembles
the row-transformed system described below in
:ref:`gstc_transformed_implementation`, then solves by LU factorization.
Its production assembly uses
:cpp:func:`::assemble_Z_calderon_eta0_scaled_kernel` and
:cpp:func:`::assemble_Z_mass_eta0_scaled_kernel`; the magnetic-current columns
and magnetic-equation rows carry vacuum-impedance scaling. The returned
coefficients are converted back to physical currents.
:cpp:func:`::assemble_V_vector_eta0_scaled_kernel` assembles the excitation
with the corresponding row scaling and source-region placement.
:cpp:func:`sie_mom::solvers::solve_gstc_dense_batch` reuses the assembled matrix
and its LU factorization for multiple excitations.

The unknown vector :math:`\mathbf{x}` and the excitation vector :math:`\mathbf{b}` are:

.. math::
   \mathbf{x} =
   \begin{bmatrix}
      \mathbf{J}_1 \\ \mathbf{M}_1 \\ \mathbf{J}_2 \\ \mathbf{M}_2
   \end{bmatrix},
   \qquad
   \mathbf{b} =
   \begin{bmatrix}
      \mathbf{V}_E \\ \mathbf{V}_H \\ \mathbf{V}_E \\ \mathbf{V}_H
   \end{bmatrix}.

The non-local **Calderón Operator Matrix** :math:`\mathbf{A_C}` contains the
:math:`\mathcal{TL}_k` and :math:`\mathcal{TK}_k` operators and is assembled using a
double loop over faces:

.. math::
   :label: eq:final_4x4_calderon

   \mathbf{A_C} =
   \begin{bmatrix}
      j\omega\mu_1\mathcal{TL}_{k_1} & \mathcal{TK}_{k_1} & j\omega\mu_2\mathcal{TL}_{k_2} & \mathcal{TK}_{k_2} \\
      -\mathcal{TK}_{k_1} & j\omega\varepsilon_1\mathcal{TL}_{k_1} & -\mathcal{TK}_{k_2} & j\omega\varepsilon_2\mathcal{TL}_{k_2} \\
      -j\omega\mu_1\mathcal{TL}_{k_1} & -\mathcal{TK}_{k_1} & j\omega\mu_2\mathcal{TL}_{k_2} & \mathcal{TK}_{k_2} \\
      \mathcal{TK}_{k_1} & -j\omega\varepsilon_1\mathcal{TL}_{k_1} & -\mathcal{TK}_{k_2} & j\omega\varepsilon_2\mathcal{TL}_{k_2}
   \end{bmatrix}.

.. note::
   This matrix has a clear structure based on the per-side Calderón operators
   (without jump terms), which we can define as:

   .. math::
      \mathbf{\tilde{C}}^{(p)} = \begin{bmatrix}
         j\omega\mu_p\mathcal{TL}_{k_p} & \mathcal{TK}_{k_p} \\
         -\mathcal{TK}_{k_p} & j\omega\varepsilon_p\mathcal{TL}_{k_p}
      \end{bmatrix}.

   The full operator matrix can then be written compactly in terms of these
   2x2 blocks:

   .. math::
      \mathbf{A_C} = \begin{bmatrix}
         \mathbf{\tilde{C}}^{(1)} & \mathbf{\tilde{C}}^{(2)} \\
        -\mathbf{\tilde{C}}^{(1)} & \mathbf{\tilde{C}}^{(2)}
      \end{bmatrix}.

The local **GSTC Mass Matrix** :math:`\mathbf{A_M}` contains the
:math:`\mathcal{M}[\cdot]` operators and is assembled using a single loop over faces:

.. math::
   :label: eq:final_4x4_gstc

   \mathbf{A_M} =
   \begin{bmatrix}
      \mathcal{M}[\mathbf{Z}_{ee}] & \mathcal{M}[\mathbf{Z}_{em}] & \mathcal{M}[\mathbf{Z}_{ee}] & \mathcal{M}[\mathbf{Z}_{em}] \\
      \mathcal{M}[\mathbf{Z}_{me}] & \mathcal{M}[\mathbf{Z}_{mm}] & \mathcal{M}[\mathbf{Z}_{me}] & \mathcal{M}[\mathbf{Z}_{mm}] \\
      -\tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{mm}^{\perp}] & \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{me}^{\perp}] & \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{mm}^{\perp}] & -\tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{me}^{\perp}] \\
      \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{em}^{\perp}] & -\tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{ee}^{\perp}] & -\tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{em}^{\perp}] & \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{ee}^{\perp}]
   \end{bmatrix}.

.. admonition:: Derivation of the GSTC Mass Matrix

   The blocks of :math:`\mathbf{A_M}` are found by expanding the GSTC terms in
   :eq:`eq:final_gstc_1` and :eq:`eq:final_gstc_2` and collecting the coefficients
   for each current in the unknown vector
   :math:`\mathbf{x} = [\mathbf{J}_1, \mathbf{M}_1, \mathbf{J}_2, \mathbf{M}_2]^T`.

   **Rows 1 and 2 (from Impedance Term):**

   The impedance term from :eq:`eq:final_gstc_1` is:

   .. math::
      + \begin{bmatrix} \mathcal{M}[\mathbf{Z}_{ee}] & \mathcal{M}[\mathbf{Z}_{em}] \\ \mathcal{M}[\mathbf{Z}_{me}] & \mathcal{M}[\mathbf{Z}_{mm}] \end{bmatrix} \begin{bmatrix} \mathbf{J}_1+\mathbf{J}_2 \\ \mathbf{M}_1+\mathbf{M}_2 \end{bmatrix}.

   The top row expands to :math:`+\mathcal{M}[\mathbf{Z}_{ee}]\mathbf{J}_1 + \mathcal{M}[\mathbf{Z}_{em}]\mathbf{M}_1 + \mathcal{M}[\mathbf{Z}_{ee}]\mathbf{J}_2 + \mathcal{M}[\mathbf{Z}_{em}]\mathbf{M}_2`.
   These four terms directly populate the first row of :math:`\mathbf{A_M}`. The second row follows the same pattern.

   **Rows 3 and 4 (from Admittance Term):**

   The admittance term from :eq:`eq:final_gstc_2` is:

   .. math::
      +\frac{1}{4} \begin{bmatrix} \mathcal{M}[\mathbf{Y}_{mm}^{\perp}] & -\mathcal{M}[\mathbf{Y}_{me}^{\perp}] \\ -\mathcal{M}[\mathbf{Y}_{em}^{\perp}] & \mathcal{M}[\mathbf{Y}_{ee}^{\perp}] \end{bmatrix} \begin{bmatrix} \mathbf{J}_2-\mathbf{J}_1 \\ \mathbf{M}_2-\mathbf{M}_1 \end{bmatrix}.

   The top row of this term expands to :math:`+\frac{1}{4}\mathcal{M}\mathbf{Y}_{mm}^{\perp} - \frac{1}{4}\mathcal{M}\mathbf{Y}_{me}^{\perp}`.
   Collecting coefficients gives:

   - For :math:`\mathbf{J}_1`: :math:`-\frac{1}{4}\mathcal{M}[\mathbf{Y}_{mm}^{\perp}]`
   - For :math:`\mathbf{M}_1`: :math:`+\frac{1}{4}\mathcal{M}[\mathbf{Y}_{me}^{\perp}]`
   - For :math:`\mathbf{J}_2`: :math:`+\frac{1}{4}\mathcal{M}[\mathbf{Y}_{mm}^{\perp}]`
   - For :math:`\mathbf{M}_2`: :math:`-\frac{1}{4}\mathcal{M}[\mathbf{Y}_{me}^{\perp}]`

   These four terms populate the third row of :math:`\mathbf{A_M}`. The fourth row is derived similarly.


.. _gstc_transformed_implementation:

Transformed System via Row Operations
-------------------------------------

To better understand the structure of the coupled system, we can perform a set of
row operations on the final system :math:`(\mathbf{A_C} + \mathbf{A_M})\mathbf{x} = \mathbf{b}`.
Let the new system be :math:`(\mathbf{A_C}'' + \mathbf{A_M}'')\mathbf{x} = \mathbf{b}''`,
where the new rows are linear combinations of the old rows:

.. math::
   \begin{aligned}
      \text{new\_row}_1 &= \text{old\_row}_1 - \text{old\_row}_3 \\
      \text{new\_row}_2 &= \text{old\_row}_2 - \text{old\_row}_4
   \end{aligned}
   \qquad
   \begin{aligned}
      \text{new\_row}_3 &= \text{old\_row}_1 + \text{old\_row}_3 \\
      \text{new\_row}_4 &= \text{old\_row}_2 + \text{old\_row}_4
   \end{aligned}

Applying these transformations and then **dividing the entire system by 2**
yields a new system with a revealing structure.

**Transformed Excitation Vector** :math:`\mathbf{b}''`

The transformation makes the top half of the excitation vector zero, and the
division by 2 simplifies the bottom half:

.. math::
   \mathbf{b}'' = \frac{1}{2}
   \begin{bmatrix}
      \mathbf{V}_E - \mathbf{V}_E \\ \mathbf{V}_H - \mathbf{V}_H \\
      \mathbf{V}_E + \mathbf{V}_E \\ \mathbf{V}_H + \mathbf{V}_H
   \end{bmatrix}
   =
   \begin{bmatrix}
      \mathbf{0} \\ \mathbf{0} \\ \mathbf{V}_E \\ \mathbf{V}_H
   \end{bmatrix}.

**Transformed Calderón Matrix** :math:`\mathbf{A_C}''`

The row operations and subsequent division by 2 perfectly decouple the
Calderón operators for the two regions into a block-diagonal matrix:

.. math::
   \mathbf{A_C}'' =
   \begin{bmatrix}
      j\omega\mu_1\mathcal{TL}_{k_1} & \mathcal{TK}_{k_1} & 0 & 0 \\
      -\mathcal{TK}_{k_1} & j\omega\varepsilon_1\mathcal{TL}_{k_1} & 0 & 0 \\
      0 & 0 & j\omega\mu_2\mathcal{TL}_{k_2} & \mathcal{TK}_{k_2} \\
      0 & 0 & -\mathcal{TK}_{k_2} & j\omega\varepsilon_2\mathcal{TL}_{k_2}
   \end{bmatrix}
   =
   \begin{bmatrix}
      \mathbf{\tilde{C}}^{(1)} & \mathbf{0} \\
      \mathbf{0} & \mathbf{\tilde{C}}^{(2)}
   \end{bmatrix}.

**Transformed GSTC Mass Matrix** :math:`\mathbf{A_M}''`

**Implementation:** :cpp:func:`::assemble_Z_calderon_kernel` assembles the
block-diagonal :math:`\mathbf{A_C}''` above using a face-pair loop.
:cpp:func:`::assemble_Z_mass_kernel` assembles :math:`\mathbf{A_M}''` below
using one thread per face, including the factors :math:`1/2` and :math:`1/8`.
The dense solver uses the corresponding
:cpp:func:`::assemble_Z_calderon_eta0_scaled_kernel` and
:cpp:func:`::assemble_Z_mass_eta0_scaled_kernel` with vacuum-impedance scaling.

The mass matrix transforms into a new structure where all terms are scaled by 1/2:

.. math::
   \mathbf{A_M}'' = \frac{1}{2}
   \begin{bmatrix}
      \mathcal{M}[\mathbf{Z}_{ee}] + \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{mm}^{\perp}] & \mathcal{M}[\mathbf{Z}_{em}] - \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{me}^{\perp}] & \mathcal{M}[\mathbf{Z}_{ee}] - \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{mm}^{\perp}] & \mathcal{M}[\mathbf{Z}_{em}] + \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{me}^{\perp}] \\
      \mathcal{M}[\mathbf{Z}_{me}] - \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{em}^{\perp}] & \mathcal{M}[\mathbf{Z}_{mm}] + \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{ee}^{\perp}] & \mathcal{M}[\mathbf{Z}_{me}] + \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{em}^{\perp}] & \mathcal{M}[\mathbf{Z}_{mm}] - \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{ee}^{\perp}] \\
      \mathcal{M}[\mathbf{Z}_{ee}] - \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{mm}^{\perp}] & \mathcal{M}[\mathbf{Z}_{em}] + \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{me}^{\perp}] & \mathcal{M}[\mathbf{Z}_{ee}] + \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{mm}^{\perp}] & \mathcal{M}[\mathbf{Z}_{em}] - \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{me}^{\perp}] \\
      \mathcal{M}[\mathbf{Z}_{me}] + \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{em}^{\perp}] & \mathcal{M}[\mathbf{Z}_{mm}] - \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{ee}^{\perp}] & \mathcal{M}[\mathbf{Z}_{me}] - \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{em}^{\perp}] & \mathcal{M}[\mathbf{Z}_{mm}] + \tfrac{1}{4}\mathcal{M}[\mathbf{Y}_{ee}^{\perp}]
   \end{bmatrix}.
   
This transformed system provides an alternative but equivalent view of the problem,
highlighting how the interior and exterior fields are coupled through the GSTC
mass matrix terms.




.. _gstc_implementation_note:

Handling Anisotropic and Bianisotropic Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A crucial point for general implementations is the correct representation and handling of the GSTC tensors (:math:`\mathbf{Y}_{ab}`, :math:`\mathbf{Z}_{ab}`) when they are anisotropic or when the material is bianisotropic (magneto-electric coupling).


Handling Bianisotropy: The 4x4 Inversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GSTC formulation often defines the surface properties in terms of an Admittance matrix :math:`\mathbb{Y}` relating fields to currents:

.. math::

   \begin{pmatrix} \mathbf{J}_S \\ \mathbf{M}_S \end{pmatrix} = 
   \underbrace{\begin{pmatrix} \bar{\bar{\mathbf{Y}}}_{ee} & \bar{\bar{\mathbf{Y}}}_{em} \\ \bar{\bar{\mathbf{Y}}}_{me} & \bar{\bar{\mathbf{Y}}}_{mm} \end{pmatrix}}_{\mathbb{Y}_{tot}}
   \begin{pmatrix} \mathbf{E}_{avg} \\ \mathbf{H}_{avg} \end{pmatrix}

However, the final integral equation system requires Impedance terms (:math:`\mathbb{Z}`) for the magnetic current equations. For general bianisotropic materials (where :math:`\bar{\bar{\mathbf{Y}}}_{em} \neq 0`), we cannot simply invert the diagonal blocks. We must invert the full structure:

.. math::

   \begin{pmatrix} \mathbf{E}_{avg} \\ \mathbf{H}_{avg} \end{pmatrix} = 
   \mathbb{Z}_{tot} \begin{pmatrix} \mathbf{J}_S \\ \mathbf{M}_S \end{pmatrix}
   \quad \text{where} \quad
   \mathbb{Z}_{tot} = \mathbb{Y}_{tot}^{-1}

This operation should be performed at the "Physical (2x2)" level for numerical stability before rotating to 3D. The resulting 4x4 matrix is then split back into four 2x2 blocks (:math:`\bar{\bar{\mathbf{Z}}}_{ee}, \bar{\bar{\mathbf{Z}}}_{em}, \dots`), each of which is rotated to the 3D Cartesian basis separately.

Detailed Matrix-Vector Mapping
""""""""""""""""""""""""""""""

The 4x4 scalar matrix used in the inversion is the "under the hood" representation of the 2x2 tensor block matrix. They are mathematically identical.

**The Vector Representation**

The "state vector" at any point on the surface is a 4-element column vector formed by stacking the two tangential vectors components (e.g., :math:`u, v` in a local basis):

.. math::

   \mathbf{U}_{field} = \begin{pmatrix} E_u \\ E_v \\ H_u \\ H_v \end{pmatrix} \quad \text{and} \quad \mathbf{U}_{current} = \begin{pmatrix} J_u \\ J_v \\ M_u \\ M_v \end{pmatrix}

**The Matrix-Block Equivalence**

When we write the tensor operation in block form:

.. math::

   \begin{pmatrix} \mathbf{J} \\ \mathbf{M} \end{pmatrix} = 
   \begin{pmatrix} \bar{\bar{\mathbf{Y}}}_{ee} & \bar{\bar{\mathbf{Y}}}_{em} \\ \bar{\bar{\mathbf{Y}}}_{me} & \bar{\bar{\mathbf{Y}}}_{mm} \end{pmatrix}
   \begin{pmatrix} \mathbf{E} \\ \mathbf{H} \end{pmatrix}

It expands to:

.. math::

   \mathbf{J} = \bar{\bar{\mathbf{Y}}}_{ee} \cdot \mathbf{E} + \bar{\bar{\mathbf{Y}}}_{em} \cdot \mathbf{H}
   \quad \text{and} \quad
   \mathbf{M} = \bar{\bar{\mathbf{Y}}}_{me} \cdot \mathbf{E} + \bar{\bar{\mathbf{Y}}}_{mm} \cdot \mathbf{H}

This matches exactly how the 4x4 matrix multiplication works on the components:

.. math::

   \begin{pmatrix} J_u \\ J_v \\ M_u \\ M_v \end{pmatrix} = 
   \begin{pmatrix} 
   Y_{ee}^{uu} & Y_{ee}^{uv} & Y_{em}^{uu} & Y_{em}^{uv} \\ 
   Y_{ee}^{vu} & Y_{ee}^{vv} & Y_{em}^{vu} & Y_{em}^{vv} \\
   Y_{me}^{uu} & Y_{me}^{uv} & Y_{mm}^{uu} & Y_{mm}^{uv} \\
   Y_{me}^{vu} & Y_{me}^{vv} & Y_{mm}^{vu} & Y_{mm}^{vv}
   \end{pmatrix} 
   \begin{pmatrix} E_{u,avg} \\ E_{v,avg} \\ H_{u,avg} \\ H_{v,avg} \end{pmatrix}

Now inverting this 4x4 matrix gives the Impedance representation directly.

.. math::
   
   \begin{pmatrix} E_{u,avg} \\ E_{v,avg} \\ H_{u,avg} \\ H_{v,avg} \end{pmatrix} = 
   \begin{pmatrix} 
   Z_{ee}^{uu} & Z_{ee}^{uv} & Z_{em}^{uu} & Z_{em}^{uv} \\ 
   Z_{ee}^{vu} & Z_{ee}^{vv} & Z_{em}^{vu} & Z_{em}^{vv} \\
   Z_{me}^{uu} & Z_{me}^{uv} & Z_{mm}^{uu} & Z_{mm}^{uv} \\
   Z_{me}^{vu} & Z_{me}^{vv} & Z_{mm}^{vu} & Z_{mm}^{vv}
   \end{pmatrix} 
   \begin{pmatrix} J_u \\ J_v \\ M_u \\ M_v \end{pmatrix}


**Connection to Assembly** (:math:`\mathbf{A_M}`)

The MoM assembly process builds the global system matrix by iterating over quadrants (:math:`Z_{JJ}, Z_{JM}`, etc.).
Even though we invert the full 4x4 matrix at once to capture potential bianisotropic coupling, **we split it back up** immediately after. The helper code `gstc_utils.hpp` performs this sequence:

1.  **Invert 4x4:** Get the globally correct mixing coefficients (Impedance from Admittance).
2.  **Extract Blocks:** Pull out :math:`\bar{\bar{\mathbf{Z}}}_{ee}` (top-left 2x2), :math:`\bar{\bar{\mathbf{Z}}}_{em}` (top-right 2x2), etc.
3.  **Output:** The utility returns these 4 blocks separately (but packed in the same array).

So, when we write the CUDA kernel for :math:`\mathbf{A_M}`, we are compatible with the standard approach:

*   To fill the **Top-Left** of the global matrix (Electric Current equation), we read the **Top-Left** block (:math:`\mathbf{Z}_{ee}` or :math:`\mathbf{Y}_{ee}` depending on equation formulation) from our array.
*   To fill the **Top-Right**, we read the **Top-Right** block, and so on.

The 4x4 inversion is just a robust pre-calculation step to ensure the inputs to the assembly are physically correct.

Data Structure and Assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CPU prepares the local admittance and impedance matrices. The CUDA mass
assembly kernel computes the face basis and rotates their 2x2 blocks to 3D.

**Storage Format:**
For each face, the input stores two row-major 4x4 matrices:
:math:`\mathbb{Y}_{tot}` followed by :math:`\mathbb{Z}_{tot}`. Together they
contain the eight local 2x2 blocks needed by the current equations.

*   The packed input contains 32 complex values per face, or one shared set
    of 32 values when the tensors are uniform.
*   The layout is equivalent to:
    
    .. code-block:: cpp

       struct FaceTensors {
           complex__float Y[4][4]; // Local admittance, row-major
           complex__float Z[4][4]; // Local impedance, row-major
       };

**Assembly Logic:**
The assembly kernel for the Mass Matrix term :math:`\mathbf{A_M}` calculates the inner product:

.. math::

   A_{ij} = \int_{f_i} \mathbf{f}_i \cdot (\mathbf{T}_{3D} \cdot \mathbf{f}_j) \, dS

where :math:`\mathbf{T}_{3D}` is formed from the corresponding local 2x2 block
using :cpp:func:`sie_mom::get_local_basis_vectors` and
:cpp:func:`sie_mom::rotate_tensor_2x2_to_3x3`.
For the admittance terms, :cpp:func:`sie_mom::apply_perp_transform` then forms
the perpendicular tensors used in :eq:`eq:final_4x4_gstc`.

**Tensor Representation (2x2 vs 3x3)**

*   **Physical View (2x2):** Physically, the GSTC parameters are defined as 2x2 tensors in the local tangent plane of the surface. For a local basis :math:`(\hat{\mathbf{u}}, \hat{\mathbf{v}})`:
    
    .. math::

        \bar{\bar{\mathbf{Y}}}_{local} = \begin{pmatrix} Y_{uu} & Y_{uv} \\ Y_{vu} & Y_{vv} \end{pmatrix}

*   **Computational View (3x3):** The Method of Moments solver uses RWG basis functions defined in the global 3D Cartesian system :math:`(\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{z}})` rather than local surface coordinates. This avoids singular points (like the poles of a sphere) where a 2D local coordinate system might break down. To operate on these vectors, we must transform the local 2x2 tensor into a degenerate 3x3 Cartesian matrix.

    This is achieved by a basis rotation. Let :math:`\mathbf{R}` be the :math:`3 \times 2` projection matrix formed by the local tangent vectors:

    .. math::

       \mathbf{R} = [\hat{\mathbf{u}} \quad \hat{\mathbf{v}}] = 
       \begin{pmatrix} 
       u_x & v_x \\ 
       u_y & v_y \\ 
       u_z & v_z 
       \end{pmatrix}

    The corresponding 3x3 global tensor :math:`\mathbf{T}_{3D}` is obtained via:

    .. math::

       \mathbf{T}_{3D} = \mathbf{R} \cdot \mathbf{T}_{local} \cdot \mathbf{R}^T

    This 3x3 matrix correctly maps tangential 3D vectors to tangential 3D vectors in the global basis.


Explicit Mass Matrix Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now detail the explicit formulas used in the GPU kernels to assemble the matrix elements. The goal is to compute the term:

.. math::
   A_{ij} = \int_{f} \mathbf{f}_i \cdot (\mathbf{T}_{\text{eff}} \cdot \mathbf{f}_j) \, dS,

where :math:`\mathbf{T}_{\text{eff}}` is the effective 3D tensor acting on the RWG basis functions. 

Depending on whether we are handling an **Impedance (Z)** term or an **Admittance (Y)** term, the tensor :math:`\mathbf{T}_{\text{eff}}` is constructed differently.

1. Impedance Case (Z Terms)
"""""""""""""""""""""""""""

For terms involving :math:`\mathbf{Z}` (e.g., :math:`\mathbf{Z}_{ee}, \mathbf{Z}_{em}, \dots`), the GSTC operator acts directly on the current vectors. The transformation is simply the rotation from the local 2D basis to the global 3D basis.

Assuming we have the local 2x2 tensor components :math:`T_{00}, T_{01}, T_{10}, T_{11}` and the local basis vectors :math:`\hat{\mathbf{u}}, \hat{\mathbf{v}}` (vectors in 3D), the steps are:

**A. Basis Rotation (2D to 3D):**

We construct the global 3x3 tensor :math:`\mathbf{Z}_{3D}` via the projection:

.. math::
   \mathbf{Z}_{3D} = \mathbf{R} \mathbf{Z}_{\text{local}} \mathbf{R}^T

Explicitly, the element :math:`(\mathbf{Z}_{3D})_{mn}` (where :math:`m,n \in \{x,y,z\}`) is:

.. math::
   (\mathbf{Z}_{3D})_{mn} = \sum_{a,b \in \{u,v\}} R_{ma} (\mathbf{Z}_{\text{local}})_{ab} R_{nb} 

   = R_{m0} R_{n0} Z_{uu} + R_{m0} R_{n1} Z_{uv} + R_{m1} R_{n0} Z_{vu} + R_{m1} R_{n1} Z_{vv}

where :math:`R = [\hat{\mathbf{u}}, \hat{\mathbf{v}}]` has columns :math:`R_{\cdot 0} = \hat{\mathbf{u}}` and :math:`R_{\cdot 1} = \hat{\mathbf{v}}`.

**B. Integration:**

The effective tensor is simply :math:`\mathbf{T}_{\text{eff}} = \mathbf{Z}_{3D}`. The integral becomes:

.. math::
   A_{ij}^{(Z)} = \int_{f} \mathbf{f}_i \cdot (\mathbf{Z}_{3D} \cdot \mathbf{f}_j) \, dS


2. Admittance Case (Y Terms)
""""""""""""""""""""""""""""

For terms involving :math:`\mathbf{Y}` (e.g., :math:`\mathbf{Y}_{mm}` in equation :eq:`eq:sie1_testing` or :math:`\mathbf{Y}_{ee}` in eq :eq:`eq:sie2_testing`), the formulation typically involves the rotation operator :math:`\hat{\mathbf{n}} \times (\cdot)`.  Specifically, the term appears as:

.. math::
   \text{Term} \propto \hat{\mathbf{n}} \times (\mathbf{Y} \cdot (\hat{\mathbf{n}} \times \mathbf{J}))

Using the vector identity for the scalar triple product, this is equivalent to testing with a rotated basis function. To implement this in the standard form :math:`\mathbf{f}_i \cdot (\mathbf{T} \cdot \mathbf{f}_j)`, we define a "perpendicular" tensor :math:`\mathbf{Y}_{\perp}`.

**A. Basis Rotation (2D to 3D):**

First, we rotate the local 2x2 :math:`\mathbf{Y}_{\text{local}}` to the global 3x3 :math:`\mathbf{Y}_{3D}` exactly as in the Z case:

.. math::
   \mathbf{Y}_{3D} = \mathbf{R} \mathbf{Y}_{\text{local}} \mathbf{R}^T

**B. The "Perp" Transform:**

The operator :math:`\mathbf{v} \to \hat{\mathbf{n}} \times \mathbf{v}` can be represented by the skew-symmetric matrix :math:`[\hat{\mathbf{n}}]_\times`:

.. math::
   [\hat{\mathbf{n}}]_\times = \begin{pmatrix} 
   0 & -n_z & n_y \\ 
   n_z & 0 & -n_x \\ 
   -n_y & n_x & 0 
   \end{pmatrix}

**Geometric Interpretation:**

This matrix :math:`[\hat{\mathbf{n}}]_\times` represents a **rotation by 90 degrees** in the plane perpendicular to :math:`\hat{\mathbf{n}}`. 
For any vector :math:`\mathbf{v}`, the matrix-vector product :math:`[\hat{\mathbf{n}}]_\times \mathbf{v}` is mathematically equivalent to the cross product :math:`\hat{\mathbf{n}} \times \mathbf{v}`. 
Physically, this operation rotates a tangent vector around the normal axis, effectively swapping the :math:`\hat{\mathbf{u}}` and :math:`\hat{\mathbf{v}}` components (e.g., :math:`\mathbf{\hat{u}} \to \mathbf{\hat{v}}` and :math:`\mathbf{\hat{v}} \to -\mathbf{\hat{u}}`). 
The "Perp" transformation of the tensor accounts for this rotation appearing in both the test function and the field term.

The effective operator :math:`\mathbf{Y}^{\perp}` that satisfies :math:`\text{Term} = \mathbf{f}_i \cdot \mathbf{Y}^{\perp} \cdot \mathbf{f}_j` is:

.. math::
   \mathbf{Y}^{\perp} = - [\hat{\mathbf{n}}]_\times \, \mathbf{Y}_{3D} \, [\hat{\mathbf{n}}]_\times

*(Note: The minus sign comes from the identity* :math:`(\hat{\mathbf{n}} \times \mathbf{A}) \cdot \mathbf{B} = - \mathbf{A} \cdot (\hat{\mathbf{n}} \times \mathbf{B})` *when moving the cross product from the test function to the inner operator).*

**Implementation:**

In the code, this matrix operations is performed explicitly.
Let :math:`\mathbf{T}_{\text{temp}} = \mathbf{Y}_{3D} [\hat{\mathbf{n}}]_\times`.
Then :math:`\mathbf{Y}^{\perp} = - [\hat{\mathbf{n}}]_\times \mathbf{T}_{\text{temp}}`.

This results in the effective tensor :math:`\mathbf{T}_{\text{eff}} = \mathbf{Y}^{\perp}`, and the integral is computed via the same standard routine:

.. math::
   A_{ij}^{(Y)} = \int_{f} \mathbf{f}_i \cdot (\mathbf{Y}^{\perp} \cdot \mathbf{f}_j) \, dS

