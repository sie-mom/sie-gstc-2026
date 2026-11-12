Benchmark with Spherical Vector Wave Functions (SVWF)
===================================================================================

This section details the mathematical definitions and implementations of the Spherical Vector Wave Functions (SVWF) used in the spherical expansion module :cite:p:`Sandeep2019fastSWVF`. These functions form the basis for solving electromagnetic scattering problems involving spherical geometries.

Vector Spherical Harmonics (Jackson Convention)
-----------------------------------------------

Our implementation follows the definition of the normalized vector spherical harmonic :math:`\mathbf{X}_{nm}` as given by Jackson :cite:p:`jackson1999classical`. We define the orbital angular momentum operator :math:`\mathbf{L}` (dimensionless, as in Jackson Eq. 9.101) as:

.. math::

   \mathbf{L} = \frac{1}{i} (\mathbf{r} \times \nabla)

The vector spherical harmonic :math:`\mathbf{X}_{nm}` is then defined (Jackson Eq. 9.119) as:

.. math::

   \mathbf{X}_{nm}(\theta, \phi) = \frac{1}{\sqrt{n(n+1)}} \mathbf{L} Y_n^m(\theta, \phi)

where the indices must satisfy :math:`n \ge 1` and :math:`|m| \le n`. The case :math:`n=0` is excluded.

Explicitly, in terms of the scalar spherical harmonics :math:`Y_n^m`, this is:

.. math::

   \mathbf{X}_{nm}(\theta, \phi) = \frac{1}{\sqrt{n(n+1)}} \left( \frac{i}{\sin\theta} \frac{\partial Y_n^m}{\partial \phi} \hat{\boldsymbol{\theta}} - i \frac{\partial Y_n^m}{\partial \theta} \hat{\boldsymbol{\phi}} \right)

This function :math:`\mathbf{X}_{nm}` is purely tangential to the sphere surface and corresponds to the angular part of the TE (Transverse Electric) modes.

Full Spherical Vector Wave Functions (M, N, Jackson 9.122)
----------------------------------------------------------

We define the full spherical vector wave functions :math:`\mathbf{M}_{nm}` and :math:`\mathbf{N}_{nm}` consistent with the multipole fields defined in Jackson Eq. 9.122.

The **Magnetic Type** function :math:`\mathbf{M}_{nm}` (corresponding to the vector part of the Magnetic Multipole field :math:`\mathbf{H}` or Electric Multipole :math:`\mathbf{E}` by duality) is defined as:

.. math::

   \mathbf{M}_{nm}(\mathbf{r}) = z_n(kr) \mathbf{X}_{nm}(\theta, \phi)

The **Electric Type** function :math:`\mathbf{N}_{nm}` (corresponding to :math:`\frac{1}{k} \nabla \times \mathbf{M}_{nm}`) is defined as:

.. math::

   \mathbf{N}_{nm}(\mathbf{r}) = \frac{1}{k} \nabla \times \mathbf{M}_{nm}(\mathbf{r})

Note: In Jackson's notation (Eq. 9.122), the magnetic multipole fields are :math:`\mathbf{H} = f_l(r) \mathbf{X}_{lm}` and :math:`\mathbf{E} = \frac{i}{k} \nabla \times \mathbf{H}`. Thus, our :math:`\mathbf{M}` corresponds to Jackson's magnetic multipole radial function profile, and our :math:`\mathbf{N}` corresponds to the associated curl field (divided by :math:`i`).

Analytic Expansions
^^^^^^^^^^^^^^^^^^^

Using the properties of :math:`\mathbf{X}_{nm}`, the components of :math:`\mathbf{N}_{nm}` are derived as follows. Let :math:`\psi_n(\rho) = \rho z_n(\rho)` be the Riccati-Bessel function and :math:`\psi_n'(\rho)` its derivative with respect to :math:`\rho = kr`.

.. math::

   \mathbf{N}_{nm} = i \frac{\sqrt{n(n+1)}}{\rho} z_n(\rho) Y_n^m \hat{\mathbf{r}} + \frac{\psi_n'(\rho)}{\rho} \hat{\mathbf{r}} \times \mathbf{X}_{nm}

Explicitly in components:

.. math::

   N_{nm, r} = i \frac{\sqrt{n(n+1)}}{\rho} z_n(\rho) Y_n^m(\theta, \phi)

.. math::

   N_{nm, \theta} = -\frac{\psi_n'(\rho)}{\rho} X_{nm, \phi}

.. math::

   N_{nm, \phi} = \frac{\psi_n'(\rho)}{\rho} X_{nm, \theta}

These analytical expressions are implemented in ``svwf_sphere.py`` to ensure high precision.


Tangential Spherical Vector Wave Functions
------------------------------------------

To solve for the scattering coefficients using GSTC, we only require the tangential components of the fields on the sphere's surface.

For a given mode :math:`(n, m)` and radial distance :math:`r` (where :math:`\rho = kr`), the tangential components are:

.. math::

   \mathbf{M}_{nm}^{tan} = z_n(\rho) \left( \hat{\mathbf{r}} \times \mathbf{X}_{nm} \right)

.. math::

   \mathbf{N}_{nm}^{tan} = \frac{\psi_n'(\rho)}{\rho} \mathbf{X}_{nm}

where :math:`\psi_n'(\rho) = z_n(\rho) + \rho z_n'(\rho)` is the derivative of the Riccati-Bessel function.

Field Expansions (Tangential)
-----------------------------

The fields are expanded in terms of spherical vector wave functions, truncated at a maximum degree :math:`N_{max}`. The summation indices generally run as follows:

.. math::

   \sum_{n,m} \equiv \sum_{n=1}^{N_{max}} \sum_{m=-n}^{n}

Note that :math:`n` starts from 1 because the vector spherical harmonics :math:`\mathbf{X}_{00}` correspond to the gradient of a scalar constant, which is zero (no monopole radiation for transverse fields).
 
We define the vectors of unknown coefficients:
:math:`\mathbf{x}_{scat} = [a_{nm}, b_{nm}]^T` for the scattered fields and :math:`\mathbf{x}_{int} = [c_{nm}, d_{nm}]^T` for the interior fields. The coefficients :math:`a_{nm}` and :math:`c_{nm}` correspond to the TE modes (:math:`\mathbf{M}_{nm}`), while :math:`b_{nm}` and :math:`d_{nm}` correspond to the TM modes (:math:`\mathbf{N}_{nm}`).

Scattered Fields
^^^^^^^^^^^^^^^^
 
The tangential scattered field (in the exterior medium 2, with :math:`k_0, \eta_0`) is expanded using outgoing SVWFs (Hankel functions, kind=3):
 
.. math::
 
   \mathbf{E}_{scat}^{tan} = \sum_{n,m} \left( a_{nm} \mathbf{M}_{nm}^{(3), tan} + b_{nm} \mathbf{N}_{nm}^{(3), tan} \right)
 
.. math::
 
   \mathbf{H}_{scat}^{tan} = \frac{1}{i\eta_0} \sum_{n,m} \left( a_{nm} \mathbf{N}_{nm}^{(3), tan} + b_{nm} \mathbf{M}_{nm}^{(3), tan} \right)
 
Interior Fields
^^^^^^^^^^^^^^^
 
Similarly, the tangential interior field (in medium 1, with :math:`k_1, \eta_1`) is expanded using regular SVWFs (Bessel functions, kind=1):
 
.. math::
 
   \mathbf{E}_{int}^{tan} = \sum_{n,m} \left( c_{nm} \mathbf{M}_{nm}^{(1), tan} + d_{nm} \mathbf{N}_{nm}^{(1), tan} \right)
 
.. math::
 
   \mathbf{H}_{int}^{tan} = \frac{1}{i\eta_1} \sum_{n,m} \left( c_{nm} \mathbf{N}_{nm}^{(1), tan} + d_{nm} \mathbf{M}_{nm}^{(1), tan} \right)
 
Total Exterior Fields
^^^^^^^^^^^^^^^^^^^^^

The total field in region 2 is the sum of the incident and scattered fields:

.. math::

   \mathbf{E}_2^{tan} = \mathbf{E}_{inc}^{tan} + \mathbf{E}_{scat}^{tan}

   \mathbf{H}_2^{tan} = \mathbf{H}_{inc}^{tan} + \mathbf{H}_{scat}^{tan}

Note: :math:`\mathbf{E}_{inc}^{tan}` and :math:`\mathbf{H}_{inc}^{tan}` are evaluated directly at the collocation points.

Matrix Operators
----------------

We construct matrix operators that map the coefficient vectors to the field values at the collocation points.
Let :math:`N_p` be the number of collocation points and :math:`N_c` be the number of coefficients.
The operators have dimensions :math:`2N_p \times N_c` (factor of 2 for :math:`\theta` and :math:`\phi` components).

We define the following operators:

*   :math:`\bar{\bar{\mathbf{E}}}_{int}`: Maps :math:`\mathbf{x}_{int}` to :math:`\mathbf{E}_1^{tan}`.
*   :math:`\bar{\bar{\mathbf{H}}}_{int}`: Maps :math:`\mathbf{x}_{int}` to :math:`\mathbf{H}_1^{tan}`.
*   :math:`\bar{\bar{\mathbf{E}}}_{scat}`: Maps :math:`\mathbf{x}_{scat}` to :math:`\mathbf{E}_{scat}^{tan}`.
*   :math:`\bar{\bar{\mathbf{H}}}_{scat}`: Maps :math:`\mathbf{x}_{scat}` to :math:`\mathbf{H}_{scat}^{tan}`.

Note that :math:`\bar{\bar{\mathbf{E}}}_{int}` and :math:`\bar{\bar{\mathbf{H}}}_{int}` are zero for the columns corresponding to :math:`\mathbf{x}_{scat}`, and vice-versa.

GSTC System Formation
---------------------

The Generalized Sheet Transition Conditions (GSTC) relate the fields across the boundary surface:

.. math::

   \hat{\mathbf{n}} \times (\mathbf{H}_2 - \mathbf{H}_1) = \mathbf{J}_S

   \hat{\mathbf{n}} \times (\mathbf{E}_2 - \mathbf{E}_1) = -\mathbf{M}_S

The surface currents are related to the average fields :math:`\mathbf{E}_{avg} = (\mathbf{E}_1 + \mathbf{E}_2)/2` and :math:`\mathbf{H}_{avg} = (\mathbf{H}_1 + \mathbf{H}_2)/2` by the susceptibility tensors :math:`\bar{\bar{Y}}, \bar{\bar{\chi}}, \bar{\bar{\gamma}}, \bar{\bar{Z}}`:

.. math::

   \mathbf{J}_S = \bar{\bar{Y}} \mathbf{E}_{avg,tan} + \bar{\bar{\chi}} \mathbf{H}_{avg,tan}

   \mathbf{M}_S = \bar{\bar{\gamma}} \mathbf{E}_{avg,tan} + \bar{\bar{Z}} \mathbf{H}_{avg,tan}

Substitution and Rearrangement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We substitute the field decompositions :math:`\mathbf{E}_2^{tan} = \mathbf{E}_{inc}^{tan} + \mathbf{E}_{scat}^{tan}` and :math:`\mathbf{H}_2^{tan} = \mathbf{H}_{inc}^{tan} + \mathbf{H}_{scat}^{tan}` (and similarly for region 1 with :math:`\mathbf{E}_{int}^{tan}, \mathbf{H}_{int}^{tan}`) into the boundary conditions.

Let :math:`\mathcal{R}` denote the rotation operator :math:`\hat{\mathbf{n}} \times`. The equations become:

Equation 1 (Electric Current):

.. math::

   \left( \mathcal{R} - \frac{\bar{\bar{\chi}}}{2} \right) \mathbf{H}_{scat}^{tan} - \frac{\bar{\bar{Y}}}{2} \mathbf{E}_{scat}^{tan} - \left( \mathcal{R} + \frac{\bar{\bar{\chi}}}{2} \right) \mathbf{H}_{int}^{tan} - \frac{\bar{\bar{Y}}}{2} \mathbf{E}_{int}^{tan} = -\left( \mathcal{R} - \frac{\bar{\bar{\chi}}}{2} \right) \mathbf{H}_{inc}^{tan} + \frac{\bar{\bar{Y}}}{2} \mathbf{E}_{inc}^{tan}

Equation 2 (Magnetic Current):

.. math::

   \left( \mathcal{R} + \frac{\bar{\bar{\gamma}}}{2} \right) \mathbf{E}_{scat}^{tan} + \frac{\bar{\bar{Z}}}{2} \mathbf{H}_{scat}^{tan} - \left( \mathcal{R} - \frac{\bar{\bar{\gamma}}}{2} \right) \mathbf{E}_{int}^{tan} + \frac{\bar{\bar{Z}}}{2} \mathbf{H}_{int}^{tan} = -\left( \mathcal{R} + \frac{\bar{\bar{\gamma}}}{2} \right) \mathbf{E}_{inc}^{tan} - \frac{\bar{\bar{Z}}}{2} \mathbf{H}_{inc}^{tan}

Discretization into Matrix Form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The linear system :math:`\mathbf{Z}\mathbf{x} = \mathbf{V}` is constructed by enforcing the GSTC equations at :math:`N_p` collocation points. The unknown vector :math:`\mathbf{x}` collects the coefficients for all :math:`N_{modes}` harmonics:

.. math::

   \mathbf{x} = [ \dots, a_{nm}, b_{nm}, c_{nm}, d_{nm}, \dots ]^T

To clearly map these coefficients to the matrix columns, we define the specific field basis pairs for each coefficient type.
Let :math:`\mathbf{M}^{(3)}, \mathbf{N}^{(3)}` denote the exterior (Hankel) SVWFs, and :math:`\mathbf{M}^{(1)}, \mathbf{N}^{(1)}` denote the interior (Bessel) SVWFs.

*   **Coeff** :math:`a_{nm}` (Ext. TE) produces: :math:`(\mathbf{E}_a, \mathbf{H}_a) = (\mathbf{M}_{nm}^{(3)}, \frac{1}{i\eta_0}\mathbf{N}_{nm}^{(3)})`
*   **Coeff** :math:`b_{nm}` (Ext. TM) produces: :math:`(\mathbf{E}_b, \mathbf{H}_b) = (\mathbf{N}_{nm}^{(3)}, \frac{1}{i\eta_0}\mathbf{M}_{nm}^{(3)})`
*   **Coeff** :math:`c_{nm}` (Int. TE) produces: :math:`(\mathbf{E}_c, \mathbf{H}_c) = (\mathbf{M}_{nm}^{(1)}, \frac{1}{i\eta_1}\mathbf{N}_{nm}^{(1)})`
*   **Coeff** :math:`d_{nm}` (Int. TM) produces: :math:`(\mathbf{E}_d, \mathbf{H}_d) = (\mathbf{N}_{nm}^{(1)}, \frac{1}{i\eta_1}\mathbf{M}_{nm}^{(1)})`

**GSTC Matrix Operators**

We define four operators representing the contribution of a field tuple :math:`(\mathbf{E}, \mathbf{H})` to the boundary conditions.
Distinguishing between exterior and interior terms is crucial because the signs of the susceptibility terms change relative to the Rotation operator.

For the **Electric Current Equation** (J-condition):

.. math::

   \mathcal{J}_{ext}(\mathbf{E}, \mathbf{H}) = \left( \mathcal{R} - \frac{\bar{\bar{\chi}}}{2} \right) \mathbf{H} - \frac{\bar{\bar{Y}}}{2} \mathbf{E}

.. math::

   \mathcal{J}_{int}(\mathbf{E}, \mathbf{H}) = -\left( \mathcal{R} + \frac{\bar{\bar{\chi}}}{2} \right) \mathbf{H} - \frac{\bar{\bar{Y}}}{2} \mathbf{E}

For the **Magnetic Current Equation** (M-condition):

.. math::

   \mathcal{M}_{ext}(\mathbf{E}, \mathbf{H}) = \left( \mathcal{R} + \frac{\bar{\bar{\gamma}}}{2} \right) \mathbf{E} + \frac{\bar{\bar{Z}}}{2} \mathbf{H}

.. math::

   \mathcal{M}_{int}(\mathbf{E}, \mathbf{H}) = -\left( \mathcal{R} - \frac{\bar{\bar{\gamma}}}{2} \right) \mathbf{E} + \frac{\bar{\bar{Z}}}{2} \mathbf{H}

**Matrix Block Structure**

For a single mode :math:`(n,m)` at a single point, the corresponding block of the matrix :math:`\mathbf{Z}` (4 columns) is constructed by applying these operators to the basis fields defined above:

.. math::   
   :label: eq-z-block
      
   \mathbf{Z}_{nm}^{block} =
   \begin{bmatrix}
   \mathcal{J}_{ext}(\mathbf{E}_a, \mathbf{H}_a) & \mathcal{J}_{ext}(\mathbf{E}_b, \mathbf{H}_b) & \mathcal{J}_{int}(\mathbf{E}_c, \mathbf{H}_c) & \mathcal{J}_{int}(\mathbf{E}_d, \mathbf{H}_d) \\
   \mathcal{M}_{ext}(\mathbf{E}_a, \mathbf{H}_a) & \mathcal{M}_{ext}(\mathbf{E}_b, \mathbf{H}_b) & \mathcal{M}_{int}(\mathbf{E}_c, \mathbf{H}_c) & \mathcal{M}_{int}(\mathbf{E}_d, \mathbf{H}_d)
   \end{bmatrix}

This block represents 4 rows (2 for J-vector, 2 for M-vector) and 4 columns (:math:`a, b, c, d`).

**Note:** Visually, the equation shows 2 rows of vector-valued functions. In the linear system, these expand into 4 scalar rows (:math:`J_\theta, J_\phi, M_\theta, M_\phi`).

The explicit signs in the interior operators ensure the boundary conditions are correctly enforced.

Substituting the definition of the partial fields :math:`(\mathbf{E}, \mathbf{H})` for each coefficient gives the explicit form in terms of SVWFs:

.. math::
   :label: eq-z-block-MN

   \mathbf{Z}_{nm}^{block} =
   \begin{bmatrix}
   \mathcal{J}_{ext}\left(\mathbf{M}^{(3)}, \frac{\mathbf{N}^{(3)}}{i\eta_0}\right) & \mathcal{J}_{ext}\left(\mathbf{N}^{(3)}, \frac{\mathbf{M}^{(3)}}{i\eta_0}\right) & \mathcal{J}_{int}\left(\mathbf{M}^{(1)}, \frac{\mathbf{N}^{(1)}}{i\eta_1}\right) & \mathcal{J}_{int}\left(\mathbf{N}^{(1)}, \frac{\mathbf{M}^{(1)}}{i\eta_1}\right) \\
   \mathcal{M}_{ext}\left(\mathbf{M}^{(3)}, \frac{\mathbf{N}^{(3)}}{i\eta_0}\right) & \mathcal{M}_{ext}\left(\mathbf{N}^{(3)}, \frac{\mathbf{M}^{(3)}}{i\eta_0}\right) & \mathcal{M}_{int}\left(\mathbf{M}^{(1)}, \frac{\mathbf{N}^{(1)}}{i\eta_1}\right) & \mathcal{M}_{int}\left(\mathbf{N}^{(1)}, \frac{\mathbf{M}^{(1)}}{i\eta_1}\right)
   \end{bmatrix}

Linear System Assembly
^^^^^^^^^^^^^^^^^^^^^^

The global linear system :math:`\mathbf{Z} \mathbf{x} = \mathbf{V}` is assembled by collecting the equations for all :math:`N_p` collocation points. The structure directly mirrors the code implementation loops.

**System Dimensions:**

*   **Total Rows (Equations):** At each collocation point, we enforce 2 vector boundary conditions (:math:`\mathbf{J}_{surf}` and :math:`\mathbf{M}_{surf}`). Each vector condition has 2 tangential components (:math:`\theta` and :math:`\phi`).
    This yields **4 scalar equations per point**.
    
    .. math:: \text{Total Equations} = 4 \times N_p

*   **Total Columns (Unknowns):** For each harmonic mode :math:`(n, m)`, there are 4 unknown coefficients (:math:`a, b, c, d`).
    
    .. math:: \text{Total Unknowns} = 4 \times N_{modes}

**Global System Structure:**

The global vector :math:`\mathbf{V}` and matrix :math:`\mathbf{Z}` are constructed by interleaving the 4 equations for each point.
For the :math:`k`-th collocation point :math:`\mathbf{r}_k`, the corresponding 4 rows in :math:`\mathbf{V}` are:

.. math::

   \mathbf{V}_k = \begin{bmatrix} 
   [-\mathcal{J}_{ext}(\mathbf{E}_{inc}, \mathbf{H}_{inc})]_k \cdot \hat{\boldsymbol{\theta}} \\
   [-\mathcal{J}_{ext}(\mathbf{E}_{inc}, \mathbf{H}_{inc})]_k \cdot \hat{\boldsymbol{\phi}} \\
   [-\mathcal{M}_{ext}(\mathbf{E}_{inc}, \mathbf{H}_{inc})]_k \cdot \hat{\boldsymbol{\theta}} \\
   [-\mathcal{M}_{ext}(\mathbf{E}_{inc}, \mathbf{H}_{inc})]_k \cdot \hat{\boldsymbol{\phi}}
   \end{bmatrix}

Similarly, the global matrix :math:`\mathbf{Z}` is built by stacking the local :math:`4 \times 4` blocks. If we view :math:`\mathbf{Z}` as a block matrix where row block :math:`k` corresponds to point :math:`\mathbf{r}_k` and column block :math:`j` corresponds to mode :math:`pq` (indices :math:`n,m`):

.. math::

   \mathbf{Z} = \begin{bmatrix} 
   \mathbf{Z}_{mode 1}^{block}(\mathbf{r}_1) & \mathbf{Z}_{mode 2}^{block}(\mathbf{r}_1) & \dots \\
   \mathbf{Z}_{mode 1}^{block}(\mathbf{r}_2) & \mathbf{Z}_{mode 2}^{block}(\mathbf{r}_2) & \dots \\
   \vdots & \vdots & \ddots
   \end{bmatrix}

Each entry :math:`\mathbf{Z}_{mode}^{block}(\mathbf{r})` corresponds to the operator block in Eq. :eq:`eq-z-block-MN`.

**Note:** While Eq. :eq:`eq-z-block-MN` is written with 2 vector rows (for :math:`\mathcal{J}` and :math:`\mathcal{M}`), each operator yields 2 scalar components (:math:`\theta` and :math:`\phi`), resulting in the :math:`4 \times 4` scalar matrix used in the implementation.

Implementation Details
----------------------

The process described above is implemented in the `solve_collocation` function within the `solver.py` module.

**System Construction and Solving:**

1.  **Collocation Grid**: A grid of collocation points (:math:`\theta_i, \phi_j`) is created on the sphere's surface. The number of points is chosen to be greater than the number of unknown coefficients, leading to an overdetermined system.

2.  **Basis Functions**: The angular parts of the SVWFs are computed using `scipy.special.sph_harm_y` as the foundation. The radial dependence is handled by `scipy.special.spherical_jn` and `scipy.special.spherical_yn` for the Bessel and Neumann functions, respectively. These are combined in `svwf_sphere.py` to generate the basis functions :math:`\mathbf{M}_{nm}` and :math:`\mathbf{N}_{nm}` at each collocation point.

3.  **Linear System**: The matrix :math:`\mathbf{Z}` and vector :math:`\mathbf{V}` are assembled by enforcing the two rearranged GSTC vector equations at each point on the grid. This results in a large, dense, and overdetermined linear system.

4.  **Solving**: The system :math:`\mathbf{Z} \mathbf{x} = \mathbf{V}` is solved using a weighted least-squares approach to minimize the error over the sphere's surface. The core solver is `scipy.linalg.lstsq`, which is configured to use the `gelsy` LAPACK driver. This driver is robust and well-suited for potentially rank-deficient systems.

**Verification and Testing:**

The correctness of the solver is verified by comparing its output against known analytical solutions (Mie theory). This is done by configuring the GSTC parameters to model simpler physical scenarios:

*   **Perfect Electric Conductor (PEC):** By setting the surface impedance to zero, the solver's results for scattering cross-section can be matched against the classical Mie solution for a PEC sphere.
*   **Dielectric Sphere:** The solver can also model a non-coated dielectric sphere, and its results can be validated against the corresponding Mie solution.

**GSTC Tensor Support:**

The current implementation is designed with flexibility in mind but has a specific scope:

*   **Current Support**: The code presently handles GSTC parameters that are **uniform** and **bi-isotropic**. This means the `gstc_params` dictionary accepts scalar values for the susceptibility and impedance terms (:math:`Y, \chi, \gamma, Z`). Internally, these scalars are converted to 2x2 diagonal matrices.

*   **Future Work**: The framework is structured to be extended. Future versions will support fully anisotropic (tensor) and spatially varying (arbitrary distribution) GSTC parameters. This will allow for the modeling of more complex and realistic metasurfaces.

Anisotropic cases
-----------------
