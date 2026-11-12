.. _singular:

#####################################
Singular term handling
#####################################

The implementation links below refer to the quadrature routines in
``include/sie_mom/cuda/dunavant_quadrature.cuh`` and the analytical
potentials in ``include/sie_mom/cuda/semi_analytic_integrals.cuh``.

.. _sie_L_singularity_extraction:

Singularity extraction for :math:`\mathcal{TL}` (surface–surface part)
===========================================================================


The goal is to evaluate the tested entry of the :math:`\mathcal{L}` operator
robustly for self and near–singular triangle pairs by splitting the Green
kernel into a regular part (handled with standard Dunavant quadrature) and a
Coulomb part :math:`1/(4\pi R)` (handled by analytic inner integrals and a
single outer quadrature loop).

Step 1 — Starting equation
--------------------------

We begin from the surface–surface contribution (line terms dropped), see :eq:`eq:TL_mjni_surface_RWG`.

Step 2 — Kernel split (regular + Coulomb)
-----------------------------------------

Split
:math:`G = G_{\mathrm{reg}} + \dfrac{1}{4\pi R}`, where
:math:`G_{\mathrm{reg}}:= G - \dfrac{1}{4\pi R}`, :math:`R=\|\mathbf r-\mathbf r'\|`.

.. admonition:: Numerical Evaluation of :math:`G_{\mathrm{reg}}`
   :class: tip

   The definition :math:`G_{\mathrm{reg}} = \frac{e^{-jkR} - 1}{4\pi R}` is numerically unstable for small :math:`R` due to catastrophic cancellation in the numerator. For a robust implementation, the limiting value as :math:`R \to 0` must be used. Using L'Hôpital's rule:

   .. math::

      \lim_{R \to 0} G_{\mathrm{reg}}(R) = \lim_{R \to 0} \frac{\frac{d}{dR}(e^{-jkR} - 1)}{\frac{d}{dR}(4\pi R)} = \lim_{R \to 0} \frac{-jk e^{-jkR}}{4\pi} = -\frac{jk}{4\pi}.

   Therefore, :math:`G_{\mathrm{reg}}` should be implemented as a piecewise function:

   .. math::
      :label: eq:Green_reg_definition

      G_{\mathrm{reg}}(R) =
      \begin{cases}
         \dfrac{e^{-jkR} - 1}{4\pi R}, & R > \epsilon \\
         -\dfrac{jk}{4\pi}, & R \le \epsilon
      \end{cases}

   where :math:`\epsilon` is a small machine-precision tolerance.

Define the modifier

.. math::

   M^{(mj,ni)}(\mathbf r,\mathbf r')
   \;=\; \frac{1}{4}\,\boldsymbol\rho_{mj}(\mathbf r)\!\cdot\!\boldsymbol\rho_{ni}(\mathbf r')
   \;-\;\frac{1}{k^2}.

Then

.. math::
   :label: eq:TL_kernel_split


   \mathcal{TL}_{mj,ni}^{(\mathrm{surf})}
   \;=\;
   \underbrace{C_{mj,ni}\!\iint M^{(mj,ni)}\,G_{\mathrm{reg}}\,\mathrm dS\,\mathrm dS'}_{
     \displaystyle \mathcal{TL}^{(\mathrm{reg})}_{mj,ni}}
   \;+\;
   \underbrace{C_{mj,ni}\!\iint M^{(mj,ni)}\,\frac{1}{4\pi R}\,\mathrm dS\,\mathrm dS'}_{
     \displaystyle \mathcal{TL}^{(\mathrm{coul})}_{mj,ni}}.

**Regular part** — standard Dunavant form :cite:p:`dunavant1985quadrature`:

.. math::
   :label: eq:TL_reg_dunavant

   \mathcal{TL}^{(\mathrm{reg})}_{mj,ni}
   \;\approx\;
   e_{mj}\,e_{ni}\!
   \sum_{p=1}^{Q_m}\sum_{q=1}^{Q_n}
   w_p\,w'_q\,
   \Big[
   \tfrac{1}{4}\,\boldsymbol\rho_{mj}(\mathbf r_{m,p})\!\cdot\!\boldsymbol\rho_{ni}(\mathbf r'_{n,q})
   -\tfrac{1}{k^2}
   \Big]\,
   G_{\mathrm{reg}}\!\big(\mathbf r_{m,p},\mathbf r'_{n,q}\big).

**Implementation:** |tl-tk-regular-quadrature| evaluates the regularized
TL and TK kernels with ``is_singular=true``.

Step 3 — Coulomb part as “outer quadrature × inner analytic”
------------------------------------------------------------

**(A) Inner panel potentials on the source face :math:`f_n`**

For each testing point :math:`\mathbf r\in f_m`, define

.. math::

   \phi_0(\mathbf r) \;:=\; \int_{f_n}\! \frac{1}{\|\mathbf r-\mathbf r'\|}\,\mathrm dS',
   \qquad
   \boldsymbol{\phi}_1(\mathbf r) \;:=\; \int_{f_n}\! \frac{\mathbf r' - \mathbf r}{\|\mathbf r-\mathbf r'\|}\,\mathrm dS'.

Then the vector inner integral entering the modifier is

.. math::
   :label: eq:Psi1_via_phi


   \boldsymbol{\Psi}_1(\mathbf r)
   \;=\; \int_{f_n}\!\frac{\boldsymbol\rho_{ni}(\mathbf r')}{\|\mathbf r-\mathbf r'\|}\,\mathrm dS'
   \;=\; \boldsymbol{\phi}_1(\mathbf r) + \big(\mathbf r - \mathbf r_{\overline{\partial e_{ni}}}\big)\,\phi_0(\mathbf r).

**Implementation:** |psi1-potential| evaluates this composite potential
from precomputed Wilton potentials.

**(B) One-loop outer quadrature form**

Using the same triangle rule on :math:`f_m`,
:math:`\int_{f_m}\phi\,\mathrm dS \approx A_m \sum_{p=1}^{Q_m} w_p\,\phi(\mathbf r_{m,p})`,
the Coulomb part becomes

.. math::
   :label: eq:TL_coulomb_outer_sum

   \mathcal{TL}^{(\mathrm{coul})}_{mj,ni}
   \;\approx\;
   \frac{e_{mj}e_{ni}}{A_n}\,\frac{1}{4\pi}\,
   \sum_{p=1}^{Q_m} w_p\,
   \Big[
   \tfrac{1}{4}\,\boldsymbol\rho_{mj}(\mathbf r_{m,p})\cdot
   \big(\underbrace{\boldsymbol{\phi}_1(\mathbf r_{m,p}) + (\mathbf r_{m,p}-\mathbf r_{\overline{\partial e_{ni}}})\,\phi_0(\mathbf r_{m,p})}_{\displaystyle \boldsymbol{\Psi}_1(\mathbf r_{m,p})}\big)
   - \tfrac{1}{k^2}\,\phi_0(\mathbf r_{m,p})
   \Big].

**Implementation:** the TL outputs of |tl-tk-singular-quadrature|
evaluate this outer quadrature using the analytical source potentials.

Closed-form :math:`\phi_0` (Wilton '84, eq. (5))
---------------------------------------------------

For a planar polygon (here a triangle) with edges indexed by :math:`i`, the scalar
panel potential is :cite:p:`wilton1984potential`

.. math::
   :label: eq:phi0_wilton

   \phi_0(\mathbf r)
   \;=\; \sum_{i}
   \big(\,\hat{\mathbf p}_i^{\,0}\!\cdot\!\hat{\mathbf u}_i\,\big)\!
   \left[
   p_i^{0}\,\ln\!\frac{R_i^{+}+l_i^{+}}{R_i^{-}+l_i^{-}}
   \;-\;
   |d|\!\left(
   \tan^{-1}\!\frac{p_i^{0}l_i^{+}}{(R_i^{0})^{2}+|d|\,R_i^{+}}
   \;-\;
   \tan^{-1}\!\frac{p_i^{0}l_i^{-}}{(R_i^{0})^{2}+|d|\,R_i^{-}}
   \right)
   \right].

**Implementation:** :cpp:func:`::calculate_phi_0` evaluates this scalar
panel potential.

**Quantities in Wilton's notation (code-ready):**

- Triangle vertices :math:`\mathbf v_0,\mathbf v_1,\mathbf v_2`.
- Unit normal :math:`\hat{\mathbf n}=\dfrac{(\mathbf v_1-\mathbf v_0)\times(\mathbf v_2-\mathbf v_0)}{\|(\mathbf v_1-\mathbf v_0)\times(\mathbf v_2-\mathbf v_0)\|}`.
- Signed distance and projection: :math:`d=\hat{\mathbf n}\!\cdot(\mathbf r-\mathbf v_0)`, :math:`\mathbf r_\pi=\mathbf r-d\,\hat{\mathbf n}`.
- For each edge :math:`i` with endpoints :math:`\mathbf r_i^{+},\mathbf r_i^{-}`:
- Unit tangent and outward co-normal: :math:`\hat{\mathbf i}_i=\dfrac{\mathbf r_i^{+}-\mathbf r_i^{-}}{\|\mathbf r_i^{+}-\mathbf r_i^{-}\|}`, :math:`\hat{\mathbf u}_i=\hat{\mathbf i}_i\times \hat{\mathbf n}`.
- In-plane projections :math:`\boldsymbol{\rho}=\mathbf r_\pi`, :math:`\boldsymbol{\rho}_i^{\pm}=\mathbf r_i^{\pm}-(\hat{\mathbf n}\!\cdot\!\mathbf r_i^{\pm})\hat{\mathbf n}`.
- Along-edge signed distances: :math:`l_i^{\pm}=(\boldsymbol{\rho}_i^{\pm}-\boldsymbol{\rho})\cdot \hat{\mathbf i}_i`.
- Perpendicular in-plane distance to the **edge line**: :math:`p_i^{0}=\big|(\boldsymbol{\rho}_i^{-}-\boldsymbol{\rho})\cdot \hat{\mathbf u}_i\big|`.
- Orientation factor (no vector build needed): :math:`\hat{\mathbf p}_i^{\,0}\!\cdot\!\hat{\mathbf u}_i=\operatorname{sign}\!\big((\boldsymbol{\rho}_i^{-}-\boldsymbol{\rho})\cdot \hat{\mathbf u}_i\big)\in\{+1,-1\}`.
- 3D distances: :math:`R_i^{0}=\sqrt{(p_i^{0})^{2}+d^{2}}`, :math:`R_i^{\pm}=\sqrt{(p_i^{0})^{2}+(l_i^{\pm})^{2}+d^{2}}`.

**Implementation:** :cpp:func:`::calculateWiltonGeometricTerms` computes
these geometric quantities. The batched evaluator
:cpp:func:`::calculateAllWiltonPotentials` reuses them to compute
:math:`\phi_0`, :math:`\phi_2`, and the free-vertex-independent parts of
:math:`\boldsymbol{\phi}_1` and :math:`\boldsymbol{\phi}_3` in one pass.

Closed-form :math:`\boldsymbol{\phi}_1` (Wilton '84, eq. (6))
--------------------------------------------------------------------

Using the same notation as above, the vector panel potential is :cite:p:`wilton1984potential`

.. math::
   :label: eq:phi1_wilton

   \boldsymbol{\phi}_1(\mathbf r)
   \;=\;
   \frac{1}{2}\,\sum_i \hat{\mathbf u}_i\,\Bigg[
      (R_i^{0})^{2}\,\ln\!\frac{R_i^{+}+l_i^{+}}{R_i^{-}+l_i^{-}}
      \;+\;
      l_i^{+}\, R_i^{+}
      \;-\;
      l_i^{-}\, R_i^{-}
   \Bigg] - d \hat{\mathbf n}\phi_0(\mathbf r).

**Implementation:** |phi1-potential| evaluates this vector panel potential.

.. note::
   The summation part of the formula calculates the integral for :math:`(\mathbf r' - \mathbf r_\pi) / R`, where :math:`\mathbf r_\pi` is the projection of the observation point onto the triangle's plane. The term :math:`- d \hat{\mathbf n}\phi_0(\mathbf r)` is a correction that transforms this result into the desired integral for :math:`(\mathbf r' - \mathbf r) / R`, since the vector difference can be decomposed as:

   .. math::

      \mathbf r' - \mathbf r = (\mathbf r' - \mathbf r_\pi) - (\mathbf r - \mathbf r_\pi) = (\mathbf r' - \mathbf r_\pi) - d\hat{\mathbf n}.

   Integrating this over the surface and dividing by :math:`R` yields the final formula.


Numerically robust primitives (for :math:`\phi_0`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use `hypot`-style evaluations:

  :math:`R_i^{0}=\mathrm{hypot}(p_i^{0},|d|)`,  
  :math:`R_i^{\pm}=\mathrm{hypot}(R_i^{0},\,l_i^{\pm})`.

- Evaluate the arctangent terms with :math:`\operatorname{atan2}`:

  .. math::

     \tan^{-1}\!\frac{p_i^{0}l_i^{\pm}}{(R_i^{0})^{2}+|d|\,R_i^{\pm}}
     \;\equiv\;
     \operatorname{atan2}\!\Big(p_i^{0}l_i^{\pm},\ (R_i^{0})^{2}+|d|\,R_i^{\pm}\Big).

- Evaluate the logarithm with `log1p` when :math:`|l_i^{\pm}|\ll R_i^{\pm}`:

  .. math::

     \ln\!\frac{R_i^{+}+l_i^{+}}{R_i^{-}+l_i^{-}}
     \;=\;
     \log1p\!\Big(\frac{l_i^{+}}{R_i^{+}}\Big)
     \;-\;
     \log1p\!\Big(\frac{l_i^{-}}{R_i^{-}}\Big).

- For nearly coplanar points (:math:`|d|\to 0`), the Wilton expression remains
  well-behaved with the :math:`\operatorname{atan2}` form; use a mesh-scaled tolerance
  to classify the “in-plane” case and keep orientation consistent for
  :math:`\hat{\mathbf u}_i` across the mesh.

.. _sie_T_singularity_extraction:

Singularity extraction for :math:`\mathcal{TK}` (surface–surface part)
===========================================================================

Similar to the :math:`\mathcal{TL}` operator, the goal is to reformulate the tested entry of the :math:`\mathcal{K}` operator to handle self and near–singular interactions.

Step 1 — Starting equation
--------------------------

We begin from the surface–surface contribution given by :eq:`eq:TK_mjni_surface_surface`:

.. math::

   TK_{mj,ni}
   =\frac{e_{mj}e_{ni}}{4A_mA_n}\,
   \iint_{f_m\times f_n}\;
   \big(\boldsymbol\rho_{mj}(\mathbf r)\times \boldsymbol\rho_{ni}(\mathbf r')\big)\cdot\;
   \Big(jk_i+\tfrac{1}{R}\Big)\,G_i(\mathbf r,\mathbf r') \hat{\mathbf R}\; dS\, dS'.

Step 2 — Isolate outer testing term
------------------------------------

Using the vector triple product identity :math:`(\mathbf A \times \mathbf B)\cdot \mathbf C = \mathbf A \cdot (\mathbf B \times \mathbf C)`,
we can rearrange the scalar triple product inside the integral to pull the testing term :math:`\boldsymbol\rho_{mj}(\mathbf r)` outside the inner integral over the source face :math:`f_n`.

Let :math:`\mathbf A = \boldsymbol\rho_{mj}(\mathbf r)`, :math:`\mathbf B = \boldsymbol\rho_{ni}(\mathbf r')`, and :math:`\mathbf C = \hat{\mathbf R}`. Then:

.. math::

   \big(\boldsymbol\rho_{mj}(\mathbf r)\times \boldsymbol\rho_{ni}(\mathbf r')\big)\cdot \hat{\mathbf R}
   \;=\;
   \boldsymbol\rho_{mj}(\mathbf r)\cdot \big(\boldsymbol\rho_{ni}(\mathbf r') \times \hat{\mathbf R}\big).

Substituting this back into the expression for :math:`TK_{mj,ni}` allows us to define an inner vector integral that only depends on the source geometry and the outer observation point :math:`\mathbf r`.

The expression now takes the form of an outer integral over the testing face :math:`f_m` and an inner integral over the source face :math:`f_n`:

.. math::
   :label: eq:TK_mjni_outer_isolated

   TK_{mj,ni}
   \;=\;
   \frac{e_{mj}e_{ni}}{4A_mA_n}\,
   \int_{f_m} \boldsymbol\rho_{mj}(\mathbf r) \cdot
   \left[
     \int_{f_n}
       \Big(jk_i+\tfrac{1}{R}\Big)\,G_i(\mathbf r,\mathbf r')\;
       \big(\boldsymbol\rho_{ni}(\mathbf r') \times \hat{\mathbf R}\big)
     \, dS'
   \right] dS.

Step 3 — Inner analytic integral and outer quadrature
--------------------------------------------------------


The evaluation proceeds by separating the two integrals. The outer integral over the testing face :math:`f_m` is computed using a standard numerical quadrature rule (e.g., Dunavant), while the inner integral over the source face :math:`f_n` is handled analytically.

Let's define the inner vector integral for a given observation point :math:`\mathbf r` as:

.. math::
   :label: eq:TK_inner_integral_def

   \boldsymbol{\Upsilon}(\mathbf r)
   \;:=\;
   \int_{f_n}
     \Big(jk_i+\tfrac{1}{R}\Big)\,G_i(\mathbf r,\mathbf r')\;
     \big(\boldsymbol\rho_{ni}(\mathbf r') \times \hat{\mathbf R}\big)
   \, dS'.

The focus now shifts to finding a closed-form or semi-analytic expression for :math:`\boldsymbol{\Upsilon}(\mathbf r)`.

Step 4 — Simplification of the Inner Integrand
-----------------------------------------------

To simplify the integrand of :math:`\boldsymbol{\Upsilon}(\mathbf r)`, we first decompose the vector :math:`\mathbf{R} = \mathbf{r} - \mathbf{r}'`.
Let :math:`\mathbf{r}'_{\overline{\partial e_{ni}}}` be the free vertex associated with the source half-basis :math:`\mathbf{f}_{ni}`. We define a constant vector (with respect to the inner integral) :math:`\mathbf{R}_i = \mathbf{r} - \mathbf{r}'_{\overline{\partial e_{ni}}}`.

The vector :math:`\mathbf{R}` can now be written as:

.. math::

   \mathbf{R} = (\mathbf{r} - \mathbf{r}'_{\overline{\partial e_{ni}}}) - (\mathbf{r}' - \mathbf{r}'_{\overline{\partial e_{ni}}}) = \mathbf{R}_i - \boldsymbol{\rho}_{ni}(\mathbf{r}').

Using this decomposition, the cross product term in the integrand becomes:

.. math::

   \boldsymbol{\rho}_{ni}(\mathbf{r}') \times \hat{\mathbf{R}}
   \;=\; \boldsymbol{\rho}_{ni}(\mathbf{r}') \times \frac{\mathbf{R}}{R}
   \;=\; \frac{1}{R} \left( \boldsymbol{\rho}_{ni}(\mathbf{r}') \times (\mathbf{R}_i - \boldsymbol{\rho}_{ni}(\mathbf{r}')) \right).

By the distributive property of the cross product, and since :math:`\boldsymbol{\rho}_{ni} \times \boldsymbol{\rho}_{ni} = 0`, this simplifies to:

.. math::

   \boldsymbol{\rho}_{ni}(\mathbf{r}') \times \hat{\mathbf{R}}
   \;=\; \frac{1}{R} \left( \boldsymbol{\rho}_{ni}(\mathbf{r}') \times \mathbf{R}_i \right)
   \;=\; -\frac{1}{R} \left( \mathbf{R}_i \times \boldsymbol{\rho}_{ni}(\mathbf{r}') \right).

Since :math:`\mathbf{R}_i` is constant with respect to the integration variable :math:`\mathbf{r}'`, we can factor it out of the integral.
The inner integral :math:`\boldsymbol{\Upsilon}(\mathbf r)` can now be expressed as:

.. math::
   :label: eq:TK_inner_integral_simplified

   \boldsymbol{\Upsilon}(\mathbf r)
   \;=\;
   -\mathbf{R}_i \times \int_{f_n}
     \frac{1}{R} \Big(jk_i+\tfrac{1}{R}\Big)\,G_i(\mathbf r,\mathbf r')\;
     \boldsymbol{\rho}_{ni}(\mathbf r')
   \, dS'.

Step 5 — Substitute Green's function and factor constants
---------------------------------------------------------

Next, we substitute the definition of the Green's function, :math:`G_i = \frac{e^{-jk_i R}}{4\pi R}`, into the expression and factor out the constant :math:`1/(4\pi)`.

.. math::

   \boldsymbol{\Upsilon}(\mathbf r)
   \;=\;
   -\frac{1}{4\pi} \mathbf{R}_i \times \int_{f_n}
     \frac{1}{R} \left(jk_i+\frac{1}{R}\right) \frac{e^{-jk_i R}}{R}
     \boldsymbol{\rho}_{ni}(\mathbf r')
   \, dS'.

Combining the terms involving :math:`R` gives the final form of the integrand before splitting into regular and singular parts:

.. math::
   :label: eq:TK_inner_integral_expanded

   \boldsymbol{\Upsilon}(\mathbf r)
   \;=\;
   -\frac{1}{4\pi} \mathbf{R}_i \times \int_{f_n}
     \left( \frac{jk_i}{R^2} + \frac{1}{R^3} \right) e^{-jk_i R}
     \boldsymbol{\rho}_{ni}(\mathbf{r}')
   \, dS'.

Step 6 — Split into Regular and Singular Parts
------------------------------------------------

To handle the singularity at :math:`R=0`, we split the kernel inside the integral into a numerically regular part and a singular part that can be related to the analytic potentials :math:`\phi_0` and :math:`\boldsymbol{\phi}_1`. This is achieved by adding and subtracting the singular part of the kernel, which is found from the small-argument Taylor expansion of the expression.

The kernel is:
:math:`K(R) = \left( \frac{jk_i}{R^2} + \frac{1}{R^3} \right) e^{-jk_i R}`.

The expansion for small :math:`R` reveals its singular behavior:
:math:`K(R) \approx \frac{1}{R^3} + \frac{k_i^2}{2R} + \mathcal{O}(1)`.

We define the singular part as :math:`S(R) = \frac{1}{R^3} + \frac{k_i^2}{2R}`.
The integral for :math:`\boldsymbol{\Upsilon}(\mathbf r)` is now split into two parts:

.. math::
   :label: eq:TK_inner_integral_split

   \boldsymbol{\Upsilon}(\mathbf r)
   \;=\;
   \boldsymbol{\Upsilon}^{(\mathrm{reg})}(\mathbf r) + \boldsymbol{\Upsilon}^{(\mathrm{coul})}(\mathbf r)

where the regular part is integrated numerically,

.. math::
   :label: eq:TK_reg_part

   \boldsymbol{\Upsilon}^{(\mathrm{reg})}(\mathbf r)
   \;=\;
   -\frac{1}{4\pi} \mathbf{R}_i \times \int_{f_n}
     \left[ \left( \frac{jk_i}{R^2} + \frac{1}{R^3} \right) e^{-jk_i R} - \left(\frac{1}{R^3} + \frac{k_i^2}{2R}\right) \right]
     \boldsymbol{\rho}_{ni}(\mathbf{r}')
   \, dS'

and the singular (Coulomb) part is handled analytically:

.. math::
   :label: eq:TK_coul_part

   \boldsymbol{\Upsilon}^{(\mathrm{coul})}(\mathbf r)
   \;=\;
   -\frac{1}{4\pi} \mathbf{R}_i \times \int_{f_n}
     \left( \frac{1}{R^3} + \frac{k_i^2}{2R} \right)
     \boldsymbol{\rho}_{ni}(\mathbf{r}')
   \, dS'.

Step 7 — Final form of the regular part
---------------------------------------

The regular part of the operator, :math:`\boldsymbol{\Upsilon}^{(\mathrm{reg})}(\mathbf r)`, can now be substituted back into the main expression for :math:`TK_{mj,ni}` from :eq:`eq:TK_mjni_outer_isolated`. This part of the operator is numerically regular and can be computed using a standard double quadrature (e.g., Dunavant-Dunavant).

First, let's define the regularized kernel function from :eq:`eq:TK_reg_part`:

.. math::
   :label: eq:TK_K_reg_func

   K_{\mathrm{reg}}(R) := \left( \frac{jk_i}{R^2} + \frac{1}{R^3} \right) e^{-jk_i R} - \left(\frac{1}{R^3} + \frac{k_i^2}{2R}\right).

.. admonition:: Numerical Evaluation of :math:`K_{\mathrm{reg}}`
   :class: tip

   Similar to :math:`G_{\mathrm{reg}}`, the definition of :math:`K_{\mathrm{reg}}(R)` is numerically unstable for small :math:`R` due to catastrophic cancellation. The limiting value as :math:`R \to 0` must be derived from the Taylor series expansion of :math:`e^{-jk_i R}`.

   .. math::

      e^{-jk_i R} = 1 - jk_i R - \frac{k_i^2 R^2}{2} + \frac{j k_i^3 R^3}{6} + \mathcal{O}(R^4)

   Substituting this into the expression for :math:`K_{\mathrm{reg}}(R)` and simplifying shows that all singular terms cancel, leaving a constant:

   .. math::

      \lim_{R \to 0} K_{\mathrm{reg}}(R) = -\frac{j k_i^3}{3}

   Therefore, :math:`K_{\mathrm{reg}}` should be implemented as a piecewise function:

   .. math::
      :label: eq:TK_K_reg_func_definition

      K_{\mathrm{reg}}(R) =
      \begin{cases}
         \left( \frac{jk_i}{R^2} + \frac{1}{R^3} \right) e^{-jk_i R} - \left(\frac{1}{R^3} + \frac{k_i^2}{2R}\right), & R > \epsilon \\
         -\frac{j k_i^3}{3}, & R \le \epsilon
      \end{cases}

   where :math:`\epsilon` is a small machine-precision tolerance.

This function is constructed to be finite as :math:`R \to 0`, making it suitable for numerical quadrature. The regular part of the operator is then:

.. math::

   TK_{mj,ni}^{(\mathrm{reg})}
   \;=\;
   \frac{e_{mj}e_{ni}}{4A_mA_n}\,
   \int_{f_m} \boldsymbol\rho_{mj}(\mathbf r) \cdot
   \left(
     -\frac{1}{4\pi} \mathbf{R}_i \times \int_{f_n}
       K_{\mathrm{reg}}(R) \, \boldsymbol{\rho}_{ni}(\mathbf{r}')
     \, dS'
   \right) dS.

Using the vector triple product identity :math:`\mathbf{A} \cdot (\mathbf{B} \times \mathbf{C}) = -(\mathbf{A} \times \mathbf{C}) \cdot \mathbf{B}` and recalling that :math:`\mathbf{R}_i - \boldsymbol{\rho}_{ni}(\mathbf{r}') = \mathbf{R}`, we can revert the expression to a more familiar form for quadrature:

.. math::

   TK_{mj,ni}^{(\mathrm{reg})}
   \;=\;
   \frac{e_{mj}e_{ni}}{4A_mA_n} \frac{1}{4\pi}
   \iint_{f_m\times f_n}
   \left( \boldsymbol{\rho}_{mj}(\mathbf{r}) \times \boldsymbol{\rho}_{ni}(\mathbf{r}') \right) \cdot \mathbf{R}
   \; K_{\mathrm{reg}}(R)
   \; dS' dS.

Since :math:`\mathbf{R} = R \hat{\mathbf{R}}`, this becomes:

.. math::
   :label: eq:TK_reg_final_quadrature

   TK_{mj,ni}^{(\mathrm{reg})}
   \;=\;
   \frac{e_{mj}e_{ni}}{16\pi A_mA_n}
   \iint_{f_m\times f_n}
   \left( \boldsymbol{\rho}_{mj}(\mathbf{r}) \times \boldsymbol{\rho}_{ni}(\mathbf{r}') \right) \cdot \hat{\mathbf{R}}
   \; \big(R \cdot K_{\mathrm{reg}}(R)\big)
   \; dS' dS.

This is now in a standard form for a double numerical quadrature, analogous to the original expression in :eq:`eq:TK_mjni_surface_surface`, but with the singular kernel replaced by the well-behaved term :math:`R \cdot K_{\mathrm{reg}}(R)`.

Step 8 — Compact “kernel × modifier” form for Quadrature
--------------------------------------------------------

To create a reusable quadrature routine, we can express both the standard (far-field) and regularized (near-field) computations in the same compact form. We follow the structure from :eq:`eq:TK_kernel_modifier`:

.. math::

   TK_{mj,ni}
   \;\approx\;
   e_{mj}\,e_{ni}\sum_{p=1}^{Q_m}\sum_{q=1}^{Q_n} w_p\,w'_q\;
   \boldsymbol{M^{(mj,ni)}_{pq}} \cdot \boldsymbol{D_{pq}}

where the geometric modifier :math:`\boldsymbol{M_{pq}}` is always the same:

.. math::
   :label: eq:TK_M_pq_def

   \boldsymbol{M^{(mj,ni)}_{pq}} = \tfrac{1}{4}\big(\boldsymbol{\rho}_{mj}(\mathbf{r}_{m,p}) \times \boldsymbol{\rho}_{ni}(\mathbf{r}'_{n,q})\big) 

The functor for the kernel term :math:`D_{pq}` is what changes depending on the case:

1.  **Standard (Far-Field) Case**:
    The kernel is the standard one, derived from the original operator definition.

    .. math::
       :label: eq:TK_D_pq_standard

       D_{pq}^{(\mathrm{std})} = \left( jk_i + \frac{1}{R_{pq}} \right) G_i(\mathbf{r}_{m,p}, \mathbf{r}'_{n,q})\;
       \hat{\mathbf{R}}_{pq}

2.  **Regularized (Near-Field) Case**:
    The kernel is derived from our regularized expression in :eq:`eq:TK_reg_final_quadrature`. By comparing the general form with that equation, we can identify :math:`D_{pq}`.

    .. math::
       :label: eq:TK_D_pq_regular

       D_{pq}^{(\mathrm{reg})} = \frac{1}{4\pi} \big( K_{\mathrm{reg}}(\mathbf{r}_{m,p}, \mathbf{r}'_{n,q})\;
       R_{pq} \hat{\mathbf{R}}_{pq}\big)

**Implementation:** the TK outputs of |tl-tk-regular-quadrature| use the
standard kernel with ``is_singular=false`` or the regularized kernel with
``is_singular=true``. The total near-field interaction adds the extracted
contribution from |tl-tk-singular-quadrature|.

Step 9 — Coulomb part: decomposition and closed forms
-----------------------------------------------------

Recall from :eq:`eq:TK_coul_part` that the Coulomb inner vector integral is

.. math::
   :label: eq:TK_coul_inner_repeat

   \boldsymbol{\Upsilon}^{(\mathrm{coul})}(\mathbf r)
   \;=\;
   -\frac{1}{4\pi} \, \mathbf{R}_i \times \int_{f_n}
     \left( \frac{1}{R^3} + \frac{k_i^2}{2R} \right)
     \boldsymbol{\rho}_{ni}(\mathbf{r}')\; dS'.

Split the integral into two parts:

.. math::
   :label: eq:TK_coul_split

   \int_{f_n} \left( \frac{1}{R^3} + \frac{k_i^2}{2R} \right) \boldsymbol{\rho}_{ni}'\, dS'
   \;=\;
   \underbrace{\int_{f_n} \frac{\boldsymbol{\rho}_{ni}'}{R^3}\, dS'}_{\displaystyle \boldsymbol{\phi}_3(\mathbf r)}
   \;+
   \;\frac{k_i^2}{2}\underbrace{\int_{f_n} \frac{\boldsymbol{\rho}_{ni}'}{R}\, dS'}_{\displaystyle \boldsymbol{\Psi}_1(\mathbf r)}.

The second integral, :math:`\boldsymbol{\Psi}_1(\mathbf r)`, is the same potential encountered in the :math:`\mathcal{TL}` operator analysis, given by :eq:`eq:Psi1_via_phi`.

The first integral defines a new vector potential, :math:`\boldsymbol{\phi}_3(\mathbf r)`. Its analytic evaluation is given by Hodges et al. (1997) :cite:p:`hodges1997mfie` and follows a structure similar to :math:`\boldsymbol{\phi}_1`.

Closed-form phi3
~~~~~~~~~~~~~~~~

The vector potential :math:`\boldsymbol{\phi}_3` is defined as:

.. math::
   :label: eq:phi3_def

   \boldsymbol{\phi}_3(\mathbf r) := \int_{f_n} \frac{\boldsymbol{\rho}_{ni}(\mathbf r')}{R^3}\, dS'

**Implementation:** :cpp:func:`::calculate_phi_3` evaluates this
free-vertex-dependent potential.

Following the notation in the reference paper :cite:p:`hodges1997mfie` (where :math:`\boldsymbol{\phi}_3` is denoted :math:`\mathbf{g}_i^m`), this can be expressed as:

.. math::
   :label: eq:phi3_hodges

   \boldsymbol{\phi}_3(\mathbf r) = -\sum_i \hat{\mathbf u}_i \ln \frac{R_i^+ + l_i^+}{R_i^- + l_i^-} + (\mathbf{r}_\pi - \mathbf{r}_{\overline{\partial e_{ni}}}) \phi_2(\mathbf r)

where :math:`\mathbf{r}_\pi` is the projection of the observation point :math:`\mathbf{r}` onto the source plane, and :math:`\phi_2` (denoted :math:`g_s^m` in the paper) is the scalar integral over :math:`1/R^3`:

.. math::
   :label: eq:phi_2_def_redef

   \phi_2(\mathbf r) := \int_{f_n} \frac{1}{R^3}\, dS'.

This scalar integral has the closed-form solution:

.. math::
   :label: eq:phi_2_hodges

   \phi_2(\mathbf r) =
   \begin{cases}
      \sum_i (\hat{\mathbf p}_i^0 \cdot \hat{\mathbf u}_i) \frac{1}{|d|} \left( \tan^{-1}\frac{p_i^0 l_i^+}{(R_i^0)^2 + |d|R_i^+} - \tan^{-1}\frac{p_i^0 l_i^-}{(R_i^0)^2 + |d|R_i^-} \right), & d \neq 0 \\
      -\sum_i (\hat{\mathbf p}_i^0 \cdot \hat{\mathbf u}_i) \left( \frac{l_i^+}{p_i^0 R_i^+} - \frac{l_i^-}{p_i^0 R_i^-} \right), & d = 0
   \end{cases}

**Implementation:** :cpp:func:`::calculate_phi_2` evaluates this scalar potential.

The notation used here (:math:`\hat{\mathbf u}_i, R_i^\pm, l_i^\pm, p_i^0, d`, etc.) is the same as that defined for the :math:`\phi_0` potential.

.. note::
   > **Correction:** many transcriptions show a **minus** in the arctangent denominators
   > (:math:`R_{0,i}^2 \color{red}{-} |d|\,R_{i,\pm}`); the correct sign is **plus**.

   - **In-plane, interior of the source triangle:**  
      :math:`\phi_2` is mathematically **unbounded** (non-integrable). However, the divergence of :math:`\phi_2 = \int_{f_n} R^{-3}\,dS'` for an observation point lying inside the source triangle does not imply that the MFIE boundary operator is divergent. The closed-form potentials with :math:`d \neq 0` describe the field away from the current sheet. Taking the one-sided limit :math:`d \to 0^{\pm}` produces the finite MFIE jump term:

      .. math::
         \mathbf{H}^{\tan}_{\mathrm{jump}} = \pm\frac{1}{2}\,\hat{\mathbf{n}}\times\mathbf{J}

      with the sign determined by the trace convention. The principal-value operator is treated separately. For a planar same-triangle RWG interaction, its surface–surface contribution vanishes geometrically (for example, in the :math:`\mathcal{K}` operator, :math:`\mathbf{f}_{mj} \cdot (\mathbf{f}_{ni} \times \mathbf{R}) = 0`). Consequently, the implementation does not evaluate the individually divergent :math:`\phi_2` potential for the interior self-panel case; it evaluates the principal-value contribution and assembles the known :math:`\pm \tfrac{1}{2}` jump term explicitly.

   - **In-plane, *outside* the triangle:**  
      Finite. Use the **coplanar** formula (:math:`d=0`) above.

   - **Off-plane (:math:`d\neq 0`):**  
      Finite, use the **general** formula with the corrected **plus** sign. But care should be taken
      when :math:`|d|` is very small (see below).

.. admonition:: Numerical Evaluation of :math:`\phi_2`
   :class: tip

   Numerical implementation notes for :math:`\phi_2`:

   - For robustness in single precision, evaluate the **sensitive pieces in double** and cast back: the two `atan2` arguments and the final division by :math:`|d|`.
   - Near-coplanar but not exactly :math:`d=0`, a tiny regularization avoids division by zero: :math:`|d|_{\text{eff}}=\max(|d|,\tau L)` with :math:`\tau\sim 10^{-6}` and :math:`L` a triangle length scale (e.g., max edge length). Use the coplanar form when truly :math:`d=0` and the point is not edge-near.
   - Use stable norms for :math:`R` values (e.g., chained `hypot` or FMA) to reduce cancellation: :math:`R_{i,\pm}=\sqrt{p_0^2+l_\pm^2+d^2}`, :math:`R_{0,i}=\sqrt{p_0^2+d^2}`.
   - Be consistent about :math:`\mathbf u_i` (whether :math:`\mathbf t_i\times\mathbf n` or :math:`\mathbf n\times\mathbf t_i`) and the definition of :math:`\mathrm{sgn}_i`.

Finally, the tested single-pair Coulomb contribution is evaluated with a one-loop quadrature over :math:`f_m`:

.. math::
   :label: eq:TK_coul_outer_sum

   TK_{mj,ni}^{(\mathrm{coul})}
   \;\approx\;
   \frac{1}{4\pi A_n}e_{mj}e_{ni}
   \sum_{p=1}^{Q_m} w_p\;
   \frac{1}{4}\,\boldsymbol\rho_{mj}(\mathbf r_{m,p})\cdot
   \boldsymbol{\Upsilon}^{(\mathrm{coul})}(\mathbf r_{m,p})

where :math:`\boldsymbol{\Upsilon}^{(\mathrm{coul})}` is assembled at each quadrature point :math:`\mathbf{r}_{m,p}` using the potentials above:

.. math::
   :label: eq:Upsilon_coul_final_assembly

   \boldsymbol{\Upsilon}^{(\mathrm{coul})}(\mathbf r)
   \;=\;
   - \, \mathbf{R}_i \times \left( \boldsymbol{\phi}_3(\mathbf r) + \frac{k_i^2}{2} \boldsymbol{\Psi}_1(\mathbf r) \right).

**Implementation:** |upsilon-potential| assembles this inner vector from
precomputed potentials. The TK outputs of |tl-tk-singular-quadrature|
perform the outer quadrature and apply the common normalization factor.

.. note::
   The same numerical care for the :math:`d=0` case and the use of ``atan2`` should be taken when implementing :math:`J_0` as was done for :math:`\phi_0`.

Singular Extraction for the :math:`n\mathcal{TL}` operator
==========================================================

For the :math:`n\mathcal{TL}` operator (used in N-PMCHWT formulations), we found in the theory formulation that avoiding surface integration by parts on the testing triangle is computationally advantageous because it eliminates the need to evaluate un-cancelling line integrals. However, it leaves us with a strongly singular inner integral over :math:`\nabla' G`. 

Recall the evaluated operator form from theory:

.. math::
   n\mathcal{TL}_{mj,ni} = \iint_{f_m \times f_n} \big(\mathbf{t}_m \cdot \mathbf{f}_{ni}\big)\, G \, dS\, dS'
   - \frac{1}{k^2} \iint_{f_m\times f_n} (\nabla'_S\!\cdot \mathbf{f}_{ni})\ \big(\mathbf{t}_m\cdot \nabla' G\big)\, dS'\, dS.

We must split both :math:`G` and :math:`\nabla' G` into their regular and analytically extracted components.
For the scalar Green's function, we use the standard split: :math:`G = G_{\mathrm{reg}} + \frac{1}{4\pi R}`.
For the gradient of the Green's function, we reuse the exact same singularity extraction we developed for the :math:`\mathcal{K}` operator in :eq:`eq:TK_reg_part`:

.. math::
   \nabla' G = \frac{1}{4\pi} K_{\mathrm{reg}}(R) \mathbf{R} + \frac{1}{4\pi} \left( \frac{1}{R^3} + \frac{k^2}{2R} \right) \mathbf{R}

where :math:`\mathbf{R} = \mathbf{r} - \mathbf{r}'` and :math:`K_{\mathrm{reg}}` is defined in :eq:`eq:TK_K_reg_func_definition`.

The Regular Part
----------------

The regular part of the :math:`n\mathcal{TL}` operator can be evaluated using a standard Dunavant double quadrature over the well-behaved integrands :math:`G_{\mathrm{reg}}` and :math:`K_{\mathrm{reg}}`. As shown in the theory section, this takes the form:

.. math::
   :label: eq:nTL_dunavant_reg

   n\mathcal{TL}_{mj,ni}^{(\mathrm{reg})} \approx e_{mj}e_{ni} \sum_{p=1}^{Q_m} \sum_{q=1}^{Q_n} w_p w_q \left[ \frac{1}{4} \big( (\boldsymbol{\rho}_{mj,p} \times \hat{\mathbf{n}}_m) \cdot \boldsymbol{\rho}_{ni,q} \big) G^{(\mathrm{reg})}_{pq} - \frac{1}{8\pi k^2} \big( (\boldsymbol{\rho}_{mj,p} \times \hat{\mathbf{n}}_m) \cdot (K_{\mathrm{reg}}(R_{pq}) \mathbf{R}_{pq}) \big) \right]

The Analytically Extracted Part
-------------------------------

The extracted part of the operator requires analytical evaluation of the inner surface integrals over :math:`f_n`. Substituting the basis function definitions :math:`\mathbf{f}_{ni} = \frac{e_{ni}}{2A_n}\boldsymbol{\rho}_{ni}` and its divergence :math:`\nabla'_S \cdot \mathbf{f}_{ni} = \frac{e_{ni}}{A_n}`, the inner vector field that dots with the testing function :math:`\mathbf{t}_m` is:

.. math::
   \mathbf{V}_{\mathrm{sing}}(\mathbf{r}) = \int_{f_n} \frac{e_{ni}}{2A_n} \boldsymbol{\rho}_{ni}(\mathbf{r}') \frac{1}{4\pi R} dS' - \frac{1}{k^2} \int_{f_n} \frac{e_{ni}}{A_n} \frac{1}{4\pi} \left( \frac{1}{R^3} + \frac{k^2}{2R} \right) \mathbf{R} \, dS'

By noting that :math:`\mathbf{R} = \mathbf{r} - \mathbf{r}' = \mathbf{R}_i - \boldsymbol{\rho}_{ni}(\mathbf{r}')` (where :math:`\mathbf{R}_i = \mathbf{r} - \mathbf{r}_{\mathrm{free}}`), we can expand the second integral and group terms exactly as was done for the :math:`\mathcal{K}` operator. This elegantly reduces to a linear combination of the pre-existing analytical potentials:

.. math::
   :label: eq:nTL_coul_inner

   \mathbf{V}_{\mathrm{sing}}(\mathbf{r}) = \frac{e_{ni}}{4\pi A_n} \left[ \boldsymbol{\Psi}_1(\mathbf{r}) + \frac{1}{k^2} \big( \boldsymbol{\phi}_3(\mathbf{r}) - \mathbf{R}_i \phi_2(\mathbf{r}) \big) - \frac{1}{2} \mathbf{R}_i \phi_0(\mathbf{r}) \right]

By substituting :math:`\boldsymbol{\Psi}_1(\mathbf{r}) = \boldsymbol{\phi}_1(\mathbf{r}) + \mathbf{R}_i \phi_0(\mathbf{r})`, we can express this fully in terms of the fundamental potentials:

.. math::
   \mathbf{V}_{\mathrm{sing}}(\mathbf{r}) = \frac{e_{ni}}{4\pi A_n} \left[ \boldsymbol{\phi}_1(\mathbf{r}) + \frac{1}{2} \mathbf{R}_i \phi_0(\mathbf{r}) + \frac{1}{k^2} \big( \boldsymbol{\phi}_3(\mathbf{r}) - \mathbf{R}_i \phi_2(\mathbf{r}) \big) \right]

Finally, the total tested single-pair singular contribution is evaluated using the outer Dunavant quadrature over :math:`f_m`. Since the testing function is :math:`\mathbf{t}_m = \frac{e_{mj}}{2 A_m} \boldsymbol{\tau}_{mj}` (with :math:`\boldsymbol{\tau}_{mj} = \boldsymbol{\rho}_{mj} \times \hat{\mathbf{n}}_m`), the area :math:`A_m` cancels with the Dunavant area Jacobian, yielding:

.. math::
   :label: eq:nTL_coul_outer_sum

   n\mathcal{TL}_{mj,ni}^{(\mathrm{sing})} \approx \frac{e_{mj}e_{ni}}{8\pi A_n} \sum_{p=1}^{Q_m} w_p \, \boldsymbol{\tau}_{mj,p} \cdot \left[ \boldsymbol{\phi}_1 + \frac{1}{2}\mathbf{R}_i \phi_0 + \frac{1}{k^2} (\boldsymbol{\phi}_3 - \mathbf{R}_i \phi_2) \right]_{\mathbf{r}=\mathbf{r}_{m,p}}

**Implementation:** :cpp:func:`::calculateSingularPartQuadrature_nTL_nTK`
evaluates the extracted Form B contribution using the batched Wilton potentials.

Numerical Stability of the Extracted Potentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A critical implementation detail arises when evaluating the combination :math:`\boldsymbol{\phi}_3(\mathbf{r}) - \mathbf{R}_i \phi_2(\mathbf{r})`. If the observation point :math:`\mathbf{r}` lies on or very near the source triangle :math:`f_n`, both :math:`\boldsymbol{\phi}_3` and :math:`\mathbf{R}_i \phi_2` independently contain terms that diverge as :math:`1/d` (where :math:`d` is the distance to the plane). Subtracting these two large numbers directly would lead to catastrophic cancellation.

However, the Wilton implementation of these potentials in the codebase provides a mathematically stable path. 
Recall the definition of :math:`\boldsymbol{\phi}_3` from :eq:`eq:phi3_hodges`:
:math:`\boldsymbol{\phi}_3(\mathbf{r}) = \boldsymbol{\phi}_{3,\mathrm{base}} + (\mathbf{r}_\pi - \mathbf{r}_{\mathrm{free}}) \phi_2(\mathbf{r})`
where :math:`\mathbf{r}_\pi` is the projection of :math:`\mathbf{r}` onto the triangle plane, and :math:`\boldsymbol{\phi}_{3,\mathrm{base}} = -\sum_i \hat{\mathbf{u}}_i \ln(\dots)`.

Since the observation point is :math:`\mathbf{r} = \mathbf{r}_\pi + d\hat{\mathbf{n}}`, we have :math:`\mathbf{R}_i = (\mathbf{r}_\pi - \mathbf{r}_{\mathrm{free}}) + d\hat{\mathbf{n}}`. 
Substituting this into the combination yields a perfect algebraic cancellation of the problematic in-plane vector:

.. math::
   \boldsymbol{\phi}_3 - \mathbf{R}_i \phi_2 = \boldsymbol{\phi}_{3,\mathrm{base}} + (\mathbf{r}_\pi - \mathbf{r}_{\mathrm{free}}) \phi_2 - \big( (\mathbf{r}_\pi - \mathbf{r}_{\mathrm{free}}) + d\hat{\mathbf{n}} \big) \phi_2 = \boldsymbol{\phi}_{3,\mathrm{base}} - d \, \phi_2 \, \hat{\mathbf{n}}

This reveals that the dangerous term :math:`(\mathbf{r}_\pi - \mathbf{r}_{\mathrm{free}}) \phi_2` vanishes entirely! We only need to evaluate :math:`\boldsymbol{\phi}_{3,\mathrm{base}} - d \phi_2 \hat{\mathbf{n}}`. 
Furthermore, since :math:`\phi_2 \sim 1/d` for small :math:`d`, the term :math:`d \phi_2 \hat{\mathbf{n}}` approaches a bounded limit as :math:`d \to 0` (evaluating to the sum of the arctangent solid angles, which mathematically represents the analytical jump term). However, a fundamental principle of boundary integral equations is that these jump limits are analytically separated from the principal-value (PV) integrals during the initial formulation. 

The numerical integration loops are designed to evaluate strictly the PV of the operator. Therefore, even in cases where the jump does not vanish identically (such as in the :math:`n\mathcal{TK}` operator, where the jump yields the mass matrix :math:`\mp \tfrac{1}{2} \int \mathbf{f}_m \cdot \mathbf{f}_n \, dS`), these terms are assembled entirely outside the PV integration loops. For the self-panel PV evaluation, the computation simply ignores the jump contribution and relies strictly on the well-behaved in-plane logarithmic sum :math:`\boldsymbol{\phi}_{3,\mathrm{base}}`. The existing implementation of :cpp:func:`::calculateAllWiltonPotentials` cleanly exposes this base term, allowing the PV combination to be evaluated with absolute numerical stability.


Singularity Extraction for the Line Integral Formulation of :math:`n\mathcal{TL}` (Form A)
------------------------------------------------------------------------------------------

Recall from theory (see :eq:`eq:nTL_formA_operator`) that the Form A representation of the :math:`n\mathcal{TL}` operator is given by:

.. math::
   n\mathcal{TL}_{mj,ni} = \iint_{f_m \times f_n} \big(\mathbf{t}_m \cdot \mathbf{f}_{ni}\big)\, G \, dS\, dS'
   + \frac{1}{k^2} \oint_{\partial f_m} \int_{f_n} (\mathbf{t}_m \cdot \boldsymbol{\nu}_m) (\nabla'_S \cdot \mathbf{f}_{ni}) \, G \, dS'\, dl

The first term is a double surface integral containing the scalar Green's function :math:`G`. Its singularity extraction was already treated in the previous section: the regularized part is evaluated numerically with 2D Dunavant quadrature, while the singular :math:`1/(4\pi R)` component is extracted and evaluated analytically using the vector potential :math:`\boldsymbol{\Phi}_1`.

Therefore, we focus exclusively on the second term: the coupled line-surface integral along the boundary :math:`\partial f_m` of the testing triangle:

.. math::
   :label: eq:nTL_formA_line_term

   \mathcal{I}_{\mathrm{line}} = \frac{1}{k^2} \oint_{\partial f_m} \int_{f_n} (\mathbf{t}_m \cdot \boldsymbol{\nu}_m) (\nabla'_S \cdot \mathbf{f}_{ni}) \, G(\mathbf{r}, \mathbf{r}') \, dS'\, dl

The Regular Part
~~~~~~~~~~~~~~~~

The regular part of the line integral from :eq:`eq:nTL_formA_line_discrete` is obtained simply by substituting the regularized Green's function :math:`G_{\mathrm{reg}}(\mathbf{r}, \mathbf{r}') = G(\mathbf{r}, \mathbf{r}') - \frac{1}{4\pi R}` for :math:`G(\mathbf{r}, \mathbf{r}')`:

.. math::
   :label: eq:nTL_formA_line_reg

   \mathcal{I}_{\mathrm{line}}^{(\mathrm{reg})} = \frac{e_{mj} e_{ni}}{2 A_m k^2} \sum_{e=1}^3 L_e \sum_{q=1}^{Q_{1D}} \sum_{p=1}^{Q_{2D}} w_q^{1D} w_p^{2D} \Big[ \big( \boldsymbol{\rho}_{mj}(\mathbf{r}_{eq}) \times \hat{\mathbf{n}}_m \big) \cdot \boldsymbol{\nu}_e \Big] G_{\mathrm{reg}}(\mathbf{r}_{eq}, \mathbf{r}_{np}')

It is important to notice that this expression accumulates the contribution over all three edges of the testing triangle. Because :math:`G_{\mathrm{reg}}` is smooth and bounded as :math:`R \to 0`, this term is evaluated numerically with standard coupled 1D Gauss-Legendre quadrature along the testing edges and 2D Dunavant quadrature over the source triangle.

The Analytically Extracted Part
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To obtain the analytically extracted singular part, we substitute the singular potential component :math:`\frac{1}{4\pi R}` into the line integral formulation of :eq:`eq:nTL_formA_line_term`:

.. math::
   :label: eq:nTL_formA_line_sing

   \mathcal{I}_{\mathrm{line}}^{(\mathrm{sing})} = \frac{1}{k^2} \oint_{\partial f_m} \int_{f_n} (\mathbf{t}_m \cdot \boldsymbol{\nu}_m) (\nabla'_S \cdot \mathbf{f}_{ni}) \frac{1}{4\pi R} \, dS'\, dl

Substituting the definitions of the rotated testing vector :math:`\mathbf{t}_m = \frac{e_{mj}}{2A_m} (\boldsymbol{\rho}_{mj} \times \hat{\mathbf{n}}_m)` and the divergence of the source basis function :math:`\nabla'_S \cdot \mathbf{f}_{ni} = \frac{e_{ni}}{A_n}`, we can extract the inner integral over the source triangle:

.. math::

   \mathcal{I}_{\mathrm{line}}^{(\mathrm{sing})} = \frac{e_{mj} e_{ni}}{2 A_m A_n k^2} \oint_{\partial f_m} \Big[ \big( \boldsymbol{\rho}_{mj}(\mathbf{r}) \times \hat{\mathbf{n}}_m \big) \cdot \boldsymbol{\nu}_m \Big] \left[ \frac{1}{4\pi} \int_{f_n} \frac{1}{R} \, dS' \right] dl

We immediately recognize the inner integral as the semi-analytic scalar Wilton potential :math:`\phi_0(\mathbf{r})` defined in :eq:`eq:phi0_wilton`:

.. math::

   \int_{f_n} \frac{1}{R} \, dS' = \phi_0(\mathbf{r})

Therefore, the singular contribution can be evaluated using a 1D Gauss-Legendre quadrature to compute the outer line integral over the three edges of the testing triangle:

.. math::
   :label: eq:nTL_formA_line_sing_discrete

   \mathcal{I}_{\mathrm{line}}^{(\mathrm{sing})} \approx \frac{e_{mj} e_{ni}}{8 \pi A_m A_n k^2} \sum_{e=1}^3 L_e \sum_{q=1}^{Q_{1D}} w_q^{1D} \Big[ \big( \boldsymbol{\rho}_{mj}(\mathbf{r}_{eq}) \times \hat{\mathbf{n}}_m \big) \cdot \boldsymbol{\nu}_e \Big] \phi_0(\mathbf{r}_{eq})

where :math:`\mathbf{r}_{eq}` are the 1D quadrature points mapped to the :math:`e`-th edge of the testing triangle, and :math:`\phi_0(\mathbf{r}_{eq})` is the scalar Wilton potential evaluated precisely at those boundary points.

Total Extracted Form (Summary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To provide a clear, unambiguous reference for implementation, the complete analytical extraction for the :math:`n\mathcal{TL}` operator (Form A) is evaluated as the sum of its four discrete components:

.. math::
   :label: eq:nTL_formA_total

   n\mathcal{TL}_{mj,ni} = \mathcal{I}_{\mathrm{area}}^{(\mathrm{reg})} + \mathcal{I}_{\mathrm{area}}^{(\mathrm{sing})} + \mathcal{I}_{\mathrm{line}}^{(\mathrm{reg})} + \mathcal{I}_{\mathrm{line}}^{(\mathrm{sing})}

The regular area and line integrals evaluate the smooth :math:`G_{\mathrm{reg}} = G - \frac{1}{4\pi R}` kernel:

.. math::

   \mathcal{I}_{\mathrm{area}}^{(\mathrm{reg})} &= \frac{e_{mj}e_{ni}}{4} \sum_{p=1}^{Q_{2D}} \sum_{q=1}^{Q_{2D}} w_p^{2D} w_q^{2D} \Big[ (\boldsymbol{\rho}_{mj}(\mathbf{r}_{mp}) \times \hat{\mathbf{n}}_m) \cdot \boldsymbol{\rho}_{ni}(\mathbf{r}_{nq}') \Big] G_{\mathrm{reg}}(\mathbf{r}_{mp}, \mathbf{r}_{nq}') \\
   \mathcal{I}_{\mathrm{line}}^{(\mathrm{reg})} &= \frac{e_{mj} e_{ni}}{2 A_m k^2} \sum_{e=1}^3 L_e \sum_{q=1}^{Q_{1D}} \sum_{p=1}^{Q_{2D}} w_q^{1D} w_p^{2D} \Big[ \big( \boldsymbol{\rho}_{mj}(\mathbf{r}_{eq}) \times \hat{\mathbf{n}}_m \big) \cdot \boldsymbol{\nu}_e \Big] G_{\mathrm{reg}}(\mathbf{r}_{eq}, \mathbf{r}_{np}')

The singular area and line integrals evaluate the extracted :math:`\frac{1}{4\pi R}` kernel analytically using the Wilton potentials :math:`\boldsymbol{\Psi}_1(\mathbf{r}) = \int_{f_n} \boldsymbol{\rho}_{ni}(\mathbf{r}') \frac{1}{R} dS'` and :math:`\phi_0(\mathbf{r}) = \int_{f_n} \frac{1}{R} dS'`:

.. math::

   \mathcal{I}_{\mathrm{area}}^{(\mathrm{sing})} &= \frac{e_{mj}e_{ni}}{16\pi A_n} \sum_{p=1}^{Q_{2D}} w_p^{2D} \Big[ (\boldsymbol{\rho}_{mj}(\mathbf{r}_{mp}) \times \hat{\mathbf{n}}_m) \cdot \boldsymbol{\Psi}_1(\mathbf{r}_{mp}) \Big] \\
   \mathcal{I}_{\mathrm{line}}^{(\mathrm{sing})} &= \frac{e_{mj} e_{ni}}{8 \pi A_m A_n k^2} \sum_{e=1}^3 L_e \sum_{q=1}^{Q_{1D}} w_q^{1D} \Big[ \big( \boldsymbol{\rho}_{mj}(\mathbf{r}_{eq}) \times \hat{\mathbf{n}}_m \big) \cdot \boldsymbol{\nu}_e \Big] \phi_0(\mathbf{r}_{eq})

This explicit separation dictates exactly how the integration loops and semi-analytic potentials should be invoked in the codebase.

**Implementation:** :cpp:func:`::calculateFormASingularPartQuadrature_nTL_nTK`
evaluates the extracted area and line terms. Its area loop uses |psi1-potential|,
and its edge loop uses :cpp:func:`::calculate_phi_0`.

Singular Extraction for the :math:`n\mathcal{TK}` operator
==========================================================

The analytical singularity extraction for the :math:`n\mathcal{TK}` operator is remarkably straightforward. As established in the theoretical derivation, the :math:`n\mathcal{TK}` operator preserves the exact same inner integral as the standard :math:`\mathcal{TK}` operator, simply replacing the testing function :math:`\mathbf{f}_{mj}` with the rotated testing function :math:`\mathbf{t}_m = \mathbf{f}_{mj} \times \hat{\mathbf{n}}_m`.

Therefore, the entire derivation of the regularized kernel :math:`K_{\mathrm{reg}}` and the analytical extraction of the Coulomb singularity :math:`\boldsymbol{\Upsilon}^{(\mathrm{coul})}` remains perfectly valid without any modifications.

The Regular Part
----------------

The regular part is evaluated with the standard Dunavant double quadrature, exactly mirroring :eq:`eq:TK_reg_final_quadrature` but substituting the rotated testing vector :math:`\boldsymbol{\tau}_{mj} = \boldsymbol{\rho}_{mj} \times \hat{\mathbf{n}}_m`:

.. math::
   :label: eq:nTK_reg_final_quadrature

   n\mathcal{TK}_{mj,ni}^{(\mathrm{reg})}
   \;\approx\;
   \frac{e_{mj}e_{ni}}{16\pi}
   \sum_{p=1}^{Q_m} \sum_{q=1}^{Q_n} w_p w_q
   \left( \boldsymbol{\tau}_{mj,p} \times \boldsymbol{\rho}_{ni,q} \right) \cdot \hat{\mathbf{R}}_{pq}
   \; \big(R_{pq} \cdot K_{\mathrm{reg}}(R_{pq})\big)

The Analytically Extracted Part
-------------------------------

The extracted singular part relies on the identical inner Coulomb vector :math:`\boldsymbol{\Upsilon}^{(\mathrm{coul})}(\mathbf{r})` defined in :eq:`eq:Upsilon_coul_final_assembly`:

.. math::
   \boldsymbol{\Upsilon}^{(\mathrm{coul})}(\mathbf{r}) = - \, \mathbf{R}_i \times \left( \boldsymbol{\phi}_3(\mathbf{r}) + \frac{k_i^2}{2} \boldsymbol{\Psi}_1(\mathbf{r}) \right)

The outer integration over the testing face :math:`f_m` applies the Dunavant rule to the dot product of the rotated testing function and this analytical inner vector. Using :math:`\mathbf{t}_m = \frac{e_{mj}}{2A_m} \boldsymbol{\tau}_{mj}`, the area :math:`A_m` cancels with the Dunavant Jacobian, leaving the exact same constant prefactor as the standard :math:`\mathcal{TK}` operator:

.. math::
   :label: eq:nTK_coul_outer_sum

   n\mathcal{TK}_{mj,ni}^{(\mathrm{sing})}
   \;\approx\;
   \frac{e_{mj}e_{ni}}{16\pi A_n}
   \sum_{p=1}^{Q_m} w_p\;
   \boldsymbol{\tau}_{mj,p} \cdot \boldsymbol{\Upsilon}^{(\mathrm{coul})}(\mathbf{r}_{m,p})

Because this relies exclusively on the existing :math:`\boldsymbol{\Upsilon}^{(\mathrm{coul})}` implementation, no new analytical potential limits or stability checks are required for the :math:`n\mathcal{TK}` operator.

**Implementation:** the nTK outputs of
:cpp:func:`::calculateSingularPartQuadrature_nTL_nTK` (Form B) and
:cpp:func:`::calculateFormASingularPartQuadrature_nTL_nTK` (Form A) both
use |upsilon-potential| for the analytical inner integral.

.. |tl-tk-regular-quadrature| replace:: :cpp:func:`calculateDoubleQuadrature() <void ::calculateDoubleQuadrature(const vec3_t[3], const int[3], const vec3_t[3], const int[3], const float*, const float*, int, real_t, real_t, bool, complex__float[3][3], complex__float[3][3], complex__float[3][3], complex__float[3][3])>`

.. |tl-tk-singular-quadrature| replace:: :cpp:func:`calculateSingularPartQuadrature() <void ::calculateSingularPartQuadrature(const vec3_t[3], const int[3], const vec3_t[3], const int[3], const float*, const float*, int, real_t, real_t, complex__float[3][3], complex__float[3][3], complex__float[3][3], complex__float[3][3])>`

.. |phi1-potential| replace:: :cpp:func:`calculate_phi_1() <vec3_t calculate_phi_1(const vec3_t&, const vec3_t[3])>`

.. |psi1-potential| replace:: :cpp:func:`calculate_Psi_1() <vec3_t calculate_Psi_1(const WiltonPotentials&, const vec3_t&)>`

.. |upsilon-potential| replace:: :cpp:func:`calculate_Upsilon_coul() <vec3_t calculate_Upsilon_coul(const WiltonPotentials&, const vec3_t&, const vec3_t&, float)>`
