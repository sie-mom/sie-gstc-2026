Implementing GSTCs in COMSOL — Case A: Zero-Thickness Sheet
============================================================

Purpose
-------

This annex documents the logic of the zero-thickness GSTC implementation
used in COMSOL for a spherical metasurface. The goal is not to reproduce
COMSOL's finite-element machinery, but to connect four levels that are visible
through *Equation View*:

1. the distributional strong Maxwell equations;
2. the bulk electric-field wave equation;
3. the weak domain and boundary contributions assembled by COMSOL; and
4. the tangential-field constraint used for the magnetic surface current.

The discussion uses the ``emw`` namespace appearing in the present RF model.
In other COMSOL interfaces or versions the corresponding namespace can be
``ewfd``; the mathematical structure is the same.

Throughout, the assumed phasor convention is

.. math::

   \mathbf{F}(\mathbf r,t)
   = \Re\!\left\{\mathbf{F}(\mathbf r)e^{i\omega t}\right\}.

Accordingly,

.. math::

   \frac{\partial}{\partial t}
   \longleftrightarrow i\omega.

Geometry and notation
---------------------

Let the computational domain be split by the zero-thickness metasurface
:math:`\Gamma`:

.. math::

   \Omega = \Omega^- \cup \Gamma \cup \Omega^+.

The traces on the two sides of the sheet are denoted by

.. math::

   \mathbf F^- = \operatorname{down}(\mathbf F),
   \qquad
   \mathbf F^+ = \operatorname{up}(\mathbf F).

Define

.. math::

   [\mathbf F] = \mathbf F^+ - \mathbf F^-,

and

.. math::

   \langle \mathbf F\rangle
   = \frac{\mathbf F^+ + \mathbf F^-}{2}.

Let :math:`\mathbf n` denote the chosen unit normal and

.. math::

   \mathbf P_t = \mathbf I - \mathbf n\mathbf n^T

the tangential projector.

A useful surface-distribution identity
--------------------------------------

For a vector field that is smooth on each side of :math:`\Gamma` but has a
jump across the sheet, the distributional curl contains a regular part and a
surface-supported singular part:

.. math::

   \overline{\nabla\times\mathbf E}
   =
   \{\nabla\times\mathbf E\}
   +
   \mathbf n\times[\mathbf E]\,\delta_\Gamma.

Here :math:`\delta_\Gamma` is the surface delta distribution, characterized by

.. math::

   \int_\Omega f(\mathbf r)\,\delta_\Gamma(\mathbf r)\,dV
   =
   \int_\Gamma f(\mathbf r)\,dS.

This identity is the bridge between the distributional description of a
zero-thickness sheet and the jump conditions used by COMSOL.

1. Distributional strong Maxwell equations
-------------------------------------------

For electric and magnetic sheet currents :math:`\mathbf J_s` and
:math:`\mathbf J_{ms}`, write

.. math::

   \nabla\times\mathbf E
   =
   -i\omega\mathbf B
   -
   \mathbf J_{ms}\,\delta_\Gamma,

.. math::

   \nabla\times\mathbf H
   =
   \mathbf J_{\mathrm{bulk}}
   +
   \mathbf J_s\,\delta_\Gamma,

where

.. math::

   \mathbf J_{\mathrm{bulk}}
   =
   \boldsymbol{\sigma}\mathbf E
   +
   i\omega\mathbf D.

The constitutive relations in the ordinary bulk media are

.. math::

   \mathbf D^\pm
   =
   \boldsymbol{\epsilon}^\pm\mathbf E^\pm,

.. math::

   \mathbf B^\pm
   =
   \boldsymbol{\mu}^\pm\mathbf H^\pm.

Matching the regular and singular parts gives the bulk Maxwell equations and

.. math::

   \boxed{\mathbf n\times[\mathbf E]=-\mathbf J_{ms}},

.. math::

   \boxed{\mathbf n\times[\mathbf H]=\mathbf J_s}.

Thus the surface distributions can be replaced by ordinary bulk equations plus
interface jump conditions.

2. Reduction to the electric-field wave equation
-------------------------------------------------

Inside either bulk region,

.. math::

   \nabla\times\mathbf E
   =
   -i\omega\boldsymbol{\mu}\mathbf H,

hence

.. math::

   \mathbf H
   =
   -\frac{1}{i\omega}
   \boldsymbol{\mu}^{-1}
   \nabla\times\mathbf E.

Substituting into Ampère's law gives

.. math::

   \boxed{
   \nabla\times
   \left(
   \boldsymbol{\mu}^{-1}\nabla\times\mathbf E
   \right)
   -
   \omega^2\boldsymbol{\epsilon}\mathbf E
   +
   i\omega\boldsymbol{\sigma}\mathbf E
   =0
   }.

Using relative tensors and :math:`k_0^2=\omega^2\mu_0\epsilon_0`,

.. math::

   \boxed{
   \nabla\times
   \left(
   \boldsymbol{\mu}_r^{-1}\nabla\times\mathbf E
   \right)
   -
   k_0^2\boldsymbol{\epsilon}_r\mathbf E
   +
   i\omega\mu_0\boldsymbol{\sigma}\mathbf E
   =0
   }.

3. COMSOL's constitutive variable chain
---------------------------------------

Equation View exposes the intermediate variables used by the RF solver.

Faraday relation
~~~~~~~~~~~~~~~~

COMSOL defines

.. code-block:: text

   emw.dBdtx = -emw.curlEx
   emw.dBdty = -emw.curlEy
   emw.dBdtz = -emw.curlEz

and

.. code-block:: text

   emw.Bx = -emw.curlEx/emw.iomega
   emw.By = -emw.curlEy/emw.iomega
   emw.Bz = -emw.curlEz/emw.iomega

which correspond to

.. math::

   \frac{\partial\mathbf B}{\partial t}
   =
   -\nabla\times\mathbf E,

and

.. math::

   \mathbf B
   =
   -\frac{1}{i\omega}
   \nabla\times\mathbf E.

.. note::

   **Why does a frequency-domain solver still contain variables called
   ``dBdt`` and ``dHdt``?**

   COMSOL is not time-stepping these quantities. Under the harmonic ansatz

   .. math::

      \mathbf F(\mathbf r,t)
      =
      \Re\{\mathbf F(\mathbf r)e^{i\omega t}\},

   a time derivative is represented algebraically:

   .. math::

      \frac{\partial\mathbf F}{\partial t}
      \longleftrightarrow i\omega\mathbf F.

   Thus ``emw.dBdt`` and ``emw.dHdt`` are phasor representations of the
   physical time derivatives. No numerical differentiation in time is
   performed.

Magnetic constitutive relation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

COMSOL forms :math:`\boldsymbol{\mu}_r^{-1}` and defines

.. math::

   \mathbf H
   =
   \frac{1}{\mu_0}
   \boldsymbol{\mu}_r^{-1}\mathbf B,

and therefore

.. math::

   \boxed{
   \mu_0
   \frac{\partial\mathbf H}{\partial t}
   =
   -
   \boldsymbol{\mu}_r^{-1}
   \nabla\times\mathbf E
   }.

Electric constitutive relation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equation View defines

.. math::

   \mathbf P
   =
   \epsilon_0
   (\boldsymbol{\epsilon}_r-\mathbf I)\mathbf E,

followed by

.. math::

   \mathbf D
   =
   \epsilon_0\mathbf E+\mathbf P
   =
   \epsilon_0\boldsymbol{\epsilon}_r\mathbf E.

The displacement current is

.. math::

   \mathbf J_d=i\omega\mathbf D,

so the total bulk current used by the wave equation is

.. math::

   \boxed{
   \mathbf J
   =
   \boldsymbol{\sigma}\mathbf E
   +
   i\omega\mathbf D
   }.

4. COMSOL's bulk weak expression
--------------------------------

For ``Wave Equation, Electric`` the exported weak expression is

.. code-block:: text

   -mu0_const*(
     -emw.dHdtx*emw.curltestdepEx
     -emw.dHdty*emw.curltestdepEy
     -emw.dHdtz*emw.curltestdepEz
     +emw.iomega*(
        emw.Jx*emw.testdepEx
       +emw.Jy*emw.testdepEy
       +emw.Jz*emw.testdepEz
     )
   )

Let

.. math::

   \mathbf v=\operatorname{test}(\mathbf E).

Then COMSOL's expression is

.. math::

   \mu_0
   \frac{\partial\mathbf H}{\partial t}
   \cdot
   \nabla\times\mathbf v
   -
   i\omega\mu_0
   \mathbf J\cdot\mathbf v.

.. note::
   This form arises directly from applying integration by parts (Green's theorem) to the strong form of Ampère's law, :math:`\nabla \times \mathbf{H} = \mathbf{J} + \dots`. Multiplying by the test function :math:`\mathbf{v}` and integrating over the volume transfers the curl operator from the trial field :math:`\mathbf{H}` onto the test field :math:`\mathbf{v}`.

   Crucially, this "curl swap" generates a natural boundary boundary integral term :math:`\int (\mathbf{n} \times \mathbf{H}) \cdot \mathbf{v} \,dS`, which perfectly matches the definition of an electric surface current :math:`\mathbf{J}_s = \mathbf{n} \times \mathbf{H}`. This explains why :math:`\mathbf{J}_s` (discussed in Section 7) is implemented cleanly as a weak boundary load rather than a harsh Nodal constraint!

Its domain contribution is

.. math::

   R_\Omega(\mathbf E;\mathbf v)
   =
   \int_\Omega
   \left[
   \mu_0
   \frac{\partial\mathbf H}{\partial t}
   \cdot\nabla\times\mathbf v
   -
   i\omega\mu_0
   \mathbf J\cdot\mathbf v
   \right]dV.

Substituting the Equation View definitions gives

.. math::

   \boxed{
   R_\Omega(\mathbf E;\mathbf v)
   =
   \int_\Omega
   \left[
   -
   (\nabla\times\mathbf v)\cdot
   \boldsymbol{\mu}_r^{-1}
   (\nabla\times\mathbf E)
   +
   k_0^2
   \mathbf v\cdot
   \boldsymbol{\epsilon}_r\mathbf E
   -
   i\omega\mu_0
   \mathbf v\cdot
   \boldsymbol{\sigma}\mathbf E
   \right]dV
   }.

.. note::

   **Why does Equation View show an expression but no right-hand side?**

   COMSOL assembles contributions in **residual form**. Instead of displaying

   .. math::

      a(\mathbf E,\mathbf v)=\ell(\mathbf v),

   it stores

   .. math::

      R(\mathbf E;\mathbf v)
      =
      a(\mathbf E,\mathbf v)-\ell(\mathbf v)
      =
      0.

   Every active physics feature adds another term to the global residual.
   After finite-element discretization, a linear problem can be rearranged as

   .. math::

      \mathbf A\mathbf u=\mathbf f,

   but Equation View displays the contributions before this rearrangement.

5. Broken :math:`H(\mathrm{curl})` structure
---------------------------------------------

Because

.. math::

   \mathbf n\times[\mathbf E]
   =
   -\mathbf J_{ms}
   \neq 0,

the sheet cannot be represented as an ordinary interior interface of one
globally conforming :math:`H(\mathrm{curl})` field. Instead,

.. math::

   \mathbf E^-
   \in H(\mathrm{curl},\Omega^-),
   \qquad
   \mathbf E^+
   \in H(\mathrm{curl},\Omega^+),

and the natural broken space is

.. math::

   \boxed{
   V
   =
   H(\mathrm{curl},\Omega^-)
   \times
   H(\mathrm{curl},\Omega^+)
   }.

COMSOL's ``up()`` and ``down()`` operators expose these two traces.

6. GSTC surface constitutive relations
--------------------------------------

Define

.. math::

   \mathbf E_{\mathrm{av}}
   =
   \frac{\mathbf E^++\mathbf E^-}{2},

.. math::

   \mathbf H_{\mathrm{av}}
   =
   \frac{\mathbf H^++\mathbf H^-}{2}.

The exported component expressions can be written compactly as

.. math::

   \boxed{
   \mathbf J_s
   =
   \mathbf P_t
   \left[
   Y_{ee}\mathbf E_{\mathrm{av}}
   -
   X_{em}\,
   \mathbf n\times\mathbf H_{\mathrm{av}}
   \right]
   },

and

.. math::

   \boxed{
   \mathbf J_{ms}
   =
   \mathbf P_t
   \left[
   Z_{mm}\mathbf H_{\mathrm{av}}
   -
   X_{em}\,
   \mathbf n\times\mathbf E_{\mathrm{av}}
   \right]
   }.

.. note::
   It is instructive to contrast what is entered into the COMSOL GUI with what COMSOL generates internally.

   In the GUI's **Surface Current** node, we input the un-projected expressions (averaging the ``up()`` and ``down()`` traces):

   .. code-block:: text

      Js0x = Yee*0.5*(up(emw.Ex)+down(emw.Ex)) - Xem*(ny*0.5*(up(emw.Hz)+down(emw.Hz)) - nz*0.5*(up(emw.Hy)+down(emw.Hy)))
      Js0y = Yee*0.5*(up(emw.Ey)+down(emw.Ey)) + Xem*(nx*0.5*(up(emw.Hz)+down(emw.Hz)) - nz*0.5*(up(emw.Hx)+down(emw.Hx)))
      Js0z = Yee*0.5*(up(emw.Ez)+down(emw.Ez)) - Xem*(nx*0.5*(up(emw.Hy)+down(emw.Hy)) - ny*0.5*(up(emw.Hx)+down(emw.Hx)))

   However, in COMSOL's **Equation View**, these currents are automatically expanded to enforce the tangent-plane projection :math:`\mathbf P_t`. Notice how the combinations of ``nmeshx/y/z`` are explicitly multiplied through our input expressions to ensure the current lies perfectly tangential to the surface:

   .. code-block:: text

      emw.scu1.Jsx = 0.5*((emw.nmeshy*(Yee*(up(emw.Ex)+down(emw.Ex))-Xem*(ny*(up(emw.Hz)+down(emw.Hz))-nz*(up(emw.Hy)+down(emw.Hy))))-emw.nmeshx*(Yee*(up(emw.Ey)+down(emw.Ey))+Xem*(nx*(up(emw.Hz)+down(emw.Hz))-nz*(up(emw.Hx)+down(emw.Hx)))))*emw.nmeshy+(emw.nmeshz*(Yee*(up(emw.Ex)+down(emw.Ex))-Xem*(ny*(up(emw.Hz)+down(emw.Hz))-nz*(up(emw.Hy)+down(emw.Hy))))-emw.nmeshx*(Yee*(up(emw.Ez)+down(emw.Ez))-Xem*(nx*(up(emw.Hy)+down(emw.Hy))-ny*(up(emw.Hx)+down(emw.Hx)))))*emw.nmeshz)

      emw.scu1.Jsy = 0.5*((-emw.nmeshy*(Yee*(up(emw.Ex)+down(emw.Ex))-Xem*(ny*(up(emw.Hz)+down(emw.Hz))-nz*(up(emw.Hy)+down(emw.Hy))))+emw.nmeshx*(Yee*(up(emw.Ey)+down(emw.Ey))+Xem*(nx*(up(emw.Hz)+down(emw.Hz))-nz*(up(emw.Hx)+down(emw.Hx)))))*emw.nmeshx+(emw.nmeshz*(Yee*(up(emw.Ey)+down(emw.Ey))+Xem*(nx*(up(emw.Hz)+down(emw.Hz))-nz*(up(emw.Hx)+down(emw.Hx))))-emw.nmeshy*(Yee*(up(emw.Ez)+down(emw.Ez))-Xem*(nx*(up(emw.Hy)+down(emw.Hy))-ny*(up(emw.Hx)+down(emw.Hx)))))*emw.nmeshz)

      emw.scu1.Jsz = 0.5*((-emw.nmeshz*(Yee*(up(emw.Ex)+down(emw.Ex))-Xem*(ny*(up(emw.Hz)+down(emw.Hz))-nz*(up(emw.Hy)+down(emw.Hy))))+emw.nmeshx*(Yee*(up(emw.Ez)+down(emw.Ez))-Xem*(nx*(up(emw.Hy)+down(emw.Hy))-ny*(up(emw.Hx)+down(emw.Hx)))))*emw.nmeshx+(-emw.nmeshz*(Yee*(up(emw.Ey)+down(emw.Ey))+Xem*(nx*(up(emw.Hz)+down(emw.Hz))-nz*(up(emw.Hx)+down(emw.Hx))))+emw.nmeshy*(Yee*(up(emw.Ez)+down(emw.Ez))-Xem*(nx*(up(emw.Hy)+down(emw.Hy))-ny*(up(emw.Hx)+down(emw.Hx)))))*emw.nmeshy)

These currents depend on the unknown fields on both sides of the sheet.

7. Electric surface current :math:`\mathbf J_s`
-------------------------------------------------

Equation View gives

.. code-block:: text

   mu0_const*emw.iomega*(
     -emw.scu1.Jsx*test(emw.tEdownx)
     -emw.scu1.Jsy*test(emw.tEdowny)
     -emw.scu1.Jsz*test(emw.tEdownz)
   )

on the GSTC boundaries. In vector notation,

.. math::

   \boxed{
   R_{J_s}(\mathbf E;\mathbf v)
   =
   -i\omega\mu_0
   \int_\Gamma
   \mathbf J_s(\mathbf E^+,\mathbf E^-)
   \cdot
   \mathbf v_t^-\,dS
   }.

Because :math:`\mathbf J_s` depends on the unknown fields, this term is part
of the system operator rather than a prescribed forcing term.

.. important::

   The use of the downside test trace in this expression is specific to the
   exported COMSOL implementation. It should be interpreted together with the
   two-trace/slit structure created by the magnetic surface current.

8. Magnetic surface current :math:`\mathbf J_{ms}`
---------------------------------------------------

The Surface Magnetic Current feature has an empty ``Weak Expressions`` table.
Instead, Equation View exposes the constraints

.. code-block:: text

   emw.tEupx-emw.tEdownx
   +emw.dnmeshz*emw.Jmsy
   -emw.dnmeshy*emw.Jmsz

   emw.tEupy-emw.tEdowny
   -emw.dnmeshz*emw.Jmsx
   +emw.dnmeshx*emw.Jmsz

   emw.tEupz-emw.tEdownz
   +emw.dnmeshy*emw.Jmsx
   -emw.dnmeshx*emw.Jmsy

Let :math:`\mathbf n_d` denote COMSOL's ``dnmesh`` normal. The three rows are

.. math::

   \boxed{
   \mathbf E_t^+
   -
   \mathbf E_t^-
   -
   \mathbf n_d\times\mathbf J_{ms}
   =
   0
   }.

For tangential :math:`\mathbf J_{ms}`, this is equivalent to

.. math::

   \boxed{
   \mathbf n_d\times
   (\mathbf E^+-\mathbf E^-)
   =
   -\mathbf J_{ms}
   }.

Thus COMSOL implements the magnetic surface current as a tangential electric
field jump constraint rather than as an ordinary weak boundary load.

The associated test/constraint-force expressions are

.. code-block:: text

   emw.testtEupx-emw.testtEdownx
   emw.testtEupy-emw.testtEdowny
   emw.testtEupz-emw.testtEdownz

which expose the jump of the tangential test field.

.. note::
   As with the electric current, COMSOL's Equation View reveals the rigorous tangent-plane projections applied under the hood. The magnetic surface current is defined under the ``smcd1`` tag (**S**urface **M**agnetic **C**urrent **D**ensity **1**):

   .. code-block:: text

      emw.smcd1.Jmsx = 0.5*((emw.nmeshy*(Zmm*(up(emw.Hx)+down(emw.Hx))-Xem*(ny*(up(emw.Ez)+down(emw.Ez))-nz*(up(emw.Ey)+down(emw.Ey))))-emw.nmeshx*(Zmm*(up(emw.Hy)+down(emw.Hy))+Xem*(nx*(up(emw.Ez)+down(emw.Ez))-nz*(up(emw.Ex)+down(emw.Ex)))))*emw.nmeshy+(emw.nmeshz*(Zmm*(up(emw.Hx)+down(emw.Hx))-Xem*(ny*(up(emw.Ez)+down(emw.Ez))-nz*(up(emw.Ey)+down(emw.Ey))))-emw.nmeshx*(Zmm*(up(emw.Hz)+down(emw.Hz))-Xem*(nx*(up(emw.Ey)+down(emw.Ey))-ny*(up(emw.Ex)+down(emw.Ex)))))*emw.nmeshz)

      emw.smcd1.Jmsy = 0.5*((-emw.nmeshy*(Zmm*(up(emw.Hx)+down(emw.Hx))-Xem*(ny*(up(emw.Ez)+down(emw.Ez))-nz*(up(emw.Ey)+down(emw.Ey))))+emw.nmeshx*(Zmm*(up(emw.Hy)+down(emw.Hy))+Xem*(nx*(up(emw.Ez)+down(emw.Ez))-nz*(up(emw.Ex)+down(emw.Ex)))))*emw.nmeshx+(emw.nmeshz*(Zmm*(up(emw.Hy)+down(emw.Hy))+Xem*(nx*(up(emw.Ez)+down(emw.Ez))-nz*(up(emw.Ex)+down(emw.Ex))))-emw.nmeshy*(Zmm*(up(emw.Hz)+down(emw.Hz))-Xem*(nx*(up(emw.Ey)+down(emw.Ey))-ny*(up(emw.Ex)+down(emw.Ex)))))*emw.nmeshz)

      emw.smcd1.Jmsz = 0.5*((-emw.nmeshz*(Zmm*(up(emw.Hx)+down(emw.Hx))-Xem*(ny*(up(emw.Ez)+down(emw.Ez))-nz*(up(emw.Ey)+down(emw.Ey))))+emw.nmeshx*(Zmm*(up(emw.Hz)+down(emw.Hz))-Xem*(nx*(up(emw.Ey)+down(emw.Ey))-ny*(up(emw.Ex)+down(emw.Ex)))))*emw.nmeshx+(-emw.nmeshz*(Zmm*(up(emw.Hy)+down(emw.Hy))+Xem*(nx*(up(emw.Ez)+down(emw.Ez))-nz*(up(emw.Ex)+down(emw.Ex))))+emw.nmeshy*(Zmm*(up(emw.Hz)+down(emw.Hz))-Xem*(nx*(up(emw.Ey)+down(emw.Ey))-ny*(up(emw.Ex)+down(emw.Ex)))))*emw.nmeshy)

   Furthermore, the Equation View explicitly reveals how the tangential field traces (e.g., ``tEupx``) and test field traces (e.g., ``testtEupx``) used in the Nodal constraint above are constructed by rigorously subtracting the normal component from the raw field:

   .. code-block:: text

      emw.tEupx = up(emw.Ex)-(emw.nmeshx*up(emw.Ex)+emw.nmeshy*up(emw.Ey)+emw.nmeshz*up(emw.Ez))*emw.nmeshx
      emw.testtEupx = up(test(emw.Ex))-(emw.nmeshx*up(test(emw.Ex))+emw.nmeshy*up(test(emw.Ey))+emw.nmeshz*up(test(emw.Ez)))*emw.nmeshx

9. Complete continuous Case-A problem
-------------------------------------

Find

.. math::

   \mathbf E^-
   \in H(\mathrm{curl},\Omega^-),
   \qquad
   \mathbf E^+
   \in H(\mathrm{curl},\Omega^+),

such that

.. math::

   R_{\Omega^-}
   +
   R_{\Omega^+}
   +
   R_{J_s}
   +
   R_{\mathrm{outer\ boundaries}}
   =
   0

for all admissible test functions, together with

.. math::

   \boxed{
   \mathbf E_t^+
   -
   \mathbf E_t^-
   -
   \mathbf n_d\times
   \mathbf J_{ms}
   (\mathbf E_{\mathrm{av}},\mathbf H_{\mathrm{av}})
   =
   0
   }.

The essential structure is therefore

.. math::

   \boxed{
   \text{bulk Maxwell weak equations}
   +
   \text{electric-current surface weak term}
   +
   \text{magnetic-current jump constraint}.
   }

10. Minimal discrete interpretation
-----------------------------------

COMSOL performs the finite-element discretization internally, so this annex
does not reproduce the full edge-element assembly.

Write only

.. math::

   \mathbf E_h^\pm
   =
   \sum_j u_j^\pm\boldsymbol{\phi}_j^\pm,

where :math:`\boldsymbol{\phi}_j^\pm` are curl-conforming basis functions.
Collecting the degrees of freedom into :math:`\mathbf u`, the discrete problem
has the abstract form

.. math::

   \boxed{\mathbf R(\mathbf u)=0}

subject to

.. math::

   \boxed{\mathbf C(\mathbf u)=0}.

For the present linear GSTC model, both are linear in :math:`\mathbf u`.
COMSOL may eliminate constraint variables internally, so no particular final
matrix layout is assumed here.

.. note::

   A saddle-point or Lagrange-multiplier block matrix can be useful for
   analysis, but it should be regarded as a conceptual representation unless
   the final COMSOL solver matrix is explicitly inspected. Equation View
   reports the present magnetic-current constraint as ``Nodal``.

11. Why Case A can be difficult for an iterative solver
-------------------------------------------------------

Even without a GSTC, the frequency-domain Maxwell operator has the wave form

.. math::

   \mathbf K-k_0^2\mathbf M,

and is indefinite rather than positive definite.

The GSTC adds additional structure:

* two tangential traces are present at the sheet;
* :math:`\mathbf J_s` depends on averages of both
  :math:`\mathbf E^\pm` and :math:`\mathbf H^\pm`;
* :math:`\mathbf J_{ms}` imposes a field-dependent jump constraint;
* :math:`X_{em}` couples electric traces to magnetic, curl-derived traces;
* the interface operator is supported on a surface while the bulk wave
  operator is volumetric.

There is therefore no general structural guarantee that the assembled matrix
will be symmetric. Strong surface coupling can also produce poorly scaled or
nearly dependent combinations of average and jump modes, leading to small
singular values and poor conditioning. In a nonsymmetric system, nonnormality
can additionally make Krylov convergence difficult to predict from eigenvalues
alone.

.. caution::

   ``up()`` and ``down()`` are not by themselves the cause of poor
   conditioning. They expose the deeper fact that the zero-thickness GSTC
   requires a broken tangential trace and a strongly coupled interface
   operator. The numerical object of interest is the assembled,
   preconditioned system.

12. Relation to the planned thin-volume formulation
----------------------------------------------------

Case A should be retained as the reference zero-thickness formulation.

Case B will replace :math:`\Gamma` by an actual thin volume and move the
magnetoelectric physics into volume constitutive relations. The later
comparison should focus on electromagnetic agreement, iterative convergence,
sensitivity to GSTC strength, sensitivity to shell thickness, and
preconditioner behavior.

The purpose is not to assume in advance that the thin-volume formulation is
better, but to determine whether replacing the two-sided sheet constraint by a
volume constitutive operator produces a more favorable discrete system while
preserving the desired GSTC physics.
