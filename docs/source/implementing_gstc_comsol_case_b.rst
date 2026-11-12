Implementing GSTCs in COMSOL — Case B: Equivalent Thin Bianisotropic Volume
============================================================================


This annex develops the **Case B** representation of a prescribed GSTC
metasurface as an electrically thin volumetric bianisotropic shell.

The objective is to preserve the logical chain from the prescribed surface
susceptibilities to the equations actually solved by COMSOL:

.. math::

   \boxed{
   \text{GSTC susceptibilities}
   \rightarrow
   \text{surface polarizations}
   \rightarrow
   \text{thin-volume constitutive matrix}
   \rightarrow
   \text{strong Maxwell equations}
   \rightarrow
   \text{electric-field equation}
   \rightarrow
   \text{weak formulation}
   \rightarrow
   \text{COMSOL Equation View implementation}.
   }

This formulation should be compared with Case A, in which the zero-thickness
sheet is represented by

* a weak boundary contribution from the electric surface current
  :math:`\mathbf J_s`, and
* a tangential electric-field jump constraint associated with the magnetic
  surface current :math:`\mathbf J_{ms}`.

In Case B there is no GSTC boundary, no surface delta distribution, and no
``up()``/``down()`` sheet constraint. The metasurface physics is transferred
to the constitutive law of a thin shell.

The derivation below assumes the phasor convention

.. math::

   \mathbf F(\mathbf r,t)
   =
   \Re\left\{
   \mathbf F(\mathbf r)e^{i\omega t}
   \right\},

so that

.. math::

   \frac{\partial}{\partial t}
   \longleftrightarrow
   i\omega.

Thin-shell geometry and constitutive model
------------------------------------------

**Geometry**

Replace the zero-thickness surface :math:`\Gamma` by a shell
:math:`\Omega_s` of thickness :math:`\tau`:

.. math::

   \boxed{
   \Omega
   =
   \Omega_{\mathrm{in}}
   \cup
   \Omega_s
   \cup
   \Omega_{\mathrm{out}}.
   }

The shell is bounded by two ordinary material interfaces,

.. math::

   \Gamma_{\mathrm{in}}
   =
   \partial\Omega_{\mathrm{in}}
   \cap
   \partial\Omega_s,

and

.. math::

   \Gamma_{\mathrm{out}}
   =
   \partial\Omega_s
   \cap
   \partial\Omega_{\mathrm{out}}.

Unlike Case A, these interfaces do not carry explicit GSTC surface currents.

In the absence of impressed interface currents, the ordinary Maxwell interface
conditions apply. In particular, the tangential electric field is represented
by a single conforming trace across the material interfaces.

**Surface susceptibilities and volume polarization**

The original zero-thickness metasurface is characterized by four surface
susceptibility tensors,

.. math::

   \boxed{
   \boldsymbol{\chi}_{ee},
   \qquad
   \boldsymbol{\chi}_{em},
   \qquad
   \boldsymbol{\chi}_{me},
   \qquad
   \boldsymbol{\chi}_{mm}.
   }

The electric and magnetic surface polarizations can be written in the form

.. math::

   \mathbf P_s^e
   =
   \epsilon_0
   \boldsymbol{\chi}_{ee}\mathbf E_{\mathrm{av}}
   +
   \sqrt{\mu_0\epsilon_0}\,
   \boldsymbol{\chi}_{em}\mathbf H_{\mathrm{av}},

.. math::

   \mathbf P_s^m
   =
   \sqrt{\frac{\epsilon_0}{\mu_0}}\,
   \boldsymbol{\chi}_{me}\mathbf E_{\mathrm{av}}
   +
   \boldsymbol{\chi}_{mm}\mathbf H_{\mathrm{av}}.

Here ``av`` denotes the representative average field used in the
zero-thickness GSTC model.


The thin-volume model spreads the integrated surface response over a finite
thickness :math:`\tau`. The corresponding volume polarization densities are

.. math::

   \boxed{
   \mathbf P_v^e
   \approx
   \frac{\mathbf P_s^e}{\tau},
   \qquad
   \mathbf P_v^m
   \approx
   \frac{\mathbf P_s^m}{\tau}.
   }

For an electrically thin shell, the representative field inside the shell is
identified with the field used by the surface model to leading order.

Therefore,

.. math::

   \mathbf P_v^e
   =
   \epsilon_0
   \frac{\boldsymbol{\chi}_{ee}}{\tau}\mathbf E
   +
   \sqrt{\mu_0\epsilon_0}\,
   \frac{\boldsymbol{\chi}_{em}}{\tau}\mathbf H,

and

.. math::

   \mathbf P_v^m
   =
   \sqrt{\frac{\epsilon_0}{\mu_0}}\,
   \frac{\boldsymbol{\chi}_{me}}{\tau}\mathbf E
   +
   \frac{\boldsymbol{\chi}_{mm}}{\tau}\mathbf H.

.. note::

   **Why does every susceptibility appear divided by** :math:`\tau` **?**

   A surface polarization is an integrated response per unit area. If the
   same integrated response is distributed approximately uniformly through a
   layer of thickness :math:`\tau`, the corresponding volume density scales as

   .. math::

      \text{volume response}
      \sim
      \frac{\text{surface response}}{\tau}.

   Consequently, decreasing :math:`\tau` improves the thin-sheet
   interpretation but simultaneously increases the magnitude of the effective
   volumetric constitutive coefficients. This will later be important for
   conditioning and mesh design.

**Constitutive matrix**

The electric displacement is

.. math::

   \mathbf D
   =
   \epsilon_0\mathbf E
   +
   \mathbf P_v^e.

Substitution gives

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mathbf D
   =
   \epsilon_0
   \left(
   \mathbf I
   +
   \frac{\boldsymbol{\chi}_{ee}}{\tau}
   \right)\mathbf E
   +
   \sqrt{\mu_0\epsilon_0}\,
   \frac{\boldsymbol{\chi}_{em}}{\tau}\mathbf H.
   }
   \tag{B1}
   \end{equation}

Similarly,

.. math::

   \mathbf B
   =
   \mu_0
   \left(
   \mathbf H+\mathbf P_v^m
   \right),

and therefore

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mathbf B
   =
   \sqrt{\mu_0\epsilon_0}\,
   \frac{\boldsymbol{\chi}_{me}}{\tau}\mathbf E
   +
   \mu_0
   \left(
   \mathbf I
   +
   \frac{\boldsymbol{\chi}_{mm}}{\tau}
   \right)\mathbf H.
   }
   \tag{B2}
   \end{equation}

Equations (B1)--(B2) can be stacked as

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \begin{bmatrix}
   \mathbf D\\[1mm]
   \mathbf B
   \end{bmatrix}
   =
   \underbrace{
   \begin{bmatrix}
   \boldsymbol{\epsilon}_s
   &
   \boldsymbol{\xi}_s
   \\[1mm]
   \boldsymbol{\zeta}_s
   &
   \boldsymbol{\mu}_s
   \end{bmatrix}
   }_{\displaystyle \mathbf C_s}
   \begin{bmatrix}
   \mathbf E\\[1mm]
   \mathbf H
   \end{bmatrix},
   }
   \tag{B3}
   \end{equation}

with

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \boldsymbol{\epsilon}_s
   =
   \epsilon_0
   \left(
   \mathbf I+
   \frac{\boldsymbol{\chi}_{ee}}{\tau}
   \right),
   }
   \tag{B4}
   \end{equation}

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \boldsymbol{\xi}_s
   =
   \sqrt{\mu_0\epsilon_0}\,
   \frac{\boldsymbol{\chi}_{em}}{\tau},
   }
   \tag{B5}
   \end{equation}

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \boldsymbol{\zeta}_s
   =
   \sqrt{\mu_0\epsilon_0}\,
   \frac{\boldsymbol{\chi}_{me}}{\tau},
   }
   \tag{B6}
   \end{equation}

and

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \boldsymbol{\mu}_s
   =
   \mu_0
   \left(
   \mathbf I+
   \frac{\boldsymbol{\chi}_{mm}}{\tau}
   \right).
   }
   \tag{B7}
   \end{equation}

The tensors :math:`\boldsymbol{\xi}` (xi) and
:math:`\boldsymbol{\zeta}` (zeta) are the magnetoelectric cross-coupling
tensors.

.. note::

   **Why does the paper define** :math:`\boldsymbol{\alpha}` **as an inverse?**

   The thin-volume constitutive matrix above maps

   .. math::

      (\mathbf E,\mathbf H)
      \longmapsto
      (\mathbf D,\mathbf B).

   The volume-integral-equation formulation in the source paper instead
   introduces tensors :math:`\boldsymbol{\alpha}_1,\ldots,
   \boldsymbol{\alpha}_4` through the inverse map

   .. math::

      \begin{bmatrix}
      \mathbf E\\
      \mathbf H
      \end{bmatrix}
      =
      \begin{bmatrix}
      \boldsymbol{\alpha}_1&
      \boldsymbol{\alpha}_2\\
      \boldsymbol{\alpha}_3&
      \boldsymbol{\alpha}_4
      \end{bmatrix}
      \begin{bmatrix}
      \mathbf D\\
      \mathbf B
      \end{bmatrix}.

   Hence

   .. math::

      \boxed{
      \begin{bmatrix}
      \boldsymbol{\alpha}_1&
      \boldsymbol{\alpha}_2\\
      \boldsymbol{\alpha}_3&
      \boldsymbol{\alpha}_4
      \end{bmatrix}
      =
      \mathbf C_s^{-1}.
      }

   For COMSOL's electric-field formulation, the direct constitutive map
   :math:`(\mathbf E,\mathbf H)\mapsto(\mathbf D,\mathbf B)` is the more useful
   representation. There is no need to explicitly evaluate the
   :math:`\boldsymbol{\alpha}` blocks unless they are required for comparison
   with the VIE formulation.

**Piecewise material law**

Define a spatially piecewise constitutive matrix

.. math::

   \mathbf C(\mathbf r)
   =
   \begin{cases}
   \begin{bmatrix}
   \boldsymbol{\epsilon}_{\mathrm{in}}&0\\
   0&\boldsymbol{\mu}_{\mathrm{in}}
   \end{bmatrix},
   & \mathbf r\in\Omega_{\mathrm{in}},
   \\[5mm]
   \begin{bmatrix}
   \boldsymbol{\epsilon}_s&\boldsymbol{\xi}_s\\
   \boldsymbol{\zeta}_s&\boldsymbol{\mu}_s
   \end{bmatrix},
   & \mathbf r\in\Omega_s,
   \\[5mm]
   \begin{bmatrix}
   \boldsymbol{\epsilon}_{\mathrm{out}}&0\\
   0&\boldsymbol{\mu}_{\mathrm{out}}
   \end{bmatrix},
   & \mathbf r\in\Omega_{\mathrm{out}}.
   \end{cases}

Thus the cross-coupling tensors can be defined globally as piecewise domain variables:

.. math::

   \boldsymbol{\xi}(\mathbf r)
   =
   \begin{cases}
   \boldsymbol{\xi}_s, & \mathbf r\in\Omega_s,\\[2mm]
   0, & \text{otherwise},
   \end{cases}

and similarly for :math:`\boldsymbol{\zeta}(\mathbf r)`.

This suggests a convenient COMSOL strategy: define
:math:`\boldsymbol{\xi}(\mathbf r)` and
:math:`\boldsymbol{\zeta}(\mathbf r)` as piecewise domain variables that are
zero outside the shell. The constitutive expressions may then be written
globally and automatically reduce to the ordinary material law outside
:math:`\Omega_s`.

Maxwell equations and weak formulation
--------------------------------------

With no explicit sheet currents, Maxwell's frequency-domain curl equations are
ordinary volume equations everywhere:

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \nabla\times\mathbf E
   =
   -i\omega\mathbf B,
   }
   \tag{B8}
   \end{equation}

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \nabla\times\mathbf H
   =
   \boldsymbol{\sigma}\mathbf E
   +
   i\omega\mathbf D
   +
   \mathbf J_{\mathrm{imp}},
   }
   \tag{B9}
   \end{equation}

where :math:`\mathbf J_{\mathrm{imp}}` denotes an optional impressed volume
source.

Inside the bianisotropic shell,

.. math::

   \mathbf D
   =
   \boldsymbol{\epsilon}\mathbf E
   +
   \boldsymbol{\xi}\mathbf H,

.. math::

   \mathbf B
   =
   \boldsymbol{\zeta}\mathbf E
   +
   \boldsymbol{\mu}\mathbf H.

Therefore,

.. math::
   :no-wrap:

   \begin{equation}
   \nabla\times\mathbf E
   =
   -i\omega
   \left(
   \boldsymbol{\zeta}\mathbf E
   +
   \boldsymbol{\mu}\mathbf H
   \right),
   \tag{B10}
   \end{equation}

and

.. math::
   :no-wrap:

   \begin{equation}
   \nabla\times\mathbf H
   =
   \boldsymbol{\sigma}\mathbf E
   +
   i\omega
   \left(
   \boldsymbol{\epsilon}\mathbf E
   +
   \boldsymbol{\xi}\mathbf H
   \right)
   +
   \mathbf J_{\mathrm{imp}}.
   \tag{B11}
   \end{equation}

These equations form a first-order :math:`(\mathbf E,\mathbf H)` system.

**Eliminating the magnetic field**

From (B10),

.. math::

   \nabla\times\mathbf E
   +
   i\omega\boldsymbol{\zeta}\mathbf E
   =
   -i\omega
   \boldsymbol{\mu}\mathbf H.

Hence

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mathbf H
   =
   -\frac{1}{i\omega}
   \boldsymbol{\mu}^{-1}
   \left(
   \nabla\times\mathbf E
   +
   i\omega\boldsymbol{\zeta}\mathbf E
   \right).
   }
   \tag{B12}
   \end{equation}

Define

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mathbf Q
   =
   \boldsymbol{\mu}^{-1}
   \left(
   \nabla\times\mathbf E
   +
   i\omega\boldsymbol{\zeta}\mathbf E
   \right).
   }
   \tag{B13}
   \end{equation}

Then

.. math::

   \mathbf H
   =
   -\frac{\mathbf Q}{i\omega}.

Substitution into Ampère's law gives

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \left(
   \nabla\times
   -
   i\omega\boldsymbol{\xi}
   \right)
   \boldsymbol{\mu}^{-1}
   \left(
   \nabla\times
   +
   i\omega\boldsymbol{\zeta}
   \right)
   \mathbf E
   -
   \omega^2\boldsymbol{\epsilon}\mathbf E
   +
   i\omega\boldsymbol{\sigma}\mathbf E
   =
   i\omega\mathbf J_{\mathrm{imp}}.
   }
   \tag{B14}
   \end{equation}

For zero impressed source the right-hand side vanishes.

.. admonition:: Consistency check

   Setting

   .. math::

      \boldsymbol{\xi}
      =
      \boldsymbol{\zeta}
      =
      0

   reduces (B14) to

   .. math::

      \nabla\times
      \boldsymbol{\mu}^{-1}
      \nabla\times\mathbf E
      -
      \omega^2\boldsymbol{\epsilon}\mathbf E
      +
      i\omega\boldsymbol{\sigma}\mathbf E
      =
      0,

   which is the ordinary electric-field curl--curl equation.

**Expanded shell operator**

Expanding the bianisotropic part of (B14) gives

.. math::
   :no-wrap:

   \begin{equation}
   \begin{aligned}
   &
   \nabla\times
   \left(
   \boldsymbol{\mu}^{-1}
   \nabla\times\mathbf E
   \right)
   \\[1mm]
   &\quad
   +
   i\omega
   \nabla\times
   \left(
   \boldsymbol{\mu}^{-1}
   \boldsymbol{\zeta}\mathbf E
   \right)
   \\[1mm]
   &\quad
   -
   i\omega
   \boldsymbol{\xi}
   \boldsymbol{\mu}^{-1}
   \nabla\times\mathbf E
   \\[1mm]
   &\quad
   +
   \omega^2
   \boldsymbol{\xi}
   \boldsymbol{\mu}^{-1}
   \boldsymbol{\zeta}\mathbf E
   \\[1mm]
   &\quad
   -
   \omega^2
   \boldsymbol{\epsilon}\mathbf E
   +
   i\omega\boldsymbol{\sigma}\mathbf E
   =
   0.
   \end{aligned}
   \tag{B15}
   \end{equation}

The first and fifth terms are familiar from the ordinary Maxwell wave
operator. The middle three terms contain the magnetoelectric coupling.

**Weak formulation**

Let

.. math::

   \mathbf v
   \in
   H(\mathrm{curl},\Omega)

be a test field.

Unlike Case A, no broken :math:`H(\mathrm{curl})` space is required by the
metasurface model itself. The shell boundaries are ordinary material
interfaces.

Using :math:`\mathbf Q` from (B13), the source-free strong equation is

.. math::

   \nabla\times\mathbf Q
   -
   i\omega\boldsymbol{\xi}\mathbf Q
   -
   \omega^2\boldsymbol{\epsilon}\mathbf E
   +
   i\omega\boldsymbol{\sigma}\mathbf E
   =
   0.

Multiplying by :math:`\mathbf v` and integrating by parts in the curl term
gives the volume weak form

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \begin{aligned}
   R(\mathbf E;\mathbf v)
   =
   \int_\Omega
   \Big[
   &
   (\nabla\times\mathbf v)\cdot\mathbf Q
   -
   i\omega
   \mathbf v\cdot
   \boldsymbol{\xi}\mathbf Q
   \\[1mm]
   &
   -
   \omega^2
   \mathbf v\cdot
   \boldsymbol{\epsilon}\mathbf E
   +
   i\omega
   \mathbf v\cdot
   \boldsymbol{\sigma}\mathbf E
   \Big]dV
   +
   R_{\partial\Omega}.
   \end{aligned}
   }
   \tag{B16}
   \end{equation}

The boundary term is

.. math::

   R_{\partial\Omega}
   =
   \int_{\partial\Omega}
   \mathbf v\cdot
   \left(
   \mathbf n\times\mathbf Q
   \right)dS,

subject to the chosen outer boundary conditions.

Because the material coefficients are piecewise,

.. math::

   R
   =
   R_{\mathrm{in}}
   +
   R_s
   +
   R_{\mathrm{out}}
   +
   R_{\partial\Omega}.

Only :math:`R_s` contains nonzero
:math:`\boldsymbol{\xi}` and :math:`\boldsymbol{\zeta}`.

.. note::

   **There is still a weak formulation in Case B.**

   What disappears is the *special GSTC surface weak contribution and jump
   constraint*. COMSOL still assembles the volume Maxwell equations in weak
   form. The distinction is

   .. math::

      \text{Case A: volume weak form + sheet operator + jump constraint},

   versus

   .. math::

      \text{Case B: piecewise volume weak form only}.

COMSOL implementation
---------------------

For the ordinary RF/Wave Optics electric-field formulation, Equation View
exposes the following constitutive chain:

.. math::

   \mathbf B
   =
   -\frac{1}{i\omega}
   \nabla\times\mathbf E,

.. math::

   \mathbf H
   =
   \frac{1}{\mu_0}
   \boldsymbol{\mu}_r^{-1}\mathbf B,

.. math::

   \mathbf D
   =
   \epsilon_0
   \boldsymbol{\epsilon}_r\mathbf E,

.. math::

   \mathbf J_d
   =
   i\omega\mathbf D,

.. math::

   \mathbf J
   =
   \boldsymbol{\sigma}\mathbf E
   +
   \mathbf J_d.

The bulk weak expression found in Equation View has the compact form

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mu_0
   \frac{\partial\mathbf H}{\partial t}
   \cdot
   \nabla\times\mathbf v
   -
   i\omega\mu_0
   \mathbf J\cdot\mathbf v.
   }
   \tag{B17}
   \end{equation}

The key observation is that the **weak expression itself need not be changed**
if the constitutive variables entering it are changed consistently.

For a bianisotropic medium, Faraday's law still gives the total magnetic flux

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mathbf B
   =
   -\frac{1}{i\omega}
   \nabla\times\mathbf E.
   }
   \tag{B18}
   \end{equation}

Thus the native ``emw.Bx``, ``emw.By``, and ``emw.Bz`` definitions can remain
unchanged.

What changes is the constitutive recovery of :math:`\mathbf H` from
:math:`\mathbf B`:

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mathbf H
   =
   \boldsymbol{\mu}^{-1}
   \left(
   \mathbf B
   -
   \boldsymbol{\zeta}\mathbf E
   \right).
   }
   \tag{B19}
   \end{equation}

Similarly,

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mathbf D
   =
   \boldsymbol{\epsilon}\mathbf E
   +
   \boldsymbol{\xi}\mathbf H.
   }
   \tag{B20}
   \end{equation}

If these definitions are supplied consistently, then the existing COMSOL weak
expression (B17) produces the bianisotropic weak equation automatically.

**Constitutive overrides in Equation View**

This section records a **provisional implementation strategy**. It should be
tested first in a small verification model before modifying the production
sphere model.

**Native material tensors**

Use the normal material input for the diagonal constitutive blocks:

.. math::

   \boldsymbol{\epsilon}_{r,s}
   =
   \mathbf I
   +
   \frac{\boldsymbol{\chi}_{ee}}{\tau},

.. math::

   \boldsymbol{\mu}_{r,s}
   =
   \mathbf I
   +
   \frac{\boldsymbol{\chi}_{mm}}{\tau}.

The inner and outer domains retain their ordinary material tensors.

**Cross-coupling variables**

Define the nine Cartesian components of :math:`\boldsymbol{\xi}(\mathbf r)` and :math:`\boldsymbol{\zeta}(\mathbf r)` as domain variables.

They should be zero outside the shell, and inside the shell they take the explicit forms:

.. math::

   \boldsymbol{\xi}_s
   =
   \frac{1}{c_0}
   \frac{\boldsymbol{\chi}_{em}}{\tau},

.. math::

   \boldsymbol{\zeta}_s
   =
   \frac{1}{c_0}
   \frac{\boldsymbol{\chi}_{me}}{\tau}.

(where we used :math:`\sqrt{\mu_0\epsilon_0} = 1/c_0`).

Their SI unit is

.. math::

   [\xi]=[\zeta]=\mathrm{s/m}.

**Magnetic field variables**

The native expressions have the form

.. code-block:: text

   emw.Hx = (murinvxx*Bx + murinvxy*By + murinvxz*Bz)/mu0_const

and similarly for ``Hy`` and ``Hz``.

For Case B they must represent

.. math::

   \mathbf H
   =
   \frac{1}{\mu_0}
   \boldsymbol{\mu}_r^{-1}
   \left(
   \mathbf B-\boldsymbol{\zeta}\mathbf E
   \right).

For example, schematically,

.. code-block:: text

   Hx =
     (
       murinvxx*(Bx-(zetaxx*Ex+zetaxy*Ey+zetaxz*Ez))
      +murinvxy*(By-(zetayx*Ex+zetayy*Ey+zetayz*Ez))
      +murinvxz*(Bz-(zetazx*Ex+zetazy*Ey+zetazz*Ez))
     )/mu0_const

with analogous expressions for ``Hy`` and ``Hz``.

.. caution::

   The tensor-component indexing in the actual COMSOL expressions must be
   checked carefully. The schematic expression above assumes the usual
   matrix convention

   .. math::

      (\boldsymbol{\zeta}\mathbf E)_i
      =
      \sum_j\zeta_{ij}E_j.

   The final implementation should use one consistent component convention
   for all four constitutive tensors.

**Time derivative of the magnetic field**

The native ``emw.dHdtx``, ``emw.dHdty``, and ``emw.dHdtz`` definitions must
also be changed. Modifying ``emw.Hx`` alone is not sufficient because the
weak equation uses ``emw.dHdt*`` directly.

In a frequency-domain model the most transparent definition is

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \frac{\partial\mathbf H}{\partial t}
   \longleftrightarrow
   i\omega\mathbf H.
   }
   \tag{B21}
   \end{equation}

Thus, schematically,

.. code-block:: text

   emw.dHdtx = emw.iomega*emw.Hx
   emw.dHdty = emw.iomega*emw.Hy
   emw.dHdtz = emw.iomega*emw.Hz

is consistent with the modified constitutive relation.

Equivalently,

.. math::
   :no-wrap:

   \begin{equation}
   \frac{\partial\mathbf H}{\partial t}
   =
   \boldsymbol{\mu}^{-1}
   \left(
   -\nabla\times\mathbf E
   -
   i\omega\boldsymbol{\zeta}\mathbf E
   \right).
   \tag{B22}
   \end{equation}

**Electric polarization and displacement**

The native polarization is

.. math::

   \mathbf P
   =
   \epsilon_0
   \left(
   \boldsymbol{\epsilon}_r-\mathbf I
   \right)\mathbf E,

followed by

.. math::

   \mathbf D
   =
   \epsilon_0\mathbf E+\mathbf P.

A convenient bianisotropic modification is therefore

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mathbf P
   =
   \epsilon_0
   \left(
   \boldsymbol{\epsilon}_r-\mathbf I
   \right)\mathbf E
   +
   \boldsymbol{\xi}\mathbf H.
   }
   \tag{B23}
   \end{equation}

Then the existing definition

.. math::

   \mathbf D
   =
   \epsilon_0\mathbf E+\mathbf P

automatically becomes

.. math::

   \mathbf D
   =
   \boldsymbol{\epsilon}\mathbf E
   +
   \boldsymbol{\xi}\mathbf H.

This suggests modifying

.. code-block:: text

   emw.Px
   emw.Py
   emw.Pz

rather than replacing ``emw.Dx``, ``emw.Dy``, and ``emw.Dz`` directly.

If ``P`` is modified consistently, the existing variables

.. code-block:: text

   emw.Dx
   emw.Dy
   emw.Dz

and

.. code-block:: text

   emw.Jdx = emw.iomega*emw.Dx
   emw.Jdy = emw.iomega*emw.Dy
   emw.Jdz = emw.iomega*emw.Dz

can remain unchanged.

Likewise, the total current definitions

.. code-block:: text

   emw.Jx
   emw.Jy
   emw.Jz

can remain unchanged because the modified displacement current propagates into
them automatically.

.. important::

   The **minimal solver-relevant constitutive override** suggested by Equation
   View is therefore

   .. code-block:: text

      emw.Hx, emw.Hy, emw.Hz
      emw.dHdtx, emw.dHdty, emw.dHdtz
      emw.Px, emw.Py, emw.Pz

   together with native shell values of :math:`\epsilon_r` and :math:`\mu_r`
   and user-defined :math:`\xi` and :math:`\zeta` tensor components.

   The native definitions of ``B``, ``D``, ``Jd``, and ``J`` can then be left
   in place, provided they continue to reference the modified variables.

**Consistency with the existing weak expression**

Start with COMSOL's compact weak contribution

.. math::

   W_{\mathrm{COMSOL}}
   =
   \mu_0
   \frac{\partial\mathbf H}{\partial t}
   \cdot\nabla\times\mathbf v
   -
   i\omega\mu_0
   \mathbf J\cdot\mathbf v.

Using

.. math::

   \mathbf J
   =
   \boldsymbol{\sigma}\mathbf E
   +
   i\omega\mathbf D,

and

.. math::

   \mathbf D
   =
   \boldsymbol{\epsilon}\mathbf E
   +
   \boldsymbol{\xi}\mathbf H,

together with (B12), gives

.. math::

   W_{\mathrm{COMSOL}}
   =
   -\mu_0
   \Big[
   (\nabla\times\mathbf v)\cdot\mathbf Q
   -
   i\omega
   \mathbf v\cdot\boldsymbol{\xi}\mathbf Q
   -
   \omega^2
   \mathbf v\cdot\boldsymbol{\epsilon}\mathbf E
   +
   i\omega
   \mathbf v\cdot\boldsymbol{\sigma}\mathbf E
   \Big].

This is exactly :math:`-\mu_0` times the weak integrand in (B16).

Therefore, if the constitutive variables are modified consistently, the
existing COMSOL electric-field weak equation already has the correct
variational structure for the bianisotropic shell.

.. admonition:: Important implementation consequence

   There are therefore two possible implementation philosophies:

   1. **Constitutive override:** preserve COMSOL's existing weak equation and
      modify the constitutive variables that feed it.
   2. **Explicit Weak Contribution:** leave the native constitutive variables
      unchanged and add the missing :math:`\xi,\zeta` terms as domain weak
      contributions.

   The constitutive-override route is attractive because it follows COMSOL's
   own internal decomposition into :math:`\mathbf B`,
   :math:`\mathbf H`, :math:`\mathbf D`, and :math:`\mathbf J`.

   The explicit-weak-contribution route may be preferable if modifying locked
   Equation View expressions proves fragile across COMSOL versions.

   Both routes should represent the same continuous operator if implemented
   consistently.

**Alternative: explicit weak contribution**

If COMSOL handles the native

.. math::

   \boldsymbol{\epsilon},
   \qquad
   \boldsymbol{\mu}

blocks and only the magnetoelectric terms are added manually, then the
additional shell weak contribution relative to the ordinary material is

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \begin{aligned}
   W_{\mathrm{cross}}
   =
   {}&
   i\omega
   (\nabla\times\mathbf v)\cdot
   \boldsymbol{\mu}^{-1}
   \boldsymbol{\zeta}\mathbf E
   \\[1mm]
   &
   -
   i\omega
   \mathbf v\cdot
   \boldsymbol{\xi}
   \boldsymbol{\mu}^{-1}
   \nabla\times\mathbf E
   \\[1mm]
   &
   +
   \omega^2
   \mathbf v\cdot
   \boldsymbol{\xi}
   \boldsymbol{\mu}^{-1}
   \boldsymbol{\zeta}\mathbf E.
   \end{aligned}
   }
   \tag{B24}
   \end{equation}

This contribution is integrated only over :math:`\Omega_s`.

COMSOL's ``Weak Contribution`` feature can be used to add such a domain term
to an existing physics interface.

**Local spherical basis and Cartesian tensors**

For a spherical metasurface the prescribed GSTC susceptibilities are naturally
expressed in a local basis.

For a purely tangential sheet, a local tensor may be embedded as

.. math::

   \boldsymbol{\chi}_{\mathrm{loc}}
   =
   \begin{bmatrix}
   \chi_{\theta\theta}
   &
   \chi_{\theta\phi}
   &
   0
   \\
   \chi_{\phi\theta}
   &
   \chi_{\phi\phi}
   &
   0
   \\
   0&0&0
   \end{bmatrix}

in the ordered basis

.. math::

   (\hat{\boldsymbol\theta},
   \hat{\boldsymbol\phi},
   \hat{\mathbf r}).

Define

.. math::

   \hat{\mathbf r}
   =
   \begin{bmatrix}
   \sin\theta\cos\phi\\
   \sin\theta\sin\phi\\
   \cos\theta
   \end{bmatrix},

.. math::

   \hat{\boldsymbol\theta}
   =
   \begin{bmatrix}
   \cos\theta\cos\phi\\
   \cos\theta\sin\phi\\
   -\sin\theta
   \end{bmatrix},

and

.. math::

   \hat{\boldsymbol\phi}
   =
   \begin{bmatrix}
   -\sin\phi\\
   \cos\phi\\
   0
   \end{bmatrix}.

Let

.. math::

   \mathbf R
   =
   \begin{bmatrix}
   \hat{\boldsymbol\theta}
   &
   \hat{\boldsymbol\phi}
   &
   \hat{\mathbf r}
   \end{bmatrix}.

Then a local constitutive tensor is transformed to Cartesian coordinates by

.. math::
   :no-wrap:

   \begin{equation}
   \boxed{
   \mathbf T_{\mathrm{xyz}}
   =
   \mathbf R
   \mathbf T_{\mathrm{loc}}
   \mathbf R^T.
   }
   \tag{B25}
   \end{equation}

The same transformation must be applied consistently to

.. math::

   \boldsymbol{\epsilon},
   \qquad
   \boldsymbol{\mu},
   \qquad
   \boldsymbol{\xi},
   \qquad
   \boldsymbol{\zeta}.

.. caution::

   A constant tensor entered directly in the global Cartesian basis generally
   does **not** represent a constant tangential material response on a curved
   spherical shell. The tensor orientation must follow the local surface
   basis.

.. important:: Special simplification for the present reciprocal omega sheet

   The normal-based construction used for the spherical omega-sheet example
   is a special case of the transformation in (B25), not an alternative to
   it.  In this example, the direct electric and magnetic sheet blocks are
   isotropic within the tangent plane,

   .. math::

      \mathbf Y_{ee}=Y\mathbf I_t,
      \qquad
      \mathbf Z_{mm}=Z\mathbf I_t,

   where the embedded tangent-plane identity is

   .. math::

      \mathbf I_{t,\mathrm{loc}}
      =
      \begin{bmatrix}
      1&0&0\\
      0&1&0\\
      0&0&0
      \end{bmatrix}_{(\theta,\phi,r)}.

   The admittance and impedance coefficients in this direct current map are
   related to the polarization susceptibilities used in (B4) and (B7) by

   .. math::

      \boldsymbol{\chi}_{ee}
      =
      \frac{Y}{i\omega\epsilon_0}\mathbf I_t,
      \qquad
      \boldsymbol{\chi}_{mm}
      =
      \frac{Z}{i\omega\mu_0}\mathbf I_t.

   Because the two tangential eigenvalues are equal, applying (B25) gives the
   global Cartesian projector directly:

   .. math::

      \mathbf R\mathbf I_{t,\mathrm{loc}}\mathbf R^T
      =
      \mathbf I-\hat{\mathbf n}\hat{\mathbf n}^{T}
      =
      \mathbf P_t.

   Consequently, the relative material tensors in the thin shell can be
   written without explicitly constructing
   :math:`\hat{\boldsymbol\theta}` and
   :math:`\hat{\boldsymbol\phi}`:

   .. math::

      \boldsymbol{\epsilon}_{r,s}
      =
      \mathbf I
      +
      \frac{Y}{i\omega\epsilon_0\tau}\mathbf P_t,
      \qquad
      \boldsymbol{\mu}_{r,s}
      =
      \mathbf I
      +
      \frac{Z}{i\omega\mu_0\tau}\mathbf P_t.

   Only the added sheet response vanishes in the normal direction; the full
   relative material tensors retain the background radial eigenvalue one.
   The three-dimensional shell is therefore anisotropic, even though each
   diagonal sheet block is isotropic within the tangent plane.

   The scalar omega coupling has a similar special representation.  Its local
   tangential quarter-turn operator transforms as

   .. math::

      \mathbf R
      \begin{bmatrix}
      0&-1&0\\
      1&0&0\\
      0&0&0
      \end{bmatrix}
      \mathbf R^T\mathbf v
      =
      \hat{\mathbf n}\times\mathbf v.

   Thus both :math:`\mathbf P_t` and
   :math:`\hat{\mathbf n}\times(\cdot)` can be assembled from the global
   normal alone for this particular model.  This shortcut is **not general**.
   If the two tangential eigenvalues differ, if a preferred tangential axis is
   present, or if any susceptibility has general off-diagonal components, the
   complete local tensor and the transformation
   :math:`\mathbf R\mathbf T_{\mathrm{loc}}\mathbf R^T` must be used.

**Material interfaces**

Because the GSTC has been replaced by a volume material, the interfaces
between

.. math::

   \Omega_{\mathrm{in}}
   \leftrightarrow
   \Omega_s

and

.. math::

   \Omega_s
   \leftrightarrow
   \Omega_{\mathrm{out}}

are ordinary material interfaces.

No ``up()``/``down()`` GSTC average is required, and no magnetic-current slit
constraint is imposed.

The finite-element space can therefore remain globally conforming in
:math:`H(\mathrm{curl})`, subject to the ordinary material-interface behavior
already built into the electromagnetic-wave interface.

Solver setup, postprocessing, and verification
----------------------------------------------

Case B removes several structural complications present in Case A:

* no duplicated GSTC tangential trace;
* no field-dependent magnetic-current jump constraint;
* no codimension-one electric-current GSTC operator;
* no explicit ``up()``/``down()`` averaging in the constitutive model.

This makes the problem more similar to a conventional piecewise-material
Maxwell problem.

However, Case B introduces different numerical difficulties.

From (B4)--(B7),

.. math::

   \boldsymbol{\epsilon}_s-\epsilon_0\mathbf I,
   \qquad
   \boldsymbol{\mu}_s-\mu_0\mathbf I,
   \qquad
   \boldsymbol{\xi}_s,
   \qquad
   \boldsymbol{\zeta}_s

all scale as

.. math::

   \frac{1}{\tau}.

Therefore,

.. math::

   \tau\downarrow
   \quad\Longrightarrow\quad
   \text{stronger coefficient contrast}.

A small :math:`\tau` also creates a geometrically thin domain that may require
high-aspect-ratio elements or a swept mesh.

Consequently, Case B is **not assumed a priori to be better conditioned**.
The useful numerical question is whether there exists a range of thicknesses
for which

.. math::

   \text{thin-volume error is small}

while

.. math::

   \text{the preconditioned Krylov system is easier than Case A}.

The final comparison should therefore consider at least

* field error relative to Case A;
* :math:`\tau/\lambda`;
* mesh aspect ratio and number of elements through the shell;
* Krylov iteration count;
* residual history;
* sensitivity to magnetoelectric coupling strength; and
* memory and wall-clock cost.

**Symmetry and nonnormality**

The Case-B electric-field operator contains the two first-order cross terms

.. math::

   i\omega
   \nabla\times
   \left(
   \boldsymbol{\mu}^{-1}
   \boldsymbol{\zeta}\mathbf E
   \right)

and

.. math::

   -i\omega
   \boldsymbol{\xi}
   \boldsymbol{\mu}^{-1}
   \nabla\times\mathbf E.

The discrete matrix is therefore not automatically symmetric. Its symmetry
properties depend on the material reciprocity relations, tensor symmetries,
boundary conditions, and the complex variational convention.

Thus Case B may still be nonsymmetric and nonnormal even though the GSTC slit
constraint has disappeared.

The comparison with Case A should therefore focus on the **assembled and
preconditioned operator**, rather than assuming that a volume formulation is
symmetric.

**Postprocessing**

Overriding the solver-relevant constitutive variables does not automatically
make every native COMSOL postprocessing quantity valid for a general
bianisotropic material.

In particular, quantities derived under the assumption

.. math::

   \mathbf D=\boldsymbol{\epsilon}\mathbf E,
   \qquad
   \mathbf B=\boldsymbol{\mu}\mathbf H

should be reviewed before use.

Examples include native expressions for

* electric and magnetic stored energy;
* material loss density;
* effective wave number and skin depth;
* constitutive energy derivatives in dispersive media; and
* some stress-tensor or material diagnostic quantities.

The Poynting vector

.. math::

   \frac12
   \Re\{\mathbf E\times\mathbf H^*\}

remains meaningful provided the modified :math:`\mathbf H` is used.

.. caution::

   Solver correctness and postprocessing correctness are separate questions.
   A model may solve the intended bianisotropic PDE while some built-in
   diagnostic variables still reflect the original non-bianisotropic
   constitutive assumptions.

**Verification sequence**

Before applying Case B to the full spherical metasurface, verify the
implementation incrementally.

1. **Native recovery test**

   Set

   .. math::

      \boldsymbol{\xi}
      =
      \boldsymbol{\zeta}
      =
      0.

   The modified model should reproduce the original COMSOL anisotropic
   :math:`\epsilon_r,\mu_r` solution.

2. **Single cross-coupling tests**

   Test :math:`\boldsymbol{\xi}\neq0`,
   :math:`\boldsymbol{\zeta}=0`, and then the reverse case.

3. **Homogeneous slab test**

   Use a simple planar bianisotropic layer for which reflection/transmission
   can be checked independently.

4. **Thickness convergence**

   Keep the prescribed surface susceptibilities fixed while varying
   :math:`\tau`. Recompute the volume tensors through (B4)--(B7) for every
   thickness.

5. **Case-A comparison**

   Compare fields, power flow, and scattering quantities against the original
   zero-thickness GSTC implementation.

6. **Solver comparison**

   Compare BiCGStab/GMRES residual histories and the behavior of the chosen
   multigrid or other preconditioner.

**Implementation recap**

The present analysis suggests the following Case-B architecture:

.. math::

   \boxed{
   \boldsymbol{\chi}_{ab}
   \rightarrow
   \left(
   \boldsymbol{\epsilon}_s,
   \boldsymbol{\xi}_s,
   \boldsymbol{\zeta}_s,
   \boldsymbol{\mu}_s
   \right)
   \rightarrow
   \left(
   \mathbf H,
   \frac{\partial\mathbf H}{\partial t},
   \mathbf P,
   \mathbf D,
   \mathbf J
   \right)
   \rightarrow
   \text{existing COMSOL weak Maxwell equation}.
   }

The most promising Equation View variables to examine or override are

.. code-block:: text

   emw.Hx, emw.Hy, emw.Hz
   emw.dHdtx, emw.dHdty, emw.dHdtz
   emw.Px, emw.Py, emw.Pz

while retaining the native relations for

.. code-block:: text

   emw.Bx, emw.By, emw.Bz
   emw.Dx, emw.Dy, emw.Dz
   emw.Jdx, emw.Jdy, emw.Jdz
   emw.Jx, emw.Jy, emw.Jz

provided their dependencies remain as shown in Equation View.

This should be treated as a hypothesis to verify in a minimal model before it
is used in the production simulation.

Mapping SIE surface parameters to COMSOL
----------------------------------------

This section establishes the direct mathematical mapping between the physical
GSTC surface parameters solved by the SIE-BEM formulation and the equivalent
thin-volume constitutive tensors solved by COMSOL in Case B.

**Surface-current conventions**

In the time-harmonic convention :math:`e^{j\omega t}` (:math:`\partial/\partial t \leftrightarrow j\omega`),
the tangential GSTCs neglecting normal surface polarizations (:math:`P_n = 0`) are

.. math::

   \hat{\mathbf{n}} \times \Delta \mathbf{H}
   =
   j\omega \mathbf{P}_s^e,
   \qquad
   \hat{\mathbf{n}} \times \Delta \mathbf{E}
   =
   -j\omega\mu_0 \mathbf{P}_s^m.

The electric and magnetic surface polarization densities are defined in terms
of surface susceptibilities :math:`\boldsymbol{\chi}_{ab}` (units of length :math:`[\mathrm{m}]`) as

.. math::

   \mathbf{P}_s^e
   =
   \epsilon_0 \boldsymbol{\chi}_{ee}\mathbf{E}_{\mathrm{av},t}
   +
   \frac{1}{c_0}\boldsymbol{\chi}_{em}\mathbf{H}_{\mathrm{av},t},

.. math::

   \mathbf{P}_s^m
   =
   \frac{1}{\eta_0}\boldsymbol{\chi}_{me}\mathbf{E}_{\mathrm{av},t}
   +
   \boldsymbol{\chi}_{mm}\mathbf{H}_{\mathrm{av},t},

where :math:`\eta_0 = \sqrt{\mu_0/\epsilon_0}` and :math:`c_0 = 1/\sqrt{\mu_0\epsilon_0}`.
Substituting the polarizations into the field jumps gives

.. math::

   \hat{\mathbf{n}} \times \Delta \mathbf{H}
   =
   j\omega\epsilon_0\boldsymbol{\chi}_{ee}\mathbf{E}_{\mathrm{av},t}
   +
   \frac{j\omega}{c_0}\boldsymbol{\chi}_{em}\mathbf{H}_{\mathrm{av},t},

.. math::

   \hat{\mathbf{n}} \times \Delta \mathbf{E}
   =
   -\frac{j\omega\mu_0}{\eta_0}\boldsymbol{\chi}_{me}\mathbf{E}_{\mathrm{av},t}
   -
   j\omega\mu_0\boldsymbol{\chi}_{mm}\mathbf{H}_{\mathrm{av},t}.

In the boundary element formulation, exterior region 2 and interior region 1 define
the surface currents as

.. math::

   \hat{\mathbf{n}} \times \mathbf{H}_2 = \mathbf{J}_2,
   \qquad
   \hat{\mathbf{n}} \times \mathbf{E}_2 = -\mathbf{M}_2,

.. math::

   \hat{\mathbf{n}} \times \mathbf{H}_1 = -\mathbf{J}_1,
   \qquad
   \hat{\mathbf{n}} \times \mathbf{E}_1 = \mathbf{M}_1.

The total field jumps across the sheet are therefore

.. math::

   \hat{\mathbf{n}} \times \Delta\mathbf{H}
   =
   \hat{\mathbf{n}} \times \mathbf{H}_2 - \hat{\mathbf{n}} \times \mathbf{H}_1
   =
   \mathbf{J}_2 + \mathbf{J}_1
   =
   \mathbf{J}_s,

.. math::

   \hat{\mathbf{n}} \times \Delta\mathbf{E}
   =
   \hat{\mathbf{n}} \times \mathbf{E}_2 - \hat{\mathbf{n}} \times \mathbf{E}_1
   =
   -\mathbf{M}_2 - \mathbf{M}_1
   =
   -\mathbf{M}_s.

Notice that the negative sign in :math:`\hat{\mathbf{n}} \times \Delta\mathbf{E} = -\mathbf{M}_s`
cancels the negative sign in the magnetic polarization jump, directly yielding:

.. math::

   \mathbf{M}_s
   =
   \frac{j\omega\mu_0}{\eta_0}\boldsymbol{\chi}_{me}\mathbf{E}_{\mathrm{av},t}
   +
   j\omega\mu_0\boldsymbol{\chi}_{mm}\mathbf{H}_{\mathrm{av},t}.

**Surface parameters and susceptibilities**

Defining the surface parameter block map

.. math::

   \begin{bmatrix}
   \mathbf{J}_s \\[2pt]
   \mathbf{M}_s
   \end{bmatrix}
   =
   \begin{bmatrix}
   \mathbf{Y} & \boldsymbol{\chi} \\
   \boldsymbol{\gamma} & \mathbf{Z}
   \end{bmatrix}
   \begin{bmatrix}
   \mathbf{E}_{\mathrm{av},t} \\[2pt]
   \mathbf{H}_{\mathrm{av},t}
   \end{bmatrix},

matching terms gives the exact correspondence (with :math:`k_0 = \omega/c_0`):

.. math::

   \boxed{
   \begin{aligned}
   \mathbf{Y} &= j\omega\epsilon_0\boldsymbol{\chi}_{ee}
   &\iff\quad
   \boldsymbol{\chi}_{ee} &= \frac{\mathbf{Y}}{j\omega\epsilon_0},
   \\[2mm]
   \mathbf{Z} &= j\omega\mu_0\boldsymbol{\chi}_{mm}
   &\iff\quad
   \boldsymbol{\chi}_{mm} &= \frac{\mathbf{Z}}{j\omega\mu_0},
   \\[2mm]
   \boldsymbol{\chi} &= \frac{j\omega}{c_0}\boldsymbol{\chi}_{em} = j k_0 \boldsymbol{\chi}_{em}
   &\iff\quad
   \boldsymbol{\chi}_{em} &= \frac{\boldsymbol{\chi}}{j k_0},
   \\[2mm]
   \boldsymbol{\gamma} &= \frac{j\omega\mu_0}{\eta_0}\boldsymbol{\chi}_{me} = j k_0 \boldsymbol{\chi}_{me}
   &\iff\quad
   \boldsymbol{\chi}_{me} &= \frac{\boldsymbol{\gamma}}{j k_0}.
   \end{aligned}
   }

.. note:: Dual use of the symbol :math:`\boldsymbol{\chi}` (Abuse of Notation)

   In the metasurface and GSTC literature, the Greek letter :math:`\boldsymbol{\chi}` is
   traditionally used in two distinct contexts:

   1. **Surface parameter block** :math:`\boldsymbol{\chi}`: The dimensionless :math:`[1]`
      off-diagonal block in the fields-to-currents map :math:`[\mathbf{Y}, \boldsymbol{\chi}; \boldsymbol{\gamma}, \mathbf{Z}]`,
      representing the magneto-electric coupling from :math:`\mathbf{H}_{\mathrm{av},t}` to :math:`\mathbf{J}_s`.
   2. **Surface polarization susceptibilities** :math:`\boldsymbol{\chi}_{ab}`: The four
      constitutive tensors :math:`\boldsymbol{\chi}_{ee}, \boldsymbol{\chi}_{em}, \boldsymbol{\chi}_{me}, \boldsymbol{\chi}_{mm}`,
      which have physical SI dimensions of length :math:`[\mathrm{m}]` and relate the fields to
      surface polarizations :math:`\mathbf{P}_s^e, \mathbf{P}_s^m`.

   To keep equations explicit and unambiguous, the subscripted notation :math:`\boldsymbol{\chi}_{ab}`
   always denotes the length-dimensioned susceptibility, while the standalone symbol :math:`\boldsymbol{\chi}`
   (without subscript) denotes the dimensionless surface parameter block:

   .. math::

      \boldsymbol{\chi} = j k_0 \boldsymbol{\chi}_{em} \iff \boldsymbol{\chi}_{em} = \frac{\boldsymbol{\chi}}{j k_0}.

Reciprocity in the surface parameters enforces

.. math::

   \mathbf{Y} = \mathbf{Y}^T,
   \qquad
   \mathbf{Z} = \mathbf{Z}^T,
   \qquad
   \boldsymbol{\chi} = -\boldsymbol{\gamma}^T
   \quad\iff\quad
   \boldsymbol{\chi}_{em} = -\boldsymbol{\chi}_{me}^T.

**Thin-shell constitutive tensors**

Substituting these expressions into the thin-volume constitutive relations
(:math:`\mathbf{D} = \epsilon_0\mathbf{E} + \mathbf{P}_s^e/\tau`,
:math:`\mathbf{B} = \mu_0\mathbf{H} + \mu_0\mathbf{P}_s^m/\tau`) produces a
uniform :math:`1/(j\omega\tau)` scaling across all four volumetric tensors:

.. math::

   \boxed{
   \begin{aligned}
   \boldsymbol{\epsilon}_{r,s}
   &=
   \mathbf{I} + \frac{\boldsymbol{\chi}_{ee}}{\tau}
   =
   \mathbf{I} + \frac{\mathbf{Y}}{j\omega\epsilon_0\tau},
   \\[2mm]
   \boldsymbol{\mu}_{r,s}
   &=
   \mathbf{I} + \frac{\boldsymbol{\chi}_{mm}}{\tau}
   =
   \mathbf{I} + \frac{\mathbf{Z}}{j\omega\mu_0\tau},
   \\[2mm]
   \boldsymbol{\xi}_s
   &=
   \frac{1}{c_0}\frac{\boldsymbol{\chi}_{em}}{\tau}
   =
   \frac{\boldsymbol{\chi}}{j\omega\tau},
   \\[2mm]
   \boldsymbol{\zeta}_s
   &=
   \frac{1}{c_0}\frac{\boldsymbol{\chi}_{me}}{\tau}
   =
   \frac{\boldsymbol{\gamma}}{j\omega\tau}.
   \end{aligned}
   }

For the 10 dimensionless normalized parameters :math:`\boldsymbol{\theta} = (y_1, y_2, y_3, c_1, c_2, c_3, c_4, z_1, z_2, z_3)`
expressed in the local spherical tangent basis :math:`(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}})`,
the local :math:`2\times 2` shell blocks are

.. math::

   \mathbf{Y}_{\mathrm{loc}}
   =
   \frac{j}{\eta_0}
   \begin{bmatrix} y_1 & y_3 \\ y_3 & y_2 \end{bmatrix}
   \quad\Longrightarrow\quad
   \Delta\boldsymbol{\epsilon}_{r,s,\mathrm{loc}}
   =
   \frac{1}{k_0\tau}
   \begin{bmatrix} y_1 & y_3 \\ y_3 & y_2 \end{bmatrix}
   \quad\text{(purely real)},

.. math::

   \mathbf{Z}_{\mathrm{loc}}
   =
   j\eta_0
   \begin{bmatrix} z_1 & z_3 \\ z_3 & z_2 \end{bmatrix}
   \quad\Longrightarrow\quad
   \Delta\boldsymbol{\mu}_{r,s,\mathrm{loc}}
   =
   \frac{1}{k_0\tau}
   \begin{bmatrix} z_1 & z_3 \\ z_3 & z_2 \end{bmatrix}
   \quad\text{(purely real)},

.. math::

   \boldsymbol{\chi}_{\mathrm{loc}}
   =
   \begin{bmatrix} c_1 & c_2 \\ c_3 & c_4 \end{bmatrix}
   \quad\Longrightarrow\quad
   \boldsymbol{\xi}_{s,\mathrm{loc}}
   =
   \frac{1}{j\omega\tau}
   \begin{bmatrix} c_1 & c_2 \\ c_3 & c_4 \end{bmatrix},

.. math::

   \boldsymbol{\gamma}_{\mathrm{loc}}
   =
   -\boldsymbol{\chi}_{\mathrm{loc}}^T
   \quad\Longrightarrow\quad
   \boldsymbol{\zeta}_{s,\mathrm{loc}}
   =
   -\boldsymbol{\xi}_{s,\mathrm{loc}}^T
   =
   \frac{1}{j\omega\tau}
   \begin{bmatrix} -c_1 & -c_3 \\ -c_2 & -c_4 \end{bmatrix}.

At each spatial point :math:`(x, y, z)` within the shell :math:`\Omega_s`, these
local :math:`2\times 2` blocks are rotated into 3D Cartesian coordinates using the
local basis projection :math:`\mathbf{R} = [\hat{\boldsymbol{\theta}} \quad \hat{\boldsymbol{\phi}}]`:

.. math::

   \mathbf{T}_{3\mathrm{D}}
   =
   \mathbf{R}\,\mathbf{T}_{\mathrm{loc}}\,\mathbf{R}^T,
   \qquad
   (\mathbf{T}_{3\mathrm{D}})_{ij}
   =
   u_i u_j T_{00} + u_i v_j T_{01} + v_i u_j T_{10} + v_i v_j T_{11},

with :math:`\mathbf{u} = \hat{\boldsymbol{\theta}}` and :math:`\mathbf{v} = \hat{\boldsymbol{\phi}}`.
These 3D tensors feed directly into the wave equation feature (``weq2``) and its
Equation View overrides for :math:`\mathbf{H}`, :math:`\partial\mathbf{H}/\partial t`,
and :math:`\mathbf{P}`.

**Source excitation and dipole normalization**

In the COMSOL RF Module implementation, point dipole excitations are modeled
using the ``ElectricPointDipole`` feature (``epd1``). In COMSOL, the dipole source
magnitude is specified by the parameter ``normpI``, representing the physical
electric current moment :math:`I\Delta z` in :math:`\mathrm{A\,m}`:

.. code-block:: text

   pI = 1.0[A*m]
   epd1.normpI = pI
   epd1.enpI = [0, 0, 1]

Because COMSOL natively solves the frequency-domain vector Helmholtz equation in
physical SI units, its field outputs (:math:`\mathbf{E}`) are directly in
:math:`\mathrm{V/m}`.

To compare these fields against integral equation solvers (SIE, PMCHWT, SVWF)
that use the normalized effective moment :math:`p_{\mathrm{eff}} = 1.0`, the
integral equation fields are converted to physical units using:

.. math::

   \text{scale} = \frac{-j\,\eta_{\mathrm{int}}}{4\pi k_{\mathrm{int}} \cdot p_{\mathrm{eff}}}.

This factor analytically accounts for both the :math:`1/(4\pi\varepsilon)` factor
omitted in the numerical Green's function source model and the :math:`I = j\omega q`
relation connecting current elements to charge dipoles.

**References and implementation notes**

The constitutive conversion follows the thin-volume relation used in the
bianisotropic-sheet formulation discussed in the source paper, in which the
:math:`\boldsymbol{\alpha}` matrix is the inverse of the direct
:math:`(\mathbf E,\mathbf H)\mapsto(\mathbf D,\mathbf B)` constitutive matrix.

COMSOL's Equation View exposes the variables, weak expressions, shape
functions, and constraints contributed by each physics feature. Editing a
predefined Equation View expression changes the governing equations solved by
COMSOL. A domain ``Weak Contribution`` is an alternative way to add the
cross-coupling terms without replacing the native weak expression.

Relevant COMSOL documentation:

* ``Equation View``:
  https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_modeling.19.067.html

* ``Weak Contribution``:
  https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_ref_equationbased.32.073.html
