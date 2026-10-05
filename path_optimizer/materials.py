"""Quad-point constitutive models shared by :mod:`~path_optimizer.plane` and
:mod:`~path_optimizer.shell`.

Both problem modules take a **layer constitutive**: a pure function of the
design scalars at one quadrature point,

.. code-block:: text

    layer(*design) -> (C_in, alpha, G_s)

    C_in  : (2, 2, 2, 2)  plane-stress in-plane stiffness, laminate axes
    alpha : (2, 2)        thermal-expansion tensor, laminate axes
    G_s   : (2, 2)        transverse-shear stiffness (ignored by plane stress)

Returning the same triple everywhere means one constitutive can drive a plane
problem and a shell layer unchanged.  ``alpha`` is only consulted when the
problem was given a ``delta_t``; ``G_s`` only by the shell.

Two are provided:

* :func:`isotropic` — one material scaled by density.  The classic SIMP
  topology-optimisation material.
* :func:`orientation_blend` — a two-phase blend between an isotropic matrix
  and an *orientation-averaged* fibre lamina, parameterised by the libertas
  orientation triple ``(x1, x2, x3)``.  This is the fibre-path material: the
  design controls both *how much* fibre and *which way* it runs.

**Every elastic constant is required.**  :class:`Lamina` and :class:`Polymer`
carry no default moduli — they are property bags, not a materials database.  A
stiffness nobody chose decides the optimum silently, and an optimiser will
happily build a part out of it.  Fields that *are* optional are optional
because the physics does not apply (thermal expansion without a ``delta_t``,
transverse shear without a shell), not because a plausible number was picked
for you.

Neither applies the SIMP exponent itself.  Projection belongs in
:meth:`path_optimizer.Pipeline.transform`, so the objective and every
constraint see the identical projected density — pass the already-projected
``rho_p`` here and the blend stays a plain linear rule of mixtures.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as np
from feax.mechanics.orientation import (
    orientation_averaged_stiffness,
    orientation_tensor_2d,
    quadratic_closure,
)
from feax.mechanics.shell import (
    isotropic_in_plane_stiffness,
    thermal_expansion_from_orientation,
)

__all__ = [
    "DEFAULT_SGN_BETA",
    "KNOCKDOWN_NOTE",
    "Lamina",
    "Polymer",
    "isotropic",
    "orientation_blend",
]


#: Default sharpness for the off-diagonal projection ``smooth_sgn(x3)`` inside
#: :func:`feax.mechanics.orientation.orientation_tensor_2d`.
#:
#: 2.0, not the 10.0 feax itself defaults to.  The projection is Nomura's
#: Eq. (34), ``a_ij = sqrt(a_ii a_jj) (2 R(q) - 1)``, and ``smooth_sgn(x3, b)``
#: equals that relaxed Heaviside with the paper's ``beta = 2b`` -- verified to
#: machine precision -- so 10 here is the paper's 20: a near-hard sign applied
#: from the first iteration, with no relaxation.
#:
#: It saturates: ``tanh(10 x3)/tanh(10)`` is 0.99996 by ``x3 = 0.3``, and the
#: compliance gradient measured 1.8e-01 at ``x3 = 0`` against 3.7e-05 at
#: ``x3 = 0.5``.  Since ``x3 = 0`` is the *only* point where ``a12 = 0`` -- and
#: there the director can only be 0 or 90 degrees, whatever ``(x1, x2)`` say --
#: each node commits to a branch while the orientation signal is still nil, then
#: locks.  Measured on the MBB example at 200x50, 150 iterations:
#:
#: ===========  =============  ====================================
#: sgn_beta     compliance     against the best *uniform* direction
#: ===========  =============  ====================================
#: 10.0         0.9153         2.1x **worse**
#: 2.0          0.1746         2.6x better
#: ===========  =============  ====================================
#:
#: At 10.0 the fibre came out transverse to the major principal strain at 74% of
#: solid nodes (median 86 degrees); at 2.0 the median is 12 degrees.
#:
#: Raising it is reasonable *late* in an optimisation, which is what the paper's
#: relaxation means; fixing it high from the start is not.
DEFAULT_SGN_BETA = 2.0


# ── Material descriptions ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Lamina:
    """Orthotropic fibre-reinforced lamina; the fibre is the local 1-axis.

    **The four elastic constants are required.**  There is no default lamina:
    a plausible-looking number nobody chose is worse than an error, because it
    silently decides the answer.  Everything optional below is optional because
    it is *not applicable* to every problem, never because a value was guessed
    for you.

    Parameters
    ----------
    E1, E2 : float
        Longitudinal (fibre) and transverse Young's moduli, Pa.
    G12 : float
        In-plane shear modulus, Pa.
    nu12 : float
        Major Poisson's ratio.  ``nu21`` follows from reciprocity, so it is not
        a separate input.
    G13, G23 : float, optional
        Transverse-shear moduli, Pa.  Read only by
        :class:`~path_optimizer.shell.LaminatedShell`; plane stress never looks
        at them, so they may be left unset there.  Setting one and not the
        other is an error.
    alpha_1, alpha_2 : float
        Longitudinal and transverse coefficients of thermal expansion, 1/K.
        Zero unless the problem was built with a ``delta_t``.  The longitudinal
        one is often small and negative while the transverse one is an order of
        magnitude larger; that asymmetry is what warps a stack of differently
        oriented layers on cooling.
    thickness : float, optional
        Ply thickness, m — geometry, not a material property.  Here only so a
        shell stack can be described in one object; plane stress ignores it.
    """

    E1: float
    E2: float
    G12: float
    nu12: float
    G13: float | None = None
    G23: float | None = None
    alpha_1: float = 0.0
    alpha_2: float = 0.0
    thickness: float | None = None

    def __post_init__(self):
        if (self.G13 is None) != (self.G23 is None):
            raise ValueError(
                "give both G13 and G23 or neither — a lamina with one "
                "transverse-shear modulus is not a material")

    def transverse_shear(self) -> tuple[float, float]:
        """``(G13, G23)``, or an error naming what is missing."""
        if self.G13 is None:
            raise ValueError(
                "this Lamina has no transverse-shear moduli; set G13 and G23 "
                "to use it in a shell")
        return self.G13, self.G23


@dataclass(frozen=True)
class Polymer:
    """Isotropic matrix filling the low-density regions.

    ``E`` and ``nu`` are required, for the same reason as :class:`Lamina`.

    Parameters
    ----------
    E : float
        Young's modulus, Pa.
    nu : float
        Poisson's ratio.
    alpha : float
        Coefficient of thermal expansion, 1/K.  Zero unless the problem was
        built with a ``delta_t``.
    shear_modulus : float, optional
        Overrides the isotropic ``E / (2(1+ν))``.  For a material that is not
        quite isotropic in shear — a filled or printed matrix, say — set it
        rather than back-solving ``nu`` to get the shear you wanted and
        corrupting the in-plane stiffness in the process.
    """

    E: float
    nu: float
    alpha: float = 0.0
    shear_modulus: float | None = None

    @property
    def G(self) -> float:
        """Shear modulus: ``shear_modulus`` if given, else ``E / (2(1+ν))``."""
        if self.shear_modulus is not None:
            return self.shear_modulus
        return self.E / (2.0 * (1.0 + self.nu))


#: Why the transverse modulus is worth knocking down, and continuing back up.
#:
#: The orientation design variables decide a *direction*, and the only thing
#: that makes one direction better than another is the contrast between ``E1``
#: and ``E2``.  When that contrast is small the compliance barely notices the
#: direction, the gradient that should choose it is drowned by everything else,
#: and the optimiser settles for an orientation field worse than simply laying
#: every fibre the same way -- which is in its own feasible set.
#:
#: Scaling ``E2`` by ``k`` in ``(0, 1]`` exaggerates the contrast: at ``k=0.02``
#: the direction is unmistakable.  Continuing ``k`` back to 1 ends on the real
#: material, so unlike optimising an exaggerated material outright, the answer
#: is an answer to the actual problem.
#:
#: Measured on the MBB example at ``E1/E2 = 2.17`` (200x50, the degenerate
#: default orientation start, every design judged on the real material):
#:
#: ======================================  ============  =====================
#: run                                     compliance    vs the best *uniform*
#: ======================================  ============  =====================
#: no continuation                         1.585e+03 J   1.12x **worse**
#: k = 0.02, 0.1, 0.3, 1.0                 1.019e+03 J   1.43x better
#: ======================================  ============  =====================
#:
#: It costs iterations -- that run spent 4 stages -- and one measurement is one
#: measurement: repeat runs of this problem land about 10% apart.
#:
#: ``k`` is not a design variable.  It rides in as a uniform fifth volume var,
#: so the pipeline passes it per iteration and :class:`path_optimizer.Continuation`
#: can schedule it:
#:
#: .. code-block:: python
#:
#:     layer = materials.orientation_blend(lamina, sgn_beta=10.0,
#:                                         transverse_knockdown=True)
#:     ...
#:     def objective(self, design, knockdown=1.0, **params):
#:         sol = self.solver(fe.TracedParams(volume_vars=(
#:             design["simp"], design["x1"], design["x2"], design["x3"],
#:             jnp.full(n, knockdown))))
#:     ...
#:     po.run(..., continuations={"knockdown": po.Continuation(0.02, 1.0,
#:                                                             update_every=40,
#:                                                             step=4.0,
#:                                                             mode="multiply")})
KNOCKDOWN_NOTE = __doc__


# ── Constitutive factories ───────────────────────────────────────────────────

def isotropic(
    E: float,
    nu: float,
    *,
    alpha: float = 0.0,
    e_min: float = 1e-9,
    G_transverse: float | None = None,
):
    """Density-scaled isotropic plane-stress material — ``layer(rho_p)``.

    .. code-block:: text

        s      = e_min + rho_p · (1 − e_min)
        C_eff  = s · C(E, ν)
        G_s    = s · G · I
        α_eff  = α · I                       (a material property, not scaled)

    ``e_min`` keeps the void stiffness away from exactly zero so the tangent
    stays invertible; with SIMP applied upstream (``rho_p = rho ** p``) this is
    the standard modified-SIMP interpolation.

    ``alpha`` is only used by problems built with a ``delta_t``.  It is *not*
    scaled by density: the thermal resultant ``C : α`` already picks up the
    stiffness scaling, which is the physically correct place for it.
    """
    if not 0.0 <= e_min < 1.0:
        raise ValueError(f"e_min must be in [0, 1), got {e_min}")
    C0 = isotropic_in_plane_stiffness(E, nu)
    G0 = (E / (2.0 * (1.0 + nu)) if G_transverse is None else G_transverse) * np.eye(2)
    alpha_tensor = alpha * np.eye(2)

    def layer(rho_p):
        s = e_min + rho_p * (1.0 - e_min)
        return s * C0, alpha_tensor, s * G0

    return layer


def orientation_blend(lamina: Lamina, polymer: Polymer | None = None,
                      sgn_beta: float = DEFAULT_SGN_BETA, *,
                      e_min: float = 1e-9,
                      transverse_knockdown: bool = False):
    """Fibre blended with a matrix, or with void — ``layer(rho_p, x1, x2, x3)``.

    ``(x1, x2, x3)`` are the libertas orientation parameters; they map to a 2D
    second-order orientation tensor ``a₂`` via
    :func:`feax.mechanics.orientation.orientation_tensor_2d`, which is then
    closed to ``a₄`` (quadratic closure) and used to orientation-average the
    lamina stiffness.  ``a₂`` carries *both* the fibre direction and the degree
    of alignment, so a single continuous field spans "unidirectional along d"
    through "randomly oriented".

    Linear rule of mixtures between the two phases:

    .. code-block:: text

        C_eff   = (1 − ρ̃)·C_poly    + ρ̃·C_fibre(a₂, a₄)
        α_eff   = (1 − ρ̃)·α_poly·I  + ρ̃·α_fibre(a₂)
        G_s,eff = (1 − ρ̃)·G_poly·I  + ρ̃·diag(G13, G23)

    **``polymer=None`` makes the low-density phase void instead.**  Then
    ``ρ̃ = 0`` carries no load, as in ordinary topology optimisation, rather than
    leaving a matrix behind.  That is the same formula with ``C_poly`` set to
    ``e_min · C_fibre``, which collapses to modified SIMP:

    .. code-block:: text

        (1 − ρ̃)·e_min·C_f + ρ̃·C_f  =  (e_min + ρ̃·(1 − e_min))·C_f

    so there is one rule of mixtures here, not two.  ``e_min`` keeps the void
    stiffness off exactly zero so the tangent stays invertible; it scales the
    *oriented* tensor, so a void element keeps its direction and simply stops
    contributing.  ``α`` is not scaled — the thermal resultant ``C : α`` already
    picks the scaling up, which is the physically correct place for it.

    Which one to use is a modelling choice, not a detail.  With a matrix, the
    design says *how much fibre to add to a part that exists everywhere* — the
    printed-composite reading, and the void is as stiff as the polymer.  With
    ``polymer=None`` the design says *where the part is at all*.  Mixing them up
    is easy to spot: a 2 GPa matrix against a 140 GPa fibre leaves the "void"
    at 1/59 of the solid stiffness, not the 1/10⁹ that SIMP assumes.

    ``sgn_beta`` sharpens the smooth sign function inside
    ``orientation_tensor_2d``; larger is closer to a hard switch but harder to
    optimise through.

    ``transverse_knockdown=True`` adds a fifth input, ``layer(rho_p, x1, x2, x3,
    k)``, where ``k`` in ``(0, 1]`` scales ``E2``.  It exists to be *continued*:
    see :data:`KNOCKDOWN_NOTE`.

    .. note::

       The layer contract always returns a transverse shear, but a lamina for a
       plane-stress problem has no reason to carry one.  If ``lamina.G13`` is
       unset, ``G_s`` is filled with ``G12`` — harmless, since plane stress
       never reads it, and wrong if you then hand the same constitutive to a
       shell.  Set ``G13`` and ``G23`` for shell work; :meth:`Lamina.transverse_shear`
       raises rather than guessing.
    """
    if polymer is None and not 0.0 <= e_min < 1.0:
        raise ValueError(f"e_min must be in [0, 1), got {e_min}")
    # Only the shell reads G_s.  A lamina with no transverse shear is legal for
    # plane stress, so fall back to the in-plane modulus here rather than
    # refusing to build -- the shell path calls transverse_shear() and gets a
    # proper error if the numbers really are missing.
    g13, g23 = (lamina.G13, lamina.G23) if lamina.G13 is not None else (
        lamina.G12, lamina.G12)
    G_fibre = np.diag(np.array([g13, g23]))
    if polymer is not None:
        C_poly = isotropic_in_plane_stiffness(polymer.E, polymer.nu)
        G_poly = polymer.G * np.eye(2)
        alpha_poly = polymer.alpha * np.eye(2)

    def layer(rho_p, x1, x2, x3, knockdown=1.0):
        a2, _, _ = orientation_tensor_2d(x1, x2, x3, sgn_beta=sgn_beta)
        a4 = quadratic_closure(a2)
        C_fibre = orientation_averaged_stiffness(
            a2, a4, E1=lamina.E1, E2=knockdown * lamina.E2,
            G12=lamina.G12, nu12=lamina.nu12,
        )
        alpha_fibre = thermal_expansion_from_orientation(
            a2, lamina.alpha_1, lamina.alpha_2,
        )
        if polymer is None:
            # The same mixture with C_poly = e_min * C_fibre, i.e. modified
            # SIMP.  alpha is a property of whatever material is there, so it is
            # not scaled; at rho = 0 there is nothing to expand anyway.
            s = e_min + rho_p * (1.0 - e_min)
            return s * C_fibre, alpha_fibre, s * G_fibre
        void = 1.0 - rho_p
        return (
            void * C_poly + rho_p * C_fibre,
            void * alpha_poly + rho_p * alpha_fibre,
            void * G_poly + rho_p * G_fibre,
        )

    if transverse_knockdown:
        return layer

    # Keep the four-argument contract when nobody asked for the fifth.
    def fixed(rho_p, x1, x2, x3):
        return layer(rho_p, x1, x2, x3, 1.0)

    return fixed
