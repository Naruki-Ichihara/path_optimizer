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
* :func:`orientation_blend` — a two-phase blend between an isotropic polymer
  matrix and an *orientation-averaged* fibre lamina, parameterised by the
  libertas orientation triple ``(x1, x2, x3)``.  This is the fibre-path
  material: the design controls both *how much* fibre and *which way* it runs.

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
    "Lamina",
    "Polymer",
    "isotropic",
    "orientation_blend",
]


# ── Material descriptions ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Lamina:
    """Orthotropic fibre-reinforced lamina; the fibre is the local 1-axis.

    Defaults are a typical CFRP lamina.  ``alpha_fibre`` is the small (often
    slightly negative) longitudinal CTE and ``alpha_trans`` the much larger
    transverse one — that asymmetry is what makes a stack of differently
    oriented layers warp when cooled.
    """

    E1: float = 140.0e9
    E2: float = 10.0e9
    G12: float = 5.0e9
    nu12: float = 0.30
    G13: float = 5.0e9
    G23: float = 3.0e9
    alpha_fibre: float = -0.5e-6
    alpha_trans: float = 30.0e-6
    thickness: float = 0.5e-3


@dataclass(frozen=True)
class Polymer:
    """Isotropic polymer matrix filling the low-density regions.

    Defaults are a typical thermoplastic: soft, with a large isotropic CTE.
    """

    E: float = 2.0e9
    nu: float = 0.40
    alpha: float = 70.0e-6

    @property
    def G(self) -> float:
        """Shear modulus ``E / (2(1+ν))``."""
        return self.E / (2.0 * (1.0 + self.nu))


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


def orientation_blend(lamina: Lamina, polymer: Polymer, sgn_beta: float = 10.0):
    """Fibre/matrix blend — ``layer(rho_p, x1, x2, x3)``.

    ``(x1, x2, x3)`` are the libertas orientation parameters; they map to a 2D
    second-order orientation tensor ``a₂`` via
    :func:`feax.mechanics.orientation.orientation_tensor_2d`, which is then
    closed to ``a₄`` (quadratic closure) and used to orientation-average the
    lamina stiffness.  ``a₂`` carries *both* the fibre direction and the degree
    of alignment, so a single continuous field spans "unidirectional along d"
    through "randomly oriented".

    Linear rule of mixtures between the two phases (no SIMP void — the
    low-density phase is polymer, not vacuum):

    .. code-block:: text

        C_eff  = (1 − ρ̃)·C_poly       + ρ̃·C_fibre(a₂, a₄)
        α_eff  = (1 − ρ̃)·α_poly·I     + ρ̃·α_fibre(a₂)
        G_s,eff = (1 − ρ̃)·G_poly·I    + ρ̃·diag(G13, G23)

    ``sgn_beta`` sharpens the smooth sign function inside
    ``orientation_tensor_2d``; larger is closer to a hard switch but harder to
    optimise through.
    """
    C_poly = isotropic_in_plane_stiffness(polymer.E, polymer.nu)
    G_poly = polymer.G * np.eye(2)
    G_fibre = np.diag(np.array([lamina.G13, lamina.G23]))
    alpha_poly = polymer.alpha * np.eye(2)

    def layer(rho_p, x1, x2, x3):
        a2, _, _ = orientation_tensor_2d(x1, x2, x3, sgn_beta=sgn_beta)
        a4 = quadratic_closure(a2)
        C_fibre = orientation_averaged_stiffness(
            a2, a4, E1=lamina.E1, E2=lamina.E2, G12=lamina.G12, nu12=lamina.nu12,
        )
        alpha_fibre = thermal_expansion_from_orientation(
            a2, lamina.alpha_fibre, lamina.alpha_trans,
        )
        void = 1.0 - rho_p
        return (
            void * C_poly + rho_p * C_fibre,
            void * alpha_poly + rho_p * alpha_fibre,
            void * G_poly + rho_p * G_fibre,
        )

    return layer
