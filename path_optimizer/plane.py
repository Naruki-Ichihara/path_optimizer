"""Plane-stress elasticity with a design-dependent quad-point material.

The workhorse for 2D topology optimisation.  A single displacement variable
``u = (u, v)`` on a 2D mesh; the stiffness at each quadrature point comes from
a :mod:`~path_optimizer.materials` layer constitutive evaluated on the design
fields, which the solver receives as ``TracedParams`` volume vars:

.. code-block:: text

    σ = t · C(design) : (ε − α ΔT),      ε = sym(∇u)

``t`` is the out-of-plane thickness — feax integrates over the 2D domain, so
multiplying the stress by ``t`` turns it into the correct force resultant.
``ΔT`` is optional: with ``delta_t=None`` (the default) no thermal term is
assembled at all, and the constitutive's ``alpha`` is never touched.

Design-field order matters — the weak form receives the volume vars positionally
and forwards them straight to the constitutive.  Build the matching
:class:`~path_optimizer.DesignSpace` with :func:`design_space` rather than by
hand, and the two cannot drift apart.

Example
-------

.. code-block:: python

    import path_optimizer as po
    from path_optimizer import materials, plane

    layer = materials.isotropic(E=1.0, nu=0.3)
    problem = plane.make_plane_stress(
        mesh, layer, location_fns=(tip,),
        surface_load_fns=[plane.uniform_traction([0.0, -1.0])],
    )
    space = plane.design_space()            # one field: "rho"
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import feax as fe
import jax.numpy as np

from path_optimizer.optimizer import DesignField, DesignSpace

__all__ = [
    "PlaneStress",
    "make_plane_stress",
    "uniform_traction",
    "fixed_bc",
    "design_space",
    "ORIENTED_FIELDS",
    "DENSITY_FIELDS",
]


#: Design-field names for a density-only (isotropic SIMP) plane problem.
DENSITY_FIELDS = ("rho",)

#: Design-field names for a density + fibre-orientation plane problem, in the
#: order :func:`path_optimizer.materials.orientation_blend` expects them.
ORIENTED_FIELDS = ("rho", "x1", "x2", "x3")


# ── Problem ──────────────────────────────────────────────────────────────────

class PlaneStress(fe.Problem):
    """Plane-stress elasticity whose material is a function of the design.

    Built with ``additional_info=(constitutive, thickness, delta_t,
    surface_load_fns)``; use :func:`make_plane_stress` rather than constructing
    it directly.
    """

    def custom_init(self, constitutive, thickness, delta_t, surface_load_fns):
        self.constitutive = constitutive
        self.thickness = float(thickness)
        self.delta_t = delta_t
        self.surface_load_fns = tuple(surface_load_fns)

    def get_tensor_map(self):
        constitutive = self.constitutive
        thickness = self.thickness
        delta_t = self.delta_t

        def stress(u_grad, *design):
            C, alpha, _ = constitutive(*design)
            eps = 0.5 * (u_grad + u_grad.T)
            if delta_t is not None:
                # Thermal eigenstrain: the elastic strain is the total strain
                # less the free expansion α·ΔT.
                eps = eps - alpha * delta_t
            return thickness * np.einsum("ijkl,kl->ij", C, eps)

        return stress

    def get_surface_maps(self):
        return list(self.surface_load_fns)


def make_plane_stress(
    mesh,
    constitutive: Callable,
    *,
    thickness: float = 1.0,
    delta_t: float | None = None,
    ele_type: str = "QUAD4",
    location_fns: Iterable[Callable] = (),
    surface_load_fns: Sequence[Callable] | None = None,
    traction: Sequence[float] | None = None,
) -> PlaneStress:
    """Construct a :class:`PlaneStress` problem on a 2D mesh.

    Parameters
    ----------
    mesh : feax.Mesh
        Any 2D feax mesh.
    constitutive : callable
        ``(*design) -> (C_in, alpha, G_s)`` — see
        :mod:`path_optimizer.materials`.  Its positional arguments must line up
        with the ``TracedParams`` volume vars you pass to the solver, i.e. with
        the design-field order of :func:`design_space`.
    thickness : float
        Out-of-plane thickness ``t``.
    delta_t : float, optional
        Uniform temperature change.  ``None`` (default) assembles **no** thermal
        term — the problem is then purely mechanical and the constitutive's
        ``alpha`` is irrelevant.
    ele_type : str
        Element type (``"QUAD4"``, ``"TRI3"``, ...), matching ``mesh``.
    location_fns : iterable of callables
        Boundary predicates, one per loaded region.
    surface_load_fns : sequence of callables, optional
        One surface map per region, signature ``(u, x, *design) -> (2,)``.  If
        omitted, every region gets :func:`uniform_traction` with ``traction``.
    traction : sequence of 2 floats, optional
        Convenience uniform traction used when ``surface_load_fns`` is omitted.
        Sign convention as in :func:`uniform_traction` — it is the residual-side
        value, i.e. the negative of the physical traction.

    Returns
    -------
    PlaneStress
    """
    location_fns = tuple(location_fns)
    if surface_load_fns is None:
        t = uniform_traction(traction if traction is not None else (0.0, 0.0))
        surface_load_fns = tuple(t for _ in location_fns)
    else:
        surface_load_fns = tuple(surface_load_fns)
        if len(surface_load_fns) != len(location_fns):
            raise ValueError(
                f"surface_load_fns ({len(surface_load_fns)}) must match "
                f"location_fns ({len(location_fns)})"
            )
    return PlaneStress(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type=ele_type,
        location_fns=list(location_fns) or None,
        additional_info=(constitutive, thickness, delta_t, surface_load_fns),
    )


# ── Loads and boundary conditions ────────────────────────────────────────────

def uniform_traction(t: Sequence[float]) -> Callable:
    """A surface map applying a constant traction.

    **Sign convention.**  feax accumulates the surface term into the *residual*
    as ``+∫ t·v dΓ``, so the value a surface map returns is the negative of the
    physical traction — a physical pull of ``+q`` along ``x`` is written
    ``uniform_traction([-q, 0.0])``, and ``uniform_traction([0.0, 1.0])`` pulls
    the boundary *downward*.  This is feax's convention, shared by every
    hand-written surface map, so the helper does not silently flip it.
    """
    t_arr = np.asarray(t, dtype=float)
    if t_arr.shape != (2,):
        raise ValueError(f"traction must have 2 components, got {t_arr.shape}")

    def surface_map(u, x, *design):
        return t_arr

    return surface_map


def fixed_bc(problem, location, component: str | int = "all") -> fe.DirichletBC:
    """Pin displacement on ``location``; ``component="all"`` fixes both u and v."""
    return fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=location, component=component, value=0.0,
                           variable_index=0),
    ]).create_bc(problem)


# ── Design space ─────────────────────────────────────────────────────────────

def design_space(
    *,
    oriented: bool = False,
    rho_bounds: tuple[float, float] = (0.0, 1.0),
    rho_init: float = 0.5,
    rho_filter_frac: float | None = 0.05,
    rho_filter_radius: float | None = None,
    theta_filter_frac: float | None = 0.05,
    theta_filter_radius: float | None = None,
    ori_tol: float = 1e-2,
    x3_bounds: tuple[float, float] = (-1.0, 1.0),
) -> DesignSpace:
    """The :class:`~path_optimizer.DesignSpace` matching a plane constitutive.

    ``oriented=False`` (default) yields the single field ``rho`` for
    :func:`path_optimizer.materials.isotropic`.  ``oriented=True`` yields
    ``(rho, x1, x2, x3)`` in the order
    :func:`path_optimizer.materials.orientation_blend` expects.

    The orientation fields start at ``+1 - ori_tol``, which ``box_to_triangle``
    maps to ``a11 = a22 = 0.5``: a full fibre population with **no preferred
    direction**.  That is what a neutral default should be — it chooses no
    direction, it just starts with fibre to point.

    Not ``-1 + ori_tol``, the other corner, which maps to ``a11 = a22 = 0.005``
    — an orientation tensor that is essentially empty.  ``a12`` is built as
    ``sqrt(a11 a22) · smooth_sgn(x3)``, so that corner scales everything ``x3``
    can do by 0.005, and ``x3`` is the *only* variable that can turn an
    isotropic ``a11 = a22`` into a direction.  Measured on the MBB example
    there, ``|d obj/dx3|`` is 69x smaller than ``|d obj/dx1|``; from
    ``+1 - ori_tol`` it is 2.5x larger.  The optimiser pushes ``x1`` and ``x2``
    together instead, which only grows the trace, and the tensor stays
    isotropic: ``ud_penalty`` sat at exactly 1.0 for every iteration of a
    200-iteration run, and the x-tolerance read the stalled design as converged
    after 4.

    Starting points measured on that example, every one judged by the
    compliance it reached (lower is better):

    ===================================  ==========  =====================
    start                                compliance  vs the best *uniform*
    ===================================  ==========  =====================
    ``+1 - ori_tol``  (a2 = I/2)         0.523       1.25x better
    uniform 0 / 45 / 135 degrees         0.516-0.534 better
    uniform 90 degrees                   0.684       about equal
    ``-1 + ori_tol``  (a2 ~ 0)           0.729       1.12x **worse**
    ===================================  ==========  =====================

    The three direction-neutral starts land within 1.04x of each other, so the
    answer does not hinge on which one; the old default was the outlier.
    """
    # An explicit radius wins over the fractional default, rather than tripping
    # DesignField's "give one or the other" guard.
    if rho_filter_radius is not None:
        rho_filter_frac = None
    if theta_filter_radius is not None:
        theta_filter_frac = None

    fields = [DesignField("rho", lower=rho_bounds[0], upper=rho_bounds[1],
                          init=rho_init, filter_frac=rho_filter_frac,
                          filter_radius=rho_filter_radius)]
    if oriented:
        start = 1.0 - ori_tol
        for name, init, (lo, hi) in (("x1", start, (-1.0, 1.0)),
                                     ("x2", start, (-1.0, 1.0)),
                                     ("x3", 0.0, x3_bounds)):
            fields.append(DesignField(name, lower=lo, upper=hi, init=init,
                                      filter_frac=theta_filter_frac,
                                      filter_radius=theta_filter_radius))
    return DesignSpace(fields)
