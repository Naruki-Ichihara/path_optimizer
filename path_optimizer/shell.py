"""Laminated FSDT (Mindlin) shell with a design-dependent quad-point laminate.

Two-variable plate bending: ``var0 = (u, v, w)``, ``var1 = (θx, θy)``.  feax 0.8
already ships :class:`feax.mechanics.shell.MindlinPlate`, but its ``A``/``D``/
``G_s`` are *constants* — one uniform plate.  Here every layer's material is
rebuilt at each quadrature point from the design fields, so density and fibre
orientation can vary per layer and per point, which is what makes the laminate
optimisable:

.. code-block:: text

    per layer k:  (C_k, α_k, G_k) = layer(*design_k)          # materials.py
    laminate:     A, B, D, G_s    = Σ_k CLT integrals over z  # feax 0.8
    resultants:   N = A:ε + B:κ − N_T,  M = B:ε + D:κ − M_T,  Q = G_s·γ

Because the layers are designed independently, the coupling ``B`` is generally
non-zero — an asymmetric laminate.  That is the whole point for bending control:
it is what turns an in-plane design into out-of-plane curvature.

**Thermal loading is opt-in.**  With ``delta_t=None`` (the default) no thermal
resultant is assembled and the bending is driven purely by mechanical load.
Give a ``delta_t`` and the cooling eigenstrain of the CTE-asymmetric stack is
added, which is the 4D-printing / self-morphing regime.  Note that with neither
a ``delta_t`` nor a mechanical load the solution is identically zero and any
displacement-based objective has a vanishing gradient — at least one driver is
required.

Design-field order matters — the weak form slices the volume vars into
``n_layers`` consecutive chunks and forwards each chunk positionally to
``layer``.  Build the matching :class:`~path_optimizer.DesignSpace` with
:func:`design_space` and the two cannot drift apart.

Example
-------

.. code-block:: python

    from path_optimizer import materials, shell

    lamina, polymer = materials.Lamina(), materials.Polymer()
    layer = materials.orientation_blend(lamina, polymer)

    problem = shell.make_laminated_shell(
        mesh, layer, vars_per_layer=4,
        thicknesses=(lamina.thickness,) * 2,     # bilayer
        delta_t=-150.0,                          # omit for pure bending
        location_fns=(free_edge,),
        surface_load_fns=[shell.uniform_transverse_load(5.0)],
    )
    space = shell.design_space(n_layers=2)       # 8 fields, matching order
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import feax as fe
import jax.numpy as np
from feax.mechanics.shell import (
    laminate_stiffness,
    laminate_thermal_loads,
    mindlin_weak_form,
)

from path_optimizer.optimizer import DesignField, DesignSpace

__all__ = [
    "LaminatedShell",
    "make_laminated_shell",
    "uniform_transverse_load",
    "clamp_bc",
    "design_space",
    "layer_field_names",
]


# ── Problem ──────────────────────────────────────────────────────────────────

class LaminatedShell(fe.Problem):
    """FSDT laminate whose per-layer material is a function of the design.

    Built with ``additional_info=(layer_fn, vars_per_layer, thicknesses,
    delta_t, dT_grad, kappa_s, nonlinear, surface_load_fns)``; use
    :func:`make_laminated_shell` rather than constructing it directly.
    """

    def custom_init(self, layer_fn, vars_per_layer, thicknesses, delta_t,
                    dT_grad, kappa_s, nonlinear, surface_load_fns):
        self.layer_fn = layer_fn
        self.vars_per_layer = int(vars_per_layer)
        self.thicknesses = np.asarray(thicknesses, dtype=float)
        self.n_layers = len(thicknesses)
        # The layer materials already carry their own orientation (via a₂), so
        # there is no additional stacking-sequence rotation on top.
        self.zero_thetas = np.zeros(self.n_layers)
        self.delta_t = delta_t
        self.dT_grad = dT_grad
        self.kappa_s = kappa_s
        self.nonlinear = nonlinear
        self.surface_load_fns = tuple(surface_load_fns)

    def get_weak_form(self):
        layer_fn = self.layer_fn
        vpl = self.vars_per_layer
        n_layers = self.n_layers
        thicks = self.thicknesses
        zero_thetas = self.zero_thetas
        delta_t = self.delta_t
        dT_grad = self.dT_grad
        kappa_s = self.kappa_s
        nonlinear = self.nonlinear

        def weak_form(vals, grads, x, *design):
            # design is (n_layers × vars_per_layer,) scalars, layer-major.
            per_layer = [
                layer_fn(*design[k * vpl:(k + 1) * vpl]) for k in range(n_layers)
            ]
            C_layers = np.stack([c for c, _, _ in per_layer])
            alpha_layers = np.stack([a for _, a, _ in per_layer])
            G_layers = np.stack([g for _, _, g in per_layer])

            A, B, D, G_s = laminate_stiffness(
                C_layers, G_layers, zero_thetas, thicks, kappa_s=kappa_s,
            )
            if delta_t is None:
                N_T = M_T = None
            else:
                N_T, M_T = laminate_thermal_loads(
                    C_layers, alpha_layers, zero_thetas, thicks,
                    dT_avg=delta_t, dT_grad=dT_grad,
                )

            # Reuse feax 0.8's tested FSDT weak form with this point's laminate.
            return mindlin_weak_form(
                A, D, G_s, B=B, N_T=N_T, M_T=M_T, nonlinear=nonlinear,
            )(vals, grads, x)

        return weak_form

    def get_surface_weak_forms(self):
        return list(self.surface_load_fns)


def make_laminated_shell(
    mesh,
    layer_fn: Callable,
    thicknesses: Sequence[float],
    *,
    vars_per_layer: int = 4,
    delta_t: float | None = None,
    dT_grad: float = 0.0,
    kappa_s: float = 5.0 / 6.0,
    nonlinear: str = "linear",
    ele_type: str = "QUAD4",
    location_fns: Iterable[Callable] = (),
    surface_load_fns: Sequence[Callable] | None = None,
    load_mag: float = 0.0,
) -> LaminatedShell:
    """Construct a :class:`LaminatedShell` on a 2D mesh.

    Parameters
    ----------
    mesh : feax.Mesh
        A single 2D mesh in the ``z = 0`` plane; used for both variables.
    layer_fn : callable
        ``(*design_k) -> (C_in, alpha, G_s)`` for one layer at one quadrature
        point — see :mod:`path_optimizer.materials`.
    thicknesses : sequence of float
        Layer thicknesses, ordered **bottom → top**.  Its length sets the number
        of layers; the midplane sits at ``z = 0``.
    vars_per_layer : int
        How many design scalars ``layer_fn`` consumes.  ``4`` for
        :func:`path_optimizer.materials.orientation_blend` (ρ, x1, x2, x3),
        ``1`` for :func:`path_optimizer.materials.isotropic`.
    delta_t : float, optional
        Through-thickness-average temperature change.  ``None`` (default)
        assembles no thermal resultant at all.
    dT_grad : float
        Linear temperature gradient in ``ΔT(z) = delta_t + z · dT_grad``.
    kappa_s : float
        Transverse-shear correction factor (5/6 for a homogeneous plate).
    nonlinear : {"linear", "von_karman"}
        Strain measure.  ``"von_karman"`` makes the residual cubic in ``w`` and
        needs a Newton solve — pass ``linear=False`` to ``feax.create_solver``
        (``path_optimizer.make_linear_solver`` will *not* do).
    ele_type : str
        Element type for both variables.
    location_fns : iterable of callables
        Boundary predicates, one per loaded region.
    surface_load_fns : sequence of callables, optional
        One weak form per region, signature
        ``(vals, x, *design) -> [t_uvw(3,), t_theta(2,)]``.  If omitted, every
        region gets :func:`uniform_transverse_load` with ``load_mag``.
    load_mag : float
        Convenience uniform transverse load used when ``surface_load_fns`` is
        omitted.

    Returns
    -------
    LaminatedShell
    """
    thicknesses = tuple(float(t) for t in thicknesses)
    if not thicknesses:
        raise ValueError("a laminate needs at least one layer")
    if any(t <= 0.0 for t in thicknesses):
        raise ValueError(f"layer thicknesses must be positive, got {thicknesses}")
    if vars_per_layer < 1:
        raise ValueError("vars_per_layer must be >= 1")

    location_fns = tuple(location_fns)
    if surface_load_fns is None:
        f = uniform_transverse_load(load_mag)
        surface_load_fns = tuple(f for _ in location_fns)
    else:
        surface_load_fns = tuple(surface_load_fns)
        if len(surface_load_fns) != len(location_fns):
            raise ValueError(
                f"surface_load_fns ({len(surface_load_fns)}) must match "
                f"location_fns ({len(location_fns)})"
            )

    return LaminatedShell(
        mesh=[mesh, mesh],
        vec=[3, 2],
        dim=2,
        ele_type=[ele_type, ele_type],
        location_fns=list(location_fns) or None,
        additional_info=(layer_fn, vars_per_layer, thicknesses, delta_t,
                         dT_grad, kappa_s, nonlinear, surface_load_fns),
    )


# ── Loads and boundary conditions ────────────────────────────────────────────

def uniform_transverse_load(load_mag: float) -> Callable:
    """A surface weak form applying a uniform transverse line load.

    feax accumulates the surface residual as ``+= t``, which is ``−t_phys``, so
    a *downward* physical load of magnitude ``load_mag`` is written as ``+`` on
    the ``w`` component.  The moment traction on ``(θx, θy)`` is zero.
    """
    def surface_weak_form(vals, x, *design):
        return [np.array([0.0, 0.0, load_mag]), np.zeros(2)]

    return surface_weak_form


def clamp_bc(problem, location) -> fe.DirichletBC:
    """Fully clamp ``location``: all five DOFs (u, v, w and θx, θy) set to zero.

    For a simple support (``w`` only, rotations free) build the
    :class:`feax.DirichletBCSpec` list yourself — this helper is deliberately
    only the fully-clamped case.
    """
    return fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=location, component="all", value=0.0,
                           variable_index=0),
        fe.DirichletBCSpec(location=location, component="all", value=0.0,
                           variable_index=1),
    ]).create_bc(problem)


# ── Design space ─────────────────────────────────────────────────────────────

def layer_field_names(n_layers: int, per_layer: Sequence[str] = ("rho", "x1", "x2", "x3")):
    """Design-field names in the order the weak form slices them.

    Layer-major: ``("rho0", "x1_0", ..., "rho1", "x1_1", ...)``.  A single-name
    ``per_layer`` (density only) gives ``("rho0", "rho1", ...)``.
    """
    return tuple(f"{base}{k}" if base == "rho" else f"{base}_{k}"
                 for k in range(n_layers) for base in per_layer)


def design_space(
    n_layers: int = 2,
    *,
    oriented: bool = True,
    rho_bounds: tuple[float, float] = (0.0, 1.0),
    rho_init: float = 0.5,
    rho_filter_frac: float | None = 0.05,
    rho_filter_radius: float | None = None,
    theta_filter_frac: float | None = 0.05,
    theta_filter_radius: float | None = None,
    ori_tol: float = 1e-2,
    x3_bounds: tuple[float, float] = (-1.0, 1.0),
) -> DesignSpace:
    """The :class:`~path_optimizer.DesignSpace` matching a laminate constitutive.

    ``oriented=True`` (default) gives four fields per layer — ``rho{k}``,
    ``x1_{k}``, ``x2_{k}``, ``x3_{k}`` — laid out layer-major, exactly the order
    :func:`path_optimizer.materials.orientation_blend` consumes.  Use
    ``oriented=False`` with :func:`path_optimizer.materials.isotropic` for one
    density field per layer.

    All layers get the same bounds and filter radii; construct the
    :class:`~path_optimizer.DesignSpace` by hand if you need them to differ.
    The orientation fields start at ``-1 + ori_tol``: the exact corner of the
    libertas box is a degenerate ``a₂`` with vanishing gradient, so a start
    there never moves.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    # An explicit radius wins over the fractional default, rather than tripping
    # DesignField's "give one or the other" guard.
    if rho_filter_radius is not None:
        rho_filter_frac = None
    if theta_filter_radius is not None:
        theta_filter_frac = None

    start = -1.0 + ori_tol
    fields = []
    for k in range(n_layers):
        fields.append(DesignField(
            f"rho{k}", lower=rho_bounds[0], upper=rho_bounds[1], init=rho_init,
            filter_frac=rho_filter_frac, filter_radius=rho_filter_radius))
        if oriented:
            for name, init, (lo, hi) in ((f"x1_{k}", start, (-1.0, 1.0)),
                                         (f"x2_{k}", start, (-1.0, 1.0)),
                                         (f"x3_{k}", 0.0, x3_bounds)):
                fields.append(DesignField(
                    name, lower=lo, upper=hi, init=init,
                    filter_frac=theta_filter_frac,
                    filter_radius=theta_filter_radius))
    return DesignSpace(fields)
