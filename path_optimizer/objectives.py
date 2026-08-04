"""Objectives and design regularisers.

Three groups, all plain functions meant to be called from inside
:meth:`path_optimizer.Pipeline.objective` (the driver jits the whole thing, so
nothing here needs its own ``jax.jit``):

**Displacement matching** — :func:`displacement_error`, :func:`transverse_error`.
Squared distance from a target displacement field.  A zero target gives the
"stay-flat" / load-compensation objective; a non-zero one gives shape matching
or form finding.  These take a :class:`feax.Solution` directly — feax 0.8's
``sol.field(i)`` means they no longer need the ``Problem`` handed to them, so
they are functions rather than factories.

**Stiffness / material** — :func:`create_compliance_fn`,
:func:`create_volume_fn`, re-exported unchanged from :mod:`feax.gene` so one
import covers every response.  These *are* factories: build them once in
:meth:`path_optimizer.Pipeline.build`.

**Design regularisers** — :func:`grey_penalty`, :func:`ud_penalty`,
:func:`magnitude_consistency`.  Pure functions of the design mapping that push
the design toward something manufacturable.  Each returns an O(1) scalar so the
weights in a weighted sum are comparable across problems.

.. important::

   Feed the regularisers the **physical** density (filtered, and Heaviside-
   projected if you project), *not* a SIMP-interpolated field.  ``rho ** 3`` is
   a stiffness, not material: a grey-scale penalty applied to it measures the
   wrong thing and a density-weighted average weights the wrong nodes.  This is
   the same distinction :meth:`path_optimizer.Pipeline.transform` documents.

Example
-------

.. code-block:: python

    from path_optimizer import objectives as obj

    class Flat(po.Pipeline):
        def build(self, mesh):
            ...
            self.u_target = obj.transverse_target(mesh, lambda x, y: 0.0 * x)
            # Normalise on the initial design so the objective starts at ~1.
            sol0 = self.solver(initial_params)
            self.denom = obj.displacement_error(sol0, self.u_target)
            self.layers = obj.group_layers(shell.layer_field_names(2))

        def objective(self, design, **params):
            sol = self.solver(...)
            return (obj.displacement_error(sol, self.u_target, denom=self.denom)
                    + 1.0 * obj.ud_penalty(design, self.layers)
                    + 0.5 * obj.grey_penalty(design, ("rho0", "rho1"))
                    + 1.0 * obj.magnitude_consistency(design, self.layers))
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as np
import numpy as onp
from feax.gene import create_compliance_fn, create_volume_fn
from feax.mechanics.orientation import orientation_tensor_2d

__all__ = [
    # displacement matching
    "displacement_error",
    "transverse_error",
    "transverse_target",
    # re-exported feax responses
    "create_compliance_fn",
    "create_volume_fn",
    # design regularisers
    "grey_penalty",
    "ud_penalty",
    "create_orientation_smoothness_fn",
    "magnitude_consistency",
    "group_layers",
]


# ── Displacement matching ────────────────────────────────────────────────────

def displacement_error(sol, u_target, *, var_index: int = 0, denom: float = 1.0):
    """``‖u − u*‖² / denom`` over the whole displacement field.

    Parameters
    ----------
    sol : feax.Solution
        A solve's result.  Variable ``var_index`` is compared component-wise.
    u_target : array, shape ``(n_nodes, vec)``
        Target displacement.  All-zero gives the stay-flat objective: the design
        must cancel whatever the load and any eigenstrain would otherwise do.
    denom : float
        Normaliser.  Pass ``1.0`` to get the raw squared error; the usual choice
        is the error of the *initial* design, so the objective starts near 1 and
        the regulariser weights mean the same thing from problem to problem::

            denom = displacement_error(sol_initial, u_target)

    Notes
    -----
    This is the error of the full vector, not just one component — an in-plane
    drift counts against the design as much as an out-of-plane one.  Use
    :func:`transverse_error` when only the out-of-plane shape matters.
    """
    u = sol.field(var_index)
    diff = u - u_target
    return np.sum(diff * diff) / np.maximum(denom, 1e-30)


def transverse_error(sol, w_target, *, var_index: int = 0, component: int = 2,
                     denom: float = 1.0):
    """``‖w − w*‖² / denom`` for a single displacement component.

    The shell version of :func:`displacement_error`: for a plate the shape is
    carried by ``w`` alone (``component=2`` of ``var0 = (u, v, w)``), and
    constraining the in-plane components as well over-specifies the problem —
    the laminate must stretch in-plane to bend at all.

    ``w_target`` is a ``(n_nodes,)`` array; build one with
    :func:`transverse_target`, or pass ``0.0`` for a flat target.
    """
    w = sol.field(var_index)[..., component]
    diff = w - w_target
    return np.sum(diff * diff) / np.maximum(denom, 1e-30)


def transverse_target(mesh, fn: Callable | None = None, *, vec: int = 3,
                      component: int = 2):
    """Nodal target displacement from a shape function ``fn(x, y) -> w``.

    Returns ``(n_nodes, vec)`` with every component zero except ``component``,
    which holds ``fn`` sampled at the nodes.  ``fn=None`` gives an all-zero
    target — the flat / stay-flat case.

    Pass ``vec=1`` and use the result with :func:`transverse_error` if you only
    want the scalar field.
    """
    pts = onp.asarray(mesh.points)
    n = pts.shape[0]
    w = onp.zeros(n) if fn is None else onp.asarray(fn(pts[:, 0], pts[:, 1]))
    if w.shape != (n,):
        raise ValueError(f"target fn returned shape {w.shape}, expected ({n},)")
    if vec == 1:
        return np.asarray(w)
    out = onp.zeros((n, vec))
    out[:, component] = w
    return np.asarray(out)


# ── Field grouping ───────────────────────────────────────────────────────────

def group_layers(names: Sequence[str], per_layer: int = 4):
    """Chunk a flat design-field name tuple into one group per layer.

    ``path_optimizer.shell.design_space`` lays its fields out layer-major, so
    ``group_layers(space.names)`` recovers
    ``(("rho0", "x1_0", "x2_0", "x3_0"), ("rho1", ...))`` — exactly the form the
    orientation regularisers below want.  A plane problem with a single oriented
    layer gives one group.
    """
    names = tuple(names)
    if per_layer < 1:
        raise ValueError("per_layer must be >= 1")
    if len(names) % per_layer:
        raise ValueError(
            f"{len(names)} field names do not divide into groups of {per_layer}")
    return tuple(names[k:k + per_layer]
                 for k in range(0, len(names), per_layer))


# ── Design regularisers ──────────────────────────────────────────────────────

def grey_penalty(design, fields: Sequence[str]):
    """Mean ``4·ρ·(1−ρ)`` over ``fields`` — pushes the density to ``{0, 1}``.

    ``1`` at ρ = 0.5 everywhere, ``0`` for a fully black-and-white design, so it
    doubles as a readable convergence diagnostic.  Averaged over the named
    fields, so a two-layer laminate is not penalised twice as hard as a plate.

    A Heaviside projection with a ramped sharpness usually does this job better;
    reach for the penalty when you want the pressure without the projection's
    extra continuation parameter.
    """
    fields = tuple(fields)
    if not fields:
        raise ValueError("grey_penalty needs at least one field")
    return sum(np.mean(4.0 * design[f] * (1.0 - design[f]))
               for f in fields) / len(fields)


def _a2_of(x1, x2, x3, sgn_beta):
    a2, _, _ = orientation_tensor_2d(x1, x2, x3, sgn_beta=sgn_beta)
    return a2


def _ud_one(x1, x2, x3, sgn_beta):
    """``4·det(a₂) / tr(a₂)²`` — 0 for unidirectional, 1 for isotropic."""
    a2 = _a2_of(x1, x2, x3, sgn_beta)
    tr = a2[0, 0] + a2[1, 1]
    det = a2[0, 0] * a2[1, 1] - a2[0, 1] * a2[0, 1]
    return 4.0 * det / (tr * tr + 1e-12)


def _mag_one(x1, x2, x3, rho, sgn_beta):
    """``(|T| − ρ)²`` with ``|T| = √((a₁₁−a₂₂)² + 4a₁₂²) ∈ [0, 1]``."""
    a2 = _a2_of(x1, x2, x3, sgn_beta)
    tx = a2[0, 0] - a2[1, 1]
    ty = 2.0 * a2[0, 1]
    t_mag = np.sqrt(tx * tx + ty * ty + 1e-30)
    return (t_mag - rho) ** 2


def ud_penalty(design, layers: Sequence[Sequence[str]], *, sgn_beta: float = 10.0):
    """Density-weighted unidirectionality penalty, averaged over layers.

    Per node, ``4·det(a₂)/tr(a₂)²`` is 0 when the orientation tensor is rank-1
    (a single fibre direction — printable as a continuous path) and 1 when it is
    isotropic (no preferred direction — not printable as fibre at all).  Driving
    it down is what turns a continuous orientation field into extractable paths.

    Weighted by the density so that **void regions do not contribute**: where
    there is no fibre, its notional direction is meaningless and penalising it
    would fight the topology for no benefit.

    ``layers`` is a sequence of ``(rho, x1, x2, x3)`` name tuples — see
    :func:`group_layers`.  ``sgn_beta`` must match the value used to build the
    material (:func:`path_optimizer.materials.orientation_blend`), or the
    penalty measures a different tensor than the physics does.
    """
    return _layer_mean(design, layers, sgn_beta, weighted=True, fn=_ud_one)


def magnitude_consistency(design, layers: Sequence[Sequence[str]], *,
                          sgn_beta: float = 10.0):
    """Mean ``(|T| − ρ)²`` over layers — ties alignment strength to density.

    ``|T|`` is the orientation tensor's anisotropy magnitude: 0 when isotropic,
    1 when fully aligned.  Without this term the optimiser can park a node at
    "half fibre, fully aligned" and "full fibre, half aligned" indifferently,
    because they give similar stiffness — but only the first is manufacturable
    at that density.  Pinning ``|T| = ρ`` removes the ambiguity.

    ``layers`` and ``sgn_beta`` as in :func:`ud_penalty`.
    """
    return _layer_mean(design, layers, sgn_beta, weighted=False, fn=_mag_one)


def create_orientation_smoothness_fn(problem, *, length_scale: float,
                                     sgn_beta: float = 10.0,
                                     var_index: int = 0):
    """Penalise how fast the fibre direction turns — ``fn(design, layers)``.

    A factory, not a plain function: it closes over the mesh's shape-function
    gradients, so build it once in :meth:`path_optimizer.Pipeline.build`.

    .. code-block:: text

        (length_scale² / A) · ∫ ρ |∇T|² dA,     T = (a₁₁ − a₂₂, 2a₁₂)

    ``T`` is the deviatoric part of ``a₂`` — the double-angle vector
    ``|T|(cos 2θ, sin 2θ)``.  **Differentiating that, rather than the angle
    itself, is the whole point**: a director is an axis, so ``θ`` jumps by π at
    arbitrary places and its gradient is meaningless there, while ``T`` is
    single-valued and its gradient is not.

    Weighted by density, so void regions — where the direction is arbitrary —
    contribute nothing.

    ``length_scale`` is required and sets what "smooth" means: the penalty is
    ``(L·|∇T|)²`` averaged over the part, so ``L`` is the distance over which a
    full change of ``T`` costs order 1.  A stripe period, or the fibre-placement
    turning radius, are the physically meaningful choices.

    Why here rather than smoothing the director afterwards: smoothing a finished
    design just degrades it, whereas a penalty inside the objective lets the
    optimiser find the best design *subject to* being smooth — it can move
    material and re-route load paths to buy the smoothness back.
    """
    cells = onp.asarray(problem.cells_list[var_index])
    shape_grads = np.asarray(problem.shape_grads)          # (nc, nq, npc, dim)
    jxw = np.asarray(problem.JxW)[:, 0, :]                 # (nc, nq)
    scale = float(length_scale) ** 2 / float(onp.asarray(problem.JxW)[:, 0, :].sum())

    def smoothness(design, layers):
        layers = tuple(tuple(g) for g in layers)
        if not layers:
            raise ValueError("need at least one layer group")
        total = 0.0
        for rho_name, n1, n2, n3 in layers:
            a2 = jax.vmap(lambda a, b, c: orientation_tensor_2d(
                a, b, c, sgn_beta=sgn_beta)[0])(
                design[n1], design[n2], design[n3])
            t_field = np.stack([a2[:, 0, 0] - a2[:, 1, 1], 2.0 * a2[:, 0, 1]],
                               axis=-1)                    # (n_nodes, 2)
            grad_sq = 0.0
            for k in range(2):
                f_cell = t_field[:, k][cells]              # (nc, npc)
                g = np.einsum("ci,cqid->cqd", f_cell, shape_grads)
                grad_sq = grad_sq + (g * g).sum(axis=-1)   # (nc, nq)
            rho_cell = design[rho_name][cells].mean(axis=-1)[:, None]
            total = total + (rho_cell * grad_sq * jxw).sum()
        return total * scale / len(layers)

    return smoothness


def _layer_mean(design, layers, sgn_beta, *, weighted, fn):
    """Average a per-node orientation penalty over layers.

    ``weighted`` selects a density-weighted mean (``Σρp/Σρ``) over a plain one.
    """
    layers = tuple(tuple(g) for g in layers)
    if not layers:
        raise ValueError("need at least one layer group")
    for g in layers:
        if len(g) != 4:
            raise ValueError(
                f"each layer group must be (rho, x1, x2, x3); got {g}")

    total = 0.0
    for rho_name, n1, n2, n3 in layers:
        rho = design[rho_name]
        if weighted:
            vmapped = jax.vmap(lambda a, b, c: fn(a, b, c, sgn_beta))
            p = vmapped(design[n1], design[n2], design[n3])
            total = total + np.sum(rho * p) / (np.sum(rho) + 1e-12)
        else:
            vmapped = jax.vmap(lambda a, b, c, r: fn(a, b, c, r, sgn_beta))
            total = total + np.mean(vmapped(design[n1], design[n2],
                                            design[n3], rho))
    return total / len(layers)
