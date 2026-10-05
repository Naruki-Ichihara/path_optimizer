"""Aligned Swift–Hohenberg stripe generation.

Turns a continuous fibre-orientation field into a *stripe* field whose crests
run parallel to the fibre — the intermediate that later becomes extractable
print paths.  The optimiser produces a director ``d(x)`` (a direction, defined
only up to sign, with no notion of spacing); a printer needs discrete paths at a
fixed pitch.  Relaxing an aligned Swift–Hohenberg field bridges the two: it is
an energy whose minimisers are stripes of a prescribed wavelength, tilted to
follow ``d``.

.. code-block:: text

    E[u] = ∫ ½((k₀² + ∇²)u)²  +  ½γ((d·∇)u)²  +  ¼u⁴  −  ½εu²  dx

* ``(k₀² + ∇²)u`` selects the wavelength: the energy is minimised by patterns
  with ``|k| = k₀``, i.e. a stripe period of ``2π/k₀``.
* ``γ((d·∇)u)²`` is the alignment ("pencil") term.  It penalises variation of
  ``u`` *along* ``d``, so the surviving stripes are the ones that run parallel
  to the fibre.  In the continuum it does not affect wavelength selection at
  all — for ``k ⊥ d`` it is identically zero — but a naive discretisation
  breaks that, which is why it gets special treatment below.
* ``¼u⁴ − ½εu²`` is the usual double well that saturates the amplitude.

Implementation notes:

* The 4th-order operator is split into two coupled 2nd-order equations
  (``v = (1 + ∇²)u`` in dimensionless coordinates) so CG1 elements suffice.
* The mesh is rescaled by ``k₀``, which makes ``k₀ = 1`` and keeps the operator
  well-conditioned regardless of the physical stripe period.
* Time stepping is semi-implicit backward Euler with a convex/concave split of
  the cubic, which makes **each step linear** — no Newton iteration.  The
  default split uses a constant stabilisation, so the operator does not change
  between steps and is factorised once per relaxation rather than once per step.
  That matters because the linear solve is 95-97% of a step's cost; see
  :class:`AlignedSHStep`.
* The alignment term is assembled **separately at 1-point quadrature** and added
  to the fully integrated remainder — selective reduced integration.  Without
  it, the alignment term corrupts the wavelength badly enough that ``γ`` has to
  be kept small, trading away the alignment it exists to provide.  See
  :func:`make_sh_solver`'s ``selective_aniso``.

Resolution is the thing to get right: the stripe period must be resolved by the
mesh.  Below roughly 6 elements per period the pattern degrades into noise, so
:func:`make_sh_solver` measures it and warns.  Optimisation meshes are usually
too coarse, hence :func:`resample` and the ``mesh=`` argument of
:func:`stripes_from_result`.

Even well resolved, the delivered period runs a few percent long — about +4% at
6 elements per period, falling with resolution.  That residue is CG1 dispersion
plus the stabilised split, not the alignment term, and it is *not* corrected
here: a relaxation's period scatters by a similar few percent from one random
seed to the next, so a single-run correction cannot be measured accurately
enough to be worth applying.  Use a finer mesh if the pitch has to be exact.

Example
-------

.. code-block:: python

    from path_optimizer import objectives as obj, shell, stripes

    fine = fe.mesh.rectangle_mesh(Nx=320, Ny=160, domain_x=Lx, domain_y=Ly,
                                  ele_type="QUAD4")
    fields = stripes.stripes_from_result(
        result, obj.group_layers(shell.layer_field_names(2)),
        stripe_period=3.0e-3, mesh=fine,
    )
    stripes.save_vtu(fields[0], "layer0.vtu")
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import feax as fe
import feax.gene as gene
import jax
import jax.numpy as jnp
import numpy as onp
from feax.assembler import create_J_bc_csr_function, create_res_bc_function
from feax.mechanics.orientation import orientation_tensor_2d

from path_optimizer.materials import DEFAULT_SGN_BETA

__all__ = [
    "AlignedSHStep",
    "SHSolver",
    "StripeField",
    "Relaxation",
    "DEFAULT_TOL",
    "DEFAULT_GAMMA",
    "EYRE",
    "make_sh_solver",
    "relax",
    "generate_stripes",
    "stripes_from_result",
    "director_from_orientation",
    "director_from_a2",
    "resample",
    "save_vtu",
    "phase_seed",
    "PhaseSeed",
    "orient_director",
    "relax_hybrid",
]


class _Eyre:
    """Sentinel selecting the Eyre split (see :class:`AlignedSHStep`)."""

    def __repr__(self):
        return "EYRE"


#: Pass as ``stabilization=`` to use the Eyre linearisation, whose matrix
#: changes every step.  Tighter per step, but ~100x more expensive overall.
EYRE = _Eyre()


#: Default alignment strength.  Measured, not guessed: with selective reduced
#: integration the period bias is flat in gamma (+3.9% at 4 through +4.3% at 32
#: on the worst mesh direction), so gamma is free to be chosen purely for
#: alignment quality.  16 captures most of the available improvement --
#: alignment error 0.0506 -> 0.0370 going from 4 to 16, then only 0.0280 at 32
#: for a further doubling.  Without selective integration this value would be
#: unusable: it cost +20.9% on the period.  See :class:`AlignedSHStep`.
DEFAULT_GAMMA = 16.0


# ── The one-step problem ─────────────────────────────────────────────────────

class AlignedSHStep(fe.Problem):
    """One backward-Euler step of aligned SH, in mixed form.

    Dimensionless coordinates — the mesh is pre-scaled by ``k₀``, so ``k₀ = 1``.
    Variables: ``var0 = u`` (the stripe field), ``var1 = v`` (auxiliary
    ``v = (1 + ∇²)u``).  Traced params per node: the director ``(d_x, d_y)``,
    the previous step ``u_old``, and the region mask ``∈ {0, 1}``.

    Built with ``additional_info=(dt, epsilon, gamma, stabilization)``; use
    :func:`make_sh_solver`.

    The cubic ``u³`` cannot be taken fully implicitly without a Newton solve, so
    it is convex-split.  Two splittings are available, chosen by
    ``stabilization``:

    * ``None`` — **Eyre**: linearise about the previous step,
      ``u³ ≈ 3u_old²·u − 2u_old³``.  The tightest linearisation, but the
      implicit coefficient contains ``u_old``, so the **matrix changes every
      step** and must be re-factorised every step.
    * a float ``S`` — **linearly stabilised**: ``u³ ≈ S·u + (u_old³ − S·u_old)``.
      The implicit coefficient is now constant, so **the matrix is constant for
      a given director and mask** and one factorisation serves the whole
      relaxation.

    Both forms have the same fixed point: substituting ``u = u_old = U`` into
    either gives ``v = εU − U³``.  They do **not** produce the same field,
    though — SH is multistable, so a different path lands on a different
    arrangement of stripe phase and defects.  What is stable across the choice
    is the pattern's statistics.  Measured on a 45-degree uniform director at 6
    elements per period, relaxed to convergence:

    .. code-block:: text

        split        steps   ms/step   wavelength   rms amp   |k.d|
        Eyre           106     129.4     1.0394mm    0.814    0.000
        Eyre (seed 7)  128     110.0     1.0594mm    0.832    0.037
        S = 3          105      34.6     1.0795mm    0.803    0.000
        S = 6          108      34.9     1.0839mm    0.794    0.000
        S = 12         133      33.5     1.0810mm    0.777    0.000

    So the stabilised split costs ~3.7x less per step at the same step count,
    and its wavelength runs ~2-4% longer than Eyre's — a bias of the same order
    as Eyre's own seed-to-seed scatter (1.9% between the two seeds above), and
    small next to the CG1 discretisation bias the mesh already imposes (+4.3% at
    6 elements per period).

    Stability needs ``S ≥ max(3u² − ε)`` along the trajectory; with ``ε = 1`` the
    field reaches ``|u| ≈ 1.2-1.4``, so ``3u² − ε ≈ 3.3`` and ``S = 6ε`` has
    margin.  ``S = 12`` is measurably over-damped (25% more steps) for no gain.
    """

    def custom_init(self, dt, epsilon, gamma, stabilization=None, mode="full"):
        self.dt = dt
        self.epsilon = epsilon
        self.gamma = gamma
        self.stabilization = stabilization
        self.mode = mode

    def get_weak_form(self):
        dt, eps, gamma = self.dt, self.epsilon, self.gamma
        stab = self.stabilization
        mode = self.mode

        def aniso_only(vals, grads, x, d_x, d_y, u_old, mask):
            """Just the alignment term, for selective reduced integration."""
            grad_u = grads[0][0]
            d = jnp.array([d_x, d_y])
            d_grad_u = d[0] * grad_u[0] + d[1] * grad_u[1]
            grad_u_term = dt * (gamma * mask) * d_grad_u * d
            return ([jnp.zeros(1), jnp.zeros(1)],
                    [grad_u_term[None, :], jnp.zeros((1, 2))])

        def weak_form(vals, grads, x, d_x, d_y, u_old, mask):
            u, v = vals[0][0], vals[1][0]
            grad_u, grad_v = grads[0][0], grads[1][0]

            d = jnp.array([d_x, d_y])
            d_grad_u = d[0] * grad_u[0] + d[1] * grad_u[1]

            # Outside the mask the linear driving term and the alignment term
            # both switch off, so the field decays to zero instead of growing
            # stripes in the void.
            eps_eff = eps * mask
            gamma_eff = gamma * mask

            u_old_m = mask * u_old
            if stab is None:
                # Eyre: implicit 3u_old²·u, explicit −2u_old³.
                u_old_sq = u_old_m * u_old_m
                cubic = 3.0 * u_old_sq * u - 2.0 * u_old_sq * u_old_m
            else:
                # Linearly stabilised: implicit S·u, explicit u_old³ − S·u_old.
                # S is masked so the void keeps the plain decay equation.
                s_eff = stab * mask
                cubic = s_eff * u + (u_old_m ** 3 - s_eff * u_old_m)

            mass_u = u - u_old_m + dt * v - dt * eps_eff * u + dt * cubic
            grad_u_term = -dt * grad_v + dt * gamma_eff * d_grad_u * d

            # v equation: v − u − ∇²u = 0.
            mass_v = v - u
            grad_v_term = grad_u

            return ([jnp.array([mass_u]), jnp.array([mass_v])],
                    [grad_u_term[None, :], grad_v_term[None, :]])

        return aniso_only if mode == "aniso" else weak_form


# ── Solver bundle ────────────────────────────────────────────────────────────

@dataclass
class SHSolver:
    """A built aligned-SH stepper, reusable across layers and designs.

    Holds the assembled-operator and residual functions rather than a finished
    solver: the operator depends on the director and mask, which are per-layer,
    so the factorisation is built inside :func:`relax`.
    """

    problem: AlignedSHStep
    assemble_J: object
    """``(x, traced_params) -> CSRMatrix`` — jitted."""
    assemble_R: object
    """``(x, traced_params) -> residual`` — jitted."""
    n_dofs: int
    mesh_physical: fe.Mesh
    """The mesh in physical units, as handed in."""
    mesh_dimensionless: fe.Mesh
    """The same mesh scaled by ``k₀``; what the problem is actually posed on."""
    n_nodes: int
    stripe_period: float
    """The pitch that was **asked for**."""
    elements_per_period: float
    stabilization: float | None
    """``None`` for the Eyre split (matrix per step), a float for the stabilised
    split (one factorisation per relaxation)."""
    assemble_J_aniso: object = None
    """``(x, traced_params) -> CSRMatrix`` for the alignment term alone, built
    on 1-point quadrature.  ``None`` when the term is folded into the main
    operator at full quadrature instead."""
    gamma: float = 0.0
    """The alignment strength the operator was built with — needed to express a
    freed-region strength as a ratio (see :func:`relax_hybrid`)."""


def make_sh_solver(mesh, stripe_period: float, *, epsilon: float = 1.0,
                   gamma: float = DEFAULT_GAMMA, dt: float = 0.5,
                   stabilization: float | None = None,
                   gauss_order: int | None = None,
                   selective_aniso: bool = True,
                   min_elements_per_period: float = 6.0) -> SHSolver:
    """Build the aligned-SH stepper on ``mesh``.

    Parameters
    ----------
    mesh : feax.Mesh
        Any 2D mesh, in physical units.  This is where the stripes are resolved,
        so it is usually finer than the optimisation mesh.
    stripe_period : float
        Period of the stripe field itself, in the mesh's units.  **Not the
        spacing of the extracted paths**: those are the zero level set, and
        ``cos(φ)`` crosses zero twice per period, so they land at
        ``stripe_period / 2``.  Pass twice the pitch you want to print at.
    epsilon : float
        Linear driving term.  Larger grows stripes faster and saturates harder.
    gamma : float
        Alignment strength — how hard the stripes are pushed to follow ``d``.
        With ``selective_aniso`` on, raising it is nearly free: measured at 6
        elements per period on the worst mesh direction (45 degrees),

        .. code-block:: text

            gamma    period bias        alignment error
                     2x2    selective   selective
              4     + 3.7%   + 3.9%      0.0506
              8     +10.7%   + 3.7%      0.0451
             16     +20.9%   + 4.0%      0.0370
             32       --     + 4.3%      0.0280

        so the default is 16 rather than a cautious 4.  Under full quadrature
        the same value costs +20.9% on the pitch, which is why this used to be
        a trade-off and no longer is.
    dt : float
        Backward-Euler step.  Both splittings are stable, so this is a
        convergence-rate knob rather than a stability one.
    stabilization : float or EYRE, optional
        Constant ``S`` for the linearly stabilised cubic split — see
        :class:`AlignedSHStep`.  ``None`` (default) picks ``6·epsilon``, which
        covers ``3u² − ε`` for the amplitudes SH actually reaches.  This is what
        makes the operator constant across a relaxation, so it can be factorised
        once instead of once per step — and since the linear solve is 95-97% of
        a step's cost, that is the whole performance story.  Pass :data:`EYRE`
        to fall back to the tighter Eyre linearisation, which re-factorises
        every step.
    selective_aniso : bool
        Assemble the alignment term on its own at 1-point (element-centre)
        quadrature and add it to the fully integrated remainder.  On by default,
        and the reason ``gamma`` can be large.

        The alignment term should be blind to a mode with ``k ⊥ d``, and on a
        bilinear element it is — but only at the element centre.  Away from it
        the ``xy`` term of ``a + bx + cy + e·xy`` tilts ``grad(u_h)`` off the
        wave direction, and the term picks up energy it should not see.  The
        leak ``int (d.grad u)^2 / int |grad u|^2``, which is 0 in the continuum,
        measured on the interpolant alone:

        .. code-block:: text

            el/period    2x2 Gauss              centre
                        0deg  22.5deg  45deg   0deg  22.5deg  45deg
                4       0     0.0300   0.0570  0     0.0032   0
                6       0     0.0122   0.0239  0     0.0006   0
               12       0     0.0029   0.0058  0     0.0000   0

        Exactly zero on both mesh symmetry axes, 42x smaller at the worst angle,
        and converging at O(h^4) instead of O(h^2).  A 3x3 rule changes nothing,
        confirming this is the shape function and not quadrature accuracy.

        Under-integrating the *whole* form is not a substitute: tried, and the
        mass and Laplacian terms went rank-deficient — 1000x the baseline power
        in checkerboard modes and no run converged at all.
    min_elements_per_period : float
        Warn below this resolution.  Under-resolved stripes come out as noise,
        and nothing downstream will tell you — hence the warning here.
    """
    if stripe_period <= 0.0:
        raise ValueError(f"stripe_period must be > 0, got {stripe_period}")

    if stabilization is EYRE:
        stab = None
    elif stabilization is None:
        stab = 6.0 * epsilon
    else:
        stab = float(stabilization)
        if stab <= 0.0:
            raise ValueError(f"stabilization must be > 0, got {stab}")

    pts = onp.asarray(mesh.points)
    h = _mean_element_size(pts, onp.asarray(mesh.cells))
    per_period = stripe_period / h
    if per_period < min_elements_per_period:
        warnings.warn(
            f"stripe period {stripe_period:g} spans only {per_period:.1f} "
            f"elements (mean size {h:.3g}); below ~{min_elements_per_period:g} "
            "the pattern degrades into noise. Refine the mesh or increase the "
            "period.", stacklevel=2)

    # Rescale so k0 = 1: the operator (k0^2 + laplacian) becomes (1 + laplacian)
    # and stays well conditioned whatever the physical period is.
    k0 = 2.0 * onp.pi / stripe_period
    dimensionless = fe.Mesh(pts * k0, onp.asarray(mesh.cells),
                            ele_type=mesh.ele_type)
    n_nodes = int(pts.shape[0])

    def _build(gamma_, mode, order):
        return AlignedSHStep(
            mesh=[dimensionless, dimensionless], vec=[1, 1], dim=2,
            ele_type=[mesh.ele_type, mesh.ele_type],
            gauss_order=None if order is None else [order, order],
            additional_info=(dt, epsilon, gamma_, stab, mode),
        )

    # Selective reduced integration.  The alignment term is exact for a mode it
    # should ignore only when sampled at the element centre: the bilinear xy
    # term, which is what tilts grad(u_h) off the wave direction, vanishes
    # there.  Measured leak int (d.grad u)^2 / int |grad u|^2 at 6 elements per
    # period, 45 degrees: 0.0239 on 2x2 Gauss, exactly 0 at the centre; worst
    # case over angles drops 42x and converges at O(h^4) instead of O(h^2).
    #
    # Under-integrating the WHOLE form instead is not an option -- it was tried
    # and the mass/Laplacian terms went rank-deficient, putting 1000x the
    # baseline power into checkerboard modes and stopping the relaxation from
    # converging at all.  So the alignment term is assembled separately at
    # 1-point quadrature and added to the fully integrated remainder.
    problem = _build(0.0 if selective_aniso else gamma, "full", gauss_order)
    problem_aniso = (_build(gamma, "aniso", 1) if selective_aniso else None)
    # No Dirichlet data: SH's natural boundary conditions are what we want, and
    # the mask already confines the pattern to the fibre region.  That also
    # means the BC-applied assembly below is a no-op, so the operator is exactly
    # the one the factorisation sees.
    bc = fe.DirichletBCConfig([]).create_bc(problem)

    # Deliberately NOT a TracedStructure: building one frees the host-side
    # assembly scratch that create_J_bc_csr_function needs.
    assemble_J = jax.jit(create_J_bc_csr_function(problem, bc))
    assemble_R = jax.jit(create_res_bc_function(problem, bc))

    assemble_J_aniso = None
    if problem_aniso is not None:
        bc_aniso = fe.DirichletBCConfig([]).create_bc(problem_aniso)
        assemble_J_aniso = jax.jit(
            create_J_bc_csr_function(problem_aniso, bc_aniso))
        # Same mesh, same variables, so the sparsity is identical and the two
        # operators can be added value-by-value.  Assert it rather than trust it.
        if not (onp.array_equal(onp.asarray(problem.csr_indptr),
                                onp.asarray(problem_aniso.csr_indptr))
                and onp.array_equal(onp.asarray(problem.csr_indices),
                                    onp.asarray(problem_aniso.csr_indices))):
            raise RuntimeError(
                "the alignment operator has a different sparsity pattern than "
                "the main operator; they cannot be summed")

    return SHSolver(
        problem=problem, assemble_J=assemble_J, assemble_R=assemble_R,
        assemble_J_aniso=assemble_J_aniso,
        n_dofs=int(problem.num_total_dofs_all_vars),
        mesh_physical=mesh, mesh_dimensionless=dimensionless,
        n_nodes=n_nodes, stripe_period=float(stripe_period),
        elements_per_period=float(per_period), gamma=float(gamma),
        stabilization=stab,
    )


def _mean_element_size(pts, cells):
    """Mean edge length of the first two edges of every cell."""
    p = pts[cells]                                   # (n_cells, n_per_cell, dim)
    e0 = onp.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e1 = onp.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    return float(0.5 * (e0.mean() + e1.mean()))


# ── Reusable factorisation ───────────────────────────────────────────────────

# cuDSS matrix-type / view enums as feax passes them: general=0, symmetric=1,
# spd=3; view full=0, upper=1, lower=2.  The SH operator in mixed form is
# non-symmetric (feax's own auto-detect reports GENERAL), so: general + full.
_CUDSS_GENERAL, _CUDSS_FULL = 0, 0


def _factorize(data, indptr, indices, n):
    """Factorise a CSR operator once and return ``solve(b) -> x``.

    Prefers cuDSS through feax's spineax shim so the factors stay on the GPU
    alongside the rest of the relaxation; falls back to SciPy's SuperLU, which
    costs a host round trip per solve but keeps this working without a GPU.
    """
    try:
        from feax.solvers._cudss_compat import factorize, solve_with
    except ImportError:
        pass
    else:
        token = factorize(data, indptr, indices,
                          mtype_id=_CUDSS_GENERAL, mview_id=_CUDSS_FULL)
        return lambda b: solve_with(token, b)

    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    lu = spla.splu(sp.csr_matrix(
        (onp.asarray(data), onp.asarray(indices), onp.asarray(indptr)),
        shape=(n, n)).tocsc())
    return lambda b: jnp.asarray(lu.solve(onp.asarray(b)))


# ── Phase-field seeding ──────────────────────────────────────────────────────

class _PhaseProblem(fe.Problem):
    """Least-squares phase: minimise ``∫|∇φ − n|²`` on the dimensionless mesh.

    Its stationarity condition is ``∫(∇φ − n)·∇w = 0`` for all ``w`` — a Poisson
    problem with ``n`` as the source.  Where ``n`` is a gradient the residual is
    zero and ``φ`` reproduces it exactly; where it is not, the least-squares fit
    spreads the mismatch, and that is precisely the frustration a stripe pattern
    has to absorb as dislocations.
    """

    def get_tensor_map(self):
        # Single-variable problems go through get_tensor_map, not
        # get_weak_form: the residual is int flux . grad(w), so returning
        # grad(phi) - n gives exactly the stationarity condition above.
        def flux(u_grad, n_x, n_y):
            return u_grad - jnp.array([[n_x, n_y]])
        return flux


def orient_director(mesh, director, mask=None):
    """Resolve the director's sign into a continuous vector field.

    A director is an axis: ``d`` and ``−d`` describe the same fibre, so the raw
    field flips sign arbitrarily between neighbouring nodes and cannot be
    integrated.  This walks a spanning tree of the mesh from one node, flipping
    each node to agree with the neighbour it was reached from.

    ``mask`` restricts the walk to the region that has material.  Pass it
    whenever the domain has holes: the director in the void is meaningless (it
    came from resampling a design that is not there), so a tree routed through
    the void carries arbitrary signs into whatever solid region it reaches next.

    The result is continuous along the tree but **not necessarily globally
    consistent**: if the field has odd winding around a loop, no choice of signs
    closes, and the leftover disagreement across non-tree edges marks where the
    pattern is topologically obliged to carry a defect.  The count of such edges
    is returned alongside.

    Returns
    -------
    (n_nodes, 2) array, int
        The oriented field, and how many mesh edges it still disagrees across.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order

    d = onp.asarray(director, dtype=float)
    cells = onp.asarray(mesh.cells)
    n_nodes = onp.asarray(mesh.points).shape[0]

    a, b = [], []
    for i in range(cells.shape[1]):
        j = (i + 1) % cells.shape[1]
        a.append(cells[:, i])
        b.append(cells[:, j])
    a = onp.concatenate(a)
    b = onp.concatenate(b)
    if mask is not None:
        solid = onp.asarray(mask) > 0.5
        keep = solid[a] & solid[b]
        a, b = a[keep], b[keep]
    else:
        solid = onp.ones(n_nodes, dtype=bool)
    adj = coo_matrix((onp.ones(a.size), (a, b)), shape=(n_nodes, n_nodes)).tocsr()
    adj = adj + adj.T

    # Holes can split the material into pieces the tree cannot span; walk each
    # connected piece from its own root rather than leaving it unoriented.
    sign = onp.ones(n_nodes)
    seen = onp.zeros(n_nodes, dtype=bool)
    for root in onp.where(solid)[0]:
        if seen[root]:
            continue
        order, pred = breadth_first_order(adj, int(root), directed=False)
        for node in order[1:]:
            par = pred[node]
            sign[node] = sign[par] * (1.0 if d[node] @ d[par] >= 0.0 else -1.0)
        seen[order] = True
    oriented = d * sign[:, None]

    # Edges the tree did not fix: a genuine topological obstruction, not noise.
    frustrated = int(((oriented[a] * oriented[b]).sum(axis=1) < 0.0).sum() // 2)
    return oriented, frustrated


@dataclass
class PhaseSeed:
    """A constructed stripe field, plus what it tells you about itself."""

    u: onp.ndarray
    """``amplitude · cos(φ) · mask`` — pass as ``relax(..., initial=...)``."""
    phi: onp.ndarray
    """The integrated phase, on the dimensionless mesh (so ``|∇φ| = 1`` is the
    requested pitch)."""
    pitch_error: onp.ndarray
    """``(1/|∇φ| − 1)·100`` — the pitch error this construction will deliver,
    per node, in percent.

    Known before any relaxation, because the least-squares fit matches the
    *direction* of ``n`` but nothing in it forces ``|∇φ| = 1``: where the fit has
    to compromise, the gradient comes out short and the stripes come out wide.
    Validated against the delivered pitch on a real truss at r = 0.932; cells in
    the top 1% of predicted error measured +67% actual against +1% overall.

    This is what :func:`relax_hybrid` gates on.
    """
    frustrated_edges: int
    """Mesh edges the sign walk could not reconcile — defects the topology
    demands regardless of method.  Zero on every case measured so far."""


def phase_seed(sh: SHSolver, director, mask, *, amplitude: float = 1.0,
               verbose: bool = False) -> PhaseSeed:
    """A defect-free-where-possible initial field, from an integrated phase.

    Random noise gives every region an independent phase, so grains collide and
    leave boundaries — and smoothing the noise does not help, because the linear
    growth stage amplifies only the band near ``|k| = k₀`` and discards the
    long-wavelength correlation that was added (measured: seed correlation from
    0 to 8 periods moved defect density by under 1%).  What survives
    amplification is *phase*, so the phase is what has to be supplied.

    Builds ``φ`` by least-squares integration of the stripe normal and returns
    ``cos(φ)``, a pattern already at the right pitch and orientation with a
    single coherent phase.  Where the normal field is a gradient this is
    defect-free; where it is not, the fit leaves the mismatch concentrated, and
    the relaxation resolves it into the dislocations the geometry demands.

    Returns
    -------
    PhaseSeed
    """
    oriented, frustrated = orient_director(sh.mesh_physical, director, mask)
    # Stripes run along d, so they vary along the perpendicular.
    normal = onp.column_stack([-oriented[:, 1], oriented[:, 0]])
    # Zero the source in the void: there is no fibre there, so the phase is
    # unconstrained and the Poisson solve just extends it harmonically.
    normal = normal * onp.asarray(mask, dtype=float)[:, None]

    # Posed on the dimensionless mesh, where k0 = 1 and the target is |∇φ| = 1.
    problem = _PhaseProblem(
        mesh=sh.mesh_dimensionless, vec=1, dim=2,
        ele_type=sh.mesh_dimensionless.ele_type,
    )
    # Pure Neumann leaves φ free up to a constant, which would make the operator
    # singular; pin one node.  The constant is irrelevant — only cos(φ) is used.
    pts = onp.asarray(sh.mesh_dimensionless.points)
    p0 = pts[0]
    tol = 1e-6 * float(onp.abs(pts).max() + 1.0)

    def pin(p):
        return (jnp.abs(p[0] - p0[0]) < tol) & (jnp.abs(p[1] - p0[1]) < tol)

    bc = fe.DirichletBCConfig(
        [fe.DirichletBCSpec(location=pin, component="all", value=0.0)]
    ).create_bc(problem)

    tp = fe.TracedParams(volume_vars=(jnp.asarray(normal[:, 0]),
                                      jnp.asarray(normal[:, 1])))
    opts = fe.DirectSolverOptions(solver="auto", verbose=False)
    solver = fe.create_solver(problem, bc=bc, solver_options=opts,
                              adjoint_solver_options=opts, linear=True,
                              traced_params=tp, return_solution=True)
    phi = onp.asarray(solver(tp).field(0)[:, 0])

    # |grad phi| on the dimensionless mesh: 1 means the pitch asked for.
    grad = _nodal_gradient(sh.mesh_dimensionless, phi)
    gmag = onp.maximum(onp.linalg.norm(grad, axis=1), 1e-6)
    pitch_error = onp.where(onp.asarray(mask) > 0.5,
                            (1.0 / gmag - 1.0) * 100.0, 0.0)

    if verbose:
        print(f"  phase seed: {frustrated} frustrated edge(s); "
              f"phi spans {(phi.max() - phi.min()) / (2 * onp.pi):.1f} periods; "
              f"predicted pitch error median "
              f"{onp.median(pitch_error[onp.asarray(mask) > 0.5]):+.2f}%, "
              f"p95 {onp.percentile(onp.abs(pitch_error[onp.asarray(mask) > 0.5]), 95):.1f}%")
    return PhaseSeed(
        u=amplitude * onp.cos(phi) * onp.asarray(mask, dtype=float),
        phi=phi, pitch_error=pitch_error, frustrated_edges=frustrated)


def _nodal_gradient(mesh, field):
    """Least-squares nodal gradient of a nodal field, via the element gradients.

    Averages each element's quadrature-point gradients onto its nodes.  Cruder
    than a proper L2 projection, but this feeds a gate threshold, not the
    physics, and it needs no extra solve.
    """
    cells = onp.asarray(mesh.cells)
    pts = onp.asarray(mesh.points)[:, :2]
    f = onp.asarray(field)
    # One bilinear gradient per element, from its corner values (centre rule).
    p = pts[cells]
    v = f[cells]
    # Fit f ~ a + b x + c y by least squares over the element's own nodes.
    ctr = p.mean(axis=1, keepdims=True)
    dp = p - ctr
    dv = v - v.mean(axis=1, keepdims=True)
    gram = onp.einsum("cni,cnj->cij", dp, dp)
    rhs = onp.einsum("cni,cn->ci", dp, dv)
    # rhs is (n_cells, 2); numpy would read that as a matrix, so give it
    # an explicit trailing axis.
    grads = onp.linalg.solve(gram + 1e-30 * onp.eye(2), rhs[..., None])[..., 0]
    out = onp.zeros((pts.shape[0], 2))
    cnt = onp.zeros(pts.shape[0])
    for k in range(cells.shape[1]):
        onp.add.at(out, cells[:, k], grads)
        onp.add.at(cnt, cells[:, k], 1.0)
    return out / onp.maximum(cnt, 1.0)[:, None]


# ── Relaxation ───────────────────────────────────────────────────────────────

@dataclass
class Relaxation:
    """Outcome of :func:`relax` — the field plus how it got there."""

    u: onp.ndarray
    steps: int
    """Backward-Euler steps actually taken."""
    converged: bool
    """Whether the tolerance was met before ``max_steps``."""
    residual: float
    """Final per-step relative change ``‖u_k − u_{k−1}‖ / ‖u_k‖``."""


@dataclass
class StripeField:
    """A relaxed stripe field and everything needed to interpret it."""

    u: onp.ndarray
    """Raw SH field at the nodes; its zero level set is the path skeleton."""
    stripe: onp.ndarray
    """``mask · (0.5 + 0.5·tanh(2u))`` — a 0..1 stripe indicator for plotting."""
    director: onp.ndarray
    mask: onp.ndarray
    mesh: fe.Mesh
    """The physical-units mesh the fields live on."""
    stripe_period: float
    steps: int
    converged: bool
    residual: float


#: Default convergence tolerance on the per-step relative change.  Chosen from
#: measured relaxations rather than as a round number: SH saturates its
#: amplitude within ~40 steps and then anneals defects, during which the
#: relative change plateaus around 2e-3 and drifts *non-monotonically*.  Below
#: ~2e-3 the criterion may simply never fire; above ~1e-2 it fires while the
#: pattern is still forming.
DEFAULT_TOL = 5.0e-3


def relax(sh: SHSolver, director, mask, *, tol: float = DEFAULT_TOL,
          patience: int = 5, max_steps: int = 1000, min_steps: int = 20,
          seed: int = 42, seed_correlation: float = 0.0, initial=None,
          verbose: bool = False, log_every: int = 50) -> Relaxation:
    """Relax the stripe field from small noise until it stops changing.

    The initial condition is low-amplitude noise inside the mask: SH is a
    pattern-forming instability, so it needs a seed to break symmetry, and the
    director then selects which orientation grows.  ``seed`` therefore changes
    where the stripes' phase lands, not their spacing or direction.

    Parameters
    ----------
    tol : float
        Stop when the per-step relative change ``‖Δu‖ / ‖u‖`` stays below this.
        See :data:`DEFAULT_TOL` for why the default is not tighter.
    patience : int
        Consecutive steps that must satisfy ``tol`` before stopping.  This is
        not belt-and-braces: the relative change is genuinely non-monotone
        during defect annealing — a measured run sat at 3.9e-3 on step 300 and
        4.4e-3 on step 400 — so a single dip below the tolerance does not mean
        the pattern has settled.
    max_steps : int
        Hard cap.  Reached without convergence, this returns the field anyway
        with ``converged=False`` rather than raising: an under-annealed pattern
        is usually still usable, but you should know it happened.
    min_steps : int
        Never stop before this many steps, whatever the residual.  Clamped to
        ``max_steps``.  Set ``tol=0`` with ``max_steps=N`` to force exactly
        ``N`` steps.
    seed_correlation : float
        Correlation length of the initial noise, **in stripe periods**.  ``0``
        (default) is white noise.

        This is a defect knob, not a cosmetic one.  White noise nucleates the
        pattern independently everywhere, so many small grains form with
        unrelated phase and every collision between them leaves a grain
        boundary — visible downstream as paths that merge and terminate.
        Smoothing the seed first means fewer, larger grains, hence fewer
        collisions.  It cannot remove defects the geometry *requires*: where the
        director field admits no constant-pitch stripe tiling, a dislocation has
        to appear somewhere no matter how the pattern is started.

        Costs one Helmholtz filter build on the stripe mesh.
    initial : array, optional
        Start from this field instead of noise — see :func:`phase_seed`.
        ``seed`` and ``seed_correlation`` are then unused.

    Returns
    -------
    Relaxation
    """
    director = onp.asarray(director, dtype=float)
    mask = onp.asarray(mask, dtype=float)
    if director.shape != (sh.n_nodes, 2):
        raise ValueError(
            f"director has shape {director.shape}, expected ({sh.n_nodes}, 2)")
    if mask.shape != (sh.n_nodes,):
        raise ValueError(
            f"mask has shape {mask.shape}, expected ({sh.n_nodes},)")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if tol < 0.0:
        raise ValueError("tol must be >= 0")
    min_steps = min(min_steps, max_steps)

    if seed_correlation < 0.0:
        raise ValueError("seed_correlation must be >= 0")

    rng = onp.random.default_rng(seed)
    d_x = jnp.asarray(director[:, 0])
    d_y = jnp.asarray(director[:, 1])
    mask_j = jnp.asarray(mask)

    if initial is not None:
        given = onp.asarray(initial, dtype=float)
        if given.shape != (sh.n_nodes,):
            raise ValueError(
                f"initial has shape {given.shape}, expected ({sh.n_nodes},)")
        u_old = jnp.asarray(given * mask)
        return _run(sh, u_old, d_x, d_y, mask_j, tol, patience, max_steps,
                    min_steps, verbose, log_every)

    noise = rng.standard_normal(sh.n_nodes) * mask
    if seed_correlation > 0.0:
        smooth = gene.create_helmholtz_filter(
            sh.mesh_physical, radius=seed_correlation * sh.stripe_period)
        noise = onp.asarray(smooth(jnp.asarray(noise))) * mask
        # Smoothing removes most of the variance; restore the amplitude so the
        # growth phase starts from the same size whatever the correlation is.
        scale = onp.sqrt((noise[mask > 0] ** 2).mean()) if (mask > 0).any() else 0.0
        if scale > 0.0:
            noise = noise / scale
    u_old = jnp.asarray(0.05 * noise)
    return _run(sh, u_old, d_x, d_y, mask_j, tol, patience, max_steps,
                min_steps, verbose, log_every)


def _run(sh, u_old, d_x, d_y, mask_j, tol, patience, max_steps, min_steps,
         verbose, log_every):
    n = sh.n_nodes
    zero = jnp.zeros(sh.n_dofs)

    def params(u):
        return fe.TracedParams(volume_vars=(d_x, d_y, u, mask_j))

    # Each step solves A x = b(u_old).  With the stabilised split A does not
    # depend on u_old, so it is assembled and factorised once here; with the
    # Eyre split it must be rebuilt inside the loop.
    def operator(u):
        """The step matrix: main form plus, when split out, the alignment term.

        The alignment term carries no ``u_old``, so it contributes nothing to
        the right-hand side -- only the matrix has to be summed.
        """
        csr = sh.assemble_J(zero, params(u))
        data = csr.data
        if sh.assemble_J_aniso is not None:
            data = data + sh.assemble_J_aniso(zero, params(u)).data
        return data, csr.indptr, csr.indices, csr.shape[0]

    frozen = None
    if sh.stabilization is not None:
        frozen = _factorize(*operator(u_old))

    # The problem is linear in x, so R(0) = -b and one solve gives the answer:
    # no Newton iteration, and the residual is the only thing reassembled.
    def solve_step(u):
        rhs = -sh.assemble_R(zero, params(u))
        solve = frozen if frozen is not None else _factorize(*operator(u))
        return solve(rhs)[:n]

    settled = 0
    delta = float("inf")
    step_i = 0
    for step_i in range(1, max_steps + 1):
        # Re-mask every step so numerical noise cannot accumulate in the void
        # and leak back across the boundary.
        u_new = solve_step(u_old) * mask_j
        delta = float(jnp.linalg.norm(u_new - u_old)
                      / jnp.maximum(jnp.linalg.norm(u_new), 1e-30))
        u_old = u_new

        if verbose and (step_i % log_every == 0 or step_i == 1):
            arr = onp.asarray(u_new)
            print(f"  step {step_i:4d}: u ∈ [{arr.min():+.3f}, {arr.max():+.3f}], "
                  f"‖u‖₂ = {onp.linalg.norm(arr):.2f}, Δ = {delta:.2e}")

        settled = settled + 1 if delta <= tol else 0
        if step_i >= min_steps and settled >= patience:
            break

    converged = settled >= patience
    if verbose:
        print(f"  {'converged' if converged else 'STOPPED at max_steps'} after "
              f"{step_i} steps (Δ = {delta:.2e}, tol = {tol:g})")
    return Relaxation(u=onp.asarray(u_old), steps=step_i, converged=converged,
                      residual=delta)


def generate_stripes(mesh, director, mask, stripe_period: float, *,
                     epsilon: float = 1.0, gamma: float = DEFAULT_GAMMA, dt: float = 0.5,
                     tol: float = DEFAULT_TOL, patience: int = 5,
                     max_steps: int = 1000, min_steps: int = 20,
                     seed: int = 42, verbose: bool = False) -> StripeField:
    """Build a solver and relax one field — the single-layer convenience.

    For several layers, build one :class:`SHSolver` with :func:`make_sh_solver`
    and call :func:`relax` per layer instead; construction is the expensive part.
    """
    sh = make_sh_solver(mesh, stripe_period, epsilon=epsilon, gamma=gamma, dt=dt)
    rel = relax(sh, director, mask, tol=tol, patience=patience,
                max_steps=max_steps, min_steps=min_steps, seed=seed,
                verbose=verbose)
    return _as_field(sh, rel, director, mask)


def _as_field(sh, rel: Relaxation, director, mask):
    mask = onp.asarray(mask, dtype=float)
    return StripeField(
        u=rel.u,
        stripe=mask * (0.5 + 0.5 * onp.tanh(2.0 * rel.u)),
        director=onp.asarray(director, dtype=float),
        mask=mask,
        mesh=sh.mesh_physical,
        stripe_period=sh.stripe_period,
        steps=rel.steps,
        converged=rel.converged,
        residual=rel.residual,
    )


def relax_hybrid(sh: SHSolver, director, mask, *, gate_lo: float = 5.0,
                 gate_hi: float = 15.0, phase_steps: int = 10,
                 free_steps: int = 250, gamma_free: float | None = None,
                 seed: int = 42, verbose: bool = False):
    """Phase field where its pitch is right, SH where it is not.

    The phase construction delivers the requested pitch wherever the director is
    integrable, and :attr:`PhaseSeed.pitch_error` says where that is — before
    anything is relaxed.  Relaxing those regions only makes them worse, because
    SH prefers a wavelength a few percent longer and inserts dislocations to get
    it.  Elsewhere the phase field has to stretch stripes to fit, and choosing a
    wavelength is precisely what SH is for.

    So: freeze the phase where it is already right, and let SH work the rest.

    Two details decide whether this helps or hurts:

    * **The freed region starts from noise**, not from the phase field.  Seeded
      with the phase pattern, SH merely deforms what is there and ends up
      colliding two inherited stripe groups; from noise, the frozen region is
      the boundary condition and SH grows a pattern that phase-locks to it.
      Measured, freed from the phase field vs from noise: tail (|err| > 10%)
      21.4% against 14.7%.
    * **The frozen region is Dirichlet data**, re-imposed after every step, so
      it is bit-identical to the phase construction at the end.

    Measured on a real truss, 1 mm pitch, against the same part relaxed whole:

    .. code-block:: text

        method              defects   pitch median   |err| > 10%
        noise                    33         +4.41%        25.2%
        phase + 10 steps          4         +0.91%        24.5%
        hybrid                   10         +1.01%        12.2%

    The hybrid trades defects for pitch: more dislocations than the pure phase
    seed, but half the pitch tail.  Which matters depends on whether the paths
    can be cut — pick by that, not by one number.

    Parameters
    ----------
    gate_lo, gate_hi : float
        Predicted |pitch error| in percent at which a node is fully frozen and
        fully freed.  Between them the gate ramps linearly.
    phase_steps : int
        Global steps run before the gate closes, to saturate the amplitude from
        1.0 to about 1.25.  See :func:`stripes_from_result`.
    free_steps : int
        Steps for the freed region, which starts from noise and so needs enough
        to grow — comparable to a full noise-seeded run.
    gamma_free : float, optional
        Alignment strength inside the freed region; ``None`` keeps the solver's.
        Lowering it is a real design knob rather than a tuning one, because the
        anisotropy term is what gives SH a preferred direction at all.  Measured
        at the same gate: 16 gives 12 defects and a 14.6% tail, 4 gives 10 and
        12.7%, 1 gives 10 and 12.2%, and **0 gives 40 defects, 151 paths against
        111 — the labyrinthine state**, the same total path length chopped into
        far more pieces.  Requires the solver's selective anisotropy (the
        default); with it off, ``gamma`` lives in the main operator instead.

    Returns
    -------
    (Relaxation, PhaseSeed, ndarray)
        The relaxed field, the seed it came from, and the gate ``w`` (1 frozen,
        0 free) so it can be plotted over the result.
    """
    if gate_hi <= gate_lo:
        raise ValueError("gate_hi must exceed gate_lo")
    if gamma_free is not None and sh.assemble_J_aniso is None:
        raise ValueError(
            "gamma_free needs the solver's selective anisotropy; rebuild with "
            "make_sh_solver(..., selective_aniso=True)")

    mask = onp.asarray(mask, dtype=float)
    seed_field = phase_seed(sh, director, mask, verbose=verbose)
    w = onp.clip((gate_hi - onp.abs(seed_field.pitch_error))
                 / (gate_hi - gate_lo), 0.0, 1.0) * mask
    if verbose:
        solid = mask > 0.5
        print(f"  gate: {100 * (w[solid] > 0.99).mean():.1f}% frozen, "
              f"{100 * (w[solid] < 0.01).mean():.1f}% freed")

    ref = relax(sh, director, mask, initial=seed_field.u, tol=0.0,
                max_steps=phase_steps, min_steps=phase_steps).u

    d = onp.asarray(director, dtype=float)
    d = d / (onp.linalg.norm(d, axis=1, keepdims=True) + 1e-30)
    d_x, d_y = jnp.asarray(d[:, 0]), jnp.asarray(d[:, 1])
    mj, wj, refj = jnp.asarray(mask), jnp.asarray(w), jnp.asarray(ref)
    zero = jnp.zeros(sh.n_dofs)

    def params(u, m):
        return fe.TracedParams(volume_vars=(d_x, d_y, u, m))

    aniso_m = mj
    if gamma_free is not None:
        scale = w + (1.0 - w) * (gamma_free / sh.gamma)
        aniso_m = jnp.asarray(scale * mask)

    csr = sh.assemble_J(zero, params(refj, mj))
    data = csr.data
    if sh.assemble_J_aniso is not None:
        data = data + sh.assemble_J_aniso(zero, params(refj, aniso_m)).data
    solve = _factorize(data, csr.indptr, csr.indices, csr.shape[0])

    rng = onp.random.default_rng(seed)
    noise = jnp.asarray(0.05 * rng.standard_normal(sh.n_nodes) * mask)
    u = wj * refj + (1.0 - wj) * noise
    delta = float("inf")
    for step_i in range(1, free_steps + 1):
        rhs = -sh.assemble_R(zero, params(u, mj))
        u_new = wj * refj + (1.0 - wj) * (solve(rhs)[:sh.n_nodes] * mj)
        delta = float(jnp.linalg.norm(u_new - u)
                      / jnp.maximum(jnp.linalg.norm(u_new), 1e-30))
        u = u_new
        if verbose and step_i % 50 == 0:
            print(f"  step {step_i:4d}: Δ = {delta:.2e}")
    return (Relaxation(u=onp.asarray(u), steps=free_steps, converged=True,
                       residual=delta), seed_field, w)


# ── Director extraction ──────────────────────────────────────────────────────

def director_from_orientation(x1, x2, x3, *,
                              sgn_beta: float = DEFAULT_SGN_BETA):
    """Unit director ``(n_nodes, 2)`` from the libertas orientation triple.

    Maps ``(x1, x2, x3)`` to the second-order orientation tensor ``a₂`` and
    takes its principal direction.  ``sgn_beta`` must match the value used to
    build the material, or the director will not be the one the physics saw.

    The result is a *direction*, not a vector: ``d`` and ``−d`` describe the
    same fibre.  The angle is recovered as ``½·atan2(2a₁₂, a₁₁−a₂₂)``, which is
    continuous in ``a₂`` and so avoids the sign flips a naive eigenvector
    routine would introduce between neighbouring nodes.
    """
    a2 = jax.vmap(lambda a, b, c: orientation_tensor_2d(a, b, c,
                                                        sgn_beta=sgn_beta)[0])(
        jnp.asarray(x1), jnp.asarray(x2), jnp.asarray(x3))
    a2 = onp.asarray(a2)
    return _director_from_components(a2[:, 0, 0], a2[:, 1, 1], a2[:, 0, 1])


def director_from_a2(a2_vec):
    """Unit director from ``a₂`` stored column-wise as ``(a₁₁, a₂₂, a₁₂)``.

    The layout feax4d writes into its XDMF history; use
    :func:`director_from_orientation` when starting from the design fields.
    """
    a2_vec = onp.asarray(a2_vec)
    if a2_vec.ndim != 2 or a2_vec.shape[1] != 3:
        raise ValueError(
            f"a2_vec has shape {a2_vec.shape}, expected (n_nodes, 3)")
    return _director_from_components(a2_vec[:, 0], a2_vec[:, 1], a2_vec[:, 2])


def _director_from_components(a11, a22, a12):
    theta = 0.5 * onp.arctan2(2.0 * a12, a11 - a22)
    return onp.column_stack([onp.cos(theta), onp.sin(theta)])


# ── Resampling an optimisation result onto a finer mesh ──────────────────────

def resample(values, src_mesh, dst_points):
    """Linearly interpolate a nodal field onto another point set.

    Works on any 2D mesh: it triangulates the *source nodes* and interpolates
    barycentrically, with nearest-node fallback outside the convex hull.  That
    keeps it independent of element type and of feax's node ordering, unlike a
    structured-grid reshape.

    ``values`` may be ``(n_src,)`` or ``(n_src, k)``.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    src = onp.asarray(src_mesh.points)[:, :2]
    dst = onp.asarray(dst_points)[:, :2]
    vals = onp.asarray(values, dtype=float)
    if vals.shape[0] != src.shape[0]:
        raise ValueError(
            f"values has {vals.shape[0]} rows, mesh has {src.shape[0]} nodes")

    linear = LinearNDInterpolator(src, vals)
    out = linear(dst)
    # Points just outside the hull (round-off on the boundary, or a genuinely
    # larger target domain) come back NaN; fill them from the nearest node.
    missing = onp.isnan(out) if out.ndim == 1 else onp.isnan(out).any(axis=1)
    if missing.any():
        out[missing] = NearestNDInterpolator(src, vals)(dst[missing])
    return out


# ── OptimizeResult adapter ───────────────────────────────────────────────────

def stripes_from_result(result, layers, stripe_period: float, *, mesh=None,
                        rho_cutoff: float = 0.5,
                        sgn_beta: float = DEFAULT_SGN_BETA,
                        epsilon: float = 1.0, gamma: float = DEFAULT_GAMMA,
                        dt: float = 0.5, seed_from: str = "hybrid",
                        phase_steps: int = 10, free_steps: int = 250,
                        gate_lo: float = 5.0, gate_hi: float = 15.0,
                        gamma_free: float | None = None,
                        tol: float = DEFAULT_TOL,
                        patience: int = 5, max_steps: int = 1000,
                        min_steps: int = 20, seed: int = 42,
                        verbose: bool = False) -> dict[int, StripeField]:
    """Relax one stripe field per laminate layer of an optimisation result.

    Reads the design straight off :class:`~path_optimizer.OptimizeResult` — the
    fields and the mesh are already in memory, so there is no round trip through
    the history file.

    Parameters
    ----------
    result : OptimizeResult
        Its ``design`` supplies the density and orientation fields and its
        ``mesh`` the source geometry.
    layers : sequence of (rho, x1, x2, x3)
        Design-field names per layer.  Build with
        :func:`path_optimizer.objectives.group_layers` so the grouping matches
        the design space rather than being retyped.
    stripe_period : float
        Physical stripe pitch, in the mesh's units.
    mesh : feax.Mesh, optional
        Mesh to relax on.  Optimisation meshes rarely resolve the stripe pitch,
        so pass a finer one and the design is resampled onto it
        (:func:`resample`).  ``None`` uses ``result.mesh`` unchanged.
    rho_cutoff : float
        Density above which a node counts as fibre.  The resulting binary mask
        is where stripes are grown; everything else stays at zero.
    sgn_beta : float
        Must match the material's value — see :func:`director_from_orientation`.
    seed_from : {"hybrid", "phase", "noise"}
        How to build each layer.  Measured on a real truss at 6 elements per
        period, 1 mm pitch:

        .. code-block:: text

            method              defects   pitch median   |err| > 10%
            noise                    33         +4.41%        25.2%
            phase + 10 steps          4         +0.91%        24.5%
            hybrid                   10         +1.01%        12.2%

        ``"noise"`` is plain SH: every region picks its own phase, grains
        collide, and the boundaries show up downstream as paths that merge and
        stop.  ``"phase"`` integrates the director into a phase field first,
        which supplies one coherent phase — far fewer defects, and the pitch
        comes out as asked.  ``"hybrid"`` (default) keeps the phase field
        wherever its own predicted pitch is right and hands the rest to SH; see
        :func:`relax_hybrid` for why that split is the useful one and for
        ``gate_lo`` / ``gate_hi`` / ``gamma_free`` / ``free_steps``.

        The hybrid trades defects for pitch — more dislocations than the pure
        phase seed, half the pitch tail.  If the paths must not be cut, use
        ``"phase"``.
    phase_steps : int
        Steps to run after a phase seed, and deliberately small.  They exist to
        saturate the amplitude (1.0 to about 1.25), which is done by step 10;
        past that SH pulls the pattern toward its own preferred wavelength — a
        few percent longer than requested — and pays for the change in
        dislocations.  Measured: 4 defects at 10 steps, 8 at 30, 15 at 60, 18 at
        250.  The convergence criterion is no help here, since it watches the
        amplitude and so fires long after the damage starts.

    Returns
    -------
    dict
        ``{layer_index: StripeField}``, indexed by position in ``layers``.

    Notes
    -----
    Each layer is seeded differently (``seed + index``) so the two layers of a
    laminate do not come out with identical stripe phase, which would stack
    every path directly on top of the one below.
    """
    layers = tuple(tuple(g) for g in layers)
    if not layers:
        raise ValueError("need at least one layer group")
    for g in layers:
        if len(g) != 4:
            raise ValueError(
                f"each layer group must be (rho, x1, x2, x3); got {g}")
    missing = {n for g in layers for n in g if n not in result.design}
    if missing:
        raise KeyError(
            f"design has no field(s) {sorted(missing)}; "
            f"available: {sorted(result.design)}")

    target = mesh if mesh is not None else result.mesh
    resampling = mesh is not None and mesh is not result.mesh
    log = print if verbose else (lambda *a, **k: None)

    sh = make_sh_solver(target, stripe_period, epsilon=epsilon, gamma=gamma,
                        dt=dt)
    log(f"SH mesh: {sh.n_nodes} nodes, "
        f"{sh.elements_per_period:.1f} elements per stripe period")

    out: dict[int, StripeField] = {}
    for k, (rho_name, n1, n2, n3) in enumerate(layers):
        fields = [result.design[n] for n in (rho_name, n1, n2, n3)]
        if resampling:
            stacked = resample(onp.column_stack([onp.asarray(f) for f in fields]),
                               result.mesh, target.points)
            rho, x1, x2, x3 = (stacked[:, i] for i in range(4))
        else:
            rho, x1, x2, x3 = (onp.asarray(f) for f in fields)

        director = director_from_orientation(x1, x2, x3, sgn_beta=sgn_beta)
        mask = (rho > rho_cutoff).astype(float)
        log(f"layer {k} ({rho_name}): {int(mask.sum())}/{mask.size} nodes solid")
        if not mask.any():
            raise ValueError(
                f"layer {k}: no node has rho > {rho_cutoff}; the mask is empty "
                "so there is nothing to grow stripes in")

        if k == 0 and seed_from != "noise":
            # tol/patience/max_steps/min_steps are relax()'s convergence budget
            # and only the noise path runs relax() to convergence.  The phase
            # and hybrid paths run a fixed number of steps instead, so a caller
            # who sets a budget here gets neither the steps they asked for nor
            # any complaint.
            ignored = [n for n, v, d in (("tol", tol, DEFAULT_TOL),
                                         ("patience", patience, 5),
                                         ("max_steps", max_steps, 1000),
                                         ("min_steps", min_steps, 20))
                       if v != d]
            if ignored:
                warnings.warn(
                    f"{', '.join(ignored)} {'are' if len(ignored) > 1 else 'is'} "
                    f"ignored by seed_from={seed_from!r}, which runs a fixed "
                    f"{'phase_steps + free_steps' if seed_from == 'hybrid' else 'phase_steps'}"
                    " budget; pass seed_from='noise' to relax to a tolerance",
                    stacklevel=2)

        if seed_from == "hybrid":
            rel, _, _ = relax_hybrid(
                sh, director, mask, gate_lo=gate_lo, gate_hi=gate_hi,
                phase_steps=phase_steps, free_steps=free_steps,
                gamma_free=gamma_free, seed=seed + k, verbose=verbose)
        elif seed_from == "phase":
            rel = relax(sh, director, mask, initial=phase_seed(sh, director, mask).u,
                        tol=0.0, max_steps=phase_steps, min_steps=phase_steps,
                        verbose=verbose)
        elif seed_from == "noise":
            rel = relax(sh, director, mask, tol=tol, patience=patience,
                        max_steps=max_steps, min_steps=min_steps, seed=seed + k,
                        verbose=verbose)
        else:
            raise ValueError(
                "seed_from must be 'hybrid', 'phase' or 'noise', got "
                f"{seed_from!r}")
        if seed_from == "noise" and not rel.converged:
            warnings.warn(
                f"layer {k} hit max_steps={max_steps} with residual "
                f"{rel.residual:.2e} > tol={tol:g}; the pattern may still be "
                "annealing", stacklevel=2)
        out[k] = _as_field(sh, rel, director, mask)
    return out


# ── Output ───────────────────────────────────────────────────────────────────

def save_vtu(field: StripeField, path):
    """Write a :class:`StripeField` to a ParaView ``.vtu``.

    Includes the director as a 3-component vector so ParaView's Glyph filter can
    draw it next to the stripes — the quickest way to confirm the pattern really
    follows the fibre.
    """
    d3 = onp.column_stack([field.director,
                           onp.zeros(field.director.shape[0])])
    fe.utils.save_sol(field.mesh, str(path), point_infos=[
        ("u", field.u),
        ("stripe", field.stripe),
        ("mask", field.mask),
        ("director", d3),
    ])
