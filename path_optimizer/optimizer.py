"""Generic multi-field design optimisation driver (feax 0.8).

This is the ``path_optimizer`` counterpart of ``feax4d.optimize``, generalised
in two directions:

* **Multi-field design vector.**  ``feax.gene.optimizer`` drives a single
  node-based scalar density.  Here the design is *N* named node fields
  (density, fibre-orientation parameters, per-layer copies, ...), each with its
  own bounds, initial value and filter radius.  They are packed into one flat
  MMA vector of length ``n_fields × n_nodes``.

* **Caller-supplied problem.**  The driver never constructs a
  :class:`feax.Problem`.  :meth:`Pipeline.build` receives the mesh and the
  subclass creates whatever problem, boundary conditions, loads and solver it
  wants — so any feax problem (shell, elasticity, thermal, coupled) can be
  optimised through the same loop.

Layering, from the driver's point of view, of one iteration::

    x (flat)  --unpack-->  raw fields  --filter-->  filtered  --transform-->  design
                                                                                |
                                     Pipeline.objective(design, **params) <-----+
                                     Pipeline.<constraint>(design, **params) <--+

The filter is generic and lives here; the projection (SIMP / Heaviside /
whatever) is problem-specific and lives in :meth:`Pipeline.transform`, so that
the objective and every constraint see exactly the same projected design.

Example
-------

.. code-block:: python

    import feax as fe
    import path_optimizer as po

    class Compliance(po.Pipeline):
        def build(self, mesh):
            self.problem = make_my_problem(mesh)
            self.bc = fe.DirichletBCConfig(specs).create_bc(self.problem)
            sample = fe.TracedParams(
                volume_vars=(fe.TracedParams.create_node_var(self.problem, 0.5),))
            self.solver = po.make_linear_solver(self.problem, self.bc, sample)
            self.compliance = fe.gene.create_compliance_fn(self.problem)
            self.volume = fe.gene.create_volume_fn(self.problem)

        def transform(self, design, penalty=3.0):
            return {"rho": design["rho"] ** penalty}

        def objective(self, design, **params):
            sol = self.solver(fe.TracedParams(volume_vars=(design["rho"],)))
            return self.compliance(sol)

        @po.constraint(target=0.4)
        def vol(self, design, **params):
            return self.volume(design["rho"])

    space = po.DesignSpace([po.DesignField("rho", 0.0, 1.0, init=0.5, filter_frac=0.05)])
    result = po.run(Compliance(), mesh, space, max_iter=100,
                    continuations={"penalty": po.Continuation(1.0, 3.0, 30, 0.5)})

Notes on the feax 0.8 API used here (these all changed from 0.7):

* ``fe.InternalVars`` → :class:`feax.TracedParams`.
* ``create_solver(..., iter_num=1, internal_vars=…)`` →
  ``create_solver(..., linear=True, traced_params=…)``.
* Solvers return a :class:`feax.Solution`; use ``sol.field(i)`` instead of
  ``problem.unflatten_fn_sol_list(sol)[i]``.
"""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import feax as fe
import feax.gene as gene
import jax
import jax.numpy as np
import nlopt
import numpy as onp

# Re-exported so callers get the whole driver vocabulary from one module.  The
# decorator/dataclass semantics are feax 0.8's, unchanged.
from feax.gene.optimizer import Continuation, constraint

__all__ = [
    "DesignField",
    "DesignSpace",
    "BoundDesignSpace",
    "Pipeline",
    "OptimizeResult",
    "run",
    "make_linear_solver",
    "Continuation",
    "constraint",
]


# A design is a plain name → node-array mapping.  It is a JAX pytree, so it
# passes through jit/grad untouched.
Design = dict[str, jax.Array]


# ── Design space ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DesignField:
    """One node-based design field.

    Parameters
    ----------
    name : str
        Key under which the field appears in the ``design`` mapping handed to
        :meth:`Pipeline.objective`.
    lower, upper : float
        Box bounds for MMA.
    init : float
        Uniform initial value (must lie within the bounds).
    filter_radius : float, optional
        Absolute Helmholtz filter radius, in mesh units.
    filter_frac : float, optional
        Filter radius as a fraction of the domain's largest span.  Mutually
        exclusive with ``filter_radius``; if neither is given the field is
        **not** filtered.

    Fields sharing the same resolved radius share one Helmholtz filter object,
    so declaring eight fields with two distinct radii builds two filters, not
    eight.
    """

    name: str
    lower: float = 0.0
    upper: float = 1.0
    init: float = 0.5
    filter_radius: float | None = None
    filter_frac: float | None = None

    def __post_init__(self) -> None:
        if self.filter_radius is not None and self.filter_frac is not None:
            raise ValueError(
                f"field {self.name!r}: give filter_radius or filter_frac, not both"
            )
        if not (self.lower <= self.init <= self.upper):
            raise ValueError(
                f"field {self.name!r}: init={self.init} outside "
                f"[{self.lower}, {self.upper}]"
            )
        if self.lower > self.upper:
            raise ValueError(f"field {self.name!r}: lower > upper")

    def resolve_radius(self, span: float) -> float | None:
        """Absolute filter radius for a domain of characteristic size ``span``."""
        if self.filter_radius is not None:
            return float(self.filter_radius)
        if self.filter_frac is not None:
            return float(self.filter_frac) * span
        return None


class DesignSpace:
    """An ordered collection of :class:`DesignField` s.

    Mesh-independent: call :meth:`bind` to attach it to a mesh and build the
    filters, bounds and initial vector.
    """

    def __init__(self, fields: Sequence[DesignField]):
        self.fields: tuple[DesignField, ...] = tuple(fields)
        if not self.fields:
            raise ValueError("DesignSpace needs at least one field")
        names = [f.name for f in self.fields]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate design field name(s): {sorted(dupes)}")
        self.names: tuple[str, ...] = tuple(names)

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    def bind(self, mesh) -> BoundDesignSpace:
        """Attach to ``mesh``: build filters and size the flat design vector."""
        return BoundDesignSpace(self, mesh)

    def __repr__(self) -> str:
        return f"DesignSpace({list(self.names)})"


class BoundDesignSpace:
    """A :class:`DesignSpace` bound to a concrete mesh.

    Owns the flat-vector layout (``field k`` occupies
    ``x[k*n_nodes : (k+1)*n_nodes]``), the Helmholtz filters and the MMA bounds.
    """

    def __init__(self, space: DesignSpace, mesh):
        self.space = space
        self.mesh = mesh
        self.n_nodes = int(mesh.points.shape[0])
        self.n_total = self.n_nodes * space.n_fields

        pts = onp.asarray(mesh.points)
        self.span = float(max(pts[:, d].max() - pts[:, d].min()
                              for d in range(pts.shape[1])))

        # One Helmholtz filter per distinct radius, shared across fields.
        cache: dict[float, Callable] = {}
        filters: list[Callable | None] = []
        for f in space.fields:
            r = f.resolve_radius(self.span)
            if r is None:
                filters.append(None)
            else:
                if r <= 0.0:
                    raise ValueError(f"field {f.name!r}: filter radius must be > 0")
                if r not in cache:
                    cache[r] = gene.create_helmholtz_filter(mesh, radius=r)
                filters.append(cache[r])
        self.filters: tuple[Callable | None, ...] = tuple(filters)
        self.filter_radii: tuple[float | None, ...] = tuple(
            f.resolve_radius(self.span) for f in space.fields
        )
        self.n_filters = len(cache)

    # -- layout -------------------------------------------------------------

    def unpack(self, x_flat) -> Design:
        """Flat design vector → ``{name: (n_nodes,) array}`` (unfiltered)."""
        n = self.n_nodes
        return {name: x_flat[k * n:(k + 1) * n]
                for k, name in enumerate(self.space.names)}

    def pack(self, design: Mapping[str, jax.Array]):
        """``{name: array}`` → flat design vector (field order preserved)."""
        return np.concatenate([
            np.asarray(design[name]).reshape(self.n_nodes)
            for name in self.space.names
        ])

    def apply_filters(self, design: Mapping[str, jax.Array]) -> Design:
        """Helmholtz-filter each field that declared a radius."""
        out: Design = {}
        for f, filt in zip(self.space.fields, self.filters, strict=True):
            v = design[f.name]
            out[f.name] = v if filt is None else filt(v)
        return out

    # -- MMA vectors --------------------------------------------------------

    def bounds(self) -> tuple[onp.ndarray, onp.ndarray]:
        """``(lower, upper)`` box bounds for the flat design vector."""
        lo = onp.empty(self.n_total)
        hi = onp.empty(self.n_total)
        n = self.n_nodes
        for k, f in enumerate(self.space.fields):
            lo[k * n:(k + 1) * n] = f.lower
            hi[k * n:(k + 1) * n] = f.upper
        return lo, hi

    def initial(self) -> onp.ndarray:
        """Uniform initial flat design vector from each field's ``init``."""
        x0 = onp.empty(self.n_total)
        n = self.n_nodes
        for k, f in enumerate(self.space.fields):
            x0[k * n:(k + 1) * n] = f.init
        return x0


# ── Solver helper ────────────────────────────────────────────────────────────

def make_linear_solver(
    problem,
    bc,
    sample_params: fe.TracedParams | None = None,
    *,
    solver_options=None,
    traced_structure: bool = True,
    verbose: bool = False,
):
    """A differentiable linear solve ``traced_params -> feax.Solution``.

    Thin wrapper over :func:`feax.create_solver` pinning the feax 0.8 spelling
    that the driver expects: single linear solve (``linear=True``), the same
    options reused for the adjoint, and ``return_solution=True`` so callers get
    a :class:`feax.Solution` (use ``sol.field(i)``).

    ``sample_params`` is a shape-correct :class:`feax.TracedParams` used to
    pre-warm the direct solver / auto solver selection.  It is required when
    the solver is ``"auto"`` (the default) or cuDSS.

    With ``traced_structure=True`` (default) a :class:`feax.TracedStructure` is
    built from ``problem`` and bound into *both* the construction and every
    subsequent call — feax 0.8 deprecates the closure assembly path that you
    otherwise land on, and building the structure releases the problem's
    host-side assembly scratch.  Pass ``traced_structure=False`` if the same
    ``problem`` is also used for a ``get_jacobian`` / non-TracedStructure
    assembly (e.g. linear buckling), which still needs those arrays.
    """
    opts = solver_options or fe.DirectSolverOptions(solver="auto", verbose=verbose)
    ts = fe.TracedStructure.from_problem(problem) if traced_structure else None
    solver = fe.create_solver(
        problem,
        bc=bc,
        solver_options=opts,
        adjoint_solver_options=opts,
        linear=True,
        traced_params=sample_params,
        traced_structure=ts,
        return_solution=True,
    )
    if ts is None:
        return solver

    def solve(traced_params, *args, **kwargs):
        kwargs.setdefault("traced_structure", ts)
        return solver(traced_params, *args, **kwargs)

    return solve


# ── Pipeline ─────────────────────────────────────────────────────────────────

class Pipeline(ABC):
    """User-supplied definition of one optimisation problem.

    Subclass and implement :meth:`build` and :meth:`objective`.  Optionally
    override :meth:`transform` (design projection) and :meth:`snapshot`
    (ParaView output), and decorate extra methods with :func:`constraint`.

    Every hook that sees a ``design`` receives the **filtered and transformed**
    fields — objective, constraints and snapshot are therefore always
    consistent with each other.

    Continuation parameters (see :class:`Continuation`) arrive as keyword
    arguments on :meth:`transform`, :meth:`objective` and the constraint
    methods; accept ``**params`` if you do not use them.
    """

    #: When True, :meth:`objective` returns ``(loss, aux)`` where ``aux`` is a
    #: mapping of scalar diagnostics.  They are logged, written to
    #: ``history.csv`` and forwarded to :meth:`snapshot`.
    has_aux: bool = False

    @abstractmethod
    def build(self, mesh) -> None:
        """Create every mesh-dependent object: problem, BCs, solver, responses.

        Called once by :func:`run` before the loop starts.  Store what the other
        hooks need as instance attributes.
        """

    @abstractmethod
    def objective(self, design: Design, **params):
        """Scalar loss to minimise (or ``(loss, aux)`` when :attr:`has_aux`)."""

    def transform(self, design: Design, **params) -> Design:
        """Derive extra fields before the design reaches objective/constraints.

        The place for SIMP (``rho ** p``), Heaviside projection, or any
        remapping whose continuation parameters should be shared by every
        response.  Default: identity.

        **Add, do not replace.**  A SIMP exponent belongs on the *stiffness*,
        not on the material budget: a volume constraint reading ``rho ** p``
        instead of ``rho`` lets the true material fraction drift far above the
        target (at ``p = 3`` a projected 0.5 is a physical 0.79) and the design
        never resolves.  Return the physical field alongside the projected one
        and let each response pick what it needs::

            def transform(self, design, penalty=3.0, **params):
                return {**design, "simp": design["rho"] ** penalty}

            def objective(self, design, **params):      # stiffness -> projected
                return compliance(solve(design["simp"]))

            @constraint(target=0.4)
            def volfrac(self, design, **params):        # budget -> physical
                return volume(design["rho"])
        """
        return design

    def snapshot(self, design: Design, aux=None) -> list[tuple[str, onp.ndarray]]:
        """Fields written to the XDMF/VTU history, as ``[(name, array), ...]``.

        Default: every design field.  Names and shapes must stay constant
        across iterations (an XDMF time-series requirement).
        """
        return [(k, onp.asarray(v)) for k, v in design.items()]


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass
class OptimizeResult:
    """Outcome of :func:`run`."""

    x_opt: onp.ndarray
    """Raw (unfiltered) flat design vector at the optimum."""
    design: Design
    """Filtered + transformed design fields at the optimum."""
    history: dict[str, list]
    final_obj: float
    """Objective of the design actually returned, re-evaluated at ``x_opt`` with
    the final continuation values.  **This is the number to report.**  Neither
    ``best_obj`` nor ``history["obj"][-1]`` is it: NLopt returns its best point,
    not its last trial, and MMA's last few trials are often wild excursions."""
    final_aux: dict[str, float] | None
    """``objective``'s aux diagnostics at ``x_opt`` (``None`` unless
    :attr:`Pipeline.has_aux`).  Same caveat as :attr:`final_obj` — do not read
    the last row of ``history`` instead."""
    best_obj: float
    """Lowest objective over all *trial* points.  Rarely what you want: with
    continuations, values from different levels are not comparable, so the
    running minimum usually lands on the softest early level."""
    best_iter: int
    n_iters: int
    stop_reason: str
    mesh: object
    space: BoundDesignSpace
    pipeline: Pipeline
    final_constraints: dict[str, float]
    xdmf_path: Path | None
    output_dir: Path | None


_NLOPT_REASONS = {
    nlopt.SUCCESS: "SUCCESS",
    nlopt.STOPVAL_REACHED: "STOPVAL_REACHED",
    nlopt.FTOL_REACHED: "FTOL_REACHED (objective settled)",
    nlopt.XTOL_REACHED: "XTOL_REACHED (design settled)",
    nlopt.MAXEVAL_REACHED: "MAXEVAL_REACHED (cap hit — not converged)",
    nlopt.MAXTIME_REACHED: "MAXTIME_REACHED",
}


def _collect_constraints(pipeline: Pipeline):
    """Discover ``@constraint``-decorated methods on ``pipeline``.

    Mirrors ``feax.gene.optimizer._collect_constraints`` (which is private and
    single-argument): equality constraints expand into an ``le`` / ``ge`` pair.
    """
    found = []
    for name in dir(type(pipeline)):
        attr = getattr(type(pipeline), name, None)
        meta = getattr(attr, "_constraint_meta", None)
        if not (callable(attr) and meta):
            continue
        bound = getattr(pipeline, name)
        if meta["type"] == "eq":
            tol = meta.get("tol", 0.0)
            found.append((name, bound, meta["target"] + tol, "le"))
            found.append((f"{name}_lb", bound, meta["target"] - tol, "ge"))
        else:
            found.append((name, bound, meta["target"], meta["type"]))
    return found


# ── Driver ───────────────────────────────────────────────────────────────────

def run(
    pipeline: Pipeline,
    mesh,
    space: DesignSpace,
    *,
    max_iter: int | None = 200,
    continuations: Mapping[str, Continuation] | None = None,
    x_init: onp.ndarray | None = None,
    ftol_rel: float = 1e-5,
    xtol_rel: float = 1e-5,
    xtol_abs: float = 1e-5,
    output_dir: Path | None = None,
    write_xdmf: bool = True,
    save_vtu: bool = False,
    snapshot_every: int = 1,
    jit: bool = True,
    verbose: bool = True,
) -> OptimizeResult:
    """Run the NLopt/MMA loop for ``pipeline`` on ``mesh``.

    Parameters
    ----------
    pipeline : Pipeline
        Problem definition.  :meth:`Pipeline.build` is called once, here.
    mesh : feax.Mesh
        Any feax mesh; the design fields live on its nodes.
    space : DesignSpace
        The multi-field design description; bound to ``mesh`` internally.
    max_iter : int, optional
        Objective-evaluation cap.  ``None`` runs until a tolerance triggers
        (not allowed together with ``continuations``, which needs epochs).
    continuations : mapping, optional
        ``{param_name: Continuation(...)}``.  Values are passed as keyword
        arguments to :meth:`Pipeline.transform`, :meth:`Pipeline.objective` and
        every constraint method.  MMA is restarted whenever a value changes.
    x_init : ndarray, optional
        Initial flat design vector.  Default: each field's ``init``.
    ftol_rel, xtol_rel, xtol_abs : float
        NLopt stopping tolerances — the loop stops when **any** is met.
    output_dir : Path, optional
        Where ``history.xdmf`` / ``history.csv`` / ``history.png`` are written.
        ``None`` disables all file output.
    write_xdmf : bool
        Write the ParaView XDMF time series (one frame per snapshot).
    save_vtu : bool
        Additionally write per-iteration ``.vtu`` files under ``vtu/``.
    snapshot_every : int
        Snapshot cadence, in objective evaluations.
    jit : bool
        JIT-compile the objective and constraints.
    verbose : bool
        Per-iteration logging.

    Returns
    -------
    OptimizeResult
    """
    log = print if verbose else (lambda *a, **k: None)

    continuations = dict(continuations or {})
    if continuations and max_iter is None:
        raise ValueError("continuations require a finite max_iter (epoch boundaries)")

    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    bound = space.bind(mesh)
    pipeline.build(mesh)

    constraints = _collect_constraints(pipeline)
    constraint_names = [c[0] for c in constraints]

    params: dict[str, float] = {k: c.initial for k, c in continuations.items()}

    # -- Traced chain: x → unpack → filter → transform → responses ------------

    def _design_of(x_flat, **p):
        return pipeline.transform(bound.apply_filters(bound.unpack(x_flat)), **p)

    def _objective_raw(x_flat, **p):
        return pipeline.objective(_design_of(x_flat, **p), **p)

    wrap = jax.jit if jit else (lambda f: f)
    obj_and_grad = wrap(jax.value_and_grad(_objective_raw, has_aux=pipeline.has_aux))

    # One value_and_grad per constraint, not a separate value and grad: NLopt
    # wants both at the same point and the history log wants the value again,
    # so computing them together is what makes the cache below a single
    # evaluation per iteration.
    compiled_constraints = []
    for name, method, target, ctype in constraints:
        def _make(method=method):
            def _fn(x_flat, **p):
                return method(_design_of(x_flat, **p), **p)
            return _fn
        compiled_constraints.append(
            (name, wrap(jax.value_and_grad(_make())), target, ctype)
        )

    design_at = wrap(_design_of)

    # -- Initial design -------------------------------------------------------

    lower, upper = bound.bounds()
    x = onp.array(bound.initial() if x_init is None else onp.asarray(x_init, float))
    if x.shape != (bound.n_total,):
        raise ValueError(
            f"x_init has shape {x.shape}, expected ({bound.n_total},) "
            f"= {space.n_fields} fields × {bound.n_nodes} nodes"
        )
    x = onp.clip(x, lower, upper)

    # -- Header ---------------------------------------------------------------

    log(f"Mesh         : {bound.n_nodes} nodes, {mesh.cells.shape[0]} cells")
    log(f"Design vars  : {bound.n_total}  "
        f"({space.n_fields} fields × {bound.n_nodes} nodes)")
    log("Fields       : " + ", ".join(
        f"{f.name}[{f.lower:g},{f.upper:g}]"
        + (f" r={r:.4g}" if r is not None else " (unfiltered)")
        for f, r in zip(space.fields, bound.filter_radii, strict=True)))
    log(f"Filters      : {bound.n_filters} Helmholtz solve(s) shared across fields")
    if compiled_constraints:
        for name, _, target, ctype in compiled_constraints:
            log(f"Constraint   : {name} {'<=' if ctype == 'le' else '>='} {target}")
    else:
        log("Constraints  : none")
    for k, c in continuations.items():
        log(f"Continuation : {k} ({c.initial} -> {c.final}, "
            f"{c.step:+g} every {c.update_every} iter)")
    log(f"Convergence  : ftol_rel={ftol_rel:g}  xtol_rel={xtol_rel:g}  "
        f"xtol_abs={xtol_abs:g}  (max-iter cap = {max_iter})")
    log("-" * 68)

    # -- History / output -----------------------------------------------------

    history: dict[str, list] = {"iter": [], "obj": [],
                                **{n: [] for n in constraint_names},
                                **{k: [] for k in continuations}}
    aux_keys: list[str] = []

    iter_count = [0]
    best = {"obj": float("inf"), "iter": 0}

    vtu_dir = None
    if out is not None and save_vtu:
        vtu_dir = out / "vtu"
        vtu_dir.mkdir(parents=True, exist_ok=True)

    xdmf_path = out / "history.xdmf" if (out is not None and write_xdmf) else None
    writer = fe.XDMFWriter(mesh, xdmf_path) if xdmf_path is not None else None

    def _write_frame(step: int, x_arr, aux=None):
        if writer is None and vtu_dir is None:
            return
        design = design_at(np.asarray(x_arr), **_jax_params())
        fields = pipeline.snapshot({k: onp.asarray(v) for k, v in design.items()},
                                   aux)
        if writer is not None:
            writer.write_iteration(step, point_infos=fields)
        if vtu_dir is not None:
            fe.utils.save_sol(mesh, str(vtu_dir / f"iter_{step:04d}.vtu"),
                              point_infos=fields)

    def _jax_params():
        # Arrays, not Python floats, so changing a continuation value does not
        # retrigger JAX compilation.
        return {k: np.asarray(v, dtype=np.float64 if fe.x64_enabled() else np.float32)
                for k, v in params.items()}

    # -- Constraint cache -----------------------------------------------------

    # MMA evaluates the objective and every constraint at the same point within
    # an iteration, and the history log wants the constraint values too.  Without
    # a cache that is three passes per constraint per iteration (log value, NLopt
    # value, NLopt gradient); cheap for a volume fraction, but ruinous for a
    # constraint that needs its own FE solve.  Memoise on (x, continuation
    # params) so each constraint is evaluated exactly once per iteration.
    _con_cache: dict[str, object] = {"x": None, "params": None, "out": {}}

    def _constraints_at(xx) -> dict[str, tuple]:
        """``{name: (value, gradient)}`` at ``xx``, computed once per point."""
        if not compiled_constraints:
            return {}
        pkey = tuple(params[k] for k in continuations)
        if (_con_cache["x"] is not None
                and _con_cache["params"] == pkey
                and onp.array_equal(_con_cache["x"], xx)):
            return _con_cache["out"]
        x_jax = np.asarray(xx)
        jp = _jax_params()
        out = {name: vg(x_jax, **jp) for name, vg, _, _ in compiled_constraints}
        # Copy xx: NLopt hands back a view it is free to overwrite in place.
        _con_cache.update(x=onp.array(xx), params=pkey, out=out)
        return out

    # -- MMA ------------------------------------------------------------------

    def _make_opt() -> nlopt.opt:
        opt = nlopt.opt(nlopt.LD_MMA, bound.n_total)
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)
        opt.set_ftol_rel(ftol_rel)
        opt.set_xtol_rel(xtol_rel)
        opt.set_xtol_abs(xtol_abs)

        def _obj(xx, grad):
            x_jax = np.asarray(xx)
            jp = _jax_params()
            out_val, g = obj_and_grad(x_jax, **jp)
            if pipeline.has_aux:
                val, aux = out_val
            else:
                val, aux = out_val, None
            if grad.size > 0:
                grad[:] = onp.asarray(g)
            val_f = float(val)

            con_vals = {name: float(v)
                        for name, (v, _) in _constraints_at(xx).items()}

            iter_count[0] += 1
            if val_f < best["obj"]:
                best["obj"], best["iter"] = val_f, iter_count[0]

            history["iter"].append(iter_count[0])
            history["obj"].append(val_f)
            for name in constraint_names:
                history[name].append(con_vals.get(name, float("nan")))
            for k in continuations:
                history[k].append(params[k])
            if aux is not None:
                for k, v in aux.items():
                    if k not in history:
                        aux_keys.append(k)
                        # Back-fill so every column has one row per iteration.
                        history[k] = [float("nan")] * (iter_count[0] - 1)
                    history[k].append(float(v))

            extra = "  ".join(f"{k}={con_vals[k]:.4f}" for k in constraint_names)
            aux_str = ("  " + "  ".join(f"{k}={float(aux[k]):.4g}" for k in aux_keys)
                       if aux is not None else "")
            log(f"Iter {iter_count[0]:4d}: obj={val_f:.4e}  {extra}{aux_str}  "
                f"best={best['obj']:.4e}")

            if iter_count[0] % snapshot_every == 0:
                _write_frame(iter_count[0], xx, aux)
            return val_f

        opt.set_min_objective(_obj)

        for name, _, target, ctype in compiled_constraints:
            def _make_con(name=name, target=target,
                          sign=(1.0 if ctype == "le" else -1.0)):
                def _con(xx, grad):
                    # Served from the cache the objective call already filled.
                    val, g = _constraints_at(xx)[name]
                    if grad.size > 0:
                        grad[:] = sign * onp.asarray(g)
                    return sign * (float(val) - target)
                return _con
            opt.add_inequality_constraint(_make_con(), 1e-8)
        return opt

    # -- Epoch loop -----------------------------------------------------------

    if writer is not None:
        writer.__enter__()
    try:
        _write_frame(0, x)

        stop_reason = "NOT_RUN"
        need_new_opt = True
        opt = None

        # Continuation values are looked up on their own clock, which normally
        # tracks iter_count.  When an epoch converges early the clock is pushed
        # forward to the next boundary, so the ramp is driven by convergence
        # rather than by burning the full iteration allowance at each level.
        sched_offset = 0

        def _sched() -> int:
            return iter_count[0] + sched_offset

        def _next_boundary() -> int | None:
            """Next scheduling index at which some continuation value changes."""
            s = _sched()
            nxt = None
            for c in continuations.values():
                b = ((s // c.update_every) + 1) * c.update_every
                if c.value_at(b) != c.value_at(s):
                    nxt = b if nxt is None else min(nxt, b)
            return nxt

        while True:
            for k, c in continuations.items():
                new = c.value_at(_sched())
                if new != params[k]:
                    params[k] = new
                    need_new_opt = True     # MMA's model is stale after a jump
                    log(f"  >>> {k} = {new:.4g}")

            if need_new_opt:
                opt = _make_opt()
                need_new_opt = False

            # Run to the next continuation boundary, or to the cap.
            if max_iter is None:
                budget = None
            else:
                remaining = max_iter - iter_count[0]
                if remaining <= 0:
                    break
                boundary = _next_boundary()
                budget = (remaining if boundary is None
                          else min(remaining, boundary - _sched()))
                opt.set_maxeval(budget)

            try:
                x = opt.optimize(x)
            except nlopt.RoundoffLimited:
                log("  NLopt: roundoff limit (treated as converged)")
                stop_reason = "ROUNDOFF_LIMITED"
                break

            code = opt.last_optimize_result()
            stop_reason = _NLOPT_REASONS.get(code, str(code))
            if max_iter is None or iter_count[0] >= max_iter:
                break

            if code in (nlopt.FTOL_REACHED, nlopt.XTOL_REACHED,
                        nlopt.STOPVAL_REACHED):
                boundary = _next_boundary()
                if boundary is None:
                    break               # fully ramped and settled — really done
                # This continuation level has settled; skip ahead to the next
                # one instead of stopping short of the final value.
                sched_offset += boundary - _sched()
                log(f"  >>> {stop_reason} at this continuation level; advancing")

        log("-" * 68)
        log(f"Done ({stop_reason}; {iter_count[0]} iters).  "
            f"best obj = {best['obj']:.6e} @ iter {best['iter']}")

        _write_frame(iter_count[0] + 1, x)
    finally:
        if writer is not None:
            writer.__exit__(None, None, None)

    # -- Final evaluation -----------------------------------------------------

    # Everything reported below is re-evaluated at the returned x_opt.  NLopt
    # hands back its best point, which is generally NOT the last one it
    # evaluated, so the tail of `history` describes a different design.
    jp = _jax_params()
    x_jax = np.asarray(x)
    final_design = {k: onp.asarray(v) for k, v in design_at(x_jax, **jp).items()}
    # Reuse obj_and_grad rather than jitting a grad-free variant: the gradient
    # is thrown away, but this is one call against a function that is already
    # compiled, instead of paying a second XLA compilation for one evaluation.
    final_out, _ = obj_and_grad(x_jax, **jp)
    if pipeline.has_aux:
        final_val, aux = final_out
        final_aux = {k: float(v) for k, v in aux.items()}
    else:
        final_val, final_aux = final_out, None
    final_obj = float(final_val)
    final_constraints = {name: float(v)
                         for name, (v, _) in _constraints_at(x).items()}
    log(f"Final objective : {final_obj:.6e}")
    for name, val in final_constraints.items():
        log(f"Final {name:12s}: {val:.6f}")

    if out is not None:
        _write_history_csv(out / "history.csv", history,
                           ["iter", "obj"] + constraint_names + aux_keys
                           + list(continuations))
        _save_history_plot(out / "history.png", history, constraint_names, aux_keys)
        log(f"Wrote {out}/history.csv"
            + (f", {xdmf_path.name}" if xdmf_path is not None else ""))

    return OptimizeResult(
        x_opt=onp.asarray(x),
        design=final_design,
        history=history,
        final_obj=final_obj,
        final_aux=final_aux,
        best_obj=best["obj"],
        best_iter=best["iter"],
        n_iters=iter_count[0],
        stop_reason=stop_reason,
        mesh=mesh,
        space=bound,
        pipeline=pipeline,
        final_constraints=final_constraints,
        xdmf_path=xdmf_path,
        output_dir=out,
    )


# ── Output helpers ───────────────────────────────────────────────────────────

def _write_history_csv(path: Path, history: dict[str, list], cols: Sequence[str]) -> None:
    n = len(history["iter"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            row = []
            for c in cols:
                col = history.get(c, [])
                row.append(col[i] if i < len(col) else "")
            w.writerow(row)


def _decorate(ax, title: str, ylabel: str, legend: bool = False) -> None:
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend()


def _save_history_plot(path: Path, history, constraint_names, aux_keys) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    panels = 1 + bool(constraint_names) + bool(aux_keys)
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4), squeeze=False)
    ax = axes[0]

    it = history["iter"]
    obj = history["obj"]
    k = 0
    if all(v > 0 for v in obj):
        ax[k].semilogy(it, obj, "b-")
    else:
        ax[k].plot(it, obj, "b-")
    _decorate(ax[k], "Objective", "objective")
    k += 1

    for names, title in ((constraint_names, "Constraints"), (aux_keys, "Diagnostics")):
        if not names:
            continue
        for name in names:
            ax[k].plot(it, history[name], label=name)
        _decorate(ax[k], title, "value", legend=True)
        k += 1

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
