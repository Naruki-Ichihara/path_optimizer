"""Tests for the multi-field optimisation driver."""
import numpy as onp
import pytest

import path_optimizer as po

# ── DesignField / DesignSpace (no mesh, no FE) ───────────────────────────────

def test_design_field_rejects_both_radius_specs():
    with pytest.raises(ValueError, match="not both"):
        po.DesignField("rho", filter_radius=0.1, filter_frac=0.05)


def test_design_field_rejects_init_outside_bounds():
    with pytest.raises(ValueError, match="outside"):
        po.DesignField("rho", lower=0.0, upper=1.0, init=1.5)


def test_design_space_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        po.DesignSpace([po.DesignField("rho"), po.DesignField("rho")])


def test_design_space_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        po.DesignSpace([])


# ── BoundDesignSpace (needs a mesh only) ─────────────────────────────────────

@pytest.fixture(scope="module")
def mesh():
    fe = pytest.importorskip("feax")
    return fe.mesh.rectangle_mesh(Nx=4, Ny=2, domain_x=2.0, domain_y=1.0,
                                  ele_type="QUAD4")


@pytest.fixture(scope="module")
def space():
    # Three fields, two distinct radii (rho / x1 share one), one unfiltered.
    return po.DesignSpace([
        po.DesignField("rho", lower=0.0, upper=1.0, init=0.5, filter_frac=0.1),
        po.DesignField("x1", lower=-1.0, upper=1.0, init=-0.9, filter_frac=0.1),
        po.DesignField("x3", lower=-1.0, upper=1.0, init=0.0),
    ])


def test_bound_space_layout(mesh, space):
    b = space.bind(mesh)
    assert b.n_nodes == mesh.points.shape[0]
    assert b.n_total == 3 * b.n_nodes
    assert b.span == pytest.approx(2.0)


def test_filters_are_shared_by_radius(mesh, space):
    b = space.bind(mesh)
    # rho and x1 declare the same frac -> one Helmholtz solver, x3 unfiltered.
    assert b.n_filters == 1
    assert b.filters[0] is b.filters[1]
    assert b.filters[2] is None
    assert b.filter_radii == (0.2, 0.2, None)


def test_pack_unpack_roundtrip(mesh, space):
    b = space.bind(mesh)
    x = onp.arange(b.n_total, dtype=float)
    fields = b.unpack(x)
    assert list(fields) == ["rho", "x1", "x3"]
    assert all(v.shape == (b.n_nodes,) for v in fields.values())
    onp.testing.assert_allclose(onp.asarray(b.pack(fields)), x)


def test_bounds_and_initial_are_per_field(mesh, space):
    b = space.bind(mesh)
    lo, hi = b.bounds()
    n = b.n_nodes
    assert (lo[:n] == 0.0).all() and (hi[:n] == 1.0).all()          # rho
    assert (lo[n:2 * n] == -1.0).all() and (hi[n:2 * n] == 1.0).all()  # x1
    x0 = b.initial()
    assert (x0[:n] == 0.5).all()
    assert (x0[n:2 * n] == -0.9).all()
    assert (x0[2 * n:] == 0.0).all()


def test_unfiltered_field_passes_through(mesh, space):
    b = space.bind(mesh)
    raw = b.unpack(onp.linspace(0.0, 1.0, b.n_total))
    out = b.apply_filters(raw)
    onp.testing.assert_allclose(onp.asarray(out["x3"]), onp.asarray(raw["x3"]))


# ── End-to-end: SIMP compliance minimisation ─────────────────────────────────

@pytest.fixture(scope="module")
def compliance_case():
    """A 2D cantilever pipeline + its mesh, built once for the e2e tests."""
    fe = pytest.importorskip("feax")
    gene = pytest.importorskip("feax.gene")
    import jax.numpy as jnp

    E0, NU, EMIN = 1.0, 0.3, 1e-6
    LX, LY, NX, NY = 2.0, 1.0, 16, 8

    class Elasticity(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, rho):
                E = EMIN + rho * (E0 - EMIN)
                mu = E / (2.0 * (1.0 + NU))
                lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
                eps = 0.5 * (u_grad + u_grad.T)
                return lam * jnp.trace(eps) * jnp.eye(2) + 2.0 * mu * eps
            return stress

        def get_surface_maps(self):
            return [lambda u, x, *iv: jnp.array([0.0, -1.0])]

    class Compliance(po.Pipeline):
        has_aux = True

        def build(self, mesh):
            tol = 1e-6

            def left(p):
                return jnp.isclose(p[0], 0.0, atol=tol)

            def tip(p):
                return jnp.isclose(p[0], LX, atol=tol) & (p[1] < LY / NY + tol)

            self.problem = Elasticity(mesh=mesh, vec=2, dim=2, ele_type="QUAD4",
                                      location_fns=[tip])
            self.bc = fe.DirichletBCConfig([
                fe.DirichletBCSpec(location=left, component="all", value=0.0),
            ]).create_bc(self.problem)
            sample = fe.TracedParams(
                volume_vars=(fe.TracedParams.create_node_var(self.problem, 0.5),))
            self.solver = po.make_linear_solver(self.problem, self.bc, sample)
            self.compliance = gene.create_compliance_fn(self.problem)
            self.volume = gene.create_volume_fn(self.problem)

        def transform(self, design, penalty=3.0, **params):
            return {"rho": design["rho"] ** penalty}

        def objective(self, design, **params):
            sol = self.solver(fe.TracedParams(volume_vars=(design["rho"],)))
            return self.compliance(sol), {"vol": self.volume(design["rho"])}

        @po.constraint(target=0.4)
        def volfrac(self, design, **params):
            return self.volume(design["rho"])

    mesh = fe.mesh.rectangle_mesh(Nx=NX, Ny=NY, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")
    space = po.DesignSpace([
        po.DesignField("rho", lower=1e-3, upper=1.0, init=0.4, filter_frac=0.05),
    ])
    return Compliance, mesh, space


@pytest.fixture(scope="module")
def compliance_result(compliance_case, tmp_path_factory):
    Compliance, mesh, space = compliance_case
    return po.run(
        Compliance(), mesh, space,
        max_iter=10,
        continuations={"penalty": po.Continuation(1.0, 3.0, 5, 1.0)},
        output_dir=tmp_path_factory.mktemp("compliance"),
        verbose=False,
    )


def test_objective_decreases(compliance_result):
    obj = compliance_result.history["obj"]
    assert len(obj) == 10
    assert obj[-1] < obj[0]


def test_volume_constraint_is_respected(compliance_result):
    # MMA allows a small violation; 1e-8 is the registered constraint tolerance.
    assert compliance_result.final_constraints["volfrac"] <= 0.4 + 1e-4


def test_result_reports_the_transformed_design(compliance_result):
    design = compliance_result.design
    assert list(design) == ["rho"]
    assert design["rho"].shape == (compliance_result.space.n_nodes,)
    # transform() applies SIMP, so values stay inside the raw bounds.
    assert design["rho"].min() >= 0.0 and design["rho"].max() <= 1.0


def test_aux_diagnostics_land_in_history(compliance_result):
    assert "vol" in compliance_result.history
    assert len(compliance_result.history["vol"]) == 10


def test_final_values_describe_the_returned_design(compliance_result):
    """final_obj / final_aux are re-evaluated at x_opt, so they must agree with
    final_constraints — which the tail of `history` need not, since NLopt
    returns its best point rather than its last trial."""
    assert compliance_result.final_aux is not None
    assert compliance_result.final_aux["vol"] == pytest.approx(
        compliance_result.final_constraints["volfrac"])
    assert compliance_result.final_obj > 0.0


def test_continuation_advances_and_is_logged(compliance_result):
    import csv
    rows = list(csv.DictReader(
        open(compliance_result.output_dir / "history.csv")))
    assert len(rows) == 10
    penalty = [float(r["penalty"]) for r in rows]
    # 10 iterations, +1 every 5 -> starts at 1.0 and steps up at least once.
    assert penalty[0] == 1.0
    assert penalty[-1] > penalty[0]
    assert penalty == sorted(penalty)          # monotone, never regresses


def test_output_files_written(compliance_result):
    out = compliance_result.output_dir
    for name in ("history.csv", "history.xdmf", "history.png"):
        assert (out / name).exists(), name


def test_constraints_are_evaluated_once_per_iteration(compliance_case):
    """MMA asks for the constraint value and its gradient, and the history log
    wants the value too — all three must come from a single evaluation.

    Run with ``jit=False`` so the constraint body executes on every call and can
    actually be counted; under jit it is traced once and the count would say
    nothing.  Before the cache this cost three evaluations per iteration.
    """
    Compliance, mesh, space = compliance_case
    calls = []

    class Counting(Compliance):
        @po.constraint(target=0.4)
        def volfrac(self, design, **params):
            calls.append(1)
            return self.volume(design["rho"])

    n_iter = 5
    result = po.run(Counting(), mesh, space, max_iter=n_iter, jit=False,
                    output_dir=None, write_xdmf=False, verbose=False)

    assert result.n_iters == n_iter
    # n_iter evaluations in the loop, plus one at x_opt for final_constraints.
    assert len(calls) <= n_iter + 1, (
        f"{len(calls)} constraint evaluations for {n_iter} iterations — "
        "the per-point cache is not being hit")


def test_constraint_cache_serves_repeated_queries(compliance_case):
    """Two constraints reading the same design must not multiply the work: the
    cached (value, gradient) pair is reused across the objective's log, NLopt's
    value query and NLopt's gradient query."""
    Compliance, mesh, space = compliance_case

    class TwoConstraints(Compliance):
        @po.constraint(target=0.4)
        def volfrac(self, design, **params):
            return self.volume(design["rho"])

        @po.constraint(target=0.9)
        def volfrac_upper(self, design, **params):
            return self.volume(design["rho"])

    result = po.run(TwoConstraints(), mesh, space, max_iter=4,
                    output_dir=None, write_xdmf=False, verbose=False)
    # Both constraints report and both are satisfied at the returned design.
    assert set(result.final_constraints) == {"volfrac", "volfrac_upper"}
    assert result.final_constraints["volfrac"] <= 0.4 + 1e-4
    assert len(result.history["volfrac"]) == result.n_iters


def test_x_init_shape_is_validated(compliance_case):
    Compliance, mesh, space = compliance_case
    with pytest.raises(ValueError, match="expected"):
        po.run(Compliance(), mesh, space, max_iter=1,
               x_init=onp.zeros(3), output_dir=None, verbose=False)


def test_continuations_require_finite_max_iter(compliance_case):
    Compliance, mesh, space = compliance_case
    with pytest.raises(ValueError, match="finite max_iter"):
        po.run(Compliance(), mesh, space, max_iter=None,
               continuations={"penalty": po.Continuation(1.0, 3.0, 5, 1.0)},
               output_dir=None, verbose=False)
