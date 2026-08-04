"""Tests for aligned Swift-Hohenberg stripe generation."""
import numpy as onp
import pytest

from path_optimizer import stripes

fe = pytest.importorskip("feax")

# Libertas orientation triples, (x1, x2, x3).
ALIGN_X = (1.0 - 1e-2, -1.0 + 1e-2, 1.0)
ALIGN_Y = (-1.0 + 1e-2, 1.0 - 1e-2, -1.0)

LX, LY = 20e-3, 10e-3
PERIOD = 3e-3


@pytest.fixture(scope="module")
def mesh():
    """Fine enough to resolve a 3 mm stripe period (~9 elements per period)."""
    return fe.mesh.rectangle_mesh(Nx=60, Ny=30, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")


# ── Director extraction ──────────────────────────────────────────────────────

def _uniform(triple, n):
    return (onp.full(n, triple[0]), onp.full(n, triple[1]), onp.full(n, triple[2]))


@pytest.mark.parametrize("triple,axis", [(ALIGN_X, 0), (ALIGN_Y, 1)])
def test_director_from_orientation_recovers_the_axis(triple, axis):
    """A corner of the libertas box maps to a near-axis director.

    Not exactly on-axis: the triples are held ``ori_tol`` off the corner (the
    corner itself is a degenerate ``a2`` with no gradient), which leaves
    ``a12 ~ sqrt(ori_tol)/2`` and tilts the director by ~3 degrees at the 1e-2
    default.  The angle, not an exact axis, is what to assert.
    """
    d = stripes.director_from_orientation(*_uniform(triple, 5))
    assert d.shape == (5, 2)
    # A director is defined up to sign, so measure the angle to the axis.
    tilt = onp.degrees(onp.arctan2(onp.abs(d[:, 1 - axis]), onp.abs(d[:, axis])))
    assert (tilt < 4.0).all(), f"director tilted {tilt.max():.2f} deg off axis"


def test_director_is_a_unit_vector():
    d = stripes.director_from_orientation(*_uniform((0.3, -0.2, 0.5), 4))
    onp.testing.assert_allclose(onp.linalg.norm(d, axis=1), 1.0, rtol=1e-10)


def test_director_from_a2_matches_the_analytic_angle():
    # a2 for a fibre at 30 deg: a11=cos^2, a22=sin^2, a12=sin*cos.
    ang = onp.deg2rad(30.0)
    c, s = onp.cos(ang), onp.sin(ang)
    a2 = onp.array([[c * c, s * s, s * c]])
    d = stripes.director_from_a2(a2)
    assert abs(float(onp.dot(d[0], [c, s]))) == pytest.approx(1.0, abs=1e-10)


def test_director_from_a2_validates_the_layout():
    with pytest.raises(ValueError, match=r"expected \(n_nodes, 3\)"):
        stripes.director_from_a2(onp.zeros((4, 2)))


# ── resample ─────────────────────────────────────────────────────────────────

def test_resample_is_exact_for_a_linear_field(mesh):
    pts = onp.asarray(mesh.points)
    values = 3.0 * pts[:, 0] - 2.0 * pts[:, 1] + 1.0
    fine = fe.mesh.rectangle_mesh(Nx=17, Ny=11, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")
    got = stripes.resample(values, mesh, fine.points)
    want = 3.0 * onp.asarray(fine.points)[:, 0] - 2.0 * onp.asarray(fine.points)[:, 1] + 1.0
    onp.testing.assert_allclose(got, want, atol=1e-10)


def test_resample_handles_multiple_columns(mesh):
    pts = onp.asarray(mesh.points)
    values = onp.column_stack([pts[:, 0], pts[:, 1], onp.ones(pts.shape[0])])
    fine = fe.mesh.rectangle_mesh(Nx=9, Ny=5, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")
    got = stripes.resample(values, mesh, fine.points)
    assert got.shape == (fine.points.shape[0], 3)
    onp.testing.assert_allclose(got[:, 2], 1.0, atol=1e-10)


def test_resample_falls_back_outside_the_hull(mesh):
    """Points beyond the source domain must get a finite nearest-node value,
    not the NaN a linear interpolator returns there."""
    values = onp.asarray(mesh.points)[:, 0]
    outside = onp.array([[-LX, -LY], [2 * LX, 2 * LY]])
    got = stripes.resample(values, mesh, outside)
    assert onp.isfinite(got).all()


def test_resample_validates_the_field_length(mesh):
    with pytest.raises(ValueError, match="rows, mesh has"):
        stripes.resample(onp.zeros(3), mesh, mesh.points)


# ── Solver construction ──────────────────────────────────────────────────────

def test_make_sh_solver_reports_the_resolution(mesh):
    sh = stripes.make_sh_solver(mesh, PERIOD)
    # 20 mm / 60 elements = 0.333 mm; 3 mm / 0.333 mm = 9.
    assert sh.elements_per_period == pytest.approx(9.0, rel=1e-6)
    assert sh.n_nodes == mesh.points.shape[0]


def test_make_sh_solver_warns_when_under_resolved(mesh):
    with pytest.warns(UserWarning, match="degrades into noise"):
        stripes.make_sh_solver(mesh, 1e-3)      # 3 elements per period


def test_make_sh_solver_rejects_a_bad_period(mesh):
    with pytest.raises(ValueError, match="must be > 0"):
        stripes.make_sh_solver(mesh, 0.0)


# ── Relaxation: shape guards ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sh(mesh):
    return stripes.make_sh_solver(mesh, PERIOD)


def test_relax_validates_the_director_shape(sh):
    with pytest.raises(ValueError, match="director has shape"):
        stripes.relax(sh, onp.zeros((3, 2)), onp.ones(sh.n_nodes), max_steps=1)


def test_relax_validates_the_mask_shape(sh):
    with pytest.raises(ValueError, match="mask has shape"):
        stripes.relax(sh, onp.zeros((sh.n_nodes, 2)), onp.ones(3), max_steps=1)


def test_relax_rejects_zero_steps(sh):
    with pytest.raises(ValueError, match="max_steps must be"):
        stripes.relax(sh, onp.zeros((sh.n_nodes, 2)), onp.ones(sh.n_nodes),
                      max_steps=0)


def test_relax_rejects_bad_patience(sh):
    with pytest.raises(ValueError, match="patience must be"):
        stripes.relax(sh, onp.zeros((sh.n_nodes, 2)), onp.ones(sh.n_nodes),
                      patience=0)


# ── Convergence ──────────────────────────────────────────────────────────────

def _uniform_director(sh):
    return onp.tile([1.0, 0.0], (sh.n_nodes, 1))


def test_zero_tolerance_runs_exactly_max_steps(sh):
    """tol=0 can never be met, so it is the way to ask for a fixed step count."""
    r = stripes.relax(sh, _uniform_director(sh), onp.ones(sh.n_nodes),
                      tol=0.0, max_steps=25)
    assert r.steps == 25
    assert not r.converged
    assert r.residual > 0.0


def test_a_loose_tolerance_stops_early(sh):
    """A tolerance the field meets quickly must cut the run well short of the
    cap — that is the whole point of convergence detection."""
    r = stripes.relax(sh, _uniform_director(sh), onp.ones(sh.n_nodes),
                      tol=5e-2, patience=3, min_steps=5, max_steps=400)
    assert r.converged
    assert r.steps < 400
    assert r.residual <= 5e-2


def test_min_steps_is_respected(sh):
    """An enormous tolerance is met on step 1; min_steps must still hold."""
    r = stripes.relax(sh, _uniform_director(sh), onp.ones(sh.n_nodes),
                      tol=1e9, patience=1, min_steps=12, max_steps=400)
    assert r.steps == 12
    assert r.converged


def test_min_steps_is_clamped_to_max_steps(sh):
    r = stripes.relax(sh, _uniform_director(sh), onp.ones(sh.n_nodes),
                      tol=1e9, patience=1, min_steps=100, max_steps=7)
    assert r.steps == 7


def test_patience_requires_consecutive_quiet_steps(sh):
    """More patience cannot converge sooner: the counter resets on any step
    that exceeds the tolerance."""
    kw = dict(tol=2e-2, min_steps=5, max_steps=400)
    few = stripes.relax(sh, _uniform_director(sh), onp.ones(sh.n_nodes),
                        patience=1, **kw)
    many = stripes.relax(sh, _uniform_director(sh), onp.ones(sh.n_nodes),
                         patience=10, **kw)
    assert many.steps >= few.steps


def test_not_converging_warns_but_still_returns(two_layer_result):
    """Hitting the cap is reported, not raised: an under-annealed pattern is
    usually still usable, but the caller must be told."""
    with pytest.warns(UserWarning, match="max_steps"):
        out = stripes.stripes_from_result(two_layer_result, LAYERS[:1], PERIOD,
                                          tol=1e-12, max_steps=6)
    assert out[0].steps == 6
    assert not out[0].converged
    assert out[0].u.shape == (two_layer_result.mesh.points.shape[0],)


# ── Physics ──────────────────────────────────────────────────────────────────

def _dominant_gradient_direction(u, nx, ny, lx, ly):
    """Principal direction of grad(u) — perpendicular to the stripes."""
    grid = onp.reshape(u, (nx + 1, ny + 1)).T          # feax orders nodes y-fastest
    gy, gx = onp.gradient(grid, ly / ny, lx / nx)
    cov = onp.array([[(gx * gx).mean(), (gx * gy).mean()],
                     [(gx * gy).mean(), (gy * gy).mean()]])
    return onp.linalg.eigh(cov)[1][:, -1]


@pytest.mark.parametrize("angle_deg", [0.0, 45.0, 90.0])
def test_stripes_run_parallel_to_the_director(sh, angle_deg):
    """The whole point of the alignment term: u must vary ACROSS the fibre, so
    its gradient is perpendicular to the director."""
    ang = onp.deg2rad(angle_deg)
    d_vec = onp.array([onp.cos(ang), onp.sin(ang)])
    director = onp.tile(d_vec, (sh.n_nodes, 1))
    u = stripes.relax(sh, director, onp.ones(sh.n_nodes),
                      tol=0.0, max_steps=250).u

    k = _dominant_gradient_direction(u, 60, 30, LX, LY)
    assert abs(float(onp.dot(k, d_vec))) < 0.15


def test_stripe_wavelength_matches_the_requested_period(mesh):
    """Fibre along x, so u oscillates along y at the requested pitch."""
    period = 2e-3
    sh = stripes.make_sh_solver(mesh, period)
    director = onp.tile([1.0, 0.0], (sh.n_nodes, 1))
    u = stripes.relax(sh, director, onp.ones(sh.n_nodes),
                      tol=0.0, max_steps=300).u

    grid = onp.reshape(u, (61, 31)).T                  # [y, x]
    column = grid[:, 30]                               # a cut across the stripes
    crossings = onp.count_nonzero(onp.sign(column[1:]) * onp.sign(column[:-1]) < 0)
    measured = 2.0 * LY / crossings
    assert measured == pytest.approx(period, rel=0.15)


def test_the_field_stays_inside_the_mask(sh):
    """Void nodes are re-masked every step, so nothing leaks out of the region."""
    director = onp.tile([1.0, 0.0], (sh.n_nodes, 1))
    pts = onp.asarray(sh.mesh_physical.points)
    mask = (pts[:, 0] < 0.5 * LX).astype(float)
    u = stripes.relax(sh, director, mask, tol=0.0, max_steps=60).u
    assert onp.abs(u[mask == 0.0]).max() == 0.0
    assert onp.abs(u[mask == 1.0]).max() > 1e-3


def test_generate_stripes_packages_the_result(mesh):
    n = mesh.points.shape[0]
    field = stripes.generate_stripes(
        mesh, onp.tile([1.0, 0.0], (n, 1)), onp.ones(n), PERIOD, tol=0.0, max_steps=40)
    assert field.u.shape == (n,)
    assert field.stripe.min() >= 0.0 and field.stripe.max() <= 1.0
    assert field.stripe_period == PERIOD
    assert field.mesh is mesh


def test_save_vtu_writes_a_file(mesh, tmp_path):
    n = mesh.points.shape[0]
    field = stripes.generate_stripes(
        mesh, onp.tile([1.0, 0.0], (n, 1)), onp.ones(n), PERIOD, tol=0.0, max_steps=5)
    out = tmp_path / "stripes.vtu"
    stripes.save_vtu(field, out)
    assert out.exists() and out.stat().st_size > 0


# ── OptimizeResult adapter ───────────────────────────────────────────────────

def _fake_result(mesh, design):
    """An OptimizeResult carrying just what stripes_from_result reads."""
    import path_optimizer as po
    return po.OptimizeResult(
        x_opt=onp.zeros(1), design=design, history={}, final_obj=0.0,
        final_aux=None, best_obj=0.0, best_iter=0, n_iters=0,
        stop_reason="", mesh=mesh, space=None, pipeline=None,
        final_constraints={}, xdmf_path=None, output_dir=None)


@pytest.fixture(scope="module")
def two_layer_result(mesh):
    n = mesh.points.shape[0]
    design = {}
    for k, triple in enumerate((ALIGN_X, ALIGN_Y)):
        design[f"rho{k}"] = onp.ones(n)
        for name, v in zip(("x1", "x2", "x3"), triple, strict=True):
            design[f"{name}_{k}"] = onp.full(n, v)
    return _fake_result(mesh, design)


LAYERS = (("rho0", "x1_0", "x2_0", "x3_0"), ("rho1", "x1_1", "x2_1", "x3_1"))


def test_stripes_from_result_gives_one_field_per_layer(two_layer_result):
    out = stripes.stripes_from_result(two_layer_result, LAYERS, PERIOD,
                                      tol=0.0, max_steps=120)
    assert set(out) == {0, 1}
    n = two_layer_result.mesh.points.shape[0]
    for f in out.values():
        assert f.u.shape == (n,)

    # Layer 0 is aligned with x and layer 1 with y, so their stripes must be
    # perpendicular to each other — the design difference survives the pipeline.
    k0 = _dominant_gradient_direction(out[0].u, 60, 30, LX, LY)
    k1 = _dominant_gradient_direction(out[1].u, 60, 30, LX, LY)
    assert abs(float(onp.dot(k0, k1))) < 0.2


def test_stripes_from_result_resamples_onto_a_finer_mesh(two_layer_result):
    fine = fe.mesh.rectangle_mesh(Nx=80, Ny=40, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")
    out = stripes.stripes_from_result(two_layer_result, LAYERS[:1], PERIOD,
                                      mesh=fine, tol=0.0, max_steps=30)
    assert out[0].u.shape == (fine.points.shape[0],)
    assert out[0].mesh is fine


def test_stripes_from_result_reports_unknown_fields(two_layer_result):
    with pytest.raises(KeyError, match="no field"):
        stripes.stripes_from_result(two_layer_result,
                                    [("rho0", "x1_0", "x2_0", "nope")], PERIOD)


def test_stripes_from_result_rejects_a_malformed_group(two_layer_result):
    with pytest.raises(ValueError, match=r"\(rho, x1, x2, x3\)"):
        stripes.stripes_from_result(two_layer_result, [("rho0", "x1_0")], PERIOD)


def test_stripes_from_result_refuses_an_empty_mask(mesh):
    n = mesh.points.shape[0]
    design = {"rho0": onp.zeros(n), "x1_0": onp.full(n, ALIGN_X[0]),
              "x2_0": onp.full(n, ALIGN_X[1]), "x3_0": onp.full(n, ALIGN_X[2])}
    result = _fake_result(mesh, design)
    with pytest.raises(ValueError, match="nothing to grow stripes in"):
        stripes.stripes_from_result(result, LAYERS[:1], PERIOD, max_steps=1)
