"""Tests for the objective and regulariser library."""
import numpy as onp
import pytest

from path_optimizer import objectives as obj

fe = pytest.importorskip("feax")
jnp = pytest.importorskip("jax.numpy")


# Libertas orientation triples, in (x1, x2, x3) order.
ALIGN_X = (1.0 - 1e-2, -1.0 + 1e-2, 1.0)     # rank-1, |T| = 1
ALIGN_Y = (-1.0 + 1e-2, 1.0 - 1e-2, -1.0)    # rank-1 the other way
ISOTROPIC = (-1.0 + 1e-2, -1.0 + 1e-2, 0.0)  # no preferred direction, |T| = 0


def _design(n, rho, triple, prefix=""):
    """A uniform single-layer design mapping."""
    ones = jnp.ones(n)
    return {f"{prefix}rho": ones * rho,
            f"{prefix}x1": ones * triple[0],
            f"{prefix}x2": ones * triple[1],
            f"{prefix}x3": ones * triple[2]}


# ── group_layers ─────────────────────────────────────────────────────────────

def test_group_layers_chunks_layer_major():
    names = ("rho0", "x1_0", "x2_0", "x3_0", "rho1", "x1_1", "x2_1", "x3_1")
    assert obj.group_layers(names) == (names[:4], names[4:])


def test_group_layers_matches_the_shell_design_space():
    from path_optimizer import shell
    space = shell.design_space(n_layers=3)
    groups = obj.group_layers(space.names)
    assert len(groups) == 3
    assert groups[2] == ("rho2", "x1_2", "x2_2", "x3_2")


def test_group_layers_rejects_a_ragged_split():
    with pytest.raises(ValueError, match="do not divide"):
        obj.group_layers(("a", "b", "c"), per_layer=2)


# ── grey_penalty ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rho,expected", [(0.5, 1.0), (0.0, 0.0), (1.0, 0.0),
                                          (0.25, 0.75)])
def test_grey_penalty_values(rho, expected):
    d = {"rho": jnp.full(7, rho)}
    assert float(obj.grey_penalty(d, ("rho",))) == pytest.approx(expected)


def test_grey_penalty_averages_over_fields():
    d = {"a": jnp.full(4, 0.5), "b": jnp.zeros(4)}      # 1.0 and 0.0
    assert float(obj.grey_penalty(d, ("a", "b"))) == pytest.approx(0.5)


def test_grey_penalty_needs_a_field():
    with pytest.raises(ValueError, match="at least one field"):
        obj.grey_penalty({}, ())


# ── ud_penalty ───────────────────────────────────────────────────────────────

def test_ud_penalty_is_zero_for_a_unidirectional_field():
    d = _design(5, 1.0, ALIGN_X)
    val = float(obj.ud_penalty(d, [("rho", "x1", "x2", "x3")]))
    assert val == pytest.approx(0.0, abs=1e-6)


def test_ud_penalty_is_one_for_an_isotropic_field():
    d = _design(5, 1.0, ISOTROPIC)
    val = float(obj.ud_penalty(d, [("rho", "x1", "x2", "x3")]))
    assert val == pytest.approx(1.0, abs=1e-6)


def test_ud_penalty_ignores_void_nodes():
    """Half the nodes are isotropic but empty — they must not be penalised."""
    n = 8
    ones = jnp.ones(n)
    # Nodes 0..3 solid + aligned, nodes 4..7 void + isotropic.
    x1 = jnp.where(jnp.arange(n) < 4, ALIGN_X[0], ISOTROPIC[0])
    x2 = jnp.where(jnp.arange(n) < 4, ALIGN_X[1], ISOTROPIC[1])
    x3 = jnp.where(jnp.arange(n) < 4, ALIGN_X[2], ISOTROPIC[2])
    rho = jnp.where(jnp.arange(n) < 4, ones, 0.0)
    d = {"rho": rho, "x1": x1, "x2": x2, "x3": x3}
    val = float(obj.ud_penalty(d, [("rho", "x1", "x2", "x3")]))
    assert val == pytest.approx(0.0, abs=1e-6)     # not 0.5


def test_ud_penalty_averages_over_layers():
    d = {**_design(5, 1.0, ALIGN_X, prefix="a_"),
         **_design(5, 1.0, ISOTROPIC, prefix="b_")}
    layers = [("a_rho", "a_x1", "a_x2", "a_x3"),
              ("b_rho", "b_x1", "b_x2", "b_x3")]
    assert float(obj.ud_penalty(d, layers)) == pytest.approx(0.5, abs=1e-6)


def test_ud_penalty_rejects_a_malformed_layer_group():
    with pytest.raises(ValueError, match=r"\(rho, x1, x2, x3\)"):
        obj.ud_penalty(_design(3, 1.0, ALIGN_X), [("rho", "x1")])


def test_ud_penalty_needs_a_layer():
    with pytest.raises(ValueError, match="at least one layer"):
        obj.ud_penalty({}, [])


# ── magnitude_consistency ────────────────────────────────────────────────────

def test_magnitude_consistency_is_zero_when_alignment_matches_density():
    """|T| = 1 for ALIGN_X, so rho = 1 is the consistent state."""
    d = _design(6, 1.0, ALIGN_X)
    val = float(obj.magnitude_consistency(d, [("rho", "x1", "x2", "x3")]))
    assert val == pytest.approx(0.0, abs=1e-3)


def test_magnitude_consistency_penalises_aligned_but_sparse():
    """Fully aligned at half density is exactly the ambiguity it exists to kill."""
    d = _design(6, 0.5, ALIGN_X)
    val = float(obj.magnitude_consistency(d, [("rho", "x1", "x2", "x3")]))
    assert val == pytest.approx(0.25, abs=1e-2)      # (1 - 0.5)^2


def test_magnitude_consistency_penalises_dense_but_unaligned():
    d = _design(6, 1.0, ISOTROPIC)
    val = float(obj.magnitude_consistency(d, [("rho", "x1", "x2", "x3")]))
    assert val == pytest.approx(1.0, abs=1e-3)       # (0 - 1)^2


# ── displacement matching ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sol():
    """A Solution with a known field, built without solving anything."""
    layout = ((4, 3), (4, 2))                        # var0 = (u,v,w), var1 = theta
    uvw = onp.array([[0.0, 0.0, 1.0],
                     [0.0, 0.0, 2.0],
                     [0.0, 0.0, 3.0],
                     [0.0, 0.0, 4.0]])
    theta = onp.zeros((4, 2))
    dofs = jnp.asarray(onp.concatenate([uvw.ravel(), theta.ravel()]))
    return fe.Solution(dofs, layout)


def test_displacement_error_against_a_zero_target(sol):
    val = float(obj.displacement_error(sol, jnp.zeros((4, 3))))
    assert val == pytest.approx(1.0 + 4.0 + 9.0 + 16.0)


def test_displacement_error_is_zero_at_the_target(sol):
    val = float(obj.displacement_error(sol, sol.field(0)))
    assert val == pytest.approx(0.0)


def test_displacement_error_normalises(sol):
    raw = float(obj.displacement_error(sol, jnp.zeros((4, 3))))
    val = float(obj.displacement_error(sol, jnp.zeros((4, 3)), denom=raw))
    assert val == pytest.approx(1.0)


def test_transverse_error_reads_only_w(sol):
    """An in-plane target offset must not change the transverse error."""
    w_target = jnp.asarray([1.0, 2.0, 3.0, 0.0])
    assert float(obj.transverse_error(sol, w_target)) == pytest.approx(16.0)


def test_transverse_error_can_read_the_second_variable(sol):
    val = float(obj.transverse_error(sol, jnp.zeros(4), var_index=1, component=0))
    assert val == pytest.approx(0.0)


# ── transverse_target ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mesh():
    return fe.mesh.rectangle_mesh(Nx=3, Ny=2, domain_x=1.0, domain_y=1.0,
                                  ele_type="QUAD4")


def test_transverse_target_defaults_to_flat(mesh):
    t = onp.asarray(obj.transverse_target(mesh))
    assert t.shape == (mesh.points.shape[0], 3)
    assert not t.any()


def test_transverse_target_fills_only_the_transverse_component(mesh):
    pts = onp.asarray(mesh.points)
    t = onp.asarray(obj.transverse_target(mesh, lambda x, y: x + 2.0 * y))
    onp.testing.assert_allclose(t[:, 2], pts[:, 0] + 2.0 * pts[:, 1])
    assert not t[:, :2].any()


def test_transverse_target_can_return_a_scalar_field(mesh):
    t = onp.asarray(obj.transverse_target(mesh, lambda x, y: x, vec=1))
    assert t.shape == (mesh.points.shape[0],)


def test_transverse_target_validates_the_shape(mesh):
    with pytest.raises(ValueError, match="expected"):
        obj.transverse_target(mesh, lambda x, y: onp.zeros(3))
