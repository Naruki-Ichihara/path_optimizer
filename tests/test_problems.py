"""Physics and wiring tests for the plane-stress and laminated-shell problems."""
import numpy as onp
import pytest

import path_optimizer as po
from path_optimizer import geometry, materials, plane, shell

fe = pytest.importorskip("feax")

LX, LY = 2.0, 1.0


def _uniform(problem, values):
    """TracedParams with each design field uniform at the given value."""
    return fe.TracedParams(volume_vars=tuple(
        fe.TracedParams.create_node_var(problem, v) for v in values))


# ── geometry ─────────────────────────────────────────────────────────────────

def test_edge_predicate_selects_the_named_edge():
    left = geometry.edge_predicate("left", LX, LY)
    right = geometry.edge_predicate("right", LX, LY)
    assert bool(left(onp.array([0.0, 0.5]))) and not bool(left(onp.array([LX, 0.5])))
    assert bool(right(onp.array([LX, 0.5])))


def test_cantilever_edges_are_opposite():
    clamp, free = geometry.cantilever_edges("right", LX, LY)
    assert bool(clamp(onp.array([LX, 0.5]))) and bool(free(onp.array([0.0, 0.5])))


def test_edge_predicate_rejects_unknown_edge():
    with pytest.raises(ValueError, match="unknown edge"):
        geometry.edge_predicate("north", LX, LY)


# ── plane stress ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bar():
    """A uniaxial bar: u_x=0 on the left, u_y=0 along the bottom, traction on the
    right.  The exact solution is uniform uniaxial stress, so displacements can
    be checked against closed form."""
    E, NU, T, THICK = 210.0e9, 0.3, 1.0e6, 0.002
    mesh = fe.mesh.rectangle_mesh(Nx=8, Ny=4, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")
    left = geometry.edge_predicate("left", LX, LY)
    right = geometry.edge_predicate("right", LX, LY)
    bottom = geometry.edge_predicate("bottom", LX, LY)

    # traction is the residual-side value, i.e. minus the physical one, so
    # -T here is a physical pull of +T along x.
    problem = plane.make_plane_stress(
        mesh, materials.isotropic(E=E, nu=NU), thickness=THICK,
        location_fns=(right,), traction=(-T, 0.0),
    )
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=left, component=0, value=0.0),
        fe.DirichletBCSpec(location=bottom, component=1, value=0.0),
    ]).create_bc(problem)
    solver = po.make_linear_solver(problem, bc, _uniform(problem, (1.0,)))
    return dict(problem=problem, solver=solver, mesh=mesh,
                E=E, NU=NU, T=T, THICK=THICK)


def test_uniaxial_bar_matches_closed_form(bar):
    sol = bar["solver"](_uniform(bar["problem"], (1.0,)))
    u = onp.asarray(sol.field(0))
    pts = onp.asarray(bar["mesh"].points)

    # thickness · E · eps_xx = T  (plane stress, sigma_yy = 0)
    eps_xx = bar["T"] / (bar["THICK"] * bar["E"])
    onp.testing.assert_allclose(u[:, 0], eps_xx * pts[:, 0], rtol=1e-8, atol=1e-12)
    # lateral contraction: eps_yy = -nu eps_xx
    onp.testing.assert_allclose(u[:, 1], -bar["NU"] * eps_xx * pts[:, 1],
                                rtol=1e-8, atol=1e-12)


def test_density_scales_stiffness(bar):
    """Halving the density must exactly double the displacement: the isotropic
    constitutive is linear in rho (with e_min negligible here)."""
    full = onp.asarray(bar["solver"](_uniform(bar["problem"], (1.0,))).field(0))
    half = onp.asarray(bar["solver"](_uniform(bar["problem"], (0.5,))).field(0))
    onp.testing.assert_allclose(half, 2.0 * full, rtol=1e-6)


def test_plane_thermal_is_opt_in():
    """delta_t=None assembles no thermal term; a free plate then does not move."""
    mesh = fe.mesh.rectangle_mesh(Nx=4, Ny=4, domain_x=1.0, domain_y=1.0,
                                  ele_type="QUAD4")
    left = geometry.edge_predicate("left", 1.0, 1.0)
    bottom = geometry.edge_predicate("bottom", 1.0, 1.0)
    layer = materials.isotropic(E=1.0e9, nu=0.3, alpha=1.0e-5)
    specs = [fe.DirichletBCSpec(location=left, component=0, value=0.0),
             fe.DirichletBCSpec(location=bottom, component=1, value=0.0)]

    disp = {}
    for label, dT in (("none", None), ("cooled", -100.0)):
        problem = plane.make_plane_stress(mesh, layer, delta_t=dT)
        bc = fe.DirichletBCConfig(specs).create_bc(problem)
        solver = po.make_linear_solver(problem, bc, _uniform(problem, (1.0,)))
        disp[label] = onp.abs(
            onp.asarray(solver(_uniform(problem, (1.0,))).field(0))).max()

    assert disp["none"] == 0.0
    assert disp["cooled"] > 1e-6


def test_surface_load_count_is_validated():
    mesh = fe.mesh.rectangle_mesh(Nx=2, Ny=2, domain_x=1.0, domain_y=1.0,
                                  ele_type="QUAD4")
    edge = geometry.edge_predicate("right", 1.0, 1.0)
    with pytest.raises(ValueError, match="must match"):
        plane.make_plane_stress(mesh, materials.isotropic(1.0, 0.3),
                                location_fns=(edge,), surface_load_fns=[])


def test_plane_design_space_field_order():
    assert plane.design_space().names == plane.DENSITY_FIELDS
    assert plane.design_space(oriented=True).names == plane.ORIENTED_FIELDS


def test_explicit_filter_radius_overrides_the_fraction():
    space = plane.design_space(rho_filter_radius=0.25)
    assert space.fields[0].filter_radius == 0.25
    assert space.fields[0].filter_frac is None


# ── laminated shell ──────────────────────────────────────────────────────────

# Spelled out, not a preset: these numbers decide what the tests assert.
LAMINA = materials.Lamina(E1=140.0e9, E2=10.0e9, G12=5.0e9, nu12=0.30,
                          G13=5.0e9, G23=3.0e9,
                          alpha_1=-0.5e-6, alpha_2=30.0e-6, thickness=0.5e-3)
POLYMER = materials.Polymer(E=2.0e9, nu=0.40, alpha=70.0e-6)
SLX, SLY = 0.2, 0.1

# libertas orientation triples: fully aligned with x, and with y.
ALIGN_X = (0.9, 1.0 - 1e-2, -1.0 + 1e-2, 1.0)
ALIGN_Y = (0.9, -1.0 + 1e-2, 1.0 - 1e-2, -1.0)


def _shell(delta_t, load_mag=0.0, nx=10, ny=5):
    mesh = fe.mesh.rectangle_mesh(Nx=nx, Ny=ny, domain_x=SLX, domain_y=SLY,
                                  ele_type="QUAD4")
    clamp, free = geometry.cantilever_edges("right", SLX, SLY)
    problem = shell.make_laminated_shell(
        mesh, materials.orientation_blend(LAMINA, POLYMER),
        (LAMINA.thickness,) * 2, vars_per_layer=4, delta_t=delta_t,
        location_fns=(free,) if load_mag else (), load_mag=load_mag,
    )
    bc = shell.clamp_bc(problem, clamp)
    solver = po.make_linear_solver(problem, bc, _uniform(problem, ALIGN_X * 2))
    return problem, solver, mesh


def _w(problem, solver, stack):
    return onp.asarray(solver(_uniform(problem, stack)).field(0)[:, 2])


def test_asymmetric_laminate_warps_when_cooled():
    problem, solver, _ = _shell(delta_t=-150.0)
    w = _w(problem, solver, ALIGN_X + ALIGN_Y)
    assert onp.abs(w).max() > 1e-4        # metres — a visible warp


def test_symmetric_laminate_does_not_warp_when_cooled():
    """Identical layers -> M_T = 0 -> pure in-plane contraction, no bending."""
    problem, solver, _ = _shell(delta_t=-150.0)
    w = _w(problem, solver, ALIGN_X + ALIGN_X)
    assert onp.abs(w).max() < 1e-9


def test_shell_thermal_is_opt_in():
    """The same asymmetric stack with delta_t=None and no load must not move."""
    problem, solver, _ = _shell(delta_t=None)
    w = _w(problem, solver, ALIGN_X + ALIGN_Y)
    assert onp.abs(w).max() == 0.0


def test_transverse_load_bends_the_cantilever():
    problem, solver, _ = _shell(delta_t=None, load_mag=5.0)
    w = _w(problem, solver, ALIGN_X + ALIGN_X)
    assert onp.abs(w).max() > 1e-6


def test_stiffer_laminate_deflects_less():
    problem, solver, _ = _shell(delta_t=None, load_mag=5.0)
    dense = onp.abs(_w(problem, solver, ALIGN_X + ALIGN_X)).max()
    sparse_stack = ((0.1,) + ALIGN_X[1:]) * 2
    sparse = onp.abs(_w(problem, solver, sparse_stack)).max()
    assert sparse > dense


def test_shell_design_space_matches_weak_form_order():
    space = shell.design_space(n_layers=2)
    assert space.names == shell.layer_field_names(2)
    assert space.names == ("rho0", "x1_0", "x2_0", "x3_0",
                           "rho1", "x1_1", "x2_1", "x3_1")
    assert len(shell.design_space(n_layers=3).names) == 12
    assert shell.design_space(n_layers=2, oriented=False).names == ("rho0", "rho1")


def test_shell_rejects_bad_laminates():
    mesh = fe.mesh.rectangle_mesh(Nx=2, Ny=2, domain_x=1.0, domain_y=1.0,
                                  ele_type="QUAD4")
    layer = materials.orientation_blend(LAMINA, POLYMER)
    with pytest.raises(ValueError, match="at least one layer"):
        shell.make_laminated_shell(mesh, layer, ())
    with pytest.raises(ValueError, match="must be positive"):
        shell.make_laminated_shell(mesh, layer, (1e-3, -1e-3))


def test_shell_design_space_rejects_zero_layers():
    with pytest.raises(ValueError, match="n_layers must be"):
        shell.design_space(n_layers=0)


# ── the two problems drive the optimiser ─────────────────────────────────────

def test_shell_optimisation_stiffens_the_plate(tmp_path):
    """A full run through the driver: minimise |w| under load, no thermal."""
    import jax.numpy as jnp

    order = shell.layer_field_names(2)

    class Stiffen(po.Pipeline):
        has_aux = True

        def build(self, mesh):
            self.problem, self.solver, _ = _shell(None, load_mag=5.0)

        def transform(self, design, **params):
            return {**design,
                    "rho0": design["rho0"] ** 3.0,
                    "rho1": design["rho1"] ** 3.0}

        def objective(self, design, **params):
            sol = self.solver(fe.TracedParams(
                volume_vars=tuple(design[n] for n in order)))
            w = sol.field(0)[:, 2]
            return jnp.sum(w * w) * 1e6, {"w_max_mm": jnp.abs(w).max() * 1e3}

    mesh = fe.mesh.rectangle_mesh(Nx=10, Ny=5, domain_x=SLX, domain_y=SLY,
                                  ele_type="QUAD4")
    result = po.run(Stiffen(), mesh, shell.design_space(n_layers=2),
                    max_iter=5, output_dir=tmp_path, verbose=False)
    assert result.history["obj"][-1] < result.history["obj"][0]
    assert result.history["w_max_mm"][-1] < result.history["w_max_mm"][0]


# ── Void vs matrix as the low-density phase ──────────────────────────────────

ALIGNED = (1.0 - 1e-2, -1.0 + 1e-2, 1.0)        # unidirectional along x


def test_a_matrix_leaves_the_void_load_bearing():
    """The two-phase reading: rho = 0 is polymer, not nothing."""
    layer = materials.orientation_blend(LAMINA, POLYMER)
    solid = float(layer(1.0, *ALIGNED)[0][0, 0, 0, 0])
    empty = float(layer(0.0, *ALIGNED)[0][0, 0, 0, 0])
    assert empty > 0.01 * solid                 # ~1/59 here, nowhere near zero


def test_no_matrix_means_no_stiffness_at_zero_density():
    layer = materials.orientation_blend(LAMINA)  # polymer=None
    solid = float(layer(1.0, *ALIGNED)[0][0, 0, 0, 0])
    empty = float(layer(0.0, *ALIGNED)[0][0, 0, 0, 0])
    assert empty == pytest.approx(1e-9 * solid, rel=1e-6)


def test_void_is_the_same_mixture_with_c_poly_set_to_e_min_c_fibre():
    """Documented equivalence -- one rule of mixtures, not two code paths."""
    e_min = 1e-4
    layer = materials.orientation_blend(LAMINA, e_min=e_min)
    C_fibre = onp.asarray(layer(1.0, *ALIGNED)[0])
    for rho in (0.0, 0.3, 0.7, 1.0):
        C = onp.asarray(layer(rho, *ALIGNED)[0])
        assert C == pytest.approx((e_min + rho * (1.0 - e_min)) * C_fibre,
                                  rel=1e-10)


def test_void_keeps_its_orientation():
    """e_min scales the oriented tensor, so direction survives into the void."""
    layer = materials.orientation_blend(LAMINA)
    C = onp.asarray(layer(0.0, *ALIGNED)[0])
    assert C[0, 0, 0, 0] > 10.0 * C[1, 1, 1, 1]


def test_e_min_is_checked_only_where_it_is_used():
    with pytest.raises(ValueError, match="e_min"):
        materials.orientation_blend(LAMINA, e_min=1.0)
    # Irrelevant with a matrix phase, so not validated there.
    materials.orientation_blend(LAMINA, POLYMER, e_min=1.0)


# ── Transverse knock-down ────────────────────────────────────────────────────

def test_knockdown_is_off_unless_asked_for():
    """The four-argument contract is what plane and shell call."""
    assert materials.orientation_blend(LAMINA).__code__.co_argcount == 4
    assert materials.orientation_blend(
        LAMINA, transverse_knockdown=True).__code__.co_argcount == 5


def test_knockdown_of_one_is_the_untouched_material():
    plain = materials.orientation_blend(LAMINA)
    kd = materials.orientation_blend(LAMINA, transverse_knockdown=True)
    assert onp.asarray(kd(1.0, *ALIGNED, 1.0)[0]) == pytest.approx(
        onp.asarray(plain(1.0, *ALIGNED)[0]), rel=1e-12)


def test_knockdown_raises_the_stiffness_contrast():
    """The whole point: it makes one fibre direction unmistakably better."""
    kd = materials.orientation_blend(LAMINA, transverse_knockdown=True)

    def contrast(k):
        C = onp.asarray(kd(1.0, *ALIGNED, k)[0])
        return C[0, 0, 0, 0] / C[1, 1, 1, 1]

    ratios = [contrast(k) for k in (1.0, 0.3, 0.1, 0.02)]
    assert ratios == sorted(ratios)                 # monotone in 1/k
    assert ratios[-1] > 5.0 * ratios[0]


def test_knockdown_leaves_the_fibre_direction_alone():
    """It scales E2, so the stiff direction must not move."""
    kd = materials.orientation_blend(LAMINA, transverse_knockdown=True)
    along = [float(onp.asarray(kd(1.0, *ALIGNED, k)[0])[0, 0, 0, 0])
             for k in (1.0, 0.02)]
    assert along[1] == pytest.approx(along[0], rel=0.05)


def test_knockdown_reaches_the_constitutive_through_the_problem():
    """A uniform fifth volume var, not a design variable -- that is how it is
    scheduled, so it has to survive the trip into get_tensor_map."""
    import jax.numpy as jnp
    layer = materials.orientation_blend(LAMINA, transverse_knockdown=True)
    mesh = fe.mesh.rectangle_mesh(Nx=4, Ny=2, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")
    stress = plane.make_plane_stress(mesh, layer, thickness=1.0).get_tensor_map()
    pull_across = jnp.array([[0.0, 0.0], [0.0, 1.0e-3]])   # transverse stretch
    full = float(stress(pull_across, 1.0, *ALIGNED, 1.0)[1, 1])
    knocked = float(stress(pull_across, 1.0, *ALIGNED, 0.02)[1, 1])
    assert knocked < 0.1 * full
