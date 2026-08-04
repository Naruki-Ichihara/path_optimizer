"""MBB beam with continuous fibre orientation, carried through to stripes.

The full pipeline in one script:

.. code-block:: text

    optimise (density + fibre orientation)
        -> aligned-SH stripes
        -> path extraction -> SVG
        -> order -> connect -> order
        -> FullControl design -> g-code

Same beam, supports and load as :mod:`examples.mbb_beam`, but the design is no
longer a single density.  Each node carries a density *and* a libertas
orientation triple, so the material at every point is a blend of polymer matrix
and orientation-averaged fibre lamina — the optimiser chooses both **how much**
fibre and **which way it runs**.

The two orientation regularisers matter as much as the compliance here:

* ``ud_penalty`` drives the orientation tensor rank-1.  A rank-2 ``a2`` means
  "fibres in several directions at once", which averages to a stiffness but is
  not a printable path.
* ``magnitude_consistency`` ties alignment strength to density, removing the
  "half the fibre, fully aligned" / "full fibre, half aligned" degeneracy.
* ``create_orientation_smoothness_fn`` limits how fast the direction turns.
  Where the director turns inside one stripe period, the paths have to collide
  no matter how the stripe stage is run, so this is the only place that problem
  can be fixed.

Without them the compliance is happy with a smeared orientation field and the
stripe stage has nothing coherent to follow.

The stripe stage then relaxes an aligned Swift-Hohenberg field along the
optimised director, on a **refined** mesh: the optimisation mesh is sized for
the topology (1 mm elements over a 200 x 50 mm beam), far too coarse to carry
the stripes, so the design is resampled onto a 0.333 mm one.

    python examples/mbb_fibre.py

Writes into ``examples/output_mbb_fibre/``:

* ``history.xdmf`` / ``history.csv`` / ``history.png`` — the optimisation
* ``design.vtu``   — the optimised density + director, for ParaView
* ``design.png``   — density with the fibre director overlaid
* ``stripes.png``  — the relaxed stripe field
* ``stripes.vtu``  — stripe field + director, for ParaView
* ``paths.svg``    — the extracted fibre print paths
* ``design.json``  — the connected, ordered toolpath as a FullControl design
* ``paths.gcode``  — g-code for ``PRINTER``, via FullControl

The last stage goes **through the SVG**: the paths are written, read back as
:class:`~path_optimizer.paths.Path` objects, sequenced, and only then handed to
:mod:`fullcontrol`.  Writing and re-reading makes the file an editable handoff
rather than a picture, and each polyline keeps the kind of the SVG group it sat
in — that label is what would let the connector and the sequencer keep several
processes apart, and what ends up as a comment in the g-code.  The names
are chosen here: :mod:`path_optimizer.paths` treats a kind as an opaque group.
It is FullControl, not this code, that owns printer profiles, extrusion models
and preview, so the same design can be retargeted without touching anything
here.
"""
from collections import Counter
from pathlib import Path

import feax as fe
import feax.gene as gene
import fullcontrol as fc
import jax.numpy as jnp
import numpy as onp

import path_optimizer as po
from path_optimizer import materials, paths, plane, stripes
from path_optimizer import objectives as obj

# ── Geometry and material (SI units, so the stripe pitch is meaningful) ──────

LX, LY = 0.200, 0.050        # full span : height (4:1)
NX, NY = 200, 50             # optimisation elements (1 mm)
THICKNESS = 2.0e-3           # out-of-plane
LOAD = 1.0e3                 # total downward force on the load patch [N]

LAMINA = materials.Lamina()          # CFRP
POLYMER = materials.Polymer()        # thermoplastic matrix
SGN_BETA = 10.0                      # orientation-tensor sharpness

VOLUME_FRACTION = 0.4
FILTER_RADIUS = 3.0 * (LX / NX)      # ~3 elements
MAX_ITER = 150

# Regulariser weights.  The compliance is normalised by its initial value, so
# these are directly comparable to it: at 0.5 a fully isotropic orientation
# costs as much as half the initial compliance.
W_UD = 0.5
W_MAG = 0.5
# Orientation smoothness.  A director that turns sharply forces the stripes to
# collide there whatever the stripe stage does -- the geometry demands it.
# Penalising the turn inside the objective, rather than smoothing the finished
# design, lets the optimiser buy the smoothness back by moving material.
W_SMOOTH = 0

# ── Stripe stage ────────────────────────────────────────────────────────────

PATH_PITCH = 1.0e-3          # centre-to-centre spacing of the printed paths
# The paths are the ZERO CONTOURS of the stripe field, and cos(phi) crosses zero
# twice per period -- so the stripe period is twice the pitch we want.  That is
# also the cheap direction: the field needs ~6 elements per STRIPE period, which
# is 3 per path spacing, a quarter of the nodes that generating at the pitch
# itself and throwing away every other contour would cost.
STRIPE_PERIOD = 2.0 * PATH_PITCH
SH_NX, SH_NY = 600, 150      # 0.333 mm -> 6 elements per stripe period
# Start from an integrated phase field, not noise, and run only long enough to
# saturate the amplitude.  Measured on this part: noise gives 33 defects and a
# +4.4% pitch after ~105 steps; the phase seed gives 4 defects and +0.9% after
# 10.  Relaxing further undoes it -- see stripes_from_result's `phase_steps`.
SH_PHASE_STEPS = 10
# Hybrid: freeze the phase field where its own predicted pitch is within
# GATE_LO%, hand SH everything past GATE_HI%.  Measured on this part -- noise
# 33 defects / +4.4% / 25.2% tail; phase alone 4 / +0.9% / 24.5%; hybrid
# 10 / +1.0% / 12.2%.  A weak alignment in the freed pockets scored best; drop
# GAMMA_FREE to 0 there and they go labyrinthine instead.
SH_GATE_LO, SH_GATE_HI = 5.0, 15.0
SH_FREE_STEPS = 250
GAMMA_FREE = 8
RHO_CUTOFF = 0.5             # density above which a node counts as fibre

# ── Print stage ─────────────────────────────────────────────────────────────

# Weld any travel shorter than this into a straight bridge, turning separate
# stripes into one continuous zigzag.  Measured on this part: 60 paths become
# 28 and the travel drops from 365 mm to 299 mm, for 65 mm of extra extruded
# length.  Raising it welds harder -- 10 mm gives 13 paths and 200 mm of travel
# -- but the bridges grow with it, and `connect` does not check whether a bridge
# lies over material, so past a few path pitches they start crossing voids.
CONNECT_TOLERANCE = 5.0e-3   # m

LAYER_HEIGHT = 0.2           # mm
EXTRUSION_WIDTH = 0.4        # mm
PRINT_SPEED = 1000           # mm/min

# FullControl's own profiles -- see paths.printers().  Nothing about the machine
# lives in this package; swapping these two names retargets the whole example.
#
# Not "generic": that profile and "custom" carry no ending procedure at all, so
# the file stops after the last move with the nozzle hot and sitting on the
# part.  They are neutral starting points to build a profile from, not machines
# to print with.  A named profile brings its own start and end.
PRINTER = "prusa_mk4"
ALT_PRINTER = "ender_3"
# Machine state, set once by the profile's starting procedure.  Anything that
# has to vary *during* the print belongs in the design steps instead.
PRINTER_SETTINGS = {"nozzle_temp": 240, "bed_temp": 60}
# Travel behaviour.  Retraction is firmware G10/G11 via the profile's command
# list; the hop lifts, crosses and descends so the nozzle does not drag over
# fibre already laid down.  Both cost moves -- with 25 chains that is cheap, but
# it would not be at 60.
RETRACT = True
Z_HOP = 0.6                  # mm above the layer

OUT = Path(__file__).with_name("output_mbb_fibre")

# One "layer" here: the physical density plus the orientation triple.  The
# stripe stage and the orientation regularisers both read this grouping, so it
# lives in one place rather than being retyped at each call site.
LAYERS = (("rho_phys", "x1", "x2", "x3"),)


class MBBFibre(po.Pipeline):
    """Compliance minimisation over density *and* fibre orientation."""

    has_aux = True

    def build(self, mesh):
        h = LX / NX
        pad = 2.0 * h

        def pin(p):
            return (p[0] < pad) & (p[1] < 1e-9)

        def roller(p):
            return (p[0] > LX - pad) & (p[1] < 1e-9)

        def load_patch(p):
            return (jnp.abs(p[0] - 0.5 * LX) < pad) & jnp.isclose(p[1], LY, atol=1e-9)

        # Four design fields per node: (rho, x1, x2, x3), in the order
        # orientation_blend consumes them — plane.design_space(oriented=True)
        # emits exactly that order.
        self.problem = plane.make_plane_stress(
            mesh,
            materials.orientation_blend(LAMINA, POLYMER, sgn_beta=SGN_BETA),
            thickness=THICKNESS,
            location_fns=(load_patch,),
            # Residual-side traction: positive ty pulls DOWN.
            traction=(0.0, LOAD / (2.0 * pad)),
        )
        self.bc = fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=pin, component=0, value=0.0),
            fe.DirichletBCSpec(location=pin, component=1, value=0.0),
            fe.DirichletBCSpec(location=roller, component=1, value=0.0),
        ]).create_bc(self.problem)

        nv = fe.TracedParams.create_node_var
        sample = fe.TracedParams(volume_vars=tuple(
            nv(self.problem, v) for v in (0.5, 0.0, 0.0, 0.0)))
        self.solver = po.make_linear_solver(self.problem, self.bc, sample)
        self.compliance = obj.create_compliance_fn(self.problem)
        self.volume = obj.create_volume_fn(self.problem)
        # Measured against the path pitch: a direction change that happens in
        # less than the gap between two paths is one they cannot follow.
        self.smoothness = obj.create_orientation_smoothness_fn(
            self.problem, length_scale=PATH_PITCH, sgn_beta=SGN_BETA)

        # Normalise on the starting design so the regulariser weights above mean
        # the same thing whatever the load and material scale are.
        n = mesh.points.shape[0]
        start = fe.TracedParams(volume_vars=(
            jnp.full(n, VOLUME_FRACTION), jnp.zeros(n), jnp.zeros(n),
            jnp.zeros(n)))
        self.c0 = float(self.compliance(self.solver(start)))
        print(f"Initial compliance (normaliser): {self.c0:.6e} J")

    def _solve(self, design):
        return self.solver(fe.TracedParams(volume_vars=(
            design["simp"], design["x1"], design["x2"], design["x3"])))

    def transform(self, design, penalty=3.0, beta=1.0, **params):
        """SIMP + Heaviside on the density; the orientation passes through.

        Only the density is projected — an orientation parameter is not a
        material fraction, and raising it to a power would just distort the
        tensor the physics and the regularisers both read.
        """
        rho_phys = gene.heaviside_projection(design["rho"], beta=beta)
        return {**design, "rho_phys": rho_phys, "simp": rho_phys ** penalty}

    def objective(self, design, **params):
        c = self.compliance(self._solve(design)) / self.c0
        ud = obj.ud_penalty(design, LAYERS, sgn_beta=SGN_BETA)
        mag = obj.magnitude_consistency(design, LAYERS, sgn_beta=SGN_BETA)
        smooth = self.smoothness(design, LAYERS)
        loss = c + W_UD * ud + W_MAG * mag + W_SMOOTH * smooth
        return loss, {"c": c, "ud": ud, "mag": mag, "smooth": smooth}

    @po.constraint(target=VOLUME_FRACTION)
    def volfrac(self, design, **params):
        return self.volume(design["rho_phys"])

    def snapshot(self, design, aux=None):
        """Add the director so ParaView can glyph it over the density."""
        d = stripes.director_from_orientation(
            design["x1"], design["x2"], design["x3"], sgn_beta=SGN_BETA)
        fields = [(k, v) for k, v in design.items()]
        fields.append(("director", onp.column_stack([d, onp.zeros(d.shape[0])])))
        return fields


# ── Output ───────────────────────────────────────────────────────────────────

def save_design_vtu(pipeline, result, path):
    """The optimised design as a ParaView ``.vtu``.

    ``history.xdmf`` already holds every iteration, but it lives on the
    optimisation mesh and is awkward to overlay on the stripe field.  This
    writes the final design alone, with the same field names, so the two ``.vtu``
    files can be opened side by side — density and director next to the stripes
    they produced.
    """
    fe.utils.save_sol(result.mesh, str(path),
                      point_infos=pipeline.snapshot(result.design))
    print(f"Wrote {path}")


# ── Plots ────────────────────────────────────────────────────────────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_design_plot(result, path, every=6):
    """Density with the fibre director drawn on top."""
    plt = _plt()
    pts = onp.asarray(result.mesh.points)
    rho = onp.asarray(result.design["rho_phys"])
    d = stripes.director_from_orientation(
        result.design["x1"], result.design["x2"], result.design["x3"],
        sgn_beta=SGN_BETA)

    fig, ax = plt.subplots(figsize=(12, 12 * LY / LX + 1))
    ax.tripcolor(pts[:, 0], pts[:, 1], rho, cmap="gray_r",
                 shading="gouraud", vmin=0.0, vmax=1.0)
    # Only glyph where there is material — a director in the void is meaningless
    # — and thin the survivors so the arrows stay readable.
    idx = onp.where(rho > RHO_CUTOFF)[0][::every]
    ax.quiver(pts[idx, 0], pts[idx, 1], d[idx, 0], d[idx, 1],
              color="tab:red", pivot="mid", headwidth=0, headlength=0,
              headaxislength=0, scale=60, width=0.0015)
    ax.set_aspect("equal")
    # final_obj is the whole weighted loss; report the compliance term itself.
    ax.set_title(
        f"density + fibre director — compliance {result.final_aux['c']:.4g} "
        f"(loss {result.final_obj:.4g}), "
        f"V/V0 = {result.final_constraints['volfrac']:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def save_stripe_plot(field, path):
    plt = _plt()
    pts = onp.asarray(field.mesh.points)
    fig, ax = plt.subplots(figsize=(12, 12 * LY / LX + 1))
    ax.tripcolor(pts[:, 0], pts[:, 1], field.stripe, cmap="gray_r",
                 shading="gouraud", vmin=0.0, vmax=1.0)
    ax.set_aspect("equal")
    ax.set_title(f"aligned-SH stripes — period {field.stripe_period * 1e3:g} mm, "
                 f"{field.steps} steps"
                 + ("" if field.converged else " (NOT converged)"))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


# ── Driver ───────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    mesh = fe.mesh.rectangle_mesh(Nx=NX, Ny=NY, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")
    space = plane.design_space(
        oriented=True,
        rho_bounds=(1e-3, 1.0),
        rho_init=VOLUME_FRACTION,
        rho_filter_radius=FILTER_RADIUS,
        theta_filter_radius=FILTER_RADIUS,
    )
    print(f"Design fields: {space.names}")

    pipeline = MBBFibre()
    result = po.run(
        pipeline, mesh, space,
        max_iter=MAX_ITER,
        continuations={
            "penalty": po.Continuation(1.0, 3.0, update_every=40, step=0.5),
            "beta": po.Continuation(1.0, 8.0, update_every=40, step=2.0),
        },
        output_dir=OUT,
        snapshot_every=5,
    )
    print(f"\ncompliance (normalised) : {result.final_aux['c']:.4f}")
    print(f"UD penalty              : {result.final_aux['ud']:.4f}  (0 = rank-1)")
    print(f"magnitude consistency   : {result.final_aux['mag']:.4f}  (0 = |T| = rho)")
    print(f"orientation smoothness  : {result.final_aux['smooth']:.4f}  (0 = uniform)")
    print(f"volume                  : {result.final_constraints['volfrac']:.4f}")
    print(f"stopped                 : {result.stop_reason} after {result.n_iters} iters")

    save_design_plot(result, OUT / "design.png")
    save_design_vtu(pipeline, result, OUT / "design.vtu")

    # ── Stripes ──
    # The optimisation mesh has 1 mm elements, far too coarse for a 1 mm pitch,
    # so relax on a refined mesh and let stripes_from_result resample onto it.
    print(f"\nRelaxing aligned-SH stripes "
          f"(period {STRIPE_PERIOD * 1e3:g} mm -> path pitch "
          f"{PATH_PITCH * 1e3:g} mm)...")
    fine = fe.mesh.rectangle_mesh(Nx=SH_NX, Ny=SH_NY, domain_x=LX, domain_y=LY,
                                  ele_type="QUAD4")
    fields = stripes.stripes_from_result(
        result, LAYERS, STRIPE_PERIOD,
        mesh=fine, rho_cutoff=RHO_CUTOFF, sgn_beta=SGN_BETA,
        seed_from="hybrid", phase_steps=SH_PHASE_STEPS,
        gate_lo=SH_GATE_LO, gate_hi=SH_GATE_HI,
        free_steps=SH_FREE_STEPS, gamma_free=GAMMA_FREE, verbose=True,
    )
    field = fields[0]
    print(f"stripes: {field.steps} steps, residual {field.residual:.2e}")
    save_stripe_plot(field, OUT / "stripes.png")
    stripes.save_vtu(field, OUT / "stripes.vtu")

    # ── Print paths ──
    # The zero level set of the stripe field, which is where the paths sit: two
    # consecutive crossings are half a period apart, so the contours come out at
    # the pitch that was asked for.
    # `paths` knows nothing about fibre -- the kind is named here, and it is what
    # carries the label through the SVG, the sequencer and into the g-code
    # comments.  Only the fibre is laid down at this stage; the matrix region
    # (paths.extract_region_contours) is a separate process and is not generated
    # here.  It is not free to carry along: as five closed contours it cost
    # 253 mm of the 568 mm of travel, and it cannot be sequenced with the fibre.
    polylines = paths.tag(paths.extract_paths(field), "fibre")
    lengths = paths.path_lengths([p.nodes for p in polylines])
    print(f"\npaths: {len(polylines)}  total {lengths.sum() * 1e3:.0f} mm  "
          f"median {onp.median(lengths) * 1e3:.1f} mm")
    svg = OUT / "paths.svg"
    paths.write_svg(polylines, svg)
    print(f"Wrote {svg}")

    # ── SVG -> Path objects ──
    # Read the paths back out of the SVG rather than reusing the array in
    # memory.  That makes the file the actual handoff: open it in a vector
    # editor, delete or redraw paths, and run from here again.  read_svg takes
    # only <polyline>, and each one keeps the kind of the group it sat in, so an
    # editor's own layers arrive under their own name instead of silently
    # becoming toolpaths.
    from_svg = paths.read_svg(svg)
    kinds = Counter(p.kind for p in from_svg)
    print(f"Read back {len(from_svg)} paths from the SVG "
          f"({sum(p.length for p in from_svg) * 1e3:.0f} mm): {dict(kinds)}")

    # ── Order, connect, order ──
    # Sequencing picks the order *and* the direction of every path, so the head
    # does not cross the beam on each travel.  It comes first because `connect`
    # welds neighbours **in the sequence it is given**: matching on absolute
    # distance does not work here, since stripe ends lie along the part boundary
    # and consecutive stripes end pitch/sin(angle) apart, not a pitch apart.
    ordered = paths.order_paths(from_svg)

    # Then weld: every travel shorter than CONNECT_TOLERANCE becomes a straight
    # bridge and the two paths become one.  Sequence again afterwards, since the
    # chains have different ends to start from than their pieces did.
    joined = paths.connect(ordered, tolerance=CONNECT_TOLERANCE)
    final = paths.order_paths(joined)
    print(f"connect: {len(ordered)} -> {len(joined)} paths, "
          f"+{(sum(p.length for p in joined) - sum(p.length for p in ordered)) * 1e3:.0f}"
          f" mm of bridges")
    print(f"travel: {paths.travel_distance(ordered) * 1e3:.0f} mm -> "
          f"{paths.travel_distance(final) * 1e3:.0f} mm")
    ordered = final

    # ── FullControl design ──
    # A design is a plain list of FullControl step objects, not a file format:
    # geometry (`fc.Point`) interleaved with state (`fc.Extruder`,
    # `fc.ExtrusionGeometry`, `fc.Printer`).  Building it is the last thing this
    # package does -- everything past here is fullcontrol's own API, used
    # directly.


    steps = paths.to_fullcontrol(ordered, z=LAYER_HEIGHT, width=EXTRUSION_WIDTH,
                                 height=LAYER_HEIGHT, print_speed=PRINT_SPEED,
                                 retract=RETRACT, hop=Z_HOP)
    print(f"\nFullControl design: {len(steps)} steps")

    # The same thing built a path at a time, which is how to get something of
    # your own in between.  `paths.path_steps` converts one Path; the rest is
    # fc.  Here: slow down for closed loops, where the head comes back round to
    # where it started and a fast pass shows the seam.
    by_hand = [fc.ExtrusionGeometry(width=EXTRUSION_WIDTH, height=LAYER_HEIGHT),
               fc.Printer(print_speed=PRINT_SPEED)]
    for path in ordered:
        if path.is_closed:
            by_hand.append(fc.Printer(print_speed=PRINT_SPEED // 2))
        by_hand += paths.path_steps(path, z=LAYER_HEIGHT, retract=RETRACT,
                                    hop=Z_HOP)
        if path.is_closed:
            by_hand.append(fc.Printer(print_speed=PRINT_SPEED))
    print(f"the same, built per path with a slow pass on the "
          f"{sum(p.is_closed for p in ordered)} closed loop(s): "
          f"{len(by_hand)} steps")

    # fc.export_design appends `.json` to whatever it is given, so pass the stem.
    fc.export_design(steps, str(OUT / "design"))
    print(f"Wrote {OUT / 'design.json'}")

    # ── G-code ──
    # We own no printer profile.  fc.GcodeControls picks one of FullControl's
    # and initialization_data overrides its settings; the profile, the start and
    # end procedures, the extrusion model and the e-axis arithmetic are all
    # fullcontrol's.
    #
    # Two things to know.  A misspelled key in initialization_data is accepted
    # in silence and simply does nothing.  And these settings only set the
    # *starting* state: the ExtrusionGeometry and Printer steps above override
    # extrusion_width, extrusion_height and print_speed from the moment they
    # appear, so the profile's values for those three never take effect here.
    print(f"\nprinter {PRINTER!r}; overriding "
          f"{', '.join(f'{k}={v}' for k, v in PRINTER_SETTINGS.items())}")
    gcode = fc.transform(steps, "gcode", fc.GcodeControls(
        printer_name=PRINTER, initialization_data=PRINTER_SETTINGS,
    ), show_tips=False)
    (OUT / "paths.gcode").write_text(gcode)
    print(f"Wrote {OUT / 'paths.gcode'} ({len(gcode.splitlines())} lines, "
          f"{len(gcode) / 1024:.0f} kB)")

    # Same design, a second machine: nothing above changes.
    other = fc.transform(steps, "gcode", fc.GcodeControls(
        printer_name=ALT_PRINTER, initialization_data=PRINTER_SETTINGS,
    ), show_tips=False)
    print(f"the same design on {ALT_PRINTER!r}: {len(other.splitlines())} lines "
          f"({len(other.splitlines()) - len(gcode.splitlines()):+d})")
    print("  fc.transform(steps, 'plot')   # look at it before printing")

    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
