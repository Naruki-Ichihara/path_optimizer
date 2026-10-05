"""Tests for the path type, sequencing, and the SVG / FullControl handoff."""
import numpy as onp
import pytest

from path_optimizer import paths as P


def line(y, x0=0.0, x1=0.05, n=20):
    return onp.stack([onp.linspace(x0, x1, n), onp.full(n, y)], axis=1)


def loop(cx=0.02, cy=0.01, r=0.005, n=40):
    t = onp.linspace(0.0, 2.0 * onp.pi, n)
    xy = onp.stack([cx + r * onp.cos(t), cy + r * onp.sin(t)], axis=1)
    xy[-1] = xy[0]                                   # exactly closed
    return xy


# ── Path ─────────────────────────────────────────────────────────────────────

def test_path_measures_itself():
    p = P.Path(line(0.0, 0.0, 0.05))
    assert p.length == pytest.approx(0.05)
    assert p.start == pytest.approx([0.0, 0.0])
    assert p.end == pytest.approx([0.05, 0.0])
    assert not p.is_closed


def test_closed_detected_and_open_is_not():
    assert P.Path(loop()).is_closed
    assert not P.Path(line(0.0)).is_closed


def test_closed_tolerance_is_relative_to_length():
    """A fixed absolute tolerance would classify by unit system, not by shape."""
    small = loop(r=1e-6)                             # micrometre loop
    big = loop(r=1.0)                                # metre loop
    assert P.Path(small).is_closed and P.Path(big).is_closed
    gapped = loop(r=1e-6)
    gapped[-1] += 1e-7                               # 10% of the radius
    assert not P.Path(gapped).is_closed


def test_reversed_preserves_length_and_swaps_ends():
    p = P.Path(line(0.0))
    r = p.reversed()
    assert r.length == pytest.approx(p.length)
    assert r.start == pytest.approx(p.end)
    assert r.kind == p.kind and r.path_id == p.path_id


def test_rolled_keeps_the_loop_closed_and_starts_where_asked():
    p = P.Path(loop())
    r = p.rolled(7)
    assert r.is_closed
    assert len(r.nodes) == len(p.nodes)
    assert r.start == pytest.approx(p.nodes[7])
    assert r.length == pytest.approx(p.length)


def test_rolling_an_open_path_is_refused():
    with pytest.raises(ValueError, match="closed"):
        P.Path(line(0.0)).rolled(3)


def test_degenerate_input_rejected():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        P.Path(onp.zeros((5, 3)))
    with pytest.raises(ValueError, match="2 nodes"):
        P.Path(onp.zeros((1, 2)))


# ── Ordering ─────────────────────────────────────────────────────────────────

def scrambled_lines(n=10, seed=0):
    polys = [line(0.001 * k) for k in range(n)]
    polys = [p if k % 2 else p[::-1] for k, p in enumerate(polys)]
    onp.random.default_rng(seed).shuffle(polys)
    return [P.Path(p, "line", i) for i, p in enumerate(polys)]


def test_travel_distance_sums_the_gaps():
    a = P.Path(line(0.0, 0.0, 0.01))
    b = P.Path(line(0.0, 0.02, 0.03))                # 10 mm gap
    assert P.travel_distance([a, b]) == pytest.approx(0.01)
    assert P.travel_distance([a]) == 0.0
    assert P.travel_distance([]) == 0.0


def test_ordering_cuts_travel_a_lot():
    scrambled = scrambled_lines()
    ordered = P.order_paths(scrambled)
    before, after = (P.travel_distance(scrambled), P.travel_distance(ordered))
    assert after < 0.2 * before


def test_ordering_conserves_every_path_and_its_length():
    scrambled = scrambled_lines()
    ordered = P.order_paths(scrambled)
    assert len(ordered) == len(scrambled)
    assert sorted(p.path_id for p in ordered) == sorted(
        p.path_id for p in scrambled)
    assert (sum(p.length for p in ordered)
            == pytest.approx(sum(p.length for p in scrambled)))


def test_ordering_is_free_to_reverse():
    """Direction is half the win: neighbouring lines must be laid boustrophedon.

    Without reversal every hop crosses the full 50 mm line, ~450 mm in all; a
    perfect sweep costs 1 mm a hop.  Greedy finds that 9 mm optimum here.
    """
    ordered = P.order_paths(scrambled_lines(), start=(0.0, 0.0), method="greedy")
    assert P.travel_distance(ordered) == pytest.approx(0.009, abs=1e-9)


def test_tsp_traverses_every_path_it_enters_on_real_geometry():
    """The zero-edge property the formulation rests on.

    A tour that enters a path and leaves without taking its zero edge makes the
    printer travel the length of that path with the extruder off.  Nothing in
    the solver guarantees against it -- on ten identical stacked lines it does
    happen, costing one 50 mm phantom travel -- so this pins the case that
    matters: the real stripe paths, where all 59 zero edges are taken.
    """
    tsp_solver = pytest.importorskip("tsp_solver.greedy")
    lines = [line(0.001 * k, 0.0, 0.02 + 0.001 * k) for k in range(12)]
    paths = [P.Path(p, "line", i) for i, p in enumerate(lines)]
    ends = onp.empty((2 * len(paths), 2))
    ends[0::2] = [p.start for p in paths]
    ends[1::2] = [p.end for p in paths]
    d = onp.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=-1)
    i = onp.arange(len(paths))
    d[2 * i, 2 * i + 1] = d[2 * i + 1, 2 * i] = 0.0
    tour = tsp_solver.solve_tsp(d, optim_steps=3)
    used = sum(1 for a, b in zip(tour, tour[1:], strict=False) if a // 2 == b // 2)
    assert used == len(paths)


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="tsp.*greedy"):
        P.order_paths(scrambled_lines(), method="annealing")


@pytest.mark.parametrize("method", ["tsp", "greedy"])
def test_ordering_conserves_paths_for_both_methods(method):
    scrambled = scrambled_lines()
    ordered = P.order_paths(scrambled, method=method)
    assert sorted(p.path_id for p in ordered) == sorted(
        p.path_id for p in scrambled)


@pytest.mark.parametrize("method", ["tsp", "greedy"])
def test_start_point_pulls_the_first_path_near(method):
    """Not exact for the TSP -- the start is one node among many."""
    first = P.order_paths(scrambled_lines(), start=(0.05, 0.009),
                          method=method)[0]
    assert onp.linalg.norm(first.start - onp.array([0.05, 0.009])) < 0.005


def test_kinds_are_not_interleaved():
    mixed = scrambled_lines(4) + [P.Path(loop(), "outline", 99)]
    kinds = [p.kind for p in P.order_paths(mixed)]
    assert kinds == ["line"] * 4 + ["outline"]


def test_a_loop_is_entered_at_its_nearest_vertex():
    """A loop has no ends to match, so where it is cut is the only choice."""
    circle = P.Path(loop(cx=0.0, cy=0.0, r=0.01), "line", 0)
    ordered = P.order_paths([circle], start=(0.05, 0.0))
    assert ordered[0].start[0] == pytest.approx(0.01, abs=1e-3)


def test_ordering_accepts_bare_arrays():
    ordered = P.order_paths([line(0.0), line(0.001)])
    assert all(isinstance(p, P.Path) for p in ordered)


def test_ordering_empty():
    assert P.order_paths([]) == []


def test_by_kind_splits_in_first_seen_order():
    mixed = [P.Path(line(0.0), "b", 0), P.Path(line(0.001), "a", 1),
             P.Path(line(0.002), "b", 2)]
    groups = P.by_kind(mixed)
    assert list(groups) == ["b", "a"]
    assert [p.path_id for p in groups["b"]] == [0, 2]


def test_tag_relabels_without_touching_the_original():
    original = [P.Path(line(0.0), "solid", 3)]
    tagged = P.tag(original, "tpu")
    assert [p.kind for p in tagged] == ["tpu"]
    assert original[0].kind == "solid"
    assert tagged[0].path_id == 3
    assert tagged[0].nodes == pytest.approx(original[0].nodes)


def test_tag_accepts_bare_arrays():
    assert P.tag([line(0.0)], "support")[0].kind == "support"


# ── Connecting ───────────────────────────────────────────────────────────────

PITCH = 0.001
TOL = 1.1 * PITCH


def cut_serpentine(rows=6, seed=0):
    """A boustrophedon chopped into overlapping thirds, then shuffled."""
    pieces = []
    for r in range(rows):
        xs = onp.linspace(0.0, 0.03, 31)
        for a, b in [(0, 11), (10, 21), (20, 31)]:
            seg = onp.stack([xs[a:b], onp.full(b - a, PITCH * r)], axis=1)
            pieces.append(seg[::-1] if (r + a) % 2 else seg)
    onp.random.default_rng(seed).shuffle(pieces)
    return [P.Path(q, "path", i) for i, q in enumerate(pieces)]


def test_connect_rebuilds_a_cut_serpentine():
    pieces = cut_serpentine()
    joined = P.connect(P.order_paths(pieces), tolerance=TOL)
    assert len(joined) == 1
    # 180 mm of path plus five 1 mm row transitions.
    assert joined[0].length == pytest.approx(0.185)


def test_connect_never_bridges_further_than_the_tolerance():
    joined = P.connect(cut_serpentine(), tolerance=TOL)
    longest = max(float(onp.linalg.norm(d))
                  for p in joined for d in onp.diff(p.nodes, axis=0))
    assert longest <= TOL + 1e-12


def test_connect_is_deterministic_for_a_fixed_order():
    """It consumes the order it is given -- the same order must weld the same."""
    ordered = P.order_paths(cut_serpentine())
    runs = [sorted(round(p.length, 9) for p in P.connect(ordered, tolerance=TOL))
            for _ in range(3)]
    assert all(r == runs[0] for r in runs)


def test_connect_removes_travel_that_ordering_could_only_shorten():
    ordered = P.order_paths(cut_serpentine())
    joined = P.connect(ordered, tolerance=TOL)
    assert P.travel_distance(ordered) > 0.0
    assert P.travel_distance(P.order_paths(joined)) < 0.1 * P.travel_distance(ordered)


def test_connect_will_not_merge_across_kinds():
    pieces = cut_serpentine()
    mixed = P.tag(pieces[:9], "a") + P.tag(pieces[9:], "b")
    joined = P.connect(P.order_paths(mixed), tolerance=TOL)
    assert {p.kind for p in joined} == {"a", "b"}
    assert len(joined) > len(P.connect(P.order_paths(pieces), tolerance=TOL))


def test_connect_leaves_closed_paths_alone():
    circle = P.Path(loop(cx=0.05, cy=0.0, r=0.002), "path", 99)
    joined = P.connect(P.order_paths(cut_serpentine() + [circle]),
                       tolerance=TOL)
    kept = [p for p in joined if p.is_closed]
    assert len(kept) == 1
    assert kept[0].length == pytest.approx(circle.length)


def test_connect_joins_nothing_at_zero_tolerance():
    pieces = P.order_paths(cut_serpentine())
    assert len(P.connect(pieces, tolerance=0.0)) == len(pieces)


def test_connect_conserves_nothing_but_never_loses_a_piece():
    """Bridges add length; no original path may vanish."""
    pieces = cut_serpentine()
    joined = P.connect(P.order_paths(pieces), tolerance=TOL)
    before = sum(p.length for p in pieces)
    after = sum(p.length for p in joined)
    assert after >= before
    assert after - before <= TOL * len(pieces)


def ordered_square(gap=0.0008):
    return P.order_paths(square_with_a_gap(gap))


def square_with_a_gap(gap=0.0008):
    corners = [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01), (0, gap)]
    return [P.Path(onp.array([corners[i], corners[i + 1]], float))
            for i in range(len(corners) - 1)]


def test_close_loops_is_opt_in():
    assert not P.connect(ordered_square(), tolerance=TOL)[0].is_closed
    closed = P.connect(ordered_square(), tolerance=TOL, close_loops=True)[0]
    assert closed.is_closed


def test_close_loops_does_not_duplicate_an_already_closed_end():
    # Given in order, so the run really is a circuit back to its own start.
    exact = square_with_a_gap(gap=0.0)
    plain = P.connect(exact, tolerance=TOL)[0]
    asked = P.connect(exact, tolerance=TOL, close_loops=True)[0]
    assert plain.is_closed
    assert len(asked.nodes) == len(plain.nodes)


def test_connect_keeps_the_first_piece_identity():
    joined = P.connect(P.order_paths(cut_serpentine()), tolerance=TOL)
    assert joined[0].kind == "path"
    assert joined[0].path_id is not None


# ── Simplifying ──────────────────────────────────────────────────────────────

def test_simplify_collapses_a_straight_run_to_its_ends():
    # 20 collinear points carry the same information as 2.
    out = P.simplify([line(0.0)], tolerance=1e-9)
    assert len(out[0].nodes) == 2
    assert out[0].nodes[0] == pytest.approx([0.0, 0.0])
    assert out[0].nodes[-1] == pytest.approx([0.05, 0.0])


def test_simplify_keeps_every_vertex_within_the_tolerance():
    t = onp.linspace(0.0, 1.0, 400)
    wiggly = onp.stack([t * 0.05, 0.002 * onp.sin(12.0 * t)], axis=1)
    tol = 1e-4
    out = P.simplify([wiggly], tolerance=tol)[0].nodes
    assert len(out) < len(wiggly)

    a, b = out[:-1], out[1:]
    ab = b - a
    ap = wiggly[:, None, :] - a[None, :, :]
    u = onp.clip((ap * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    d = onp.linalg.norm(ap - u[..., None] * ab, axis=-1).min(axis=1)
    assert d.max() <= tol


def test_simplify_keeps_a_loop_closed():
    out = P.simplify([loop()], tolerance=1e-4)[0]
    assert out.is_closed
    assert out.nodes[0] == pytest.approx(out.nodes[-1])
    assert len(out.nodes) < len(loop())


def test_simplify_preserves_order_kind_and_endpoints():
    src = P.tag([line(0.0)], "a") + P.tag([loop()], "b")
    out = P.simplify(src, tolerance=1e-4)
    assert [p.kind for p in out] == ["a", "b"]
    for got, want in zip(out, src, strict=True):
        assert got.start == pytest.approx(want.start)
        assert got.end == pytest.approx(want.end)


def test_simplify_leaves_short_paths_and_zero_tolerance_alone():
    two = onp.array([[0.0, 0.0], [0.01, 0.0]])
    assert len(P.simplify([two], tolerance=1.0)[0].nodes) == 2
    assert len(P.simplify([line(0.0)], tolerance=0.0)[0].nodes) == 20


def test_simplify_rejects_a_negative_tolerance():
    with pytest.raises(ValueError, match="must not be negative"):
        P.simplify([line(0.0)], tolerance=-1e-6)


def test_path_lengths_takes_path_objects_too():
    src = [line(0.0, 0.0, 0.05)]
    assert P.path_lengths(src) == pytest.approx([0.05])
    assert P.path_lengths(P.tag(src, "a")) == pytest.approx([0.05])


# ── Hairpins ─────────────────────────────────────────────────────────────────

def hairpin(gap=0.0007, leg=0.03, n=16, spread=0.004, m=30):
    """A half turn of radius ``gap/2`` with two legs running back out of it.

    The legs diverge, as a stripe pattern's do: a pair that stayed exactly
    parallel could never be pulled a bead's width apart, and trimming would
    only ever hit its ``max_trim`` bound.
    """
    t = onp.linspace(-onp.pi / 2, onp.pi / 2, n)
    turn = onp.stack([0.5 * gap * onp.cos(t), 0.5 * gap * onp.sin(t)], axis=1)
    x = onp.linspace(0.0, -leg, m)
    dy = onp.linspace(0.0, spread, m)
    lower = onp.stack([x, -0.5 * gap - dy], axis=1)[::-1]
    upper = onp.stack([x, 0.5 * gap + dy], axis=1)
    return onp.vstack([lower, turn, upper])


def test_turn_radius_matches_a_known_circle():
    t = onp.linspace(0.0, 2.0 * onp.pi, 200)
    circle = onp.stack([0.007 * onp.cos(t), 0.007 * onp.sin(t)], axis=1)
    assert P.turn_radius(circle) == pytest.approx(0.007, rel=1e-3)
    assert onp.isinf(P.turn_radius(line(0.0))).all()
    assert len(P.turn_radius(onp.zeros((2, 2)))) == 0


def test_trim_hairpins_cuts_the_turn_and_leaves_the_legs():
    out = P.trim_hairpins([hairpin()], radius=0.001)
    assert len(out) == 2
    for piece in out:
        assert P.turn_radius(piece.nodes).min(initial=onp.inf) >= 0.001
    # Nothing moved: every surviving vertex is one of the originals.
    src = {tuple(q) for q in hairpin()}
    assert all(tuple(q) in src for piece in out for q in piece.nodes)


def test_trim_hairpins_opens_a_gap_of_one_clearance():
    out = P.trim_hairpins([hairpin()], radius=0.001, clearance=0.002)
    assert len(out) == 2
    assert onp.linalg.norm(out[1].start - out[0].end) >= 0.002


def test_trim_hairpins_leaves_a_gentle_path_untouched():
    t = onp.linspace(0.0, onp.pi, 60)
    arc = onp.stack([0.01 * onp.cos(t), 0.01 * onp.sin(t)], axis=1)
    out = P.trim_hairpins([arc], radius=0.001)
    assert len(out) == 1
    assert out[0].nodes == pytest.approx(arc)


def test_trim_hairpins_keeps_the_kind_and_drops_stubs():
    out = P.trim_hairpins(P.tag([hairpin()], "fibre"), radius=0.001)
    assert {p.kind for p in out} == {"fibre"}
    # A short leg is a dot of plastic plus a travel move, so it goes.
    short = P.trim_hairpins([hairpin(leg=0.0008, spread=0.0002, m=4)],
                            radius=0.001, min_length=0.01)
    assert short == []


def test_trim_hairpins_stops_trimming_at_max_trim():
    # Legs that never reach the asked-for clearance must not be eaten whole:
    # the walk-back is bounded and what is left still gets printed.
    straight = hairpin(spread=0.0)
    out = P.trim_hairpins([straight], radius=0.001, clearance=0.05,
                          max_trim=0.002, min_length=0.0)
    assert len(out) == 2
    assert all(p.length > 0.02 for p in out)


def test_trim_hairpins_rejects_a_non_positive_radius():
    with pytest.raises(ValueError, match="radius must be positive"):
        P.trim_hairpins([hairpin()], radius=0.0)


def test_trim_hairpins_does_not_cut_a_closed_path_at_its_seam():
    # A closed path repeats its first vertex, so the first and last pieces run
    # into each other through it.  Emitted separately they would be two paths
    # whose ends sit on top of each other -- 38 of them on a real part.  One
    # hairpin, well away from the seam, therefore has to give back one piece.
    t = onp.linspace(0.0, 2.0 * onp.pi, 121)
    loop = onp.stack([0.01 * onp.cos(t), 0.01 * onp.sin(t)], axis=1)
    loop[-1] = loop[0]
    # A *shallow* spike is the sharp one: the turn radius goes as half the leg
    # length, so pulling the vertex 1 mm in off a 10 mm circle gives ~0.5 mm,
    # while pulling it 4 mm in gives 2 mm and is not a hairpin at all.
    loop[60] = loop[60] * 0.9
    assert P.Path(loop).is_closed
    assert (P.turn_radius(loop) < 0.001).sum() > 0

    out = P.trim_hairpins([loop], radius=0.001)
    assert len(out) == 1
    assert P.turn_radius(out[0].nodes).min(initial=onp.inf) >= 0.001


def test_trim_hairpins_finds_a_hairpin_sitting_on_the_seam():
    # `turn_radius` reads interior vertices, so a closed path whose sharpest
    # turn is its own first vertex hides it from the threshold entirely.
    t = onp.linspace(0.0, 2.0 * onp.pi, 61)
    loop = onp.stack([0.01 * onp.cos(t), 0.01 * onp.sin(t)], axis=1)
    loop[-1] = loop[0]
    # Pinch the seam into a hairpin by pulling its neighbours in.
    loop[1] = loop[0] + onp.array([0.0002, -0.0003])
    loop[-2] = loop[0] + onp.array([0.0002, 0.0003])
    out = P.trim_hairpins([loop], radius=0.001)
    assert all(P.turn_radius(p.nodes).min(initial=onp.inf) >= 0.001 for p in out)


# ── Placing the free ends ────────────────────────────────────────────────────

def test_spread_ends_moves_an_end_away_from_its_neighbours():
    # One path's end sits beside a wall of another's nodes; it must step away
    # from the wall, not along it.
    wall = onp.stack([onp.full(20, 0.001), onp.linspace(-0.01, 0.01, 20)], axis=1)
    stub = onp.array([[-0.01, 0.0], [0.0, 0.0]])
    out = P.spread_ends([stub, wall], move=0.0002, radius=0.01)
    moved = out[0].end
    assert moved[0] < stub[-1][0]                       # away from the wall
    assert onp.linalg.norm(moved - stub[-1]) == pytest.approx(0.0002, rel=1e-2)


def test_spread_ends_leaves_closed_paths_and_isolated_paths_alone():
    out = P.spread_ends([loop()], move=0.0002)
    assert out[0].nodes == pytest.approx(loop())
    lonely = P.spread_ends([line(0.0)], move=0.0002, radius=1e-6)
    assert lonely[0].nodes == pytest.approx(line(0.0))


def test_spread_ends_ignores_the_ends_own_tail():
    # Without `skip` the end runs forward along its own path, undoing the trim.
    stub = onp.stack([onp.linspace(0.0, 0.01, 30), onp.zeros(30)], axis=1)
    out = P.spread_ends([stub], move=0.0002, radius=0.02, skip=0.02)
    assert out[0].nodes == pytest.approx(stub)


def test_spread_ends_rejects_a_non_positive_move():
    with pytest.raises(ValueError, match="move must be positive"):
        P.spread_ends([line(0.0)], move=0.0)


# ── SVG round trip ───────────────────────────────────────────────────────────

def test_svg_round_trip_recovers_geometry_and_kind(tmp_path):
    fib = P.tag([line(0.001 * k) for k in range(5)], "solid")
    con = P.tag([loop()], "outline")
    svg = tmp_path / "p.svg"
    P.write_svg(fib + con, svg)

    back = P.read_svg(svg)
    assert len(back) == 6
    assert [p.kind for p in back] == ["solid"] * 5 + ["outline"]
    for got, want in zip(back, [p.nodes for p in fib + con], strict=True):
        # The file carries 10 significant digits, so that is the floor.
        assert got.nodes == pytest.approx(want, rel=1e-9, abs=1e-11)
    assert back[-1].is_closed


def test_read_svg_needs_no_out_of_band_scale(tmp_path):
    """data-scale and the viewBox make the file self-describing."""
    svg = tmp_path / "p.svg"
    P.write_svg([line(0.002)], svg, scale=1.0)       # metres in the file
    assert P.read_svg(svg)[0].nodes == pytest.approx(line(0.002), abs=1e-9)


def test_arbitrary_kinds_round_trip(tmp_path):
    """Kinds are the caller's vocabulary, not this package's two."""
    kinds = ["pla_white", "tpu", "support", "conductive"]
    mixed = [P.Path(line(0.001 * (4 * k + i)), kinds[k], 4 * k + i)
             for k in range(4) for i in range(3)]
    svg = tmp_path / "mm.svg"
    P.write_svg(mixed, svg)

    text = svg.read_text()
    for k in kinds:
        assert f'<g id="{k}"' in text
    back = P.read_svg(svg)
    assert [p.kind for p in back] == [p.kind for p in mixed]
    assert len({p.kind for p in back}) == 4


def test_each_kind_gets_its_own_colour(tmp_path):
    svg = tmp_path / "mm.svg"
    P.write_svg([P.Path(line(0.0), "a"), P.Path(line(0.001), "b")], svg)
    used = [ln.split('stroke="')[1].split('"')[0]
            for ln in svg.read_text().splitlines() if "<g " in ln]
    assert used == list(P.KIND_COLOURS[:2])


def test_stroke_can_be_given_per_kind(tmp_path):
    svg = tmp_path / "mm.svg"
    P.write_svg([P.Path(line(0.0), "a"), P.Path(line(0.001), "b")], svg,
                stroke={"a": "#111111", "b": "#222222"})
    assert '<g id="a" fill="none" stroke="#111111"' in svg.read_text()
    with pytest.raises(ValueError, match="no stroke"):
        P.write_svg([P.Path(line(0.0), "a")], svg, stroke={"b": "#111111"})


def test_too_many_kinds_warns_rather_than_reusing_a_colour(tmp_path):
    many = [P.Path(line(0.001 * k), f"m{k}", k)
            for k in range(len(P.KIND_COLOURS) + 2)]
    with pytest.warns(UserWarning, match="distinct colours"):
        P.write_svg(many, tmp_path / "mm.svg")
    text = (tmp_path / "mm.svg").read_text()
    for colour in P.KIND_COLOURS:
        assert text.count(f'stroke="{colour}"') == 1


def test_a_kind_that_cannot_be_an_xml_id_is_refused(tmp_path):
    with pytest.raises(ValueError, match="SVG group id"):
        P.write_svg([P.Path(line(0.0), "two words")], tmp_path / "x.svg")


def test_ungrouped_polylines_take_the_default_kind(tmp_path):
    svg = tmp_path / "loose.svg"
    P.write_svg([line(0.0)], svg)
    svg.write_text(svg.read_text().replace('id="path"', 'id="renamed"'))
    assert P.read_svg(svg)[0].kind == "renamed"
    assert P.Path(line(0.0)).kind == "path"


def test_read_svg_warns_about_what_it_skipped(tmp_path):
    svg = tmp_path / "p.svg"
    P.write_svg([line(0.0)], svg)
    text = svg.read_text().replace(
        "</svg>", '  <g id="notes"><path d="M0 0 C1 1 2 2 3 3"/></g>\n</svg>')
    svg.write_text(text)
    with pytest.warns(UserWarning, match="polyline"):
        got = P.read_svg(svg)
    assert len(got) == 1                             # the curve is not a toolpath


# ── DXF export ───────────────────────────────────────────────────────────────

def dxf_pairs(text):
    """DXF as a flat list of (group code, value) -- enough to check a writer."""
    lines = text.splitlines()
    return [(int(lines[i]), lines[i + 1]) for i in range(0, len(lines) - 1, 2)]


def dxf_polylines(text):
    """Layer name and vertex array for each POLYLINE, plus its closed flag."""
    out, cur, layer, closed = [], None, None, False
    for code, value in dxf_pairs(text):
        if code == 0 and value == "POLYLINE":
            cur, closed = [], False
        elif code == 0 and value == "SEQEND":
            out.append((layer, onp.array(cur, dtype=float), closed))
            cur = None
        elif cur is not None and code == 0 and value == "VERTEX":
            cur.append([None, None])
        elif cur is not None and code == 10:
            cur[-1][0] = float(value)
        elif cur is not None and code == 20:
            cur[-1][1] = float(value)
        elif cur is not None and not cur and code == 70:
            closed = value == "1"
        elif cur is not None and not cur and code == 8:
            layer = value
    return out


def test_dxf_writes_one_polyline_per_path_without_the_y_flip(tmp_path):
    src = [line(0.001 * k) for k in range(3)]
    f = tmp_path / "p.dxf"
    P.write_dxf(src, f)

    got = dxf_polylines(f.read_text())
    assert len(got) == 3
    for (_, nodes, _), want in zip(got, src, strict=True):
        # Metres to millimetres, y unchanged -- DXF counts y upward like the
        # mesh, so unlike SVG there is nothing to undo.
        assert nodes == pytest.approx(want * 1000.0, rel=1e-9, abs=1e-11)


def test_dxf_gives_each_kind_its_own_layer(tmp_path):
    f = tmp_path / "p.dxf"
    P.write_dxf(P.tag([line(0.0)], "solid") + P.tag([line(0.002)], "outline"), f)

    text = f.read_text()
    assert [layer for layer, _, _ in dxf_polylines(text)] == ["solid", "outline"]
    names = [v for i, (c, v) in enumerate(dxf_pairs(text))
             if c == 2 and dxf_pairs(text)[i - 1] == (0, "LAYER")]
    assert names == ["solid", "outline"]


def test_dxf_closes_loops_with_the_flag_not_a_repeated_vertex(tmp_path):
    f = tmp_path / "p.dxf"
    P.write_dxf([loop()], f)

    (_, nodes, closed), = dxf_polylines(f.read_text())
    assert closed
    # A repeated first vertex would leave CAM a zero-length segment.
    assert len(nodes) == len(loop()) - 1
    assert nodes[0] != pytest.approx(nodes[-1])


def test_dxf_scale_is_honoured(tmp_path):
    f = tmp_path / "p.dxf"
    P.write_dxf([line(0.002)], f, scale=1.0)
    (_, nodes, _), = dxf_polylines(f.read_text())
    assert nodes == pytest.approx(line(0.002), rel=1e-9, abs=1e-11)


def test_dxf_colours_cycle_per_kind_and_can_be_overridden(tmp_path):
    f = tmp_path / "p.dxf"
    P.write_dxf(P.tag([line(0.0)], "a") + P.tag([line(0.002)], "b"), f)
    aci = [int(v) for c, v in dxf_pairs(f.read_text()) if c == 62]
    assert aci == [P.KIND_ACI[0], P.KIND_ACI[1]]

    P.write_dxf(P.tag([line(0.0)], "a") + P.tag([line(0.002)], "b"), f,
                layer_colours={"a": 7, "b": 7})
    assert [int(v) for c, v in dxf_pairs(f.read_text()) if c == 62] == [7, 7]

    with pytest.raises(ValueError, match="no entry for"):
        P.write_dxf(P.tag([line(0.0)], "a"), f, layer_colours={"b": 7})


def test_dxf_rejects_a_kind_that_is_not_a_layer_name(tmp_path):
    for bad in ("two words", "a/b", "", "x" * 32):
        with pytest.raises(ValueError, match="DXF layer name"):
            P.write_dxf([P.Path(line(0.0), bad)], tmp_path / "x.dxf")


def test_dxf_warns_on_an_empty_drawing(tmp_path):
    with pytest.warns(UserWarning, match="no paths"):
        P.write_dxf([], tmp_path / "x.dxf")


def test_dxf_keeps_a_connected_bridge_as_ordinary_geometry(tmp_path):
    # `connect` welds two contours into one path; the bridge between them is a
    # printed move, so it has to leave as vertices like any other.
    joined = P.connect(P.tag([line(0.0), line(0.0005)[::-1]], "fibre"),
                       tolerance=0.01)
    assert len(joined) == 1
    f = tmp_path / "p.dxf"
    P.write_dxf(joined, f)

    (_, nodes, _), = dxf_polylines(f.read_text())
    assert nodes == pytest.approx(joined[0].nodes * 1000.0, rel=1e-9, abs=1e-11)


# ── STEP export ──────────────────────────────────────────────────────────────

def step_instances(text):
    """``{"#7": ("POLYLINE", "...params...")}`` -- enough to check a writer."""
    out = {}
    for raw in text.splitlines():
        if not raw.startswith("#") or "=" not in raw:
            continue
        ref, body = raw.split("=", 1)
        body = body.rstrip(";")
        if "(" not in body:
            continue
        name, _, params = body.partition("(")
        out[ref] = (name, params.rsplit(")", 1)[0])
    return out


def step_points(text, poly_ref):
    """The xyz of each CARTESIAN_POINT a POLYLINE refers to, in order."""
    inst = step_instances(text)
    refs = inst[poly_ref][1].split("(", 1)[1].rstrip(")").split(",")
    return onp.array([
        [float(v) for v in inst[r.strip()][1].split("(")[1].rstrip(")").split(",")]
        for r in refs])


def test_step_writes_one_polyline_per_path_at_the_given_z(tmp_path):
    src = [line(0.001 * k) for k in range(3)]
    f = tmp_path / "p.step"
    P.write_step(src, f, z=0.2)

    text = f.read_text()
    polys = [r for r, (n, _) in step_instances(text).items() if n == "POLYLINE"]
    assert len(polys) == 3
    for ref, want in zip(polys, src, strict=True):
        xyz = step_points(text, ref)
        assert xyz[:, :2] == pytest.approx(want * 1000.0, rel=1e-9, abs=1e-11)
        assert xyz[:, 2] == pytest.approx(0.2)


def test_step_gives_each_kind_its_own_curve_set(tmp_path):
    f = tmp_path / "p.step"
    P.write_step(P.tag([line(0.0)], "solid") + P.tag([line(0.002)], "outline"), f)
    sets = [params for _, (n, params) in step_instances(f.read_text()).items()
            if n == "GEOMETRIC_CURVE_SET"]
    assert len(sets) == 2
    assert sets[0].startswith("'solid'") and sets[1].startswith("'outline'")


def test_step_declares_the_unit_it_was_scaled_to(tmp_path):
    f = tmp_path / "p.step"
    P.write_step([line(0.0)], f)
    assert "SI_UNIT(.MILLI.,.METRE.)" in f.read_text()

    P.write_step([line(0.0)], f, scale=1.0, units="m")
    assert "SI_UNIT($,.METRE.)" in f.read_text()

    with pytest.raises(ValueError, match="units must be one of"):
        P.write_step([line(0.0)], f, units="inch")


def test_step_reals_always_carry_a_decimal_point(tmp_path):
    # `1` is an integer in Part 21 and a conforming reader rejects it where a
    # real is required, so a whole-number coordinate has to come out as `1.`.
    f = tmp_path / "p.step"
    P.write_step([onp.array([[0.001, 0.002], [0.003, 0.004]])], f)
    for nm, params in step_instances(f.read_text()).values():
        if nm == "CARTESIAN_POINT":
            for v in params.split("(")[1].rstrip(")").split(","):
                assert "." in v or "E" in v.upper(), v


def test_step_has_the_product_skeleton_cad_needs(tmp_path):
    f = tmp_path / "p.step"
    P.write_step([line(0.0)], f, name="beam")
    names = {n for n, _ in step_instances(f.read_text()).values()}
    assert {"PRODUCT", "PRODUCT_DEFINITION", "SHAPE_DEFINITION_REPRESENTATION",
            "GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION",
            "APPLICATION_PROTOCOL_DEFINITION"} <= names
    text = f.read_text()
    assert text.startswith("ISO-10303-21;")
    assert text.rstrip().endswith("END-ISO-10303-21;")
    assert "'beam'" in text


def test_step_escapes_a_quote_in_a_kind(tmp_path):
    f = tmp_path / "p.step"
    P.write_step([P.Path(line(0.0), "o'brien")], f)
    assert "'o''brien'" in f.read_text()


def test_step_warns_on_an_empty_drawing(tmp_path):
    with pytest.warns(UserWarning, match="no paths"):
        P.write_step([], tmp_path / "x.step")


# ── FullControl handoff ──────────────────────────────────────────────────────

fc = pytest.importorskip("fullcontrol")


def test_to_fullcontrol_emits_travel_then_extrude():
    steps = P.to_fullcontrol([line(0.0, 0.0, 0.05, n=3)], z=0.2, annotate=False)
    kinds = [type(s).__name__ for s in steps]
    assert kinds == ["Extruder", "Point", "Extruder", "Point", "Point"]
    assert steps[0].on is False and steps[2].on is True
    assert steps[1].x == pytest.approx(0.0)          # 1000x scale: m -> mm
    assert steps[-1].x == pytest.approx(50.0)
    assert steps[-1].z == pytest.approx(0.2)


def test_path_steps_is_one_path_on_its_own():
    steps = P.path_steps(P.Path(line(0.0, 0.0, 0.05, n=3), "a", 4), z=0.2)
    assert [type(s).__name__ for s in steps] == [
        "GcodeComment", "Extruder", "Point", "Extruder", "Point", "Point"]
    assert steps[0].text == "a path 4"
    assert steps[2].x == pytest.approx(0.0)          # 1000x scale: m -> mm
    assert steps[-1].x == pytest.approx(50.0)


def test_path_steps_can_skip_the_approach():
    """After a weld the head is already there; a travel would be spurious."""
    steps = P.path_steps(line(0.0, 0.0, 0.05, n=3), z=0.2, travel=False,
                         annotate=False)
    assert [type(s).__name__ for s in steps] == ["Point"] * 3
    assert steps[0].x == pytest.approx(0.0)          # start is kept


def test_retract_wraps_the_travel():
    steps = P.path_steps(line(0.0), z=0.2, retract=True, annotate=False)
    kinds = [type(s).__name__ for s in steps]
    assert kinds[:3] == ["Extruder", "PrinterCommand", "Point"]
    assert steps[1].id == "retract"
    assert [s.id for s in steps if type(s).__name__ == "PrinterCommand"] == [
        "retract", "unretract"]
    # Unretract lands after the approach and before printing resumes.
    un = next(i for i, s in enumerate(steps)
              if getattr(s, "id", None) == "unretract")
    assert type(steps[un - 1]).__name__ == "Point"
    assert type(steps[un + 1]).__name__ == "Extruder" and steps[un + 1].on


def test_hop_lifts_crosses_and_descends():
    steps = P.path_steps(line(0.0), z=0.2, hop=0.6, annotate=False)
    zs = [(s.x, s.z) for s in steps if type(s).__name__ == "Point"][:3]
    assert zs[0] == (None, pytest.approx(0.8))       # straight up, x carries over
    assert zs[1][1] == pytest.approx(0.8)            # cross at height
    assert zs[2] == (None, pytest.approx(0.2))       # back down in place


def test_no_travel_means_no_retract_or_hop():
    steps = P.path_steps(line(0.0), z=0.2, travel=False, retract=True, hop=0.6,
                         annotate=False)
    assert all(type(s).__name__ == "Point" for s in steps)
    assert all(s.z == pytest.approx(0.2) for s in steps)


def test_the_first_path_gets_no_hop():
    """A bare z move at the head of a design has no x or y to carry over."""
    job = [P.Path(line(0.0), "a", 0), P.Path(line(0.001), "a", 1)]
    steps = P.to_fullcontrol(job, z=0.2, hop=0.6)
    lifts = [s for s in steps
             if type(s).__name__ == "Point" and s.x is None and s.y is None]
    assert len(lifts) == 2                           # one lift + one descent
    assert all(s.z is not None for s in lifts)
    first_point = next(s for s in steps if type(s).__name__ == "Point")
    assert first_point.x is not None and first_point.y is not None


def test_retract_and_hop_reach_the_gcode():
    job = [P.Path(line(0.0), "a", 0), P.Path(line(0.001), "a", 1)]
    steps = P.to_fullcontrol(job, z=0.2, print_speed=1000, retract=True, hop=0.6)
    gcode = fc.transform(steps, "gcode",
                         fc.GcodeControls(printer_name="generic"),
                         show_tips=False)
    lines = [ln.split(";")[0].strip() for ln in gcode.splitlines()]
    # The first G10 belongs to path 0, which gets no hop; path 1 is the one
    # with the full lift-cross-descend.
    i = len(lines) - 1 - lines[::-1].index("G10")
    assert lines[i:i + 5] == ["G10", "G0 F8000 Z0.8", "G0 X0 Y1", "G0 Z0.2",
                              "G11"]


def test_path_steps_label_overrides_the_comment():
    steps = P.path_steps(line(0.0), z=0.2, label="purge line")
    assert steps[0].text == "purge line"


def test_building_a_design_by_hand_matches_to_fullcontrol():
    """The bulk call is the per-path call in a loop -- nothing more."""
    job = [P.Path(line(0.0), "a", 0), P.Path(line(0.001), "a", 1)]
    bulk = P.to_fullcontrol(job, z=0.2, width=0.4, print_speed=1000)
    byhand = [fc.ExtrusionGeometry(width=0.4), fc.Printer(print_speed=1000)]
    for path in job:
        byhand += P.path_steps(path, z=0.2)
    assert [type(s).__name__ for s in byhand] == [
        type(s).__name__ for s in bulk]
    assert [getattr(s, "text", None) for s in byhand] == [
        getattr(s, "text", None) for s in bulk]


def test_per_path_loop_can_interleave_other_steps():
    fast = fc.Printer(print_speed=3000)
    steps = [fc.ExtrusionGeometry(width=0.4)]
    for path in [P.Path(line(0.0), "a", 0), P.Path(line(0.001), "support", 1)]:
        if path.kind == "support":
            steps.append(fast)
        steps += P.path_steps(path, z=0.2)
    speeds = [i for i, s in enumerate(steps) if type(s).__name__ == "Printer"]
    comments = [i for i, s in enumerate(steps)
                if type(s).__name__ == "GcodeComment"]
    assert len(speeds) == 1
    assert comments[0] < speeds[0] < comments[1]     # fired before the support


def test_annotation_carries_the_kind_into_the_design():
    steps = P.to_fullcontrol([P.Path(loop(), "outline", 7)], z=0.2)
    comments = [s.text for s in steps if type(s).__name__ == "GcodeComment"]
    assert comments == ["outline path 7 (closed)"]


def test_on_kind_fires_once_per_kind_when_ordered():
    kinds = ["a", "b", "c"]
    mixed = [P.Path(line(0.001 * (3 * k + i)), kinds[k], 3 * k + i)
             for k in range(3) for i in range(3)]
    steps = P.to_fullcontrol(P.order_paths(mixed), z=0.2, on_kind={
        "a": P.tool_change(0), "b": P.tool_change(1),
        "c": P.tool_change("M280 P0 S160")})
    changes = [s.text for s in steps if type(s).__name__ == "ManualGcode"]
    assert changes == ["T0", "T1", "M280 P0 S160"]


def test_on_kind_is_not_only_for_tools():
    """Any steps at all -- a kind is a group, not a tool."""
    steps = P.to_fullcontrol([P.Path(line(0.0), "wide"), P.Path(line(0.001), "a")],
                             z=0.2, annotate=False,
                             on_kind={"wide": [fc.ExtrusionGeometry(width=0.8)]})
    widths = [s.width for s in steps
              if type(s).__name__ == "ExtrusionGeometry"]
    assert widths == [0.8]
    assert steps[0].width == 0.8                     # emitted before the travel


def test_on_kind_fires_again_when_a_kind_recurs():
    """Emission is on change, so an unordered design pays per run."""
    alternating = [P.Path(line(0.001 * i), "a" if i % 2 else "b", i)
                   for i in range(4)]
    steps = P.to_fullcontrol(alternating, z=0.2,
                             on_kind={"a": P.tool_change(0)})
    assert sum(type(s).__name__ == "ManualGcode" for s in steps) == 2


def test_relative_e_resets_the_datum_at_each_tool_change():
    """FullControl tracks one extruder, so absolute E runs on across a change."""
    ps = [P.Path(line(0.0), "a", 0), P.Path(line(0.001), "b", 1)]
    controls = fc.GcodeControls(printer_name="toolchanger_T0",
                                initialization_data={"relative_e": False})

    def on_kind(relative_e):
        return {"a": P.tool_change(0, relative_e=relative_e),
                "b": P.tool_change(1, relative_e=relative_e)}

    def e_per_path(steps):
        """E words of each path's extruding moves, keyed by the path comment."""
        gcode = fc.transform(steps, "gcode", controls, show_tips=False)
        out, here = {}, None
        for ln in gcode.splitlines():
            if ln.startswith("; ") and " path " in ln:
                here = out.setdefault(ln[2:], [])
            elif here is not None and ln.startswith("G1 "):
                here += [float(w[1:]) for w in ln.split() if w.startswith("E")]
        return out

    kwargs = dict(z=0.2, width=0.4, height=0.2)
    ran_on = e_per_path(P.to_fullcontrol(ps, **kwargs, on_kind=on_kind(None)))
    reset = e_per_path(P.to_fullcontrol(ps, **kwargs, on_kind=on_kind(False)))

    # Without the reset, tool 1 picks up where tool 0's total left off.
    assert min(ran_on["b path 1"]) > max(ran_on["a path 0"])
    # With it, each tool starts from its own zero, so the two identical-length
    # paths extrude identical E words.
    assert reset["b path 1"] == pytest.approx(reset["a path 0"])
    assert reset["a path 0"][0] < 1.0


def test_tool_change_without_relative_e_is_just_the_t_word():
    assert [type(s).__name__ for s in P.tool_change(1)] == ["ManualGcode"]
    assert [type(s).__name__ for s in P.tool_change(1, relative_e=True)] == [
        "ManualGcode", "Extruder"]


def test_a_kind_with_no_entry_gets_nothing():
    steps = P.to_fullcontrol([P.Path(line(0.0), "a"), P.Path(line(0.001), "b")],
                             z=0.2, on_kind={"a": P.tool_change(0)})
    assert [s.text for s in steps
            if type(s).__name__ == "ManualGcode"] == ["T0"]


def test_multiple_kinds_alone_change_nothing():
    """A multi-kind design is not by itself a multi-tool design."""
    plain = P.to_fullcontrol([P.Path(line(0.0), "a")], z=0.2)
    many = P.to_fullcontrol([P.Path(line(0.0), "a"), P.Path(line(0.001), "b")],
                            z=0.2)
    assert not any(type(s).__name__ == "ManualGcode" for s in many)
    assert [type(s).__name__ for s in many[:len(plain)]] == [
        type(s).__name__ for s in plain]


def test_design_reaches_gcode():
    """Straight into fullcontrol's own API -- nothing here wraps it."""
    steps = P.to_fullcontrol([line(0.0, 0.0, 0.05, n=3)], z=0.2, print_speed=1000)
    gcode = fc.transform(steps, "gcode",
                         fc.GcodeControls(printer_name="generic"),
                         show_tips=False)
    assert "; path path 0" in gcode
    assert any(ln.startswith("G1 ") for ln in gcode.splitlines())


def test_design_survives_fullcontrols_own_json_round_trip(tmp_path):
    steps = P.to_fullcontrol(P.order_paths(scrambled_lines(3)), z=0.2,
                             width=0.4, height=0.2, print_speed=1000)
    fc.export_design(steps, str(tmp_path / "design"))
    back = fc.import_design(fc, str(tmp_path / "design"))
    assert [type(s).__name__ for s in back] == [type(s).__name__ for s in steps]
    pt = [(s.x, s.y, s.z) for s in back if type(s).__name__ == "Point"]
    assert pt == [(s.x, s.y, s.z) for s in steps if type(s).__name__ == "Point"]


