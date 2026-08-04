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


