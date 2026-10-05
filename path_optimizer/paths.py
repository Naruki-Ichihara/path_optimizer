"""Stripe field → print paths → FullControl.

The last stage: turn the relaxed Swift–Hohenberg field into polylines a printer
can follow, and hand them to FullControl as a design.

.. code-block:: text

    StripeField --extract_paths--> polylines --to_fullcontrol--> fc steps
                                       |
                                       +--write_svg--> .svg    (edit, inspect)
                                       +--write_dxf--> .dxf    (CAD, CAM)
                                       +--write_step-> .step   (CAD)

:func:`simplify` goes before any of the three file writers.  Contours come off
the stripe mesh at about one vertex per element, which is ~20x more than the
geometry needs; on a real part 0.05 mm of tolerance removes 95% of them and
moves nothing measurably.

The path skeleton is the **zero level set** of ``u``.  Crests and troughs are
where material would go if you were printing the stripe pattern itself, but a
toolpath is a curve, not a band, and the zeros sit centred between them.

.. warning::

   ``cos(φ)`` crosses zero every **half** period, so the extracted paths are
   spaced at ``stripe_period / 2`` — **not** at ``stripe_period``.  To print at
   a pitch ``p``, relax the stripe field at ``stripe_period = 2p``.

   That is also the cheaper way round.  The stripe field has to be resolved at
   ~6 elements per *stripe* period, so asking for a ``2p`` period leaves 3
   elements between neighbouring paths and needs a quarter of the nodes that
   generating at period ``p`` and discarding every other contour would.

No G-code is emitted here.  :func:`to_fullcontrol` returns a FullControl
*design* — a list of ``fc.Point`` / ``fc.Extruder`` / ... objects — and what
happens to it is the caller's choice:

.. code-block:: python

    import fullcontrol as fc
    from path_optimizer import paths

    polylines = paths.extract_paths(stripe_field)
    joined = paths.connect(polylines, tolerance=1.1e-3)
    steps = paths.to_fullcontrol(paths.order_paths(joined),
                                 z=0.2, width=0.4, height=0.2)

    fc.transform(steps, "plot")                                   # look at it
    fc.transform(steps, "gcode", fc.GcodeControls(printer_name="prusa_i3"))

Going through FullControl rather than writing G-code directly means printer
profiles, extrusion models, travel handling and previewing are all somebody
else's problem, and the same design can be retargeted without touching this
code.
"""
from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import dataclass
from xml.etree import ElementTree

import numpy as onp

from path_optimizer.stripes import StripeField, resample

__all__ = [
    "Path",
    "extract_paths",
    "extract_region_contours",
    "clip_to_region",
    "KIND_COLOURS",
    "write_svg",
    "read_svg",
    "write_dxf",
    "write_step",
    "KIND_ACI",
    "by_kind",
    "tag",
    "connect",
    "smooth",
    "simplify",
    "trim_hairpins",
    "spread_ends",
    "turn_radius",
    "order_paths",
    "travel_distance",
    "tool_change",
    "path_steps",
    "to_fullcontrol",
    "to_fullcontrol_layers",
    "path_lengths",
]


# ── The path type ────────────────────────────────────────────────────────────

@dataclass
class Path:
    """One print path, with the little semantics the later stages need.

    Bare coordinate arrays are enough to draw a picture but not to *sequence* a
    print: sequencing needs to know where each path starts and ends, whether it
    is a closed loop (which can be entered anywhere, and has no ends to match),
    and which group it belongs to (groups are laid down by different means and
    must not be interleaved).  That is what this carries.

    ``kind`` is a free-form label, not an enum: it is whatever distinction the
    job needs to keep — one kind per material, per tool, per process, or none at
    all.  Nothing in this module interprets the string.  It becomes an SVG group
    id, a :func:`connect` and :func:`order_paths` group, a g-code comment, and a
    key into :func:`to_fullcontrol`'s ``on_kind``, all by identity alone.  It
    only has to be usable as an XML id.  The default, ``"path"``, means "not
    distinguished from anything".
    """

    nodes: onp.ndarray
    kind: str = "path"
    path_id: int | None = None

    def __post_init__(self):
        self.nodes = onp.asarray(self.nodes, dtype=float)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError(
                f"nodes must be (n, 2), got {self.nodes.shape}")
        if len(self.nodes) < 2:
            raise ValueError("a path needs at least 2 nodes")

    @property
    def start(self) -> onp.ndarray:
        return self.nodes[0]

    @property
    def end(self) -> onp.ndarray:
        return self.nodes[-1]

    @property
    def length(self) -> float:
        return float(onp.linalg.norm(onp.diff(self.nodes, axis=0), axis=1).sum())

    @property
    def is_closed(self) -> bool:
        """Ends meet to within a thousandth of the path's own length.

        Relative, not absolute: these come from contouring at whatever scale the
        mesh is in, so a fixed tolerance would be wrong in one unit system or
        the other.
        """
        return bool(onp.linalg.norm(self.end - self.start)
                    <= 1e-3 * max(self.length, 1e-30))

    def reversed(self) -> Path:
        """The same path walked the other way — the other option when ordering."""
        return Path(self.nodes[::-1].copy(), self.kind, self.path_id)

    def rolled(self, index: int) -> Path:
        """A closed path re-cut to begin at ``index``.

        Only meaningful for a loop: an open path re-cut this way would leave a
        gap.  Ordering uses it to start a loop at whichever vertex is nearest.
        """
        if not self.is_closed:
            raise ValueError("only a closed path can be re-cut")
        n = self.nodes[:-1]                       # drop the duplicated end
        k = int(index) % len(n)
        rolled = onp.vstack([n[k:], n[:k], n[k:k + 1]])
        return Path(rolled, self.kind, self.path_id)


def _as_paths(items, kind: str = "path") -> list[Path]:
    """Accept either Path objects or bare ``(n, 2)`` arrays."""
    out = []
    for i, item in enumerate(items):
        if isinstance(item, Path):
            out.append(item)
        else:
            out.append(Path(item, kind, i))
    return out


def by_kind(paths) -> dict[str, list[Path]]:
    """Split paths by ``kind``, keeping first-seen order.

    The basic thing a kind is for: pull one group out to draw it, measure it,
    resample it, or hand it to a different stage.  Everything else in this
    module that is kind-aware — the SVG groups, the sequencer, the per-kind
    steps in :func:`to_fullcontrol` — is this operation with something done to
    each group afterwards.
    """
    groups: dict[str, list[Path]] = {}
    for path in _as_paths(paths):
        groups.setdefault(path.kind, []).append(path)
    return groups


def tag(paths, kind: str) -> list[Path]:
    """Relabel paths, leaving the originals alone.

    How a multi-material set gets assembled: extract each material's paths
    however it is generated, tag each batch, and concatenate.
    """
    return [Path(p.nodes, kind, p.path_id) for p in _as_paths(paths)]


# ── Connecting ───────────────────────────────────────────────────────────────

def _crosses(bridge, segments) -> bool:
    """Does the straight bridge cut across any of these segments?

    Proper crossings only: two segments that merely share an endpoint, which
    every weld does with the paths it joins, are not a crossing.
    """
    if not len(segments):
        return False
    p, r = bridge[0], bridge[1] - bridge[0]
    q, s_ = segments[:, 0], segments[:, 1] - segments[:, 0]
    denom = r[0] * s_[:, 1] - r[1] * s_[:, 0]
    qp = q - p
    with onp.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * s_[:, 1] - qp[:, 1] * s_[:, 0]) / denom
        u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / denom
    eps = 1e-9
    hit = (onp.abs(denom) > 1e-30) & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)
    return bool(hit.any())


def _segments_of(paths, skip_ids):
    """Every segment of every path except the two ends of the weld."""
    out = []
    for k, path in enumerate(paths):
        if k in skip_ids:
            continue
        q = onp.asarray(path.nodes)
        if len(q) >= 2:
            out.append(onp.stack([q[:-1], q[1:]], axis=1))
    return onp.concatenate(out) if out else onp.zeros((0, 2, 2))


def connect(paths, *, tolerance: float, group_by_kind: bool = True,
            close_loops: bool = False, no_cross: bool = False) -> list[Path]:
    """Weld consecutive paths whose ends nearly meet, **in the order given**.

    Contouring cuts a field wherever the level set happens to break, so what the
    printer would run as one continuous zigzag arrives as a pile of separate
    strokes.  This welds them back: walking the sequence, whenever the travel
    from one path's end to the next path's start is within ``tolerance``, the two
    become one path.

    .. important::

       Run :func:`order_paths` **first**.  This consumes the order it is given —
       it welds neighbours in the sequence, not whichever ends happen to be
       closest in space.  Sequencing is what puts the right paths next to each
       other and turns each one the right way round, so the gap this measures is
       exactly the travel that sequencing could not avoid.

       Matching on absolute distance instead does not work here.  Stripe ends
       lie along the part boundary, so consecutive stripes end ``pitch /
       sin(angle)`` apart, not ``pitch`` — on a shallow boundary far more.  What
       identifies a weldable pair is that the sequencer put them together, not
       that they are within a pitch of each other.

    The gap is spanned by a straight segment: nothing is invented in between,
    and the welded path passes through both original endpoints.  So ``tolerance``
    is also the longest bridge the result may contain.

    Closed paths have no free ends; they break the chain and pass through
    untouched.

    Parameters
    ----------
    paths : sequence of :class:`Path` or of ``(n, 2)`` arrays
        In print order — the output of :func:`order_paths`.
    tolerance : float
        Longest travel to weld, in the paths' own units.  Zero or negative welds
        nothing.
    group_by_kind : bool
        Never weld across a change of ``kind``.  A welded path is one continuous
        run, and two kinds usually mean two materials or two tools.
    close_loops : bool
        Also close a finished chain whose two ends are within ``tolerance``,
        making it a closed path.  Off by default — a loop is a different thing
        to sequence and to print, so becoming one is worth asking for.
    no_cross : bool
        Refuse a weld whose bridge would cut across another path.  A bridge is
        a straight line drawn through whatever lies between the two ends, and
        at a few path pitches it starts running over the part instead of along
        it.  Costs one segment-intersection pass per candidate weld.
    Returns
    -------
    list of Path
        Welded paths, in the same order, each keeping the ``kind`` and
        ``path_id`` of its first piece.

    Notes
    -----
    Run :func:`order_paths` again afterwards: welding changes which paths exist,
    and a chain has different ends to sequence from than its pieces did.
    """
    paths = _as_paths(paths)
    if not paths:
        return []

    runs: list[list[Path]] = []
    last_index: list[int] = []          # which input path each run ends on
    for index, path in enumerate(paths):
        weldable = (
            runs
            and tolerance > 0
            and not path.is_closed
            and not runs[-1][-1].is_closed
            and (not group_by_kind or path.kind == runs[-1][-1].kind)
            and onp.linalg.norm(path.start - runs[-1][-1].end) <= tolerance
        )
        if weldable and no_cross:
            prev = runs[-1][-1]
            bridge = onp.vstack([prev.end, path.start])
            others = _segments_of(paths, {index, last_index[-1]})
            weldable = not _crosses(bridge, others)
        if weldable:
            runs[-1].append(path)
            last_index[-1] = index
        else:
            runs.append([path])
            last_index.append(index)
    return [_join(r, tolerance, close_loops) for r in runs]


def _join(pieces: list[Path], tolerance: float, close_loop: bool) -> Path:
    """Concatenate a run of already-oriented paths into one."""
    if len(pieces) == 1 and not close_loop:
        return pieces[0]
    # A gap far below the tolerance is two contours meeting at a shared point;
    # keeping both copies would leave a zero-length segment in the g-code.
    eps = 1e-6 * tolerance
    nodes = [pieces[0].nodes]
    for prev, nxt in zip(pieces, pieces[1:]):    # noqa: B905
        gap = onp.linalg.norm(nxt.start - prev.end)
        nodes.append(nxt.nodes[1:] if gap <= eps else nxt.nodes)
    joined = onp.vstack(nodes)
    if close_loop:
        gap = onp.linalg.norm(joined[-1] - joined[0])
        # A chain can already end where it began; appending would only add a
        # zero-length segment.
        if eps < gap <= tolerance:
            joined = onp.vstack([joined, joined[:1]])
    return Path(joined, pieces[0].kind, pieces[0].path_id)


# ── Ordering ─────────────────────────────────────────────────────────────────

def travel_distance(paths) -> float:
    """Total distance moved between paths, in the order given.

    The quantity ordering exists to reduce: every one of these is an
    extruder-off move, and on a machine that cuts its feedstock also a cut and
    a re-anchor.
    """
    paths = _as_paths(paths)
    if len(paths) < 2:
        return 0.0
    return float(sum(onp.linalg.norm(b.start - a.end)
                     for a, b in zip(paths, paths[1:])))    # noqa: B905


def order_paths(paths, *, start=None, group_by_kind: bool = True,
                method: str = "tsp", optim_steps: int = 3) -> list[Path]:
    """Sequence paths to cut the travel between them.

    Picks both the order and the **direction** of every path — an open path can
    be run either way, so a sequencer that only permutes leaves half the saving
    on the table.  Closed loops are re-cut to begin at their nearest vertex,
    since a loop can be entered anywhere.

    Parameters
    ----------
    start : (2,) array-like, optional
        Where the head begins.  Defaults to the first path's start, so the
        result is deterministic.
    group_by_kind : bool
        Keep paths of one ``kind`` together, sequencing within each group and
        emitting groups in first-seen order.  Different kinds usually mean
        different material or a different tool, so interleaving them costs a
        change per path rather than one per kind.
    method : {"tsp", "greedy"}
        ``"tsp"`` states the problem properly — two nodes per path (its ends)
        joined by a zero-cost edge, so choosing the tour chooses the direction
        too — and hands it to ``tsp-solver2``: greedy edge matching followed by
        ``optim_steps`` rounds of 2-opt.  Falls back to ``"greedy"`` with a
        warning if that package is missing.

        ``"greedy"`` is nearest-free-end, taking whichever unused path has an
        end closest to the head.  Fast and locally sensible, but it strands
        paths: having consumed a neighbourhood it must cross the part to reach
        what it skipped.  On a 65-path stripe set that cost 738 mm of travel
        against the TSP's 570, with a worst single jump of 159 mm against 86.

        Neither dominates — both are heuristics.  On ten stacked lines started
        from a corner, greedy finds the 9 mm optimum and the TSP returns 13.
        Measure with :func:`travel_distance` rather than assuming.

    optim_steps : int
        2-opt rounds for ``method="tsp"``.  More is better and costs time.

    Returns
    -------
    list of Path
        The same paths, reordered and possibly reversed or re-cut.

    Notes
    -----
    Neither method is exact — this is a travelling-salesman problem — so measure
    the result with :func:`travel_distance` rather than assuming it helped.
    """
    paths = _as_paths(paths)
    if not paths:
        return []

    if group_by_kind:
        order, groups = [], by_kind(paths)
        for kind in groups:                        # dicts keep insertion order
            order += order_paths(groups[kind], start=start, group_by_kind=False,
                                 method=method, optim_steps=optim_steps)
            start = order[-1].end
        return order

    if method == "tsp":
        try:
            from tsp_solver.greedy import solve_tsp
        except ImportError:
            warnings.warn(
                "tsp-solver2 is not installed, falling back to greedy ordering; "
                "install it with `pip install tsp-solver2`", stacklevel=2)
        else:
            return _order_tsp(paths, start, solve_tsp, optim_steps)
    elif method != "greedy":
        raise ValueError(f"method must be 'tsp' or 'greedy', got {method!r}")
    return _order_greedy(paths, start)


def _order_tsp(paths, start, solve_tsp, optim_steps: int) -> list[Path]:
    """Two nodes per path — its two ends — with a zero-cost edge between them.

    The formulation feax4d uses.  Because the intra-path edges cost nothing they
    are the first the solver takes, so a tour enters each path at one end and
    leaves at the other: choosing the tour chooses each path's direction as well
    as its place in the order.

    ``start``, if given, enters the matrix as a virtual node 0 and is dropped
    from the answer.

    A closed path goes in as well, with both its nodes at its start, since it
    has no direction to choose; once the tour has placed it, it is re-cut to
    begin at whichever vertex the head is actually nearest.  feax4d instead
    pins closed paths at the front in input order.  That is what it costs: on
    the measured part a single 30.6 mm closed stripe, pinned ahead of eleven
    welded chains, added a 169.2 mm travel to reach it and get back — 46% of the
    whole tour, for one loop.
    """
    if len(paths) == 1:
        return list(paths)

    ends = onp.empty((2 * len(paths), 2))
    ends[0::2] = [p.start for p in paths]
    ends[1::2] = [p.end for p in paths]
    d = onp.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=-1)
    i = onp.arange(len(paths))
    d[2 * i, 2 * i + 1] = d[2 * i + 1, 2 * i] = 0.0

    if start is not None:
        head = onp.asarray(start, dtype=float)
        aug = onp.zeros((len(d) + 1, len(d) + 1))
        aug[0, 1:] = aug[1:, 0] = onp.linalg.norm(ends - head, axis=1)
        aug[1:, 1:] = d
        tour = [n - 1 for n in solve_tsp(aug, optim_steps=optim_steps) if n > 0]
    else:
        head = None
        tour = solve_tsp(d, optim_steps=optim_steps)

    out, seen = [], set()
    for node in tour:
        idx, entered_at_end = node // 2, node % 2 == 1
        if idx in seen:
            continue
        seen.add(idx)
        path = paths[idx]
        if path.is_closed:
            if head is not None:
                k = int(onp.argmin(onp.linalg.norm(path.nodes[:-1] - head,
                                                   axis=1)))
                path = path.rolled(k)
        elif entered_at_end:
            path = path.reversed()
        out.append(path)
        head = path.end
    return out


def _order_greedy(paths, start) -> list[Path]:
    """Nearest free end, taking each path forwards or backwards as it faces."""
    remaining = list(paths)
    head = onp.asarray(start, dtype=float) if start is not None else remaining[0].start
    out: list[Path] = []
    while remaining:
        best_i, best_d, best_p = 0, onp.inf, remaining[0]
        for i, p in enumerate(remaining):
            if p.is_closed:
                d2 = onp.linalg.norm(p.nodes[:-1] - head, axis=1)
                k = int(onp.argmin(d2))
                d, cand = float(d2[k]), p.rolled(k)
            else:
                d_start = float(onp.linalg.norm(p.start - head))
                d_end = float(onp.linalg.norm(p.end - head))
                d, cand = ((d_start, p) if d_start <= d_end
                           else (d_end, p.reversed()))
            if d < best_d:
                best_i, best_d, best_p = i, d, cand
        remaining.pop(best_i)
        out.append(best_p)
        head = best_p.end
    return out


def _structured(mesh):
    """``(nx, ny, xs, ys)`` if ``mesh`` is a regular grid, else ``None``.

    Verified, not assumed: the reshape is checked against the actual node
    coordinates before it is trusted, so an unstructured mesh falls through to
    interpolation rather than being silently scrambled.
    """
    pts = onp.asarray(mesh.points)[:, :2]
    xs = onp.unique(pts[:, 0])
    ys = onp.unique(pts[:, 1])
    if xs.size * ys.size != pts.shape[0]:
        return None
    try:
        gx = pts[:, 0].reshape(xs.size, ys.size)
        gy = pts[:, 1].reshape(xs.size, ys.size)
    except ValueError:
        return None
    # feax's rectangle_mesh is x-major / y-fastest.
    if (onp.allclose(gx[:, 0], xs) and onp.allclose(gy[0, :], ys)
            and onp.allclose(gx, xs[:, None]) and onp.allclose(gy, ys[None, :])):
        return xs.size, ys.size, xs, ys
    return None


def _rasterise(mesh, columns, step):
    """Sample nodal fields onto a regular ``[y, x]`` grid of spacing ~``step``.

    Contour finding needs an array, and the stripe field lives on a mesh.  When
    the mesh already *is* a regular grid (the usual case — the stripe stage runs
    on a refined rectangle) its own nodes are used directly, which skips a
    Delaunay triangulation of several hundred thousand points.
    """
    stacked = onp.column_stack([onp.asarray(c, dtype=float) for c in columns])
    grid = _structured(mesh)
    if grid is not None:
        nx, ny, xs, ys = grid
        out = [stacked[:, k].reshape(nx, ny).T for k in range(stacked.shape[1])]
        return out, xs, ys

    pts = onp.asarray(mesh.points)[:, :2]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    nx = max(4, int(round((hi[0] - lo[0]) / step)) + 1)
    ny = max(4, int(round((hi[1] - lo[1]) / step)) + 1)
    xs = onp.linspace(lo[0], hi[0], nx)
    ys = onp.linspace(lo[1], hi[1], ny)
    gx, gy = onp.meshgrid(xs, ys)
    vals = resample(stacked, mesh, onp.column_stack([gx.ravel(), gy.ravel()]))
    vals = onp.atleast_2d(vals.T).T if vals.ndim == 1 else vals
    out = [vals[:, k].reshape(ny, nx) for k in range(vals.shape[1])]
    return out, xs, ys


def _to_physical(rows, cols, xs, ys):
    """Contour indices (row, col) → physical (x, y)."""
    x = onp.interp(cols, onp.arange(xs.size), xs)
    y = onp.interp(rows, onp.arange(ys.size), ys)
    return onp.column_stack([x, y])


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_paths(field: StripeField, *, level: float = 0.0,
                  min_length: float | None = None, boundary_erode: int = 1,
                  samples_per_period: int = 8) -> list[onp.ndarray]:
    """Zero-level polylines of the stripe field, clipped to the solid region.

    Parameters
    ----------
    field : StripeField
        From :mod:`path_optimizer.stripes`.
    level : float
        Contour level.  ``0.0`` gives paths at the requested pitch; a non-zero
        level shifts them off centre and makes alternate gaps uneven.
    extra : int
        Drop this many further vertices from each side once ``clearance`` is
        met.  The clearance test is satisfied the moment the two ends are far
        enough apart, which on a gentle turn is barely past the apex and leaves
        the two beads still running alongside each other.  Two or three extra
        vertices open it properly.  ``max_trim`` still bounds the total.
    min_length : float, optional
        Drop polylines shorter than this.  Defaults to two stripe periods —
        anything shorter is a fragment at a defect or a mask corner, and costs
        more in travel and stops than it contributes.
    boundary_erode : int
        Grid cells to erode the mask by before keeping contour vertices.
    samples_per_period : int
        Raster resolution, used only when the mesh is not already a regular grid.

    Returns
    -------
    list of ``(n, 2)`` arrays
        Polylines in the mesh's physical units.

    Notes
    -----
    Masking is why this is more than a call to ``find_contours``.  Filling the
    void with a large sentinel keeps contours out of it, but then a zero contour
    runs *along* the solid/void boundary and threads separate stripes into one
    boundary-hugging curve.  So each contour is kept only where it lies in the
    **eroded interior** of the mask and is split wherever it leaves — stripes end
    at the boundary instead of joining along it.
    """
    from scipy.ndimage import binary_erosion
    from skimage import measure

    step = field.stripe_period / max(1, samples_per_period)
    (u_grid, mask_grid), xs, ys = _rasterise(
        field.mesh, (field.u, field.mask), step)

    solid = mask_grid > 0.5
    if not solid.any():
        return []
    clipped = onp.asarray(u_grid).copy()
    clipped[~solid] = 1e10                      # no isoline in the void
    interior = (binary_erosion(solid, iterations=int(boundary_erode))
                if boundary_erode > 0 else solid)

    if min_length is None:
        min_length = 2.0 * field.stripe_period

    paths: list[onp.ndarray] = []

    def emit(chunk):
        if chunk.shape[0] < 2:
            return
        seg = _to_physical(chunk[:, 0], chunk[:, 1], xs, ys)
        if onp.linalg.norm(onp.diff(seg, axis=0), axis=1).sum() < min_length:
            return
        paths.append(seg)

    for contour in measure.find_contours(clipped, level=level):
        if contour.shape[0] < 2:
            continue
        ri = onp.clip(onp.round(contour[:, 0]).astype(int), 0, ys.size - 1)
        ci = onp.clip(onp.round(contour[:, 1]).astype(int), 0, xs.size - 1)
        inside = interior[ri, ci]
        start = None
        for k, ins in enumerate(inside):
            if ins and start is None:
                start = k
            elif not ins and start is not None:
                emit(contour[start:k])
                start = None
        if start is not None:
            emit(contour[start:])
    return paths


def extract_region_contours(field: StripeField, *, invert: bool = False,
                            offset: float = 0.0, min_points: int = 4,
                            samples_per_period: int = 8) -> list[onp.ndarray]:
    """Closed boundary polygons of the solid region (or its complement).

    ``invert=True`` gives the void region instead — the area the stripes do not
    cover, which is what a filling process needs bounded.  The grid is
    zero-padded first so a region touching the domain edge still closes, and
    interior holes come back as their own loops.

    Parameters
    ----------
    offset : float
        Move the loops this far **into** the region, in the mesh's units.  A
        bead printed on the boundary itself hangs half its width over the edge
        and lands on whatever the fill put there; half a bead width in, it sits
        inside the part and beside the fill rather than on it.

        The offset is a level set of the distance to the boundary, not a
        displaced polygon, so it cannot self-intersect: where the part is
        narrower than twice ``offset`` the loop closes up and disappears, which
        is the correct answer — there is no room for a perimeter there.

    Returns
    -------
    list of ``(n, 2)`` arrays
        Closed loops.  Fewer than at ``offset = 0`` when thin features drop
        out, and a loop may split in two where a waist pinches off.
    """
    from skimage import measure

    step = field.stripe_period / max(1, samples_per_period)
    (mask_grid,), xs, ys = _rasterise(field.mesh, (field.mask,), step)
    region = (mask_grid <= 0.5) if invert else (mask_grid > 0.5)
    padded = onp.pad(region.astype(float), 1, constant_values=0.0)

    level = 0.5
    if offset:
        from scipy.ndimage import distance_transform_edt

        if offset < 0:
            raise ValueError(f"offset moves into the region, so it must not be "
                             f"negative; got {offset}")
        # `sampling` makes the transform physical, so the level is the offset
        # itself.  The half cell is where the binary boundary actually lies:
        # the first foreground cell centre is half a cell inside it.
        dy = float(ys[1] - ys[0]) if ys.size > 1 else step
        dx = float(xs[1] - xs[0]) if xs.size > 1 else step
        padded = distance_transform_edt(padded > 0.5, sampling=(dy, dx))
        level = offset + 0.5 * min(dx, dy)

    polys = []
    for c in measure.find_contours(padded, level=level):
        if c.shape[0] < min_points:
            continue
        rows = onp.clip(c[:, 0] - 1.0, 0.0, ys.size - 1)
        cols = onp.clip(c[:, 1] - 1.0, 0.0, xs.size - 1)
        poly = _to_physical(rows, cols, xs, ys)
        if not onp.allclose(poly[0], poly[-1]):
            poly = onp.vstack([poly, poly[0]])
        polys.append(poly)
    return polys


def _inset_field(field, samples_per_period: int):
    """Distance from the region's boundary, as a grid plus its axes."""
    from scipy.ndimage import distance_transform_edt

    step = field.stripe_period / max(1, samples_per_period)
    (mask_grid,), xs, ys = _rasterise(field.mesh, (field.mask,), step)
    dy = float(ys[1] - ys[0]) if ys.size > 1 else step
    dx = float(xs[1] - xs[0]) if xs.size > 1 else step
    inside = onp.pad(mask_grid, 1, constant_values=0.0) > 0.5
    depth = distance_transform_edt(inside, sampling=(dy, dx))[1:-1, 1:-1]
    return depth - 0.5 * min(dx, dy), xs, ys


def clip_to_region(paths, field: StripeField, *, inset: float,
                   samples_per_period: int = 8, min_length: float | None = None):
    """Keep only the parts of each path that lie ``inset`` inside the boundary.

    The companion to :func:`extract_region_contours`'s ``offset``, and the half
    that actually stops the two fouling each other.  Moving a perimeter inwards
    does not get it clear of the fill — the fill is everywhere, so the
    perimeter just lands further onto it.  What a slicer does, and what this
    does, is cut the fill back and leave the perimeter the room.

    For a perimeter whose centre line sits ``w/2`` in, its bead reaches ``w``
    from the edge, so the fill's centre line has to start at ``1.5 * w``: that
    is the ``inset`` to ask for.

    Where a path leaves the region it is cut, and the crossing point is
    interpolated rather than snapped to the last vertex inside, so the end
    lands on the inset line and not up to one sample short of it.

    Parameters
    ----------
    paths : sequence of :class:`Path` or of ``(n, 2)`` arrays
    field : StripeField
        Supplies the mask the distance is measured from.
    inset : float
        How far inside the boundary a point must be to survive, in the mesh's
        units.
    min_length : float, optional
        Drop surviving pieces shorter than this.  Defaults to ``2 * inset``.

    Returns
    -------
    list of :class:`Path`
        More paths than went in, each keeping its parent's kind.  Re-run
        :func:`order_paths`: the sequence has changed.
    """
    if inset < 0:
        raise ValueError(f"inset must not be negative, got {inset}")
    depth, xs, ys = _inset_field(field, samples_per_period)
    min_length = 2.0 * inset if min_length is None else min_length

    def depth_at(q):
        """Bilinear sample of the distance field at arbitrary points."""
        fx = onp.clip((q[:, 0] - xs[0]) / (xs[1] - xs[0]), 0, xs.size - 1.000001)
        fy = onp.clip((q[:, 1] - ys[0]) / (ys[1] - ys[0]), 0, ys.size - 1.000001)
        i0, j0 = fy.astype(int), fx.astype(int)
        ty, tx = fy - i0, fx - j0
        d00, d01 = depth[i0, j0], depth[i0, j0 + 1]
        d10, d11 = depth[i0 + 1, j0], depth[i0 + 1, j0 + 1]
        return ((1 - ty) * ((1 - tx) * d00 + tx * d01)
                + ty * ((1 - tx) * d10 + tx * d11))

    out: list[Path] = []
    for path in _as_paths(paths):
        q = onp.asarray(path.nodes, dtype=float)
        d = depth_at(q) - inset
        keep = d >= 0.0
        if keep.all():
            out.append(path)
            continue
        for lo, hi in _runs(keep):
            piece = [q[lo:hi]]
            # Walk out to the crossing on each side, where there is one.
            if lo > 0:
                t = d[lo - 1] / (d[lo - 1] - d[lo])
                piece.insert(0, (q[lo - 1] + t * (q[lo] - q[lo - 1]))[None])
            if hi < len(q):
                t = d[hi - 1] / (d[hi - 1] - d[hi])
                piece.append((q[hi - 1] + t * (q[hi] - q[hi - 1]))[None])
            nodes = onp.vstack(piece)
            candidate = Path(nodes, path.kind, path.path_id)
            if len(nodes) >= 2 and candidate.length >= min_length:
                out.append(candidate)
    return out


def path_lengths(paths) -> onp.ndarray:
    """Arc length of each polyline.

    Takes :class:`Path` objects or bare ``(n, 2)`` arrays, like everything else
    here -- after :func:`order_paths` the usual thing to have in hand is the
    former.
    """
    return onp.array([p.length for p in _as_paths(paths)])


# ── SVG ──────────────────────────────────────────────────────────────────────

# Assigned to kinds in first-seen order.  A validated categorical order, not a
# rainbow: the drawing is how a multi-material job gets checked by eye before it
# is printed, so the kinds have to stay apart for a colourblind reader too.
KIND_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948")

def _svg_id(kind: str) -> str:
    """A kind as an XML id, or an error saying so."""
    if not kind or any(c in kind for c in ' "<>&\t\n'):
        raise ValueError(
            f"kind {kind!r} cannot be an SVG group id: it must be non-empty and "
            "free of whitespace, quotes and angle brackets")
    return kind


def _kind_colours(kinds, stroke):
    """Resolve the stroke for each kind, in first-seen order."""
    if isinstance(stroke, dict):
        missing = [k for k in kinds if k not in stroke]
        if missing:
            raise ValueError(f"no stroke given for kind(s) {missing}")
        return {k: stroke[k] for k in kinds}
    if stroke is not None:
        return dict.fromkeys(kinds, stroke)
    if len(kinds) > len(KIND_COLOURS):
        # Cycling would put two kinds in the same colour with nothing to say
        # which is which.  Better to stop colouring and leave the group ids as
        # the identity, which they are anyway.
        warnings.warn(
            f"{len(kinds)} kinds but only {len(KIND_COLOURS)} distinct colours; "
            "the extra kinds are drawn black -- pass `stroke` as a dict to "
            "choose, or read the <g id> instead of the colour", stacklevel=3)
    return {k: (KIND_COLOURS[i] if i < len(KIND_COLOURS) else "#000000")
            for i, k in enumerate(kinds)}


def write_svg(paths, filename, *, scale: float = 1000.0,
              units: str = "mm", stroke_width: float = 0.3,
              stroke=None) -> None:
    """Write paths to SVG, for inspection or for a slicer that wants one.

    ``scale`` converts the mesh's units to the SVG's (default metres → mm).  The
    y axis is flipped so the drawing reads the same way up as the mesh, since
    SVG counts y downward.

    **Each ``kind`` becomes its own ``<g id="{kind}">``**, in first-seen order,
    and :func:`read_svg` reads the id straight back as the kind.  Nothing here
    knows what a kind means — a multi-material job names one group per material,
    a single-process job leaves them all at the default — but a kind has to be
    usable as an XML id: non-empty, no whitespace or markup characters.

    Parameters
    ----------
    paths : sequence of :class:`Path` or of ``(n, 2)`` arrays
        Bare arrays are taken as the default kind; use :func:`tag` to name
        them.
    stroke : str or dict, optional
        One colour for everything, or ``{kind: colour}``.  By default each kind
        takes the next entry of :data:`KIND_COLOURS`.
    """
    items = _as_paths(paths)
    if not items:
        warnings.warn("writing an SVG with no paths", stacklevel=2)
    allpts = onp.vstack([p.nodes for p in items]) if items else onp.zeros((1, 2))
    x_min, y_min = allpts.min(axis=0)
    x_max, y_max = allpts.max(axis=0)

    groups = by_kind(items)
    for kind in groups:
        _svg_id(kind)
    colours = _kind_colours(list(groups), stroke)

    def pts_attr(poly):
        x = poly[:, 0] * scale
        y = (y_max + y_min - poly[:, 1]) * scale       # flip to SVG's y-down
        # Significant digits, not decimals: %.4f would be 0.1 um in millimetres
        # but 0.1 mm if someone writes metres, quietly making the file's
        # resolution a function of `scale`.
        return " ".join(f"{a:.10g},{b:.10g}" for a, b in zip(x, y, strict=True))

    w, h = (x_max - x_min) * scale, (y_max - y_min) * scale
    # data-scale makes the file self-describing: read_svg needs it to undo the
    # unit conversion and the y flip without being told out of band.
    desc = ", ".join(f"{len(v)} {k}" for k, v in groups.items()) or "empty"
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
           # The viewBox is not decoration: read_svg reconstructs the y flip
           # from it, so it is written at the same precision as the points.
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.10g}{units}" '
           f'height="{h:.10g}{units}" viewBox="{x_min * scale:.10g} '
           f'{y_min * scale:.10g} {w:.10g} {h:.10g}" data-scale="{scale!r}">',
           f'  <desc>print paths by kind: {desc}</desc>']
    for kind, members in groups.items():
        out.append(f'  <g id="{kind}" fill="none" stroke="{colours[kind]}" '
                   f'stroke-width="{stroke_width}">')
        out += [f'    <polyline points="{pts_attr(m.nodes)}"/>' for m in members]
        out.append("  </g>")
    out.append("</svg>")
    with open(filename, "w") as f:
        f.write("\n".join(out))


def read_svg(filename) -> list[Path]:
    """Read an SVG written by :func:`write_svg` back into :class:`Path` objects.

    The round trip exists so the drawing can be inspected — or hand-edited — and
    still be the thing that gets printed.  Coordinates are returned in the mesh's
    units: the ``data-scale`` attribute and the ``viewBox`` carry everything
    needed to undo the unit conversion and the y flip, so no arguments are
    required.

    **``kind`` is the enclosing group's id**, whatever it says — so the kinds
    :func:`write_svg` wrote come back unchanged, and a group a vector editor
    added arrives under its own name rather than being folded into its
    neighbours.  Polylines outside any group take the default kind, which is what
    a foreign file with no grouping most likely means.

    Only ``<polyline>`` is understood, which is all :func:`write_svg` emits;
    ``<path>`` elements with curve commands are skipped rather than
    mis-flattened, and a warning says how many were dropped.
    """
    tree = ElementTree.parse(str(filename))
    root = tree.getroot()

    scale = float(root.get("data-scale", 1000.0))
    view = root.get("viewBox")
    if view is not None:
        _, vy, _, vh = (float(t) for t in view.replace(",", " ").split())
        y_sum = (2.0 * vy + vh) / scale          # == y_max + y_min in mesh units
    else:                                        # foreign file: leave y alone
        y_sum = None

    def parse(points: str) -> onp.ndarray:
        nums = [float(t) for t in points.replace(",", " ").split()]
        xy = onp.asarray(nums, dtype=float).reshape(-1, 2) / scale
        if y_sum is not None:
            xy[:, 1] = y_sum - xy[:, 1]
        return xy

    ns = "{http://www.w3.org/2000/svg}"
    out, skipped = [], 0
    for group in [root] + root.findall(f"{ns}g"):
        gid = group.get("id")
        kind = gid if gid else "path"
        for el in group.findall(f"{ns}polyline"):
            xy = parse(el.get("points", ""))
            if len(xy) >= 2:
                out.append(Path(xy, kind, len(out)))
        skipped += len(group.findall(f"{ns}path"))
    if skipped:
        warnings.warn(
            f"skipped {skipped} <path> element(s): only <polyline> is read",
            stacklevel=2)
    return out


#: AutoCAD colour indices standing in for :data:`KIND_COLOURS`, in the same
#: order.  R12 has no true colour -- a layer carries one integer out of a fixed
#: 255-entry palette -- so these are the nearest hue, not the same colour.
KIND_ACI = (5, 30, 3, 2, 6, 94, 170, 1)

#: Characters AutoCAD rejects in a layer name, plus whitespace.  Layer names are
#: also capped at 31 characters in R12.
_DXF_BAD = set('<>/\\":;?*|=\'' + " \t\n")


def _dxf_layer(kind: str) -> str:
    """A kind as a DXF layer name, or an error saying why it is not one."""
    if not kind or _DXF_BAD & set(kind) or len(kind) > 31:
        raise ValueError(
            f"kind {kind!r} cannot be a DXF layer name: it must be non-empty, "
            f"at most 31 characters, and free of whitespace and <>/\\\":;?*|='"
        )
    return kind


def _dxf_pairs(*pairs):
    """DXF is a flat stream of (group code, value) lines, two lines each."""
    return "".join(f"{code}\n{value}\n" for code, value in pairs)


def write_dxf(paths, filename, *, scale: float = 1000.0,
              layer_colours=None) -> None:
    """Write paths to DXF R12, for CAD or for a CAM package that wants one.

    Each path becomes one ``POLYLINE``, so a run that has been through
    :func:`connect` exports as it prints: the bridges that welded two contours
    together are interior vertices of the polyline, indistinguishable from the
    contour they joined.  Travel moves are not geometry and are not written --
    what is in the file is what gets deposited.

    **Each ``kind`` becomes its own layer**, in first-seen order, which is the
    DXF counterpart of :func:`write_svg`'s ``<g id>``.  Freezing a layer in CAD
    then isolates one material or one process.

    ``scale`` converts the mesh's units to the file's, default metres to
    millimetres.  Unlike SVG there is **no y flip**: DXF counts y upward, the
    same way the mesh does, so the coordinates go out unchanged apart from the
    scale factor.  R12 stores no unit, so a reader will assume whatever its
    own drawing is set to -- millimetres, by the default above.

    Closed paths are written with the polyline's closed flag set rather than by
    repeating the first vertex, so CAM sees a loop and not a gap of zero length.

    Parameters
    ----------
    paths : sequence of :class:`Path` or of ``(n, 2)`` arrays
        Bare arrays are taken as the default kind; use :func:`tag` to name them.
    layer_colours : int or dict, optional
        One AutoCAD colour index for every layer, or ``{kind: index}``.  By
        default each kind takes the next entry of :data:`KIND_ACI`.

    See Also
    --------
    write_svg : the same drawing with colour, for a browser or a slicer.
    """
    items = _as_paths(paths)
    if not items:
        warnings.warn("writing a DXF with no paths", stacklevel=2)

    groups = by_kind(items)
    layers = [_dxf_layer(k) for k in groups]
    if isinstance(layer_colours, dict):
        missing = set(groups) - set(layer_colours)
        if missing:
            raise ValueError(f"layer_colours has no entry for {sorted(missing)}")
        aci = {k: int(layer_colours[k]) for k in groups}
    elif layer_colours is None:
        aci = {k: KIND_ACI[i % len(KIND_ACI)] for i, k in enumerate(groups)}
    else:
        aci = dict.fromkeys(groups, int(layer_colours))

    allpts = (onp.vstack([p.nodes for p in items]) * scale if items
              else onp.zeros((1, 2)))
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)

    out = [_dxf_pairs(
        (0, "SECTION"), (2, "HEADER"),
        (9, "$ACADVER"), (1, "AC1009"),
        (9, "$EXTMIN"), (10, f"{lo[0]:.10g}"), (20, f"{lo[1]:.10g}"), (30, "0.0"),
        (9, "$EXTMAX"), (10, f"{hi[0]:.10g}"), (20, f"{hi[1]:.10g}"), (30, "0.0"),
        (0, "ENDSEC"),
        (0, "SECTION"), (2, "TABLES"),
        (0, "TABLE"), (2, "LAYER"), (70, len(layers)))]
    for kind, name in zip(groups, layers, strict=True):
        out.append(_dxf_pairs((0, "LAYER"), (2, name), (70, 0),
                              (62, aci[kind]), (6, "CONTINUOUS")))
    out.append(_dxf_pairs((0, "ENDTAB"), (0, "ENDSEC"),
                          (0, "SECTION"), (2, "ENTITIES")))

    for kind, name in zip(groups, layers, strict=True):
        for path in groups[kind]:
            nodes = onp.asarray(path.nodes, dtype=float) * scale
            closed = path.is_closed
            if closed:
                # The closed flag already draws the last segment; keeping the
                # repeated vertex as well would leave a zero-length one.
                nodes = nodes[:-1]
            # 66 = "vertices follow", which R12 requires on every POLYLINE.
            out.append(_dxf_pairs((0, "POLYLINE"), (8, name), (66, 1),
                                  (70, 1 if closed else 0)))
            for x, y in nodes:
                out.append(_dxf_pairs((0, "VERTEX"), (8, name),
                                      (10, f"{x:.10g}"), (20, f"{y:.10g}"),
                                      (30, "0.0")))
            out.append(_dxf_pairs((0, "SEQEND"), (8, name)))

    out.append(_dxf_pairs((0, "ENDSEC"), (0, "EOF")))
    with open(filename, "w") as f:
        f.write("".join(out))


def turn_radius(nodes) -> onp.ndarray:
    """Radius of the circle through each consecutive triple of vertices.

    One value per interior vertex, so ``len(nodes) - 2`` of them.  Collinear
    triples give ``inf``.  This is the measure of whether a bead of a given
    width can follow the path: below ``width / 2`` it cannot, and lays material
    over material instead.
    """
    q = onp.asarray(nodes, dtype=float)
    if len(q) < 3:
        return onp.full(max(len(q) - 2, 0), onp.inf)
    a, b, c = q[:-2], q[1:-1], q[2:]
    ab, bc, ac = b - a, c - b, c - a
    twice_area = onp.abs(ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0])
    sides = (onp.linalg.norm(ab, axis=1) * onp.linalg.norm(bc, axis=1)
             * onp.linalg.norm(ac, axis=1))
    return onp.where(twice_area > 1e-12,
                     sides / (2.0 * onp.maximum(twice_area, 1e-12)), onp.inf)


def _runs(flags) -> list[tuple[int, int]]:
    """Contiguous ``True`` spans of a boolean array as half-open ranges."""
    idx = onp.flatnonzero(flags)
    if not len(idx):
        return []
    breaks = onp.flatnonzero(onp.diff(idx) > 1)
    starts = onp.concatenate([[0], breaks + 1])
    ends = onp.concatenate([breaks + 1, [len(idx)]])
    return [(int(idx[a]), int(idx[b - 1]) + 1) for a, b in zip(starts, ends, strict=True)]


def trim_hairpins(paths, *, radius: float, clearance: float | None = None,
                  max_trim: float | None = None, min_length: float | None = None,
                  extra: int = 0):
    """Cut the path where it turns tighter than the bead, and pull the ends apart.

    A toolpath off a stripe field turns back on itself at the end of every
    stripe, and some of those turns are tighter than the bead is wide.  Printed,
    that is not a turn: the nozzle lays material on material and leaves a blob,
    and it is also what makes a CAD solid of the bead self-intersect.  Measured
    on a real part, 74% of the paths had at least one such turn.

    So: split at each one, then walk the two new ends back until they stand
    ``clearance`` apart, which is the gap the bead needs in order not to
    overlap.  Nothing is moved -- only dropped -- so every vertex that survives
    is still exactly where the optimiser put it.

    Parameters
    ----------
    radius : float
        Turns tighter than this are hairpins, in the mesh's units.  The bead's
        half-width is the physical choice.
    clearance : float, optional
        How far apart to pull the two cut ends.  Defaults to ``2 * radius`` --
        one bead width, so the two bead ends meet without overlapping.
    max_trim : float, optional
        Give up walking back after this much arc length on each side and cut
        anyway.  Defaults to ``5 * clearance``.  It bounds the damage when the
        two legs stay close for a long way, which is a path that was going to
        print badly whatever is done to it.
    extra : int
        Drop this many further vertices from each side once ``clearance`` is
        met.  The clearance test is satisfied the moment the two ends are far
        enough apart, which on a gentle turn is barely past the apex and leaves
        the two beads still running alongside each other.  Two or three extra
        vertices open it properly.  ``max_trim`` still bounds the total.
    min_length : float, optional
        Drop pieces shorter than this.  Defaults to ``clearance`` -- a stub
        shorter than the bead is wide is a dot of plastic and a travel move.

    Returns
    -------
    list of :class:`Path`
        In path order, each piece keeping its parent's kind.  Expect more paths
        out than in; run :func:`order_paths` afterwards, since the sequence has
        changed.

    Notes
    -----
    Run this **after** :func:`connect`, never before.  The gap it opens is one
    bead wide and ``connect`` is given a tolerance of several millimetres, so
    running it the other way round welds every cut straight back together.

    Measured on a real part -- 106 paths, 63161 mm, a 2 mm bead:

    .. code-block:: text

                           paths    length   travel   tight turns
        as printed           106   63161mm   3050mm            78
        hairpins trimmed     252   62154mm   3944mm             0

    1.6% of the path length and 29% more travel, for a bead that never crosses
    itself.  The path count is the real cost: 146 more stops and starts, each
    one a retraction and a restart.  Whether that trade is worth making is a
    print decision, which is why this is a separate step and not something
    :func:`extract_paths` does on its own.

    The cut lands on whichever vertex the sampling provided, which is often
    further from the neighbour than it needs to be and sometimes closer.
    :func:`spread_ends` places it properly afterwards.

    See Also
    --------
    turn_radius : the measure this thresholds, to see what a part is carrying.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    clearance = 2.0 * radius if clearance is None else clearance
    max_trim = 5.0 * clearance if max_trim is None else max_trim
    min_length = clearance if min_length is None else min_length

    out: list[Path] = []
    for path in _as_paths(paths):
        for piece in _split_hairpins(onp.asarray(path.nodes, dtype=float),
                                     radius, clearance, max_trim, path.is_closed,
                                     extra):
            candidate = Path(piece, path.kind, path.path_id)
            if candidate.length >= min_length:
                out.append(candidate)
    return out


def _split_hairpins(nodes, radius, clearance, max_trim, closed=False, extra=0):
    """Pieces of one polyline, cut and trimmed at every turn tighter than ``radius``."""
    if len(nodes) < 3:
        return [nodes]
    if closed and len(nodes) > 3:
        # `turn_radius` reads interior vertices only, so the turn *at* the seam
        # is the one vertex it never sees -- and a hairpin sitting there would
        # survive the whole operation.  Rolling the seam onto the gentlest turn
        # puts every sharp one where it can be found.  A closed path has no
        # first vertex in any real sense, so this costs nothing.
        gentle = int(onp.argmax(turn_radius(nodes))) + 1
        core = nodes[:-1]
        nodes = onp.vstack([onp.roll(core, -gentle, axis=0), core[gentle][None]])
    tight = turn_radius(nodes) < radius
    runs = _runs(tight)
    if not runs:
        return [nodes]

    # One cut per run of tight vertices, at its middle.  +1 converts a
    # turn_radius index (interior vertices) to a node index.
    cuts = [((lo + hi) // 2) + 1 for lo, hi in runs]

    step = onp.concatenate([[0.0], onp.cumsum(
        onp.linalg.norm(onp.diff(nodes, axis=0), axis=1))])
    pieces, start = [], 0
    for cut in cuts:
        left, right = _pull_apart(nodes, step, cut, clearance, max_trim, extra)
        if left > start:
            pieces.append(nodes[start:left + 1])
        start = right
    pieces.append(nodes[start:])
    pieces = [p for p in pieces if len(p) >= 2]

    if closed and len(pieces) >= 2:
        # The seam is not a cut.  A closed path repeats its first vertex at the
        # end, so the last piece runs straight into the first one through it --
        # emitted separately they are two paths whose ends sit on top of each
        # other, which is the opposite of what the trimming is for.
        pieces = [onp.vstack([pieces[-1][:-1], pieces[0]])] + pieces[1:-1]
    return pieces


def _pull_apart(nodes, step, cut, clearance, max_trim, extra=0):
    """Walk out from ``cut`` until the two ends stand ``clearance`` apart.

    Symmetric in arc length rather than in index, so an unevenly sampled turn
    is not trimmed harder on whichever side happens to carry more vertices.

    The walk stops one vertex short of each end of the polyline.  Running it to
    the very end would leave a one-vertex piece, which is not a path at all --
    whether the remaining stub is worth printing is ``min_length``'s decision
    to make, not a side effect of where the walk happened to stop.
    """
    lo, hi = 1, len(nodes) - 2
    left = right = cut
    while onp.linalg.norm(nodes[right] - nodes[left]) < clearance:
        moved_left = step[cut] - step[left]
        moved_right = step[right] - step[cut]
        if min(moved_left, moved_right) >= max_trim:
            break
        can_left, can_right = left > lo, right < hi
        if not (can_left or can_right):
            break
        if can_left and (moved_left <= moved_right or not can_right):
            left -= 1
        else:
            right += 1

    # `extra` is counted after the clearance is met, and still inside the same
    # bounds: never past the ends of the polyline, never past `max_trim`.
    for _ in range(max(extra, 0)):
        if left > lo and step[cut] - step[left - 1] <= max_trim:
            left -= 1
        if right < hi and step[right + 1] - step[cut] <= max_trim:
            right += 1
    return left, right


def spread_ends(paths, *, move: float, radius: float | None = None,
                skip: float | None = None, samples: int = 72):
    """Nudge each free end into the clearest space around it.

    :func:`trim_hairpins` can only stop at a vertex, so a cut end lands wherever
    the sampling happened to put one -- often still closer to its neighbour than
    it needs to be.  This moves it, by at most ``move``, to the point that
    maximises the summed distance to the material around it.

    The objective is a sum of distances, which is **convex** in the position, so
    its maximum over any disc is on the rim: the end always moves the full
    ``move``, in the direction that leads away from the surrounding nodes.
    ``move`` is therefore not a bound that is sometimes reached -- it is the
    step length, and it is the one number that decides how far the geometry may
    depart from the optimiser's own curve.

    Every point within ``radius`` counts the same, near or far, since each
    contributes one unit vector to the gradient.  That is what "sum of
    distances" means; if what you want is clearance, keep ``radius`` close to
    the pitch so distant stripes cannot outvote the neighbour that matters.

    Parameters
    ----------
    move : float
        How far an end may travel, in the mesh's units.  A fraction of the bead
        width is the sane range; past that the end leaves the path the
        optimiser designed.
    radius : float, optional
        Only nodes within this distance are "around it".  Defaults to
        ``10 * move``.
    skip : float, optional
        Ignore nodes of the end's **own** path within this arc length of it.
        Defaults to ``radius``.  Without it the end simply runs forward along
        its own tail, undoing the trim that put it there.
    samples : int
        Directions tried around the rim.  The objective is smooth, so 72 (every
        5 degrees) resolves the optimum to well under a tenth of ``move``.

    Returns
    -------
    list of :class:`Path`
        Same count, same order, same kinds.  Only the first and last vertex of
        each open path can differ; closed paths are returned untouched, since
        they have no free end to move and shifting one would break the loop.

    Notes
    -----
    This is the one operation here that puts a vertex where the optimiser did
    not. :func:`simplify` and :func:`trim_hairpins` only ever drop vertices.

    Measured on a real part after trimming -- 252 paths, a 2 mm bead, a 0.2 mm
    step -- against the distance from each free end to the nearest material
    that is not its own:

    .. code-block:: text

                        min gap     p5    ends closer than the bead
        trimmed          1.559mm  1.91mm                      7.1%
        + spread_ends    1.759mm  2.03mm                      1.6%

    7 ms for the lot.  No solver: the maximum of a convex function over a disc
    is on its rim, so the rim is all that is searched.
    """
    if move <= 0:
        raise ValueError(f"move must be positive, got {move}")
    radius = 10.0 * move if radius is None else radius
    skip = radius if skip is None else skip
    items = _as_paths(paths)
    if not items:
        return []

    from scipy.spatial import cKDTree

    nodes = [onp.asarray(p.nodes, dtype=float) for p in items]
    cloud = onp.vstack(nodes)
    owner = onp.concatenate([onp.full(len(q), i) for i, q in enumerate(nodes)])
    arc = onp.concatenate([
        onp.concatenate([[0.0], onp.cumsum(onp.linalg.norm(onp.diff(q, axis=0), axis=1))])
        for q in nodes])
    tree = cKDTree(cloud)

    angles = onp.linspace(0.0, 2.0 * onp.pi, samples, endpoint=False)
    rim = move * onp.stack([onp.cos(angles), onp.sin(angles)], axis=1)

    out = []
    for i, path in enumerate(items):
        q = nodes[i].copy()
        if len(q) >= 2 and not path.is_closed:
            total = arc[onp.flatnonzero(owner == i)][-1]
            for end, own_arc in ((0, 0.0), (len(q) - 1, total)):
                near = tree.query_ball_point(q[end], radius)
                if not near:
                    continue
                near = onp.asarray(near)
                mine = owner[near] == i
                keep = ~(mine & (onp.abs(arc[near] - own_arc) < skip))
                if not keep.any():
                    continue
                around = cloud[near[keep]]
                # Convex objective: the best point on the rim is the answer, so
                # only the rim is searched.
                dist = onp.linalg.norm(
                    (q[end] + rim)[:, None, :] - around[None, :, :], axis=-1)
                q[end] = q[end] + rim[int(onp.argmax(dist.sum(axis=1)))]
        out.append(Path(q, path.kind, path.path_id))
    return out


def _laplacian_step(q, closed, weight):
    """One pass of ``x += weight * (neighbour mean - x)``."""
    if closed:
        core = q[:-1]
        mean = 0.5 * (onp.roll(core, 1, axis=0) + onp.roll(core, -1, axis=0))
        core = core + weight * (mean - core)
        return onp.vstack([core, core[:1]])
    out = q.copy()
    mean = 0.5 * (q[:-2] + q[2:])
    out[1:-1] = q[1:-1] + weight * (mean - q[1:-1])
    return out


def smooth(paths, *, iterations: int = 10, lam: float = 0.5,
           mu: float | None = None):
    """Taubin smoothing: take the stair-steps out without shrinking the loop.

    A contour off a grid is a staircase — marching squares can only put a
    vertex on a cell edge, so a boundary crossing the grid at a shallow angle
    comes back as a flight of steps one cell high.  Printed, every step is a
    direction change the machine has to decelerate through.

    Plain Laplacian smoothing takes the steps out and the area with them: it is
    diffusion, and a closed loop under diffusion shrinks towards a point.  That
    matters here because the perimeter's offset is deliberate.  Taubin's fix is
    to follow each smoothing pass with a slightly larger negative one, which
    undoes the shrinkage while leaving the high-frequency removal in place.

    Parameters
    ----------
    iterations : int
        Number of λ/μ pairs.  Cost is linear; the effect saturates.
    lam : float
        The smoothing weight, in ``(0, 1)``.
    mu : float, optional
        The un-shrinking weight, negative and larger in magnitude than ``lam``.
        Defaults to Taubin's rule ``1 / (k - 1 / lam)`` with a pass band of
        ``k = 0.1``, which leaves wavelengths longer than about twenty vertices
        alone and removes what is shorter.  For ``lam = 0.5`` that is -0.526.

    Returns
    -------
    list of :class:`Path`
        Same count, same order, same kinds.  Closed paths stay closed, and an
        open path keeps both its endpoints exactly.

    Notes
    -----
    This moves vertices, so run it before anything that measures the path, and
    follow it with :func:`simplify` — smoothing makes the polyline much easier
    to describe with fewer points than it arrived with.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must lie in (0, 1), got {lam}")
    if mu is None:
        mu = 1.0 / (0.1 - 1.0 / lam)
    if mu >= 0.0:
        raise ValueError(f"mu must be negative, got {mu}")

    out = []
    for path in _as_paths(paths):
        q = onp.asarray(path.nodes, dtype=float).copy()
        if len(q) >= 4:
            closed = path.is_closed
            for _ in range(max(int(iterations), 0)):
                q = _laplacian_step(q, closed, lam)
                q = _laplacian_step(q, closed, mu)
        out.append(Path(q, path.kind, path.path_id))
    return out


def _rdp_keep(points, tolerance: float):
    """Ramer-Douglas-Peucker: which vertices to keep, as a boolean mask.

    Iterative rather than recursive -- a contour off a fine mesh runs to a few
    thousand vertices, and the worst case recurses once per vertex.
    """
    n = len(points)
    keep = onp.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        span = points[j] - points[i]
        length = float(onp.hypot(*span))
        rel = points[i + 1:j] - points[i]
        if length <= 1e-30:
            # A closed path starts and ends at the same vertex, so the first
            # split has no chord at all; distance to the point is the measure.
            d = onp.hypot(rel[:, 0], rel[:, 1])
        else:
            d = onp.abs(rel[:, 0] * span[1] - rel[:, 1] * span[0]) / length
        k = int(onp.argmax(d))
        if d[k] > tolerance:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return keep


def simplify(paths, *, tolerance: float) -> list[Path]:
    """Drop vertices that no one would miss, keeping the shape to ``tolerance``.

    Contours come off the stripe mesh at roughly one vertex per element, which
    is far more than the geometry needs: a straight run through a member is
    hundreds of collinear points.  This removes a vertex whenever doing so moves
    the polyline by less than ``tolerance``, so the error is bounded everywhere
    rather than on average.

    Use it before writing a file, not before measuring.  Spacing and travel are
    properties of the curve and survive it, but anything that counts nodes --
    :func:`path_lengths` is fine, a per-node statistic is not -- sees a
    different sampling afterwards.

    Parameters
    ----------
    paths : sequence of :class:`Path` or of ``(n, 2)`` arrays
    tolerance : float
        Maximum distance, **in the mesh's units**, that the simplified polyline
        may sit from the original.  At print scale the useful range is well
        under the nozzle: 0.02 mm is invisible, 0.1 mm starts to flatten tight
        corners.

    Returns
    -------
    list of :class:`Path`
        Same order, same kinds, same endpoints.  Closed paths stay closed: the
        shared first and last vertex is always kept.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")
    out = []
    for path in _as_paths(paths):
        nodes = onp.asarray(path.nodes, dtype=float)
        if len(nodes) <= 2 or tolerance == 0:
            out.append(path)
            continue
        out.append(Path(nodes[_rdp_keep(nodes, tolerance)].copy(),
                        path.kind, path.path_id))
    return out


#: SI prefixes STEP names, for the length unit the file declares.
_STEP_UNITS = {"m": "$", "mm": ".MILLI.", "cm": ".CENTI.", "um": ".MICRO."}


def _step_real(v: float) -> str:
    """STEP reals must carry a point: ``1`` is an integer, ``1.`` is a real."""
    t = f"{v:.10g}"
    return t if ("." in t or "e" in t or "E" in t) else t + "."


def _step_text(t: str) -> str:
    """A STEP string literal.  Quotes double; control characters are refused."""
    if any(ord(c) < 32 for c in t):
        raise ValueError(f"{t!r} cannot go in a STEP string: control characters")
    return "'" + t.replace("'", "''") + "'"


def write_step(paths, filename, *, scale: float = 1000.0, units: str = "mm",
               z: float = 0.0, name: str = "toolpath") -> None:
    """Write paths to STEP (AP214 ``automotive_design``) as wireframe curves.

    AP214 rather than AP203 because every CAD package reads it and it is the
    more permissive of the two about geometry that bounds no solid, which is
    all a toolpath ever is.

    Each path becomes one ``POLYLINE`` and each ``kind`` one named
    ``GEOMETRIC_CURVE_SET``, so the grouping matches :func:`write_svg`'s groups
    and :func:`write_dxf`'s layers.  As with DXF, a run that has been through
    :func:`connect` exports as it prints: the welded bridges are interior
    vertices like any other.

    This is curves, not solids.  STEP's reason for existing is boundary
    representation, and a toolpath is not one -- what it is good for here is
    bringing the paths into CAD beside the part model to measure and to draw.
    A bead swept into a solid is a different file and needs a geometry kernel.

    One vertex becomes one ``CARTESIAN_POINT`` entity, so the file is large in
    proportion to the sampling.  Run :func:`simplify` first: on a real part it
    removes ~95% of the vertices while moving the curve less than the tolerance
    you name.

    Parameters
    ----------
    scale : float
        Mesh units to file units, default metres to millimetres.
    units : {"mm", "m", "cm", "um"}
        The length unit the file **declares**.  STEP records this, unlike DXF,
        so it has to agree with ``scale`` -- the default pair does.
    z : float
        Plane to place the curves on, in file units.  One layer per file; STEP
        has no notion of a print layer.
    name : str
        Product name, which is what CAD shows in its tree.

    See Also
    --------
    write_dxf : the same curves in DXF R12, if the receiving tool prefers it.
    simplify : thin the vertices first.
    """
    if units not in _STEP_UNITS:
        raise ValueError(f"units must be one of {sorted(_STEP_UNITS)}, got {units!r}")
    items = _as_paths(paths)
    if not items:
        warnings.warn("writing a STEP file with no paths", stacklevel=2)
    groups = by_kind(items)

    lines: list[str] = []
    nid = 0

    def put(body: str) -> str:
        """Append one entity instance and hand back its name."""
        nonlocal nid
        nid += 1
        lines.append(f"#{nid}={body};")
        return f"#{nid}"

    # The AP203 product skeleton.  None of it is optional: a bare curve set
    # with no product behind it loads as an empty assembly in most CAD.
    ctx = put("APPLICATION_CONTEXT('automotive design')")
    put(f"APPLICATION_PROTOCOL_DEFINITION('international standard',"
        f"'automotive_design',2000,{ctx})")
    pctx = put(f"PRODUCT_CONTEXT('',{ctx},'mechanical')")
    prod = put(f"PRODUCT({_step_text(name)},{_step_text(name)},'',({pctx}))")
    pdf = put("PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',"
              f"{prod},.NOT_KNOWN.)")
    dctx = put(f"PRODUCT_DEFINITION_CONTEXT('part definition',{ctx},'design')")
    pdef = put(f"PRODUCT_DEFINITION('design','',{pdf},{dctx})")
    shape = put(f"PRODUCT_DEFINITION_SHAPE('','',{pdef})")

    origin = put("CARTESIAN_POINT('',(0.,0.,0.))")
    zdir = put("DIRECTION('',(0.,0.,1.))")
    xdir = put("DIRECTION('',(1.,0.,0.))")
    axis = put(f"AXIS2_PLACEMENT_3D('',{origin},{zdir},{xdir})")
    length_unit = put(f"( LENGTH_UNIT() NAMED_UNIT(*) "
                      f"SI_UNIT({_STEP_UNITS[units]},.METRE.) )")
    angle_unit = put("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    solid_unit = put("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    tolerance = put(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),"
                    f"{length_unit},'distance_accuracy_value','confusion accuracy')")
    geo_ctx = put(f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) "
                  f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({tolerance})) "
                  f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({length_unit},{angle_unit},"
                  f"{solid_unit})) REPRESENTATION_CONTEXT('','3D') )")

    zs = _step_real(z)
    sets = []
    for kind, members in groups.items():
        curves = []
        for path in members:
            # A closed path already repeats its first vertex, which is exactly
            # how a STEP POLYLINE closes -- nothing to add or remove.
            nodes = onp.asarray(path.nodes, dtype=float) * scale
            pts = [put(f"CARTESIAN_POINT('',({_step_real(x)},{_step_real(y)},{zs}))")
                   for x, y in nodes]
            curves.append(put(f"POLYLINE('',({','.join(pts)}))"))
        sets.append(put(f"GEOMETRIC_CURVE_SET({_step_text(kind)},"
                        f"({','.join(curves)}))"))

    rep = put(f"GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION("
              f"{_step_text(name)},({axis}{''.join(',' + s for s in sets)}),"
              f"{geo_ctx})")
    put(f"SHAPE_DEFINITION_REPRESENTATION({shape},{rep})")

    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S")
    desc = ", ".join(f"{len(v)} {k}" for k, v in groups.items()) or "empty"
    header = [
        "ISO-10303-21;",
        "HEADER;",
        f"FILE_DESCRIPTION(('print paths by kind: {desc}'),'2;1');",
        f"FILE_NAME({_step_text(str(filename))},'{stamp}',(''),(''),"
        "'path_optimizer','path_optimizer','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
        "ENDSEC;",
        "DATA;",
    ]
    with open(filename, "w") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(lines))
        f.write("\nENDSEC;\nEND-ISO-10303-21;\n")


# ── FullControl handoff ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Fc:
    """The handful of FullControl classes this module needs."""

    Point: type
    Extruder: type
    ExtrusionGeometry: type
    Printer: type
    GcodeComment: type
    ManualGcode: type
    PrinterCommand: type


def _import_fullcontrol():
    """The module itself, with a message that says how to get it."""
    try:
        import fullcontrol as fc
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "FullControl is required to build a design; install it with "
            "`pip install git+https://github.com/FullControlXYZ/fullcontrol`"
        ) from exc
    return fc


def _fullcontrol():
    fc = _import_fullcontrol()
    return _Fc(fc.Point, fc.Extruder, fc.ExtrusionGeometry, fc.Printer,
               fc.GcodeComment, fc.ManualGcode, fc.PrinterCommand)


def tool_change(tool, *, relative_e: bool | None = None) -> list:
    """Steps that select ``tool``, for use as one entry of ``on_kind``.

    ``tool`` is a tool number, emitted as ``Tn``, or a string passed through
    verbatim for machines whose change is not a bare T word.  It goes out as
    ``fc.ManualGcode``, which is how FullControl's own ``toolchanger_T0``..``T3``
    printer profiles select theirs; there is no portable tool-change step, so
    this will not show up in ``fc.transform(steps, "plot")``.

    Parameters
    ----------
    relative_e : bool, optional
        **Set this to whatever the ``GcodeControls`` will use.**  Those profiles
        choose one tool for the whole print, and FullControl tracks a single
        extruder position (``MultitoolPrinter`` is commented out upstream), so a
        mid-print change leaves the E word running on from the previous tool's
        total.  Harmless under relative E, where every E is a delta; wrong under
        absolute E, where the incoming tool is told to move to a position it was
        never at.

        Given, this appends ``fc.Extruder(relative_gcode=relative_e)``, which
        re-declares the mode *and* resets FullControl's own datum, so the emitted
        ``G92 E0`` and the internal total stay in step.  Adding a bare
        ``G92 E0`` yourself does **not** work: the internal total keeps counting
        and the next move jumps straight back to it.  Measured on a two-tool
        design under absolute E: 29.3, 30.1, 30.9, 31.8 across the change
        without this; 0.83, 1.66, 0.83, 1.66 with it.

    Examples
    --------
    >>> to_fullcontrol(order_paths(paths), z=0.2, on_kind={
    ...     "pla": tool_change(0, relative_e=False),
    ...     "tpu": tool_change(1, relative_e=False),
    ... })                                                  # doctest: +SKIP
    """
    fc = _fullcontrol()
    steps = [fc.ManualGcode(text=tool if isinstance(tool, str) else f"T{tool}")]
    if relative_e is not None:
        steps.append(fc.Extruder(relative_gcode=relative_e))
    return steps


def path_steps(path, *, z: float = 0.0, scale: float = 1000.0,
               annotate: bool = True, travel: bool = True,
               retract: bool = False, hop: float | None = None,
               label: str | None = None) -> list:
    """One path as FullControl steps — the unit :func:`to_fullcontrol` is built from.

    Use this when the design needs something of yours between the paths: a
    purge, a pause, a z hop, a comment of your own, a speed change for one
    stripe.  Building the list yourself and calling this per path is the
    supported way to do that; :func:`to_fullcontrol` is the same loop with no
    room in the middle.

    .. code-block:: python

        steps = [fc.ExtrusionGeometry(width=0.4, height=0.2)]
        for path in paths.order_paths(job):
            if path.kind == "support":
                steps.append(fc.Printer(print_speed=3000))
            steps += paths.path_steps(path, z=0.2)

    Parameters
    ----------
    path : :class:`Path` or ``(n, 2)`` array
        In the mesh's units.
    z : float
        Layer height, **already in FullControl's units** (millimetres) — unlike
        the path coordinates, which are scaled.
    scale : float
        Multiplies the path coordinates.  The default 1000 takes metres, which
        is what the FE side works in, to the millimetres FullControl expects.
        Getting this wrong is the easiest way to print something 1000x too
        small, so it is explicit rather than inferred.
    annotate : bool
        Emit a leading ``fc.GcodeComment`` naming the path's id and kind.  It is
        the only place the path semantics survive into the printer file.
    travel : bool
        Move to the path's start with the extruder off.  Turn it off when the
        head is already there — after a weld, or when you have emitted your own
        approach — so no spurious travel is inserted.  ``retract`` and ``hop``
        do nothing without it.
    retract : bool
        Retract before the travel and unretract on arrival, as
        ``fc.PrinterCommand(id="retract")`` / ``"unretract"``.  What those
        become is the printer profile's ``printer_command_list`` — ``G10`` and
        ``G11`` by default, i.e. firmware retraction.  A profile whose command
        list lacks the keys raises rather than silently omitting them.
    hop : float, optional
        Z lift for the travel, in millimetres, same units as ``z``.  The head
        rises in place, crosses at ``z + hop``, and descends at the far end, so
        it does not drag over what has already been laid down.  Costs two extra
        moves per path.
    label : str, optional
        Text for the comment, replacing the default ``"{kind} path {id}"``.

    Returns
    -------
    list
        FullControl design steps for this path alone.
    """
    fc = _fullcontrol()
    if z is None:
        raise ValueError("z is required: FullControl points are 3D")
    path = path if isinstance(path, Path) else Path(path)
    pts = path.nodes * scale

    steps: list = []
    if annotate:
        steps.append(fc.GcodeComment(text=label if label is not None else (
            f"{path.kind} path {path.path_id}"
            f"{' (closed)' if path.is_closed else ''}")))
    if travel:
        # Extruder off first: the lift is a move like any other, and it must not
        # extrude on the way up.
        steps.append(fc.Extruder(on=False))
        if retract:
            steps.append(fc.PrinterCommand(id="retract"))
        if hop:
            # Straight up: x and y carry over from wherever the head is.  That
            # only works if something has set them -- as the very first Point of
            # a design, FullControl fills them with 0 and the head dives for the
            # origin.  to_fullcontrol suppresses the hop on the first path for
            # exactly that reason; doing the same if you call this yourself.
            steps.append(fc.Point(z=z + hop))
        steps.append(fc.Point(x=float(pts[0, 0]), y=float(pts[0, 1]),
                              z=z + hop if hop else z))
        if hop:
            steps.append(fc.Point(z=z))
        if retract:
            steps.append(fc.PrinterCommand(id="unretract"))
        steps.append(fc.Extruder(on=True))
        pts = pts[1:]
    return steps + [fc.Point(x=float(x), y=float(y), z=z) for x, y in pts]


def to_fullcontrol(paths, *, z: float = 0.0, scale: float = 1000.0,
                   width: float | None = None, height: float | None = None,
                   print_speed: float | None = None,
                   travel_speed: float | None = None,
                   annotate: bool = True, retract: bool = False,
                   hop: float | None = None,
                   on_kind: dict | None = None) -> list:
    """Build a FullControl design from paths.

    Returns a ``steps`` list ready for ``fc.transform(steps, "gcode", ...)`` or
    ``fc.transform(steps, "plot")``.  Nothing is written to disk and no printer
    profile is assumed — those belong to the ``fc.GcodeControls`` the caller
    supplies.

    This is an ``fc.ExtrusionGeometry`` and an ``fc.Printer`` followed by
    :func:`path_steps` per path, with ``on_kind`` fired on each change of kind.
    When you need something of your own in between, write that loop yourself —
    :func:`path_steps` is the only part of it this package has to supply.

    Parameters
    ----------
    paths : sequence of :class:`Path` or of ``(n, 2)`` arrays
        In the mesh's units, **in the order they should be printed** — this
        function lays them down as given.  Run them through :func:`order_paths`
        first unless the order already means something.
    z : float
        Layer height to place this set of paths at, **already in FullControl's
        units** (millimetres) — unlike the paths, which are scaled.
    scale : float
        Multiplies the path coordinates.  The default 1000 takes metres, which
        is what the FE side works in, to the millimetres FullControl expects.
        Getting this wrong is the easiest way to print something 1000x too small,
        so it is explicit rather than inferred.
    width, height : float, optional
        Extrusion cross-section in mm.  Emitted once, up front, as an
        ``fc.ExtrusionGeometry``; omit to leave whatever the printer profile
        sets.
    print_speed, travel_speed : float, optional
        mm/min, emitted once as an ``fc.Printer``.
    annotate : bool
        Emit a G-code comment before each path naming its id and kind.  Cheap,
        and it is the only place the path semantics survive into the printer
        file — without it the output is an undifferentiated stream of moves.
    retract, hop : see :func:`path_steps`
        Applied to every travel except the first, which gets no hop — there is
        nothing laid down to clear, and a bare z move at the head of a design
        has no x or y to carry over.
    on_kind : dict, optional
        ``{kind: steps}`` — FullControl steps to emit when that kind starts,
        and again whenever it starts after a different kind has intervened.
        Kinds absent from the dict get nothing.

        This is the general form of "do something per group": a purge, a fan
        change, a different extrusion width, a comment, or a tool change.
        Nothing here decides what a kind means, so a multi-kind design is not by
        itself a multi-tool design — you opt in by naming the steps.
        :func:`tool_change` builds the tool case.

        Emission is on *change*, so ordering with :func:`order_paths` — which
        keeps each kind contiguous — is what makes this once per kind instead of
        once per path.

    Returns
    -------
    list
        FullControl design steps.
    """
    fc = _fullcontrol()
    steps: list = []
    if width is not None or height is not None:
        steps.append(fc.ExtrusionGeometry(width=width, height=height))
    if print_speed is not None or travel_speed is not None:
        steps.append(fc.Printer(print_speed=print_speed,
                                travel_speed=travel_speed))
    current_kind = object()                          # nothing started yet
    for i, path in enumerate(_as_paths(paths)):
        if path.kind != current_kind:
            steps += list((on_kind or {}).get(path.kind, ()))
            current_kind = path.kind
        label = None if path.path_id is not None else (
            f"{path.kind} path {i}{' (closed)' if path.is_closed else ''}")
        # No hop on the first path: there is nothing laid down to clear yet, and
        # a bare z-Point at the head of a design has no x or y to carry over.
        steps += path_steps(path, z=z, scale=scale, annotate=annotate,
                            retract=retract, hop=None if i == 0 else hop,
                            label=label)
    return steps


def to_fullcontrol_layers(layers, *, z0: float = 0.0,
                          layer_height: float = 0.2, **kwargs) -> list:
    """Stack several path sets into one FullControl design, one per layer.

    ``layers`` is a sequence of path lists, bottom first — for example the
    per-layer output of :func:`path_optimizer.stripes.stripes_from_result`,
    each run through :func:`extract_paths`.  Layer *k* is placed at
    ``z0 + k * layer_height``; ``height`` defaults to ``layer_height`` so the
    extrusion model matches the spacing unless told otherwise.
    """
    kwargs.setdefault("height", layer_height)
    steps: list = []
    for k, paths in enumerate(layers):
        steps += to_fullcontrol(paths, z=z0 + k * layer_height, **kwargs)
    return steps
