"""Print paths → B-rep solids → STEP.

A toolpath is a curve; what the machine lays down is a bead of finite width and
height.  This turns the one into the other, as real boundary representation
rather than a mesh: each bead is the path's footprint -- the region swept by a
disc of radius ``width / 2`` -- pushed up by the layer height.

.. code-block:: python

    from path_optimizer import paths, solids

    final = paths.order_paths(paths.connect(paths.order_paths(polylines),
                                            tolerance=5.0e-3))
    solids.write_step_solid(final, "beads.step", width=2.0, height=0.2)

Why the footprint and not a left/right offset of the polyline: a toolpath turns
much tighter than the bead is wide.  On a real part 3.4% of the vertices turned
inside the bead's own half-width, down to a radius of 0.29 mm for a 1.0 mm
half-width, and a plain two-sided offset crosses itself at every one of them.
The swept region is what the printer actually deposits there -- the material
simply overlaps -- and it is always a valid area.

This is the only part of :mod:`path_optimizer` that needs a geometry kernel.
OpenCASCADE arrives through ``cadquery-ocp`` and is installed with the package:
it ships manylinux wheels for every Python a notebook is likely to be running,
so there is no build step even on Colab.  It is still imported on first use
rather than at module scope, so an installation that somehow lacks it fails
where it is used and not on ``import path_optimizer``.

.. note::

   A solid is worth what its tolerance is worth.  Run
   :func:`path_optimizer.paths.simplify` first -- the contours carry about one
   vertex per stripe-mesh element and every one of them becomes faces in the
   file.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as onp

from path_optimizer.paths import _as_paths, turn_radius

#: Re-exported from :mod:`path_optimizer.paths` -- the same measure decides
#: whether a bead can follow a path and whether it needs trimming first.
__all__ = ["BeadReport", "bead_compound", "write_step_solid", "turn_radius",
           "read_step", "tessellate", "to_pyvista", "to_plotly", "plot"]


_OCC_HINT = (
    "path_optimizer.solids needs OpenCASCADE, which is not importable.\n"
    "    pip install cadquery-ocp\n"
    "It installs with path_optimizer, so finding it missing means the install "
    "was partial or the wrong environment is active. For the toolpath as curves "
    "rather than solids, paths.write_step and paths.write_dxf need no kernel."
)


def _occ():
    """The OpenCASCADE names this module uses, or an error saying how to get them."""
    try:
        from OCP.BRep import BRep_Builder
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakePolygon,
            BRepBuilderAPI_Transform,
        )
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.GeomAbs import GeomAbs_JoinType
        from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
        from OCP.GProp import GProp_GProps
        from OCP.ShapeFix import ShapeFix_Shape
        from OCP.TopAbs import TopAbs_ShapeEnum
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Compound
    except ImportError as exc:                           # pragma: no cover
        raise ImportError(_OCC_HINT) from exc
    return SimpleNamespace(
        Builder=BRep_Builder, Compound=TopoDS_Compound, TopoDS=TopoDS,
        Fuse=BRepAlgoAPI_Fuse,
        MakeFace=BRepBuilderAPI_MakeFace, MakePolygon=BRepBuilderAPI_MakePolygon,
        Transform=BRepBuilderAPI_Transform, Analyzer=BRepCheck_Analyzer,
        GProp=BRepGProp, Props=GProp_GProps, MakeOffset=BRepOffsetAPI_MakeOffset,
        MakePrism=BRepPrimAPI_MakePrism, JoinType=GeomAbs_JoinType,
        Pnt=gp_Pnt, Trsf=gp_Trsf, Vec=gp_Vec, ShapeFix=ShapeFix_Shape,
        ShapeEnum=TopAbs_ShapeEnum, Explorer=TopExp_Explorer)


@dataclass
class BeadReport:
    """What became of each path on the way to a solid.

    Attributes
    ----------
    beads : int
        Paths that produced material.  Compare it with the number you passed:
        anything missing is in ``dropped``.
    solids : int
        ``TopoDS_Solid`` count in the compound, which is larger than ``beads``
        when a path had to be ``split``.
    faces : int
        Faces across all of them -- the first thing that decides file size.
    dropped : list of int
        Paths that produced no face at all.  Their geometry is missing from
        the output; they are not merely untidy.
    repaired : list of int
        Paths that needed ``ShapeFix`` before they checked out.
    split : list of int
        Paths whose offset failed whole and had to be cut and unioned.  The
        region is the same; the bead may arrive as several solids.
    invalid : list of int
        Paths still failing ``BRepCheck`` after repair.  They are kept, since
        most CAD will heal them on import, but they are named so the decision
        is yours.
    tight : list of int
        Paths that turn tighter than the bead's half-width somewhere.  This is
        a property of the toolpath, not of the export, and it is what the other
        two lists usually trace back to.
    """

    beads: int = 0
    solids: int = 0
    faces: int = 0
    dropped: list[int] = field(default_factory=list)
    repaired: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    tight: list[int] = field(default_factory=list)
    split: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        bits = [f"{self.beads} beads, {self.solids} solids, "
                f"{self.faces} faces"]
        for name, xs in (("dropped", self.dropped), ("split", self.split),
                         ("repaired", self.repaired), ("invalid", self.invalid),
                         ("tight turns", self.tight)):
            if xs:
                bits.append(f"{len(xs)} {name}")
        return "; ".join(bits)


def _offset_wires(occ, spine, distance):
    """Outward (or inward, if negative) offset of a wire, with arc joins.

    Arc joins, not mitred ones: a mitre at a hairpin runs off to infinity and
    the printed bead really does turn in a semicircle there.
    """
    mk = occ.MakeOffset(
        spine, occ.JoinType.GeomAbs_Arc, False)
    mk.Perform(distance)
    if not mk.IsDone():
        return []
    out = []
    exp = occ.Explorer(mk.Shape(), occ.ShapeEnum.TopAbs_WIRE)
    while exp.More():
        out.append(occ.TopoDS.Wire(exp.Current()))
        exp.Next()
    return out


def _wire_area(occ, wire) -> float:
    mk = occ.MakeFace(wire)
    if not mk.IsDone():
        return 0.0
    props = occ.Props()
    occ.GProp.SurfaceProperties_s(mk.Face(), props)
    return abs(props.Mass())


def _offset_footprint(occ, nodes, half_width, closed):
    """The bead's footprint straight from one offset, or ``None`` if OCC balks."""
    if not closed and len(nodes) == 2:
        # OCC's offset wants more than one edge and returns nothing for a lone
        # segment -- the simplest path there is.  Splitting it at its midpoint
        # changes no geometry and gives the offset the two edges it wants.
        nodes = onp.vstack([nodes[0], 0.5 * (nodes[0] + nodes[1]), nodes[1]])
    poly = occ.MakePolygon()
    for x, y in nodes:
        poly.Add(occ.Pnt(float(x), float(y), 0.0))
    if closed:
        poly.Close()
    spine = poly.Wire()

    wires = _offset_wires(occ, spine, half_width)
    if not wires:
        return None
    if closed:
        # A closed path's bead is a ring: the inward offset is its hole.  That
        # offset collapses when the loop is narrower than the bead, and then a
        # filled disc is the right answer and there is nothing to subtract.
        wires += _offset_wires(occ, spine, -half_width)

    # The enclosing wire is the one of greatest area -- not of most edges, which
    # a hairpin gets wrong -- and every other wire is a hole.  A hole has to run
    # opposite to it, or OCC reads it as a second outer boundary and fills the
    # ring: that mistake put the model 20% over its true volume.
    wires.sort(key=lambda w: _wire_area(occ, w), reverse=True)
    mk = occ.MakeFace(wires[0])
    for hole in wires[1:]:
        mk.Add(occ.TopoDS.Wire(hole.Reversed()))
    return mk.Face() if mk.IsDone() else None


#: How many times :func:`_footprint` may halve a path before giving up on it.
#: Each level doubles the pieces, so six is 64 -- far past where a path that is
#: going to work ever needs.
_MAX_SPLITS = 6


def _footprint(occ, nodes, half_width, closed, depth: int = 0):
    """The planar face the bead covers, splitting the path if OCC needs it.

    A single hairpin can make the offset of a whole 600 mm path fail, taking
    the other 599 mm of it with it.  Cutting the node list at the tightest turn
    and unioning the halves recovers it **exactly**: the footprint of a
    polyline is the union of its segments' footprints, so a cut at a shared
    vertex changes nothing about the region, only how it is computed.

    The union is a real boolean over two or three faces, not an overlap left
    lying in the compound -- the volume stays right.
    """
    face = _offset_footprint(occ, nodes, half_width, closed)
    if face is not None or depth >= _MAX_SPLITS or len(nodes) < 4:
        return face

    radius = turn_radius(nodes)
    cut = int(onp.argmin(radius)) + 1 if len(radius) else len(nodes) // 2
    cut = min(max(cut, 1), len(nodes) - 2)
    # Both halves are open, whatever the original was: they end at the cut.
    # A closed path still comes out a ring, because the footprint of its
    # segments is a ring however the segments are grouped.
    left = _footprint(occ, nodes[:cut + 1], half_width, False, depth + 1)
    right = _footprint(occ, nodes[cut:], half_width, False, depth + 1)
    if left is None or right is None:
        return left if right is None else right
    fuse = occ.Fuse(left, right)
    return fuse.Shape() if fuse.IsDone() else None


def _count(occ, shape, kind) -> int:
    exp, n = occ.Explorer(shape, kind), 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def bead_compound(paths, *, width: float, height: float, z: float = 0.0,
                  scale: float = 1000.0, repair: bool = True):
    """One solid per path, gathered into a ``TopoDS_Compound``.

    The beads are kept as separate solids rather than fused.  At a pitch wider
    than the bead they do not touch, and a boolean over a hundred-odd of them
    costs minutes and splits every face it crosses -- a fused bead measured
    25020 faces where the offset gives 239.

    Parameters
    ----------
    paths : sequence of :class:`~path_optimizer.paths.Path` or ``(n, 2)`` arrays
    width, height : float
        Bead cross-section in the **file's** units, so millimetres under the
        default ``scale``.
    z : float
        Underside of the layer, in file units.
    scale : float
        Mesh units to file units, default metres to millimetres.
    repair : bool
        Run ``ShapeFix`` on a solid that fails ``BRepCheck``.  It recovers
        about a third of them and costs little.

    Returns
    -------
    (compound, :class:`BeadReport`)
        Read the report: a path whose footprint cannot be built is **dropped**,
        and its material is simply absent from the compound.
    """
    occ = _occ()
    items = _as_paths(paths)
    if not items:
        warnings.warn("building a bead compound with no paths", stacklevel=2)
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}, {height}")

    builder = occ.Builder()
    compound = occ.Compound()
    builder.MakeCompound(compound)
    report = BeadReport()
    face_kind = occ.ShapeEnum.TopAbs_FACE

    for k, path in enumerate(items):
        nodes = onp.asarray(path.nodes, dtype=float) * scale
        if turn_radius(nodes).min(initial=onp.inf) < 0.5 * width:
            report.tight.append(k)
        face = _offset_footprint(occ, nodes, 0.5 * width, path.is_closed)
        if face is None:
            face = _footprint(occ, nodes, 0.5 * width, path.is_closed)
            if face is not None:
                report.split.append(k)
        if face is None:
            report.dropped.append(k)
            continue
        solid = occ.MakePrism(
            face, occ.Vec(0.0, 0.0, height)).Shape()
        if z:
            solid = _translated(occ, solid, z)
        if not occ.Analyzer(solid).IsValid():
            if repair:
                fix = occ.ShapeFix(solid)
                fix.Perform()
                solid = fix.Shape()
            if occ.Analyzer(solid).IsValid():
                report.repaired.append(k)
            else:
                report.invalid.append(k)
        report.beads += 1
        report.solids += _count(occ, solid, occ.ShapeEnum.TopAbs_SOLID)
        report.faces += _count(occ, solid, face_kind)
        builder.Add(compound, solid)

    if report.dropped:
        warnings.warn(
            f"{len(report.dropped)} path(s) produced no bead and are missing from "
            f"the output: {report.dropped}", stacklevel=2)
    return compound, report


def _translated(occ, shape, dz):
    trsf = occ.Trsf()
    trsf.SetTranslation(occ.Vec(0.0, 0.0, float(dz)))
    return occ.Transform(shape, trsf, True).Shape()


def write_step_solid(paths, filename, *, width: float, height: float,
                     z: float = 0.0, scale: float = 1000.0, units: str = "MM",
                     repair: bool = True, name: str = "beads") -> BeadReport:
    """Write the beads as STEP solids, analytic surfaces and all.

    Unlike :func:`path_optimizer.paths.write_step`, which writes the toolpath as
    curves, this writes what the material occupies: planes, cylinders and the
    arcs that join them, as a real B-rep that CAD can fillet, section and cut
    against.

    A ``kind`` becomes nothing here.  STEP has no layer, and the compound is
    flat; group the paths before calling if the kinds must stay apart in
    separate files.

    Parameters
    ----------
    width, height, z, scale, repair
        See :func:`bead_compound`.
    units : str
        What the file declares -- ``"MM"``, ``"M"``, ``"INCH"`` and the other
        names OCC accepts.  It has to agree with ``scale``.
    name : str
        Unused by the writer today; kept so a caller can name the product when
        OCC's static settings are given a product name.

    Returns
    -------
    :class:`BeadReport`
        Printed by ``str()`` as one line.  Check ``dropped`` before trusting
        the file to be complete.
    """
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    compound, report = bead_compound(paths, width=width, height=height, z=z,
                                     scale=scale, repair=repair)
    Interface_Static.SetCVal_s("write.step.unit", units)
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    status = writer.Write(str(filename))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"OCC refused to write {filename}: status {status}")
    return report


# ── Looking at it ────────────────────────────────────────────────────────────

def read_step(filename):
    """Read a STEP file back as one OpenCASCADE shape.

    Anything OCC can read, not only what :func:`write_step_solid` wrote.
    """
    occ = _occ()                                      # the import check
    del occ
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(filename)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise OSError(f"OCC could not read {filename}")
    reader.TransferRoots()
    return reader.OneShape()


def tessellate(shape, *, deflection: float = 0.05, angle: float = 0.5):
    """Triangulate a B-rep shape into plain arrays.

    Parameters
    ----------
    shape : TopoDS_Shape or path
        From :func:`bead_compound`, or a STEP file, which is read first.
    deflection : float
        Greatest distance the triangles may sit from the true surface, in the
        shape's own units.  A bead's flat faces are exact at any value; this
        decides how finely the round ends and the arc joins are cut.
    angle : float
        Angular deflection, in radians.

    Returns
    -------
    (vertices, triangles)
        ``(n, 3)`` float and ``(m, 3)`` int arrays.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    if isinstance(shape, (str, os.PathLike)):
        shape = read_step(shape)
    BRepMesh_IncrementalMesh(shape, float(deflection), False, float(angle), True)

    points: list[onp.ndarray] = []
    faces: list[onp.ndarray] = []
    offset = 0
    exp = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        location = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, location)
        exp.Next()
        if tri is None:
            continue
        transform = location.Transformation()
        nodes = onp.array([
            [(p := tri.Node(i).Transformed(transform)).X(), p.Y(), p.Z()]
            for i in range(1, tri.NbNodes() + 1)])
        cells = onp.array([
            [t.Value(1) - 1, t.Value(2) - 1, t.Value(3) - 1]
            for t in (tri.Triangle(i) for i in range(1, tri.NbTriangles() + 1))])
        points.append(nodes)
        if len(cells):
            faces.append(cells + offset)
        offset += len(nodes)
        del location

    if not points:
        return onp.zeros((0, 3)), onp.zeros((0, 3), dtype=int)
    return onp.vstack(points), (onp.vstack(faces) if faces
                                else onp.zeros((0, 3), dtype=int))


def to_pyvista(shape, *, deflection: float = 0.05, angle: float = 0.5):
    """The shape as a :class:`pyvista.PolyData`, ready to plot or to save.

    ``shape`` may also be a path to a STEP file, which is read first.
    """
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:                # pragma: no cover
        if "guarded_eval" not in str(exc):
            raise
        import IPython

        raise ImportError(
            f"pyvista cannot be imported under IPython {IPython.__version__}. "
            "From 0.49 it checks that IPython is loaded and then imports "
            "IPython.core.guarded_eval, which only exists from IPython 8.11, so "
            "in a notebook on an older IPython -- Colab's, at the time of "
            "writing -- importing it raises. path_optimizer asks for "
            "pyvista<0.49 for that reason; this environment has a newer one.\n"
            '    pip install -q "pyvista[jupyter]<0.49"\n'
            "then restart the kernel."
        ) from exc

    vertices, triangles = tessellate(shape, deflection=deflection, angle=angle)
    if not len(triangles):
        return pv.PolyData()
    cells = onp.hstack([onp.full((len(triangles), 1), 3), triangles]).ravel()
    return pv.PolyData(vertices, cells)


def _in_notebook() -> bool:
    """True inside a Jupyter/Colab kernel, false in a plain interpreter."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ in (
        "ZMQInteractiveShell", "Shell")          # Jupyter, Colab


#: Eye directions for the named views, in units of the scene's own size.
_EYE = {"iso": (1.4, -1.4, 1.1), "xy": (0.0, 0.0, 2.2),
        "xz": (0.0, -2.2, 0.0), "yz": (2.2, 0.0, 0.0)}


def to_plotly(shape, *, deflection: float = 0.05, angle: float = 0.5,
              color: str = "#2a78d6", view: str = "iso",
              background: str = "#fcfcfb", flatshading: bool = True):
    """The shape as a :class:`plotly.graph_objects.Figure`.

    The same tessellation as :func:`to_pyvista`, in the library the rest of the
    notebook already draws with.  Plotly renders in the page from data the cell
    carries, so unlike PyVista's interactive backends it needs no server to
    reach -- which is what makes it the one that works on Colab.

    The whole mesh travels into the output: 181k triangles came to 10.7 MB of
    inline HTML.  ``deflection`` is the knob, and it is worth turning on a
    large part.
    """
    import plotly.graph_objects as go

    if view not in _EYE:
        raise ValueError(
            f"view must be one of {sorted(_EYE)}; got {view!r}")
    vertices, triangles = tessellate(shape, deflection=deflection, angle=angle)
    eye = _EYE[view]
    fig = go.Figure(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color=color, flatshading=flatshading, hoverinfo="skip"))
    blank = dict(showbackground=False, showgrid=False, showticklabels=False,
                 zeroline=False, title="")
    fig.update_layout(
        # aspectmode="data" keeps the part's proportions; anything else
        # stretches a long beam to fill a cube.
        scene=dict(xaxis=blank, yaxis=blank, zaxis=blank, aspectmode="data",
                   camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                               projection=dict(type="orthographic"))),
        paper_bgcolor=background, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False)
    return fig


def plot(shape, *, deflection: float = 0.05, angle: float = 0.5,
         color: str = "#2a78d6", view: str = "iso", zoom: float = 1.0,
         engine: str | None = None, window_size=(1600, 900),
         show_edges: bool = False, background: str = "#fcfcfb", screenshot=None,
         jupyter_backend: str | None = None, **kwargs):
    """Open PyVista's viewer on the beads.

    Interactive: rotate, pan, section, measure.  In a notebook the viewer
    appears in the output cell, rendered by trame, which ``pyvista[jupyter]``
    brings in and which this package installs.

    In a notebook this draws with **Plotly** and returns the figure, so the cell
    renders it.  Outside one it opens a PyVista window.  Plotly because it
    renders from data the cell carries and needs nothing to connect to, which
    is the difference that matters: PyVista's interactive backends all point
    the page at a trame server on localhost, and a Colab notebook cannot reach
    it -- the cell comes back "connection refused" and shows nothing.  Pass
    ``engine="pyvista"`` to insist, on a kernel that can serve it.

    .. code-block:: python

        from path_optimizer import solids

        solids.plot("beads.step")                     # straight from the file
        solids.plot(compound, view="xy", show_edges=True)

    Either way the scene travels into the notebook, so ``deflection`` decides
    both how responsive the viewer is and how large the saved ``.ipynb``
    becomes: 181k triangles came to 10.7 MB of inline HTML.

    Parameters
    ----------
    shape : TopoDS_Shape or path
        A shape from :func:`bead_compound`, or a STEP file, which is read first.
    deflection : float
        Tessellation tolerance, in the shape's units.  The flat faces are exact
        whatever it is; this is how finely the round ends are cut.  Loosen it on
        a large part -- 0.2 mm took a 1 m beam to 162k triangles in 1.8 s.
    color : str
        One colour for the whole part.  Toolpath beads are one material and one
        process; colouring them apart would say something that is not so.
    view : {"iso", "xy", "xz", "yz"}
        Starting camera.  ``"iso"`` shows the extrusion, ``"xy"`` looks
        straight down, which is how a toolpath is read.  Projection is parallel
        either way: under perspective the paths at the far side of a metre-long
        part look finer than the ones in front, and they are not.
    zoom : float
        Applied once the camera has been fitted to the part.  PyVista only;
        Plotly fits the part and leaves the zoom to the mouse.
    engine : {"plotly", "pyvista"}, optional
        ``None`` means Plotly in a notebook and PyVista outside one, which is
        where each works.  ``screenshot`` forces PyVista, since saving a Plotly
        figure as an image needs kaleido on top.
    show_edges : bool
        Draw the triangle edges.  Off by default -- the tessellation's edges
        are not the part's, and at this scale they fill the picture.
    screenshot : path, optional
        Save to this file instead of opening the viewer.  For a machine with no
        display; PyVista needs ``PYVISTA_OFF_SCREEN=true`` or an X server.
    jupyter_backend : str, optional
        PyVista engine only.  ``None`` means ``"html"`` in a notebook, which is
        the one PyVista backend that embeds the scene instead of serving it;
        ``"static"`` gives a still image.  ``"client"`` and ``"server"`` need a
        reachable trame server and do not work on Colab.

    Returns
    -------
    figure, viewer or :class:`pyvista.Plotter`
        Whatever has to be the cell's value for it to render: a Plotly figure,
        or a PyVista viewer.  Outside a notebook, the plotter, after showing or
        saving, so a caller can take the camera or add to the scene.
    """
    notebook = _in_notebook()
    if engine is None:
        engine = "pyvista" if screenshot is not None or not notebook else "plotly"
    if engine not in ("plotly", "pyvista"):
        raise ValueError(
            f"engine must be 'plotly' or 'pyvista'; got {engine!r}")
    if engine == "plotly":
        if screenshot is not None:
            raise ValueError("screenshot needs engine='pyvista'")
        fig = to_plotly(shape, deflection=deflection, angle=angle, color=color,
                        view=view, background=background, **kwargs)
        return fig if notebook else fig.show()

    import pyvista as pv

    mesh = to_pyvista(shape, deflection=deflection, angle=angle)
    plotter = pv.Plotter(off_screen=screenshot is not None,
                         window_size=list(window_size))
    plotter.set_background(background)
    plotter.add_mesh(mesh, color=color, show_edges=show_edges,
                     smooth_shading=False, **kwargs)
    plotter.enable_parallel_projection()
    try:
        {"xy": plotter.view_xy, "xz": plotter.view_xz, "yz": plotter.view_yz,
         "iso": plotter.view_isometric}[view]()
    except KeyError:
        raise ValueError(
            f"view must be one of 'iso', 'xy', 'xz', 'yz'; got {view!r}") from None
    plotter.reset_camera()
    if zoom != 1.0:
        plotter.camera.zoom(zoom)

    if screenshot is not None:
        plotter.screenshot(str(screenshot))
        plotter.close()
        return plotter

    if jupyter_backend is None and notebook:
        jupyter_backend = "html"
    if notebook:
        # Returned, not shown: a widget renders because it is the cell's value.
        # Calling show() and then returning the plotter prints its repr and
        # nothing else.
        return plotter.show(jupyter_backend=jupyter_backend, return_viewer=True)
    if jupyter_backend is not None:
        plotter.show(jupyter_backend=jupyter_backend)
    else:
        plotter.show()
    return plotter
