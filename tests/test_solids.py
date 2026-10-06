"""Tests for the bead solids and their STEP export.

OpenCASCADE is optional, so every test that builds geometry is skipped without
it; `turn_radius` is pure numpy and always runs.
"""
import math

import numpy as onp
import pytest

from path_optimizer import paths as P
from path_optimizer import solids

OCC = pytest.importorskip("OCP", reason="needs cadquery-ocp")


def line(y=0.0, x0=0.0, x1=0.05, n=2):
    return onp.stack([onp.linspace(x0, x1, n), onp.full(n, y)], axis=1)


def circle(r=0.02, n=80):
    t = onp.linspace(0.0, 2.0 * onp.pi, n)
    xy = onp.stack([r * onp.cos(t), r * onp.sin(t)], axis=1)
    xy[-1] = xy[0]
    return xy


def volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


# ── turn_radius (no kernel needed) ───────────────────────────────────────────

def test_turn_radius_is_infinite_on_a_straight_run():
    assert onp.isinf(solids.turn_radius(line(n=10))).all()


def test_turn_radius_recovers_a_known_circle():
    r = solids.turn_radius(circle(r=7.0, n=200))
    assert r == pytest.approx(7.0, rel=1e-3)


def test_turn_radius_is_empty_for_a_two_point_path():
    assert len(solids.turn_radius(line(n=2))) == 0


# ── the bead itself ──────────────────────────────────────────────────────────

def test_a_lone_segment_is_not_dropped():
    # OCC's offset returns nothing for a single-edge wire; the module splits it
    # rather than losing the simplest path there is.
    _, rep = solids.bead_compound([line(n=2)], width=2.0, height=0.2)
    assert rep.beads == 1 and not rep.dropped


def test_a_straight_bead_has_the_volume_of_its_sweep():
    # A 50 mm run of 2 mm bead, 0.2 mm tall, with a semicircular cap at each
    # end: w*h*L for the body plus one full disc's worth between the two caps.
    comp, rep = solids.bead_compound([line()], width=2.0, height=0.2)
    want = 2.0 * 0.2 * 50.0 + math.pi * 1.0**2 * 0.2
    assert volume(comp) == pytest.approx(want, rel=1e-3)
    assert rep.beads == 1 and rep.solids == 1 and not rep.dropped


def test_a_closed_bead_is_a_ring_and_not_a_disc():
    comp, rep = solids.bead_compound([circle(r=0.02)], width=2.0, height=0.2)
    # Hollow: 2*pi*R*w*h, not pi*(R+w/2)^2*h.
    ring = 2.0 * math.pi * 20.0 * 2.0 * 0.2
    assert volume(comp) == pytest.approx(ring, rel=2e-3)
    assert volume(comp) < 0.2 * math.pi * 21.0**2
    assert not rep.dropped


def test_height_and_z_place_the_layer():
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    comp, _ = solids.bead_compound([line()], width=2.0, height=0.3, z=1.5)
    box = Bnd_Box()
    BRepBndLib.Add_s(comp, box)
    assert box.CornerMin().Z() == pytest.approx(1.5, abs=1e-6)
    assert box.CornerMax().Z() == pytest.approx(1.8, abs=1e-6)


def test_scale_converts_mesh_units_to_file_units():
    comp, _ = solids.bead_compound([line()], width=2.0, height=0.2, scale=1.0)
    # The same path read as metres is 1000x smaller, so a 2 mm bead swallows it
    # whole and the solid is just the end cap.
    assert volume(comp) < 2.0 * 0.2 * 50.0


def test_a_hairpin_tighter_than_the_bead_is_reported_not_hidden():
    # A 180 degree turn of 0.2 mm radius, resolved: a 2 mm bead cannot follow
    # that, and the report has to say so rather than quietly deform.
    t = onp.linspace(-onp.pi / 2, onp.pi / 2, 24)
    turn = onp.stack([0.0002 * onp.cos(t), 0.0002 * onp.sin(t)], axis=1)
    hair = onp.vstack([[-0.02, -0.0002], turn, [-0.02, 0.0002]])
    assert solids.turn_radius(hair * 1000.0).min() < 1.0
    _, rep = solids.bead_compound([hair], width=2.0, height=0.2)
    assert rep.tight == [0]


def test_empty_input_warns():
    with pytest.warns(UserWarning, match="no paths"):
        solids.bead_compound([], width=2.0, height=0.2)


def test_a_non_positive_section_is_refused():
    for w, h in ((0.0, 0.2), (2.0, -1.0)):
        with pytest.raises(ValueError, match="must be positive"):
            solids.bead_compound([line()], width=w, height=h)


# ── STEP ─────────────────────────────────────────────────────────────────────

def test_step_solid_round_trips_through_occ(tmp_path):
    from OCP.STEPControl import STEPControl_Reader

    f = tmp_path / "bead.step"
    rep = solids.write_step_solid([line(), circle()], f, width=2.0, height=0.2)
    assert rep.beads == 2
    assert f.stat().st_size > 0

    reader = STEPControl_Reader()
    reader.ReadFile(str(f))
    reader.TransferRoots()
    back = reader.OneShape()
    comp, _ = solids.bead_compound([line(), circle()], width=2.0, height=0.2)
    assert volume(back) == pytest.approx(volume(comp), rel=1e-6)


def test_a_path_whose_offset_fails_is_split_rather_than_lost():
    # Two legs 0.7 mm apart joined by a 0.35 mm hairpin -- the shape that made
    # OCC refuse a whole 620 mm path and drop its material from the file.
    t = onp.linspace(-onp.pi / 2, onp.pi / 2, 16)
    turn = onp.stack([0.00035 * onp.cos(t), 0.00035 * onp.sin(t)], axis=1)
    hair = onp.vstack([[-0.03, -0.00035], turn, [-0.03, 0.00035]])
    _, rep = solids.bead_compound([hair], width=2.0, height=0.2)
    assert not rep.dropped
    assert rep.beads == 1

    # The split is a union, not two overlapping copies: the volume must come
    # out *under* the swept figure, because the two legs of a 0.7 mm hairpin
    # lie inside one 2 mm bead and that material is counted once.  It must also
    # exceed one leg's worth, or something was lost rather than merged.
    comp, _ = solids.bead_compound([hair], width=2.0, height=0.2)
    length = onp.linalg.norm(onp.diff(hair, axis=0), axis=1).sum() * 1000.0
    swept = 2.0 * 0.2 * length + math.pi * 1.0**2 * 0.2
    assert 0.5 * swept < volume(comp) < swept


def test_step_solid_declares_millimetres(tmp_path):
    f = tmp_path / "bead.step"
    solids.write_step_solid([line()], f, width=2.0, height=0.2)
    text = f.read_text(errors="replace")
    assert "MILLI" in text and "METRE" in text


def test_report_prints_as_one_line():
    _, rep = solids.bead_compound([line()], width=2.0, height=0.2)
    assert str(rep) == f"1 beads, 1 solids, {rep.faces} faces"
    rep.dropped.append(7)
    assert str(rep).endswith("; 1 dropped")


def test_simplify_cuts_the_face_count_without_moving_the_bead():
    dense = [line(n=40)]
    thin = P.simplify(dense, tolerance=1e-9)
    a, ra = solids.bead_compound(dense, width=2.0, height=0.2)
    b, rb = solids.bead_compound(thin, width=2.0, height=0.2)
    assert rb.faces < ra.faces
    assert volume(b) == pytest.approx(volume(a), rel=1e-6)
