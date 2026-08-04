"""Boundary-location predicates for rectangular domains.

feax identifies boundaries by a predicate ``point -> bool`` handed to
``location_fns`` / ``DirichletBCSpec``.  For an arbitrary domain you write your
own; these cover the rectangle that most benchmark problems live on.

Nothing here is specific to a plane or shell problem — both modules use them.
"""
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as np

__all__ = ["edge_predicate", "cantilever_edges", "EDGES"]


EDGES = ("left", "right", "bottom", "top")

_OPPOSITE = {"right": "left", "left": "right", "top": "bottom", "bottom": "top"}


def edge_predicate(edge: str, Lx: float, Ly: float, tol: float = 1e-6) -> Callable:
    """A predicate selecting one edge of the rectangle ``[0,Lx] × [0,Ly]``.

    Parameters
    ----------
    edge : {"left", "right", "bottom", "top"}
        ``left``/``right`` are ``x = 0`` / ``x = Lx``; ``bottom``/``top`` are
        ``y = 0`` / ``y = Ly``.
    tol : float
        Absolute coordinate tolerance.  Make it a fraction of the element size,
        not of the domain, or neighbouring nodes get swept in.
    """
    if edge not in _OPPOSITE:
        raise ValueError(f"unknown edge {edge!r}; use one of {EDGES}")
    axis, target = ((0, 0.0) if edge == "left" else
                    (0, Lx) if edge == "right" else
                    (1, 0.0) if edge == "bottom" else (1, Ly))

    def on_edge(point):
        return np.isclose(point[axis], target, atol=tol)

    return on_edge


def cantilever_edges(clamp: str, Lx: float, Ly: float, tol: float = 1e-6):
    """``(clamped_edge_fn, free_edge_fn)`` for a cantilever on a rectangle.

    ``clamp`` names the supported edge; the free (typically loaded) edge is the
    one opposite it.
    """
    return (edge_predicate(clamp, Lx, Ly, tol),
            edge_predicate(_OPPOSITE[clamp], Lx, Ly, tol))
