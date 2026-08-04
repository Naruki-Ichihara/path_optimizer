"""path_optimizer: GPU-accelerated path optimization built on feax and JAX."""

# optimizer first: plane/shell import the design-space types from it.
from path_optimizer import (
    geometry,
    materials,
    objectives,
    paths,
    plane,
    shell,
    stripes,
)
from path_optimizer.geometry import cantilever_edges, edge_predicate
from path_optimizer.materials import Lamina, Polymer
from path_optimizer.optimizer import (
    BoundDesignSpace,
    Continuation,
    DesignField,
    DesignSpace,
    OptimizeResult,
    Pipeline,
    constraint,
    make_linear_solver,
    run,
)
from path_optimizer.plane import PlaneStress, make_plane_stress
from path_optimizer.shell import LaminatedShell, make_laminated_shell

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # submodules
    "geometry",
    "materials",
    "objectives",
    "paths",
    "plane",
    "shell",
    "stripes",
    # design space
    "DesignField",
    "DesignSpace",
    "BoundDesignSpace",
    # pipeline
    "Pipeline",
    "constraint",
    "Continuation",
    # driver
    "run",
    "OptimizeResult",
    "make_linear_solver",
    # materials
    "Lamina",
    "Polymer",
    # problems
    "PlaneStress",
    "make_plane_stress",
    "LaminatedShell",
    "make_laminated_shell",
    # geometry
    "edge_predicate",
    "cantilever_edges",
]
