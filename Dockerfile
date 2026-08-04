FROM nvcr.io/nvidia/jax:26.05-py3

RUN apt update
RUN apt upgrade -y

RUN apt -y install gmsh python3-gmsh libsuitesparse-dev swig libnlopt-dev
RUN pip install --upgrade pip
# nlopt's PyPI sdist pins cmake_minimum_required(VERSION <3.5); CMake 4.x dropped
# compat for policy versions <3.5 and hard-errors. CMAKE_POLICY_VERSION_MINIMUM=3.5
# tells CMake to configure anyway.
RUN CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-build-isolation nlopt

# Copy feax source code
COPY . /workspace
WORKDIR /workspace

# Install feax with CUDA dependencies.
# JAX is intentionally not reinstalled here: the NVCR base image already ships
# JAX compiled for CUDA 13.1.1. Use pip install .[cuda13,jax] outside containers.
RUN pip install .[cuda13,sksparse]

# Install spineax WITHOUT touching the base JAX (--no-deps: spineax and its
# lineax dep both declare jax[cuda13], which would clobber the NVCR base JAX).
# Build from the Naruki-Ichihara fork: the int64 CSR-index + hybrid-memory
# features feax uses live on its feat/int64-csr-offsets branch (not yet in the
# PyPI wheel / upstream johnviljoen). No prebuilt wheel exists for it, so every
# arch builds it from source against the container's own jaxlib (matching the
# cuDSS FFI ABI). cuDSS 0.8 (libcudss.so.0) and lineax are installed here;
# jaxtyping / equinox / cusparse come from the feax install above.
# TODO: point at the fork's main once the branch merges, or the PyPI wheel once
# the features are upstreamed.
RUN pip install nvidia-cudss-cu13 && \
    pip install --no-deps lineax && \
    pip install --no-build-isolation "scikit-build-core>=0.5" nanobind && \
    pip install --no-build-isolation --no-deps \
        "git+https://github.com/Naruki-Ichihara/spineax.git@feat/int64-csr-offsets"

# Algebraic-multigrid solver (feax[amg]): smoothed-aggregation AMG via PyAMG,
# run on JAX/GPU through AMJax, as a preconditioner for a matrix-free outer
# Krylov solve. Pure-Python (pyamg pulled in by amjax) — no native AmgX build.
RUN pip install --no-deps amjax pyamg

# Optional: Node.js 20 + pydoc-markdown + Docusaurus dependencies
# Build with: docker build --build-arg INSTALL_DOCS=true .
RUN if [ "$INSTALL_DOCS" = "true" ]; then \
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
        apt-get install -y nodejs && \
        pip install pydoc-markdown && \
        cd /workspace/docs && npm install; \
    fi

CMD ["/bin/bash"]
