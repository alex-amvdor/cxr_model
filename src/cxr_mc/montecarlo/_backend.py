"""
montecarlo._backend

Array backend selection (GPU via CuPy, else CPU NumPy) shared by every
montecarlo submodule. Importing this module performs the one-time device
probe and prints the GPU/CPU banner; ``xp`` is the active array module
(``cupy`` or ``numpy``), ``cp`` is CuPy or ``None`` on a CPU box, ``REAL`` is
the on-device spectrum precision, and ``_to_cpu`` moves arrays back to NumPy.
"""

import os
import warnings

import numpy as np

try:
    # cupy-cuda* imports cleanly even with no usable CUDA runtime (e.g. on the
    # viz laptop, which has the wheel but no GPU/driver), but on such a machine it
    # emits a UserWarning at import ("CUDA path could not be detected ..."). We
    # fall back to CPU below anyway, so silence just that import-time warning to
    # keep the notebook output clean. Importing is NOT proof the GPU path works --
    # a later cp.float32 / kernel call would blow up with AttributeError or a
    # CUDARuntimeError. Probe for a real device and fall back to CPU on ANY
    # failure, so the laptop runs the notebook on numpy.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")
        import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device")
    _GPU = True
    xp = cp
    print("Using GPU")
except Exception:
    _GPU = False
    cp = None  # so submodules can `from ._backend import cp` on a CPU box
    xp = np
    print("No GPU found, or cupy not installed!\nFalling back to CPU execution.")

# On-GPU spectrum precision. Consumer GPUs run fp64 at 1/32-1/64 of their fp32
# rate, so the big sinc/brem matmuls dominate -- single precision ~halves their
# cost and device-memory traffic. The one cancellation-sensitive spot, the
# (E_grid - E_res) subtraction, stays >3 orders below the line width, so fp32 is
# safe here; the complex couplings fall to complex64 automatically. Set
# CXR_FP64=1 to force double precision (reference/validation runs). The CPU
# fallback, where fp32 buys no speed, always stays double.
REAL = xp.float32 if (_GPU and os.environ.get("CXR_FP64") != "1") else xp.float64


def _to_cpu(a):
    """Move array to CPU (numpy). No-op if already numpy."""
    if cp is not None and isinstance(a, cp.ndarray):
        return a.get()
    return np.asarray(a)
