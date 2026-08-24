"""ctypes bindings for the Mojo distance-correlation kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_DCOR_LIB") or os.path.join(
    ROOT, "dist", "libmojo-dcor.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mdcor_stats": ([I, I, I, I, I, I, I, F, I], None),
    "mdcor_pairwise_distances": ([I, I, I, I, F], None),
    "mdcor_center": ([I, I, I, I, I], None),
    "mdcor_mean_product": ([I, I, I], F),
    "mdcor_u_product": ([I, I, I], F),
}

_library: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "dcor.mojo")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    process = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((process.stderr or process.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value: object) -> np.ndarray:
    source = np.asarray(value)
    if not (
        np.issubdtype(source.dtype, np.number)
        or np.issubdtype(source.dtype, np.bool_)
    ):
        raise TypeError("input must contain real numeric values")
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex-valued arrays are not supported")
    if source.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError(
            f"{source.dtype} arrays cannot be converted to float64 without "
            "silent precision loss"
        )
    if np.issubdtype(source.dtype, np.integer) and source.size:
        exact_limit = 2**53
        if np.any(source > exact_limit) or np.any(source < -exact_limit):
            raise ValueError(
                "integer values outside [-2**53, 2**53] cannot be represented "
                "exactly as float64"
            )
    try:
        return np.ascontiguousarray(source, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("input must contain real numeric values") from error


def address(array: np.ndarray, *, writable: bool = False) -> int:
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    if writable and not array.flags.writeable:
        raise ValueError("FFI output buffers must be writable")
    result = int(array.ctypes.data)
    if array.size and result == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return result
