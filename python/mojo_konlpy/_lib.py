"""ctypes bridge for the Mojo morphology kernel."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(os.environ.get("MOJO_KONLPY_LIB", ROOT / "dist" / "libmojo-konlpy.so"))
I = ctypes.c_int64

_library: ctypes.CDLL | None = None


def library() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not LIB_PATH.exists():
            raise RuntimeError(f"Mojo library not found at {LIB_PATH}; run `pixi run build`")
        _library = ctypes.CDLL(str(LIB_PATH))
        _library.mkl_analyze.argtypes = [I] * 15
        _library.mkl_analyze.restype = I
    return _library


def address(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.ndim != 1 or not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be one-dimensional and C-contiguous")
    if array.size and not array.ctypes.data:
        raise ValueError("FFI buffer has a null data pointer")
    return int(array.ctypes.data)
