"""Minimal in-place vector arithmetic for NumPy and pyopencl arrays."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from pyopencl.elementwise import ElementwiseKernel

_AXPBY_KERNELS: dict[tuple[cl.Context, np.dtype], ElementwiseKernel] = {}


def order_of(array: Any) -> Literal["C", "F"]:
    """Return the memory order of a contiguous array."""
    return "C" if array.flags.c_contiguous else "F"


def zeros(shape: tuple[int, ...], like: Any) -> Any:
    """Return zeros of the given shape with dtype, order and queue of ``like``."""
    if isinstance(like, clarray.Array):
        assert like.queue is not None
        return clarray.zeros(
            like.queue,
            shape,
            dtype=like.dtype,
            order=order_of(like),
            allocator=like.allocator,
        )
    return np.zeros(shape, dtype=like.dtype, order=order_of(like))


def axpby(a: float, x: Any, b: float, y: Any) -> None:
    """Compute ``x = a * x + b * y`` in place."""
    if not isinstance(x, clarray.Array):
        x *= a
        x += b * y
        return

    if order_of(x) != order_of(y) or x.shape != y.shape:
        raise ValueError("axpby requires arrays of equal shape and memory order")
    dtype = x.dtype
    assert x.queue is not None
    kernel = _axpby_kernel(x.queue.context, dtype)
    event = kernel(
        dtype.type(a),
        x,
        dtype.type(b),
        y,
        wait_for=x.events + y.events,
    )
    x.add_event(event)


def _axpby_kernel(context: cl.Context, dtype: np.dtype) -> ElementwiseKernel:
    key = (context, np.dtype(dtype))
    kernel = _AXPBY_KERNELS.get(key)
    if kernel is None:
        ctype = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}[key[1]]
        kernel = ElementwiseKernel(
            context,
            f"{ctype} a, {ctype} *x, {ctype} b, const {ctype} *y",
            "x[i] = a * x[i] + b * y[i]",
            name="gratopy_axpby",
        )
        _AXPBY_KERNELS[key] = kernel
    return kernel
