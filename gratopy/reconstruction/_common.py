"""Input handling shared by the reconstruction functions."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Callable

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from gratopy.operator.base import Operator
from gratopy.operator.space import Space
from gratopy.reconstruction._vector import order_of, zeros

Callback = Callable[[int, Any], "bool | None"]

_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def prepare_data(
    operator: Operator,
    data: Any,
    *,
    queue: cl.CommandQueue | None,
    x0: Any | None = None,
) -> Any:
    """Validate ``data`` and move it to the device if a queue is available.

    The queue is taken from ``queue``, ``data`` or ``x0``, in this order. Without
    any queue, ``data`` stays a NumPy array and the operator has to handle it.
    """
    if operator.input_shape is None or operator.output_shape is None:
        raise ValueError(
            "Reconstruction requires an operator with known input_shape and output_shape."
        )

    data = _coerce_data(data, _infer_queue(queue, data, x0))
    if data.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"data must be float32 or float64, got {data.dtype}")

    output_shape = tuple(operator.output_shape)
    if tuple(data.shape[: len(output_shape)]) != output_shape:
        raise ValueError(
            f"Data shape mismatch: expected {output_shape} (plus trailing "
            f"slice dimensions), got {data.shape}"
        )
    return data


def _infer_queue(queue: cl.CommandQueue | None, *arrays: Any) -> cl.CommandQueue | None:
    if queue is not None:
        return queue
    for array in arrays:
        if isinstance(array, clarray.Array) and array.queue is not None:
            return array.queue
    return None


def _coerce_data(data: Any, queue: cl.CommandQueue | None) -> Any:
    """Return contiguous data, on the device if a queue is given."""
    if isinstance(data, clarray.Array):
        if not (data.flags.c_contiguous or data.flags.f_contiguous):
            raise ValueError("data must be a contiguous array")
        return data if queue is None else data.with_queue(queue)

    data = np.asarray(data)
    if not (data.flags.c_contiguous or data.flags.f_contiguous):
        data = np.ascontiguousarray(data)
    return data if queue is None else clarray.to_device(queue, data)


def solution_shape(operator: Operator, data: Any) -> tuple[int, ...]:
    """Return the shape of the reconstruction belonging to ``data``."""
    assert operator.input_shape is not None and operator.output_shape is not None
    extra_dims = tuple(data.shape[len(operator.output_shape) :])
    return tuple(operator.input_shape) + extra_dims


def initial_guess(operator: Operator, data: Any, x0: Any | None) -> Any:
    """Return a fresh iterate: zeros, or a copy of ``x0`` matching ``data``."""
    shape = solution_shape(operator, data)
    if x0 is None:
        return zeros(shape, like=data)

    if tuple(x0.shape) != shape:
        raise ValueError(f"x0 shape mismatch: expected {shape}, got {x0.shape}")

    order = order_of(data)
    if isinstance(data, clarray.Array):
        if isinstance(x0, clarray.Array):
            if x0.dtype != data.dtype:
                raise ValueError(
                    f"x0 dtype {x0.dtype} does not match the data dtype {data.dtype}"
                )
            if order_of(x0) != order:
                raise ValueError(
                    "x0 and data must have the same memory order "
                    f"(data is {order}-contiguous)"
                )
            return x0.with_queue(data.queue).copy()
        assert data.queue is not None
        return clarray.to_device(data.queue, np.require(x0, data.dtype, order))

    if isinstance(x0, clarray.Array):
        raise TypeError("x0 is a device array, but data is a NumPy array")
    return np.array(x0, dtype=data.dtype, order=order)


def space_or_euclidean(space: Space | None, shape: tuple[int, ...] | None) -> Space:
    """Return ``space``, falling back to the Euclidean space of ``shape``."""
    if space is not None:
        return space
    assert shape is not None
    return Space(shape)


def check_iterations(number_iterations: Any) -> int:
    if (
        not isinstance(number_iterations, Integral)
        or isinstance(number_iterations, bool)
        or number_iterations < 0
    ):
        raise ValueError("number_iterations must be a non-negative integer")
    return int(number_iterations)


def check_positive(value: Any, name: str) -> float:
    if not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def resolve_operator_norm(
    operator: Operator,
    operator_norm: float | None,
    data: Any,
) -> float:
    """Return the given operator norm or estimate it once."""
    if operator_norm is None:
        queue = data.queue if isinstance(data, clarray.Array) else None
        operator_norm = operator.norm_estimate(dtype=data.dtype, queue=queue)
    return check_positive(operator_norm, "operator_norm")


def notify(callback: Callback | None, iteration: int, x: Any) -> bool:
    """Call the callback and return whether it requested to stop."""
    return callback is not None and bool(callback(iteration, x))
