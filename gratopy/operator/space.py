"""Function spaces of operators.

A :class:`Space` describes the Hilbert space an operator acts on or maps into:
a discretized :math:`L^2` space with a diagonal weighting of its inner product.
Projection operators use weighted spaces (pixel area for images, detector
pixel width and angle weights for sinograms); their adjoints :attr:`T` are
adjoint with respect to exactly these inner products.

Arrays may carry trailing dimensions beyond :attr:`Space.shape` (e.g., a stack
of slices); these are summed over without additional weights, consistent with
the slicewise application of operators.
"""

from __future__ import annotations

from math import prod
from typing import Any

import numpy as np
import numpy.typing as npt
import pyopencl as cl
import pyopencl.array as clarray
from pyopencl.reduction import ReductionKernel


class Space:
    """Discretized :math:`L^2` space with a diagonal inner-product weighting.

    The inner product is

    .. math::

        \\langle x, y \\rangle = \\mathrm{scale} \\sum_i w_i x_i y_i,

    where :math:`w_i` is the entry of ``axis_weights`` belonging to the index
    of :math:`i` along the weighted axis (or 1 without axis weights).

    **Parameters**

    ``shape``:
        Base shape of the elements of the space.
    ``scale``:
        Positive constant factor of the inner product, e.g., the pixel area.
    ``axis_weights``:
        Optional pair ``(axis, weights)`` of an axis of ``shape`` and positive
        weights along it, e.g., the angle weights of a sinogram.

    **Example**

    >>> import numpy as np
    >>> from gratopy.operator import Space
    >>> X = Space((2, 3), scale=0.5, axis_weights=(1, [1.0, 2.0, 1.0]))
    >>> X.inner(np.ones((2, 3)), np.ones((2, 3)))
    4.0
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        scale: float = 1.0,
        axis_weights: tuple[int, npt.ArrayLike] | None = None,
    ) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be positive and finite")

        self.weight_axis: int | None = None
        self.axis_weights: np.ndarray | None = None
        if axis_weights is not None:
            axis, weights = axis_weights
            if not 0 <= axis < len(self.shape):
                raise ValueError(
                    f"weight axis {axis} out of range for shape {self.shape}"
                )
            weights = np.array(weights, dtype=np.float64)
            if weights.shape != (self.shape[axis],):
                raise ValueError(
                    f"axis weights must have shape ({self.shape[axis]},), "
                    f"got {weights.shape}"
                )
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("axis weights must be positive and finite")
            weights.flags.writeable = False
            self.weight_axis = int(axis)
            self.axis_weights = weights

        self._device_weights: dict[tuple[cl.Context, np.dtype], clarray.Array] = {}

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle/deepcopy state without device buffers."""
        state = self.__dict__.copy()
        state["_device_weights"] = {}
        return state

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Space):
            return NotImplemented
        if (self.shape, self.scale, self.weight_axis) != (
            other.shape,
            other.scale,
            other.weight_axis,
        ):
            return False
        if self.axis_weights is None or other.axis_weights is None:
            return self.axis_weights is other.axis_weights
        return bool(np.array_equal(self.axis_weights, other.axis_weights))

    def __hash__(self) -> int:
        return hash((self.shape, self.scale, self.weight_axis))

    def __repr__(self) -> str:
        weights = ""
        if self.weight_axis is not None:
            weights = f", axis_weights=(axis {self.weight_axis}, ...)"
        return f"Space(shape={self.shape}, scale={self.scale!r}{weights})"

    def inner(self, x: Any, y: Any) -> float:
        """Return the inner product of ``x`` and ``y``.

        ``x`` and ``y`` must both be NumPy arrays or both be
        :class:`pyopencl.array.Array` instances of the same shape (and, on the
        device, the same memory order).
        """
        self._validate(x, y)
        if isinstance(x, clarray.Array):
            return self.scale * self._device_inner(x, y)

        x = np.asarray(x)
        y = np.asarray(y)
        product = x * y
        if self.axis_weights is not None:
            assert self.weight_axis is not None
            broadcast_shape = [1] * x.ndim
            broadcast_shape[self.weight_axis] = len(self.axis_weights)
            product = product * self.axis_weights.reshape(broadcast_shape)
        return self.scale * float(np.real(product.sum()))

    def norm(self, x: Any) -> float:
        """Return the norm of ``x`` induced by :meth:`inner`."""
        return float(np.sqrt(max(self.inner(x, x), 0.0)))

    def _validate(self, x: Any, y: Any) -> None:
        if isinstance(x, clarray.Array) != isinstance(y, clarray.Array):
            raise TypeError(
                "Both arguments must be NumPy arrays or both pyopencl arrays."
            )
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} and {y.shape}")
        if tuple(x.shape[: len(self.shape)]) != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape} (plus trailing "
                f"dimensions), got {x.shape}"
            )

    def _device_inner(self, x: clarray.Array, y: clarray.Array) -> float:
        if x.dtype != y.dtype:
            raise ValueError(f"dtype mismatch: {x.dtype} and {y.dtype}")
        if x.flags.c_contiguous and y.flags.c_contiguous:
            order = "C"
        elif x.flags.f_contiguous and y.flags.f_contiguous:
            order = "F"
        else:
            raise ValueError("Both arrays must be contiguous with the same order.")

        if self.axis_weights is None:
            return float(np.real(clarray.vdot(x, y).get()))

        assert self.weight_axis is not None
        axis = self.weight_axis
        if order == "F":
            stride = prod(x.shape[:axis])
        else:
            stride = prod(x.shape[axis + 1 :])
        queue = x.queue
        assert queue is not None
        kernel = _weighted_dot_kernel(queue.context, x.dtype)
        weights = self._weights_on_device(queue, x.dtype)
        result = kernel(
            x,
            y,
            weights,
            np.int64(stride),
            np.int64(len(self.axis_weights)),
            queue=queue,
            return_event=False,
        )
        return float(result.get())

    def _weights_on_device(
        self, queue: cl.CommandQueue, dtype: np.dtype
    ) -> clarray.Array:
        assert self.axis_weights is not None
        key = (queue.context, np.dtype(dtype))
        weights = self._device_weights.get(key)
        if weights is None:
            weights = clarray.to_device(queue, self.axis_weights.astype(dtype))
            self._device_weights[key] = weights
        return weights


_WEIGHTED_DOT_KERNELS: dict[tuple[cl.Context, np.dtype], ReductionKernel] = {}


def _weighted_dot_kernel(context: cl.Context, dtype: np.dtype) -> ReductionKernel:
    """Return a reduction computing ``sum_i x[i] y[i] w[(i / stride) % count]``."""
    key = (context, np.dtype(dtype))
    kernel = _WEIGHTED_DOT_KERNELS.get(key)
    if kernel is None:
        ctype = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}[key[1]]
        kernel = ReductionKernel(
            context,
            dtype_out=key[1],
            neutral="0",
            reduce_expr="a+b",
            map_expr="x[i] * y[i] * w[(i / weight_stride) % weight_count]",
            arguments=(
                f"const {ctype} *x, const {ctype} *y, const {ctype} *w, "
                "const long weight_stride, const long weight_count"
            ),
            name="gratopy_weighted_dot",
        )
        _WEIGHTED_DOT_KERNELS[key] = kernel
    return kernel
