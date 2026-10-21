"""Variational reconstruction: total-variation regularization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from gratopy.operator.base import Operator
from gratopy.operator.opencl import _expand_gratopy_template
from gratopy.reconstruction._common import (
    Callback,
    check_iterations,
    check_positive,
    notify,
    prepare_data,
    resolve_operator_norm,
    solution_shape,
)
from gratopy.reconstruction._vector import axpby, order_of, zeros

_KERNEL_PATH = Path(__file__).resolve().parent.parent / "total_variation.cl"
_PROGRAMS: dict[cl.Context, cl.Program] = {}

# Squared norm bound of the 3D gradient (12) plus the normalized operator (1).
_STACKED_NORM_SQ = 13


def total_variation(
    operator: Operator,
    data: Any,
    mu: float,
    number_iterations: int = 1000,
    *,
    slice_thickness: float = 1.0,
    stepsize_weighting: float = 10.0,
    operator_norm: float | None = None,
    queue: cl.CommandQueue | None = None,
    callback: Callback | None = None,
) -> Any:
    """Total-variation regularized reconstruction via a primal-dual algorithm.

    Approximately solves

    .. math::

        \\min_{u \\geq 0} \\frac{\\mu}{2}\\|Au - f\\|^2 + \\mathrm{TV}(u)

    with the primal-dual algorithm of [CP2011]_, as
    :func:`gratopy.total_variation`. As there, the reconstruction is
    constrained to be nonnegative.

    The pixel size :math:`\\delta_x` entering the total variation is taken from
    ``operator.domain`` (an image space with pixel area :math:`\\delta_x^2`);
    an unspecified domain means :math:`\\delta_x = 1`.

    :param operator: The forward operator :math:`A` with a two-dimensional
        input shape.
    :param data: The data :math:`f` as a device array, or a NumPy array
        together with ``queue``. A trailing third dimension holds slices.
    :param mu: Positive regularization parameter; smaller values regularize
        more strongly.
    :param number_iterations: Number of iterations.
    :param slice_thickness: For slice stacks, the ratio of slice spacing to
        pixel size used to couple the slices; 0 reconstructs slices
        independently. Ignored for two-dimensional data.
    :param stepsize_weighting: Ratio of the dual and primal step sizes
        :math:`\\sigma` and :math:`\\tau` (with :math:`\\sigma\\tau` fixed).
    :param operator_norm: The norm :math:`\\|A\\|`; estimated once if
        :obj:`None`.
    :param queue: OpenCL queue; inferred from ``data`` if omitted.
    :param callback: Called as ``callback(iteration, u)`` after every
        iteration; returning :obj:`True` stops the iteration. ``u`` is owned by
        the solver; copy it to keep it.
    :return: The reconstruction as a device array.

    """
    number_iterations = check_iterations(number_iterations)
    mu = check_positive(mu, "mu")
    stepsize_weighting = check_positive(stepsize_weighting, "stepsize_weighting")
    if slice_thickness < 0:
        raise ValueError("slice_thickness must be non-negative")
    if operator.input_shape is None or len(operator.input_shape) != 2:
        raise ValueError(
            "total_variation requires an operator with a two-dimensional "
            f"input shape, got {operator.input_shape}"
        )

    data = prepare_data(operator, data, queue=queue)
    if not isinstance(data, clarray.Array):
        raise ValueError(
            "total_variation runs on the OpenCL device; pass device data or "
            "NumPy data together with queue=."
        )
    image_shape = solution_shape(operator, data)
    if len(image_shape) == 2:
        slice_thickness = 0.0
    elif len(image_shape) > 3:
        raise ValueError("total_variation supports at most one slice dimension")

    delta_x = _pixel_size(operator)
    norm = resolve_operator_norm(operator, operator_norm, data)

    assert data.queue is not None
    kernels = _Kernels(data.queue.context, data.dtype, order_of(data))
    adjoint = operator.T

    u = zeros(image_shape, like=data)
    u_extrapolated = zeros(image_shape, like=data)
    # Dual variable of the gradient, stored as OpenCL real3 (4 entries) per pixel.
    v = zeros((4,) + image_shape, like=data)
    dual_data = zeros(data.shape, like=data)
    projected = zeros(data.shape, like=data)
    backprojected = zeros(image_shape, like=data)

    sigma = stepsize_weighting / np.sqrt(_STACKED_NORM_SQ)
    tau = 1.0 / np.sqrt(_STACKED_NORM_SQ) / stepsize_weighting
    # The algorithm works with A / ||A|| and the gradient in pixel units;
    # rescale mu accordingly and precompute the resolvent factor.
    mu_scaled = mu * norm**2 * delta_x
    resolvent_factor = mu_scaled / (sigma + mu_scaled)

    for iteration in range(1, number_iterations + 1):
        # Dual updates
        kernels.update_v(v, u_extrapolated, sigma, slice_thickness)
        kernels.project_v(v)
        projected = operator.apply_to(u_extrapolated, output=projected)
        kernels.update_lambda(dual_data, projected, data, sigma / norm, resolvent_factor)

        # Primal update (nonnegative), written into u_extrapolated
        backprojected = adjoint.apply_to(dual_data, output=backprojected)
        kernels.update_u(
            u_extrapolated, u, v, backprojected, tau, 1.0 / norm, slice_thickness
        )

        # Extrapolation: u <- 2 u_new - u_old; then swap roles
        axpby(-1.0, u, 2.0, u_extrapolated)
        u, u_extrapolated = u_extrapolated, u

        if notify(callback, iteration, u):
            break
    return u


def _pixel_size(operator: Operator) -> float:
    domain = operator.domain
    if domain is None:
        return 1.0
    if domain.axis_weights is not None:
        raise ValueError(
            "total_variation requires an image domain with uniform pixel "
            f"weights, got {domain}"
        )
    return float(np.sqrt(domain.scale))


def _program(context: cl.Context) -> cl.Program:
    program = _PROGRAMS.get(context)
    if program is None:
        dtypes = ["float"]
        if any(device.double_fp_config for device in context.devices):
            dtypes.append("double")
        source = _expand_gratopy_template(
            _KERNEL_PATH.read_text(), dtypes=dtypes, two_orders=False
        )
        program = cl.Program(context, source).build()
        _PROGRAMS[context] = program
    return program


class _Kernels:
    """The fused primal-dual update kernels of ``total_variation.cl``."""

    def __init__(self, context: cl.Context, dtype: np.dtype, order: str) -> None:
        if np.dtype(dtype) == np.dtype(np.float64) and not any(
            device.double_fp_config for device in context.devices
        ):
            raise TypeError("The OpenCL device does not support double precision.")
        program = _program(context)
        suffix = f"{'float' if dtype == np.float32 else 'double'}_{order.lower()}"
        self._update_v = cl.Kernel(program, f"update_v_{suffix}")
        self._project_v = cl.Kernel(program, f"update_NormV_unchor_{suffix}")
        self._update_lambda = cl.Kernel(program, f"update_lambda_L2_{suffix}")
        self._update_u = cl.Kernel(program, f"update_u_{suffix}")

    @staticmethod
    def _run(kernel, shape, output, *arguments, inputs=()):
        event = kernel(
            output.queue,
            shape,
            None,
            output.data,
            *arguments,
            wait_for=output.events + [e for array in inputs for e in array.events],
        )
        output.add_event(event)

    def update_v(self, v, u, sigma, slice_thickness):
        self._run(
            self._update_v,
            u.shape,
            v,
            u.data,
            np.float32(sigma),
            np.float32(slice_thickness),
            inputs=(u,),
        )

    def project_v(self, v):
        # The second kernel argument is unused by the kernel.
        self._run(self._project_v, v.shape[1:], v, v.data)

    def update_lambda(self, dual, projected, data, sigma, factor):
        self._run(
            self._update_lambda,
            dual.shape,
            dual,
            projected.data,
            data.data,
            np.float32(sigma),
            np.float32(factor),
            inputs=(projected, data),
        )

    def update_u(self, u_new, u_old, v, backprojected, tau, scaling, slice_thickness):
        self._run(
            self._update_u,
            u_new.shape,
            u_new,
            u_old.data,
            v.data,
            backprojected.data,
            np.float32(tau),
            np.float32(scaling),
            np.float32(slice_thickness),
            inputs=(u_old, v, backprojected),
        )
