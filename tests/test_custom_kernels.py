from __future__ import annotations

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from copy import copy
from pathlib import Path

from gratopy.operator.opencl import OpenCLKernelSpec, _OpenCLOperator


TEST_AFFINE_KERNEL = Path(__file__).parent / "affine.cl"
TEST_AFFINE_ALT_KERNEL = Path(__file__).parent / "affine_alt.cl"


class AffineOperator(_OpenCLOperator):
    def __init__(
        self,
        shape: tuple[int, ...],
        adjoint: bool = False,
        kernel_spec: OpenCLKernelSpec | None = None,
    ):
        state = {"adjoint": adjoint}
        super().__init__(
            name="AffineOperator",
            state=state,
            input_shape=shape,
            output_shape=shape,
            kernel_spec=kernel_spec,
        )

    def _default_kernel_spec(self) -> OpenCLKernelSpec:
        return OpenCLKernelSpec.from_path(TEST_AFFINE_KERNEL, base_name="affine")

    @property
    def adjoint(self) -> bool:
        return self.state["adjoint"]

    @property
    def T(self) -> "AffineOperator":
        operator_copy = copy(self)
        operator_copy.state = copy(self.state)
        operator_copy.state["adjoint"] = not self.state["adjoint"]
        return operator_copy


def test_custom_kernel_spec_end_to_end():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    shape = (4, 5)
    spec = OpenCLKernelSpec.from_path(TEST_AFFINE_KERNEL, base_name="affine")
    operator = AffineOperator(shape=shape, kernel_spec=spec)

    expected = 2 * np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1

    for input_order in ["C", "F"]:
        for output_order in ["C", "F"]:
            host_input = np.require(
                np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
                requirements=input_order,
            )
            device_input = clarray.to_device(queue, host_input)
            device_output = clarray.zeros(
                queue, shape, dtype=np.float32, order=output_order
            )

            result = operator.apply_to(device_input, output=device_output)
            np.testing.assert_allclose(result.get(), expected)


def test_custom_kernel_spec_adjoint_kernel_is_used():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    shape = (4, 5)
    spec = OpenCLKernelSpec.from_path(TEST_AFFINE_KERNEL, base_name="affine")
    operator = AffineOperator(shape=shape, kernel_spec=spec)

    host_input = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = 3 * host_input - 1

    result = operator.T.apply_to(host_input, queue=queue)
    np.testing.assert_allclose(result.get(), expected)


def test_custom_kernel_spec_cache_isolation_by_source_signature():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    shape = (4, 5)
    host_input = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    spec1 = OpenCLKernelSpec.from_path(TEST_AFFINE_KERNEL, base_name="affine")
    spec2 = OpenCLKernelSpec.from_path(TEST_AFFINE_ALT_KERNEL, base_name="affine")

    op1 = AffineOperator(shape=shape, kernel_spec=spec1)
    op2 = AffineOperator(shape=shape, kernel_spec=spec2)

    expected1 = 2 * host_input + 1
    expected2 = 5 * host_input - 3

    result1_before = op1.apply_to(host_input, queue=queue).get()
    result2 = op2.apply_to(host_input, queue=queue).get()
    result1_after = op1.apply_to(host_input, queue=queue).get()

    np.testing.assert_allclose(result1_before, expected1)
    np.testing.assert_allclose(result2, expected2)
    np.testing.assert_allclose(result1_after, expected1)
