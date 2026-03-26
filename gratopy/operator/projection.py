"""Gratopy projection operators.

This module contains concrete implementations of projection operators,
most notably :class:`.Radon` and :class:`.Fanbeam`.

Examples
--------

>>> import gratopy
>>> Nx = 300
>>> phantom = gratopy.easy_phantom(N=Nx)
>>> R = gratopy.operator.Radon(image_domain=Nx, angles=180)
>>> sinogram = R.apply_to(phantom, queue=queue)
>>> backprojection = R.T.apply_to(sinogram)

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyopencl as cl
import pyopencl.array as clarray

from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gratopy.gratopy import radon_struct
from gratopy.operator.base import Operator
from gratopy.operator.opencl import OpenCLKernelSpec, _OpenCLOperator
from gratopy.utilities import Angles, Detectors, ExtentPlaceholder, ImageDomain


class Radon(_OpenCLOperator):
    """A Radon transform operator."""

    def __init__(
        self,
        image_domain: int | tuple[int, int] | ImageDomain,
        angles: Angles | int,
        detectors: Detectors | int | None = None,
        adjoint: bool = False,
        kernel_spec: OpenCLKernelSpec | None = None,
    ):
        if not isinstance(image_domain, ImageDomain):
            image_domain = ImageDomain(size=image_domain, extent=2.0)

        if not isinstance(angles, Angles):
            angles = Angles.uniform(number=angles)

        if not isinstance(detectors, Detectors):
            if detectors is None:
                detectors = int(np.ceil(np.hypot(*image_domain.size)))
            detectors = Detectors(number=detectors)

        state = {
            "image_domain": image_domain,
            "angles": angles,
            "detectors": detectors,
            "adjoint": adjoint,
        }
        super().__init__(name="Radon", state=state, kernel_spec=kernel_spec)

        self.substitute_placeholder()
        self.projection_settings: SimpleNamespace | None = None
        self._host_struct: dict[str, Any] | None = None
        self._device_struct: dict[tuple[cl.Context, np.dtype], dict[str, Any]] = {}

        image_shape = self.image_domain.size
        sinogram_shape = (self.detectors.number, len(self.angles))

        if self.adjoint:
            self.input_shape = sinogram_shape
            self.output_shape = image_shape
        else:
            self.input_shape = image_shape
            self.output_shape = sinogram_shape

    def _default_kernel_spec(self) -> OpenCLKernelSpec:
        kernel_path = Path(__file__).resolve().parent.parent / "radon.cl"
        return OpenCLKernelSpec.from_path(kernel_path, base_name="radon")

    @property
    def image_domain(self) -> ImageDomain:
        return self.state["image_domain"]

    @property
    def angles(self) -> Angles:
        return self.state["angles"]

    @property
    def detectors(self) -> Detectors:
        return self.state["detectors"]

    @property
    def adjoint(self) -> bool:
        return self.state["adjoint"]

    @property
    def T(self) -> "Radon":
        operator_copy = copy(self)
        operator_copy.state = copy(self.state)
        operator_copy.state["adjoint"] = not self.state["adjoint"]
        operator_copy.input_shape, operator_copy.output_shape = (
            operator_copy.output_shape,
            operator_copy.input_shape,
        )
        return operator_copy

    def substitute_placeholder(self) -> None:
        """Substitute any placeholder values in the operator settings."""
        if all(
            [
                isinstance(self.image_domain.extent, ExtentPlaceholder),
                isinstance(self.detectors.extent, ExtentPlaceholder),
            ]
        ):
            raise ValueError(
                "At least one of image_domain.extent or detectors.extent must be specified."
            )
        elif self.detectors.extent == ExtentPlaceholder.FULL and not isinstance(
            self.image_domain.extent, ExtentPlaceholder
        ):
            self.detectors.extent = self.image_domain.extent
        elif self.image_domain.extent == ExtentPlaceholder.FULL and not isinstance(
            self.detectors.extent, ExtentPlaceholder
        ):
            self.image_domain.extent = self.detectors.extent

    def _ensure_host_struct(self, queue: cl.CommandQueue) -> None:
        if self._host_struct is not None:
            return

        self._host_struct = radon_struct(
            queue=queue,
            img_shape=self.image_domain.size,
            angles=self.angles.angles,
            angle_weights=self.angles.weights,
            n_detectors=self.detectors.number,
            detector_width=float(self.detectors.extent),
            image_width=float(self.image_domain.extent),
            midpoint_shift=self.image_domain.center,
            detector_shift=self.detectors.center,
        )

    def _ensure_device_struct(
        self,
        queue: cl.CommandQueue,
        dtype: npt.DTypeLike,
    ) -> dict[str, Any]:
        self._ensure_host_struct(queue)
        dtype = np.dtype(dtype)
        cache_key = (queue.context, dtype)
        if cache_key in self._device_struct:
            return self._device_struct[cache_key]

        assert self._host_struct is not None
        ofs = self._host_struct["ofs_dict"][dtype]
        geometry = self._host_struct["geo_dict"][dtype]
        angle_weights = self._host_struct["angle_diff_dict"][dtype]

        ofs_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
        geometry_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, geometry.nbytes)
        angle_weights_buf = cl.Buffer(
            queue.context,
            cl.mem_flags.READ_ONLY,
            angle_weights.nbytes,
        )

        cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
        cl.enqueue_copy(queue, geometry_buf, geometry.data).wait()
        cl.enqueue_copy(queue, angle_weights_buf, angle_weights.data).wait()

        device_struct = {
            "ofs": ofs_buf,
            "geometry": geometry_buf,
            "angle_weights": angle_weights_buf,
        }
        self._device_struct[cache_key] = device_struct
        return device_struct

    def _expected_output_shape(self, argument_shape: tuple[int, ...]) -> tuple[int, ...]:
        z_dimension = ()
        if len(argument_shape) > 2:
            z_dimension = (argument_shape[2],)

        if self.adjoint:
            return self.image_domain.size + z_dimension
        return (self.detectors.number, len(self.angles)) + z_dimension

    def _validate_argument(self, argument: clarray.Array) -> None:
        assert argument.shape[0:2] == self.input_shape, (
            f"Input shape mismatch: expected {self.input_shape}, got {argument.shape}"
        )

    def _validate_output(
        self,
        output: clarray.Array,
        argument: clarray.Array,
    ) -> clarray.Array:
        output = output.with_queue(argument.queue)
        expected_shape = self._expected_output_shape(argument.shape)
        assert output.dtype == argument.dtype, (
            f"Output dtype mismatch: expected {argument.dtype}, got {output.dtype}"
        )
        assert output.shape == expected_shape, (
            f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
        )
        return output

    def apply_to(
        self,
        argument: npt.ArrayLike | clarray.Array,
        output: clarray.Array | None = None,
        queue: cl.CommandQueue | None = None,
        return_event: bool = False,
    ) -> clarray.Array | tuple[clarray.Array, list[cl.Event]]:
        queue = self._infer_queue(argument=argument, output=output, queue=queue)
        self.projection_settings = SimpleNamespace(queue=queue)
        argument = self._coerce_argument(argument, queue)
        self._validate_argument(argument)

        if output is None:
            output = self._allocate_output(
                queue=queue,
                shape=self._expected_output_shape(argument.shape),
                dtype=argument.dtype,
                order=self._default_order(argument),
                allocator=argument.allocator,
            )
        output = self._validate_output(output, argument)

        device_struct = self._ensure_device_struct(queue, argument.dtype)
        output_order = self._default_order(output)
        input_order = self._default_order(argument)
        kernel = self._get_projection_kernel(
            queue.context,
            dtype=argument.dtype,
            output_order=output_order,
            input_order=input_order,
            adjoint=self.adjoint,
        )

        if self.adjoint:
            cl_event = self._invoke_kernel(
                kernel,
                queue,
                output.shape,
                output.data,
                argument.data,
                device_struct["ofs"],
                device_struct["geometry"],
                wait_for=output.events + argument.events,
            )
        else:
            cl_event = self._invoke_kernel(
                kernel,
                queue,
                output.shape,
                output.data,
                argument.data,
                device_struct["ofs"],
                device_struct["geometry"],
                wait_for=output.events + argument.events,
            )

        output.add_event(cl_event)
        output = self.scalar * output

        if return_event:
            return output, [cl_event]
        return output


class Fanbeam(Operator):
    def __init__(
        self,
        source_distances: float | tuple[float, float],
        image_domain: int | tuple[int, int] | ImageDomain,
        angles: Angles | int,
        detectors: Detectors | int | None = None,
        adjoint: bool = False,
    ):
        super().__init__(name="Fanbeam")


# R, R_E parameters for fanbeam tranform: pass as _one_ argument, tuple
# or new dataclass. source_detector_distance, source_origin_distance
# if only one value is given, it is R, and R_E is R/2.
