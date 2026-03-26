"""Gratopy projection operators.

This module contains concrete operator implementations for gratopy's
experimental operator interface, most notably :class:`.Radon`.

The focus is currently on a compositional, operator-style interface for Radon
transforms and their adjoints. Fanbeam support is not yet implemented at the
same level.

Examples
--------

>>> import numpy as np
>>> import pyopencl as cl
>>> import gratopy
>>> ctx = cl.create_some_context(interactive=False)
>>> queue = cl.CommandQueue(ctx)
>>> Nx = 300
>>> img = np.zeros((Nx, Nx), dtype=np.float32)
>>> R = gratopy.operator.Radon(image_domain=Nx, angles=180)
>>> sinogram = R.apply_to(img, queue=queue)
>>> backprojection = R.T.apply_to(sinogram)

The same forward and adjoint applications can also be written via operator
syntax:

>>> sinogram = R * img
>>> backprojection = R.T * sinogram
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
    """Parallel-beam Radon transform operator.

    This class provides the main entry point to gratopy's experimental
    operator-based projection interface. A :class:`Radon` object represents
    either the forward projection operator or, when created through
    :attr:`T`, its adjoint.

    **Parameters**

    ``image_domain``:
        Description of the image grid and its physical extent. This can be
        given as:

        - an ``int`` for a square image domain of that size,
        - a ``(Nx, Ny)`` tuple for a rectangular grid,
        - or an explicit :class:`gratopy.utilities.ImageDomain` instance.

        Plain integer and tuple inputs are converted to
        :class:`~gratopy.utilities.ImageDomain` with a default extent of
        ``2.0``.
    ``angles``:
        Angular sampling of the operator. This can be given either as an
        integer or as an explicit :class:`gratopy.utilities.Angles` object.

        If an integer is given, uniformly weighted angles are created via
        :meth:`gratopy.utilities.Angles.uniform`.
    ``detectors``:
        Detector configuration. This can be given as:

        - ``None`` to use the default detector count
          ``ceil(hypot(Nx, Ny))``,
        - an integer specifying the number of detector pixels,
        - or an explicit :class:`gratopy.utilities.Detectors` object.

        Plain integer inputs are converted to
        :class:`~gratopy.utilities.Detectors` with default extent handling.
    ``adjoint``:
        If ``False`` (the default), construct the forward Radon transform.
        If ``True``, construct the adjoint operator directly. In practice, the
        adjoint is typically accessed via :attr:`T`.
    ``kernel_spec``:
        Optional :class:`gratopy.operator.opencl.OpenCLKernelSpec` describing
        which OpenCL kernel bundle to use. When omitted, the operator uses the
        Radon kernels shipped with gratopy.

    **Notes**

    The operator accepts both :class:`pyopencl.array.Array` inputs and NumPy
    arrays. When applying the operator to a NumPy array, a queue must be
    available so that the data can be transferred to the device.

    The operator supports 2D as well as slicewise 3D data. For example,
    a forward operator with image shape ``(Nx, Ny)`` maps:

    - ``(Nx, Ny)`` to ``(Ns, Na)``,
    - ``(Nx, Ny, Nz)`` to ``(Ns, Na, Nz)``.

    Correspondingly, the adjoint maps:

    - ``(Ns, Na)`` to ``(Nx, Ny)``,
    - ``(Ns, Na, Nz)`` to ``(Nx, Ny, Nz)``.

    The placeholder mechanism for extents in the experimental operator API is
    not yet fully implemented. In particular, ``ExtentPlaceholder.FULL``
    currently only supports limited inherited behavior, while
    ``ExtentPlaceholder.VALID`` should be considered unsupported for now.

    **Examples**

    >>> import numpy as np
    >>> import pyopencl as cl
    >>> import gratopy
    >>> ctx = cl.create_some_context(interactive=False)
    >>> queue = cl.CommandQueue(ctx)
    >>> img = np.zeros((128, 128), dtype=np.float32)
    >>> R = gratopy.operator.Radon(image_domain=128, angles=180)
    >>> sino = R.apply_to(img, queue=queue)
    >>> backproj = R.T.apply_to(sino)

    Operator algebra is also supported:

    >>> G = R.T * R
    >>> gram_img = G.apply_to(img, queue=queue)
    """

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

    def _kernel_arguments(
        self,
        output: clarray.Array,
        argument: clarray.Array,
        queue: cl.CommandQueue,
    ) -> tuple[Any, ...]:
        device_struct = self._ensure_device_struct(queue, argument.dtype)
        return device_struct["ofs"], device_struct["geometry"]

    def apply_to(
        self,
        argument: npt.ArrayLike | clarray.Array,
        output: clarray.Array | None = None,
        queue: cl.CommandQueue | None = None,
        return_event: bool = False,
    ) -> clarray.Array | tuple[clarray.Array, list[cl.Event]]:
        queue = self._infer_queue(argument=argument, output=output, queue=queue)
        self.projection_settings = SimpleNamespace(queue=queue)
        return super().apply_to(
            argument,
            output=output,
            queue=queue,
            return_event=return_event,
        )


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
