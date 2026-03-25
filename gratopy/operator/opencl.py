"""OpenCL helpers for gratopy operators.

This module contains small building blocks for OpenCL-backed operators:

- :class:`OpenCLKernelSpec` describes where kernels are loaded from.
- :class:`_OpenCLOperator` provides shared execution helpers.

The classes in this module are intentionally lightweight. They are meant to
support concrete operators such as :class:`gratopy.operator.Radon` without
pulling OpenCL-specific concerns into :class:`gratopy.operator.base.Operator`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyopencl as cl
import pyopencl.array as clarray

from gratopy.operator.base import Operator


@dataclass(frozen=True)
class OpenCLKernelSpec:
    """Description of an OpenCL kernel bundle.

    Parameters are intentionally minimal for now:

    - ``paths`` point to one or more kernel source files,
    - ``base_name`` is used for kernel name construction,
    - ``build_options`` are forwarded to ``cl.Program.build``.
    """

    paths: tuple[str, ...]
    base_name: str
    build_options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_paths = tuple(str(Path(path).expanduser().resolve()) for path in self.paths)
        object.__setattr__(self, "paths", normalized_paths)
        object.__setattr__(self, "build_options", tuple(self.build_options))

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        base_name: str,
        build_options: tuple[str, ...] = (),
    ) -> "OpenCLKernelSpec":
        """Create a kernel spec from a single source file."""
        return cls(paths=(str(path),), base_name=base_name, build_options=build_options)

    @classmethod
    def from_paths(
        cls,
        paths: list[str | Path] | tuple[str | Path, ...],
        base_name: str,
        build_options: tuple[str, ...] = (),
    ) -> "OpenCLKernelSpec":
        """Create a kernel spec from multiple source files."""
        return cls(
            paths=tuple(str(path) for path in paths),
            base_name=base_name,
            build_options=build_options,
        )

    def read_sources(self) -> tuple[str, ...]:
        """Return the source code of all configured kernel files."""
        return tuple(Path(path).read_text() for path in self.paths)

    @property
    def signature(self) -> str:
        """Return a content-based signature used for program cacheing."""
        digest = hashlib.sha256()
        digest.update(self.base_name.encode())
        for option in self.build_options:
            digest.update(b"\0")
            digest.update(option.encode())
        for path, source in zip(self.paths, self.read_sources()):
            digest.update(b"\0")
            digest.update(path.encode())
            digest.update(b"\0")
            digest.update(source.encode())
        return digest.hexdigest()


class _OpenCLOperator(Operator):
    """Internal base class for OpenCL-backed operators.

    This class intentionally only implements shared execution helpers.
    Concrete subclasses are still responsible for geometry handling and for
    deciding which kernels to load.
    """

    _PROGRAM_CACHE: dict[tuple[cl.Context, str], dict[str, cl.Kernel]] = {}

    def __init__(
        self,
        *,
        name: str | None = None,
        kernel_spec: OpenCLKernelSpec | None = None,
        **operator_kwargs: Any,
    ):
        super().__init__(name=name, **operator_kwargs)
        self.kernel_spec = kernel_spec or self._default_kernel_spec()
        self._last_queue: cl.CommandQueue | None = None

    def _default_kernel_spec(self) -> OpenCLKernelSpec:
        """Return the default kernel spec for this operator.

        Concrete subclasses are expected to override this if they want to rely
        on `_OpenCLOperator` construction directly.
        """
        raise NotImplementedError("Concrete OpenCL operators must define a kernel spec")

    def _infer_queue(
        self,
        argument: npt.ArrayLike | clarray.Array | None = None,
        output: clarray.Array | None = None,
        queue: cl.CommandQueue | None = None,
    ) -> cl.CommandQueue:
        """Infer the queue to use for computation."""
        if queue is not None:
            self._last_queue = queue
            return queue

        if isinstance(argument, clarray.Array) and argument.queue is not None:
            self._last_queue = argument.queue
            return argument.queue

        if isinstance(output, clarray.Array) and output.queue is not None:
            self._last_queue = output.queue
            return output.queue

        if self._last_queue is not None:
            return self._last_queue

        raise ValueError(
            "No OpenCL queue available. Either pass an explicit queue, provide "
            "a clarray.Array as input, or provide an output array with an "
            "associated queue."
        )

    def _coerce_argument(
        self,
        argument: npt.ArrayLike | clarray.Array,
        queue: cl.CommandQueue,
    ) -> clarray.Array:
        """Coerce a NumPy/array-like input to a device array."""
        if isinstance(argument, clarray.Array):
            return argument.with_queue(queue)

        array = np.asarray(argument)
        if not array.flags["C_CONTIGUOUS"] and not array.flags["F_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        return clarray.to_device(queue, array)

    def _default_order(self, reference: clarray.Array) -> str:
        """Return the preferred output order based on a reference array."""
        return "C" if reference.flags.c_contiguous else "F"

    def _allocate_output(
        self,
        queue: cl.CommandQueue,
        shape: tuple[int, ...],
        dtype: npt.DTypeLike,
        order: str = "F",
        allocator: Any = None,
    ) -> clarray.Array:
        """Allocate an output array on the device."""
        return clarray.zeros(
            queue=queue,
            shape=shape,
            dtype=dtype,
            order=order,
            allocator=allocator,
        )

    def _supports_double_precision(self, context: cl.Context) -> bool:
        """Return whether any device in the context supports double precision."""
        return any(device.double_fp_config for device in context.devices)

    def _render_template(self, source: str, two_orders: bool = True) -> str:
        """Expand gratopy-style kernel templates.

        The current gratopy kernels use placeholders of the form:

        - ``\\my_variable_type``
        - ``\\order1``
        - ``\\order2``

        This helper expands the same scheme for operator-local kernel specs.
        """
        dtypes = ["float"]
        if self._supports_double_precision(self._last_queue.context):
            dtypes.append("double")

        rendered = []
        for dtype in dtypes:
            for order1 in ["f", "c"]:
                if two_orders:
                    for order2 in ["f", "c"]:
                        rendered.append(
                            source.replace("\\my_variable_type", dtype)
                            .replace("\\order1", order1)
                            .replace("\\order2", order2)
                        )
                else:
                    rendered.append(
                        source.replace("\\my_variable_type", dtype).replace(
                            "\\order1", order1
                        )
                    )
        return "".join(rendered)

    def _build_code(self, context: cl.Context, two_orders: bool = True) -> str:
        """Build OpenCL source code from the configured kernel spec."""
        dtypes = ["float"]
        if self._supports_double_precision(context):
            dtypes.append("double")

        rendered = []
        for source in self.kernel_spec.read_sources():
            for dtype in dtypes:
                for order1 in ["f", "c"]:
                    if two_orders:
                        for order2 in ["f", "c"]:
                            rendered.append(
                                source.replace("\\my_variable_type", dtype)
                                .replace("\\order1", order1)
                                .replace("\\order2", order2)
                            )
                    else:
                        rendered.append(
                            source.replace("\\my_variable_type", dtype).replace(
                                "\\order1", order1
                            )
                        )
        return "".join(rendered)

    def _get_program(
        self,
        context: cl.Context,
        two_orders: bool = True,
    ) -> dict[str, cl.Kernel]:
        """Compile kernels for the given context if necessary and return them."""
        cache_key = (context, f"{self.kernel_spec.signature}:two_orders={two_orders}")
        if cache_key in self._PROGRAM_CACHE:
            return self._PROGRAM_CACHE[cache_key]

        code = self._build_code(context, two_orders=two_orders)
        program = cl.Program(context, code)
        program.build(options=list(self.kernel_spec.build_options))
        kernels = {kernel.function_name: kernel for kernel in program.all_kernels()}
        self._PROGRAM_CACHE[cache_key] = kernels
        return kernels

    def _kernel_name(
        self,
        *,
        dtype: npt.DTypeLike,
        output_order: str,
        input_order: str,
        adjoint: bool = False,
    ) -> str:
        """Construct a concrete kernel name from the current kernel spec."""
        precision = "float" if np.dtype(dtype) == np.dtype("float32") else "double"
        base_name = self.kernel_spec.base_name
        if adjoint:
            return f"{base_name}_ad_{precision}_{output_order.lower()}{input_order.lower()}"
        return f"{base_name}_{precision}_{output_order.lower()}{input_order.lower()}"

    def _get_projection_kernel(
        self,
        context: cl.Context,
        *,
        dtype: npt.DTypeLike,
        output_order: str,
        input_order: str,
        adjoint: bool = False,
    ) -> cl.Kernel:
        """Return the compiled projection kernel for a given execution mode."""
        kernels = self._get_program(context, two_orders=True)
        kernel_name = self._kernel_name(
            dtype=dtype,
            output_order=output_order,
            input_order=input_order,
            adjoint=adjoint,
        )
        return kernels[kernel_name]

    def _invoke_kernel(
        self,
        kernel: cl.Kernel,
        queue: cl.CommandQueue,
        global_shape: tuple[int, ...],
        *arguments: Any,
        wait_for: list[cl.Event] | None = None,
    ) -> cl.Event:
        """Invoke an OpenCL kernel with a shared minimal wrapper."""
        if wait_for is None:
            wait_for = []
        return kernel(queue, global_shape, None, *arguments, wait_for=wait_for)


def invalidate_kernel_cache() -> None:
    """Clear the internal operator OpenCL program cache."""
    _OpenCLOperator._PROGRAM_CACHE.clear()
