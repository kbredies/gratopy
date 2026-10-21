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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock, local
from typing import Any, Callable, Literal
from weakref import WeakValueDictionary

import numpy as np
import numpy.typing as npt
import pyopencl as cl
import pyopencl.array as clarray

from gratopy.operator.base import Operator


def _expand_gratopy_template(
    source: str,
    *,
    dtypes: list[str],
    two_orders: bool,
) -> str:
    """Expand gratopy-style OpenCL kernel templates."""
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


@dataclass(frozen=True)
class OpenCLKernelSpec:
    """Description of an OpenCL kernel bundle.

    This class describes the OpenCL source files used by an experimental
    gratopy operator as well as the kernel base name used for lookup.

    **Parameters**

    ``paths``:
        One or more kernel source files. Paths are normalized to absolute
        paths during initialization.
    ``base_name``:
        Base name used for generated kernel lookup. With the current naming
        convention, forward and adjoint kernels are expected to follow the
        pattern

        ``{base_name}_{dtype}_{out_order}{in_order}``

        and

        ``{base_name}_ad_{dtype}_{out_order}{in_order}``,

        respectively.
    ``build_options``:
        Optional extra compiler flags forwarded to :meth:`cl.Program.build`.

    **Notes**

    The current operator implementation expects gratopy-style template
    placeholders in the source files, most notably ``\\my_variable_type``,
    ``\\order1``, and ``\\order2``.

    The :attr:`signature` property describes compiled program identity using
    the source paths, source contents, their order, and build options. Operators
    with the same compilation inputs share one compiled program, while
    ``base_name`` selects a kernel family from that program.

    **Examples**
    Use the default shipped Radon kernels implicitly via
    :class:`gratopy.operator.projection.Radon`, or provide an explicit custom
    kernel bundle:

    >>> from gratopy.operator import OpenCLKernelSpec
    >>> spec = OpenCLKernelSpec.from_path("scratch/my_radon.cl", base_name="radon")

    The resulting spec can then be passed to a custom or built-in operator.
    """

    paths: tuple[str, ...]
    base_name: str
    build_options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_paths = tuple(
            str(Path(path).expanduser().resolve()) for path in self.paths
        )
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

    def source_snapshot(self) -> tuple[tuple[str, ...], str]:
        """Read the configured sources and return them with their signature."""
        sources = self.read_sources()
        digest = hashlib.sha256()
        for option in self.build_options:
            digest.update(b"\0")
            digest.update(option.encode())
        for path, source in zip(self.paths, sources):
            digest.update(b"\0")
            digest.update(path.encode())
            digest.update(b"\0")
            digest.update(source.encode())
        return sources, digest.hexdigest()

    @property
    def signature(self) -> str:
        """Return a content-based signature used for program cacheing."""
        return self.source_snapshot()[1]


_ProgramCacheKey = tuple[cl.Context, str, bool]
_ProgramLeaseKey = tuple[cl.Context, bool]


class _ProgramBundle:
    """Compiled program shared by operators through strong runtime leases."""

    def __init__(
        self,
        *,
        cache_key: _ProgramCacheKey,
        generation: int,
        program: cl.Program,
    ) -> None:
        self.cache_key = cache_key
        self.generation = generation
        self.program = program
        self._thread_state = local()

    def kernel(self, name: str) -> cl.Kernel:
        """Return a kernel instance local to the calling thread."""
        kernels = getattr(self._thread_state, "kernels", None)
        if kernels is None:
            kernels = {}
            self._thread_state.kernels = kernels

        kernel = kernels.get(name)
        if kernel is None:
            kernel = cl.Kernel(self.program, name)
            kernels[name] = kernel
        return kernel


class _ProgramCache:
    """Weak registry interning compiled OpenCL program bundles."""

    def __init__(self) -> None:
        self._bundles: WeakValueDictionary[_ProgramCacheKey, _ProgramBundle] = (
            WeakValueDictionary()
        )
        self._generation = 0
        self._lock = RLock()

    @property
    def generation(self) -> int:
        """Return the current invalidation generation."""
        with self._lock:
            return self._generation

    def acquire(
        self,
        cache_key: _ProgramCacheKey,
        build: Callable[[], cl.Program],
    ) -> _ProgramBundle:
        """Return a shared bundle, compiling it once on a cache miss."""
        with self._lock:
            bundle = self._bundles.get(cache_key)
            if bundle is not None:
                return bundle

            bundle = _ProgramBundle(
                cache_key=cache_key,
                generation=self._generation,
                program=build(),
            )
            self._bundles[cache_key] = bundle
            return bundle

    def invalidate(self) -> None:
        """Invalidate all entries while allowing active leases to finish."""
        with self._lock:
            self._generation += 1
            self._bundles.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._bundles)


_PROGRAM_CACHE = _ProgramCache()


class _OpenCLOperator(Operator):
    """Internal base class for OpenCL-backed operators.

    This class intentionally only implements shared execution helpers.
    Concrete subclasses are still responsible for geometry handling and for
    deciding which kernels to load.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        kernel_spec: OpenCLKernelSpec | None = None,
        **operator_kwargs: Any,
    ):
        super().__init__(name=name, **operator_kwargs)
        self.kernel_spec = kernel_spec or self._default_kernel_spec()
        self._program_bundles: dict[_ProgramLeaseKey, _ProgramBundle] = {}

    def _default_kernel_spec(self) -> OpenCLKernelSpec:
        """Return the default kernel spec for this operator.

        Concrete subclasses are expected to override this if they want to rely
        on `_OpenCLOperator` construction directly.
        """
        raise NotImplementedError("Concrete OpenCL operators must define a kernel spec")

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle/deepcopy state without live OpenCL program leases."""
        state = super().__getstate__()
        state["_program_bundles"] = {}
        return state

    def _infer_queue(
        self,
        argument: npt.ArrayLike | clarray.Array | None = None,
        output: clarray.Array | None = None,
        queue: cl.CommandQueue | None = None,
    ) -> cl.CommandQueue:
        """Infer the queue to use for computation."""
        if queue is not None:
            return queue

        if isinstance(argument, clarray.Array) and argument.queue is not None:
            return argument.queue

        if isinstance(output, clarray.Array) and output.queue is not None:
            return output.queue

        raise ValueError(
            "No OpenCL queue available. Either pass an explicit queue, provide "
            "a clarray.Array as input, or provide an output array with an "
            "associated queue."
        )

    @staticmethod
    def _coerce_argument(
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

    @staticmethod
    def _default_order(reference: clarray.Array) -> Literal["C", "F"]:
        """Return the preferred output order based on a reference array."""
        return "C" if reference.flags.c_contiguous else "F"

    @staticmethod
    def _allocate_output(
        queue: cl.CommandQueue,
        shape: tuple[int, ...],
        dtype: npt.DTypeLike,
        order: Literal["C", "F"] = "F",
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

    @staticmethod
    def _upload_read_only_buffers(
        queue: cl.CommandQueue,
        host_arrays: Mapping[str, np.ndarray],
    ) -> dict[str, cl.Buffer]:
        """Upload named host arrays as immutable kernel argument buffers."""
        device_buffers = {}
        for name, host_array in host_arrays.items():
            buffer = cl.Buffer(
                queue.context,
                cl.mem_flags.READ_ONLY,
                host_array.nbytes,
            )
            cl.enqueue_copy(queue, buffer, host_array.data).wait()
            device_buffers[name] = buffer
        return device_buffers

    @staticmethod
    def _supports_double_precision(context: cl.Context) -> bool:
        """Return whether any device in the context supports double precision."""
        return any(device.double_fp_config for device in context.devices)

    def _build_code(
        self,
        context: cl.Context,
        two_orders: bool = True,
        sources: tuple[str, ...] | None = None,
    ) -> str:
        """Build OpenCL source code from the configured kernel spec."""
        dtypes = ["float"]
        if self._supports_double_precision(context):
            dtypes.append("double")
        if sources is None:
            sources = self.kernel_spec.read_sources()

        return "".join(
            _expand_gratopy_template(source, dtypes=dtypes, two_orders=two_orders)
            for source in sources
        )

    def _get_program(
        self,
        context: cl.Context,
        two_orders: bool = True,
    ) -> _ProgramBundle:
        """Acquire a compiled program bundle for the given context."""
        sources, signature = self.kernel_spec.source_snapshot()
        cache_key = (context, signature, two_orders)
        lease_key = (context, two_orders)
        bundle = self._program_bundles.get(lease_key)
        if (
            bundle is not None
            and bundle.cache_key == cache_key
            and bundle.generation == _PROGRAM_CACHE.generation
        ):
            return bundle

        def build() -> cl.Program:
            code = self._build_code(
                context,
                two_orders=two_orders,
                sources=sources,
            )
            program = cl.Program(context, code)
            program.build(options=list(self.kernel_spec.build_options))
            return program

        bundle = _PROGRAM_CACHE.acquire(cache_key, build)
        self._program_bundles[lease_key] = bundle
        return bundle

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
            return (
                f"{base_name}_ad_{precision}_{output_order.lower()}{input_order.lower()}"
            )
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
        program = self._get_program(context, two_orders=True)
        kernel_name = self._kernel_name(
            dtype=dtype,
            output_order=output_order,
            input_order=input_order,
            adjoint=adjoint,
        )
        return program.kernel(kernel_name)

    def _expected_output_shape(
        self,
        argument_shape: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[int, ...]:
        """Return the expected output shape for an execution direction."""
        input_shape = self.output_shape if adjoint else self.input_shape
        output_shape = self.input_shape if adjoint else self.output_shape
        if input_shape is None or output_shape is None:
            raise NotImplementedError(
                "Concrete OpenCL operators must define _expected_output_shape() or "
                "set input_shape/output_shape appropriately."
            )

        extra_dims = argument_shape[len(input_shape) :]
        return output_shape + extra_dims

    def _validate_argument(
        self,
        argument: clarray.Array,
        *,
        adjoint: bool = False,
    ) -> None:
        """Validate an input argument before kernel execution."""
        input_shape = self.output_shape if adjoint else self.input_shape
        if input_shape is None:
            return

        if argument.shape[0 : len(input_shape)] != input_shape:
            raise ValueError(
                f"Input shape mismatch: expected {input_shape}, got {argument.shape}"
            )

    def _validate_output(
        self,
        output: clarray.Array,
        argument: clarray.Array,
        *,
        adjoint: bool = False,
    ) -> clarray.Array:
        """Validate and normalize an output array before kernel execution."""
        output = output.with_queue(argument.queue)
        expected_shape = self._expected_output_shape(
            argument.shape,
            adjoint=adjoint,
        )

        if output.dtype != argument.dtype:
            raise ValueError(
                f"Output dtype mismatch: expected {argument.dtype}, got {output.dtype}"
            )
        if output.shape != expected_shape:
            raise ValueError(
                f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
            )
        return output

    def _vector_norm(self, vector: Any) -> float:
        """Compute the Euclidean norm of an OpenCL device array."""
        if not isinstance(vector, clarray.Array):
            return super()._vector_norm(vector)
        squared_norm = clarray.vdot(vector, vector).get()
        return float(np.sqrt(np.real(squared_norm)))

    def _kernel_arguments(
        self,
        output: clarray.Array,
        argument: clarray.Array,
        queue: cl.CommandQueue,
    ) -> tuple[Any, ...]:
        """Return additional kernel arguments beyond output/input buffers."""
        return ()

    def _get_kernel(
        self,
        output: clarray.Array,
        argument: clarray.Array,
        queue: cl.CommandQueue,
        *,
        adjoint: bool = False,
    ) -> cl.Kernel:
        """Return the kernel used for an OpenCL execution direction."""
        return self._get_projection_kernel(
            queue.context,
            dtype=argument.dtype,
            output_order=self._default_order(output),
            input_order=self._default_order(argument),
            adjoint=adjoint,
        )

    def _global_shape(
        self,
        output: clarray.Array,
        argument: clarray.Array,
    ) -> tuple[int, ...]:
        """Return the global shape used for kernel execution."""
        return output.shape

    def _apply_opencl(
        self,
        argument: Any,
        output: Any | None,
        queue: cl.CommandQueue | None,
        return_event: bool,
        *,
        adjoint: bool,
    ) -> clarray.Array | tuple[clarray.Array, list[cl.Event]]:
        """Execute one direction through the shared OpenCL pipeline."""
        queue = self._infer_queue(argument=argument, output=output, queue=queue)
        argument = self._coerce_argument(argument, queue)
        self._validate_argument(argument, adjoint=adjoint)

        if output is None:
            output = self._allocate_output(
                queue=queue,
                shape=self._expected_output_shape(argument.shape, adjoint=adjoint),
                dtype=argument.dtype,
                order=self._default_order(argument),
                allocator=argument.allocator,
            )
        output = self._validate_output(output, argument, adjoint=adjoint)

        kernel = self._get_kernel(output, argument, queue, adjoint=adjoint)
        event = self._invoke_kernel(
            kernel,
            queue,
            self._global_shape(output, argument),
            output.data,
            argument.data,
            *self._kernel_arguments(output, argument, queue),
            wait_for=output.events + argument.events,
        )
        output.add_event(event)

        if return_event:
            return output, [event]
        return output

    def apply_to(
        self,
        argument: Any,
        output: Any | None = None,
        queue: cl.CommandQueue | None = None,
        return_event: bool = False,
        **kwargs: Any,
    ) -> clarray.Array | tuple[clarray.Array, list[cl.Event]]:
        """Apply the forward OpenCL kernel."""
        return self._apply_opencl(
            argument,
            output,
            queue,
            return_event,
            adjoint=False,
        )

    def apply_adjoint_to(
        self,
        argument: Any,
        output: Any | None = None,
        queue: cl.CommandQueue | None = None,
        return_event: bool = False,
        **kwargs: Any,
    ) -> clarray.Array | tuple[clarray.Array, list[cl.Event]]:
        """Apply the adjoint OpenCL kernel."""
        return self._apply_opencl(
            argument,
            output,
            queue,
            return_event,
            adjoint=True,
        )

    @staticmethod
    def _invoke_kernel(
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
    """Invalidate compiled programs used by experimental OpenCL operators.

    Existing invocations may finish with their current programs. Live operators
    acquire a fresh bundle on their next application, while bundles without any
    remaining operator leases are released automatically.
    """
    _PROGRAM_CACHE.invalidate()
