"""Gratopy projection operators.

This module contains concrete operator implementations for gratopy's
experimental operator interface, most notably :class:`.Radon`.

The module provides compositional parallel-beam and fan-beam transforms,
including pixel-, ray-, and strip-driven discretizations and their adjoints.

Examples
--------

>>> import numpy as np
>>> import pyopencl as cl
>>> import pyopencl.array as clarray
>>> import gratopy
>>> ctx = cl.create_some_context(interactive=False)
>>> queue = cl.CommandQueue(ctx)
>>> Nx = 300
>>> img = np.zeros((Nx, Nx), dtype=np.float32)
>>> R = gratopy.operator.Radon(image_domain=Nx, angles=180)
>>> sinogram = R.apply_to(img, queue=queue)
>>> backprojection = R.T.apply_to(sinogram)

The same forward and adjoint applications can also be written via operator
syntax when the arrays already reside on the OpenCL device:

>>> device_img = clarray.to_device(queue, img)
>>> sinogram = R * device_img
>>> backprojection = R.T * sinogram
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyopencl as cl
import pyopencl.array as clarray

from dataclasses import replace
from pathlib import Path
from typing import Any

from gratopy.gratopy import fanbeam_struct, radon_struct
from gratopy.operator.geometry import ProjectionGeometry
from gratopy.operator.opencl import OpenCLKernelSpec, _OpenCLOperator
from gratopy.utilities import (
    Angles,
    Detectors,
    ExtentPlaceholder,
    ImageDomain,
    full_detector_given_image_fanbeam,
    full_detector_given_image_fullcircle,
    full_detector_given_image_halfcircle,
    full_image_given_detector_fanbeam,
    full_image_given_detector_fullcircle,
    full_image_given_detector_halfcircle,
    valid_detector_given_image_fanbeam,
    valid_detector_given_image_fullcircle,
    valid_detector_given_image_halfcircle,
    valid_image_given_detector_fanbeam,
    valid_image_given_detector_fullcircle,
    valid_image_given_detector_halfcircle,
)


class _ProjectionOperator(_OpenCLOperator):
    """Shared runtime plumbing for concrete projection geometries."""

    _operator_name = "Projection"
    _kernel_filename = ""
    _kernel_base_name = ""
    _device_array_specs: tuple[tuple[str, str], ...] = ()

    def __init__(
        self,
        *,
        state: dict[str, Any],
        kernel_spec: OpenCLKernelSpec | None,
    ) -> None:
        super().__init__(
            name=self._operator_name,
            state=state,
            kernel_spec=kernel_spec,
        )
        self._host_struct: dict[str, Any] | None = None
        self._device_struct: dict[tuple[cl.Context, np.dtype], dict[str, Any]] = {}
        self.input_shape = self.image_domain.size
        self.output_shape = (self.detectors.number, len(self.angles))

    def _default_kernel_spec(self) -> OpenCLKernelSpec:
        kernel_path = Path(__file__).resolve().parent.parent / self._kernel_filename
        return OpenCLKernelSpec.from_path(
            kernel_path,
            base_name=self._kernel_base_name,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle/deepcopy state without live OpenCL runtime objects."""
        state = super().__getstate__()
        state["_host_struct"] = None
        state["_device_struct"] = {}
        return state

    @property
    def image_domain(self) -> ImageDomain:
        return self.state["image_domain"]

    @property
    def angles(self) -> Angles:
        return self.state["angles"]

    @property
    def detectors(self) -> Detectors:
        return self.state["detectors"]

    def _ensure_host_struct(self, queue: cl.CommandQueue) -> None:
        """Populate the geometry-specific host structure."""
        raise NotImplementedError

    def _geometry_at(self, angles: npt.ArrayLike | None = None) -> ProjectionGeometry:
        """Return the physical geometry of the operator at the given angles.

        :param angles: Angles in radians. Defaults to all angles of the
            operator's angular sampling.
        :return: The source, detector and image-domain positions, see
            :class:`gratopy.operator.geometry.ProjectionGeometry`.
        """
        if angles is None:
            angles = self.angles.angles
        angles = np.atleast_1d(np.asarray(angles, dtype=float))
        image_extent = self.image_domain.extent
        detector_extent = self.detectors.extent
        assert not isinstance(image_extent, ExtentPlaceholder)
        assert not isinstance(detector_extent, ExtentPlaceholder)

        Nx, Ny = self.image_domain.size
        image_dims = np.array([Nx, Ny], dtype=float) * image_extent / max(Nx, Ny)
        beam = self._beam_geometry(angles, image_dims)
        return ProjectionGeometry(
            angles=angles,
            image_center=np.array(self.image_domain.center, dtype=float),
            image_dims=image_dims,
            detector_center=float(self.detectors.center),
            detector_width=float(detector_extent),
            detector_number=self.detectors.number,
            **beam,
        )

    def show_geometry(
        self,
        angles: npt.ArrayLike | None = None,
        ax: Any = None,
        *,
        n_rays: int = 15,
        coverage: bool = True,
        trajectory: bool = True,
        legend: bool = True,
    ) -> Any:
        """Draw the physical geometry of the operator.

        The drawing shows the image domain, the detector (with an arrow
        towards increasing pixel indices, pixel 0 and the detector center),
        sample rays over the detector and, for fan beams, the source. With
        ``coverage``, the illuminated region is shaded and rays missing the
        image domain are drawn dashed in red, which makes the
        :class:`~gratopy.utilities.ExtentPlaceholder` conventions visible.

        :param angles: Angle or angles in radians; several angles are overlaid
            in one axes. Defaults to the first angle of the operator.
        :param ax: :class:`matplotlib.axes.Axes` to draw into. Existing
            content is kept. If ``None``, a new figure is created; call
            :func:`matplotlib.pyplot.show` to display it.
        :param n_rays: Number of sample rays per angle.
        :param coverage: Whether to highlight the coverage of the image.
        :param trajectory: For fan beams, whether to draw the source circle
            with the source positions of all operator angles.
        :param legend: Whether to add a legend to the right of the axes, e.g.
            ``False`` when drawing a grid of geometries.
        :return: The :class:`matplotlib.axes.Axes` drawn into.
        """
        from gratopy.operator.geometry import plot_projection_geometry

        if angles is None:
            angles = self.angles.angles[:1]
        geometry = self._geometry_at(angles)
        return plot_projection_geometry(
            geometry,
            ax,
            n_rays=n_rays,
            coverage=coverage,
            trajectory=(
                self._geometry_at().sources if trajectory and geometry.is_fanbeam else None
            ),
            legend=legend,
            title=self._operator_name if ax is None else None,
        )

    def _beam_geometry(
        self, angles: np.ndarray, image_dims: np.ndarray
    ) -> dict[str, Any]:
        """Return the geometry-specific fields of :class:`ProjectionGeometry`.

        Geometries override this to provide ``detector_origin``,
        ``detector_axis`` and either ``sources`` or ``beam_direction``.
        """
        raise NotImplementedError(
            f"{self._operator_name} does not describe its physical geometry."
        )

    def _image_extent_formula(
        self,
        placeholder: ExtentPlaceholder,
        M: tuple[float, float, float],
        detector_extent: float,
        c: float,
    ) -> tuple[float, float] | None:
        """Return the image dimensions ``(Dx, Dy)`` for an image placeholder.

        Geometries supporting extent placeholders override this with their
        geometry-specific formulas; returning ``None`` signals that no image
        extent with the requested coverage exists.
        """
        raise NotImplementedError(
            f"{self._operator_name} does not support resolving an "
            "ExtentPlaceholder for the image domain; provide a numeric extent."
        )

    def _detector_extent_formula(
        self,
        placeholder: ExtentPlaceholder,
        M: tuple[float, float, float],
        D: tuple[float, float],
    ) -> float | None:
        """Return the detector width for a detector placeholder.

        Geometries supporting extent placeholders override this with their
        geometry-specific formulas; returning ``None`` signals that no
        detector width with the requested coverage exists.
        """
        raise NotImplementedError(
            f"{self._operator_name} does not support resolving an "
            "ExtentPlaceholder for the detectors; provide a numeric extent."
        )

    def _resolved_image_extent(
        self,
        placeholder: ExtentPlaceholder,
        detector_extent: float,
    ) -> float:
        """Compute a numeric image extent from a fixed detector extent."""
        Mx, My = self.image_domain.center
        Md = self.detectors.center
        Nx, Ny = self.image_domain.size
        c = Nx / Ny
        M = (Md, Mx, My)

        result = self._image_extent_formula(placeholder, M, detector_extent, c)

        if result is None:
            if placeholder == ExtentPlaceholder.FULL:
                meaning = (
                    "FULL means the largest image extent such that every "
                    "ray through the image also hits the detector"
                )
            else:
                meaning = (
                    "VALID means the smallest image extent such that every "
                    "ray hitting the detector also passes through the image"
                )
            raise ValueError(
                f"Cannot resolve ExtentPlaceholder.{placeholder.name} for the "
                f"image domain ({meaning}): no such image extent exists with "
                f"the given detector width Dd={detector_extent}, detector center "
                f"Md={Md}, and image center ({Mx}, {My}). Consider increasing "
                f"the detector width or adjusting the center offsets."
            )

        Dx, Dy = result
        return max(Dx, Dy)

    def _resolved_detector_extent(
        self,
        placeholder: ExtentPlaceholder,
        image_extent: float,
    ) -> float:
        """Compute a numeric detector extent from a fixed image extent."""
        Mx, My = self.image_domain.center
        Md = self.detectors.center
        Nx, Ny = self.image_domain.size
        Dx = image_extent * Nx / max(Nx, Ny)
        Dy = image_extent * Ny / max(Nx, Ny)
        M = (Md, Mx, My)
        D = (Dx, Dy)

        result = self._detector_extent_formula(placeholder, M, D)

        if result is None:
            if placeholder == ExtentPlaceholder.FULL:
                meaning = (
                    "FULL means the smallest detector width such that every "
                    "ray through the image also hits the detector"
                )
            else:
                meaning = (
                    "VALID means the largest detector width such that every "
                    "ray hitting the detector also passes through the image"
                )
            raise ValueError(
                f"Cannot resolve ExtentPlaceholder.{placeholder.name} for the "
                f"detector ({meaning}): no such detector width exists with image "
                f"dimensions ({Dx}, {Dy}), detector center Md={Md}, and image "
                f"center ({Mx}, {My}). Consider increasing the image extent or "
                f"adjusting the center offsets."
            )
        return result

    def _resolve_extent_placeholders(self) -> None:
        """Resolve extent placeholders without mutating the input geometry."""
        image_extent = self.image_domain.extent
        detector_extent = self.detectors.extent
        if isinstance(image_extent, ExtentPlaceholder) and isinstance(
            detector_extent, ExtentPlaceholder
        ):
            raise NotImplementedError(
                "Both the ImageDomain and Detectors use an ExtentPlaceholder. "
                "Please set at least one of the two extents to a specific "
                "value; the remaining placeholder will be resolved "
                "automatically."
            )

        if isinstance(image_extent, ExtentPlaceholder):
            assert not isinstance(detector_extent, ExtentPlaceholder)
            self.state["image_domain"] = replace(
                self.image_domain,
                extent=self._resolved_image_extent(
                    image_extent,
                    float(detector_extent),
                ),
            )

        if isinstance(detector_extent, ExtentPlaceholder):
            assert not isinstance(image_extent, ExtentPlaceholder)
            self.state["detectors"] = replace(
                self.detectors,
                extent=self._resolved_detector_extent(
                    detector_extent,
                    float(image_extent),
                ),
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
        host_arrays = {
            device_name: self._host_struct[host_name][dtype]
            for device_name, host_name in self._device_array_specs
        }
        device_struct = self._upload_read_only_buffers(queue, host_arrays)
        self._device_struct[cache_key] = device_struct
        return device_struct

    def _kernel_arguments(
        self,
        output: clarray.Array,
        argument: clarray.Array,
        queue: cl.CommandQueue,
    ) -> tuple[Any, ...]:
        device_struct = self._ensure_device_struct(queue, argument.dtype)
        return tuple(
            device_struct[device_name]
            for device_name, _host_name in self._device_array_specs
        )


class Radon(_ProjectionOperator):
    """Parallel-beam Radon transform operator.

    A :class:`Radon` object represents the pixel-driven forward projection.
    Accessing :attr:`T` creates the corresponding weighted adjoint and shares
    the geometry and OpenCL runtime caches of the forward operator.

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
        :meth:`gratopy.utilities.Angles.uniform` with the ``half_circle``
        parameter set to ``True``.
    ``detectors``:
        Detector configuration. This can be given as:

        - ``None`` to use the default detector count
          ``ceil(hypot(Nx, Ny))``,
        - an integer specifying the number of detector pixels,
        - or an explicit :class:`gratopy.utilities.Detectors` object.

        Plain integer inputs are converted to
        :class:`~gratopy.utilities.Detectors` with default extent handling.
    ``kernel_spec``:
        Optional :class:`gratopy.operator.opencl.OpenCLKernelSpec` describing
        which OpenCL kernel bundle to use. When omitted, the operator uses the
        Radon kernels shipped with gratopy.

    **Notes**

    The operator accepts both :class:`pyopencl.array.Array` inputs and NumPy
    arrays. Every application to a NumPy array must receive an explicit queue
    or a device output from which the queue can be inferred. Operators do not
    remember queues from earlier applications. Device inputs carry their queue
    and can therefore be used with the shorter ``R * image`` syntax.

    The operator supports 2D as well as slicewise 3D data. For example,
    a forward operator with image shape ``(Nx, Ny)`` maps:

    - ``(Nx, Ny)`` to ``(Ns, Na)``,
    - ``(Nx, Ny, Nz)`` to ``(Ns, Na, Nz)``.

    Correspondingly, the adjoint maps:

    - ``(Ns, Na)`` to ``(Nx, Ny)``,
    - ``(Ns, Na, Nz)`` to ``(Nx, Ny, Nz)``.

    A numeric image extent can be paired with a detector
    :class:`gratopy.utilities.ExtentPlaceholder` to infer detector coverage.
    Conversely, a numeric detector extent can be paired with an image extent
    placeholder to infer the image domain. ``FULL`` and ``VALID`` select the
    corresponding coverage convention.

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

    _operator_name = "Radon"
    _kernel_filename = "radon.cl"
    _kernel_base_name = "radon"
    _device_array_specs = (("ofs", "ofs_dict"), ("geometry", "geo_dict"))

    def __init__(
        self,
        image_domain: int | tuple[int, int] | ImageDomain,
        angles: Angles | int,
        detectors: Detectors | int | None = None,
        kernel_spec: OpenCLKernelSpec | None = None,
    ):
        if not isinstance(image_domain, ImageDomain):
            image_domain = ImageDomain(size=image_domain, extent=2.0)

        if not isinstance(angles, Angles):
            angles = Angles.uniform(number=angles, half_circle=True)

        if not isinstance(detectors, Detectors):
            if detectors is None:
                detectors = int(np.ceil(np.hypot(*image_domain.size)))
            detector_extent = image_domain.extent
            if isinstance(detector_extent, ExtentPlaceholder):
                detector_extent = ExtentPlaceholder.FULL
            detectors = Detectors(number=detectors, extent=detector_extent)

        state = {
            "image_domain": image_domain,
            "angles": angles,
            "detectors": detectors,
        }
        super().__init__(state=state, kernel_spec=kernel_spec)
        self._resolve_extent_placeholders()

    def _use_full_circle(self) -> bool:
        """Decide whether extent placeholders use full-circle geometry.

        The choice follows the angular sampling's ``half_circle`` flag. That
        flag is load-bearing only for the equispaced constructors
        (:meth:`Angles.sparse` / :meth:`Angles.uniform`); for limited-angle and
        explicit samplings it is plain metadata defaulting to ``False``. So a
        placeholder resolved on a sampling that was never explicitly flagged
        could silently fall back to the full-circle formulas. Guard against
        that: a full-circle resolution is only trusted when the sampled angles
        actually cover more than half a circle.
        """
        if self.angles.half_circle:
            return False

        angles = np.asarray(self.angles.angles, dtype=float)
        angular_range = float(angles.max() - angles.min())
        # POLICY (review): "covers the full circle" == angular range > pi.
        # This cleanly separates half-circle data (range < pi) from full-circle
        # data (range ~= 2*pi); a partial scan in between is treated as full.
        if angular_range <= np.pi + 1e-9:
            raise ValueError(
                "Cannot resolve an ExtentPlaceholder: the angular sampling is "
                "marked half_circle=False (full circle), but the angles only "
                f"span {angular_range:.4f} rad (<= pi). If this is half-circle "
                "data, build the Angles with half_circle=True; if it is a "
                "genuine full-circle scan, widen the angular range."
            )
        return True

    def _image_extent_formula(
        self,
        placeholder: ExtentPlaceholder,
        M: tuple[float, float, float],
        detector_extent: float,
        c: float,
    ) -> tuple[float, float] | None:
        full_circle = self._use_full_circle()
        if placeholder == ExtentPlaceholder.FULL:
            if full_circle:
                return full_image_given_detector_fullcircle(M, detector_extent, c)
            return full_image_given_detector_halfcircle(M, detector_extent, c)
        if full_circle:
            return valid_image_given_detector_fullcircle(M, detector_extent, c)
        return valid_image_given_detector_halfcircle(M, detector_extent, c)

    def _detector_extent_formula(
        self,
        placeholder: ExtentPlaceholder,
        M: tuple[float, float, float],
        D: tuple[float, float],
    ) -> float | None:
        full_circle = self._use_full_circle()
        if placeholder == ExtentPlaceholder.FULL:
            if full_circle:
                return full_detector_given_image_fullcircle(M, D)
            return full_detector_given_image_halfcircle(M, D)
        if full_circle:
            return valid_detector_given_image_fullcircle(M, D)
        return valid_detector_given_image_halfcircle(M, D)

    def _beam_geometry(
        self, angles: np.ndarray, image_dims: np.ndarray
    ) -> dict[str, Any]:
        # The parallel-beam detector can lie at any distance along the beam;
        # place it just outside the scene so that drawings stay compact.
        scene_radius = max(
            np.hypot(*self.image_domain.center) + np.hypot(*image_dims) / 2,
            abs(self.detectors.center) + float(self.detectors.extent) / 2,  # type: ignore[arg-type]
        )
        beam_direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        # radon_struct has no detector reversal, so neither has the geometry.
        detector_axis = np.stack([np.sin(angles), -np.cos(angles)], axis=1)
        return {
            "detector_origin": 1.1 * scene_radius * beam_direction,
            "detector_axis": detector_axis,
            "beam_direction": beam_direction,
        }

    def _ensure_host_struct(self, queue: cl.CommandQueue) -> None:
        if self._host_struct is not None:
            return

        self._host_struct = radon_struct(
            queue=queue,
            img_shape=self.image_domain.size,
            angles=self.angles.angles,
            angle_weights=self.angles.weights,
            n_detectors=self.detectors.number,
            detector_width=float(self.detectors.extent),  # type: ignore[arg-type]
            image_width=float(self.image_domain.extent),  # type: ignore[arg-type]
            midpoint_shift=self.image_domain.center,
            detector_shift=self.detectors.center,
        )


class Fanbeam(_ProjectionOperator):
    """Pixel-driven fan-beam projection operator.

    **Parameters**

    ``image_domain``:
        Image grid, physical extent, and center. An integer creates a square
        image domain; a tuple creates a rectangular domain.
    ``angles``:
        Angular sampling. An integer creates a uniformly weighted full-circle
        sampling; an :class:`gratopy.utilities.Angles` value supplies explicit
        angles and quadrature weights.
    ``detectors``:
        Explicit detector count, physical extent, center, and orientation.
    ``source_detector_distance``:
        Orthogonal distance from the source to the detector line.
    ``source_origin_distance``:
        Distance from the source to the rotation center. The source lies
        outside the image domain and this distance is smaller than
        ``source_detector_distance``.
    ``kernel_spec``:
        Optional OpenCL kernel bundle. The bundled pixel-driven Fanbeam kernels
        are selected by default.

    The operator maps image arrays to sinograms of shape
    ``(detectors.number, len(angles))`` and applies slicewise to trailing
    dimensions. :attr:`T` provides the weighted adjoint while sharing geometry
    and OpenCL runtime caches with the forward operator.

    As for :class:`Radon`, either the image or the detector extent can be a
    :class:`gratopy.utilities.ExtentPlaceholder`, which is then resolved from
    the other, numeric extent. Fan-beam placeholders are always resolved with
    full-circle geometry, independent of the angular sampling; for
    limited-angle scans, choose numeric extents explicitly.

    **Example**

    >>> from gratopy.operator import Fanbeam
    >>> from gratopy.utilities import Detectors
    >>> F = Fanbeam(
    ...     image_domain=128,
    ...     angles=360,
    ...     detectors=Detectors(number=192, extent=3.0),
    ...     source_detector_distance=8.0,
    ...     source_origin_distance=4.0,
    ... )
    """

    _operator_name = "Fanbeam"
    _kernel_filename = "fanbeam.cl"
    _kernel_base_name = "fanbeam"
    _device_array_specs: tuple[tuple[str, str], ...] = (
        ("ofs", "ofs_dict"),
        ("sdpd", "sdpd_dict"),
        ("geometry", "geo_dict"),
    )

    def __init__(
        self,
        image_domain: int | tuple[int, int] | ImageDomain,
        angles: Angles | int,
        detectors: Detectors,
        *,
        source_detector_distance: float,
        source_origin_distance: float,
        kernel_spec: OpenCLKernelSpec | None = None,
    ) -> None:
        if not isinstance(image_domain, ImageDomain):
            image_domain = ImageDomain(size=image_domain, extent=2.0)

        if not isinstance(angles, Angles):
            angles = Angles.uniform(number=angles)

        if not isinstance(detectors, Detectors):
            raise TypeError("Fanbeam detectors must be an explicit Detectors instance.")

        state = {
            "source_detector_distance": float(source_detector_distance),
            "source_origin_distance": float(source_origin_distance),
            "image_domain": image_domain,
            "angles": angles,
            "detectors": detectors,
        }
        super().__init__(state=state, kernel_spec=kernel_spec)
        self._validate_source_distances()
        resolved_placeholder = self._placeholder_description()
        self._resolve_extent_placeholders()
        self._validate_geometry(resolved_placeholder)

    @property
    def source_detector_distance(self) -> float:
        return self.state["source_detector_distance"]

    @property
    def source_origin_distance(self) -> float:
        return self.state["source_origin_distance"]

    def _image_extent_formula(
        self,
        placeholder: ExtentPlaceholder,
        M: tuple[float, float, float],
        detector_extent: float,
        c: float,
    ) -> tuple[float, float] | None:
        RE = self.source_origin_distance
        R = self.source_detector_distance
        if placeholder == ExtentPlaceholder.FULL:
            return full_image_given_detector_fanbeam(M, detector_extent, RE, R, c)
        return valid_image_given_detector_fanbeam(M, detector_extent, RE, R, c)

    def _detector_extent_formula(
        self,
        placeholder: ExtentPlaceholder,
        M: tuple[float, float, float],
        D: tuple[float, float],
    ) -> float | None:
        RE = self.source_origin_distance
        R = self.source_detector_distance
        if placeholder == ExtentPlaceholder.FULL:
            return full_detector_given_image_fanbeam(M, D, RE, R)
        return valid_detector_given_image_fanbeam(M, D, RE, R)

    def _beam_geometry(
        self, angles: np.ndarray, image_dims: np.ndarray
    ) -> dict[str, Any]:
        RE = self.source_origin_distance
        R = self.source_detector_distance
        towards_detector = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        orientation = -1.0 if self.detectors.reversed else 1.0
        detector_axis = orientation * np.stack([np.sin(angles), -np.cos(angles)], axis=1)
        return {
            "detector_origin": (R - RE) * towards_detector,
            "detector_axis": detector_axis,
            "sources": -RE * towards_detector,
        }

    def _placeholder_description(self) -> str | None:
        """Describe the extent placeholder in use, for error messages."""
        if isinstance(self.image_domain.extent, ExtentPlaceholder):
            return f"image extent ExtentPlaceholder.{self.image_domain.extent.name}"
        if isinstance(self.detectors.extent, ExtentPlaceholder):
            return f"detector extent ExtentPlaceholder.{self.detectors.extent.name}"
        return None

    def _validate_geometry(self, resolved_placeholder: str | None = None) -> None:
        """Validate immutable fan-beam geometry during construction."""
        image_extent = self._numeric_image_extent()
        self._validate_discretization()
        self._validate_source_geometry(image_extent, resolved_placeholder)

    def _numeric_image_extent(self) -> float:
        """Validate fan-beam extents and return the numeric image extent."""
        image_extent = self.image_domain.extent
        detector_extent = self.detectors.extent
        assert not isinstance(image_extent, ExtentPlaceholder)
        assert not isinstance(detector_extent, ExtentPlaceholder)
        if not np.isfinite(image_extent) or image_extent <= 0:
            raise ValueError("image extent must be positive and finite")
        if not np.isfinite(detector_extent) or detector_extent <= 0:
            raise ValueError("detector extent must be positive and finite")
        return image_extent

    def _validate_discretization(self) -> None:
        """Validate fan-beam grid sizes, centers, and angular sampling."""
        if any(size <= 0 for size in self.image_domain.size):
            raise ValueError("image dimensions must be positive")
        if self.detectors.number <= 0:
            raise ValueError("number of detectors must be positive")
        if not np.all(np.isfinite(self.image_domain.center)):
            raise ValueError("image center must be finite")
        if not np.isfinite(self.detectors.center):
            raise ValueError("detector center must be finite")
        if len(self.angles) == 0:
            raise ValueError("at least one angle is required")

    def _validate_source_distances(self) -> None:
        """Validate the source distances used by the placeholder formulas."""
        source_detector_distance = self.source_detector_distance
        source_origin_distance = self.source_origin_distance
        if not np.isfinite(source_origin_distance) or source_origin_distance <= 0:
            raise ValueError("source_origin_distance must be positive and finite")
        if (
            not np.isfinite(source_detector_distance)
            or source_detector_distance <= source_origin_distance
        ):
            raise ValueError(
                "source_detector_distance must be finite and greater than "
                "source_origin_distance"
            )

    def _validate_source_geometry(
        self,
        image_extent: float,
        resolved_placeholder: str | None = None,
    ) -> None:
        """Ensure the source stays outside the image domain."""
        Nx, Ny = self.image_domain.size
        scale = image_extent / max(Nx, Ny)
        corner_radius = 0.5 * np.hypot(scale * Nx, scale * Ny)
        center_distance = np.hypot(*self.image_domain.center)
        if corner_radius + center_distance >= self.source_origin_distance:
            message = "source must lie outside the image domain"
            if resolved_placeholder is not None:
                message += (
                    f"; resolving the {resolved_placeholder} gave image extent "
                    f"{image_extent} and detector extent {self.detectors.extent}, "
                    "which places the source inside the image domain. Consider "
                    "a smaller detector extent, a larger source_origin_distance, "
                    "or numeric extents."
                )
            raise ValueError(message)

    def _ensure_host_struct(self, queue: cl.CommandQueue) -> None:
        if self._host_struct is not None:
            return

        detector_extent = self.detectors.extent
        image_extent = self.image_domain.extent
        assert not isinstance(detector_extent, ExtentPlaceholder)
        assert not isinstance(image_extent, ExtentPlaceholder)
        self._host_struct = fanbeam_struct(
            queue=queue,
            img_shape=self.image_domain.size,
            angles=self.angles.angles,
            detector_width=float(detector_extent),
            source_detector_dist=float(self.source_detector_distance),
            source_origin_dist=float(self.source_origin_distance),
            angle_weights=self.angles.weights,
            n_detectors=self.detectors.number,
            detector_shift=self.detectors.center,
            image_width=float(image_extent),
            midpoint_shift=self.image_domain.center,
            reverse_detector=self.detectors.reversed,
        )


class RayDrivenRadon(Radon):
    """Parallel-beam projection using the ray-driven kernel discretization."""

    _operator_name = "RayDrivenRadon"
    _kernel_base_name = "radon_ray"


class StripDrivenRadon(Radon):
    """Parallel-beam projection using the strip-driven kernel discretization."""

    _operator_name = "StripDrivenRadon"
    _kernel_base_name = "radon_strip"


class RayDrivenFanbeam(Fanbeam):
    """Fan-beam projection using the ray-driven kernel discretization."""

    _operator_name = "RayDrivenFanbeam"
    _kernel_base_name = "fanbeam_ray"
    _device_array_specs: tuple[tuple[str, str], ...] = (
        ("ofs", "ofs_dict"),
        ("geometry", "geo_dict"),
    )
