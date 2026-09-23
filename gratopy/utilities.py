"""Miscellaneous utility functions for gratopy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

Numeric: TypeAlias = float | int | np.int_ | np.float32 | np.double


class ExtentPlaceholder(Enum):
    """Placeholder values for image and detector extents.

    These placeholders are primarily intended for the experimental operator
    API, where image and detector extents can be specified independently.
    They express geometric intent rather than an immediate numerical value.

    The placeholder mechanism in the experimental operator API is still
    evolving. The experimental Radon and Fanbeam operators can resolve one
    placeholder extent at a time: either the image extent from a fixed detector
    extent, or the detector extent from a fixed image extent. Passing
    placeholders for both extents at once is unsupported and raises
    :class:`NotImplementedError`. Fanbeam placeholders are always resolved for
    a full-circle scan.
    """

    FULL = "full"
    """From the perspective of the detector: smallest detector extent such
    that each ray hitting the image domain also hits the detector.
    From the perspective of the image domain: largest image extent such
    that each ray hitting the image domain also hits the detector.
    """

    VALID = "valid"
    """From the perspective of the detector: the largest extent such
    that each ray hitting the detector also hits the image domain.
    From the perspective of the image domain: the smallest extent such that
    each ray hitting the detector also hits the image domain.
    """


class GeometryType(Enum):
    """
    Enum for the different geometry types.
    """

    RADON = "radon"
    FANBEAM = "fanbeam"


@dataclass(frozen=True, eq=False)
class Angles:
    r"""Angular sampling together with associated weights.

    An :class:`Angles` object stores the projection angles in radians and the
    corresponding angle weights used, for instance, in the adjoint operator.
    It serves as the main explicit representation of angular geometry in the
    experimental operator API.

    **Parameters**

    ``angles``:
        One-dimensional array-like object containing angles in radians.
    ``weights``:
        One-dimensional array-like object containing the weights associated
        with the given angles.
    ``half_circle``:
        Whether the angular sampling should be interpreted with half-circle
        (:math:`\pi`) periodicity instead of full-circle (:math:`2\pi`)
        periodicity. For constructors that generate a complete equispaced
        sampling, this also determines whether angles are generated in
        :math:`[0, \pi)` or :math:`[0, 2\pi)`. For limited-angle or explicit
        samplings, it is stored as metadata for downstream code.

    **Notes**

    Instances are immutable. Input arrays are copied during construction and
    stored as read-only arrays, so changing the caller's arrays cannot alter an
    existing angular sampling.

    The static constructors on this class provide a few common ways of
    constructing angular samplings:

    - :meth:`sparse` for unweighted equispaced angles,
    - :meth:`uniform` for equispaced angles with uniform weights,
    - :meth:`uniform_interval` for one limited-angle interval,
    - :meth:`intervals` for multiple limited-angle intervals,
    - :meth:`from_list` for explicit angles with automatically inferred
      *natural* weights.
    """

    angles: np.ndarray
    weights: np.ndarray
    half_circle: bool = False

    def __init__(
        self,
        angles: npt.ArrayLike,
        weights: npt.ArrayLike,
        half_circle: bool = False,
    ):
        angle_values = np.asarray(angles)
        weight_values = np.asarray(weights)
        angles_array = np.frombuffer(
            angle_values.tobytes(),
            dtype=angle_values.dtype,
        ).reshape(angle_values.shape)
        weights_array = np.frombuffer(
            weight_values.tobytes(),
            dtype=weight_values.dtype,
        ).reshape(weight_values.shape)
        if len(angles_array) != len(weights_array):
            raise ValueError("Angles and weights must have the same length.")

        angles_array.setflags(write=False)
        weights_array.setflags(write=False)
        object.__setattr__(self, "angles", angles_array)
        object.__setattr__(self, "weights", weights_array)
        object.__setattr__(self, "half_circle", half_circle)

    def __repr__(self):
        return f"Angles({self.angles}, weights={self.weights})"

    def __reduce__(
        self,
    ) -> tuple[type[Angles], tuple[np.ndarray, np.ndarray, bool]]:
        """Reconstruct through ``__init__`` to preserve read-only arrays."""
        return type(self), (self.angles, self.weights, self.half_circle)

    def __len__(self):
        return len(self.angles)

    @staticmethod
    def sparse(number: int, half_circle: bool = False) -> Angles:
        r"""
        Store a list of angles with all weights set to 1.

        :param number: Number of angles.
        :param half_circle: If False (the default), angles are in :math:`[0, 2\pi)`.
            Otherwise, angles are in :math:`[0, \pi)`.
        :return: Angles object with unitary weights.
        """
        max_angle = np.pi if half_circle else 2 * np.pi
        angles = np.linspace(0, max_angle, number, endpoint=False)
        weights = np.ones_like(angles)
        return Angles(angles=angles, weights=weights, half_circle=half_circle)

    @staticmethod
    def uniform(number: int, half_circle: bool = False) -> Angles:
        r"""
        Generate uniformly distributed and weighted angles.

        :param number: Number of angles.
        :param half_circle: If False (the default), angles are in :math:`[0, 2\pi)`.
            Otherwise, angles are in :math:`[0, \pi)`.
        :return: Angles object with uniform angles and weights.
        """
        max_angle = np.pi if half_circle else 2 * np.pi
        angles = Angles.sparse(number, half_circle=half_circle)
        weights = angles.weights * (max_angle / number)
        return Angles(
            angles=angles.angles,
            weights=weights,
            half_circle=half_circle,
        )

    @staticmethod
    def uniform_interval(
        start: float,
        end: float,
        number: int,
        half_circle: bool = False,
    ) -> Angles:
        r"""
        Generate uniformly distributed and weighted angles in a specified interval.

        :param start: Start of the interval.
        :param end: End of the interval.
        :param number: Number of angles.
        :param half_circle: Angular periodicity metadata. If True, the sampling
            is marked as using :math:`\pi` periodicity; otherwise, it is marked
            as using :math:`2\pi` periodicity.
        :return: Angles object with uniform angles and weights.
        """
        delta = abs(end - start) / (2 * number)

        angles = np.linspace(start + delta, end - delta, number)
        weights = 2 * delta * np.ones_like(angles)
        return Angles(angles=angles, weights=weights, half_circle=half_circle)

    @staticmethod
    def intervals(
        number_list: list[int],
        start_list: list[float],
        end_list: list[float],
        half_circle: bool = False,
    ) -> Angles:
        r"""
        Create angles uniformly distributed across multiple intervals.

        :param number_list: List of numbers of angles for each interval.
        :param start_list: List of start points for each interval.
        :param end_list: List of end points for each interval.
        :param half_circle: Angular periodicity metadata. If True, the sampling
            is marked as using :math:`\pi` periodicity; otherwise, it is marked
            as using :math:`2\pi` periodicity.
        :return: Angles object with angles and weights across all intervals.
        """
        if not len(number_list) == len(start_list) == len(end_list):
            raise ValueError("All input lists must have the same length.")

        angles: list[float] = []
        weights: list[float] = []

        for n, start, end in zip(number_list, start_list, end_list):
            interval_angles = Angles.uniform_interval(
                start, end, n, half_circle=half_circle
            )
            angles.extend(interval_angles.angles)
            weights.extend(interval_angles.weights)

        return Angles(
            angles=np.array(angles), weights=np.array(weights), half_circle=half_circle
        )

    @staticmethod
    def from_list(
        angles: npt.ArrayLike,
        half_circle: bool = False,
    ) -> Angles:
        r"""
        Automatically compute *natural* weights for a given list of angles.

        :param angles: List of angles.
        :param half_circle: If False (the default), angles are in :math:`[0, 2\pi)`.
            Otherwise, angles are in :math:`[0, \pi)`.
        :return: Angles object with unitary weights.
        """
        angles = np.asarray(angles)
        max_angle = np.pi if half_circle else 2 * np.pi

        angles_index = np.argsort(angles % max_angle)
        angles_sorted = angles[angles_index] % max_angle
        angles_extended = np.array(
            np.hstack(
                [
                    -max_angle + angles_sorted[-1],
                    angles_sorted,
                    angles_sorted[0] + max_angle,
                ]
            )
        )
        na = len(angles_sorted)
        angle_weights = 0.5 * (abs(angles_extended[2 : na + 2] - angles_extended[0:na]))

        # Correct for multiple occurrence of angles, for example
        # angles in [0,2pi] are considered instead of [0,pi]
        # and mod pi has same value) The weight of the same angles
        # is distributed equally.
        tol = 0.000001
        na = len(angles_sorted)
        i = 0
        while i < na - 1:
            count = 1
            my_sum = angle_weights[i]
            while abs(angles_sorted[i] - angles_sorted[i + count]) < tol:
                my_sum += angle_weights[i + count]
                count += 1
                if i + count > na - 1:
                    break

            val = my_sum / count
            for j in range(i, i + count):
                angle_weights[j] = val
            i += count

        angle_weights[angles_index] = angle_weights
        return Angles(angles=angles, weights=angle_weights, half_circle=half_circle)


# TODO: write tests for reversed in particular
@dataclass(frozen=True)
class Detectors:
    """Detector discretization and physical placement.

    This class describes the detector line used by an operator:

    - ``number`` is the number of detector pixels,
    - ``extent`` is the physical detector width or an extent placeholder,
    - ``center`` shifts the detector along its detector axis,
    - ``reversed`` flips the detector orientation.

    **Parameters**

    ``number``:
        Number of detector pixels. Negative values are accepted as a shorthand
        for ``reversed=True`` with ``abs(number)`` detector pixels.
    ``extent``:
        Physical detector width or an :class:`ExtentPlaceholder`.
    ``center``:
        Physical shift of the detector center along the detector axis.
    ``reversed``:
        Whether the detector orientation is reversed. If omitted, the sign of
        ``number`` is used to infer the orientation.

    **Notes**

    Instances are immutable. Construct a new detector value when changing its
    discretization or physical placement.

    In the experimental Radon and Fanbeam operators, ``ExtentPlaceholder.FULL``
    and ``ExtentPlaceholder.VALID`` can be used here to resolve the detector extent
    from a fixed image extent. Passing placeholders for both detector and image
    extents at once is unsupported and raises :class:`NotImplementedError`.
    """

    number: int
    extent: float | ExtentPlaceholder = ExtentPlaceholder.FULL
    center: float = 0.0
    reversed: bool = False

    def __init__(
        self,
        number: int,
        extent: float | ExtentPlaceholder = ExtentPlaceholder.FULL,
        center: float = 0.0,
        reversed: bool | None = None,
    ):
        object.__setattr__(self, "number", abs(number))
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "reversed", number < 0 if reversed is None else reversed)


@dataclass(frozen=True)
class ImageDomain:
    """Image grid and physical image extent.

    This class bundles the discrete image shape together with the physical
    extent and center shift used by an operator.

    **Parameters**

    ``size``:
        Image grid size. An integer is interpreted as a square domain of shape
        ``(N, N)``; a tuple specifies ``(Nx, Ny)`` directly.
    ``extent``:
        Physical extent of the image domain or an :class:`ExtentPlaceholder`.
        In gratopy's conventions, this corresponds to the longer side length of
        the rectangular image domain.
    ``center``:
        Physical shift of the image center relative to the rotation center.

    **Notes**

    Instances are immutable. Construct a new image-domain value when changing
    its grid, extent, or center.

    In the experimental Radon and Fanbeam operators, ``ExtentPlaceholder.FULL``
    and ``ExtentPlaceholder.VALID`` can be used here to resolve the image extent
    from a fixed detector extent. Passing placeholders for both image and
    detector extents at once is unsupported and raises
    :class:`NotImplementedError`.
    """

    size: tuple[int, int]
    extent: float | ExtentPlaceholder = 2.0
    center: tuple[float, float] = (0.0, 0.0)

    def __init__(
        self,
        size: int | tuple[int, int],
        extent: float | ExtentPlaceholder = 2.0,
        center: tuple[float, float] = (0.0, 0.0),
    ):
        if isinstance(size, int):
            size = (size, size)
        else:
            size = (size[0], size[1])
        center = (center[0], center[1])
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "center", center)


# ---------------------------------------------------------------------------
# Halfcircle geometry formulas (phi in [0, pi])
# ---------------------------------------------------------------------------


def _solve_quadratic(a: float, b: float, d: float) -> tuple[float, float]:
    """Solve ax^2 + bx + d = 0 and return (x1, x2) with x1 <= x2.

    Returns (nan, nan) when no real roots exist; callers handle this via
    the downstream NaN comparisons (always False).
    """
    discriminant = b**2 - 4 * a * d
    if discriminant < 0:
        return (float("nan"), float("nan"))
    sqrt_disc = np.sqrt(discriminant)
    x1 = (-b - sqrt_disc) / (2 * a)
    x2 = (-b + sqrt_disc) / (2 * a)
    return (x1, x2)


def _select_extent_from_linear_case(
    lower_root: float, upper_root: float, linear_bound: float
) -> float | None:
    """Select the image extent in one of the linear half-circle cases."""
    if upper_root <= linear_bound:
        return upper_root
    if lower_root <= linear_bound:
        return linear_bound
    return None


def _point_image_fits_detector_halfcircle(
    M: tuple[float, float, float], Dd: float
) -> bool:
    """Return whether a point image at ``(Mx, My)`` fits into the detector."""
    (Md, Mx, My) = M
    # Half-circle point-image feasibility: s_center(phi) over phi in [0, pi]
    # ranges over [-|My|, r] for Mx>=0 and [-r, |My|] for Mx<0, where
    # r = sqrt(Mx^2 + My^2). Both endpoints must lie inside [Md-Dd/2, Md+Dd/2].
    r = np.sqrt(Mx**2 + My**2)
    s = 1.0 if Mx >= 0 else -1.0
    return r <= s * Md + Dd / 2 and abs(My) <= Dd / 2 - s * Md


def _select_extent_from_case_a(
    x1: float, x2: float, y1: float, y2: float
) -> float | None:
    """Select the image extent where both sqrt constraints apply."""
    if (y2 <= x2) and (x1 <= y2):
        return y2
    if (x2 <= y2) and (y1 <= x2):
        return x2
    return None


def full_detector_given_image_halfcircle(
    M: tuple[float, float, float],
    D: tuple[float, float],
) -> float:
    """Smallest detector width so every ray through the image hits the detector.

    Half-circle variant (phi in [0, pi]).  See PDF Section 3.1, Eq. (5).

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    D : (Dx, Dy)
        Physical image dimensions.

    Returns
    -------
    float
        Required detector width Dd.
    """
    (Md, Mx, My) = M
    (Dx, Dy) = D
    # Coordinate rotation: the Max/Min formulas (1)-(2) were derived for
    # f(phi) = a cos(phi) + b sin(phi), while the sinogram convention is
    # s(phi) = x sin(phi) - y cos(phi).
    (Mx, My) = (-My, Mx)
    (Dx, Dy) = (Dy, Dx)

    if My >= -Dy / 2:
        MAX = np.sqrt((abs(Mx) + Dx / 2) ** 2 + (My + Dy / 2) ** 2)
    else:
        MAX = abs(Mx) + Dx / 2

    if My <= Dy / 2:
        MIN = -np.sqrt((abs(Mx) + Dx / 2) ** 2 + (My - Dy / 2) ** 2)
    else:
        MIN = -abs(Mx) - Dx / 2

    Dd = 2 * max(Md - MIN, MAX - Md)
    return Dd


def full_image_given_detector_halfcircle(
    M: tuple[float, float, float],
    Dd: float,
    c: float = 1.0,
) -> tuple[float, float] | None:
    """Largest image so every ray through it hits the detector.

    Half-circle variant (phi in [0, pi]).  See PDF Section 3.2, Eqs. (6)-(12).

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    Dd : float
        Detector width.
    c : float
        Aspect ratio Dx / Dy.

    Returns
    -------
    tuple[float, float] or None
        Image dimensions (Dx, Dy), or None if no valid geometry exists.
    """
    (Md, Mx, My) = M

    if not _point_image_fits_detector_halfcircle(M, Dd):
        return None

    a1 = (1 + c**2) / (4 * c**2)
    b1 = abs(My) / c + Mx
    d1 = -((Md + Dd / 2) ** 2) + Mx**2 + My**2
    (x1, x2) = _solve_quadratic(a1, b1, d1)

    a2 = (1 + c**2) / (4 * c**2)
    b2 = abs(My) / c - Mx
    d2 = -((Md - Dd / 2) ** 2) + Mx**2 + My**2
    (y1, y2) = _solve_quadratic(a2, b2, d2)

    # Case A: the image interval crosses the rotation center in x-direction
    # (Dx >= 2|Mx|), so both extrema are given by the sqrt formulas (1)-(2).
    Dx = _select_extent_from_case_a(x1, x2, y1, y2)

    if Dx is not None and Dx >= 2 * abs(Mx):
        return (Dx, Dx / c)

    # Case B: the image lies to the negative x side (Mx < 0, Dx < 2|Mx|),
    # so the upper detector constraint becomes linear.
    if Mx < 0:
        z = (Md + Dd / 2 - abs(My)) * 2 * c
        Dx = _select_extent_from_linear_case(y1, y2, z)
    # Case C: the image lies to the positive x side (Mx > 0, Dx < 2Mx),
    # so the lower detector constraint becomes linear.
    elif Mx > 0:
        z = (Dd / 2 - Md - abs(My)) * 2 * c
        Dx = _select_extent_from_linear_case(x1, x2, z)
    else:
        # Mx == 0: rectangle always straddles x=0, Case A is the only path.
        return None

    if Dx is None or Dx < 0:
        return None

    Dy = Dx / c
    return (Dx, Dy)


def valid_image_given_detector_halfcircle(
    M: tuple[float, float, float],
    Dd: float,
    c: float = 1.0,
) -> tuple[float, float]:
    """Smallest image so every ray hitting the detector passes through it.

    Half-circle variant (phi in [0, pi]).  See PDF Section 3.3, Eqs. (13)-(14).

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    Dd : float
        Detector width.
    c : float
        Aspect ratio Dx / Dy.

    Returns
    -------
    tuple[float, float]
        Image dimensions (Dx, Dy).
    """
    (Md, Mx, My) = M

    Dx = 2 * max(Mx - Md + Dd / 2, -Mx + Md + Dd / 2)
    Dy = 2 * max(abs(Md) + Dd / 2 - My, abs(Md) + Dd / 2 + My)

    Dx = max(Dx, c * Dy)
    Dy = Dx / c
    return (Dx, Dy)


def valid_detector_given_image_halfcircle(
    M: tuple[float, float, float],
    D: tuple[float, float],
) -> float | None:
    """Largest detector so every ray hitting it passes through the image.

    Half-circle variant (phi in [0, pi]).  See PDF Section 3.4, Eq. (15).

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    D : (Dx, Dy)
        Physical image dimensions.

    Returns
    -------
    float or None
        Detector width Dd, or None if no valid geometry exists.
    """
    (Md, Mx, My) = M
    (Dx, Dy) = D

    Dd = 2 * min(
        -Mx + Dx / 2 + Md,
        Mx - Md + Dx / 2,
        My - abs(Md) + Dy / 2,
        -My - abs(Md) + Dy / 2,
    )
    if Dd <= 0:
        return None
    return Dd


# ---------------------------------------------------------------------------
# Full-circle variants (phi in [0, 2*pi]).
#
# Over the full circle, max_phi s(x, y, phi) = +sqrt(x^2 + y^2) and
# min_phi s(x, y, phi) = -sqrt(x^2 + y^2) for every point, with no case
# distinction.  This removes the corner/branch bookkeeping of the half-circle
# formulas and, crucially, makes the detector offset enter only through
# abs(Md): the detector-axis sign convention that affects the half-circle
# variants is irrelevant here.
# ---------------------------------------------------------------------------


def full_detector_given_image_fullcircle(
    M: tuple[float, float, float],
    D: tuple[float, float],
) -> float:
    """Smallest detector width so every ray through the image hits the detector.

    Full-circle variant (phi in [0, 2*pi]).  See PDF Section 4.1.

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    D : (Dx, Dy)
        Physical image dimensions.

    Returns
    -------
    float
        Required detector width Dd.
    """
    (Md, Mx, My) = M
    (Dx, Dy) = D
    # No coordinate rotation is needed (unlike the half-circle variant):
    # over the full circle the extremal projection is +/- the corner radius.
    MAX = np.sqrt((abs(Mx) + Dx / 2) ** 2 + (abs(My) + Dy / 2) ** 2)
    Dd = 2 * (MAX + abs(Md))
    return Dd


def full_image_given_detector_fullcircle(
    M: tuple[float, float, float],
    Dd: float,
    c: float = 1.0,
) -> tuple[float, float] | None:
    """Largest image so every ray through it hits the detector.

    Full-circle variant (phi in [0, 2*pi]).  See PDF Section 4.2.

    The constraint sqrt((|Mx| + Dx/2)^2 + (|My| + Dy/(2c))^2) <= Dd/2 - |Md|
    reduces to a single quadratic in Dx (with Dy = Dx / c).

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    Dd : float
        Detector width.
    c : float
        Aspect ratio Dx / Dy.

    Returns
    -------
    tuple[float, float] or None
        Image dimensions (Dx, Dy), or None if no valid geometry exists.
    """
    (Md, Mx, My) = M

    radius = Dd / 2 - abs(Md)
    if radius <= 0:
        return None

    a = (1 + c**2) / (4 * c**2)
    b = abs(Mx) + abs(My) / c
    d = Mx**2 + My**2 - radius**2

    # The larger root is the largest feasible Dx; _solve_quadratic returns
    # (nan, nan) when no real root exists, which the guard below rejects.
    (_, Dx) = _solve_quadratic(a, b, d)
    if not np.isfinite(Dx) or Dx <= 0:
        return None

    Dy = Dx / c
    return (Dx, Dy)


def valid_image_given_detector_fullcircle(
    M: tuple[float, float, float],
    Dd: float,
    c: float = 1.0,
) -> tuple[float, float]:
    """Smallest image so every ray hitting the detector passes through it.

    Full-circle variant (phi in [0, 2*pi]).  See PDF Section 4.3.

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    Dd : float
        Detector width.
    c : float
        Aspect ratio Dx / Dy.

    Returns
    -------
    tuple[float, float]
        Image dimensions (Dx, Dy).
    """
    (Md, Mx, My) = M

    Dx = 2 * (abs(Mx) + abs(Md) + Dd / 2)
    Dy = 2 * (abs(My) + abs(Md) + Dd / 2)

    Dx = max(Dx, c * Dy)
    Dy = Dx / c
    return (Dx, Dy)


def valid_detector_given_image_fullcircle(
    M: tuple[float, float, float],
    D: tuple[float, float],
) -> float | None:
    """Largest detector so every ray hitting it passes through the image.

    Full-circle variant (phi in [0, 2*pi]).  See PDF Section 4.4.

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    D : (Dx, Dy)
        Physical image dimensions.

    Returns
    -------
    float or None
        Detector width Dd, or None if no valid geometry exists.
    """
    (Md, Mx, My) = M
    (Dx, Dy) = D

    Dd = 2 * min(
        Dx / 2 - abs(Md) - abs(Mx),
        Dy / 2 - abs(Md) - abs(My),
    )
    if Dd <= 0:
        return None
    return Dd


# ---------------------------------------------------------------------------
# Fan-beam variants (full circle, phi in [0, 2*pi]).
#
# The source lies at distance RE from the rotation center and the flat
# detector at orthogonal distance R from the source.  A ray hitting the
# detector at position t has the parallel-beam offset
#     s = RE * t / sqrt(R^2 + t^2),     inversely  t = R * s / sqrt(RE^2 - s^2).
# Both maps are odd and strictly increasing, so the full-circle parallel-beam
# reasoning carries over by mapping the relevant offsets between s and t.
# As in the parallel-beam full-circle case, the detector offset enters only
# through abs(Md).
# ---------------------------------------------------------------------------


def _fanbeam_s_to_t(s: float, RE: float, R: float) -> float:
    """Map a parallel-beam offset s to the fan-beam detector position t."""
    return R * s / np.sqrt(RE**2 - s**2)


def _fanbeam_t_to_s(t: float, RE: float, R: float) -> float:
    """Map a fan-beam detector position t to the parallel-beam offset s."""
    return RE * t / np.sqrt(R**2 + t**2)


def full_detector_given_image_fanbeam(
    M: tuple[float, float, float],
    D: tuple[float, float],
    RE: float,
    R: float,
) -> float | None:
    """Smallest detector width so every ray through the image hits the detector.

    Fan-beam variant (full circle).  With the squared corner radius
    Z = (|Mx| + Dx/2)^2 + (|My| + Dy/2)^2 the result is
    Dd = 2 * (sqrt(Z * R^2 / (RE^2 - Z)) + |Md|).

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    D : (Dx, Dy)
        Physical image dimensions.
    RE : float
        Distance from the source to the rotation center.
    R : float
        Orthogonal distance from the source to the detector line.

    Returns
    -------
    float or None
        Required detector width Dd, or None if the image reaches the
        source circle (Z >= RE^2).
    """
    (Md, Mx, My) = M
    (Dx, Dy) = D

    Z = (abs(Mx) + Dx / 2) ** 2 + (abs(My) + Dy / 2) ** 2
    if Z >= RE**2:
        return None

    Dd = 2 * (_fanbeam_s_to_t(np.sqrt(Z), RE, R) + abs(Md))
    return Dd


def full_image_given_detector_fanbeam(
    M: tuple[float, float, float],
    Dd: float,
    RE: float,
    R: float,
    c: float = 1.0,
) -> tuple[float, float] | None:
    """Largest image so every ray through it hits the detector.

    Fan-beam variant (full circle).  The detector covers the parallel-beam
    offsets |s| <= Reff = s(Dd/2 - |Md|), so the constraint
    sqrt((|Mx| + Dx/2)^2 + (|My| + Dy/(2c))^2) <= Reff reduces to a single
    quadratic in Dx (with Dy = Dx / c).

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    Dd : float
        Detector width.
    RE : float
        Distance from the source to the rotation center.
    R : float
        Orthogonal distance from the source to the detector line.
    c : float
        Aspect ratio Dx / Dy.

    Returns
    -------
    tuple[float, float] or None
        Image dimensions (Dx, Dy), or None if no valid geometry exists.
    """
    (Md, Mx, My) = M

    half_width = Dd / 2 - abs(Md)
    if half_width <= 0:
        return None
    radius = _fanbeam_t_to_s(half_width, RE, R)

    a = (1 + c**2) / (4 * c**2)
    b = abs(Mx) + abs(My) / c
    d = Mx**2 + My**2 - radius**2

    (_, Dx) = _solve_quadratic(a, b, d)
    if not np.isfinite(Dx) or Dx <= 0:
        return None

    Dy = Dx / c
    return (Dx, Dy)


def valid_image_given_detector_fanbeam(
    M: tuple[float, float, float],
    Dd: float,
    RE: float,
    R: float,
    c: float = 1.0,
) -> tuple[float, float]:
    """Smallest image so every ray hitting the detector passes through it.

    Fan-beam variant (full circle).  The rays hitting the detector have
    parallel-beam offsets up to s_max = s(|Md| + Dd/2) in absolute value.

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    Dd : float
        Detector width.
    RE : float
        Distance from the source to the rotation center.
    R : float
        Orthogonal distance from the source to the detector line.
    c : float
        Aspect ratio Dx / Dy.

    Returns
    -------
    tuple[float, float]
        Image dimensions (Dx, Dy).
    """
    (Md, Mx, My) = M

    s_max = _fanbeam_t_to_s(abs(Md) + Dd / 2, RE, R)

    Dx = 2 * (abs(Mx) + s_max)
    Dy = 2 * (abs(My) + s_max)

    Dx = max(Dx, c * Dy)
    Dy = Dx / c
    return (Dx, Dy)


def valid_detector_given_image_fanbeam(
    M: tuple[float, float, float],
    D: tuple[float, float],
    RE: float,
    R: float,
) -> float | None:
    """Largest detector so every ray hitting it passes through the image.

    Fan-beam variant (full circle).  With L = min(Dx/2 - |Mx|, Dy/2 - |My|)
    the result is Dd = 2 * (L * R / sqrt(RE^2 - L^2) - |Md|).

    Parameters
    ----------
    M : (Md, Mx, My)
        Detector center offset and image center coordinates.
    D : (Dx, Dy)
        Physical image dimensions.
    RE : float
        Distance from the source to the rotation center.
    R : float
        Orthogonal distance from the source to the detector line.

    Returns
    -------
    float or None
        Detector width Dd, or None if no valid geometry exists.
    """
    (Md, Mx, My) = M
    (Dx, Dy) = D

    L = min(Dx / 2 - abs(Mx), Dy / 2 - abs(My))
    if L <= 0 or L >= RE:
        return None

    Dd = 2 * (_fanbeam_s_to_t(L, RE, R) - abs(Md))
    if Dd <= 0:
        return None
    return Dd
