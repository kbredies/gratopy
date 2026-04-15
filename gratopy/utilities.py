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
    evolving. At the moment, placeholders should be considered unsupported in
    the operator API and may raise :class:`NotImplementedError`.
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


class Angles:
    """Angular sampling together with associated weights.

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

    **Notes**

    The static constructors on this class provide a few common ways of
    constructing angular samplings:

    - :meth:`sparse` for unweighted equispaced angles,
    - :meth:`uniform` for equispaced angles with uniform weights,
    - :meth:`uniform_interval` for one limited-angle interval,
    - :meth:`intervals` for multiple limited-angle intervals,
    - :meth:`from_list` for explicit angles with automatically inferred
      *natural* weights.
    """

    def __init__(self, angles: npt.ArrayLike, weights: npt.ArrayLike):
        angles = np.asarray(angles)
        weights = np.asarray(weights)
        if len(angles) != len(weights):
            raise ValueError("Angles and weights must have the same length.")
        self.angles = angles
        self.weights = weights

    def __repr__(self):
        return f"Angles({self.angles}, weights={self.weights})"

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
        return Angles(angles=angles, weights=weights)

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
        angles.weights = angles.weights * (max_angle / number)
        return angles

    @staticmethod
    def uniform_interval(start: float, end: float, number: int) -> Angles:
        """
        Generate uniformly distributed and weighted angles in a specified interval.

        :param start: Start of the interval.
        :param end: End of the interval.
        :param number: Number of angles.
        :return: Angles object with uniform angles and weights.
        """
        delta = abs(end - start) / (2 * number)

        angles = np.linspace(start + delta, end - delta, number)
        weights = 2 * delta * np.ones_like(angles)
        return Angles(angles=angles, weights=weights)

    @staticmethod
    def intervals(
        number_list: list[int], start_list: list[float], end_list: list[float]
    ) -> Angles:
        """
        Create angles uniformly distributed across multiple intervals.

        :param number_list: List of numbers of angles for each interval.
        :param start_list: List of start points for each interval.
        :param end_list: List of end points for each interval.
        :return: Angles object with angles and weights across all intervals.
        """
        if not len(number_list) == len(start_list) == len(end_list):
            raise ValueError("All input lists must have the same length.")

        angles = []
        weights = []

        for n, start, end in zip(number_list, start_list, end_list):
            interval_angles = Angles.uniform_interval(start, end, n)
            angles.extend(interval_angles.angles)
            weights.extend(interval_angles.weights)

        return Angles(angles=np.array(angles), weights=np.array(weights))

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
        return Angles(angles=angles, weights=angle_weights)


# TODO: write tests for reversed in particular
@dataclass
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

    In the current operator API, placeholder-based extent handling is not yet
    implemented and may raise :class:`NotImplementedError`.
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
        self.number = abs(number)
        self.extent = extent
        self.center = center

        if reversed is None:
            self.reversed = number < 0
        else:
            self.reversed = reversed


@dataclass
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

    In the experimental operator API, placeholders for extents are not yet
    implemented and may raise :class:`NotImplementedError`.
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
        self.size = size
        self.extent = extent
        self.center = center


# ---------------------------------------------------------------------------
# Halfcircle geometry formulas (phi in [0, pi])
# ---------------------------------------------------------------------------


def _solve_quadratic(a: float, b: float, d: float) -> tuple[float, float]:
    """Solve ax^2 + bx + d = 0 and return (x1, x2) with x1 <= x2."""
    discriminant = b**2 - 4 * a * d
    sqrt_disc = np.sqrt(discriminant)
    x1 = (-b - sqrt_disc) / (2 * a)
    x2 = (-b + sqrt_disc) / (2 * a)
    return (x1, x2)


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

    a1 = (1 + c**2) / (4 * c**2)
    b1 = abs(My) / c + Mx
    d1 = -(Md + Dd / 2) ** 2 + Mx**2 + My**2
    (x1, x2) = _solve_quadratic(a1, b1, d1)

    a2 = (1 + c**2) / (4 * c**2)
    b2 = abs(My) / c - Mx
    d2 = -(Md - Dd / 2) ** 2 + Mx**2 + My**2
    (y1, y2) = _solve_quadratic(a2, b2, d2)

    # Case A: Dx >= 2|Mx| — both sqrt formulas in (1)-(2) apply.
    if (y2 <= x2) and (x1 <= y2):
        Dx = y2
    elif (x2 <= y2) and (y1 <= x2):
        Dx = x2
    else:
        return None

    Dy = Dx / c

    if Dx >= 2 * abs(Mx):
        return (Dx, Dy)

    # Case B: Mx < 0, Dx < 2|Mx| — Max constraint becomes linear.
    if Mx < 0:
        z = (Md + Dd / 2 - abs(My)) * 2 * c
        if y2 <= z:
            Dx = y2
        elif y1 <= z:
            Dx = z
        else:
            Dx = None
    # Case C: Mx > 0, Dx < 2Mx — Min constraint becomes linear.
    elif Mx > 0:
        z = (Dd / 2 - Md - abs(My)) * 2 * c
        if x2 <= z:
            Dx = x2
        elif x1 <= z:
            Dx = z
        else:
            Dx = None

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
