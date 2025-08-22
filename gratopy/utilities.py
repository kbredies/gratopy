"""Miscellaneous utility functions for gratopy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from enum import Enum


class GeometryType(Enum):
    """
    Enum for the different geometry types.
    """

    RADON = "radon"
    FANBEAM = "fanbeam"


class Angles:
    """
    Utility class storing angles and their weights.

    :param angles: Angles in radians.
    :param weights: Weights for the given angles.
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
    def uniform_interval(number: int, start: float, end: float) -> Angles:
        """
        Generate uniformly distributed and weighted angles in a specified interval.

        :param number: Number of angles.
        :param start: Start of the interval.
        :param end: End of the interval.
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
            interval_angles = Angles.uniform_interval(n, start, end)
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
