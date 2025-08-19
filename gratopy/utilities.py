"""Miscellaneous utility functions for gratopy."""

from __future__ import annotations

import numpy as np

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
    def __init__(self, angles: np.ndarray, weights: np.ndarray):
        if len(angles) != len(weights):
            raise ValueError("Angles and weights must have the same length.")
        self.angles = np.asarray(angles)
        self.weights = np.asarray(weights)

    def __repr__(self):
        return f"Angles({self.angles}, weights={self.weights})"
    
    def __len__(self):
        return len(self.angles)
    
    @staticmethod
    def uniform(number: int, geometry: GeometryType) -> Angles:
        """
        Generate uniformly distributed and weighted angles.

        :param number: Number of angles.
        :param geometry: The underlying projection geometry.
        :return: Angles object with uniform angles and weights.
        """
        geometry = GeometryType(geometry)
        if geometry == GeometryType.RADON:
            max_angle = np.pi
        elif geometry == GeometryType.FANBEAM:
            max_angle = 2 * np.pi
        else:
            raise NotImplementedError(
                f"Geometry type {geometry} is not implemented."
            )

        angles = np.linspace(0, max_angle, number + 1)[:-1]
        weights = np.ones_like(angles) * (max_angle / number)
        return Angles(angles=angles, weights=weights)
    
    @staticmethod
    def uniform_interval(
        number: int, start: float, end: float
    ) -> Angles:
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
    def uniform_multiple_intervals(
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
            raise ValueError(
                "All input lists must have the same length."
            )

        angles = []
        weights = []
        
        for n, start, end in zip(number_list, start_list, end_list):
            interval_angles = Angles.uniform_interval(n, start, end)
            angles.extend(interval_angles.angles)
            weights.extend(interval_angles.weights)

        return Angles(angles=np.array(angles), weights=np.array(weights))
    
    @staticmethod
    def with_unitary_weights(angles: np.ndarray) -> Angles:
        """
        Store a list of angles with all weights set to 1.

        :param angles: Angles in radians.
        :return: Angles object with unitary weights.
        """
        angles = np.asarray(angles)
        weights = np.ones_like(angles)
        return Angles(angles=angles, weights=weights)
