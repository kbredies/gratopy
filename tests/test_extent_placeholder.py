"""Tests for ExtentPlaceholder resolution in the Radon operator.

These tests verify that ExtentPlaceholder.FULL and ExtentPlaceholder.VALID
are correctly resolved to concrete float values when constructing a Radon
operator with one placeholder extent and one fixed extent.

The tests cover both directions:
- Fixed image extent, placeholder on the detector side.
- Fixed detector extent, placeholder on the image side.

Each direction is tested across a range of image and detector center
offsets to exercise the geometry formulas for off-center configurations.

Expected reference values were computed independently and are compared
with a tolerance of 0.02 to account for discretization effects.
"""

import numpy as np
import pytest
import pyopencl as cl

from gratopy.operator import Radon
from gratopy.utilities import Detectors, ExtentPlaceholder, ImageDomain


ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

Nx = 400
Ns = 100
Na = 180

IMAGE_CENTERS = [
    (+0.5, 0.2),
    (-0.5, 0.2),
    (0.5, -0.2),
    (-0.5, -0.2),
]

DETECTOR_CENTERS = [0, 0.2, -0.2, 0.5, -0.5]


# -- Detector placeholder with fixed image extent ---------------------------

DETECTOR_PLACEHOLDER_EXPECTED = {
    ExtentPlaceholder.VALID: [
        # detector_center=0
        1.0, 1.0, 1.0, 1.0,
        # detector_center=0.2
        1.2, 0.6, 1.2, 0.6,
        # detector_center=-0.2
        0.6, 1.2, 0.6, 1.2,
        # detector_center=0.5
        0.6, None, 0.6, None,
        # detector_center=-0.5
        None, 0.6, None, 0.6,
    ],
    ExtentPlaceholder.FULL: [
        # detector_center=0
        3.841, 3.841, 3.841, 3.841,
        # detector_center=0.2
        3.441, 4.241, 3.441, 4.241,
        # detector_center=-0.2
        4.241, 3.441, 4.241, 3.441,
        # detector_center=0.5
        3.6, 4.841, 3.6, 4.841,
        # detector_center=-0.5
        4.841, 3.6, 4.841, 3.6,
    ],
}


def _detector_placeholder_cases():
    """Generate (placeholder, detector_center, image_center, expected) tuples."""
    for placeholder in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]:
        expected_list = list(DETECTOR_PLACEHOLDER_EXPECTED[placeholder])
        idx = 0
        for dc in DETECTOR_CENTERS:
            for ic in IMAGE_CENTERS:
                yield placeholder, dc, ic, expected_list[idx]
                idx += 1


@pytest.mark.parametrize(
    "placeholder, detector_center, image_center, expected",
    list(_detector_placeholder_cases()),
    ids=[
        f"{p.name}-dc{dc}-ic{ic}"
        for p in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]
        for dc in DETECTOR_CENTERS
        for ic in IMAGE_CENTERS
    ],
)
def test_detector_extent_placeholder(
    placeholder, detector_center, image_center, expected
):
    """Resolve a detector ExtentPlaceholder with a fixed image extent of 2.0.

    Verifies that the resolved detector extent matches the independently
    computed reference value. Cases where no valid geometry exists (expected
    is None) must raise a ValueError.
    """
    if expected is None:
        with pytest.raises(ValueError):
            Radon(
                image_domain=ImageDomain(
                    size=Nx, center=image_center, extent=2.0
                ),
                angles=Na,
                detectors=Detectors(
                    number=Ns, center=detector_center, extent=placeholder
                ),
            )
    else:
        radon = Radon(
            image_domain=ImageDomain(
                size=Nx, center=image_center, extent=2.0
            ),
            angles=Na,
            detectors=Detectors(
                number=Ns, center=detector_center, extent=placeholder
            ),
        )
        assert isinstance(radon.detectors.extent, float)
        assert radon.detectors.extent == pytest.approx(expected, abs=0.02), (
            f"Detector extent mismatch for {placeholder.name} with "
            f"detector_center={detector_center}, image_center={image_center}: "
            f"expected {expected}, got {radon.detectors.extent}"
        )


# -- Image placeholder with fixed detector extent ---------------------------

IMAGE_PLACEHOLDER_EXPECTED = {
    ExtentPlaceholder.VALID: [
        # detector_center=0
        3.0, 3.0, 3.0, 3.0,
        # detector_center=0.2
        2.8, 3.4, 2.8, 3.4,
        # detector_center=-0.2
        3.4, 2.8, 3.4, 2.8,
        # detector_center=0.5
        3.4, 4.0, 3.4, 4.0,
        # detector_center=-0.5
        4.0, 3.4, 4.0, 3.4,
    ],
    ExtentPlaceholder.FULL: [
        # detector_center=0
        0.682, 0.682, 0.682, 0.682,
        # detector_center=0.2
        0.970, 0.390, 0.970, 0.390,
        # detector_center=-0.2
        0.390, 0.970, 0.390, 0.970,
        # detector_center=0.5
        0.6, None, 0.6, None,
        # detector_center=-0.5
        None, 0.6, None, 0.6,
    ],
}


def _image_placeholder_cases():
    """Generate (placeholder, detector_center, image_center, expected) tuples."""
    for placeholder in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]:
        expected_list = list(IMAGE_PLACEHOLDER_EXPECTED[placeholder])
        idx = 0
        for dc in DETECTOR_CENTERS:
            for ic in IMAGE_CENTERS:
                yield placeholder, dc, ic, expected_list[idx]
                idx += 1


@pytest.mark.parametrize(
    "placeholder, detector_center, image_center, expected",
    list(_image_placeholder_cases()),
    ids=[
        f"{p.name}-dc{dc}-ic{ic}"
        for p in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]
        for dc in DETECTOR_CENTERS
        for ic in IMAGE_CENTERS
    ],
)
def test_image_extent_placeholder(
    placeholder, detector_center, image_center, expected
):
    """Resolve an image ExtentPlaceholder with a fixed detector extent of 2.0.

    Verifies that the resolved image extent matches the independently
    computed reference value. Cases where no valid geometry exists (expected
    is None) must raise a ValueError.
    """
    if expected is None:
        with pytest.raises(ValueError):
            Radon(
                image_domain=ImageDomain(
                    size=Nx, center=image_center, extent=placeholder
                ),
                angles=Na,
                detectors=Detectors(
                    number=Ns, center=detector_center, extent=2.0
                ),
            )
    else:
        radon = Radon(
            image_domain=ImageDomain(
                size=Nx, center=image_center, extent=placeholder
            ),
            angles=Na,
            detectors=Detectors(
                number=Ns, center=detector_center, extent=2.0
            ),
        )
        assert isinstance(radon.image_domain.extent, float)
        assert radon.image_domain.extent == pytest.approx(expected, abs=0.02), (
            f"Image extent mismatch for {placeholder.name} with "
            f"detector_center={detector_center}, image_center={image_center}: "
            f"expected {expected}, got {radon.image_domain.extent}"
        )
