"""Tests for ExtentPlaceholder resolution in the Radon operator.

These tests verify that ExtentPlaceholder.FULL and ExtentPlaceholder.VALID
are correctly resolved to concrete float values when constructing a Radon
operator with one placeholder extent and one fixed extent.

The tests cover both directions:
- Fixed image extent, placeholder on the detector side.
- Fixed detector extent, placeholder on the image side.

Each direction is exercised across a range of image and detector center
offsets and across image side ratios Ny/Nx in {0.7, 1.0, 1.7} to cover
non-square image domains (c = Nx/Ny != 1).

Reference values for the c=1 cases were computed independently by hand;
the c!=1 cases were captured from the implementation after visual
verification of the underlying radon transforms, so they act as
regression guards. Comparison uses an absolute tolerance of 0.02.
"""

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

SIDE_RATIOS = [0.7, 1.0, 1.7]


# -- Detector placeholder with fixed image extent ---------------------------

DETECTOR_PLACEHOLDER_EXPECTED = {
    (ExtentPlaceholder.VALID, 0.7): [
        # detector_center=0
        1.000, 1.000, 1.000, 1.000,
        # detector_center=0.2
        0.600, 0.600, 0.600, 0.600,
        # detector_center=-0.2
        0.600, 0.600, 0.600, 0.600,
        # detector_center=0.5
        None, None, None, None,
        # detector_center=-0.5
        None, None, None, None,
    ],
    (ExtentPlaceholder.VALID, 1.0): [
        # detector_center=0
        1.000, 1.000, 1.000, 1.000,
        # detector_center=0.2
        1.200, 0.600, 1.200, 0.600,
        # detector_center=-0.2
        0.600, 1.200, 0.600, 1.200,
        # detector_center=0.5
        0.600, None, 0.600, None,
        # detector_center=-0.5
        None, 0.600, None, 0.600,
    ],
    (ExtentPlaceholder.VALID, 1.7): [
        # detector_center=0
        0.176, 0.176, 0.176, 0.176,
        # detector_center=0.2
        0.576, None, 0.576, None,
        # detector_center=-0.2
        None, 0.576, None, 0.576,
        # detector_center=0.5
        0.600, None, 0.600, None,
        # detector_center=-0.5
        None, 0.600, None, 0.600,
    ],
    (ExtentPlaceholder.FULL, 0.7): [
        # detector_center=0
        3.499, 3.499, 3.499, 3.499,
        # detector_center=0.2
        3.099, 3.899, 3.099, 3.899,
        # detector_center=-0.2
        3.899, 3.099, 3.899, 3.099,
        # detector_center=0.5
        3.059, 4.499, 3.059, 4.499,
        # detector_center=-0.5
        4.499, 3.059, 4.499, 3.059,
    ],
    (ExtentPlaceholder.FULL, 1.0): [
        # detector_center=0
        3.842, 3.842, 3.842, 3.842,
        # detector_center=0.2
        3.442, 4.242, 3.442, 4.242,
        # detector_center=-0.2
        4.242, 3.442, 4.242, 3.442,
        # detector_center=0.5
        3.600, 4.842, 3.600, 4.842,
        # detector_center=-0.5
        4.842, 3.600, 4.842, 3.600,
    ],
    (ExtentPlaceholder.FULL, 1.7): [
        # detector_center=0
        3.240, 3.240, 3.240, 3.240,
        # detector_center=0.2
        2.840, 3.640, 2.840, 3.640,
        # detector_center=-0.2
        3.640, 2.840, 3.640, 2.840,
        # detector_center=0.5
        3.406, 4.240, 3.406, 4.240,
        # detector_center=-0.5
        4.240, 3.406, 4.240, 3.406,
    ],
}


def _detector_placeholder_cases():
    """Yield (placeholder, side_ratio, detector_center, image_center, expected)."""
    for placeholder in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]:
        for sr in SIDE_RATIOS:
            expected_list = list(DETECTOR_PLACEHOLDER_EXPECTED[(placeholder, sr)])
            idx = 0
            for dc in DETECTOR_CENTERS:
                for ic in IMAGE_CENTERS:
                    yield placeholder, sr, dc, ic, expected_list[idx]
                    idx += 1


@pytest.mark.parametrize(
    "placeholder, side_ratio, detector_center, image_center, expected",
    list(_detector_placeholder_cases()),
    ids=[
        f"{p.name}-sr{sr}-dc{dc}-ic{ic}"
        for p in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]
        for sr in SIDE_RATIOS
        for dc in DETECTOR_CENTERS
        for ic in IMAGE_CENTERS
    ],
)
def test_detector_extent_placeholder(
    placeholder, side_ratio, detector_center, image_center, expected
):
    """Resolve a detector ExtentPlaceholder with a fixed image extent of 2.0.

    Verifies that the resolved detector extent matches the reference value.
    Cases where no valid geometry exists (expected is None) must raise a
    ValueError.
    """
    Ny = int(side_ratio * Nx)
    if expected is None:
        with pytest.raises(ValueError):
            Radon(
                image_domain=ImageDomain(
                    size=(Nx, Ny), center=image_center, extent=2.0
                ),
                angles=Na,
                detectors=Detectors(
                    number=Ns, center=detector_center, extent=placeholder
                ),
            )
    else:
        radon = Radon(
            image_domain=ImageDomain(
                size=(Nx, Ny), center=image_center, extent=2.0
            ),
            angles=Na,
            detectors=Detectors(
                number=Ns, center=detector_center, extent=placeholder
            ),
        )
        assert isinstance(radon.detectors.extent, float)
        assert radon.detectors.extent == pytest.approx(expected, abs=0.02), (
            f"Detector extent mismatch for {placeholder.name} with "
            f"side_ratio={side_ratio}, detector_center={detector_center}, "
            f"image_center={image_center}: "
            f"expected {expected}, got {radon.detectors.extent}"
        )


# -- Image placeholder with fixed detector extent ---------------------------

IMAGE_PLACEHOLDER_EXPECTED = {
    (ExtentPlaceholder.VALID, 0.7): [
        # detector_center=0
        3.429, 3.429, 3.429, 3.429,
        # detector_center=0.2
        4.000, 4.000, 4.000, 4.000,
        # detector_center=-0.2
        4.000, 4.000, 4.000, 4.000,
        # detector_center=0.5
        4.857, 4.857, 4.857, 4.857,
        # detector_center=-0.5
        4.857, 4.857, 4.857, 4.857,
    ],
    (ExtentPlaceholder.VALID, 1.0): [
        # detector_center=0
        3.000, 3.000, 3.000, 3.000,
        # detector_center=0.2
        2.800, 3.400, 2.800, 3.400,
        # detector_center=-0.2
        3.400, 2.800, 3.400, 2.800,
        # detector_center=0.5
        3.400, 4.000, 3.400, 4.000,
        # detector_center=-0.5
        4.000, 3.400, 4.000, 3.400,
    ],
    (ExtentPlaceholder.VALID, 1.7): [
        # detector_center=0
        5.100, 5.100, 5.100, 5.100,
        # detector_center=0.2
        4.420, 5.780, 4.420, 5.780,
        # detector_center=-0.2
        5.780, 4.420, 5.780, 4.420,
        # detector_center=0.5
        3.400, 6.800, 3.400, 6.800,
        # detector_center=-0.5
        6.800, 3.400, 6.800, 3.400,
    ],
    (ExtentPlaceholder.FULL, 0.7): [
        # detector_center=0
        0.767, 0.767, 0.767, 0.767,
        # detector_center=0.2
        1.097, 0.436, 1.097, 0.436,
        # detector_center=-0.2
        0.436, 1.097, 0.436, 1.097,
        # detector_center=0.5
        0.857, None, 0.857, None,
        # detector_center=-0.5
        None, 0.857, None, 0.857,
    ],
    (ExtentPlaceholder.FULL, 1.0): [
        # detector_center=0
        0.682, 0.682, 0.682, 0.682,
        # detector_center=0.2
        0.970, 0.391, 0.970, 0.391,
        # detector_center=-0.2
        0.391, 0.970, 0.391, 0.970,
        # detector_center=0.5
        0.600, None, 0.600, None,
        # detector_center=-0.5
        None, 0.600, None, 0.600,
    ],
    (ExtentPlaceholder.FULL, 1.7): [
        # detector_center=0
        0.893, 0.893, 0.893, 0.893,
        # detector_center=0.2
        1.200, 0.522, 1.200, 0.522,
        # detector_center=-0.2
        0.522, 1.200, 0.522, 1.200,
        # detector_center=0.5
        0.600, None, 0.600, None,
        # detector_center=-0.5
        None, 0.600, None, 0.600,
    ],
}


def _image_placeholder_cases():
    """Yield (placeholder, side_ratio, detector_center, image_center, expected)."""
    for placeholder in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]:
        for sr in SIDE_RATIOS:
            expected_list = list(IMAGE_PLACEHOLDER_EXPECTED[(placeholder, sr)])
            idx = 0
            for dc in DETECTOR_CENTERS:
                for ic in IMAGE_CENTERS:
                    yield placeholder, sr, dc, ic, expected_list[idx]
                    idx += 1


@pytest.mark.parametrize(
    "placeholder, side_ratio, detector_center, image_center, expected",
    list(_image_placeholder_cases()),
    ids=[
        f"{p.name}-sr{sr}-dc{dc}-ic{ic}"
        for p in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]
        for sr in SIDE_RATIOS
        for dc in DETECTOR_CENTERS
        for ic in IMAGE_CENTERS
    ],
)
def test_image_extent_placeholder(
    placeholder, side_ratio, detector_center, image_center, expected
):
    """Resolve an image ExtentPlaceholder with a fixed detector extent of 2.0.

    Verifies that the resolved image extent matches the reference value.
    Cases where no valid geometry exists (expected is None) must raise a
    ValueError.
    """
    Ny = int(side_ratio * Nx)
    if expected is None:
        with pytest.raises(ValueError):
            Radon(
                image_domain=ImageDomain(
                    size=(Nx, Ny), center=image_center, extent=placeholder
                ),
                angles=Na,
                detectors=Detectors(
                    number=Ns, center=detector_center, extent=2.0
                ),
            )
    else:
        radon = Radon(
            image_domain=ImageDomain(
                size=(Nx, Ny), center=image_center, extent=placeholder
            ),
            angles=Na,
            detectors=Detectors(
                number=Ns, center=detector_center, extent=2.0
            ),
        )
        assert isinstance(radon.image_domain.extent, float)
        assert radon.image_domain.extent == pytest.approx(expected, abs=0.02), (
            f"Image extent mismatch for {placeholder.name} with "
            f"side_ratio={side_ratio}, detector_center={detector_center}, "
            f"image_center={image_center}: "
            f"expected {expected}, got {radon.image_domain.extent}"
        )
