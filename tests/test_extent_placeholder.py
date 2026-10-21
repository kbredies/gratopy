"""Tests for ExtentPlaceholder resolution in the Radon and Fanbeam operators.

These tests verify that ExtentPlaceholder.FULL and ExtentPlaceholder.VALID
are correctly resolved to concrete float values when constructing a Radon
operator with one placeholder extent and one fixed extent.

The tests cover both directions:
- Fixed image extent, placeholder on the detector side.
- Fixed detector extent, placeholder on the image side.

Each direction is exercised across a range of image and detector center
offsets and across image side ratios Ny/Nx in {0.7, 1.0, 1.7} to cover
non-square image domains (c = Nx/Ny != 1).

Both angular settings are covered:
- half circle [0, pi)  (angles given as an int -> half_circle=True), and
- full circle  [0, 2*pi) (angles given as Angles(half_circle=False)),
which resolve placeholders with the half- and full-circle geometry formulas
respectively. The two settings have their own reference-value tables.

Reference values for the half-circle c=1 cases were computed independently by
hand; the half-circle c!=1 cases and all full-circle cases were captured from
the implementation after visual verification of the underlying radon
transforms, so they act as regression guards. Over the full circle the
resolution depends only on |Mx|, |My|, |Md| (no sign / case distinction), so
all four image-center sign variants share the same expected value.
Comparison uses an absolute tolerance of 0.02.

The Fanbeam operator is exercised on the same grid of centers and side
ratios. Its placeholders are always resolved for the full circle, and the
resolved extents are checked by ray tracing instead of reference tables.
"""

import pytest

from gratopy.operator import Fanbeam, Radon
from gratopy.utilities import Angles, Detectors, ExtentPlaceholder, ImageDomain

from .helpers import fanbeam_detector_covers_image, fanbeam_image_meets_detector_rays


Nx = 400
Ns = 100
Na = 180

# Full-circle angular sampling: 180 angles equispaced over [0, 2*pi).
FULL_ANGLES = Angles.uniform(Na, half_circle=False)

IMAGE_CENTERS = [
    (+0.5, 0.2),
    (-0.5, 0.2),
    (0.5, -0.2),
    (-0.5, -0.2),
]

DETECTOR_CENTERS = [0, 0.2, -0.2, 0.5, -0.5]

SIDE_RATIOS = [0.7, 1.0, 1.7]


# -- Detector placeholder with fixed image extent ---------------------------

DETECTOR_PLACEHOLDER_EXPECTED_HALFCIRCLE = {
    (ExtentPlaceholder.VALID, 0.7): [
        # detector_center=0
        1.000,
        1.000,
        1.000,
        1.000,
        # detector_center=0.2
        0.600,
        0.600,
        0.600,
        0.600,
        # detector_center=-0.2
        0.600,
        0.600,
        0.600,
        0.600,
        # detector_center=0.5
        None,
        None,
        None,
        None,
        # detector_center=-0.5
        None,
        None,
        None,
        None,
    ],
    (ExtentPlaceholder.VALID, 1.0): [
        # detector_center=0
        1.000,
        1.000,
        1.000,
        1.000,
        # detector_center=0.2
        1.200,
        0.600,
        1.200,
        0.600,
        # detector_center=-0.2
        0.600,
        1.200,
        0.600,
        1.200,
        # detector_center=0.5
        0.600,
        None,
        0.600,
        None,
        # detector_center=-0.5
        None,
        0.600,
        None,
        0.600,
    ],
    (ExtentPlaceholder.VALID, 1.7): [
        # detector_center=0
        0.176,
        0.176,
        0.176,
        0.176,
        # detector_center=0.2
        0.576,
        None,
        0.576,
        None,
        # detector_center=-0.2
        None,
        0.576,
        None,
        0.576,
        # detector_center=0.5
        0.600,
        None,
        0.600,
        None,
        # detector_center=-0.5
        None,
        0.600,
        None,
        0.600,
    ],
    (ExtentPlaceholder.FULL, 0.7): [
        # detector_center=0
        3.499,
        3.499,
        3.499,
        3.499,
        # detector_center=0.2
        3.099,
        3.899,
        3.099,
        3.899,
        # detector_center=-0.2
        3.899,
        3.099,
        3.899,
        3.099,
        # detector_center=0.5
        3.059,
        4.499,
        3.059,
        4.499,
        # detector_center=-0.5
        4.499,
        3.059,
        4.499,
        3.059,
    ],
    (ExtentPlaceholder.FULL, 1.0): [
        # detector_center=0
        3.842,
        3.842,
        3.842,
        3.842,
        # detector_center=0.2
        3.442,
        4.242,
        3.442,
        4.242,
        # detector_center=-0.2
        4.242,
        3.442,
        4.242,
        3.442,
        # detector_center=0.5
        3.600,
        4.842,
        3.600,
        4.842,
        # detector_center=-0.5
        4.842,
        3.600,
        4.842,
        3.600,
    ],
    (ExtentPlaceholder.FULL, 1.7): [
        # detector_center=0
        3.240,
        3.240,
        3.240,
        3.240,
        # detector_center=0.2
        2.840,
        3.640,
        2.840,
        3.640,
        # detector_center=-0.2
        3.640,
        2.840,
        3.640,
        2.840,
        # detector_center=0.5
        3.406,
        4.240,
        3.406,
        4.240,
        # detector_center=-0.5
        4.240,
        3.406,
        4.240,
        3.406,
    ],
}


def _detector_placeholder_cases_halfcircle():
    """Yield (placeholder, side_ratio, detector_center, image_center, expected)."""
    for placeholder in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]:
        for sr in SIDE_RATIOS:
            expected_list = list(
                DETECTOR_PLACEHOLDER_EXPECTED_HALFCIRCLE[(placeholder, sr)]
            )
            idx = 0
            for dc in DETECTOR_CENTERS:
                for ic in IMAGE_CENTERS:
                    yield placeholder, sr, dc, ic, expected_list[idx]
                    idx += 1


@pytest.mark.parametrize(
    "placeholder, side_ratio, detector_center, image_center, expected",
    list(_detector_placeholder_cases_halfcircle()),
    ids=[
        f"{p.name}-sr{sr}-dc{dc}-ic{ic}"
        for p in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]
        for sr in SIDE_RATIOS
        for dc in DETECTOR_CENTERS
        for ic in IMAGE_CENTERS
    ],
)
def test_detector_extent_placeholder_halfcircle(
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
                image_domain=ImageDomain(size=(Nx, Ny), center=image_center, extent=2.0),
                angles=Na,
                detectors=Detectors(
                    number=Ns, center=detector_center, extent=placeholder
                ),
            )
    else:
        radon = Radon(
            image_domain=ImageDomain(size=(Nx, Ny), center=image_center, extent=2.0),
            angles=Na,
            detectors=Detectors(number=Ns, center=detector_center, extent=placeholder),
        )
        assert isinstance(radon.detectors.extent, float)
        assert radon.detectors.extent == pytest.approx(expected, abs=0.02), (
            f"Detector extent mismatch for {placeholder.name} with "
            f"side_ratio={side_ratio}, detector_center={detector_center}, "
            f"image_center={image_center}: "
            f"expected {expected}, got {radon.detectors.extent}"
        )


# -- Image placeholder with fixed detector extent ---------------------------

IMAGE_PLACEHOLDER_EXPECTED_HALFCIRCLE = {
    (ExtentPlaceholder.VALID, 0.7): [
        # detector_center=0
        3.429,
        3.429,
        3.429,
        3.429,
        # detector_center=0.2
        4.000,
        4.000,
        4.000,
        4.000,
        # detector_center=-0.2
        4.000,
        4.000,
        4.000,
        4.000,
        # detector_center=0.5
        4.857,
        4.857,
        4.857,
        4.857,
        # detector_center=-0.5
        4.857,
        4.857,
        4.857,
        4.857,
    ],
    (ExtentPlaceholder.VALID, 1.0): [
        # detector_center=0
        3.000,
        3.000,
        3.000,
        3.000,
        # detector_center=0.2
        2.800,
        3.400,
        2.800,
        3.400,
        # detector_center=-0.2
        3.400,
        2.800,
        3.400,
        2.800,
        # detector_center=0.5
        3.400,
        4.000,
        3.400,
        4.000,
        # detector_center=-0.5
        4.000,
        3.400,
        4.000,
        3.400,
    ],
    (ExtentPlaceholder.VALID, 1.7): [
        # detector_center=0
        5.100,
        5.100,
        5.100,
        5.100,
        # detector_center=0.2
        4.420,
        5.780,
        4.420,
        5.780,
        # detector_center=-0.2
        5.780,
        4.420,
        5.780,
        4.420,
        # detector_center=0.5
        3.400,
        6.800,
        3.400,
        6.800,
        # detector_center=-0.5
        6.800,
        3.400,
        6.800,
        3.400,
    ],
    (ExtentPlaceholder.FULL, 0.7): [
        # detector_center=0
        0.767,
        0.767,
        0.767,
        0.767,
        # detector_center=0.2
        1.097,
        0.436,
        1.097,
        0.436,
        # detector_center=-0.2
        0.436,
        1.097,
        0.436,
        1.097,
        # detector_center=0.5
        0.857,
        None,
        0.857,
        None,
        # detector_center=-0.5
        None,
        0.857,
        None,
        0.857,
    ],
    (ExtentPlaceholder.FULL, 1.0): [
        # detector_center=0
        0.682,
        0.682,
        0.682,
        0.682,
        # detector_center=0.2
        0.970,
        0.391,
        0.970,
        0.391,
        # detector_center=-0.2
        0.391,
        0.970,
        0.391,
        0.970,
        # detector_center=0.5
        0.600,
        None,
        0.600,
        None,
        # detector_center=-0.5
        None,
        0.600,
        None,
        0.600,
    ],
    (ExtentPlaceholder.FULL, 1.7): [
        # detector_center=0
        0.893,
        0.893,
        0.893,
        0.893,
        # detector_center=0.2
        1.200,
        0.522,
        1.200,
        0.522,
        # detector_center=-0.2
        0.522,
        1.200,
        0.522,
        1.200,
        # detector_center=0.5
        0.600,
        None,
        0.600,
        None,
        # detector_center=-0.5
        None,
        0.600,
        None,
        0.600,
    ],
}


def _image_placeholder_cases_halfcircle():
    """Yield (placeholder, side_ratio, detector_center, image_center, expected)."""
    for placeholder in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]:
        for sr in SIDE_RATIOS:
            expected_list = list(IMAGE_PLACEHOLDER_EXPECTED_HALFCIRCLE[(placeholder, sr)])
            idx = 0
            for dc in DETECTOR_CENTERS:
                for ic in IMAGE_CENTERS:
                    yield placeholder, sr, dc, ic, expected_list[idx]
                    idx += 1


@pytest.mark.parametrize(
    "placeholder, side_ratio, detector_center, image_center, expected",
    list(_image_placeholder_cases_halfcircle()),
    ids=[
        f"{p.name}-sr{sr}-dc{dc}-ic{ic}"
        for p in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]
        for sr in SIDE_RATIOS
        for dc in DETECTOR_CENTERS
        for ic in IMAGE_CENTERS
    ],
)
def test_image_extent_placeholder_halfcircle(
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
                detectors=Detectors(number=Ns, center=detector_center, extent=2.0),
            )
    else:
        radon = Radon(
            image_domain=ImageDomain(
                size=(Nx, Ny), center=image_center, extent=placeholder
            ),
            angles=Na,
            detectors=Detectors(number=Ns, center=detector_center, extent=2.0),
        )
        assert isinstance(radon.image_domain.extent, float)
        assert radon.image_domain.extent == pytest.approx(expected, abs=0.02), (
            f"Image extent mismatch for {placeholder.name} with "
            f"side_ratio={side_ratio}, detector_center={detector_center}, "
            f"image_center={image_center}: "
            f"expected {expected}, got {radon.image_domain.extent}"
        )


# ===========================================================================
# Full-circle counterpart: angles span [0, 2*pi) via Angles(half_circle=False),
# so the operator resolves placeholders with the full-circle geometry formulas.
# Same case grid; its own reference-value tables.
# ===========================================================================


# -- Detector placeholder with fixed image extent (full circle) -------------

DETECTOR_PLACEHOLDER_EXPECTED_FULLCIRCLE = {
    (ExtentPlaceholder.VALID, 0.7): [
        # detector_center=0
        1.000,
        1.000,
        1.000,
        1.000,
        # detector_center=0.2
        0.600,
        0.600,
        0.600,
        0.600,
        # detector_center=-0.2
        0.600,
        0.600,
        0.600,
        0.600,
        # detector_center=0.5
        None,
        None,
        None,
        None,
        # detector_center=-0.5
        None,
        None,
        None,
        None,
    ],
    (ExtentPlaceholder.VALID, 1.0): [
        # detector_center=0
        1.000,
        1.000,
        1.000,
        1.000,
        # detector_center=0.2
        0.600,
        0.600,
        0.600,
        0.600,
        # detector_center=-0.2
        0.600,
        0.600,
        0.600,
        0.600,
        # detector_center=0.5
        None,
        None,
        None,
        None,
        # detector_center=-0.5
        None,
        None,
        None,
        None,
    ],
    (ExtentPlaceholder.VALID, 1.7): [
        # detector_center=0
        0.176,
        0.176,
        0.176,
        0.176,
        # detector_center=0.2
        None,
        None,
        None,
        None,
        # detector_center=-0.2
        None,
        None,
        None,
        None,
        # detector_center=0.5
        None,
        None,
        None,
        None,
        # detector_center=-0.5
        None,
        None,
        None,
        None,
    ],
    (ExtentPlaceholder.FULL, 0.7): [
        # detector_center=0
        3.499,
        3.499,
        3.499,
        3.499,
        # detector_center=0.2
        3.899,
        3.899,
        3.899,
        3.899,
        # detector_center=-0.2
        3.899,
        3.899,
        3.899,
        3.899,
        # detector_center=0.5
        4.499,
        4.499,
        4.499,
        4.499,
        # detector_center=-0.5
        4.499,
        4.499,
        4.499,
        4.499,
    ],
    (ExtentPlaceholder.FULL, 1.0): [
        # detector_center=0
        3.842,
        3.842,
        3.842,
        3.842,
        # detector_center=0.2
        4.242,
        4.242,
        4.242,
        4.242,
        # detector_center=-0.2
        4.242,
        4.242,
        4.242,
        4.242,
        # detector_center=0.5
        4.842,
        4.842,
        4.842,
        4.842,
        # detector_center=-0.5
        4.842,
        4.842,
        4.842,
        4.842,
    ],
    (ExtentPlaceholder.FULL, 1.7): [
        # detector_center=0
        3.240,
        3.240,
        3.240,
        3.240,
        # detector_center=0.2
        3.640,
        3.640,
        3.640,
        3.640,
        # detector_center=-0.2
        3.640,
        3.640,
        3.640,
        3.640,
        # detector_center=0.5
        4.240,
        4.240,
        4.240,
        4.240,
        # detector_center=-0.5
        4.240,
        4.240,
        4.240,
        4.240,
    ],
}


def _detector_placeholder_cases_fullcircle():
    """Yield (placeholder, side_ratio, detector_center, image_center, expected)."""
    for placeholder in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]:
        for sr in SIDE_RATIOS:
            expected_list = list(
                DETECTOR_PLACEHOLDER_EXPECTED_FULLCIRCLE[(placeholder, sr)]
            )
            idx = 0
            for dc in DETECTOR_CENTERS:
                for ic in IMAGE_CENTERS:
                    yield placeholder, sr, dc, ic, expected_list[idx]
                    idx += 1


@pytest.mark.parametrize(
    "placeholder, side_ratio, detector_center, image_center, expected",
    list(_detector_placeholder_cases_fullcircle()),
    ids=[
        f"{p.name}-sr{sr}-dc{dc}-ic{ic}"
        for p in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]
        for sr in SIDE_RATIOS
        for dc in DETECTOR_CENTERS
        for ic in IMAGE_CENTERS
    ],
)
def test_detector_extent_placeholder_fullcircle(
    placeholder, side_ratio, detector_center, image_center, expected
):
    """Resolve a detector ExtentPlaceholder (full circle), fixed image extent 2.0.

    Verifies that the resolved detector extent matches the reference value.
    Cases where no valid geometry exists (expected is None) must raise a
    ValueError.
    """
    Ny = int(side_ratio * Nx)
    if expected is None:
        with pytest.raises(ValueError):
            Radon(
                image_domain=ImageDomain(size=(Nx, Ny), center=image_center, extent=2.0),
                angles=FULL_ANGLES,
                detectors=Detectors(
                    number=Ns, center=detector_center, extent=placeholder
                ),
            )
    else:
        radon = Radon(
            image_domain=ImageDomain(size=(Nx, Ny), center=image_center, extent=2.0),
            angles=FULL_ANGLES,
            detectors=Detectors(number=Ns, center=detector_center, extent=placeholder),
        )
        assert isinstance(radon.detectors.extent, float)
        assert radon.detectors.extent == pytest.approx(expected, abs=0.02), (
            f"Detector extent mismatch for {placeholder.name} with "
            f"side_ratio={side_ratio}, detector_center={detector_center}, "
            f"image_center={image_center}: "
            f"expected {expected}, got {radon.detectors.extent}"
        )


# -- Image placeholder with fixed detector extent (full circle) -------------

IMAGE_PLACEHOLDER_EXPECTED_FULLCIRCLE = {
    (ExtentPlaceholder.VALID, 0.7): [
        # detector_center=0
        3.429,
        3.429,
        3.429,
        3.429,
        # detector_center=0.2
        4.000,
        4.000,
        4.000,
        4.000,
        # detector_center=-0.2
        4.000,
        4.000,
        4.000,
        4.000,
        # detector_center=0.5
        4.857,
        4.857,
        4.857,
        4.857,
        # detector_center=-0.5
        4.857,
        4.857,
        4.857,
        4.857,
    ],
    (ExtentPlaceholder.VALID, 1.0): [
        # detector_center=0
        3.000,
        3.000,
        3.000,
        3.000,
        # detector_center=0.2
        3.400,
        3.400,
        3.400,
        3.400,
        # detector_center=-0.2
        3.400,
        3.400,
        3.400,
        3.400,
        # detector_center=0.5
        4.000,
        4.000,
        4.000,
        4.000,
        # detector_center=-0.5
        4.000,
        4.000,
        4.000,
        4.000,
    ],
    (ExtentPlaceholder.VALID, 1.7): [
        # detector_center=0
        5.100,
        5.100,
        5.100,
        5.100,
        # detector_center=0.2
        5.780,
        5.780,
        5.780,
        5.780,
        # detector_center=-0.2
        5.780,
        5.780,
        5.780,
        5.780,
        # detector_center=0.5
        6.800,
        6.800,
        6.800,
        6.800,
        # detector_center=-0.5
        6.800,
        6.800,
        6.800,
        6.800,
    ],
    (ExtentPlaceholder.FULL, 0.7): [
        # detector_center=0
        0.767,
        0.767,
        0.767,
        0.767,
        # detector_center=0.2
        0.436,
        0.436,
        0.436,
        0.436,
        # detector_center=-0.2
        0.436,
        0.436,
        0.436,
        0.436,
        # detector_center=0.5
        None,
        None,
        None,
        None,
        # detector_center=-0.5
        None,
        None,
        None,
        None,
    ],
    (ExtentPlaceholder.FULL, 1.0): [
        # detector_center=0
        0.682,
        0.682,
        0.682,
        0.682,
        # detector_center=0.2
        0.391,
        0.391,
        0.391,
        0.391,
        # detector_center=-0.2
        0.391,
        0.391,
        0.391,
        0.391,
        # detector_center=0.5
        None,
        None,
        None,
        None,
        # detector_center=-0.5
        None,
        None,
        None,
        None,
    ],
    (ExtentPlaceholder.FULL, 1.7): [
        # detector_center=0
        0.893,
        0.893,
        0.893,
        0.893,
        # detector_center=0.2
        0.522,
        0.522,
        0.522,
        0.522,
        # detector_center=-0.2
        0.522,
        0.522,
        0.522,
        0.522,
        # detector_center=0.5
        None,
        None,
        None,
        None,
        # detector_center=-0.5
        None,
        None,
        None,
        None,
    ],
}


def _image_placeholder_cases_fullcircle():
    """Yield (placeholder, side_ratio, detector_center, image_center, expected)."""
    for placeholder in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]:
        for sr in SIDE_RATIOS:
            expected_list = list(IMAGE_PLACEHOLDER_EXPECTED_FULLCIRCLE[(placeholder, sr)])
            idx = 0
            for dc in DETECTOR_CENTERS:
                for ic in IMAGE_CENTERS:
                    yield placeholder, sr, dc, ic, expected_list[idx]
                    idx += 1


@pytest.mark.parametrize(
    "placeholder, side_ratio, detector_center, image_center, expected",
    list(_image_placeholder_cases_fullcircle()),
    ids=[
        f"{p.name}-sr{sr}-dc{dc}-ic{ic}"
        for p in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]
        for sr in SIDE_RATIOS
        for dc in DETECTOR_CENTERS
        for ic in IMAGE_CENTERS
    ],
)
def test_image_extent_placeholder_fullcircle(
    placeholder, side_ratio, detector_center, image_center, expected
):
    """Resolve an image ExtentPlaceholder (full circle), fixed detector extent 2.0.

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
                angles=FULL_ANGLES,
                detectors=Detectors(number=Ns, center=detector_center, extent=2.0),
            )
    else:
        radon = Radon(
            image_domain=ImageDomain(
                size=(Nx, Ny), center=image_center, extent=placeholder
            ),
            angles=FULL_ANGLES,
            detectors=Detectors(number=Ns, center=detector_center, extent=2.0),
        )
        assert isinstance(radon.image_domain.extent, float)
        assert radon.image_domain.extent == pytest.approx(expected, abs=0.02), (
            f"Image extent mismatch for {placeholder.name} with "
            f"side_ratio={side_ratio}, detector_center={detector_center}, "
            f"image_center={image_center}: "
            f"expected {expected}, got {radon.image_domain.extent}"
        )


# -- Fan-beam (always resolved for the full circle) ---------------------------
#
# Instead of reference tables, every resolved Fanbeam geometry is checked by
# ray tracing: the coverage condition must hold, and must break once the
# resolved extent is changed by FANBEAM_SLACK. Where resolution fails, ray
# tracing must confirm that no extent can satisfy the condition.

FANBEAM_RE = 4.0
FANBEAM_R = 8.0
FANBEAM_SLACK = 1.05
FANBEAM_POINT = 1e-6
# The fixed extents are chosen so that the grid mixes resolvable and
# infeasible cases in both directions (the parallel-beam values 2.0 would make
# every FULL image case and, for Ny/Nx = 1.7, every off-center VALID detector
# case infeasible, since the fan magnifies by R / RE = 2):
# - image extent 2.5: VALID detector infeasible only for Ny/Nx = 1.7 and
#   |detector_center| = 0.5 (8 cases),
# - detector extent 3.0: FULL image infeasible only for
#   |detector_center| = 0.5 (24 cases).
FANBEAM_IMAGE_EXTENT = 2.5
FANBEAM_DETECTOR_EXTENT = 3.0


def _fanbeam_cases():
    """Yield (placeholder, side_ratio, detector_center, image_center)."""
    for placeholder in [ExtentPlaceholder.VALID, ExtentPlaceholder.FULL]:
        for sr in SIDE_RATIOS:
            for dc in DETECTOR_CENTERS:
                for ic in IMAGE_CENTERS:
                    yield placeholder, sr, dc, ic


FANBEAM_CASES = list(_fanbeam_cases())
FANBEAM_IDS = [f"{p.name}-sr{sr}-dc{dc}-ic{ic}" for p, sr, dc, ic in FANBEAM_CASES]


def _fanbeam(image_extent, detector_extent, side_ratio, detector_center, image_center):
    Ny = int(side_ratio * Nx)
    return Fanbeam(
        image_domain=ImageDomain(size=(Nx, Ny), center=image_center, extent=image_extent),
        angles=FULL_ANGLES,
        detectors=Detectors(number=Ns, center=detector_center, extent=detector_extent),
        source_detector_distance=FANBEAM_R,
        source_origin_distance=FANBEAM_RE,
    )


def _image_dims(extent, side_ratio):
    Ny = int(side_ratio * Nx)
    return (extent * Nx / max(Nx, Ny), extent * Ny / max(Nx, Ny))


def _covers(image_center, image_dims, detector_center, detector_width):
    return fanbeam_detector_covers_image(
        image_center, image_dims, detector_center, detector_width, FANBEAM_RE, FANBEAM_R
    )


def _meets(image_center, image_dims, detector_center, detector_width):
    return fanbeam_image_meets_detector_rays(
        image_center, image_dims, detector_center, detector_width, FANBEAM_RE, FANBEAM_R
    )


@pytest.mark.parametrize(
    "placeholder, side_ratio, detector_center, image_center",
    FANBEAM_CASES,
    ids=FANBEAM_IDS,
)
def test_detector_extent_placeholder_fanbeam(
    placeholder, side_ratio, detector_center, image_center
):
    """Resolve a Fanbeam detector ExtentPlaceholder with image extent 2.5."""
    D = _image_dims(FANBEAM_IMAGE_EXTENT, side_ratio)
    if placeholder == ExtentPlaceholder.FULL:
        fanbeam = _fanbeam(
            FANBEAM_IMAGE_EXTENT, placeholder, side_ratio, detector_center, image_center
        )
        Dd = fanbeam.detectors.extent
        assert _covers(image_center, D, detector_center, Dd)
        assert not _covers(image_center, D, detector_center, Dd / FANBEAM_SLACK)
        return

    if not _meets(image_center, D, detector_center, FANBEAM_POINT):
        # Not even the rays through the detector center always meet the image.
        with pytest.raises(ValueError, match="VALID.*for the detector"):
            _fanbeam(
                FANBEAM_IMAGE_EXTENT,
                placeholder,
                side_ratio,
                detector_center,
                image_center,
            )
        return
    fanbeam = _fanbeam(
        FANBEAM_IMAGE_EXTENT, placeholder, side_ratio, detector_center, image_center
    )
    Dd = fanbeam.detectors.extent
    assert _meets(image_center, D, detector_center, Dd)
    assert not _meets(image_center, D, detector_center, Dd * FANBEAM_SLACK)


@pytest.mark.parametrize(
    "placeholder, side_ratio, detector_center, image_center",
    FANBEAM_CASES,
    ids=FANBEAM_IDS,
)
def test_image_extent_placeholder_fanbeam(
    placeholder, side_ratio, detector_center, image_center
):
    """Resolve a Fanbeam image ExtentPlaceholder with detector extent 3.0."""
    Dd = FANBEAM_DETECTOR_EXTENT
    if placeholder == ExtentPlaceholder.VALID:
        fanbeam = _fanbeam(placeholder, Dd, side_ratio, detector_center, image_center)
        extent = fanbeam.image_domain.extent
        D = _image_dims(extent, side_ratio)
        smaller = _image_dims(extent / FANBEAM_SLACK, side_ratio)
        assert _meets(image_center, D, detector_center, Dd)
        assert not _meets(image_center, smaller, detector_center, Dd)
        return

    point = (FANBEAM_POINT, FANBEAM_POINT)
    if not _covers(image_center, point, detector_center, Dd):
        # Not even a point image at the image center fits the detector.
        with pytest.raises(ValueError, match="FULL.*for the image domain"):
            _fanbeam(placeholder, Dd, side_ratio, detector_center, image_center)
        return
    fanbeam = _fanbeam(placeholder, Dd, side_ratio, detector_center, image_center)
    extent = fanbeam.image_domain.extent
    D = _image_dims(extent, side_ratio)
    larger = _image_dims(extent * FANBEAM_SLACK, side_ratio)
    assert _covers(image_center, D, detector_center, Dd)
    assert not _covers(image_center, larger, detector_center, Dd)
