import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gratopy.utilities import (
    Angles,
    Detectors,
    ExtentPlaceholder,
    ImageDomain,
    full_detector_given_image_fanbeam,
    full_image_given_detector_fanbeam,
    valid_detector_given_image_fanbeam,
    valid_image_given_detector_fanbeam,
)

from .helpers import (
    fanbeam_detector_covers_image,
    fanbeam_image_meets_detector_rays,
)


def test_angles_are_immutable_and_own_read_only_array_copies():
    angle_values = np.array([0.0, 1.0])
    weight_values = np.array([0.5, 0.5])
    angles = Angles(angle_values, weight_values, half_circle=True)

    angle_values[0] = 2.0
    weight_values[0] = 3.0

    np.testing.assert_array_equal(angles.angles, [0.0, 1.0])
    np.testing.assert_array_equal(angles.weights, [0.5, 0.5])
    assert not angles.angles.flags.writeable
    assert not angles.weights.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        angles.angles[0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        angles.weights[0] = 3.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        angles.angles.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        angles.half_circle = False


def test_angles_remain_immutable_after_copying_and_pickling():
    angles = Angles([0.0, 1.0], [0.5, 0.5])

    restored_angles = (
        copy.deepcopy(angles),
        pickle.loads(pickle.dumps(angles)),
    )

    for restored in restored_angles:
        np.testing.assert_array_equal(restored.angles, angles.angles)
        assert not restored.angles.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            restored.angles.flags.writeable = True


def test_detectors_are_immutable():
    detectors = Detectors(number=-20, extent=3.0, center=0.25)

    assert detectors.number == 20
    assert detectors.reversed is True
    with pytest.raises(FrozenInstanceError):
        detectors.extent = 4.0


def test_image_domains_are_immutable():
    image_domain = ImageDomain(
        size=(16, 12),
        extent=2.0,
        center=(0.1, -0.2),
    )

    assert image_domain.size == (16, 12)
    assert image_domain.center == (0.1, -0.2)
    with pytest.raises(FrozenInstanceError):
        image_domain.extent = ExtentPlaceholder.FULL


# -- Fan-beam extent formulas, checked by ray tracing -------------------------

RE = 4.0
R = 8.0
FANBEAM_CASES = [
    # (Md, Mx, My), (Dx, Dy), Dd
    ((0.0, 0.0, 0.0), (1.5, 1.5), 3.0),
    ((0.2, 0.3, -0.1), (2.0, 1.2), 4.0),
    ((-0.3, -0.2, 0.25), (1.0, 1.8), 3.5),
]


def _covers(M, D, Dd):
    return fanbeam_detector_covers_image(M[1:], D, M[0], Dd, RE, R)


def _meets(M, D, Dd):
    return fanbeam_image_meets_detector_rays(M[1:], D, M[0], Dd, RE, R)


@pytest.mark.parametrize(("M", "D", "_Dd"), FANBEAM_CASES)
def test_full_detector_given_image_fanbeam_covers_image_tightly(M, D, _Dd):
    Dd = full_detector_given_image_fanbeam(M, D, RE, R)

    assert _covers(M, D, Dd)
    assert not _covers(M, D, 0.98 * Dd)


@pytest.mark.parametrize(("M", "D", "Dd"), FANBEAM_CASES)
def test_full_image_given_detector_fanbeam_fits_detector_tightly(M, D, Dd):
    (Dx, Dy) = full_image_given_detector_fanbeam(M, Dd, RE, R, c=D[0] / D[1])

    assert _covers(M, (Dx, Dy), Dd)
    assert not _covers(M, (1.02 * Dx, 1.02 * Dy), Dd)


@pytest.mark.parametrize(("M", "D", "Dd"), FANBEAM_CASES)
def test_valid_image_given_detector_fanbeam_meets_all_rays_tightly(M, D, Dd):
    (Dx, Dy) = valid_image_given_detector_fanbeam(M, Dd, RE, R, c=D[0] / D[1])

    assert _meets(M, (Dx, Dy), Dd)
    assert not _meets(M, (0.98 * Dx, 0.98 * Dy), Dd)


@pytest.mark.parametrize(("M", "D", "_Dd"), FANBEAM_CASES)
def test_valid_detector_given_image_fanbeam_meets_image_tightly(M, D, _Dd):
    Dd = valid_detector_given_image_fanbeam(M, D, RE, R)

    assert _meets(M, D, Dd)
    # The binding ray direction falls between the sampled angles, so a
    # slightly larger margin is needed to see the violation.
    assert not _meets(M, D, 1.05 * Dd)


def test_fanbeam_extent_formulas_report_infeasible_geometries():
    # Image reaching the source circle, image not containing the rotation
    # center, and detector not containing the ray through the center.
    assert full_detector_given_image_fanbeam((0, 0, 0), (6, 6), RE, R) is None
    assert valid_detector_given_image_fanbeam((0, 0.3, 0), (0.4, 0.4), RE, R) is None
    assert full_image_given_detector_fanbeam((2.0, 0, 0), 3.0, RE, R) is None
