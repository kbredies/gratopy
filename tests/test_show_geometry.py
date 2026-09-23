"""Examples for visualizing operator geometries with ``show_geometry``.

Each test draws a typical use of
:meth:`show_geometry <gratopy.operator.Radon.show_geometry>` and checks a
basic property of the drawn geometry. The drawing shows

- the image domain and the rotation center (0, 0),
- the detector as a thick line, with an arrow towards increasing pixel
  index, an open circle at pixel 0 and a tick at the detector center,
- sample rays over the detector and the central ray; rays hitting the
  detector but missing the image are dashed in red and the illuminated
  region is shaded,
- for fan beams the source and the source circle with all operator angles.

The second part of this file checks that the drawing is correct: the drawn
geometry is compared with the angle vectors of the kernel structures and with
the sinogram support of an off-center image (including shifted and reversed
detectors), and the drawn artists are compared with the geometry.

Plots are deactivated by default and can be activated by setting
``export GRATOPY_TEST_PLOT=true`` in the terminal.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import pytest

from gratopy.operator import Fanbeam, Radon, RayDrivenFanbeam, RayDrivenRadon
from gratopy.operator.projection import _ProjectionOperator
from gratopy.utilities import Angles, Detectors, ExtentPlaceholder, ImageDomain

# Plots are deactivated by default, can be activated
# by setting 'export GRATOPY_TEST_PLOT=true' in the terminal
plot_parameter = os.environ.get("GRATOPY_TEST_PLOT")
if plot_parameter is None:
    plot_parameter = "0"
if plot_parameter.lower() not in ["0", "false"]:
    PLOT = True
else:
    PLOT = False


SOURCE_ORIGIN_DISTANCE = 4.0
SOURCE_DETECTOR_DISTANCE = 8.0
IMAGE_SIZE = (200, 140)
IMAGE_CENTER = (0.4, -0.2)


def finish(figure):
    """Show the figure if plotting is activated, close it otherwise."""
    if PLOT:
        plt.show()
    plt.close(figure)


def make_fanbeam(image_extent, detector_extent, detector_center, reversed=False):
    return Fanbeam(
        image_domain=ImageDomain(IMAGE_SIZE, extent=image_extent, center=IMAGE_CENTER),
        angles=36,
        detectors=Detectors(
            100, extent=detector_extent, center=detector_center, reversed=reversed
        ),
        source_detector_distance=SOURCE_DETECTOR_DISTANCE,
        source_origin_distance=SOURCE_ORIGIN_DISTANCE,
    )


def test_show_geometry_single_operator():
    """Draw a Radon and a Fanbeam geometry at one angle.

    Without an ``ax``, :meth:`show_geometry` creates its own figure with the
    operator name as title and returns the axes. Without ``angles``, the
    first angle of the operator is drawn.
    """
    radon = Radon(
        image_domain=ImageDomain(IMAGE_SIZE, extent=2.0, center=IMAGE_CENTER),
        angles=Angles.uniform(90, half_circle=True),
        detectors=Detectors(100, extent=2.0, center=0.3),
    )
    fanbeam = make_fanbeam(2.0, 5.0, 0.4)

    for operator in (radon, fanbeam):
        ax = operator.show_geometry(np.pi / 5)
        assert ax.get_title() == repr(operator)
        finish(ax.figure)


def test_show_geometry_extent_placeholders():
    """Draw the four placeholder resolutions of a fan-beam geometry.

    All four are drawn at the angle 3.4. For VALID, every ray hitting the
    detector passes through the image, so no ray is dashed in red. For
    FULL, every ray through the image hits the detector; rays hitting the
    detector past the image are allowed and drawn dashed in red. The
    binding angle of FULL may lie anywhere on the circle, so FULL is checked
    over all angles of the operator, not only the drawn one.
    """
    cases = {
        "detector FULL": (2.0, ExtentPlaceholder.FULL),
        "detector VALID": (2.0, ExtentPlaceholder.VALID),
        "image FULL": (ExtentPlaceholder.FULL, 3.0),
        "image VALID": (ExtentPlaceholder.VALID, 3.0),
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), layout="constrained")
    angle = 3.4

    for ax, (title, (image_extent, detector_extent)) in zip(axes.flat, cases.items()):
        operator = make_fanbeam(image_extent, detector_extent, 0.3)
        operator.show_geometry(angle, ax=ax, legend=ax is axes[0, 1])
        ax.set_title(
            f"Fanbeam, {title}: image {operator.image_domain.extent:.3f}, "
            f"detector {operator.detectors.extent:.3f}"
        )

        geometry = operator._geometry_at()
        lower, upper = geometry.detector_interval
        if "VALID" in title:
            missing = [a for a in ax.get_children() if a.get_gid() == "ray-miss"]
            assert not missing
        else:
            lowest, highest = geometry.image_detector_range().T
            assert lowest.min() >= lower - 1e-9
            assert highest.max() <= upper + 1e-9

    finish(figure)


@pytest.mark.parametrize("reversed", [False, True])
def test_show_geometry_reversed_detector(reversed):
    """Draw a fan-beam geometry with a shifted, possibly reversed detector.

    Reversing the detector swaps the end holding pixel 0 and the direction
    of the arrow; the detector center Md moves to the other side of the
    detector foot point accordingly.
    """
    operator = make_fanbeam(2.0, 5.0, 1.0, reversed=reversed)
    ax = operator.show_geometry(np.pi / 5)
    ax.set_title(f"Fanbeam, Md = 1.0, reversed={reversed}")

    geometry = operator._geometry_at(np.pi / 5)
    (first_pixel,) = [a for a in ax.get_children() if a.get_gid() == "first-pixel"]
    expected = geometry.detector_points(geometry.pixel_positions()[0])[0, 0]
    np.testing.assert_allclose(first_pixel.get_xydata()[0], expected)
    finish(ax.figure)


def test_show_geometry_several_angles():
    """Overlay several angles of a Radon and a Fanbeam geometry.

    Up to eight angles are listed with their color in the legend; the
    sample rays can be reduced via ``n_rays`` to keep the drawing readable.
    """
    radon = Radon(
        image_domain=ImageDomain(IMAGE_SIZE, extent=2.0, center=IMAGE_CENTER),
        angles=Angles.uniform(90, half_circle=True),
        detectors=Detectors(100, extent=2.5, center=0.3),
    )
    fanbeam = make_fanbeam(2.0, ExtentPlaceholder.FULL, 0.3)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), layout="constrained")
    radon_angles = np.linspace(0, np.pi, 4, endpoint=False)
    radon.show_geometry(radon_angles, ax=axes[0], n_rays=7)
    axes[0].set_title("Radon, four angles in [0, pi)")
    fanbeam.show_geometry([0, np.pi / 2, np.pi], ax=axes[1], n_rays=7)
    axes[1].set_title("Fanbeam (FULL detector), three angles")

    detectors = [a for a in axes[0].get_children() if a.get_gid() == "detector"]
    assert len(detectors) == len(radon_angles)
    finish(figure)


# ============================================================================
# Correctness of the drawn geometry (no plots)
# ============================================================================

CHECK_ANGLES = Angles.uniform(24)


@pytest.fixture(scope="module")
def queue():
    context = cl.create_some_context(interactive=False)
    return cl.CommandQueue(context)


def _checked_radon(operator_class=Radon, **image):
    return operator_class(
        image_domain=ImageDomain(**({"size": (20, 16), "center": (0.3, -0.2)} | image)),
        angles=CHECK_ANGLES,
        detectors=Detectors(number=60, extent=3.0, center=0.25),
    )


def _checked_fanbeam(operator_class=Fanbeam, reversed=False, **image):
    return operator_class(
        image_domain=ImageDomain(**({"size": (20, 16), "center": (0.3, -0.2)} | image)),
        angles=CHECK_ANGLES,
        detectors=Detectors(number=90, extent=6.0, center=0.4, reversed=reversed),
        source_detector_distance=8.0,
        source_origin_distance=4.0,
    )


def test_fanbeam_geometry_matches_host_struct(queue):
    for reversed in (False, True):
        operator = _checked_fanbeam(reversed=reversed)
        geometry = operator._geometry_at()
        operator._ensure_host_struct(queue)
        ofs = operator._host_struct["ofs_dict"][np.dtype("float64")]
        geo = operator._host_struct["geo_dict"][np.dtype("float64")]
        delta_x = geo[10]
        delta_s = geo[2] * delta_x

        np.testing.assert_allclose(geometry.detector_axis * delta_s, ofs[0:2].T * delta_x)
        np.testing.assert_allclose(geometry.sources, ofs[2:4].T * delta_x)
        np.testing.assert_allclose(geometry.detector_origin, ofs[4:6].T * delta_x)
        np.testing.assert_allclose(
            geometry.pixel_positions(),
            (np.arange(operator.detectors.number) - geo[5]) * delta_s,
        )


def test_radon_geometry_matches_host_struct(queue):
    operator = _checked_radon()
    geometry = operator._geometry_at()
    operator._ensure_host_struct(queue)
    ofs = operator._host_struct["ofs_dict"][np.dtype("float64")]
    Nx, Ny = operator.image_domain.size
    delta_x = operator.image_domain.extent / max(Nx, Ny)
    delta_s = operator.detectors.extent / operator.detectors.number
    Mx, My = operator.image_domain.center

    # Detector index of image pixel (i, j) as evaluated by the kernels.
    i, j = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    kernel_index = ofs[0] * i[..., None] + ofs[1] * j[..., None] + ofs[2]

    x = (i - (Nx - 1) / 2) * delta_x + Mx
    y = (j - (Ny - 1) / 2) * delta_x + My
    t = np.einsum(
        "ijak,ak->ija",
        np.stack([x, y], axis=-1)[:, :, None, :] - geometry.detector_origin[None, None],
        geometry.detector_axis,
    )
    n = operator.detectors.number
    geometry_index = (n - 1) / 2 + (t - operator.detectors.center) / delta_s

    np.testing.assert_allclose(geometry_index, kernel_index, atol=1e-9)


@pytest.mark.parametrize(
    "make_operator",
    [
        lambda: _checked_radon(Radon, extent=0.4),
        lambda: _checked_radon(RayDrivenRadon, extent=0.4),
        lambda: _checked_fanbeam(Fanbeam, extent=0.4),
        lambda: _checked_fanbeam(Fanbeam, reversed=True, extent=0.4),
        lambda: _checked_fanbeam(RayDrivenFanbeam, reversed=True, extent=0.4),
    ],
    ids=["radon", "ray-radon", "fanbeam", "fanbeam-reversed", "ray-fanbeam-reversed"],
)
def test_sinogram_support_matches_image_detector_range(make_operator, queue):
    operator = make_operator()
    geometry = operator._geometry_at()
    image = np.ones(operator.input_shape, dtype=np.float32)
    sinogram = operator.apply_to(image, queue=queue).get()

    delta_s = geometry.detector_width / geometry.detector_number
    n = geometry.detector_number
    predicted = (n - 1) / 2 + (
        geometry.image_detector_range() - geometry.detector_center
    ) / delta_s

    for a in range(len(geometry.angles)):
        support = np.flatnonzero(sinogram[:, a] > 1e-6 * sinogram[:, a].max())
        # Pixel-driven kernels spread each image pixel over neighbouring
        # detector pixels, so the support may exceed the prediction slightly.
        assert predicted[a, 0] - 2 <= support[0] <= predicted[a, 0] + 1
        assert predicted[a, 1] - 1 <= support[-1] <= predicted[a, 1] + 2


@pytest.mark.parametrize("make_operator", [_checked_radon, _checked_fanbeam])
def test_rays_hit_image_exactly_inside_image_detector_range(make_operator):
    geometry = make_operator()._geometry_at()
    t = np.linspace(*geometry.detector_interval, 401)
    hits = geometry.rays_hit_image(t)
    lowest, highest = geometry.image_detector_range().T

    inside = (t[None, :] > lowest[:, None] + 1e-9) & (
        t[None, :] < highest[:, None] - 1e-9
    )
    outside = (t[None, :] < lowest[:, None] - 1e-9) | (
        t[None, :] > highest[:, None] + 1e-9
    )
    assert hits[inside].all()
    assert not hits[outside].any()


def test_full_detector_placeholder_encloses_image_detector_range():
    operator = Fanbeam(
        image_domain=ImageDomain(size=(20, 16), extent=2.0, center=(0.3, -0.2)),
        angles=Angles.uniform(720),
        detectors=Detectors(number=90, extent=ExtentPlaceholder.FULL, center=0.4),
        source_detector_distance=8.0,
        source_origin_distance=4.0,
    )
    geometry = operator._geometry_at()
    lower, upper = geometry.detector_interval
    lowest, highest = geometry.image_detector_range().T

    assert lowest.min() >= lower - 1e-9
    assert highest.max() <= upper + 1e-9
    assert min(lowest.min() - lower, upper - highest.max()) < 1e-3 * (upper - lower)


def test_geometry_at_single_angle():
    geometry = _checked_fanbeam()._geometry_at(np.pi / 3)

    assert geometry.angles.shape == (1,)
    assert geometry.sources.shape == (1, 2)
    assert geometry.detector_points([0.0, 1.0]).shape == (1, 2, 2)


def _artists(ax, gid):
    return [a for a in ax.get_children() if a.get_gid() == gid]


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


@pytest.mark.parametrize("make_operator", [_checked_radon, _checked_fanbeam])
def test_show_geometry_draws_into_given_axes_without_clearing(make_operator, ax):
    (existing,) = ax.plot([0, 1], [0, 1])

    returned = make_operator().show_geometry(0.4, ax=ax)

    assert returned is ax
    assert existing in ax.get_lines()
    assert ax.get_aspect() == 1.0
    assert ax.get_legend() is not None


@pytest.mark.parametrize("make_operator", [_checked_radon, _checked_fanbeam])
def test_show_geometry_marks_exactly_the_rays_missing_the_image(make_operator, ax):
    operator = make_operator()
    angles = [0.4, 2.0]
    geometry = operator._geometry_at(angles)
    hits = geometry.rays_hit_image(np.linspace(*geometry.detector_interval, 11))

    operator.show_geometry(angles, ax=ax, n_rays=11)

    assert len(_artists(ax, "ray-miss")) == (~hits).sum() > 0
    assert len(_artists(ax, "ray-hit")) == hits.sum()
    assert len(_artists(ax, "detector")) == len(angles)


def test_show_geometry_without_coverage_draws_no_missing_rays(ax):
    _checked_radon().show_geometry(0.4, ax=ax, n_rays=11, coverage=False, legend=False)

    assert not _artists(ax, "ray-miss")
    assert not _artists(ax, "illuminated")
    assert len(_artists(ax, "ray-hit")) == 11
    assert ax.get_legend() is None


def test_show_geometry_draws_fanbeam_source_and_trajectory(ax):
    operator = _checked_fanbeam()
    operator.show_geometry(ax=ax)

    (source,) = _artists(ax, "source")
    np.testing.assert_allclose(
        source.get_xydata()[0],
        operator._geometry_at(operator.angles.angles[0]).sources[0],
    )
    (samples,) = _artists(ax, "trajectory-samples")
    assert len(samples.get_xydata()) == len(operator.angles)


def test_show_geometry_draws_no_trajectory_for_parallel_beams(ax):
    _checked_radon().show_geometry(ax=ax)

    assert not _artists(ax, "trajectory")
    assert not _artists(ax, "source")


def test_show_geometry_creates_figure_when_no_axes_given():
    axes = _checked_fanbeam().show_geometry()

    assert axes.get_title() == "Fanbeam"
    plt.close(axes.figure)


def test_projection_operator_without_beam_geometry_rejects_geometry_queries():
    class UnsupportedProjection(_ProjectionOperator):
        _operator_name = "UnsupportedProjection"
        _kernel_filename = "radon.cl"
        _kernel_base_name = "radon"

    operator = UnsupportedProjection(
        state={
            "image_domain": ImageDomain(size=16, extent=2.0),
            "angles": Angles.uniform(20),
            "detectors": Detectors(number=20, extent=3.0),
        },
        kernel_spec=None,
    )

    with pytest.raises(NotImplementedError, match="UnsupportedProjection"):
        operator._geometry_at()


def test_show_geometry_legend_uses_the_angle_color_for_one_angle(ax):
    _checked_fanbeam().show_geometry(0.4, ax=ax)
    ray_hit = _artists(ax, "ray-hit")[0]
    entries = {handle.get_label(): handle for handle in ax.get_legend().legend_handles}

    for label in ("ray meeting image", "central ray", "detector, arrow: index ↑"):
        assert entries[label].get_color() == ray_hit.get_color()
