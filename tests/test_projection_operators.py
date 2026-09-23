import numpy as np
import pytest
import pyopencl as cl
import pyopencl.array as clarray

from gratopy.operator import (
    Fanbeam,
    Radon,
    RayDrivenFanbeam,
    RayDrivenRadon,
    StripDrivenRadon,
)
from gratopy.operator.base import AdjointOperator
from gratopy.operator.projection import _ProjectionOperator
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


PARALLEL_OPERATORS = (Radon, RayDrivenRadon, StripDrivenRadon)
FANBEAM_OPERATORS = (Fanbeam, RayDrivenFanbeam)


@pytest.fixture(scope="module")
def queue():
    context = cl.create_some_context(interactive=False)
    return cl.CommandQueue(context)


def make_parallel_operator(operator_class):
    return operator_class(
        image_domain=ImageDomain(size=(24, 20), extent=2.0, center=(0.05, -0.03)),
        angles=Angles.uniform(17, half_circle=True),
        detectors=Detectors(number=29, extent=3.0, center=0.1, reversed=True),
    )


def make_fanbeam_operator(operator_class):
    return operator_class(
        image_domain=ImageDomain(size=(24, 20), extent=2.0, center=(0.05, -0.03)),
        angles=Angles.uniform(24),
        detectors=Detectors(number=29, extent=3.0, center=0.1, reversed=True),
        source_detector_distance=8.0,
        source_origin_distance=4.0,
    )


@pytest.mark.parametrize(
    ("operator_class", "expected_name", "kernel_base_name"),
    [
        (Radon, "Radon", "radon"),
        (RayDrivenRadon, "RayDrivenRadon", "radon_ray"),
        (StripDrivenRadon, "StripDrivenRadon", "radon_strip"),
    ],
)
def test_parallel_operator_variants_use_stable_leaf_architecture(
    operator_class,
    expected_name,
    kernel_base_name,
):
    operator = make_parallel_operator(operator_class)

    assert repr(operator) == expected_name
    assert operator.kernel_spec.base_name == kernel_base_name
    assert "adjoint" not in operator.state
    assert isinstance(operator.T, AdjointOperator)
    assert operator.T.operand is operator
    assert operator.T.T is operator


@pytest.mark.parametrize(
    ("operator_class", "expected_name", "kernel_base_name"),
    [
        (Fanbeam, "Fanbeam", "fanbeam"),
        (RayDrivenFanbeam, "RayDrivenFanbeam", "fanbeam_ray"),
    ],
)
def test_fanbeam_operator_variants_use_stable_leaf_architecture(
    operator_class,
    expected_name,
    kernel_base_name,
):
    operator = make_fanbeam_operator(operator_class)

    assert repr(operator) == expected_name
    assert operator.kernel_spec.base_name == kernel_base_name
    assert "adjoint" not in operator.state
    assert isinstance(operator.T, AdjointOperator)
    assert operator.T.operand is operator
    assert operator.T.T is operator


@pytest.mark.parametrize("operator_class", PARALLEL_OPERATORS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_parallel_operator_variants_execute_slicewise(operator_class, order, queue):
    operator = make_parallel_operator(operator_class)
    rng = np.random.default_rng(1)
    image = np.require(
        rng.standard_normal(operator.input_shape + (2,)),
        dtype=np.float32,
        requirements=order,
    )
    device_image = clarray.to_device(queue, image)

    sinogram = operator.apply_to(device_image)
    backprojection = operator.T.apply_to(sinogram)

    assert sinogram.shape == operator.output_shape + (2,)
    assert backprojection.shape == operator.input_shape + (2,)
    assert np.all(np.isfinite(sinogram.get()))
    assert np.all(np.isfinite(backprojection.get()))


@pytest.mark.parametrize("operator_class", FANBEAM_OPERATORS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_fanbeam_operator_variants_execute_slicewise(operator_class, order, queue):
    operator = make_fanbeam_operator(operator_class)
    rng = np.random.default_rng(1)
    image = np.require(
        rng.standard_normal(operator.input_shape + (2,)),
        dtype=np.float32,
        requirements=order,
    )
    device_image = clarray.to_device(queue, image)

    sinogram = operator.apply_to(device_image)
    backprojection = operator.T.apply_to(sinogram)

    assert sinogram.shape == operator.output_shape + (2,)
    assert backprojection.shape == operator.input_shape + (2,)
    assert np.all(np.isfinite(sinogram.get()))
    assert np.all(np.isfinite(backprojection.get()))


def assert_physical_adjointness(operator, queue):
    rng = np.random.default_rng(2)
    image = clarray.to_device(
        queue,
        rng.standard_normal(operator.input_shape).astype(np.float32),
    )
    sinogram = clarray.to_device(
        queue,
        rng.standard_normal(operator.output_shape).astype(np.float32),
    )

    projected = operator.apply_to(image).get()
    backprojected = operator.T.apply_to(sinogram).get()
    image_values = image.get()
    sinogram_values = sinogram.get()

    image_extent = operator.image_domain.extent
    detector_extent = operator.detectors.extent
    assert not isinstance(image_extent, ExtentPlaceholder)
    assert not isinstance(detector_extent, ExtentPlaceholder)
    delta_x = image_extent / max(operator.image_domain.size)
    delta_s = detector_extent / operator.detectors.number

    image_pairing = np.vdot(image_values, backprojected) * delta_x**2
    weighted_sinogram = sinogram_values * operator.angles.weights[np.newaxis, :]
    sinogram_pairing = np.vdot(weighted_sinogram, projected) * delta_s

    assert image_pairing == pytest.approx(sinogram_pairing, rel=2e-4, abs=2e-5)


@pytest.mark.parametrize("operator_class", PARALLEL_OPERATORS)
def test_parallel_operator_variants_are_physical_adjoints(operator_class, queue):
    assert_physical_adjointness(make_parallel_operator(operator_class), queue)


@pytest.mark.parametrize("operator_class", FANBEAM_OPERATORS)
def test_fanbeam_operator_variants_are_physical_adjoints(operator_class, queue):
    assert_physical_adjointness(make_fanbeam_operator(operator_class), queue)


@pytest.mark.parametrize("operator_class", PARALLEL_OPERATORS)
def test_parallel_operator_variants_share_runtime_with_adjoint(operator_class, queue):
    operator = make_parallel_operator(operator_class)
    image = clarray.zeros(queue, operator.input_shape, dtype=np.float32)

    for _ in range(3):
        (operator.T * operator).apply_to(image)
    queue.finish()

    assert len(operator._device_struct) == 1
    assert set(next(iter(operator._device_struct.values()))) == {"ofs", "geometry"}


@pytest.mark.parametrize("operator_class", FANBEAM_OPERATORS)
def test_fanbeam_operator_variants_share_runtime_with_adjoint(operator_class, queue):
    operator = make_fanbeam_operator(operator_class)
    image = clarray.zeros(queue, operator.input_shape, dtype=np.float32)

    for _ in range(3):
        (operator.T * operator).apply_to(image)
    queue.finish()

    assert len(operator._device_struct) == 1
    expected_buffers = {"ofs", "geometry"}
    if operator_class is Fanbeam:
        expected_buffers.add("sdpd")
    assert set(next(iter(operator._device_struct.values()))) == expected_buffers


def test_fanbeam_uses_explicit_source_distances():
    operator = Fanbeam(
        image_domain=16,
        angles=20,
        detectors=Detectors(number=24, extent=3.0),
        source_detector_distance=8.0,
        source_origin_distance=4.0,
    )

    assert operator.source_detector_distance == 8.0
    assert operator.source_origin_distance == 4.0
    assert operator.angles.half_circle is False


@pytest.mark.parametrize(
    ("source_detector_distance", "source_origin_distance"),
    [(4.0, 4.0), (3.0, 4.0), (-2.0, -1.0), (np.inf, 4.0)],
)
def test_fanbeam_rejects_invalid_source_distances(
    source_detector_distance,
    source_origin_distance,
):
    with pytest.raises(ValueError, match="source_.*distance"):
        Fanbeam(
            image_domain=16,
            angles=20,
            detectors=Detectors(number=24, extent=3.0),
            source_detector_distance=source_detector_distance,
            source_origin_distance=source_origin_distance,
        )


@pytest.mark.parametrize("operator_class", FANBEAM_OPERATORS)
@pytest.mark.parametrize(
    ("placeholder", "expected"),
    [
        (
            ExtentPlaceholder.FULL,
            full_detector_given_image_fanbeam((0.1, 0.2, -0.1), (2.0, 1.6), 4.0, 8.0),
        ),
        (
            ExtentPlaceholder.VALID,
            valid_detector_given_image_fanbeam((0.1, 0.2, -0.1), (2.0, 1.6), 4.0, 8.0),
        ),
    ],
)
def test_fanbeam_resolves_detector_placeholder_without_mutating_geometry(
    operator_class, placeholder, expected
):
    image_domain = ImageDomain(size=(20, 16), extent=2.0, center=(0.2, -0.1))
    detectors = Detectors(number=20, extent=placeholder, center=0.1)

    operator = operator_class(
        image_domain=image_domain,
        angles=20,
        detectors=detectors,
        source_detector_distance=8.0,
        source_origin_distance=4.0,
    )

    assert operator.detectors.extent == pytest.approx(expected)
    assert operator.image_domain.extent == 2.0
    assert detectors.extent is placeholder


@pytest.mark.parametrize("operator_class", FANBEAM_OPERATORS)
@pytest.mark.parametrize(
    ("placeholder", "expected_dimensions"),
    [
        (
            ExtentPlaceholder.FULL,
            full_image_given_detector_fanbeam((0.1, 0.2, -0.1), 3.0, 6.0, 8.0, 1.25),
        ),
        (
            ExtentPlaceholder.VALID,
            valid_image_given_detector_fanbeam((0.1, 0.2, -0.1), 3.0, 6.0, 8.0, 1.25),
        ),
    ],
)
def test_fanbeam_resolves_image_placeholder_without_mutating_geometry(
    operator_class, placeholder, expected_dimensions
):
    image_domain = ImageDomain(size=(20, 16), extent=placeholder, center=(0.2, -0.1))
    detectors = Detectors(number=20, extent=3.0, center=0.1)

    operator = operator_class(
        image_domain=image_domain,
        angles=20,
        detectors=detectors,
        source_detector_distance=8.0,
        source_origin_distance=6.0,
    )

    assert operator.image_domain.extent == pytest.approx(max(expected_dimensions))
    assert operator.detectors.extent == 3.0
    assert image_domain.extent is placeholder


def test_fanbeam_placeholder_uses_full_circle_for_half_circle_angles():
    kwargs = {
        "image_domain": ImageDomain(size=16, extent=2.0, center=(0.2, -0.1)),
        "detectors": Detectors(number=20, extent=ExtentPlaceholder.FULL, center=0.1),
        "source_detector_distance": 8.0,
        "source_origin_distance": 4.0,
    }

    half = Fanbeam(angles=Angles.uniform(20, half_circle=True), **kwargs)
    full = Fanbeam(angles=Angles.uniform(20), **kwargs)

    assert half.detectors.extent == full.detectors.extent


def test_fanbeam_rejects_two_extent_placeholders():
    with pytest.raises(NotImplementedError, match="Both the ImageDomain"):
        Fanbeam(
            image_domain=ImageDomain(size=16, extent=ExtentPlaceholder.FULL),
            angles=20,
            detectors=Detectors(number=20, extent=ExtentPlaceholder.FULL),
            source_detector_distance=8.0,
            source_origin_distance=4.0,
        )


def test_fanbeam_reports_placeholder_that_places_source_inside_image():
    # VALID needs an image meeting all rays hitting the 40.0-wide detector;
    # those rays pass up to 3.7 from the center, so the image corners reach
    # beyond the source circle of radius 4.0.
    with pytest.raises(ValueError, match="resolving the image extent"):
        Fanbeam(
            image_domain=ImageDomain(size=16, extent=ExtentPlaceholder.VALID),
            angles=20,
            detectors=Detectors(number=20, extent=40.0),
            source_detector_distance=8.0,
            source_origin_distance=4.0,
        )


def test_projection_operator_without_formulas_rejects_placeholders():
    class UnsupportedProjection(_ProjectionOperator):
        _operator_name = "UnsupportedProjection"
        _kernel_filename = "radon.cl"
        _kernel_base_name = "radon"

    operator = UnsupportedProjection(
        state={
            "image_domain": ImageDomain(size=16, extent=2.0),
            "angles": Angles.uniform(20),
            "detectors": Detectors(number=20, extent=ExtentPlaceholder.FULL),
        },
        kernel_spec=None,
    )

    with pytest.raises(NotImplementedError, match="UnsupportedProjection"):
        operator._resolve_extent_placeholders()
