"""Tests for operator function spaces."""

import copy

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pytest

from gratopy.operator import (
    Fanbeam,
    Operator,
    Radon,
    RayDrivenFanbeam,
    RayDrivenRadon,
    Space,
    StripDrivenRadon,
)
from gratopy.utilities import Detectors, ImageDomain


@pytest.fixture(scope="module")
def queue():
    return cl.CommandQueue(cl.create_some_context(interactive=False))


def make_fanbeam(operator_class=Fanbeam):
    return operator_class(
        image_domain=ImageDomain(size=(24, 20), extent=2.0, center=(0.1, -0.2)),
        angles=30,
        detectors=Detectors(number=40, extent=4.0, center=0.1),
        source_detector_distance=8.0,
        source_origin_distance=4.0,
    )


def make_radon(operator_class=Radon, extent=2.0):
    return operator_class(
        image_domain=ImageDomain(size=(24, 20), extent=extent), angles=25
    )


PROJECTION_OPERATORS = [
    lambda: make_radon(Radon),
    lambda: make_radon(RayDrivenRadon),
    lambda: make_radon(StripDrivenRadon),
    lambda: make_fanbeam(Fanbeam),
    lambda: make_fanbeam(RayDrivenFanbeam),
]


def test_space_validates_arguments():
    with pytest.raises(ValueError, match="scale"):
        Space((3,), scale=0.0)
    with pytest.raises(ValueError, match="out of range"):
        Space((3,), axis_weights=(1, [1.0]))
    with pytest.raises(ValueError, match="shape"):
        Space((3, 2), axis_weights=(0, [1.0, 2.0]))
    with pytest.raises(ValueError, match="positive"):
        Space((2,), axis_weights=(0, [1.0, -1.0]))


def test_space_equality():
    weights = (1, [1.0, 2.0])
    assert Space((3, 2), 0.5, weights) == Space((3, 2), 0.5, weights)
    assert hash(Space((3, 2), 0.5, weights)) == hash(Space((3, 2), 0.5, weights))
    assert Space((3, 2), 0.5, weights) != Space((3, 2), 0.5, (1, [1.0, 3.0]))
    assert Space((3, 2), 0.5, weights) != Space((3, 2), 0.5)
    assert Space((3, 2)) != Space((2, 3))


def test_numpy_inner_product_applies_scale_and_axis_weights():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4, 2))
    y = rng.standard_normal((3, 4, 2))
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    space = Space((3, 4), scale=0.5, axis_weights=(1, weights))

    expected = 0.5 * np.sum(x * y * weights[None, :, None])
    assert space.inner(x, y) == pytest.approx(expected)
    assert space.norm(x) == pytest.approx(
        np.sqrt(0.5 * np.sum(x * x * weights[None, :, None]))
    )


def test_inner_product_rejects_wrong_shapes():
    space = Space((3, 4))
    with pytest.raises(ValueError, match="Shape mismatch"):
        space.inner(np.ones((4, 3)), np.ones((4, 3)))
    with pytest.raises(ValueError, match="Shape mismatch"):
        space.inner(np.ones((3, 4)), np.ones((3, 4, 1)))


@pytest.mark.parametrize("dtype, rel", [(np.float32, 1e-5), (np.float64, 1e-12)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("weight_axis", [0, 1])
@pytest.mark.parametrize("extra_dims", [(), (3,)])
def test_device_inner_product_matches_numpy(
    queue, dtype, rel, order, weight_axis, extra_dims
):
    if dtype == np.float64 and not queue.device.double_fp_config:
        pytest.skip("OpenCL device does not support double precision")
    rng = np.random.default_rng(1)
    shape = (5, 7)
    x = np.require(rng.standard_normal(shape + extra_dims), dtype, order)
    y = np.require(rng.standard_normal(shape + extra_dims), dtype, order)
    weights = rng.uniform(0.5, 2.0, shape[weight_axis])
    space = Space(shape, scale=0.3, axis_weights=(weight_axis, weights))

    device_value = space.inner(clarray.to_device(queue, x), clarray.to_device(queue, y))
    assert device_value == pytest.approx(space.inner(x, y), rel=rel)


def test_device_inner_product_rejects_mixed_orders(queue):
    space = Space((4, 3))
    x = clarray.to_device(queue, np.ones((4, 3), np.float32, order="C"))
    y = clarray.to_device(queue, np.ones((4, 3), np.float32, order="F"))
    with pytest.raises(ValueError, match="same order"):
        space.inner(x, y)


def test_projection_spaces_carry_physical_weights():
    R = make_radon()
    delta_x = 2.0 / 24
    delta_s = float(R.detectors.extent) / R.detectors.number

    assert R.domain == Space((24, 20), scale=delta_x**2)
    assert R.codomain == Space(
        R.output_shape, scale=delta_s, axis_weights=(1, R.angles.weights)
    )


@pytest.mark.parametrize("make_operator", PROJECTION_OPERATORS)
def test_projection_operators_are_adjoint_in_their_spaces(make_operator, queue):
    A = make_operator()
    rng = np.random.default_rng(2)
    x = clarray.to_device(queue, rng.standard_normal(A.input_shape).astype(np.float32))
    y = clarray.to_device(queue, rng.standard_normal(A.output_shape).astype(np.float32))

    assert A.codomain.inner(A * x, y) == pytest.approx(
        A.domain.inner(x, A.T * y), rel=2e-4, abs=2e-5
    )


def test_spaces_propagate_through_expressions():
    R = make_radon()
    R2 = make_radon(RayDrivenRadon)

    assert R.T.domain is R.codomain and R.T.codomain is R.domain
    assert (2 * R).domain is R.domain
    assert (R + R2).codomain == R.codomain
    gram = R.T * R
    assert gram.domain is R.domain and gram.codomain is R.domain


def test_mismatching_spaces_cannot_be_combined():
    R = make_radon(extent=2.0)
    R_wide = make_radon(extent=3.0)
    assert R.input_shape == R_wide.input_shape

    with pytest.raises(ValueError, match="Domain in sum mismatch"):
        R + R_wide
    with pytest.raises(ValueError, match="Space mismatch in composition"):
        R_wide.T * R


def test_custom_operator_spaces():
    operator = Operator(input_shape=(3,), output_shape=(2,))
    assert operator.domain is None and operator.codomain is None

    operator.domain = Space((3,), scale=2.0)
    assert operator.T.codomain == Space((3,), scale=2.0)
    with pytest.raises(ValueError, match="does not match the operator shape"):
        operator.codomain = Space((3,))
    with pytest.raises(TypeError, match="must be a Space"):
        Operator(domain=(3,))


def test_spaces_survive_copies_without_device_buffers(queue):
    R = make_radon()
    sinogram = clarray.to_device(queue, np.ones(R.output_shape, np.float32))
    R.codomain.inner(sinogram, sinogram)
    assert R.codomain._device_weights

    copied = copy.deepcopy(R)
    assert copied.codomain == R.codomain
    assert copied.codomain._device_weights == {}


def test_norm_estimate_is_a_monotone_lower_bound_in_weighted_spaces():
    rng = np.random.default_rng(5)
    matrix = rng.standard_normal((40, 30))
    domain_weights = rng.uniform(0.01, 100.0, 30)
    codomain_weights = rng.uniform(0.1, 10.0, 40)

    class WeightedMatrix(Operator):
        def apply_to(self, argument, output=None, **kwargs):
            return matrix @ argument

        def apply_adjoint_to(self, argument, output=None, **kwargs):
            return (matrix.T @ (codomain_weights * argument)) / domain_weights

    operator = WeightedMatrix(
        input_shape=(30,),
        output_shape=(40,),
        domain=Space((30,), axis_weights=(0, domain_weights)),
        codomain=Space((40,), axis_weights=(0, codomain_weights)),
    )
    exact = np.linalg.norm(
        np.sqrt(codomain_weights)[:, None] * matrix / np.sqrt(domain_weights)[None, :],
        ord=2,
    )

    estimates = [
        operator.norm_estimate(
            number_iterations=k, dtype=np.float64, rng=np.random.default_rng(0)
        )
        for k in range(1, 31)
    ]
    assert np.all(np.diff(estimates) >= -1e-12 * exact)
    assert max(estimates) <= exact * (1 + 1e-12)
    converged = operator.norm_estimate(
        number_iterations=300, dtype=np.float64, rng=np.random.default_rng(0)
    )
    assert converged == pytest.approx(exact, rel=1e-8)
