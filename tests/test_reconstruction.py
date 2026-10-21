"""Tests for the operator-based reconstruction functions."""

import importlib

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pytest

import gratopy
from gratopy.operator import Operator, Radon, Space
from gratopy.reconstruction import conjugate_gradients, landweber, total_variation


@pytest.fixture(scope="module")
def queue():
    return cl.CommandQueue(cl.create_some_context(interactive=False))


class WeightedMatrixOperator(Operator):
    """Matrix operator whose adjoint is taken in weighted spaces."""

    def __init__(self, matrix, domain, codomain):
        self.matrix = matrix
        super().__init__(
            input_shape=(matrix.shape[1],),
            output_shape=(matrix.shape[0],),
            domain=domain,
            codomain=codomain,
        )

    def _weights(self, space):
        weights = np.full(space.shape, space.scale)
        if space.axis_weights is not None:
            weights = weights * space.axis_weights
        return weights

    def apply_to(self, argument, output=None, **kwargs):
        output[...] = self.matrix @ argument
        return output

    def apply_adjoint_to(self, argument, output=None, **kwargs):
        weighted = self._weights(self.codomain) * argument
        output[...] = (self.matrix.T @ weighted) / self._weights(self.domain)
        return output

    def exact_norm(self):
        scaled = (
            np.sqrt(self._weights(self.codomain))[:, None]
            * self.matrix
            / np.sqrt(self._weights(self.domain))[None, :]
        )
        return np.linalg.norm(scaled, ord=2)


@pytest.fixture
def weighted_problem():
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((40, 10))
    domain = Space((10,), scale=0.5)
    codomain = Space((40,), scale=2.0, axis_weights=(0, rng.uniform(0.1, 10.0, 40)))
    operator = WeightedMatrixOperator(matrix, domain, codomain)
    data = rng.standard_normal(40)

    sqrt_weights = np.sqrt(operator._weights(codomain))
    solution = np.linalg.lstsq(sqrt_weights[:, None] * matrix, sqrt_weights * data)[0]
    unweighted = np.linalg.lstsq(matrix, data)[0]
    assert not np.allclose(solution, unweighted, rtol=1e-2)
    return operator, data, solution


def make_radon():
    return Radon(image_domain=64, angles=40)


def make_legacy_settings(R, queue):
    return gratopy.ProjectionSettings(
        queue,
        gratopy.RADON,
        img_shape=R.image_domain.size,
        angles=R.angles.angles,
        angle_weights=R.angles.weights,
        n_detectors=R.detectors.number,
        image_width=float(R.image_domain.extent),
        detector_width=float(R.detectors.extent),
        midpoint_shift=R.image_domain.center,
        detector_shift=R.detectors.center,
    )


@pytest.fixture
def radon_problem(queue):
    R = make_radon()
    phantom = gratopy.phantom(queue, 64, dtype=np.float32)
    return R, R * phantom


# Tolerances against the legacy solvers, which run the same algorithms with
# differently ordered reductions.
LEGACY_TOLERANCES = {
    "float32": {"landweber": 1e-5, "conjugate_gradients": 1e-3, "total_variation": 1e-5},
    "float64": {
        "landweber": 1e-12,
        "conjugate_gradients": 1e-9,
        "total_variation": 1e-12,
    },
}


@pytest.fixture(params=["float32", "float64"])
def legacy_case(request, queue):
    """A Radon problem in the given precision, with legacy tolerances."""
    dtype = np.dtype(request.param)
    if dtype == np.float64 and not queue.device.double_fp_config:
        pytest.skip("OpenCL device does not support double precision")
    R = make_radon()
    sinogram = R * gratopy.phantom(queue, 64, dtype=dtype)
    return R, sinogram, LEGACY_TOLERANCES[request.param]


def relative_difference(a, b):
    a = a.get() if isinstance(a, clarray.Array) else a
    b = b.get() if isinstance(b, clarray.Array) else b
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# -- Mathematics with NumPy operators -------------------------------------


def test_conjugate_gradients_solves_weighted_least_squares(weighted_problem):
    operator, data, solution = weighted_problem
    x = conjugate_gradients(operator, data, 30)
    np.testing.assert_allclose(x, solution, rtol=1e-8)


def test_landweber_converges_to_weighted_least_squares(weighted_problem):
    operator, data, solution = weighted_problem
    x = landweber(operator, data, 3000, w=1.5, operator_norm=operator.exact_norm())
    np.testing.assert_allclose(x, solution, rtol=1e-6)


def test_conjugate_gradients_stops_at_epsilon(weighted_problem):
    operator, data, _ = weighted_problem
    residuals = []

    def record(iteration, x):
        residual = operator.codomain.norm(operator.matrix @ x - data)
        residuals.append(residual / operator.codomain.norm(data))

    conjugate_gradients(operator, data, 30, epsilon=0.9, callback=record)
    assert residuals[-1] < 0.9
    assert all(r >= 0.9 for r in residuals[:-1])


def test_callback_stops_iteration(weighted_problem):
    operator, data, _ = weighted_problem
    norm = operator.exact_norm()
    calls = []

    def stop_after_three(iteration, x):
        calls.append(iteration)
        return iteration == 3

    x = landweber(operator, data, 50, operator_norm=norm, callback=stop_after_three)
    assert calls == [1, 2, 3]
    np.testing.assert_allclose(x, landweber(operator, data, 3, operator_norm=norm))


def test_landweber_continues_from_x0_without_modifying_it(weighted_problem):
    operator, data, _ = weighted_problem
    norm = operator.exact_norm()
    first = landweber(operator, data, 5, operator_norm=norm)
    first_copy = first.copy()

    continued = landweber(operator, data, 5, operator_norm=norm, x0=first)
    np.testing.assert_allclose(
        continued, landweber(operator, data, 10, operator_norm=norm)
    )
    np.testing.assert_array_equal(first, first_copy)


def test_operators_without_spaces_use_euclidean_inner_products():
    rng = np.random.default_rng(4)
    matrix = rng.standard_normal((30, 8))
    data = rng.standard_normal(30)

    class PlainMatrixOperator(Operator):
        def apply_to(self, argument, output=None, **kwargs):
            output[...] = matrix @ argument
            return output

        def apply_adjoint_to(self, argument, output=None, **kwargs):
            output[...] = matrix.T @ argument
            return output

    operator = PlainMatrixOperator(input_shape=(8,), output_shape=(30,))
    x = conjugate_gradients(operator, data, 20)
    np.testing.assert_allclose(x, np.linalg.lstsq(matrix, data)[0], rtol=1e-8)


def test_input_validation(weighted_problem):
    operator, data, _ = weighted_problem
    with pytest.raises(ValueError, match="Data shape mismatch"):
        landweber(operator, np.zeros(39), operator_norm=1.0)
    with pytest.raises(TypeError, match="float32 or float64"):
        landweber(operator, np.zeros(40, dtype=int), operator_norm=1.0)
    with pytest.raises(ValueError, match="x0 shape mismatch"):
        conjugate_gradients(operator, data, x0=np.zeros(9))
    with pytest.raises(ValueError, match="number_iterations"):
        conjugate_gradients(operator, data, -1)
    with pytest.raises(ValueError, match="w must be positive"):
        landweber(operator, data, w=0.0)
    with pytest.raises(ValueError, match="operator_norm must be positive"):
        landweber(operator, data, operator_norm=0.0)
    with pytest.raises(ValueError, match="mu must be positive"):
        total_variation(operator, data, mu=-1.0)
    with pytest.raises(ValueError, match="two-dimensional input shape"):
        total_variation(operator, data, mu=1.0)


# -- Projection operators on the device -----------------------------------


@pytest.fixture
def fixed_legacy_norm(monkeypatch):
    """Make the legacy solvers use a given norm instead of a random estimate."""
    # ``import gratopy.gratopy`` yields the package, as gratopy/__init__.py
    # rebinds the name; patch the module the legacy solvers actually use.
    legacy = importlib.import_module("gratopy.gratopy")
    assert gratopy.landweber.__globals__ is vars(legacy)

    def fix(norm):
        monkeypatch.setattr(legacy, "normest", lambda *args, **kwargs: norm)

    return fix


def test_landweber_matches_legacy(legacy_case, queue, fixed_legacy_norm):
    R, sinogram, tolerance = legacy_case
    norm = R.norm_estimate(queue=queue)
    fixed_legacy_norm(norm)

    x = landweber(R, sinogram, 50, operator_norm=norm)
    legacy_x = gratopy.landweber(sinogram, make_legacy_settings(R, queue), 50)
    assert x.dtype == sinogram.dtype
    assert relative_difference(x, legacy_x) < tolerance["landweber"]


def test_conjugate_gradients_matches_legacy(legacy_case, queue):
    R, sinogram, tolerance = legacy_case
    x = conjugate_gradients(R, sinogram, 10)
    legacy_x = gratopy.conjugate_gradients(sinogram, make_legacy_settings(R, queue), 10)
    assert x.dtype == sinogram.dtype
    assert relative_difference(x, legacy_x) < tolerance["conjugate_gradients"]


def test_total_variation_matches_legacy(legacy_case, queue, fixed_legacy_norm):
    R, sinogram, tolerance = legacy_case
    norm = R.norm_estimate(queue=queue)
    fixed_legacy_norm(norm)

    x = total_variation(R, sinogram, 100.0, 200, operator_norm=norm)
    legacy_x = gratopy.total_variation(
        sinogram, make_legacy_settings(R, queue), 100.0, 200
    )
    assert x.dtype == sinogram.dtype
    assert relative_difference(x, legacy_x) < tolerance["total_variation"]
    assert x.get().min() >= 0


def test_numpy_data_with_queue_returns_device_array(radon_problem, queue):
    R, sinogram = radon_problem
    x = conjugate_gradients(R, sinogram.get(), 5, queue=queue)
    assert isinstance(x, clarray.Array)
    assert relative_difference(x, conjugate_gradients(R, sinogram, 5)) < 1e-6


def test_numpy_data_without_queue_raises(radon_problem):
    R, sinogram = radon_problem
    with pytest.raises(ValueError, match="No OpenCL queue available"):
        conjugate_gradients(R, sinogram.get(), 5)
    with pytest.raises(ValueError, match="queue="):
        total_variation(R, sinogram.get(), 1.0, operator_norm=1.0)


def test_memory_order_does_not_change_the_result(radon_problem, queue):
    R, sinogram = radon_problem
    fortran_sinogram = clarray.to_device(queue, np.asfortranarray(sinogram.get()))

    x = conjugate_gradients(R, fortran_sinogram, 10)
    assert x.flags.f_contiguous
    assert relative_difference(x, conjugate_gradients(R, sinogram, 10)) < 1e-4


def test_scaled_operator_uses_propagated_spaces(radon_problem):
    R, sinogram = radon_problem
    x = conjugate_gradients(R, sinogram, 10)
    x_scaled = conjugate_gradients(2.0 * R, sinogram, 10)
    assert relative_difference(2.0 * x_scaled.get(), x) < 1e-4


@pytest.mark.parametrize("solver", ["landweber", "total_variation"])
def test_slice_stacks_are_reconstructed_slicewise(radon_problem, queue, solver):
    R, sinogram = radon_problem
    single = sinogram.get()
    stack = clarray.to_device(queue, np.stack([single, 0.5 * single], axis=-1))

    if solver == "landweber":
        reconstruct = lambda data: landweber(R, data, 20, operator_norm=1.0)  # noqa: E731
    else:
        reconstruct = lambda data: total_variation(  # noqa: E731
            R, data, 100.0, 50, slice_thickness=0.0, operator_norm=1.0
        )

    x = reconstruct(stack)
    assert x.shape == R.input_shape + (2,)
    for index, factor in enumerate((1.0, 0.5)):
        expected = reconstruct(clarray.to_device(queue, factor * single))
        assert relative_difference(x.get()[..., index], expected) < 1e-5


def test_total_variation_couples_slices(radon_problem, queue):
    R, sinogram = radon_problem
    single = sinogram.get()
    stack = clarray.to_device(queue, np.stack([single, 0.5 * single], axis=-1))

    independent = total_variation(
        R, stack, 100.0, 50, slice_thickness=0.0, operator_norm=1.0
    )
    coupled = total_variation(R, stack, 100.0, 50, slice_thickness=1.0, operator_norm=1.0)
    assert relative_difference(coupled, independent) > 1e-3
