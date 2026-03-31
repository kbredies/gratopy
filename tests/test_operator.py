import numpy as np
import pytest
import pyopencl as cl
import pyopencl.array as clarray

from gratopy.operator.base import IDENTITY, ZERO, Operator
from gratopy.operator import Radon
from gratopy.utilities import Detectors, ExtentPlaceholder, ImageDomain


def test_identity_repr():
    assert repr(IDENTITY) == "[Id]"


def test_identity_apply_to():
    assert IDENTITY.apply_to([1, 2, 3]) == [1, 2, 3]
    assert IDENTITY * [1, 2, 3] == [1, 2, 3]


def test_zero_repr():
    assert repr(ZERO) == "[0]"


def test_zero_apply_to():
    data = np.array([1, 2, 3])

    np.testing.assert_equal(ZERO.apply_to(data), np.zeros_like(data))
    np.testing.assert_equal(ZERO * data, np.zeros_like(data))


def test_zero_addition():
    assert ZERO + ZERO == ZERO
    assert ZERO - ZERO == ZERO
    assert ZERO + IDENTITY == IDENTITY
    assert IDENTITY + ZERO == IDENTITY
    assert IDENTITY - ZERO == IDENTITY


def test_zero_scalar_multiplication():
    assert 0 * ZERO == ZERO
    assert 42 * ZERO == ZERO


def test_zero_multiplication():
    assert ZERO * ZERO == ZERO
    assert ZERO * IDENTITY == ZERO
    assert IDENTITY * ZERO == ZERO


def test_identity_multiplication():
    A = Operator(name="A")

    assert IDENTITY * A == A
    assert A * IDENTITY == A


def test_operator_representation():
    A = Operator(name="A")
    B = Operator(name="B")
    C = Operator(name="C")

    assert repr(A) == "A"
    assert repr(B) == "B"
    assert repr(A * B) == "A*B"
    assert repr(A + B) == "A + B"
    assert repr(5 * (A + IDENTITY) - B) == "5*A + 5*[Id] + (-1)*B"
    assert repr(5 * (A + IDENTITY) - B) == "5*A + 5*[Id] + (-1)*B"
    assert repr(5 * (A + IDENTITY) * C - B) == "(5*A + 5*[Id])*C + (-1)*B"
    assert repr(A * B * A * B * A * B) == "A*B*A*B*A*B"
    assert repr(5 * A * B * A * B * A * B) == "5*A*B*A*B*A*B"
    assert repr(5 * (A * B * B * A)) == "5*A*B*B*A"


def test_operator_composition():
    from gratopy.operator.base import OperatorArithmeticOperation

    A = Operator(name="A")
    B = Operator(name="B")

    composed_op = 5 * (A + IDENTITY) - B
    assert composed_op.is_composite()
    assert composed_op._arithmetic_operation == OperatorArithmeticOperation.ADDITION
    assert len(composed_op._operands) == 3


def test_operator_arithmetic_references():
    A = Operator(name="A")
    B = Operator(name="B")

    assert 5 * (A + B + A) == 5 * A + 5 * B + 5 * A
    assert 5 * (A * B * B * A) == 5 * A * B * B * A


def test_radon_numpy_coercion():
    """Test that Radon.apply_to() correctly converts NumPy arrays to clarray."""
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    Nx = 16
    n_angles = 10
    R = Radon(image_domain=Nx, angles=n_angles)

    # Test with zero array and explicit queue
    img_np = np.zeros((Nx, Nx), dtype=np.float32)
    sinogram = R.apply_to(img_np, queue=queue)

    assert isinstance(sinogram, clarray.Array)
    assert sinogram.shape == (R.detectors.number, n_angles)
    np.testing.assert_array_equal(sinogram.get(), 0.0)


def test_radon_numpy_coercion_clarray_input():
    """Test that Radon.apply_to() works with clarray input without explicit queue."""
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    Nx = 16
    n_angles = 10
    R = Radon(image_domain=Nx, angles=n_angles)

    img_np = np.zeros((Nx, Nx), dtype=np.float32)
    img_cl = clarray.to_device(queue, img_np)
    sinogram = R.apply_to(img_cl)

    assert isinstance(sinogram, clarray.Array)
    assert sinogram.shape == (R.detectors.number, n_angles)


def test_radon_numpy_coercion_reuses_queue():
    """Test that Radon.apply_to() reuses the queue from projection_settings."""
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    Nx = 16
    n_angles = 10
    R = Radon(image_domain=Nx, angles=n_angles)

    img_np = np.zeros((Nx, Nx), dtype=np.float32)

    # First call with explicit queue
    R.apply_to(img_np, queue=queue)
    assert R.projection_settings is not None

    # Second call without queue should reuse the stored queue
    sinogram2 = R.apply_to(img_np)
    assert isinstance(sinogram2, clarray.Array)


def test_radon_numpy_coercion_no_queue_error():
    """Test that Radon.apply_to() raises ValueError when no queue is available."""
    Nx = 16
    R = Radon(image_domain=Nx, angles=10)

    img_np = np.zeros((Nx, Nx), dtype=np.float32)

    with pytest.raises(ValueError, match="No OpenCL queue available"):
        R.apply_to(img_np)


def test_composite_operator_forwards_queue():
    """Test that composite operators forward queue kwargs to child operators."""
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    Nx = 16
    R = Radon(image_domain=Nx, angles=10)
    gram = R.T * R

    img_np = np.zeros((Nx, Nx), dtype=np.float32)
    result = gram.apply_to(img_np, queue=queue)

    assert isinstance(result, clarray.Array)
    assert result.shape == (Nx, Nx)


def test_radon_placeholder_rejected_for_detector_extent():
    with pytest.raises(NotImplementedError, match="Extent placeholders"):
        Radon(
            image_domain=ImageDomain(size=16, extent=2.0),
            angles=10,
            detectors=Detectors(number=20, extent=ExtentPlaceholder.FULL),
        )


def test_radon_placeholder_rejected_for_image_extent():
    with pytest.raises(NotImplementedError, match="Extent placeholders"):
        Radon(
            image_domain=ImageDomain(size=16, extent=ExtentPlaceholder.FULL),
            angles=10,
            detectors=Detectors(number=20, extent=2.0),
        )
