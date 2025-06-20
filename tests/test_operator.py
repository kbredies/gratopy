import pytest

import numpy as np

from gratopy.operator import IDENTITY, ZERO, Operator


def test_identity_repr():
    from gratopy.operator import IDENTITY

    assert repr(IDENTITY) == "[Id]"

def test_identity_apply_to():
    from gratopy.operator import IDENTITY

    assert IDENTITY.apply_to([1, 2, 3]) == [1, 2, 3]
    assert IDENTITY * [1, 2, 3] == [1, 2, 3]

def test_zero_repr():
    from gratopy.operator import ZERO

    assert repr(ZERO) == "[0]"

def test_zero_apply_to():
    from gratopy.operator import ZERO

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
    assert repr(5*(A + IDENTITY) - B) == "5*A + 5*[Id] + (-1)*B"
    assert repr(5*(A + IDENTITY) - B) == "5*A + 5*[Id] + (-1)*B"
    assert repr(5*(A + IDENTITY)*C - B) == "(5*A + 5*[Id])*C + (-1)*B"
    assert repr(A*B*A*B*A*B) == "A*B*A*B*A*B"
    assert repr(5*A*B*A*B*A*B) == "5*A*B*A*B*A*B"
    assert repr(5*(A*B*B*A)) == "5*A*B*B*A"

def test_operator_composition():
    from gratopy.operator import OperatorArithmeticOperation

    A = Operator(name="A")
    B = Operator(name="B")

    composed_op = 5*(A + IDENTITY) - B
    assert composed_op.is_composite()
    assert composed_op._arithmetic_operation == OperatorArithmeticOperation.ADDITION
    assert len(composed_op._operands) == 3


def test_operator_arithmetic_references():
    A = Operator(name="A")
    B = Operator(name="B")

    assert 5*(A + B + A) == 5*A + 5*B + 5*A
    assert 5*(A*B*B*A) == 5*A*B*B*A