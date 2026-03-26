
import pytest
from gratopy.operator.base import Operator, IDENTITY, ZERO

def test_shape_init():
    op = Operator(input_shape=(10,), output_shape=(20,))
    assert op.input_shape == (10,)
    assert op.output_shape == (20,)

def test_shape_mismatch_sum_input():
    A = Operator(name="A", input_shape=(10,), output_shape=(20,))
    B = Operator(name="B", input_shape=(11,), output_shape=(20,))
    
    with pytest.raises(ValueError, match="Input shape mismatch"):
        _ = A + B

def test_shape_mismatch_sum_output():
    A = Operator(name="A", input_shape=(10,), output_shape=(20,))
    B = Operator(name="B", input_shape=(10,), output_shape=(21,))
    
    with pytest.raises(ValueError, match="Output shape mismatch"):
        _ = A + B

def test_shape_propagation_sum():
    A = Operator(name="A", input_shape=(10,), output_shape=(20,))
    B = Operator(name="B", input_shape=(10,), output_shape=(20,))
    C = A + B
    assert C.input_shape == (10,)
    assert C.output_shape == (20,)

def test_shape_propagation_sum_partial():
    # A has shapes, B has None
    A = Operator(name="A", input_shape=(10,), output_shape=(20,))
    B = Operator(name="B")
    
    C = A + B
    assert C.input_shape == (10,)
    assert C.output_shape == (20,)
    
    D = B + A
    assert D.input_shape == (10,)
    assert D.output_shape == (20,)

def test_shape_propagation_product():
    A = Operator(name="A", input_shape=(20,), output_shape=(30,))
    B = Operator(name="B", input_shape=(10,), output_shape=(20,))
    
    # C(x) = A(B(x))
    C = A * B
    assert C.input_shape == (10,)
    assert C.output_shape == (30,)

def test_shape_mismatch_product():
    A = Operator(name="A", input_shape=(20,), output_shape=(30,))
    B = Operator(name="B", input_shape=(10,), output_shape=(25,)) # Mismatch
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        _ = A * B

def test_shape_propagation_product_chain():
    A = Operator(name="A", input_shape=(20,), output_shape=(30,))
    B = Operator(name="B", input_shape=(15,), output_shape=(20,))
    C = Operator(name="C", input_shape=(10,), output_shape=(15,))
    
    # D = A * B * C
    D = A * B * C
    assert D.input_shape == (10,)
    assert D.output_shape == (30,)

def test_shape_propagation_product_chain_with_none():
    A = Operator(name="A", input_shape=(20,), output_shape=(30,))
    B = Operator(name="B")
    C = Operator(name="C")
    D = Operator(name="D", input_shape=(10,), output_shape=(15,))
    
    E = A * B * C * D
    assert E.input_shape == (10,)
    assert E.output_shape == (30,)

def test_shape_propagation_nested():
    # (A + B) * C
    A = Operator(name="A", input_shape=(20,), output_shape=(30,))
    B = Operator(name="B", input_shape=(20,), output_shape=(30,))
    C = Operator(name="C", input_shape=(10,), output_shape=(20,))
    
    D = (A + B) * C
    assert D.input_shape == (10,)
    assert D.output_shape == (30,)

def test_identity_preserves_shape():
    A = Operator(name="A", input_shape=(10,), output_shape=(20,))
    
    assert (IDENTITY * A).input_shape == (10,)
    assert (IDENTITY * A).output_shape == (20,)
    
    assert (A * IDENTITY).input_shape == (10,)
    assert (A * IDENTITY).output_shape == (20,)
    
    assert (IDENTITY + ZERO) is IDENTITY

def test_zero_preserves_shape():
    A = Operator(name="A", input_shape=(10,), output_shape=(20,))
    
    assert (A + ZERO).input_shape == (10,)
    assert (A + ZERO).output_shape == (20,)
    
    assert (ZERO + A).input_shape == (10,)
    assert (ZERO + A).output_shape == (20,)

