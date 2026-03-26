
import pytest
from gratopy.operator import Radon

def test_radon_shapes():
    N = 128
    angles = 180
    detectors = 200
    
    R = Radon(image_domain=N, angles=angles, detectors=detectors)
    
    assert R.input_shape == (N, N)
    assert R.output_shape == (detectors, angles)
    
    RT = R.T
    assert RT.input_shape == (detectors, angles)
    assert RT.output_shape == (N, N)
    
    # Composition R.T * R maps image to image
    gram = R.T * R
    assert gram.input_shape == (N, N)
    assert gram.output_shape == (N, N)
    
    # Composition R * R.T maps sinogram to sinogram
    dual_gram = R * R.T
    assert dual_gram.input_shape == (detectors, angles)
    assert dual_gram.output_shape == (detectors, angles)

def test_radon_shape_mismatch():
    R1 = Radon(image_domain=128, angles=180)
    R2 = Radon(image_domain=129, angles=180) # Different image size
    
    # R1 + R2 should fail (input shape mismatch (128,128) vs (129,129))
    with pytest.raises(ValueError, match="Input shape mismatch"):
        _ = R1 + R2
        
    # R1 * R2 impossible because R2 output is sinogram, R1 input is image
    # (unless sinogram shape matches image shape, which is unlikely here)
    # R2 output: (det, 180). R1 input: (128, 128).
    with pytest.raises(ValueError, match="Shape mismatch"):
        _ = R1 * R2

def test_radon_shape_composition_mismatch():
    R1 = Radon(image_domain=128, angles=180)
    # R1 input (128,128), output (det, 180)
    
    R2 = Radon(image_domain=128, angles=100)
    # R2 input (128,128), output (det, 100)
    
    # R1.T * R2
    # R2 output (det, 100). R1.T input (det, 180).
    # Mismatch!
    with pytest.raises(ValueError, match="Shape mismatch"):
        _ = R1.T * R2
