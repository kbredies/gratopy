"""Gratopy projection operators.

This module contains concrete implementations of projection operators,
most notably :class:`.Radon` and :class:`.Fanbeam`.

Examples
--------

>>> import gratopy
>>> Nx = 300
>>> phantom = gratopy.easy_phantom(N=Nx)
>>> R = gratopy.operator.Radon(image_domain=Nx, angles=180)
>>> sinogram = R.apply_to(phantom)
>>> backprojection = R.adjoint.apply_to(sinogram)

"""

from __future__ import annotations

import numpy as np

from gratopy.operator.base import Operator
from gratopy.utilities import ImageDomain, Angles, Detectors

class Radon(Operator):
    """A Radon transform operator."""
    def __init__(
        self,
        image_domain: int | tuple[int, int] | ImageDomain,
        angles: Angles | int,
        detectors: Detectors | int | None = None,
        adjoint: bool = False,
    ):
        super().__init__(name="Radon")

        if not isinstance(image_domain, ImageDomain):
            image_domain = ImageDomain(size=image_domain)
        
        if not isinstance(angles, Angles):
            angles = Angles.uniform(number=angles)

        if not isinstance(detectors, Detectors):
            if detectors is None:
                detectors = int(np.ceil(np.hypot(*image_domain.size)))
            detectors = Detectors(number=detectors)

        self.state = {
            "image_domain": image_domain,
            "angles": angles,
            "detectors": detectors,
            "adjoint": adjoint,
        }




class Fanbeam(Operator):
    pass