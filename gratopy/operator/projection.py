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
import numpy.typing as npt
import pyopencl as cl
import pyopencl.array as clarray

from copy import copy

from gratopy.gratopy import radon, radon_ad
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

    @property
    def image_domain(self) -> ImageDomain:
        return self.state["image_domain"]
    
    @property
    def angles(self) -> Angles:
        return self.state["angles"]
    
    @property
    def detectors(self) -> Detectors:
        return self.state["detectors"]
    
    @property
    def adjoint(self) -> "Radon":
        operator_copy = copy(self)
        operator_copy.state["adjoint"] = not self.state["adjoint"]
        return operator_copy

    def apply_to(self, argument: npt.ArrayLike, output: clarray.Array | None = None) -> clarray.Array:
        if not isinstance(argument, clarray.Array):
            pass  # TODO: if input is e.g. numpy array, turn into pyopencl array using default queue

        if output is None:
            pass  # TODO: allocate output using same queue as argument

        assert isinstance(argument, clarray.Array)
        assert isinstance(output, clarray.Array)

        if not self.state["adjoint"]:
            radon(
                sino=output,
                img=argument,
                projectionsetting=None,
            )

        else:
            radon_ad(
                sino=argument,
                img=output,
                projectionsetting=None,
            )
        
        return output



class Fanbeam(Operator):
    pass