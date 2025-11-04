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

from gratopy.gratopy import ProjectionSettings, radon, radon_ad
from gratopy.operator.base import Operator
from gratopy.utilities import ImageDomain, Angles, Detectors, GeometryType

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
        self.projection_settings = None

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
    def adjoint(self) -> bool:
        return self.state["adjoint"]
    
    @property
    def T(self) -> "Radon":
        operator_copy = copy(self)
        operator_copy.state = copy(self.state)
        operator_copy.state["adjoint"] = not self.state["adjoint"]
        return operator_copy

    def apply_to(
            self,
            argument: npt.ArrayLike,
            output: clarray.Array | None = None,
            queue: cl.CommandQueue | None = None,
            return_event: bool = False,
        ) -> clarray.Array | tuple[clarray.Array, list[cl.Event]]:
        if not isinstance(argument, clarray.Array):
            pass  # TODO: if input is e.g. numpy array, turn into pyopencl array using default queue

        if output is None:
            if self.adjoint:
                output_shape = self.image_domain.size
            else:
                output_shape = (self.detectors.number, len(self.angles))
            output = clarray.zeros(
                queue=argument.queue,
                shape=output_shape,
                dtype=argument.dtype,
            )  # TODO: adjust shape according to projection

        assert isinstance(argument, clarray.Array)
        assert isinstance(argument.queue, cl.CommandQueue)
        assert isinstance(output, clarray.Array)

        if self.projection_settings is None:
            self.projection_settings = ProjectionSettings(
                queue=argument.queue,
                geometry=GeometryType.RADON,
                img_shape=self.image_domain.size,
                image_width=self.image_domain.extent,
                midpoint_shift=self.image_domain.center,
                angles=self.angles.angles,
                angle_weights=self.angles.weights,
                n_detectors=self.detectors.number,
                detector_width=self.detectors.extent,
                detector_shift=self.detectors.center,
                reverse_detector=self.detectors.reversed,
            )

        if not self.state["adjoint"]:
            radon(
                img=argument,
                sino=output,
                projectionsetting=self.projection_settings,
            )

        else:
            radon_ad(
                sino=argument,
                img=output,
                projectionsetting=self.projection_settings,
            )
        output = self.scalar * output
        return output



class Fanbeam(Operator):
    pass