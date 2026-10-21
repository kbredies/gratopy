"""Reconstruction algorithms for gratopy operators.

All functions take an :class:`~gratopy.operator.Operator` and data in its
output space and return the reconstruction. They work with projection
operators as well as with operator expressions and custom operators; inner
products are taken in the operator's :attr:`~gratopy.operator.Operator.domain`
and :attr:`~gratopy.operator.Operator.codomain`.

>>> from gratopy.operator import Radon
>>> from gratopy.reconstruction import conjugate_gradients
>>> R = Radon(image_domain=256, angles=180)
>>> x = conjugate_gradients(R, sinogram, 50)  # doctest: +SKIP
"""

from gratopy.reconstruction.least_squares import conjugate_gradients, landweber
from gratopy.reconstruction.variational import total_variation

__all__ = ["conjugate_gradients", "landweber", "total_variation"]
