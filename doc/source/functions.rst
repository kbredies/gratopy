Function reference
==================

.. module:: gratopy


Definition of geometry
----------------------

A cornerstone in applying projection methods is to define for which geometry the projection has to be computed.
Thus, the first step in using gratopy is always creating an instance of :class:`gratopy.ProjectionSettings` defining the geometry, and thus internally precomputing relevant quantities.

.. autoclass:: gratopy.ProjectionSettings
	:exclude-members: ensure_dtype, set_angle_weights
	:members:

Transforms
----------

The functions :func:`forwardprojection` and :func:`backprojection` perform the projection operations based on the geometry defined in **projectionsetting**. The images **img** and the sinograms **sino** need to be interpreted and
behave as described in `Getting started <getting_started.html>`_.

.. autofunction:: gratopy.forwardprojection
.. autofunction:: gratopy.backprojection

Solvers
-------

Based on these forward and backward operators, one can implement a variety of reconstruction algorithms, where the toolbox's focus is on iterative methods (as those in particular are dependent on efficient implementation).
The following constitute a few easy-to-use examples which also serve as illustration on how gratopy can be included in custom :mod:`pyopencl` implementations.


.. autofunction:: gratopy.landweber
.. autofunction:: gratopy.conjugate_gradients
.. autofunction:: gratopy.total_variation
.. autofunction:: gratopy.normest
.. autofunction:: gratopy.weight_sinogram

Data generation
---------------

For convenient testing, a phantom generator is included which creates a modified two-dimensional phantom of arbitrary size.

.. autofunction:: gratopy.phantom

Internal functions
------------------

The following contains the documentation for a set of internal functions which could be of interest for developers. Note that these might be subject to change in the future.

.. autofunction:: gratopy.radon
.. autofunction:: gratopy.radon_ad
.. autofunction:: gratopy.radon_struct

.. autofunction:: gratopy.fanbeam
.. autofunction:: gratopy.fanbeam_ad
.. autofunction:: gratopy.fanbeam_struct

.. autofunction:: gratopy.create_code
