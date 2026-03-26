Reference manual
================

.. module:: gratopy

For the experimental operator-based interface, see :doc:`operator_api`.


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

Operator package
----------------

The operator package provides an experimental operator-oriented interface to a
subset of gratopy's functionality.

.. automodule:: gratopy.operator
   :members:
   :undoc-members:
   :show-inheritance:

Operator algebra
----------------

.. automodule:: gratopy.operator.base
   :members:
   :undoc-members:
   :show-inheritance:

Projection operators
--------------------

.. automodule:: gratopy.operator.projection
   :members:
   :undoc-members:
   :show-inheritance:

OpenCL-backed operator helpers
------------------------------

.. automodule:: gratopy.operator.opencl
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: gratopy.operator.opencl._OpenCLOperator
   :members: apply_to, _default_kernel_spec, _expected_output_shape, _validate_argument, _validate_output, _kernel_arguments, _get_kernel, _global_shape, _infer_queue, _coerce_argument, _allocate_output
   :undoc-members:
   :show-inheritance:

Utility geometry classes
------------------------

The experimental operator API uses a small collection of helper classes for
image domains, detector geometry, angular sampling, and extent placeholders.

.. automodule:: gratopy.utilities
   :members: ExtentPlaceholder, Angles, Detectors, ImageDomain
   :undoc-members:
