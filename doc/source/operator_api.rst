Operator syntax
===============

The operator API provides a more compositional interface to gratopy's
projection operators. It is currently **experimental** and primarily focused on
parallel-beam Radon transforms.

The legacy :class:`gratopy.ProjectionSettings` API remains the main documented
interface for the full feature set of gratopy. The operator API complements it
with a syntax that is often more convenient when working with operator algebra,
adjoint operators, and experimental kernels.

.. warning::

   The operator API is **experimental**. Backward-incompatible changes may be
   introduced without a full deprecation cycle while the interface and internal
   abstractions are still settling.

   In particular, extent placeholders such as
   :class:`gratopy.utilities.ExtentPlaceholder` are not yet implemented in the
   operator API and currently raise :class:`NotImplementedError`.

Current scope
-------------

The current operator API supports in particular:

- the :class:`gratopy.operator.projection.Radon` operator,
- adjoints via :attr:`T`,
- operator composition and arithmetic,
- custom OpenCL kernels via :class:`gratopy.operator.opencl.OpenCLKernelSpec`.

At the moment, the operator API should be understood as **Radon-first**.
Fanbeam support is not yet implemented at the same level.

Quick example
-------------

A Radon transform and its adjoint can be used as follows:

.. code-block:: python

    import numpy as np
    import pyopencl as cl
    import gratopy

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    Nx = 128
    img = np.zeros((Nx, Nx), dtype=np.float32)

    R = gratopy.operator.Radon(image_domain=Nx, angles=180)

    sino = R.apply_to(img, queue=queue)
    backprojection = R.T.apply_to(sino)

The same operations can be written with operator syntax:

.. code-block:: python

    sino = R * img
    backprojection = R.T * sino

When the input is a NumPy array, a queue is still required for the first
application so that the data can be transferred to the device.

Detailed geometry example
-------------------------

The operator API also supports explicit geometry helper objects from
:mod:`gratopy.utilities`. This is often the clearest way to specify image
extent, detector geometry, shifts, and angular sampling explicitly.

.. code-block:: python

    import numpy as np
    import pyopencl as cl
    import gratopy
    from gratopy.utilities import Angles, Detectors, ImageDomain

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    img = np.zeros((192, 128), dtype=np.float32)

    image_domain = ImageDomain(
        size=(192, 128),
        extent=3.0,
        center=(0.1, -0.2),
    )

    angles = Angles.uniform_interval(
        start=0.0,
        end=np.pi / 2,
        number=120,
    )

    detectors = Detectors(
        number=220,
        extent=3.0,
        center=0.15,
        reversed=False,
    )

    R = gratopy.operator.Radon(
        image_domain=image_domain,
        angles=angles,
        detectors=detectors,
    )

    sino = R.apply_to(img, queue=queue)
    backproj = R.T.apply_to(sino)

The same setup can also be written directly inline when constructing the
operator:

.. code-block:: python

    R = gratopy.operator.Radon(
        image_domain=ImageDomain(size=(192, 128), extent=3.0, center=(0.1, -0.2)),
        angles=Angles.uniform_interval(0.0, np.pi / 2, 120),
        detectors=Detectors(number=220, extent=3.0, center=0.15),
    )

This explicit style is particularly useful when experimenting with geometry in
Python code, because image domain, angles, and detector settings become
first-class objects that can be reused and modified independently.

Operator algebra
----------------

Operators inherit from :class:`gratopy.operator.base.Operator`, which supports
basic arithmetic and composition. For example, one can form a Gram operator

.. code-block:: python

    G = R.T * R

and apply it to an image:

.. code-block:: python

    gram_img = G.apply_to(img, queue=queue)

This is one of the main motivations for the operator interface: projection
operators can be combined with a syntax that mirrors the underlying linear
algebra.

Class structure
---------------

The operator implementation is intentionally layered in a small number of
classes.

As an **experimental** interface, the operator API may still change in
backward-incompatible ways without a full deprecation cycle while the design is
settling.

:class:`gratopy.operator.base.Operator`
    Provides generic operator arithmetic such as addition, scalar
    multiplication, composition, and application.

:class:`gratopy.operator.opencl._OpenCLOperator`
    Internal helper base for OpenCL-backed operators. It implements shared
    execution plumbing such as queue inference, array coercion, output
    allocation, program cacheing, and kernel lookup.

:class:`gratopy.operator.projection.Radon`
    Concrete Radon transform operator. It owns its operator state,
    geometry-specific preparation, and binds these to the OpenCL kernels.

This separation keeps the generic algebra in :mod:`gratopy.operator.base`
backend-agnostic while concentrating OpenCL-specific behavior in a gratopy-
specific internal layer.

Custom kernels
--------------

One goal of the new operator API is to make kernel experimentation easier.
Kernel sources can be specified via
:class:`gratopy.operator.opencl.OpenCLKernelSpec`.

A custom kernel spec can be passed directly to the operator:

.. code-block:: python

    from gratopy.operator import OpenCLKernelSpec

    spec = OpenCLKernelSpec.from_path("scratch/my_radon.cl", base_name="radon")
    R = gratopy.operator.Radon(image_domain=128, angles=180, kernel_spec=spec)

This allows experimenting with alternative kernels while keeping the Python-side
operator interface unchanged.

Subclassing `_OpenCLOperator`
-----------------------------

For experimental custom operators, the internal class
:class:`gratopy.operator.opencl._OpenCLOperator` provides a default
:py:meth:`apply_to() <gratopy.operator.opencl._OpenCLOperator.apply_to>`
implementation. Although `_OpenCLOperator` is internal, its documented hook
methods form the intended customization surface for OpenCL-backed operators.

The default execution pipeline performs, in order:

1. queue inference,
2. coercion of array-like inputs to :class:`pyopencl.array.Array`,
3. input validation,
4. output allocation (if needed),
5. output validation,
6. kernel lookup,
7. kernel invocation,
8. multiplication by the operator scalar.

Subclasses can adapt this behavior mostly via hooks instead of overriding the
entire method.

Important hooks are:

- :py:meth:`gratopy.operator.opencl._OpenCLOperator._default_kernel_spec`
  for the default kernel bundle,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._expected_output_shape`
  for shape inference,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._validate_argument`
  and
  :py:meth:`gratopy.operator.opencl._OpenCLOperator._validate_output`
  for validation,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._get_kernel`
  for choosing the compiled kernel,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._kernel_arguments`
  for supplying additional kernel arguments beyond output and input buffers,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._global_shape`
  for customizing the OpenCL launch shape.

In simple cases, a custom operator only needs to provide a kernel spec,
`adjoint` / `T` behavior, and static input/output shapes.

Limitations and status
----------------------

The operator API is still evolving. In particular:

- the focus is currently on :class:`gratopy.operator.projection.Radon`,
- extent placeholders are not yet implemented and currently raise
  :class:`NotImplementedError`,
- higher-level solver interfaces are still centered around the legacy API.

For the full and mature feature set of gratopy, the legacy API documented in
:doc:`getting_started` and :doc:`functions` remains the main reference.
