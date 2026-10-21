Operator syntax
===============

The experimental operator API provides a compositional interface to gratopy's
parallel-beam and fan-beam projections. It is designed for projection geometry,
adjoints, operator arithmetic, reusable outputs, and custom OpenCL kernels.

Available projection operators
------------------------------

The following projection discretizations are available:

- :class:`gratopy.operator.projection.Radon` uses the pixel-driven
  parallel-beam kernels.
- :class:`gratopy.operator.projection.RayDrivenRadon` uses ray-driven
  parallel-beam kernels.
- :class:`gratopy.operator.projection.StripDrivenRadon` uses strip-driven
  parallel-beam kernels.
- :class:`gratopy.operator.projection.Fanbeam` uses the pixel-driven fan-beam
  kernels.
- :class:`gratopy.operator.projection.RayDrivenFanbeam` uses ray-driven
  fan-beam kernels.

Every projection operator provides its adjoint through :attr:`T` and supports
composition, addition, scaling, output reuse, norm estimation, and custom
kernels through :class:`gratopy.operator.opencl.OpenCLKernelSpec`. Expressions
share the immutable geometry and OpenCL runtime caches of their projection
leaves.

Quick example
-------------

A Radon transform and its adjoint can be used as follows:

.. code-block:: python

    import numpy as np
    import pyopencl as cl
    import pyopencl.array as clarray
    import gratopy

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    Nx = 128
    img = np.zeros((Nx, Nx), dtype=np.float32)

    R = gratopy.operator.Radon(image_domain=Nx, angles=180)

    sino = R.apply_to(img, queue=queue)
    backprojection = R.T.apply_to(sino)

A fan-beam transform is defined by an explicit detector geometry, the
source-to-detector distance, and the source-to-origin distance:

.. code-block:: python

    from gratopy.utilities import Detectors

    F = gratopy.operator.Fanbeam(
        image_domain=Nx,
        angles=360,
        detectors=Detectors(number=192, extent=3.0),
        source_detector_distance=8.0,
        source_origin_distance=4.0,
    )
    fan_sino = F.apply_to(img, queue=queue)
    fan_backprojection = F.T.apply_to(fan_sino)

``source_detector_distance`` measures the orthogonal distance from the source
to the detector line. ``source_origin_distance`` measures the distance from the
source to the rotation center. The source-to-detector distance is larger than
the source-to-origin distance, and the source lies outside the image domain.
Integer angle counts produce a full-circle sampling for fan-beam operators.

The same operations can be written with operator syntax when the arrays
already reside on the OpenCL device:

.. code-block:: python

    device_img = clarray.to_device(queue, img)
    sino = R * device_img
    backprojection = R.T * sino

Queue selection follows a fixed precedence: an explicit ``queue=`` is used
first, followed by the queue associated with a device argument and then the
queue associated with a device output. Apply NumPy arrays with ``queue=`` or a
caller-provided :class:`pyopencl.array.Array` output. Device arrays support the
multiplication shorthand because they carry their queue.

Output reuse and events
-----------------------

Applications allocate a result when ``output`` is omitted. Allocation-sensitive
iterative code can provide a compatible device output instead:

.. code-block:: python

    output = clarray.empty(queue, R.output_shape, dtype=np.float32)
    R.apply_to(device_img, output=output)

Compositions forward ``output`` to their final operation, while sums write
their first summand directly into it before accumulating the remaining terms.
Intermediate results use temporary device arrays. Reusing outputs is
recommended in allocation-sensitive iterative loops.

OpenCL execution is asynchronous. Passing ``return_event=True`` returns
``(result, events)`` with the result's current event list in addition to
recording those events on the result array.

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

Image domains, angles, and detector settings are immutable values that can be
reused by multiple operators. Construct modified geometry with a new value, for
example by using :func:`dataclasses.replace`. ``Angles`` stores private,
read-only copies of its angle and weight arrays.

Extent inference
----------------

One physical extent can be inferred from the other by using
:class:`gratopy.utilities.ExtentPlaceholder`. This is useful when one
wants either the smallest detector covering a fixed image domain, or the
largest image domain covered by a fixed detector.

For example, the detector extent can be inferred from a fixed image extent:

.. code-block:: python

    from gratopy.utilities import Detectors, ExtentPlaceholder, ImageDomain

    R = gratopy.operator.Radon(
        image_domain=ImageDomain(size=128, extent=2.0),
        angles=180,
        detectors=Detectors(number=200, extent=ExtentPlaceholder.FULL),
    )

Conversely, the image extent can be inferred from a fixed detector extent:

.. code-block:: python

    R = gratopy.operator.Radon(
        image_domain=ImageDomain(size=128, extent=ExtentPlaceholder.FULL),
        angles=180,
        detectors=Detectors(number=200, extent=2.0),
    )

Fan-beam operators support placeholders in the same way:

.. code-block:: python

    F = gratopy.operator.Fanbeam(
        image_domain=ImageDomain(size=128, extent=2.0),
        angles=360,
        detectors=Detectors(number=192, extent=ExtentPlaceholder.FULL),
        source_detector_distance=8.0,
        source_origin_distance=4.0,
    )

Use one placeholder together with one numeric extent. ``FULL`` chooses the
extent such that every ray through the image domain hits the detector, while
``VALID`` chooses it such that every ray hitting the detector intersects the
image domain. If no such extent exists for the configured image and detector
centers, construction raises a :class:`ValueError`.

Radon placeholders take the angular coverage into account, as given by the
``half_circle`` flag of :class:`~gratopy.utilities.Angles`. Fan-beam
placeholders are always resolved for a full-circle scan; for limited-angle
fan-beam scans, choose numeric extents explicitly.

Geometry visualization
----------------------

Projection operators can draw their physical geometry via
:meth:`~gratopy.operator.projection._ProjectionOperator.show_geometry`. The
drawing shows the image domain, the detector (with an arrow towards increasing
pixel indices, a marker at pixel 0 and a tick at the detector center), sample
rays and, for fan beams, the source and its trajectory. Rays missing the image
domain are drawn dashed in red, which makes it easy to check the coverage
resulting from an :class:`~gratopy.utilities.ExtentPlaceholder`. No OpenCL
queue is required.

.. code-block:: python

    import matplotlib.pyplot as plt
    from gratopy.utilities import Detectors, ExtentPlaceholder, ImageDomain

    F = gratopy.operator.Fanbeam(
        image_domain=ImageDomain(size=128, extent=ExtentPlaceholder.VALID),
        angles=360,
        detectors=Detectors(number=192, extent=3.0),
        source_detector_distance=8.0,
        source_origin_distance=4.0,
    )
    F.show_geometry()          # first angle of the operator, in a new figure
    plt.show()

Several angles are overlaid in one axes, and existing axes can be drawn into,
e.g., to compare geometries side by side:

.. code-block:: python

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    R.show_geometry(np.pi / 4, ax=axes[0], legend=False)
    F.show_geometry([0.0, np.pi / 2], ax=axes[1], legend=False)
    plt.show()

Adjoint convention
------------------

Projection adjoints are defined with respect to gratopy's physical image and
sinogram pairings. Angular quadrature weights from
:class:`gratopy.utilities.Angles` are included in each backprojection kernel.
Consequently, ``R.T`` and ``F.T`` can be used directly in variational methods
and normal operators such as ``R.T * R`` with the configured quadrature.

Operator algebra
----------------

Operators inherit from :class:`gratopy.operator.base.Operator`, which builds
non-mutating expression nodes for arithmetic, scaling, adjoints, and
composition. Expression nodes retain references to their original operands;
they do not copy concrete operators or their runtime caches. For example, one
can form a Gram operator

.. code-block:: python

    G = R.T * R

and apply it to an image:

.. code-block:: python

    gram_img = G.apply_to(img, queue=queue)

This is one of the main motivations for the operator interface: projection
operators can be combined with a syntax that mirrors the underlying linear
algebra. The multiplication syntax is intentionally overloaded:

- ``A * B`` composes two operators,
- ``alpha * A`` and ``A * alpha`` scale an operator,
- ``A * x`` applies an operator to a non-operator argument.

Addition and subtraction construct sum expressions. Algebra creates dedicated
expression nodes and never mutates or copies concrete leaves.

Norm estimation
---------------

Every operator provides :meth:`gratopy.operator.base.Operator.norm_estimate`.
The default ``"poweriteration"`` algorithm applies power iteration to
``A.T * A`` and supports OpenCL operators via an explicit queue:

.. code-block:: python

    estimate = R.norm_estimate(queue=queue, number_iterations=30)

The alternative ``"naive"`` algorithm combines leaf values structurally using
the triangle inequality for sums and submultiplicativity of operator norms for
compositions. These inequalities preserve certified upper bounds, but unknown
leaf norms are currently obtained from finite power iterations and are not
themselves certified upper bounds. The resulting combined value is therefore a
heuristic unless certified bounds are available for every leaf.

Function spaces
---------------

Every operator has a :attr:`~gratopy.operator.base.Operator.domain` and a
:attr:`~gratopy.operator.base.Operator.codomain`, given as
:class:`gratopy.operator.space.Space` objects (or ``None`` if unspecified,
which solvers treat as Euclidean). A space carries the weighted inner product
in which :attr:`T` is the adjoint. For projection operators, images are
weighted by the pixel area :math:`\delta_x^2` and sinograms by the detector
pixel width :math:`\delta_s` and the angle weights:

.. code-block:: python

    residual = R.codomain.norm(R * x - sino)       # physically weighted norm
    pairing = R.domain.inner(x, R.T * sino)        # equals R.codomain.inner(R * x, sino)

Spaces propagate through operator algebra: the adjoint swaps domain and
codomain, compositions take the domain of their last and the codomain of their
first operand, and combining operators with different (specified) spaces
raises a :class:`ValueError`. Custom operators can declare their spaces via
``domain=`` and ``codomain=`` or by assigning the attributes.

Reconstruction
--------------

:mod:`gratopy.reconstruction` provides the reconstruction algorithms
:func:`~gratopy.reconstruction.landweber`,
:func:`~gratopy.reconstruction.conjugate_gradients`, and
:func:`~gratopy.reconstruction.total_variation`. They take the operator first
and work with any operator, including expressions and custom operators:

.. code-block:: python

    from gratopy.reconstruction import landweber, conjugate_gradients, total_variation

    x_lw = landweber(R, sino, 200)
    x_cg = conjugate_gradients(R, sino, 50, epsilon=1e-4)
    x_tv = total_variation(R, sino, mu=20.0, number_iterations=2000)

Inputs follow the operator conventions: the queue is taken from ``queue=``,
the data, or ``x0``; dtype and memory order follow the data; trailing
dimensions are treated as a stack of slices. The algorithms never print
progress. Instead, ``callback(iteration, x)`` is called after every iteration
and stops the iteration by returning ``True``:

.. code-block:: python

    def monitor(iteration, x):
        if iteration % 10 == 0:               # each check synchronizes with the host
            return R.codomain.norm(R * x - sino) < 1e-3 * R.codomain.norm(sino)

    norm = R.norm_estimate(queue=queue)       # estimate once, reuse across calls
    x = landweber(R, sino, 1000, operator_norm=norm, callback=monitor)
    x = landweber(R, sino, 100, operator_norm=norm, x0=x)   # continue

As in :func:`gratopy.total_variation`, the total-variation reconstruction is
constrained to be nonnegative.

Class structure
---------------

The operator implementation is layered in a small number of classes.

:class:`gratopy.operator.base.Operator`
    Provides the common operator interface and constructs dedicated adjoint,
    scale, sum, and composition expression nodes. Concrete operands are shared
    across the expression tree rather than copied.

:class:`gratopy.operator.opencl._OpenCLOperator`
    Internal helper base for OpenCL-backed operators. It implements shared
    execution plumbing such as queue inference, array coercion, output
    allocation, program cacheing, and kernel lookup.

:class:`gratopy.operator.projection.Radon` and :class:`gratopy.operator.projection.Fanbeam`
    Concrete projection operators. They own immutable geometry state,
    geometry-specific preparation, and bindings to the OpenCL kernels. The
    ray- and strip-driven classes are dedicated subclasses selecting alternate
    kernel bundles while preserving the same execution and cache lifecycle.

This separation keeps the generic algebra in :mod:`gratopy.operator.base`
backend-agnostic while concentrating OpenCL-specific behavior in a gratopy-
specific internal layer.

Compiled-program lifecycle
--------------------------

Compiled OpenCL programs are shared across concrete operators using the same
context, kernel source, build options, and template-expansion mode. The global
registry retains these program bundles weakly; each operator that has actually
used a bundle retains a strong runtime lease. Consequently, equivalent live
operators share compilation work, while a bundle is released automatically
after its last operator lease disappears. Adjoint and composite expressions
retain their concrete leaves and therefore participate in the same lifecycle.

Kernel instances are local to each calling thread because OpenCL kernel
arguments are mutable. Threads share compiled programs while maintaining
independent kernel argument state.

The registry can be invalidated explicitly when required:

.. code-block:: python

    from gratopy.operator import invalidate_kernel_cache

    invalidate_kernel_cache()

Live operators acquire newly compiled bundles on their next application.
Invocations already in progress may finish with their existing program.
Invalidation removes bundles from future lookup; an existing operator may keep
its old lease alive until its next application or until the operator is
released.

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
3. direction-aware input validation,
4. output allocation (if needed),
5. output validation,
6. forward or adjoint kernel lookup,
7. kernel invocation.

Scalar multiplication is represented by a dedicated expression node instead
of mutating or copying the concrete OpenCL operator.

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

In simple cases, a custom operator only needs to provide a kernel spec and
static input/output shapes. The shared OpenCL implementation dispatches the
forward and adjoint kernels through :meth:`apply_to` and
:meth:`apply_adjoint_to`; :attr:`T` is an adjoint expression wrapper around the
same concrete operator and therefore shares its runtime caches.
