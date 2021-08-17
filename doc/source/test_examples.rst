.. _test-examples:

Test examples
==================

The following documents a number of tests covering essentially
all functions and features contained in the package. These functions
serve the double purpose of showing that the package
is indeed working as desired (via ``pytest`` or ``nosetests``, see :ref:`installation`),
and illustrating to users how to set various parameters of the
gratopy toolbox and what their effect are (cf. the source code for the tests).

The tests are also able to
produce plots of the results. To turn on plotting, the
environment variable ``GRATOPY_TEST_PLOT`` needs to be set, e.g.
the command

::

   GRATOPY_TEST_PLOT=true pytest

can be issued in the ``gratopy`` directory.


Radon transform
---------------------

.. module:: tests.test_radon

.. autofunction:: tests.test_radon.test_projection()
.. autofunction:: tests.test_radon.test_types_contiguity()
.. autofunction:: tests.test_radon.test_weighting()
.. autofunction:: tests.test_radon.test_adjointness()
.. autofunction:: tests.test_radon.test_nonquadratic()
.. autofunction:: tests.test_radon.test_limited_angles()
.. autofunction:: tests.test_radon.test_angle_input_variants()
.. autofunction:: tests.test_radon.test_midpoint_shift()
.. autofunction:: tests.test_radon.test_create_sparse_matrix()


Fanbeam transform
-----------------------

.. module:: tests.test_fanbeam

.. autofunction:: tests.test_fanbeam.test_projection()
.. autofunction:: tests.test_fanbeam.test_types_contiguity()
.. autofunction:: tests.test_fanbeam.test_weighting()
.. autofunction:: tests.test_fanbeam.test_adjointness()
.. autofunction:: tests.test_fanbeam.test_nonquadratic()
.. autofunction:: tests.test_fanbeam.test_limited_angles()
.. autofunction:: tests.test_fanbeam.test_midpoint_shift()
.. autofunction:: tests.test_fanbeam.test_geometric_orientation()
.. autofunction:: tests.test_fanbeam.test_range_check_walnut()
.. autofunction:: tests.test_fanbeam.test_landweber()
.. autofunction:: tests.test_fanbeam.test_conjugate_gradients()
.. autofunction:: tests.test_fanbeam.test_total_variation()
.. autofunction:: tests.test_fanbeam.test_create_sparse_matrix()
