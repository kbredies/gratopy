Test examples
==================

The following documents a number of tests covering essentially
all functions and features contained in the package. These functions
serve the double purpose of showing that the package
is indeed working as desired (via ``pytest`` or ``nosetests``, see `installation <installation.html>`_),
and illustrating to users how to set various parameters of the
gratopy toolbox and what their effect are (cf. the source code for the tests).


Radon transform
---------------------

.. module:: tests.test_radon

.. autofunction:: tests.test_radon.test_projection()
.. autofunction:: tests.test_radon.test_weighting()
.. autofunction:: tests.test_radon.test_adjointness()
.. autofunction:: tests.test_radon.test_nonquadratic()
.. autofunction:: tests.test_radon.test_fullangle()
.. autofunction:: tests.test_radon.test_midpoint_shift()
.. autofunction:: tests.test_radon.test_extract_sparse_matrix()


Fanbeam transform
-----------------------

.. module:: tests.test_fanbeam

.. autofunction:: tests.test_fanbeam.test_projection()
.. autofunction:: tests.test_fanbeam.test_weighting()
.. autofunction:: tests.test_fanbeam.test_adjointness()
.. autofunction:: tests.test_fanbeam.test_fullangle()
.. autofunction:: tests.test_fanbeam.test_midpoint_shift()
.. autofunction:: tests.test_fanbeam.test_range_check_walnut()
.. autofunction:: tests.test_fanbeam.test_landweber()
.. autofunction:: tests.test_fanbeam.test_conjugate_gradients()
.. autofunction:: tests.test_fanbeam.test_total_variation()
.. autofunction:: tests.test_fanbeam.test_nonquadratic()
.. autofunction:: tests.test_fanbeam.test_extract_sparse_matrix()
