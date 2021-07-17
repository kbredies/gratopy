
     
Test examples
==================

In the following you find a number of references to the additional 
functions contained in the package. These functions
serve the double purpose of showing that the package 
is indeed working as desired (via ``pytest`` or ``nosetests``, see `installation <installation.html>`_),
and illustrating to users how to set various parameters of the 
gratopy toolbox and what their effect are. 
  

Radon transform
---------------------

.. module:: tests.test_radon

.. autoclass:: tests.test_radon.test_projection()
	
.. autoclass:: tests.test_radon.test_weighting()

.. autoclass:: tests.test_radon.test_adjointness()

.. autoclass:: tests.test_radon.test_nonquadratic()

.. autoclass:: tests.test_radon.test_fullangle()



Fanbeam transform
-----------------------

.. module:: tests.test_fanbeam


.. autoclass:: tests.test_fanbeam.test_projection()

.. autoclass:: tests.test_fanbeam.test_weighting()

.. autoclass:: tests.test_fanbeam.test_adjointness()

.. autoclass:: tests.test_fanbeam.test_fullangle()

.. autoclass:: tests.test_fanbeam.test_midpointshift()

.. autoclass:: tests.test_fanbeam.test_landweber()

.. autoclass:: tests.test_fanbeam.test_conjugate_gradients()

.. autoclass:: tests.test_fanbeam.test_total_variation()

.. autoclass:: tests.test_fanbeam.test_nonquadratic()

