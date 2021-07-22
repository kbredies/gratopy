
     
Function reference
==================

.. module:: gratopy


Definition of geometry
----------------------
A  cornerstone in applying projection methods is to define for which geometry the projection has to be computed.
Thus, the first step in using gratopy is always creating an instance of :class:`gratopy.ProjectionSettings` defining the geometry, and thus internally precomputing relevant quantities.

.. autoclass:: gratopy.ProjectionSettings
	:members:


Transforms
----------
The functions :func:`forwardprojection` and :func:`backprojection` perform the projection operations based on the geometry defined in **projectionsetting**. The images **img** and the sinograms **sino** need to be interpreted and 
behave as described in `Getting Started <getting_started.html>`_.

.. autofunction:: gratopy.forwardprojection
.. autofunction:: gratopy.backprojection

Solvers
-------
Based on these forward and backward operators, one can implement a variety of reconstruction algorithms, where the toolbox's focus is on iterative methods (as those in particular are dependent on efficient implementation). 
The following constitute a few easy-to-use examples which also serve as illustration on how gratopy can be included in custom implementations.


.. autofunction:: gratopy.landweber
.. autofunction:: gratopy.conjugate_gradients
.. autofunction:: gratopy.total_variation
.. autofunction:: gratopy.normest
		  
Data generation
---------------

For convinient testing we also include a phantom generator which creates a modified phantom of arbitrary dimension.

.. autofunction:: gratopy.phantom

Internal functions
------------------

In the following we conclude with the documentation of a series of internal functions, which are probably only of interest for more advanced users.  

.. autofunction:: gratopy.radon
.. autofunction:: gratopy.radon_ad
.. autofunction:: gratopy.radon_struct

.. autofunction:: gratopy.fanbeam
.. autofunction:: gratopy.fanbeam_ad
.. autofunction:: gratopy.fanbeam_struct

.. autofunction:: gratopy.create_code



