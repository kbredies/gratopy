
     
Function reference
==================

.. module:: gratopy


Definition of Geometry
----------------------
A  cornerstone in applying projection methods is to define in which geometry the projection are supposed to executed.
Thus the first step in using gratopy is always creating an instance of :class:`gratopy.ProjectionSettings` defining the geometry, and thus internally precomputing relevant quantities.

.. autoclass:: gratopy.ProjectionSettings
	:members:


Transforms
----------
The forwardprojection and backprojection operations execute based on the geometry defined in a projectionsetting the basic forward and backward operations. The images **img** and the sinograms **sino** 
behave and need to be interpreted as described in `Getting Started <getting_started.html>`_.

.. autofunction:: gratopy.forwardprojection
.. autofunction:: gratopy.backprojection

Solvers
-------
Based on these forward and backward operators, one can implement a variety of reconstruction algorithms, where we focus on iterative methods (as those in particular are dependent on efficient implementation). 
These are but a few examples for easy use, which also serve to illustrate how gratopy can included in your implementations.


.. autofunction:: gratopy.landweber
.. autofunction:: gratopy.conjugate_gradients
.. autofunction:: gratopy.normest
		  
Data generation
---------------

For convinient testing we also include a phantom generator which creates a modified phantom of arbitrary dimension.

.. autofunction:: gratopy.phantom

Internal functions
------------------

In the following we conclude with the documentation of a series of internal functions, which however are probably only of interest for more advanced users.  

.. autofunction:: gratopy.radon
.. autofunction:: gratopy.radon_ad
.. autofunction:: gratopy.radon_struct

.. autofunction:: gratopy.fanbeam
.. autofunction:: gratopy.fanbeam_ad
.. autofunction:: gratopy.fanbeam_struct

.. autofunction:: gratopy.create_code


