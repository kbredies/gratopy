.. gratopy documentation master file, created by
   sphinx-quickstart on Wed May  5 20:36:37 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gratopy's documentation!
=====================================

The **Gr**\az **a**\ccelerated **to**\mographic projections for **Py**\thon **(Gratopy)**  is a software toolbox for Python3 developed to allow for efficient, high quality execution of projection methods
such as the Radon and fanbeam transform.  The operators contained in the toolbox are based on pixel-driven projection methods, which were shown to possess `suitable approximation properties <https://epubs.siam.org/doi/abs/10.1137/20M1326635>`_.
The code is based on a powerful parallel OpenCL/GPU implementation, resulting in high execution speed, while allowing for seamless integration into
`PyOpenCL  <https://documen.tician.de/pyopencl/index.html>`_. 
Hence, gratopy can efficiently be combined with other PyOpenCL code, and is well-suited to be used, for instance, in all kinds of tomographic reconstruction approaches, in particular, those involving optimization algorithms.

Highlights
==================
* Easy-to-use tomographic projection toolbox.
* High-quality projection operators.
* Fast projection due to custom OpenCL/GPU-implementation.
* Seamless integration into PyOpenCL.
* Various reconstruction schemes included.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   getting_started
   test_examples
   functions
   acknowledgements
	 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
