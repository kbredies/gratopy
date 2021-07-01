.. gratopy documentation master file, created by
   sphinx-quickstart on Wed May  5 20:36:37 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gratopy's documentation!
=====================================

The **Gr**\az **a**\ccelerated **to**\mographic projection for **P**\ython **(Gratopy)**  is a software tool for Python3 developed to allow for efficient, high quality execution of projection methods
such as Radon and fanbeam transform.  The operations contained in the toolbox are based on pixel-driven projection methods, which were shown to possess `suitable approximation properties <https://epubs.siam.org/doi/abs/10.1137/20M1326635>`_.
The code is based in a powerful OpenCL/GPU implementation, resulting in high execution speed, while allowing for seamless integration into `PyOpenCL  <https://documen.tician.de/pyopencl/index.html>`_. 
Hence this can efficiently be paired with other PyOpenCL code, in particular OpenCL based optimization algorithms.

Highlights
==================
* High quality projection implementation.
* Fast projection due to custom OpenCL/GPU-implementation.
* Seamless integration into PyOpenCL code.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   getting_started
   test_examples
   functions
   acknowledgement
	 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
