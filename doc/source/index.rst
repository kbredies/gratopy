.. gratopy documentation master file, created by
   sphinx-quickstart on Wed May  5 20:36:37 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*************************************
Welcome to gratopy's documentation!
*************************************

The gratopy_ (**Gr**\az **a**\ccelerated **to**\mographic projections for **Py**\thon) toolbox is a Python3 software package for the efficient and high-quality computation of Radon transforms, fanbeam transforms as well as the associated backprojections. The included operators are based on pixel-driven projection methods which were shown to possess `favorable approximation properties <https://epubs.siam.org/doi/abs/10.1137/20M1326635>`_. The toolbox offers a powerful parallel OpenCL/GPU implementation which admits high execution speed and allows for seamless integration into PyOpenCL_. Gratopy can efficiently be combined with other PyOpenCL code and is well-suited for the development of iterative tomographic reconstruction approaches, in particular, for those involving optimization algorithms.

.. _gratopy: https://github.com/kbredies/gratopy/
.. _pyopencl: https://documen.tician.de/pyopencl/

**Highlights**

* Easy-to-use tomographic projection toolbox.
* High-quality 2D projection operators.
* Fast projection due to custom OpenCL/GPU implementation.
* Seamless integration into PyOpenCL.
* Basic iterative reconstruction schemes included (Landweber, CG, total variation).
* Comprehensive documentation, tests and example code.

.. |beginfigref| raw:: latex

                     \begin{minipage}{\textwidth}


.. |endfigref| raw:: latex

                   \end{minipage}

.. table:: The fanbeam projection of a walnut and gratopy's Landweber and total variation reconstructions (from left to right).
    :widths: 20 40 40

    +----------------------------------------+-------------------------------------+------------------------------------------+
    | .. image:: graphics/walnut_sinogram.png| .. image:: graphics/landweber.png   | .. image:: graphics/total_variation.png  |
    |      :height: 250                      |      :height: 250                   |       :height: 250                       |
    |      :align: center                    |      :align: center                 |       :align: center                     |
    +----------------------------------------+-------------------------------------+------------------------------------------+




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
