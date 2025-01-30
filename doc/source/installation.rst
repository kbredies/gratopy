.. _installation:

Installation
============

Stable releases can be found on the
`GitHub Release page <https://github.com/kbredies/gratopy/releases>`__
and are distributed via the
`Python Packaging Index, PyPI <https://pypi.org/project/gratopy/>`__.


.. _pip: https://pypi.org/project/pip/

Installation in Python
----------------------

The gratopy toolbox can easily be installed using pip_
::

    pip install gratopy

Alternatively, the release can be downloaded from https://github.com/kbredies/gratopy and installed (after unpacking inside the corresponding folder) via
::

    pip install .

In case installation fails due to the dependency on other packages
(see `pyproject.toml <https://github.com/kbredies/gratopy/blob/master/pyproject.toml>`_),
it is advised to install the packages by hand before retrying to install gratopy.

In particular, the PyOpenCL package may require some additional
effort as it depends on additional drivers and C libraries which might needed to be installed by hand. We refer to the documentation of PyOpenCL_.

.. _pyopencl: https://documen.tician.de/pyopencl/

For development, we recommend installing the project and dependency management
tool `uv <https://docs.astral.sh/uv/>`__. The local development environment
is then setup from within the cloned repository by running

::

    uv sync


Testing correct installation
----------------------------



The release archive (or GitHub repository) includes a ``tests`` folder which contains a variety of tests that allow to observe visually and numerically whether gratopy was installed correctly and works as desired.

One can perform these tests by using, for instance, pytest_ (which, if you are not using ``uv``
for managing a local development environment, needs to be installed by running
``pip install pytest``). The tests are then executed by running
::

    pytest

or ``uv run pytest`` when using ``uv``.

In case multiple OpenCL devices are registered in :mod:`pyopencl`, but the default device is not suitably configured for the tests to work, one might need to choose the context to use manually. This a-priori choice of context to use in :mod:`pyopencl` can be done via
::

    export PYOPENCL_CTX=<context_number>

The context number can, for instance, be determined in Python by
::

   import pyopencl
   pyopencl.create_some_context()

following the interactive instructions and observing the console output.

By default, the plots of the tests are disabled, but can be activated, e.g., by
::

    export GRATOPY_TEST_PLOT=true

Moreover, the :ref:`getting-started` guide contains two example code segments which can be executed to quickly check that no errors occur and the output is as desired.

.. _pytest: https://pypi.org/project/pytest/

Requirements
------------

The `pyproject.toml <https://github.com/kbredies/gratopy/blob/master/pyproject.toml>`_ file specifies Python packages
required for the use of ``gratopy``. Amongst them the most relevant are

* `pyopencl <https://pypi.org/project/pyopencl/>`_
* `numpy <https://pypi.org/project/numpy/>`_
* `scipy <https://pypi.org/project/scipy/>`_
* `matplotlib <https://pypi.org/project/matplotlib/>`_
* `Pillow <https://pypi.org/project/Pillow/>`_
* `Mako <https://pypi.org/project/Mako/>`_

Most users aiming for scientific computing applications will probably have these packages already installed as they can be considered standard for numerical computations in Python.
Let us again point out that correctly installing PyOpenCL might take some time and effort though, as dependent on the used hardware/GPU, the installation of suitable drivers might be required, see, for instance, https://documen.tician.de/pyopencl/.
