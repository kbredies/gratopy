Installation
*************
Gratopy can easily be installed via pip.

Installation via pip
=====================

For many users the `wheel <https://pypi.org/project/wheel/>`_ package is probably already installed. As not having it installed can lead to issues though, we advise to install it beforehand by hand via
::

    pip install wheel

Then the gratopy toolbox can be  installed from the internet directly via 
::

    pip install gratopy   

Alternatively, you can download the package from `<https://github.com/kbredies/gratopy>`_ 
as tar and install it (after unpacking inside the corresponding folder) via 
::

    pip install .
    
or installing the wheel file directly via
::

    pip install gratopy*.whl

In case these installations fails due to the dependency on other packages (see requirements.txt), it is advised to install the packages by hand before retrying to install gratopy. In particular the PyOpenCL package may require some additional
effort as it depends on additional drivers and c libraries which might need to be installed by hand. 

Testing correct installation
============================

In the tar.gz file (or on github) you find a tests folder, which contains a variety of tests visually and numerically observing whether gratopy was installed correctly and works as desired.

One can execute these tests by using `pytest <https://pypi.org/project/pytest/>`_  
::

    pytest  
    
or `nose <https://pypi.org/project/nose/>`_
::

    nosetests 

In case multiple context are available, but not all are suitably configured for the tests to work one might need to choose the context to use (as the test will choose the context determined as default by the system). You can a-priori choose which context to use in PyOpenCL by default via
::

    export PYOPENCL_CTX=<context_number>


Moreover, `getting started <getting_started.html>`_ contains two example code segments which can be executed to quickly check that no errors occur and the output is as desired.

Requirements
==================
The requirements.txt file contains references to python packages  relevant to the use of gratopy.
Amongst them the most relevant ones are

* numpy 
* matplotlib
* scipy
* pyopencl

Most accomplished user will probably have these packages installed as they are considered standard for numerical calculations in python.
Particularly, correctly installing PyOpenCl might take some time and effort though, as dependent on the Hardware/GPU used suitable drivers might need to be installed, we refer to `<https://documen.tician.de/pyopencl/>`_.    


