Installation
*************

Installation via pip
=====================
Pip can be use to install this package easily and straight-forward.

For many users the `wheels <https://pypi.org/project/wheel/>`_ toolbox is probably already installed. As not having it installed can lead to issues, we advise to install it beforehand by hand via
::
    pip install wheel

Then the gratopy toolbox can be  installed from the internet directly via 
::
    pip install gratopy   

Alternatively, you can download the package from `<https://github.com/kbredies/gratopy>`_ 
and install it (when inside the corresponding folder) via 
::
    pip install .

In case these installations fails due to the dependency on other packages (see inside setup.py), it is advised to install the packages by hand. In particular the pyopencl package may require some additional
considerations as it depends drivers and c libraries which might need to be installed by hand. 

Testing correct installation
===============================
One can check wether gratopy was installed correctly, by using pytest 
::
    pytest-3 <location of gratopy> 

For this to work one might need to choose the context to use in case multiple context are available, but perhaps not all are suitably configured (as the test will choose the context determined default by the system). You can apriori choose which context to use in pyopencl by default via
::
    export PYOPENCL_CTX=<context_number>


Requirements
==================
The requirements.txt file contains references to the relevant python packages
Amongst them the most relevant ones are

* numpy 
* matplotlib
* scipy
* pyopencl

Particularly, correctly installing PyOpenCl might take some time and efford, as dependent on the Hardware/GPU used suitable drivers might need to be installed, we refer you to `<https://documen.tician.de/pyopencl/>`_.    


