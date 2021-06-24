Installation
*************



This package can be installed via pip:
::
    pip install gratopy   

Alternatively, you can download the package from `<https://github.com/kbredies/gratopy>`_ 
and install it via 
::
    python setup.py build
    python setup.py install --user


Testing correct installation
===============================
One can check wether gratopy was installed correctly, by using pytest 
::
    pytest-3 <location of gratopy> 

For this to work one might need to choose the context to use in case multiple context are available, but perhaps not all are suitably configured. You can apriori choose which context to use in pyopencl by default via
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


