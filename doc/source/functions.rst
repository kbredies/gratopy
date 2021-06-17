Gratopy
*******

The - *Gr*\az *a*\ccelerated *to*mographic projection for *P*\ython *(Gratopy)*  is a software tool developed to allow for efficient, high quality execution of projection methods such as Radon and fanbeam transform.  The operations contained in the toolbox are based on pixel-driven projection methods, which were shown to possess suitable approximation properties. The code is based in a powerful OpenCL/GPU implementation, resulting in high execution speed, while allowing for seamless integration into PyOpenCL. Hence this can efficiently be paired with other PyOpenCL code, in particular OpenCL based optimization algorithms.

Highlights
==================
* High quality projection implementation.
* Fast projection due to custom OpenCL/GPU-implementation.
* Seamless integration into PyOpenCL code.

Installation
==================

This package can be installed via pip:
::
	pip install grato   
    
Requirements
==================
The requirements.txt file contains references to the relevant python packages
Amongst them the most relevant ones are

* numpy 
* matplotlib
* scipy
* pyopencl

Particularly, correctly installing PyOpenCl might take some time and efford, as dependent on the Hardware/GPU used suitable drivers might need to be installed, we refer you to `https://documen.tician.de/pyopencl/`


Getting startetd
==================
One can start in Python via
::
	from numpy import *
	import pyopencl as cl
	import grato
	import matplotlib .pyplot as plt 
	
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	phantom=grato.phantom(queue,300)
	number_angles=60
	number_detector=300
	PS=grato.ProjectionSettings(queue,grato.RADON,phantom.shape,number_angles,number_detector)
	sino=grato.forwardprojection(phantom,PS)
	backproj=grato.backprojection(sino,PS)
	
	figure()
	title("Generated Phantom")
	imshow(phantom)
	
	figure()
	title("Sinogram")
	imshow(sino)
	
	figure()
	title
	imshow(backproj)
	show()

	

Authors
==================
* *Kristian Bredies* University of Graz, kristian.bredies@uni-graz.at
* *Richard Huber* University of Graz, richard.huber@uni-graz.at


Publications
==================
If you found the toolbox useful, please cite the following associated publications.



Acknowledgement
==================

Licence
==================

     
Function reference
==================

.. module:: grato

Transforms
----------

.. autoclass :: grato.ProjectionSettings
	:members:
.. autofunction:: grato.forwardprojection
.. autofunction:: grato.backprojection

Solvers
-------

.. autofunction:: grato.landweber
.. autofunction:: grato.cg
		  
Data generation
---------------

.. autofunction:: grato.phantom

Internal functions
------------------

.. autofunction:: grato.radon
.. autofunction:: grato.radon_ad
.. autofunction:: grato.radon_struct

.. autofunction:: grato.fanbeam
.. autofunction:: grato.fanbeam_ad
.. autofunction:: grato.fanbeam_struct

.. autofunction:: grato.create_code
