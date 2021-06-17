Getting started
************


Installation
==================

This package can be installed via pip:
::
	pip install gratopy   
    
Requirements
==================
The requirements.txt file contains references to the relevant python packages
Amongst them the most relevant ones are

* numpy 
* matplotlib
* scipy
* pyopencl

Particularly, correctly installing PyOpenCl might take some time and efford, as dependent on the Hardware/GPU used suitable drivers might need to be installed, we refer you to `https://documen.tician.de/pyopencl/`


First example
==================
One can start in Python via
::
	from numpy import *
	import pyopencl as cl
	import gratopy
	import matplotlib .pyplot as plt

	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	phantom=gratopy.phantom(queue,300)
	number_angles=60
	number_detector=300
	PS=gratopy.ProjectionSettings(queue,gratopy.RADON,phantom.shape
		,number_angles,number_detector)
	sino=gratopy.forwardprojection(phantom,PS)
	backproj=gratopy.backprojection(sino,PS)

	plt.figure()
	plt.title("Generated Phantom")
	plt.imshow(phantom.get())

	plt.figure()
	plt.title("Sinogram")
	plt.imshow(sino.get())

	plt.figure()
	plt.title
	plt.imshow(backproj.get())
	plt.show()
	

