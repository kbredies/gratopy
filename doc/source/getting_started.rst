Getting started
***************


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

One can check wether gratopy was installed correctly, by using the pytest option.
::
    pytest-3 <location of gratopy> 

For this to work one might need to choose the context to use in case multiple context are available, but perhaps not all are suitably configured. You can apriori choose which context to use in pyopencl by default via
::
    export PYOPENCL_CTX=<context_number>
    


Introduction 
==================
The cornerstone of the gratopy toolbox is the `ProjectionSettings` class. It collects all relevant information to create the kernels and precomputes and save values necessary for the computation. Thus virtually all functions of gratopy require an object of this class. To do so, the geometric situation to be observed needs to be discribed, which ranges from very basic settings to more envolved settings. 

Image *img* is considered as pyopencl.Array of the form (Nx,Ny)  ( or (Nx,Ny,Nz) in case of an 3-dimensional image, where the Radon transform is computed slice-wise) These dimensions must be compatible with the `ProjectionSettings` used together with, i.e. `ProjectionSettings.image_width`=(Nx,Ny) must coincide while the Nz dimension can be chosen independently. The image has the physical width corresponds to image_width and represents diameter of a circular obeject contained in the image, and \delta_x the sidelength of the quadratic pixels. These pixels are equidantly positioned in a grid and the values assosciated to them saved in *img* correspond to the mean-value in this pixel of a continuous image counterpart. 

Sinogram *sino* is considered as pyopencl.Array  of the form (Ns,Na) or (Ns,Na,Nz) for Ns the number of detecotors and Na the number of angles. These dimensions must be compatible with the `ProjectionSettings` used together with, i.e., Ns. The width of the detector is given by detector_width, and the detector pixels are equidistantly partioning the detectorline with width delta_s. The angles on the other hand are given by the user and need not be equidistant or even partion the entire angular domain. The values associated to pixel again correspond to the average value of a continuous counterpart sinogram.   

For the Radon transform the geometry is mainly determined by the ratio of image_width to detector_width, for most standard examples these parameters coincide.
	
The considerations concerning *img* and *sino* remain valid for the fanbeam transform. However the geometry gets slightly more complex, as the distance from source to the center of rotation denoted by RE and the distance from the source to the detector denoted by R. 


	.. image:: grafics/Radon-1.png
Geometry of the Radon transform.

	
	.. image:: grafics/Fanbeam-1.png
Geometry of the Fanbeam transform.




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




