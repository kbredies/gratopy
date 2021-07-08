# Gratopy
The **Gr**az **a**ccelerated **to**mographic projection for **P**ython **(Gratopy)**  is a software tool for Python 3 developed to allow for efficient, high quality execution of projection methods
such as Radon and fanbeam transform.  The operations contained in the toolbox are based on pixel-driven projection methods, which were shown to possess suitable approximation properties.
The code is based in a powerful OpenCL/GPU implementation, resulting in high execution speed, while allowing for seamless integration into [PyOpenCL](https://documen.tician.de/pyopencl/). 
Hence this can efficiently be paired with other PyOpenCL code, in particular OpenCL based optimization algorithms.

## Highlights
* Easy to use tomography toolbox.
* High quality projection implementation.
* Fast projection due to custom OpenCL/GPU-implementation.
* Seamless integration into PyOpenCL code.
* Contains various reconstruction schemes.

## Install
```bash
pip install gratopy
```

or alternatively directly download and unpack the tar folder and install inside the folder via

```bash
pip install .
```
or via the wheel file

```bash
pip install gratopy*.whl
```

For more details we refer you to  [documentation](doc/build/html/installation.html).

Alternatively, no dedicated installation is needed for the program, simply download the code, and copy it to the python libraries or set the insert the corresponding path and get started. Be sure to have the following Python modules installed, most of which should be standard.
 
## Requirements


* [pyopencl](https://pypi.org/project/pyopencl/)
* [numpy](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [matplotlib](https://pypi.org/project/matplotlib/)

Particularly, correctly installing and configuring PyOpenCL might take some time, as dependent on the used platform/GPU, suitable drivers must be installed.


## Getting started
We refer you to the extensive [documentation](doc/build/html/index.html), inparticular the point [getting started](doc/build/html/getting_started.html), as well as to the test files for [Radon transform](doc/build/html/_modules/test_radon.html) or [fanbeam transform](doc/build/html/_modules/test_fanbeam.html). In the documentary you also find the following [rudimentary example](doc/build/html/getting_started.html#first-example-radon-transform).

```python

#Initial import and definitions
from numpy import *
import pyopencl as cl
import gratopy
import matplotlib.pyplot as plt
number_angles=60
number_detector=300
Nx=300

#create pyopencl context
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
    
#create phantom as testimage
phantom=gratopy.phantom(queue,Nx)
    
#create suitable ProjectionSettings
PS=gratopy.ProjectionSettings(queue,gratopy.RADON,phantom.shape,
                               number_angles,number_detector)
	    
#Compute forward projection and backprojection of created sinogram	
sino=gratopy.forwardprojection(phantom,PS)
backproj=gratopy.backprojection(sino,PS)

#Plot results
plt.figure()
plt.title("Generated Phantom")
plt.imshow(phantom.get(),cmap="gray")

plt.figure()
plt.title("Sinogram")
plt.imshow(sino.get(),cmap="gray")

plt.figure()
plt.title("Backprojection")
plt.imshow(backproj.get(),cmap="gray")
plt.show()

```







## Known issues


## Authors

* **Richard Huber** richard.huber@uni-graz.at
* **Kristian Bredies** kristian.bredies@uni-graz.at

All authors are affiliated with the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en).

## Publications
If you find this tool useful, please cite the following associated publication.

*
*

## Acknowledgements

The development of this software was partially supported by the following projects:

* *Regularization Graphs for Variational Imaging*, funded by the Austrian Science Fund (FWF), grant P-29192,

* *Lifting-based regularization for dynamic image data*, funded by the Austrian Science Fund (FWF), grant J-4112,

* *International Research Training Group IGDK 1754 Optimization and Numerical Analysis for Partial Differential Equations with Nonsmooth
Structures*, funded by the German Research Council (DFG) and the Austrian Science Fund (FWF), grant W-1244.

## License

This project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.
