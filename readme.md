# Gratop
The **Gr**az  **To**mographic **P**rojections (Gratop) is a software tool developed to allow for efficient, high quality execution of projection methods such as Radon or fanbeam transform. It originates from a project concerning Scanning Transmission Electron Tomography of the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en) together with the [Institute for Electron Microscopy and Nanoanalysis and the Centre for Electron Microscopy](https://www.felmi-zfe.at) at [Graz University of Technology](https://www.tugraz.at), though it is not limited to such applications. The operations contained in the toolbox are based on pixel-driven projection methods, which were shown to possess suitable approximation properties. The code is based in a powerful OpenCL/GPU implementation, resulting in high execution speed, while allowing for seamless integration into PyOpenCL. Hence this can efficiently be paired with other PyOpenCL code, in particular OpenCL based optimization algorithms.

## Highlights
* High quality projection implementation.
* Fast projection due to custom OpenCL/GPU-implementation.
* Seamless integration into PyOpenCL code.

## Install
```bash
pip install gratop
```

Alternatively, no dedicated installation is needed for the program, simply download the code, and copy it to the python libraries or set the insert the corresponding path and get started. Be sure to have the following Python modules installed, most of which should be standard.
 
## Requirements


* [pyopencl](https://pypi.org/project/pyopencl/) (>=pyopencl-2017.1)
* [numpy](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [matplotlib](https://pypi.org/project/matplotlib/)

Particularly, correctly installing and configuring PyOpenCL might take some time, as dependent on the used platform/GPU, suitable drivers must be installed.


## Getting started
We refer to the article cited below which introduces and shows the basic properties. Additionally, the example folder contains various further examples showing how the toolbox can be used. The most basic example can be executed as follows, where it is assumed that a pyopencl ``queue'' queue and a pyopencl.array ``img'' coresponding to an image be given: 

```python
import grato
PS=grato.projection_settings(queue,"parallel",img.shape,angles,Ns)
sino_gpu=grato.forwardprojection(None,img,PS,wait_for=None)
bp_gpu=grato.backprojection(None,sino_gpu,PS,wait_for=None)
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
