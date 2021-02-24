## Graptor - Graz Application for Tomographic Reconstruction
##
## Copyright (C) 2019 Richard Huber, Martin Holler, Kristian Bredies
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.
##
##
## radon_tgv_primal_dual_3d.py:
## Code for 3D variational reconstruction of tomographic data.
##
## -------------------------
## Richard Huber (richard.huber@uni-graz.at)
## Martin Holler (martin.holler@uni-graz.at)
## Kristian Bredies (kristian.bredies@uni-graz.at)
## 
## 21.02.2019
## -------------------------
## If you consider this code to be useful, please cite:
## 
## [1] R. Huber, G. Haberfehlner, M. Holler, G. Kothleitner,
##     K. Bredies. Total Generalized Variation regularization for
##     multi-modal electron tomography. *Nanoscale*, 2019. 
##     DOI: [10.1039/C8NR09058K](https://doi.org/10.1039/C8NR09058K).
##
## [2] M. Holler, R. Huber, F. Knoll. Coupled regularization with
##     multiple data discrepancies. Inverse Problems, Special
##     issue on joint reconstruction and multi-modality/multi-spectral
##     imaging, 34(8):084003, 2018.


from numpy import *
import matplotlib
#matplotlib.use('gtkagg')
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy
import copy

""" Reconstruction of tomographic data for multiple using Primal Dual optimization
Input:	
		sino		... A list(length is Number_Channels) containing the sinogram data (Array with shape [Number_of_channels,Number_of_angles,Nx])
		angles		... A list or array of angles corresponding to Radon transform
		Parameters	... Regularization parameters corresponding to TGV or TV (List, array or scalar)
		mu			... Weights of the subproblems (must be same length as Number_Channels) (array or list)
		maxiter		... Number of iterations used in the primal dual algorithm (scalar)
		ctx			... A pyopencl context corresponding to the device used for GPU implementation
		plott		... A list with two entries, first is 0 or 1 for no live plott or live plott, second for frequence of updates (between 0 and 1 transformed in percent of reconstruction, for greater 1 corresponds to every so much iterations)
		Stepsize	... An integer in case of skipping slices (e.g. 2 if every second slice is reconstructed)
		discrepancy	... A string stating whether L2 or KL should be used as discrepancy functions. Options 'L2' or 'KL'
		regularisationmethod	...	 A string stating what regularisation method is used for coupled regularization, i.e. uncorrelated, Frobenius or Nuclear coupling. Options are 'Uncorrelated 2D Reconstruction', 'Frobenius 2D Reconstruction','Nuclear 2D Reconstruction'
		regularization ...String: state what regularization functional to consider, options 'TV' or 'TGV'
		Info		... A list of additional plotting information containing information on the overall progress of the reconstruction, where first entry corresponds to percent already complete, the second the width (in percent) of the subproblem currently computed,	last on at which slices we currently compute .

Output:
		Solution   ... A numpy array containing the reconstuction
		Sinogram   ... A numpy array containing the sinograms corresponding to the solution
"""
def Reconstructionapproach3d(sino,angles,Parameter,mu,maxiter,ctx,plott,Stepsize,discrepancy,regularisationmethod,regularisation,Info,adapt_mu):
	
	start=Info[0]
	sectionwidth=Info[1]
	current=Info[2]
	
	#This is a hack that avoids a bug that seems to occur when zero initializing arrays of size >2GB with clarray.zeros
	def zeros_hack(*args, **kwargs):
		res = clarray.empty(*args, **kwargs)
		res[:] = 0
		return res
	clarray.zeros = zeros_hack
	
	#Create Py Opencl Program
	class Program(object):
		def __init__(self, ctx, code):
			self._cl_prg = cl.Program(ctx, code)
			self._cl_prg.build()
			self._cl_kernels = self._cl_prg.all_kernels()
			for kernel in self._cl_kernels:
					self.__dict__[kernel.function_name] = kernel
	
	#Choose Context, GPUdevice	
	
	queue = cl.CommandQueue(ctx)

######Kernel Code (Actual code for computation on the Graphical Processing Unit)
	prg = Program(ctx, r"""
	

__kernel void radon(__global float *sino, __global float *img,
					__constant float4 *ofs, const int X,
					const int Y)
{
   size_t I = get_global_size(0);
   size_t J = get_global_size(1); 

  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t thirddim = get_global_id(2);
  int ii=i;
  float4 o = ofs[j];
  float acc = 0.0f;
  img+=X*Y*thirddim;
  for(int y = 0; y < Y; y++) {
	int x_low, x_high;
	float d = y*o.y + o.z;

	// compute bounds
	if (o.x == 0) {
	  if ((d > ii-1) && (d < ii+1)) {
		x_low = 0; x_high = X-1;
	  } else {
		img += X; continue;
	  }
	} else if (o.x > 0) {
	  x_low = (int)((ii-1 - d)*o.w);
	  x_high = (int)((ii+1 - d)*o.w);
	} else {
	  x_low = (int)((ii+1 - d)*o.w);
	  x_high = (int)((ii-1 - d)*o.w);
	}
	x_low = max(x_low, 0);
	x_high = min(x_high, X-1);

	// integrate
	for(int x = x_low; x <= x_high; x++) {
	  float weight = 1.0 - fabs(x*o.x + d - ii);
	  if (weight > 0.0f) acc += weight*img[x];
	}
	img += X;
  }
  sino[j*I + i+I*J*thirddim] = acc;
}

__kernel void radon_ad(__global float *img, __global float *sino,
					   __constant float4 *ofs, const int I,
					   const int J)
{
  size_t X = get_global_size(0);
	size_t Y = get_global_size(1);
  
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
   size_t thirddim = get_global_id(2);

  float4 c = (float4)(x,y,1,0);
  float acc = 0.0f;
  sino += I*J*thirddim;
  
  for (int j=0; j < J; j++) {
	float i = dot(c, ofs[j]);
	if ((i > -1) && (i < I)) {
	  float i_floor;
	  float p = fract(i, &i_floor);
	  if (i_floor >= 0)	  acc += (1.0f - p)*sino[(int)i_floor];
	  if (i_floor <= I-2) acc += p*sino[(int)(i_floor+1)];
	}
	sino += I;
  }
  img[y*X + x+X*Y*thirddim] = acc;
}



""")

###########################

	""" Is used to ensure that no data is present where it makes no sense for data to be by setting sino to zero in such positions
		Input		
				sino		...	Np.array with sinogram data in question
				r_struct	... Defining the geometry of an object for the radon transform
				imageshape	... Imagedimensions of the image corresponding to the image (could probably be removed since information is also in r_struct)
		Ouput
				sinonew		... A new sinogram where pixels such that R1 is zero, where R is the radon transform and 1 is the constant 1 image. (This corresponds to all pixels	 we cann not obtain any mass in due to the geometry of the radontransform)
	"""

#############
## Radon code


"""Creates the structure of radon geometry required for radontransform and its adjoint
Input:
		queue ... a queue object corresponding to a context in pyopencl
		shape ... the shape of the object (image) in pixels
		angles ... a list of angles considered
		n_detectors ... Number of detectors, i.e. resolution of the sinogram
		detector_with ... Width of one detector relatively to a pixel in the image (default 1.0)
		detector_shift ... global shift of ofsets (default 0)
Output:
		ofs_buf ... a buffer object with 4 x number of angles entries corresponding to the cos and sin divided by the detectorwidth, also offset depending on the angle and the inverse of the cos values
		shape ... The same shape as in the input.
		sinogram_shape ... The sinogram_shape is a list with first the number of detectors, then number of angles.
"""
def radon_struct(queue, shape, angles, n_detectors=None,
			 detector_width=1.0, detector_shift=0.0):
	if isscalar(angles):
		angles = linspace(0,pi,angles+1)[:-1]
	if n_detectors is None:
		nd = int(ceil(hypot(shape[0],shape[1])))
	else:
		nd = n_detectors
	midpoint_domain = array([shape[0]-1, shape[1]-1])/2.0
	midpoint_detectors = (nd-1.0)/2.0

	X = cos(angles)/detector_width
	Y = sin(angles)/detector_width
	Xinv = 1.0/X

	# set near vertical lines to vertical
	mask = abs(Xinv) > 10*nd
	X[mask] = 0
	Y[mask] = sin(angles[mask]).round()/detector_width
	Xinv[mask] = 0

	offset = midpoint_detectors - X*midpoint_domain[0] \
			- Y*midpoint_domain[1] + detector_shift/detector_width

	ofs = zeros((4, len(angles)), dtype=float32, order='F')
	ofs[0,:] = X; ofs[1,:] = Y; ofs[2,:] = offset; ofs[3,:] = Xinv

	ofs_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
	cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()

	sinogram_shape = (nd, len(angles))

	return (ofs_buf, shape, sinogram_shape)


"""Starts the GPU Radon transform code 
input	sino ... A pyopencl.array in which result will be saved.
		img ...	 A pyopencl.array in which the image for the radontransform is contained
		r_struct. ..	The r_struct corresponding the given topology (geometry), see radon_struct
output	An event for the queue to compute the radon transform of image saved into img w.r.t. r_struct geometry
"""
def radon(sino, img, r_struct, wait_for=None):
	(ofs_buf, shape, sinogram_shape) = r_struct

	return prg.radon(sino.queue, sino.shape, None,
				 sino.data, img.data, ofs_buf,
				 int32(shape[0]), int32(shape[1]),
				 wait_for=wait_for)

"""Starts the GPU backprojection code 
input	sino ... A pyopencl.array in which the sinogram for transformation will be saved.
		img ...	 A pyopencl.array in which the result for the adjoint radontransform is contained
		r_struct. ..	The r_struct corresponding the given topology, see radon_struct
output	An event for the queue to compute the adjoint radon transform of image saved into img w.r.t. r_struct geometry
"""
def radon_ad(img, sino, r_struct, wait_for=None):
	(ofs_buf, shape, sinogram_shape) = r_struct
	
	return prg.radon_ad(img.queue, img.shape, None,
					img.data, sino.data, ofs_buf,
					int32(sinogram_shape[0]),
					int32(sinogram_shape[1]), wait_for=wait_for)

"""Estimation of the norm of the radontransform with geometry r_struct is computed
input
		queue ... queue object of some context in pyopencl
		the r_struct corresponding the given topology of a radon transform, see radon_struct
output
		norm ... an estimate of the norm of the radon transform (square of largest singular value)
An Power iteration method is applied onto R^T R (Adjoint operator times operator of radontransform)
"""
def radon_normest(queue, r_struct):
	img = clarray.to_device(queue, require(random.randn(*r_struct[1]), float32, 'F'))
	sino = clarray.zeros(queue, r_struct[2], dtype=float32, order='F')

	V=(radon(sino, img, r_struct, wait_for=img.events))
	
	for i in range(10):
		normsqr = float(clarray.sum(img).get())
		img /= normsqr
		sino.add_event(radon(sino, img, r_struct, wait_for=img.events))
		img.add_event(radon_ad(img, sino, r_struct, wait_for=sino.events))

	return sqrt(normsqr)
	
	
def get_gpu_context(GPU_choice=-1):
	platforms = cl.get_platforms()
	my_gpu_devices = []
	try:
		for platform in platforms:
			my_gpu_devices.extend(platform.get_devices())
		gpu_devices = [device for device in my_gpu_devices if device.type == cl.device_type.GPU]
		non_gpu_devices = [device for device in my_gpu_devices if device not in gpu_devices]
		my_gpu_devices = gpu_devices + non_gpu_devices
	except:
		pass
	if my_gpu_devices == []:
		raise cl.Error('No device found, make sure PyOpenCL was installed correctly.')
	while GPU_choice not in range(0,len(my_gpu_devices)):
		for i in range(0,len(my_gpu_devices)):
			print( '['+str(i)+']' ,my_gpu_devices[i])
		GPU_choice=input('Choose device to use by entering the number preceeding it: ' )
		try:
			GPU_choice = int(GPU_choice)
		except ValueError:
			print('Please enter an integer value')

	if GPU_choice in range(0,len(my_gpu_devices)):
		gerat=[my_gpu_devices[GPU_choice]]
		ctx = cl.Context(devices=gerat)
	else:
		ctx = cl.create_some_context()
	return ctx, my_gpu_devices, GPU_choice


ctx, my_gpu_devices, GPU_choice=get_gpu_context()	
queue = cl.CommandQueue(ctx)





