from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from fanbeam_source import *

# test
if __name__ == '__main__':
	



	N=900
	
	#img=np.ones([N,N])		
	img=np.zeros([N,N])
	img[int(N/3.):int(2*N/3.)][:,int(N/3.):int(2*N/3.)]=1

	s_axis=[]
	Resulting_Sino=[]

	angles=30

	my_dtype=float32


	p=4
	
	Ns=500
	#r_struct = radon_struct(queue,img.shape, angles,Ns,1/float(p),detector_shift=400,fullangle=True)
	
	
	PS=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=p)
	
	#(self, geometry, img_shape, angles, n_detectors=None, 
	#				detector_width=1,detector_shift = 0.0, fullangle=True,data_type=float)
	
	
	img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))
	sino_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=my_dtype, order='F')

	sino_gpu.events.append(radon(sino_gpu,img_gpu,PS,wait_for=sino_gpu.events))


		
		
			
	A=np.sum(img)*PS.delta_x**2
	B=np.sum(sino_gpu.get())*PS.delta_xi/PS.N_angles
	C=np.sum(sino_gpu.get()[:,10])*PS.delta_xi

	print(A,B,abs(1-A/B))
