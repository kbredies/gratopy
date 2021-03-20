from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from fanbeam_source import *
	
	
Coefficienttest=True
if Coefficienttest==True:
	
	number_detectors=230
	img=np.zeros([400,400])
	angles=390
	#f_struct_gpu = fanbeam_struct_gpu(img.shape, angles,  83, 900, 300, number_detectors,0,None)
	
	PS=projection_settings(img.shape, angles, n_detectors=number_detectors, geometry="fan",
					detector_width=83,detector_shift = 0.0, midpointshift=[0,0],
					R=900, RE=300,
					image_width=None, fullangle=True,data_type=float32)
	
	delta_x=PS.delta_x
	delta_xi_ratio=PS.delta_ratio

	Fehler=[]
	count=0
	for i in range(100):
		
		img1_gpu = clarray.to_device(queue, require(np.random.random(PS.shape), float32, 'F'))
		sino1_gpu = clarray.to_device(queue, require(np.random.random(PS.sinogram_shape), float32, 'F'))
		img2_gpu = clarray.zeros(queue, PS.shape, dtype=float32, order='F')
		sino2_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=float32, order='F')
		fanbeam_richy_gpu(sino2_gpu,img1_gpu,PS)
					
		fanbeam_richy_gpu_add(img2_gpu,sino1_gpu,PS)
		sino1=sino1_gpu.get().reshape(sino1_gpu.size)
		sino2=sino2_gpu.get().reshape(sino2_gpu.size)
		img1=img1_gpu.get().reshape(img1_gpu.size)
		img2=img2_gpu.get().reshape(img2_gpu.size)
		
		
		a=np.dot(img1,img2)*delta_x**2
		#import pdb;pdb.set_trace()
		b=np.dot(sino1,sino2)*(2*np.pi)/angles *(delta_xi_ratio*delta_x)
		if abs(a-b)/min(abs(a),abs(b))>0.00001:
			print (a,b,a/b)
		#	count+=1
		#	Fehler.append((a,b))
	#print 'Number of Errors: ',count,' Errors were ',Fehler
