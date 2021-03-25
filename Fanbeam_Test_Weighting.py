from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from fanbeam_source import *

for number_detectors in [50,100,200,400,800,1600]:
	Nx=400
	img=np.ones([Nx,Nx])
	angles=720

	rescaling=1/40.*sqrt(2)
	detector_width=400*rescaling
	R=1200*rescaling
	RE=200*rescaling
	image_width=40*rescaling

	PS = projection_settings(queue,"fan",img_shape=img.shape, angles= angles,  detector_width=detector_width, R=R,RE= RE, n_detectors=number_detectors,image_width=image_width,data_type='single')
	delta_x=PS.delta_x
	delta_xi_ratio=PS.delta_ratio
	


	img_gpu = clarray.to_device(queue, require(img, float32, 'F'))

	sino_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=float32, order='F')

	fanbeam_richy_gpu(sino_gpu,img_gpu,PS)


	print(np.max(sino_gpu.get()))
	print(np.min(sino_gpu.get()))
	print(np.sum(sino_gpu.get())*delta_xi_ratio/angles)

	a=np.sum(img_gpu.get())*delta_x**2
	b=np.sum(sino_gpu.get())*(delta_xi_ratio*delta_x)/angles
	print(a,b,b/a)
	#imshow(sino_gpu.get())
	#show()
#from scipy import misc
#misc.imshow(sino_gpu.get())


