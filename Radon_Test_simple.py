from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from fanbeam_source import *


####################
## fan beam CPU code
def rgb2gray(rgb):
    if len(rgb.shape)>2:
    	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
    	gray=rgb
    return gray
    

# test
if __name__ == '__main__':
	

	from scipy import  misc	
	img=misc.imread('Shepp_Logan_backprojection_grey_reversed.png')

	img=np.array(rgb2gray(img),dtype=float)

	N=img.shape[0]
			
	#from scipy import misc;misc.imshow(img)

	#import pdb;pdb.set_trace()
	#img=img[:,:,0]

	s_axis=[]
	Resulting_Sino=[]

	angles=360

	img=255-img
	my_dtype=float32


	p=1
	
	Ns=int(2*img.shape[0])

	r_struct = radon_struct(queue,img.shape, angles,Ns,1/float(p),detector_shift=400,fullangle=True)
	PS=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=p,detector_shift=0,fullangle=True,data_type=my_dtype)
	#(self, geometry, img_shape, angles, n_detectors=None, 
	#				detector_width=1,detector_shift = 0.0, fullangle=True,data_type=float)
	
	
	img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))
	sino_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=my_dtype, order='F')

	sino_gpu.events.append(radon(sino_gpu,img_gpu,PS,wait_for=sino_gpu.events))


	figure(1)
	imshow(img, cmap=cm.gray)
#	if p==1:
#		misc.imsave('SheppLogan_Phantom_sino'+my_dtype+'.png',sino_gpu.get())
#	sino_extended=sino_gpu.get()
#	sino_extended2=np.zeros([Ns,10*angles])
#	for i in range(angles):
#		for j in range(10):
#			sino_extended2[:,10*i+j]=sino_extended[:,i]
		
#		misc.imsave('SheppLogan_Phantom_backprojection'+my_dtype+'.png',img_gpu.get())

	img_gpu.events.append(radon_ad(img_gpu,sino_gpu,PS,wait_for=img_gpu.events))
	figure(2)
	imshow(sino_gpu.get(), cmap=cm.gray)
	figure(3)
	imshow(img_gpu.get(), cmap=cm.gray)
	show()			
		
		
	A=np.sum(img_gpu.get())*PS.delta_x**2
	B=np.sum(sino_gpu.get()[:,20])*PS.delta_xi*(2*np.pi)/PS.N_angles
	print(A,B)
