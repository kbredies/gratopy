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
	A=misc.imread('Shepp_Logan_backprojection_grey_reversed.png')
	A=np.array(rgb2gray(A),dtype=float)
	N=A.shape[0]
	
	
	B=np.ones(A.shape)*120
	B[int(N/float(3)):int(2*N/float(3))]=255
	B[0:int(N/float(4))]=255
	
	B[int(N-N/float(4)):N]=255
	
	
	img=np.zeros( (A.shape+tuple([2])))
	img[:,:,0]=A
	img[:,:,1]=B
	
			
	#from scipy import misc;misc.imshow(img)

	#import pdb;pdb.set_trace()
	#img=img[:,:,0]

	s_axis=[]
	Resulting_Sino=[]

	angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8
	
	img=255-img
	my_dtype=float32


	p=2
	
	Ns=int(0.3*img.shape[0])
	shift=0
	#r_struct = radon_struct(queue,img.shape, angles,Ns,1/float(p),detector_shift=400,fullangle=True)
	
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	PScorrect=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=p,detector_shift=shift,fullangle=False,data_type=my_dtype)
	PSincorrect=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=p,detector_shift=shift,fullangle=True,data_type=my_dtype)

	#(self, geometry, img_shape, angles, n_detectors=None, 
	#				detector_width=1,detector_shift = 0.0, fullangle=True,data_type=float)
	
	
	img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))
	#sino_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=my_dtype, order='F')

	sino_gpu_correct=forwardprojection(None,img_gpu,PScorrect)
	sino_gpu_incorrect=forwardprojection(None,img_gpu,PSincorrect)

	figure(1)
	imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
#	if p==1:
#		misc.imsave('SheppLogan_Phantom_sino'+my_dtype+'.png',sino_gpu.get())
#	sino_extended=sino_gpu.get()
#	sino_extended2=np.zeros([Ns,10*angles])
#	for i in range(angles):
#		for j in range(10):
#			sino_extended2[:,10*i+j]=sino_extended[:,i]
		
#		misc.imsave('SheppLogan_Phantom_backprojection'+my_dtype+'.png',img_gpu.get())

	backprojected_correct=backprojection(None,sino_gpu_correct,PScorrect)
	backprojected_incorrect=backprojection(None,sino_gpu_correct,PSincorrect)
	figure(2)

	imshow(np.vstack([np.hstack([sino_gpu_correct.get()[:,:,0],sino_gpu_correct.get()[:,:,1]]),np.hstack([sino_gpu_incorrect.get()[:,:,0],sino_gpu_incorrect.get()[:,:,1]])]), cmap=cm.gray)
	figure(3)
	imshow(np.vstack([np.hstack([backprojected_correct.get()[:,:,0],backprojected_correct.get()[:,:,1]]),np.hstack([backprojected_incorrect.get()[:,:,0],backprojected_incorrect.get()[:,:,1]])]), cmap=cm.gray)
	show()			
		
########		


#	angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8
	angles=180
	R=5
	RE=2
	Detector_width=6
	image_width=2
	PScorrect=projection_settings(queue,"fan",img.shape,angles,Ns,image_width=image_width,R=R,RE=RE,detector_width=Detector_width,detector_shift=shift,fullangle=False,data_type=my_dtype)
	PSincorrect=projection_settings(queue,"fan",img.shape,angles,Ns,image_width=image_width,R=R,RE=RE,detector_width=Detector_width,detector_shift=shift,fullangle=True,data_type=my_dtype)
	
	PScorrect.show_geometry(0)
	img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))

	sino_gpu_correct=forwardprojection(None,img_gpu,PScorrect)
	sino_gpu_incorrect=forwardprojection(None,img_gpu,PSincorrect)

	figure(1)
	imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
#	if p==1:
#		misc.imsave('SheppLogan_Phantom_sino'+my_dtype+'.png',sino_gpu.get())
#	sino_extended=sino_gpu.get()
#	sino_extended2=np.zeros([Ns,10*angles])
#	for i in range(angles):
#		for j in range(10):
#			sino_extended2[:,10*i+j]=sino_extended[:,i]
		
#		misc.imsave('SheppLogan_Phantom_backprojection'+my_dtype+'.png',img_gpu.get())

	backprojected_correct=backprojection(None,sino_gpu_correct,PScorrect)
	backprojected_incorrect=backprojection(None,sino_gpu_correct,PSincorrect)
	figure(2)

	imshow(np.vstack([np.hstack([sino_gpu_correct.get()[:,:,0],sino_gpu_correct.get()[:,:,1]]),np.hstack([sino_gpu_incorrect.get()[:,:,0],sino_gpu_incorrect.get()[:,:,1]])]), cmap=cm.gray)
	figure(3)
	imshow(np.vstack([np.hstack([backprojected_correct.get()[:,:,0],backprojected_correct.get()[:,:,1]]),np.hstack([backprojected_incorrect.get()[:,:,0],backprojected_incorrect.get()[:,:,1]])]), cmap=cm.gray)
	show()			
