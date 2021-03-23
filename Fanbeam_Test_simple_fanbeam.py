from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from fanbeam_source import *


####################
## fan beam CPU code

    

# test
if __name__ == '__main__':
	
	img=np.zeros([225,225,2])
	A =  imread('brain.png')[:,:,0]
	A/=np.max(A)
	B=imread('Ente.jpeg')[:,:,0]
	B=np.array(B,dtype=float)
	B/=np.max(B)
	
	img[:,:,0]=A
	img[:,:,1] = B
	#from scipy import misc
	#misc.imshow(img[:,:,1])
	
	img=np.array(img,dtype=float)
	number_detectors = 600
	#from scipy import misc;misc.imshow(img)
	angles=220
		
	
	PS = projection_settings("fan",img_shape=img.shape,angles= angles,  detector_width=400, R=752, RE=200, n_detectors=number_detectors,data_type='single')
	
	
	#f_struct_cpu = fanbeam_struct_richy_cpu(img.shape, angles,  114.8, 700, 350, number_detectors,0,None)
	#img2=img[:,:,0]
	#sino_cpu=fanbeam_cpu_individual(img2, f_struct_cpu,250,10)
	#import pdb;pdb.set_trace()
		
	img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
	sino_gpu = clarray.zeros(queue, (PS.n_detectors,PS.N_angles,2), dtype=float32, order='F')
	
	a=time.clock()
	for i in range(100):
		sino_gpu.events.append(fanbeam_richy_gpu(sino_gpu,img_gpu,PS,wait_for=sino_gpu.events))
		
	print ('Time Required Forward',(time.clock()-a)/100)
	#from scipy import misc;misc.imshow(sino_gpu.get())
	
	#import pdb; pdb.set_trace()

	#exit
	sino=sino_gpu.get()
	
	numberofangles=180
	angles = linspace(0,2*pi,numberofangles+1)[:-1] + pi
	
	
	#misc.imshow(sino_gpu.get())
	a=time.clock()
	for i in range(100):
		img_gpu.events.append(fanbeam_richy_gpu_add(img_gpu,sino_gpu,PS,wait_for=img_gpu.events))
	print ('Time Required Backprojection',(time.clock()-a)/100)
	#misc.imshow(img_gpu.get())
	figure(1)
	imshow(hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
	title('original image')
	figure(2)
	imshow(hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
	title('Fanbeam transformed image')
	figure(3)
	imshow(hstack([img_gpu.get()[:,:,0],img_gpu.get()[:,:,1]]), cmap=cm.gray)
	title('Backprojected image')
	show()
	
#	ASDF=Landweberiteration(sino_gpu,f_struct_gpu,1000)
#	import pdb;pdb.set_trace()
	
	
	
	
	img = imread('brain.png')
	img=img[:,:,0]
	#img=np.ones(img.shape)
	img=np.array(img,dtype=float)
	number_detectors = 512
	#from scipy import misc;misc.imshow(img)
	angles=360
	
	#f_struct = fanbeam_struct_richy_cpu(img.shape, angles, 114.8, 700, 350, number_detectors)
	#fanbeam_add_new(sino,f_struct)
	
	PS = projection_settings(img_shape=img.shape, angles= angles,  detector_width=114.8,R= 700, RE=350, n_detectors=number_detectors,geometry="fan", data_type='single')
	PS.show_geometry(np.pi/4)
	
	#sinonew=fanbeam_new(img,f_struct)
	#sinocpu=fanbeam_richy_cpu(img,f_struct)
	#import pdb; pdb.set_trace()

	#misc.imshow(sino)
	img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
	sino_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=float32, order='F')
	
	a=time.clock()
	for i in range(100):
		sino_gpu.events.append(fanbeam_richy_gpu(sino_gpu,img_gpu,PS,wait_for=sino_gpu.events))
		
	print ('Time Required Forward',(time.clock()-a)/100)
	#from scipy import misc;misc.imshow(sino_gpu.get())
	
	#import pdb; pdb.set_trace()
	
	#exit
	wsdf=sino_gpu.get()
#	import pdb;pdb.set_trace()
	numberofangles=180
	angles = linspace(0,2*pi,numberofangles+1)[:-1] + pi
	
	
	#misc.imshow(sino_gpu.get())
	a=time.clock()
	for i in range(100):
		img_gpu.events.append(fanbeam_richy_gpu_add(img_gpu,sino_gpu,PS,wait_for=img_gpu.events))
	print ('Time Required Backprojection',(time.clock()-a)/100)
#	import pdb;pdb.set_trace()
	#misc.imshow(img_gpu.get())
	figure(1)
	imshow(img, cmap=cm.gray)
	title('original image')
	figure(2)
	imshow(sino_gpu.get(), cmap=cm.gray)
	title('Fanbeam transformed image')
	figure(3)
	imshow(img_gpu.get(), cmap=cm.gray)
	title('Backprojected image')
	show()



						
		
					
