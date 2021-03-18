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
	number_detectors = 512
	#from scipy import misc;misc.imshow(img)
	angles=360
		
	
	f_struct_gpu = fanbeam_struct_richy_gpu(img.shape, angles,  114.8, 700, 350, number_detectors,0,None)
	
	
	f_struct_cpu = fanbeam_struct_richy_cpu(img.shape, angles,  114.8, 700, 350, number_detectors,0,None)
	img2=img[:,:,0]
	#sino_cpu=fanbeam_cpu_individual(img2, f_struct_cpu,250,10)
	#import pdb;pdb.set_trace()
	
	
	
	
	
	
	
	
		
	img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
	sino_gpu = clarray.zeros(queue, (f_struct_gpu[1][0],f_struct_gpu[1][1],2), dtype=float32, order='F')
	
	a=time.clock()
	for i in range(100):
		sino_gpu.events.append(fanbeam_richy_gpu(sino_gpu,img_gpu,f_struct_gpu,wait_for=sino_gpu.events))
		
	print 'Time Required Forward',(time.clock()-a)/100
	#from scipy import misc;misc.imshow(sino_gpu.get())
	
	#import pdb; pdb.set_trace()

	#exit
	sino=sino_gpu.get()
	
	numberofangles=180
	angles = linspace(0,2*pi,numberofangles+1)[:-1] + pi
	
	
	#misc.imshow(sino_gpu.get())
	a=time.clock()
	for i in range(100):
		img_gpu.events.append(fanbeam_richy_gpu_add(img_gpu,sino_gpu,f_struct_gpu,wait_for=img_gpu.events))
	print 'Time Required Backprojection',(time.clock()-a)/100
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
	
	f_struct_gpu = fanbeam_struct_richy_gpu(img.shape, angles,  114.8, 700, 350, number_detectors,0,None)
	show_geometry(np.pi/4,f_struct_gpu)
	
	#sinonew=fanbeam_new(img,f_struct)
	#sinocpu=fanbeam_richy_cpu(img,f_struct)
	#import pdb; pdb.set_trace()

	#misc.imshow(sino)
	img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
	sino_gpu = clarray.zeros(queue, f_struct_gpu[1], dtype=float32, order='F')
	
	a=time.clock()
	for i in range(100):
		sino_gpu.events.append(fanbeam_richy_gpu(sino_gpu,img_gpu,f_struct_gpu,wait_for=sino_gpu.events))
		
	print 'Time Required Forward',(time.clock()-a)/100
	#from scipy import misc;misc.imshow(sino_gpu.get())
	
	#import pdb; pdb.set_trace()
	
	#exit
	wsdf=sino_gpu.get()
	import pdb;pdb.set_trace()
	numberofangles=180
	angles = linspace(0,2*pi,numberofangles+1)[:-1] + pi
	
	
	#misc.imshow(sino_gpu.get())
	print 'jetz'
	a=time.clock()
	for i in range(100):
		img_gpu.events.append(fanbeam_richy_gpu_add(img_gpu,sino_gpu,f_struct_gpu,wait_for=img_gpu.events))
	print 'Time Required Backprojection',(time.clock()-a)/100
	import pdb;pdb.set_trace()
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
	import pdb;pdb.set_trace()

	Coefficienttest=True
	if Coefficienttest==True:
		Fehler=[]
		count=0
		for i in range(100):
			
			img1_gpu = clarray.to_device(queue, require(np.random.random(f_struct_gpu[0]), float32, 'F'))
			sino1_gpu = clarray.to_device(queue, require(np.random.random(f_struct_gpu[1]), float32, 'F'))
			img2_gpu = clarray.zeros(queue, f_struct_gpu[0], dtype=float32, order='F')
			sino2_gpu = clarray.zeros(queue, f_struct_gpu[1], dtype=float32, order='F')
			fanbeam_richy_gpu(sino2_gpu,img1_gpu,f_struct_gpu)
						
			fanbeam_richy_gpu_add(img2_gpu,sino1_gpu,f_struct_gpu)
			sino1=sino1_gpu.get().reshape(sino1_gpu.size)
			sino2=sino2_gpu.get().reshape(sino2_gpu.size)
			img1=img1_gpu.get().reshape(img1_gpu.size)
			img2=img2_gpu.get().reshape(img2_gpu.size)
			
			
			a=np.dot(img1,img2)
			b=np.dot(sino1,sino2)
			#import pdb; pdb.set_trace()
			if abs(a-b)/min(abs(a),abs(b))>0.001:
				print a,b
				count+=1
				Fehler.append((a,b))
		print 'Number of Errors: ',count,' Errors were ',Fehler
						
		
					
	from scipy import misc				
	Wallnut=misc.imread('Wallnut/Image.png')
	Wallnut=Wallnut/np.mean(Wallnut)
	Wallnut[np.where(Wallnut<=1.5)]=0
	Wallnut=scipy.misc.imresize(Wallnut,[328,328])
	
#	Wallnut=np.ones(Wallnut.shape)
	Wallnut_gpu=clarray.to_device(queue,require(Wallnut,float32,'F'))
	number_detectors=328
	Detectorwidth=114.8
	FOD=110
	FDD=300
	numberofangles=120
	geometry=[Detectorwidth,FDD,FOD,number_detectors]
	#misc.imshow(Wallnut)
	f_struct_gpu_wallnut = fanbeam_struct_richy_gpu(Wallnut.shape, numberofangles, Detectorwidth, FDD, FOD, number_detectors)
	sino2_gpu = clarray.zeros(queue, f_struct_gpu_wallnut[1], dtype=float32, order='F')
	
	fanbeam_richy_gpu(sino2_gpu,Wallnut_gpu,f_struct_gpu_wallnut)	
	#misc.imshow(sino2_gpu.get())
	Wallnut_gpu2=clarray.to_device(queue,require(Wallnut,float32,'F'))
	fanbeam_richy_gpu_add(Wallnut_gpu2,sino2_gpu,f_struct_gpu_wallnut)
	
	figure(1)
	imshow(Wallnut, cmap=cm.gray)
	title('original Wallnut')
	figure(2)
	imshow(sino2_gpu.get(), cmap=cm.gray)
	title('Fanbeam transformed image')
	figure(3)
	imshow(Wallnut_gpu2.get(), cmap=cm.gray)
	title('Backprojected image')
	show()
	#misc.imshow(Wallnut_gpu2.get())
	
	
	#U=Landweberiteration(sino2_gpu,f_struct_gpu_wallnut)
	#misc.imshow(U.get())
	
	
	
	sinonew=misc.imread('Wallnut/Sinogram.png')
	print 'asdf',np.max(sinonew)
	sinonew[np.where(sinonew<2000)]=0
	sinonew=np.array(sinonew,dtype=float)
	sinonew/=np.mean(sinonew)
	
	
	#h=scipy.hstack([sino2_gpu.get()/np.mean(sino2_gpu.get()),sinonew])
	#misc.imshow(h)
	#misc.imsave('Comparison_sinogram.png',h)
	Wallnut_gpu2new=clarray.to_device(queue,require(sinonew,float32,'F'))
	ULW=Landweberiteration(Wallnut_gpu2new,f_struct_gpu_wallnut,100)
	misc.imshow(ULW.get())
	#fanbeam_richy_gpu(Wallnut,U,f_struct_gpu_wallnut)
	#misc.imshow(Wallnut.get())
	#import pdb;pdb.set_trace()

	sinonew=[sinonew.T]
	import pdb; pdb.set_trace()
