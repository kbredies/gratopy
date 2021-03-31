from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from grato import *
from scipy import misc
from PIL import Image
import matplotlib.image as mpimg



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
	


	A=mpimg.imread('Testfiles/Shepp_Logan_backprojection_grey_reversed.png')
	A=np.array(rgb2gray(A),dtype=float)
	N=A.shape[0]
	
	B=np.ones(A.shape)*120
	B[int(N/float(3)):int(2*N/float(3))]=255
	B[0:int(N/float(4))]=255
	
	B[int(N-N/float(4)):N]=255
	
	
	img=np.zeros( (A.shape+tuple([2])))
	img[:,:,0]=A*255/np.max(A)
	img[:,:,1]=B
	img=255-img
	
	angles=360
	my_dtype=float32
	detector_width=4
	image_width=4
	Ns=int(0.5*img.shape[0])
	
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	PS=projection_settings(queue,"parallel",img.shape,angles,Ns,image_width=image_width,detector_width=detector_width,detector_shift=0,fullangle=True,data_type=my_dtype)
	
	PS.show_geometry(0)
	PS.show_geometry(np.pi/8)
	PS.show_geometry(np.pi/4)
	PS.show_geometry(np.pi*3/8.)
		
	img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))

	sino_gpu=forwardprojection(None,img_gpu,PS)


	figure(1)
	imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
	backprojected=backprojection(None,sino_gpu,PS)
	figure(2)
	imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
	figure(3)
	imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]), cmap=cm.gray)
	show()			
		
		
	
	
	
	###Weighting
	print("")
	print("weighting")
	N=900
	img=np.zeros([N,N])
	img[int(N/3.):int(2*N/3.)][:,int(N/3.):int(2*N/3.)]=1
	angles=30
	my_dtype=float32
	detector_width=4
	image_width=4
	Ns=500
	

	PS=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=detector_width,image_width=image_width)
		
	img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))
	sino_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=my_dtype, order='F')

	forwardprojection(sino_gpu,img_gpu,PS,wait_for=sino_gpu.events)	
			
	A=np.sum(img)*PS.delta_x**2
	B=np.sum(sino_gpu.get())*PS.delta_xi/PS.n_angles
	C=np.sum(sino_gpu.get()[:,10])*PS.delta_xi

	print("The mass inside the image is "+str(A)+" was carried over in the mass inside an projection is "+str(B)+" i.e. the relative error is "+ str(abs(1-A/B)))
	
	
	##Adjointness
	print("")
	print("Adjointness")
	Nx=900
	number_detectors=400
	img=np.zeros([Nx,Nx])
	angles=360
	
	PS=projection_settings(queue,"parallel",img.shape, angles, n_detectors=number_detectors, 
					fullangle=True,data_type='single')
	
	delta_x=PS.delta_x
	delta_xi_ratio=PS.delta_ratio

	Fehler=[]
	count=0
	eps=0.00001
	for i in range(100):
		
		img1_gpu = clarray.to_device(queue, require(np.random.random(PS.shape), float32, 'F'))
		sino1_gpu = clarray.to_device(queue, require(np.random.random(PS.sinogram_shape), float32, 'F'))
		img2_gpu = clarray.zeros(queue, PS.shape, dtype=float32, order='F')
		sino2_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=float32, order='F')
		forwardprojection(sino2_gpu,img1_gpu,PS)
					
		backprojection(img2_gpu,sino1_gpu,PS)
		sino1=sino1_gpu.get().reshape(sino1_gpu.size)
		sino2=sino2_gpu.get().reshape(sino2_gpu.size)
		img1=img1_gpu.get().reshape(img1_gpu.size)
		img2=img2_gpu.get().reshape(img2_gpu.size)
		
		
		a=np.dot(img1,img2)*delta_x**2
		b=np.dot(sino1,sino2)*(np.pi)/angles *(delta_xi_ratio*delta_x)
		if abs(a-b)/min(abs(a),abs(b))>eps:
			print (a,b,a/b)
			count+=1
			Fehler.append((a,b))
	print("Adjointness:")
	print ('Number of Errors: '+str(count)+' out of 100 tests adjointness-errors were bigger than '+str(eps))

