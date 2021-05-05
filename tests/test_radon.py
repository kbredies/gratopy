import unittest

import os
from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
from grato import *
import matplotlib.image as mpimg

ctx = None
queue = None

####################
## fan beam CPU code
def rgb2gray(rgb):
    if len(rgb.shape)>2:
    	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
    	gray=rgb
    return gray

def create_phantoms(N):
    A=phantom(N)
    
    B=np.ones(A.shape)*(255-120)
    B[int(N/float(3)):int(2*N/float(3))]=0
    B[0:int(N/float(4))]=0
    B[int(N-N/float(4)):N]=0
    
    img=np.zeros(A.shape+(2,))
    img[:,:,0]=A*255/np.max(A)
    img[:,:,1]=B

    return img


class TestRadon(unittest.TestCase):
    def test_projection(self):
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)

        print("Projection test")

        N = 1200
        img = create_phantoms(N)
	
        angles=360
        dtype=float32
        detector_width=4
        image_width=4
        Ns=int(0.5*img.shape[0])

        PS=projection_settings(queue,"parallel",img.shape,angles,Ns,image_width=image_width,detector_width=detector_width,detector_shift=0,fullangle=True)
	
        PS.show_geometry(0)
        PS.show_geometry(np.pi/8)
        PS.show_geometry(np.pi/4)
        PS.show_geometry(np.pi*3/8.)
		
        img_gpu = cl.array.to_device(queue, require(img, dtype, 'F'))

        sino_gpu=forwardprojection(None,img_gpu,PS)


        figure(1)
        imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
        backprojected=backprojection(None,sino_gpu,PS)
        figure(2)
        imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
        figure(3)
        imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]), cmap=cm.gray)
        show()

    def test_weighting(self):
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)

        ###Weighting
        print("Weighting test")
        N=900
        img=np.zeros([N,N])
        img[int(N/3.):int(2*N/3.)][:,int(N/3.):int(2*N/3.)]=1
        angles=30
        dtype=float32
        detector_width=4
        image_width=4
        Ns=500
	
        PS=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=detector_width,image_width=image_width)
		
        img_gpu = cl.array.to_device(queue, require(img, dtype, 'F'))
        sino_gpu = cl.array.zeros(queue, PS.sinogram_shape, dtype=dtype, order='F')

        forwardprojection(sino_gpu,img_gpu,PS,wait_for=sino_gpu.events)	
        
        A=np.sum(img)*PS.delta_x**2
        B=np.sum(sino_gpu.get())*PS.delta_s/PS.n_angles
        C=np.sum(sino_gpu.get()[:,10])*PS.delta_s

        print("The mass inside the image is "+str(A)+" was carried over in the mass inside an projection is "+str(B)+" i.e. the relative error is "+ str(abs(1-A/B)))
        
    def test_adjointness(self):
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)

        ##Adjointness
        print("Adjointness test")
        Nx=900
        number_detectors=400
        img=np.zeros([Nx,Nx])
        angles=360
    
        PS=projection_settings(queue,"parallel",img.shape, angles, n_detectors=number_detectors, fullangle=True)
	
        delta_x=PS.delta_x
        delta_s_ratio=PS.delta_ratio

        Fehler=[]
        count=0
        eps=0.00001
        for i in range(100):
            img1_gpu = cl.array.to_device(queue, require(np.random.random(PS.shape), float32, 'F'))
            sino1_gpu = cl.array.to_device(queue, require(np.random.random(PS.sinogram_shape), float32, 'F'))
            img2_gpu = cl.array.zeros(queue, PS.shape, dtype=float32, order='F')
            sino2_gpu = cl.array.zeros(queue, PS.sinogram_shape, dtype=float32, order='F')
            forwardprojection(sino2_gpu,img1_gpu,PS)
					
            backprojection(img2_gpu,sino1_gpu,PS)
            sino1=sino1_gpu.get().reshape(sino1_gpu.size)
            sino2=sino2_gpu.get().reshape(sino2_gpu.size)
            img1=img1_gpu.get().reshape(img1_gpu.size)
            img2=img2_gpu.get().reshape(img2_gpu.size)
		
            a=np.dot(img1,img2)*delta_x**2
            b=np.dot(sino1,sino2)*(np.pi)/angles *(delta_s_ratio*delta_x)
            if abs(a-b)/min(abs(a),abs(b))>eps:
                print (a,b,a/b)
                count+=1
                Fehler.append((a,b))
                
        print ('Adjointness: Number of Errors: '+str(count)+' out of 100 tests adjointness-errors were bigger than '+str(eps))

    def test_fullangle(self):
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)

        ###Fullangle
        N = 1200
        img = create_phantoms(N)

        s_axis=[]
        Resulting_Sino=[]
        angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8
        dtype=float32
        p=2

        Ns=int(0.3*img.shape[0])
        shift=0

        PScorrect=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=p,detector_shift=shift,fullangle=False)
        PSincorrect=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=p,detector_shift=shift,fullangle=True)

        img_gpu = cl.array.to_device(queue, require(img, dtype, 'F'))

        sino_gpu_correct=forwardprojection(None,img_gpu,PScorrect)
        sino_gpu_incorrect=forwardprojection(None,img_gpu,PSincorrect)

        figure(1)
        imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)

        backprojected_correct=backprojection(None,sino_gpu_correct,PScorrect)
        backprojected_incorrect=backprojection(None,sino_gpu_correct,PSincorrect)
        figure(2)
        title("Sinograms with vs without fullangle")
        imshow(np.vstack([np.hstack([sino_gpu_correct.get()[:,:,0],sino_gpu_correct.get()[:,:,1]]),np.hstack([sino_gpu_incorrect.get()[:,:,0],sino_gpu_incorrect.get()[:,:,1]])]), cmap=cm.gray)
        figure(3)
        title("Backprojection with vs without fullangle")
        imshow(np.vstack([np.hstack([backprojected_correct.get()[:,:,0],backprojected_correct.get()[:,:,1]]),np.hstack([backprojected_incorrect.get()[:,:,0],backprojected_incorrect.get()[:,:,1]])]), cmap=cm.gray)
        show()			
        
    def test_nonquadratic(self):
        ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(ctx)

        N = 1200
        img = create_phantoms(N)

        N1=img.shape[0]
        N2=int(img.shape[0]*2/3.)

        img=np.array(img[:,0:N2,:])

        angles=360
        Ns=int(0.5*img.shape[0])

        dtype=float32

        PS=projection_settings(queue,"parallel",img.shape,angles,Ns)
        img_gpu = cl.array.to_device(queue, require(img, dtype, 'F'))
        sino_gpu=forwardprojection(None,img_gpu,PS)
        backprojected=backprojection(None,sino_gpu,PS)

        figure(1)
        title("original non square images")
        imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
        figure(2)
        title("Radon sinogram for non-square image")
        imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
        figure(3)
        title("backprojection for non-square image")
        imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]), cmap=cm.gray)
        show()	
        
# test
if __name__ == '__main__':
    unittest.main()
