import os, pkgutil
from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
from gratopy import *
import matplotlib.image as mpimg

ctx = None
queue = None


def create_phantoms(queue, N, dtype='double'):
    #Use gratopy phantom method to create Shepp-Logan
    A=phantom(queue, N, dtype=dtype)
    A *= 255/cl.array.max(A).get()
    
    #Second test image consisting of 2 bars across the image
    B=cl.array.empty(queue, A.shape, dtype=dtype)
    B[:] = 255-120
    B[int(N/3):int(2*N/3)]=0
    B[0:int(N/4)]=0
    B[int(N-N/4):N]=0
    
    img=cl.array.to_device(queue, np.stack([A.get(), B.get()], axis=-1))
    return img

def test_projection():
    ###Projection test: simply computes forward and backprojection of two test images, to visually confurm the correctness of the method 
    print("Projection test")

    #Create PyopenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    
                
    #Create testimage
    dtype=float32
    N = 1200
    img = create_phantoms(queue, N, dtype=dtype)
	
    #Define relevant quantities to determine the geometry
    angles=360
    detector_width=4
    image_width=4
    Ns=int(0.5*img.shape[0])

    #Create projectionsetting with parallel beam setting with 360 equi-distant angles, the detector has a width of 4 and the observed object has a diameter of 4 (i.e. is captured by the detector), we consider half the amount of detector pixels as image pixels
    PS=ProjectionSettings(queue, PARALLEL, img.shape, angles, Ns,
                              image_width=image_width, detector_width=detector_width,
                              detector_shift=0,
                              fullangle=True)


    #Plot geometry via show_geometry method to consider geometry
    figure(0)
    PS.show_geometry(0, axes=subplot(2,2,1))
    PS.show_geometry(np.pi/8, axes=subplot(2,2,2))
    PS.show_geometry(np.pi/4, axes=subplot(2,2,3))
    PS.show_geometry(np.pi*3/8., axes=subplot(2,2,4))
    
    #Compute Radon transform for given test images
    sino_gpu = forwardprojection(img, PS)
    #Compute backprojection of computed sinogram
    backprojected = backprojection(sino_gpu, PS)
    
    #Plot Results
    figure(1)
    imshow(np.hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    figure(2)
    imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
    figure(3)
    imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]), cmap=cm.gray)
    show()

def test_weighting():
    ###Weighting:  Checks wether the mass of an image is correctly transported into the mass inside a projection
    print("Weighting test")
    
    #Create PyopenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    
   #Relevant quantities
    dtype=float32
    N=900
    img_shape=(N,N)
    angles=30
    detector_width=4
    image_width=4
    Ns=500
	
    #define projectionsetting
    PS=ProjectionSettings(queue, PARALLEL,img_shape,angles,Ns,detector_width=detector_width,image_width=image_width)
    
    #Consider image as rectangular of side-length (4/3) 
    img=np.zeros([N,N])
    img[int(N/3.):int(2*N/3.)][:,int(N/3.):int(2*N/3.)]=1
    img_gpu = cl.array.to_device(queue, require(img, dtype, 'F'))
    
    #compute corresponding Sinogram
    sino_gpu = forwardprojection(img_gpu,PS)	
    
    #Mass inside the image must correspond to the mass any projection
    mass_image=np.sum(img)*PS.delta_x**2
    mass_sinogram_average=np.sum(sino_gpu.get())*PS.delta_s/PS.n_angles
    mass_sino_rdm=np.sum(sino_gpu.get()[:, random.random_integers(0, angles-1) ])*PS.delta_s
    
    print("The mass inside the image is "+str(mass_image)+" was carried over in the mass inside an projection is "+str(mass_sino_rdm)+" i.e. the relative error is "+ str(abs(1-mass_image/mass_sino_rdm)))
    assert((abs(1-mass_image/mass_sino_rdm)<0.001)*(abs(1-mass_image/mass_sino_rdm)<0.001)), "The mass was not carried over correctly into  projections, as the relative difference is "+str(abs(1-mass_image/mass_sino_rdm))
    
def test_adjointness():
    ##Adjointness: Check wether forward and backprojection are indeed adjoint to one another by consider random images and there dual pairing values
    print("Adjointness test")
    
    #Create PyopenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    Nx=900
    number_detectors=400
    img=np.zeros([Nx,Nx])
    angles=360
    
    #define projectionsetting
    PS=ProjectionSettings(queue, PARALLEL, img.shape, angles, n_detectors=number_detectors, fullangle=True)
	

    
    
    #define zero images and sinograms
    sino2_gpu = cl.array.zeros(queue, PS.sinogram_shape, dtype=float32, order='F')
    img2_gpu = cl.array.zeros(queue, PS.img_shape, dtype=float32, order='F')
    
    Error=[]
    count=0
    eps=0.00001
    #Loop through a number of experiments
    for i in range(100):
	#Create random image and sinogram
        img1_gpu = cl.array.to_device(queue, require(np.random.random(PS.img_shape), float32, 'F'))
        sino1_gpu = cl.array.to_device(queue, require(np.random.random(PS.sinogram_shape), float32, 'F'))
        
	#Compute corresponding forward and backprojections
        forwardprojection(img1_gpu,PS,sino=sino2_gpu)
        backprojection(sino1_gpu,PS,img=img2_gpu)
            
        #Extract suitable Information
        sino1=sino1_gpu.get().flatten()
        sino2=sino2_gpu.get().flatten()
        img1=img1_gpu.get().flatten()
        img2=img2_gpu.get().flatten()
	
	#Dual pairing in imagedomain
        a=np.dot(img1,img2)*PS.delta_x**2
	#Dual pairing in sinogram domain
        b=np.dot(sino1,sino2)*(np.pi)/angles*(PS.delta_ratio*PS.delta_x)
        
	#Check whether an error occurred
        if abs(a-b)/min(abs(a),abs(b))>eps:
            count+=1
            Error.append((a,b))
                
    print ('Adjointness: Number of Errors: '+str(count)+' out of 100 tests adjointness-errors were bigger than '+str(eps))
    assert(len(Error)<10),'A large number of experiments for adjointness turned out negative, number of errors: '+str(count)+' out of 100 tests adjointness-errors were bigger than '+str(eps) 

def test_fullangle():
    ###Fullangle: Shows the effect of considering a limited angle setting correctly vs incorrectly.   
    
    #Create PyopenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    

    #relevant quantities
    dtype=float32
    N = 1200
    img = create_phantoms(queue, N, dtype=dtype)
    p=2
    Ns=int(0.3*img.shape[0])
    shift=0

    #Angles cover only part of the angular range
    angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8
    
    #Create two projecetionsettings, one with the correct "fullangle=False" parameter for limited-angle situation, incorrectly using "fullangle=True"
    PScorrect=ProjectionSettings(queue, PARALLEL, img.shape,angles,Ns,detector_width=p,detector_shift=shift,fullangle=False)
    PSincorrect=ProjectionSettings(queue, PARALLEL, img.shape,angles,Ns,detector_width=p,detector_shift=shift,fullangle=True)

    #Forward and backprojection for the two settings
    sino_gpu_correct=forwardprojection(img,PScorrect)
    sino_gpu_incorrect=forwardprojection(img,PSincorrect)
    backprojected_correct=backprojection(sino_gpu_correct,PScorrect)
    backprojected_incorrect=backprojection(sino_gpu_correct,PSincorrect)
        
    #Plot results
    figure(1)
    imshow(np.hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Sinograms with vs without fullangle")
    imshow(np.vstack([np.hstack([sino_gpu_correct.get()[:,:,0],sino_gpu_correct.get()[:,:,1]]),np.hstack([sino_gpu_incorrect.get()[:,:,0],sino_gpu_incorrect.get()[:,:,1]])]), cmap=cm.gray)
    figure(3)
    title("Backprojection with vs without fullangle")
    imshow(np.vstack([np.hstack([backprojected_correct.get()[:,:,0],backprojected_correct.get()[:,:,1]]),np.hstack([backprojected_incorrect.get()[:,:,0],backprojected_incorrect.get()[:,:,1]])]), cmap=cm.gray)
    show()			
        
def test_nonquadratic():
    ####Non-quadratic: Consider the case when non quadratic images are transformed.
    
    #Create PyopenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    #Creat phantom but cut of one side
    dtype=float32
    N1 = 1200
    img = create_phantoms(queue,N1,dtype=dtype)
    N2=int(img.shape[0]*2/3.)
    img=cl.array.to_device(queue, img.get()[:,0:N2,:].copy())
    
    #Additional quantities and setting
    angles=360
    Ns=int(0.5*img.shape[0])
    PS=ProjectionSettings(queue, PARALLEL, img.shape,angles,Ns)
    
    #Compute forward and backprojection
    sino_gpu=forwardprojection(img,PS)
    backprojected=backprojection(sino_gpu,PS)


    #Plot results
    figure(1)
    title("original non square images")
    imshow(np.hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Radon sinogram for non-square image")
    imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
    figure(3)
    title("backprojection for non-square image")
    imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]), cmap=cm.gray)
    show()	
        
# test
if __name__ == '__main__':
    pass
