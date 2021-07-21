import os, pkgutil
from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
from gratopy import *
import matplotlib.image as mpimg

from numpy import random

ctx = None
queue = None

def create_random_test_numbers():
    m=1000
    M=2000
    rng=np.random.default_rng(1)
    mylist=[]
    
    mylist.append(rng.integers(0,M,m))#s
    mylist.append(rng.integers(0,M,m))#phi
    mylist.append(rng.integers(0,M,m))#z
    mylist.append(rng.normal(0,1,m)) #factors
    
    mylist.append(rng.integers(0,M,m))#
    mylist.append(rng.integers(0,M,m))
    
    myfile=open("rnd.txt","w")
    
    for j in range(6):	
        for i in range(m):
             myfile.write(str(mylist[j][i])+"\n")
    myfile.close()
	


def read_random_test_numbers(Nx,Ny,Ns,Na, Nz=1):
    
    myfile=open("rnd.txt","r")
    m = 1000
    test_s=[]
    test_phi=[]
    test_z=[]
    factors=[]
    test_x=[]
    test_y=[]
        
    text=myfile.readlines()
    
    for i in range(m):
        test_s.append(int(text[i])%Ns)
        test_phi.append(int(text[i+m])%Na)
        test_z.append(int(text[i+2*m])%Nz)
        factors.append(float(text[i+3*m]))
        test_x.append(int(text[i+4*m])%Nx)
        test_y.append(int(text[i+5*m])%Ny)
    myfile.close()
    return test_s,test_phi,test_z,factors,test_x,test_y


def create_phantoms(queue, N, dtype='double'):
    # use gratopy phantom method to create Shepp-Logan phantom
    A=phantom(queue, N, dtype=dtype)
    A *= 255/cl.array.max(A).get()
    
    # second test image consisting of 2 horizontal bars
    B=cl.array.empty(queue, A.shape, dtype=dtype)
    B[:] = 255-120
    B[int(N/3):int(2*N/3)]=0
    B[0:int(N/4)]=0
    B[int(N-N/4):N]=0
    
    img=cl.array.to_device(queue, np.stack([A.get(), B.get()], axis=-1))
    return img

def test_projection():
    """  Basic projection test. Simply computes forward and backprojection 
    of the Radon transform for two test images in order to visually confirm 
    the correctness of the method. """

    print("Projection test")

    # create PyopenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    
    # create test image
    dtype=float32
    N = 1200
    img = create_phantoms(queue, N, dtype=dtype)
	
    # define relevant quantities to determine the geometry
    angles=360
    detector_width=4
    image_width=4
    Ns=int(0.5*img.shape[0])

    # create projectionsetting with parallel beam setting with 360
    # equi-distant angles, the detector has a width of 4 and the observed
    # object has a diameter of 4 (i.e. is captured by the detector), we
    # consider half the amount of detector pixels as image pixels
    PS=ProjectionSettings(queue, PARALLEL, img.shape, angles, Ns,
                              image_width=image_width,
			      detector_width=detector_width,
                              detector_shift=0,
                              fullangle=True)

    # plot geometry via show_geometry method to visualize geometry
    figure(0)
    PS.show_geometry(0, axes=subplot(2,2,1))
    PS.show_geometry(np.pi/8, axes=subplot(2,2,2))
    PS.show_geometry(np.pi/4, axes=subplot(2,2,3))
    PS.show_geometry(np.pi*3/8., axes=subplot(2,2,4))
    
    # compute Radon transform for given test images
    sino_gpu = forwardprojection(img, PS)

    # compute backprojection of computed sinogram
    backprojected_gpu = backprojection(sino_gpu, PS)
    
    img=img.get()
    sino=sino_gpu.get()
    backprojected=backprojected_gpu.get()
    
    
    # plot results
    figure(1)
    imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
    figure(2)
    imshow(np.hstack([sino[:,:,0],sino[:,:,1]]),
        cmap=cm.gray)
    figure(3)
    imshow(np.hstack([backprojected[:,:,0],
        backprojected[:,:,1]]), cmap=cm.gray)
    
    show()
    
    
    # Computing a controlnumbers to quantitatively verify correctness 
    m=1000
    test_s,test_phi,test_z,factors,test_x,test_y=read_random_test_numbers(
                                                     N,N,Ns,angles,2)
    mysum0=0
    mysum1=0
    mysum2=0
    for i in range(0,m):    
        mysum0+=factors[i]*img[test_x[i],test_y[i],test_z[i]]
        mysum1+=factors[i]*sino[test_s[i],test_phi[i],test_z[i]]
        mysum2+=factors[i]*backprojected[test_x[i],test_y[i],test_z[i]]

    mysumtrue0=2949.373863867869       
    mysumtrue1=1221.223836630344
    mysumtrue2=7427.706215239802
    
    assert(abs(mysum0-mysumtrue0)<0.001),\
        "A control-sum for the original image did not match the expected value,\
        expected: "+str(mysumtrue0) +", received: "+str(mysum0)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error."  
    
    assert(abs(mysum1-mysumtrue1)<0.001),\
        "A control-sum for the sinogram did not match the expected value,\
        expected: "+str(mysumtrue1) +", received: "+str(mysum1)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error."
	
    assert(abs(mysum2-mysumtrue2)<0.001),\
        "A control-sum for the backprojection did not match the expected value,\
        expected: "+str(mysumtrue2) +", received: "+str(mysum2)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error." 
    

def test_weighting():
    """ Mass preservation test. Check whether the total mass of an image 
    (square with side length 4/3 and pixel values, i.e. density, 1) 
    is correctly transported into the total mass of a projection, i.e., 
    the scaling is adequate.
    """
    print("Weighting test")
    
    # create PyopenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    
    # relevant quantities
    dtype=float32
    N=900
    img_shape=(N,N)
    angles=30
    detector_width=4
    image_width=4
    Ns=500
	
    # define projectionsetting
    PS=ProjectionSettings(queue, PARALLEL,img_shape,angles,Ns,
        detector_width=detector_width,image_width=image_width)
    
    # consider image as rectangular of side-length (4/3)
    img=np.zeros([N,N])
    img[int(N/3.):int(2*N/3.)][:,int(N/3.):int(2*N/3.)]=1
    img_gpu = cl.array.to_device(queue, require(img, dtype, 'F'))
    
    # compute corresponding sinogram
    sino_gpu = forwardprojection(img_gpu,PS)	
    
    # mass inside the image must correspond to the mass any projection
    mass_image=np.sum(img)*PS.delta_x**2
    mass_sinogram_average=np.sum(sino_gpu.get())*PS.delta_s/PS.n_angles
    mass_sino_rdm=np.sum(sino_gpu.get()[:, random.randint(0, angles) ])\
        *PS.delta_s
    
    print("The mass inside the image is "+str(mass_image)+
        " was carried over in the mass inside an projection is "
	+str(mass_sino_rdm)+" i.e. the relative error is "
	+ str(abs(1-mass_image/mass_sino_rdm)))
	
    assert((abs(1-mass_image/mass_sino_rdm)<0.001)*\
        (abs(1-mass_image/mass_sino_rdm)<0.001)),\
        "The mass was not carried over correctly into  projections,\
        as the relative difference is "\
	+str(abs(1-mass_image/mass_sino_rdm))
    
def test_adjointness():
    """ Adjointness test. Creates random images 
    and sinograms to check whether forward and backprojection are indeed 
    adjoint to one another (by comparing the corresponding dual pairings). 
    This comparison is carried out for multiple experiments.
    """
    print("Adjointness test")
    
    # create PyOpenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    Nx=900
    number_detectors=400
    img=np.zeros([Nx,Nx])
    angles=360
    
    # define projectionsetting
    PS=ProjectionSettings(queue, PARALLEL, img.shape, angles, 
        n_detectors=number_detectors, fullangle=True)
    
    # define zero images and sinograms
    sino2_gpu = cl.array.zeros(queue, PS.sinogram_shape, dtype=float32, 
        order='F')
    
    img2_gpu = cl.array.zeros(queue, PS.img_shape, dtype=float32, order='F')
    
    Error=[]
    count=0
    eps=0.00001
    # loop through a number of experiments
    for i in range(100):
	# create random image and sinogram
        img1_gpu = cl.array.to_device(queue, 
            require(np.random.random(PS.img_shape), float32, 'F'))
        
        sino1_gpu = cl.array.to_device(queue,\
            require(np.random.random(PS.sinogram_shape), float32, 'F'))
        
	# compute corresponding forward and backprojections
        forwardprojection(img1_gpu,PS,sino=sino2_gpu)
        backprojection(sino1_gpu,PS,img=img2_gpu)
            
        # extract suitable Information
        sino1=sino1_gpu.get().flatten()
        sino2=sino2_gpu.get().flatten()
        img1=img1_gpu.get().flatten()
        img2=img2_gpu.get().flatten()
	
	# dual pairing in imagedomain
        a=np.dot(img1,img2)*PS.delta_x**2

	# dual pairing in sinogram domain
        b=np.dot(sino1,sino2)*(np.pi)/angles*(PS.delta_ratio*PS.delta_x)
        
	# check whether an error occurred
        if abs(a-b)/min(abs(a),abs(b))>eps:
            count+=1
            Error.append((a,b))
                
    print ('Adjointness: Number of Errors: '+str(count)+' out of\
        100 tests adjointness-errors were bigger than '+str(eps))
    
    assert(len(Error)<10),'A large number of experiments for adjointness\
        turned out negative, number of errors: '+str(count)+' out of 100\
	tests adjointness-errors were bigger than '+str(eps) 

def test_fullangle():
    """ Full-angle test. Tests and illustrates the impact of the fullangle 
    parameter, in particular showing artifacts resulting from the incorrect 
    use of the limited angle setting.
    """
       
    # create PyOpenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    dtype=float32
    N = 1200
    img_gpu = create_phantoms(queue, N, dtype=dtype)
    p=2
    Ns=int(0.3*N)
    shift=0

    # angles cover only a part of the angular range
    angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8
    
    # create two projecetionsettings, one with the correct "fullangle=False"
    # parameter for limited-angle situation, incorrectly using "fullangle=True"
    PScorrect=ProjectionSettings(queue, PARALLEL, img_gpu.shape,angles,Ns,
       detector_width=p,detector_shift=shift,fullangle=False)
    PSincorrect=ProjectionSettings(queue, PARALLEL, img_gpu.shape,angles,Ns,
        detector_width=p,detector_shift=shift,fullangle=True)

    # forward and backprojection for the two settings
    sino_gpu_correct=forwardprojection(img_gpu,PScorrect)
    sino_gpu_incorrect=forwardprojection(img_gpu,PSincorrect)
    backprojected_gpu_correct=backprojection(sino_gpu_correct,PScorrect)
    backprojected_gpu_incorrect=backprojection(sino_gpu_correct,PSincorrect)
        
    sino_correct=sino_gpu_correct.get()
    sino_incorrect=sino_gpu_incorrect.get()
    backprojected_correct=backprojected_gpu_correct.get()
    backprojected_incorrect=backprojected_gpu_incorrect.get()
    
    img=img_gpu.get()

        
    # plot results
    figure(1)
    imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Sinograms with vs without fullangle")
    imshow(np.vstack([np.hstack([sino_correct[:,:,0],\
        sino_correct[:,:,1]]),\
	np.hstack([sino_incorrect[:,:,0],\
	sino_incorrect[:,:,1]])]), cmap=cm.gray)
    
    figure(3)
    title("Backprojection with vs without fullangle")
    imshow(np.vstack([np.hstack([backprojected_correct[:,:,0],\
        backprojected_correct[:,:,1]]),\
	np.hstack([backprojected_incorrect[:,:,0],\
	backprojected_incorrect[:,:,1]])]), cmap=cm.gray)
    
    show()
    
    # Computing a controlnumbers to quantitatively verify correctness 
    m=1000
    test_s,test_phi,test_z,factors,test_x,test_y=read_random_test_numbers(
                                                     N,N,Ns,len(angles),2)
    
    
    mysum0=0
    mysum1=0
    mysum2=0
    mysum3=0
    mysum4=0
    for i in range(0,m):    
        
        mysum0+=factors[i]*img[test_x[i],test_y[i],test_z[i]]
        mysum1+=factors[i]*sino_correct[test_s[i],test_phi[i],test_z[i]]
        mysum2+=factors[i]*sino_incorrect[test_s[i],test_phi[i],test_z[i]]
        mysum3+=factors[i]*backprojected_correct[test_x[i],test_y[i],test_z[i]]
        mysum4+=factors[i]*backprojected_incorrect[test_x[i],test_y[i],test_z[i]]

    mysumtrue0=2949.373863867869       
    mysumtrue1=990.31814758003
    mysumtrue2=990.31814758003
    mysumtrue3=1357.265059450704
    mysumtrue4=2409.5415802098414
    assert(abs(mysum0-mysumtrue0)<0.001),\
        "A control-sum for the original image did not match the expected value,\
        expected: "+str(mysumtrue0) +", received: "+str(mysum0)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error."  
    
    assert(abs(mysum1-mysumtrue1)<0.001),\
        "A control-sum for the sinogram with correct fullangle did not match the expected value,\
        expected: "+str(mysumtrue1) +", received: "+str(mysum1)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error."
    
    assert(abs(mysum2-mysumtrue2)<0.001),\
        "A control-sum for the sinogram with incorrect fullangle did not match the expected value,\
        expected: "+str(mysumtrue2) +", received: "+str(mysum2)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error."

	
    assert(abs(mysum3-mysumtrue3)<0.001),\
        "A control-sum for the backprojection with correct fullangle did not match the expected value,\
        expected: "+str(mysumtrue3) +", received: "+str(mysum3)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error." 

    assert(abs(mysum4-mysumtrue4)<0.001),\
        "A control-sum for the backprojection with correct fullangle did not match the expected value,\
        expected: "+str(mysumtrue4) +", received: "+str(mysum4)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error." 

		
        
def test_nonquadratic():
    """ Nonquadratic image test. Tests and illustrates the projection 
    operator for non-quadratic images. """
    
    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # create phantom but cut of one side
    dtype=float32
    N1 = 1200
    img = create_phantoms(queue,N1,dtype=dtype)
    N2=int(img.shape[0]*2/3.)
    img_gpu=cl.array.to_device(queue, img.get()[:,0:N2,:].copy())
    
    # additional quantities and setting
    angles=360
    Ns=int(0.5*img_gpu.shape[0])
    PS=ProjectionSettings(queue, PARALLEL, img_gpu.shape,angles,Ns)
    
    # compute forward and backprojection
    sino_gpu=forwardprojection(img_gpu,PS)
    backprojected_gpu=backprojection(sino_gpu,PS)
    
    img=img_gpu.get()
    sino=sino_gpu.get()
    backprojected=backprojected_gpu.get()

    # plot results
    figure(1)
    title("original non square images")
    imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Radon sinogram for non-square image")
    imshow(np.hstack([sino[:,:,0],sino[:,:,1]]),\
        cmap=cm.gray)
    
    figure(3)
    title("backprojection for non-square image")
    imshow(np.hstack([backprojected[:,:,0],\
        backprojected[:,:,1]]), cmap=cm.gray)
    
    show()
    
    # Computing a controlnumbers to quantitatively verify correctness 
    m=1000
    test_s,test_phi,test_z,factors,test_x,test_y=read_random_test_numbers(
                                                     N1,N2,Ns,angles,2)
    mysum0=0
    mysum1=0
    mysum2=0
    for i in range(0,m):    
        mysum0+=factors[i]*img[test_x[i],test_y[i],test_z[i]]
        mysum1+=factors[i]*sino[test_s[i],test_phi[i],test_z[i]]
        mysum2+=factors[i]*backprojected[test_x[i],test_y[i],test_z[i]]

    mysumtrue0=999.4965329492022       
    mysumtrue1=-782.3501868881585
    mysumtrue2=3310.347556576363
    
    assert(abs(mysum0-mysumtrue0)<0.001),\
        "A control-sum for the original image did not match the expected value,\
        expected: "+str(mysumtrue0) +", received: "+str(mysum0)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error."  
    
    assert(abs(mysum1-mysumtrue1)<0.001),\
        "A control-sum for the sinogram did not match the expected value,\
        expected: "+str(mysumtrue1) +", received: "+str(mysum1)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error."
	
    assert(abs(mysum2-mysumtrue2)<0.001),\
        "A control-sum for the backprojection did not match the expected value,\
        expected: "+str(mysumtrue2) +", received: "+str(mysum2)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error." 
    


def test_extract_sparse_matrix():
    order="F"
    dtype=float64
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    Nx=150
    number_detectors=100
    img=np.zeros([Nx,Nx])
    angles=30
    
    # define projectionsetting
    PS=ProjectionSettings(queue, PARALLEL, img.shape, angles, 
                          n_detectors=number_detectors, fullangle=True)
    
    #Create corresponding sparse matrix
    sparsematrix=PS.create_sparse_matrix(dtype=dtype,order=order)
    
    #Testimage
    img=phantom(queue, Nx, dtype)
    img=img.get()
    img=img.reshape(Nx**2,order=order)
    #Compute forward and backprojection
    sino=sparsematrix*img
    backproj=sparsematrix.T*sino

    #reshape
    img=img.reshape(Nx,Nx,order=order)
    sino=sino.reshape(number_detectors,angles,order=order)
    backproj=backproj.reshape(Nx,Nx,order=order)

    #plot results
    figure(1)
    title("Testimage")
    imshow(img,cmap=cm.gray)

    figure(2)
    title("projection via spase Matrix")
    imshow(sino,cmap=cm.gray)
    
    figure(3)
    title("backprojection via spase Matrix")
    imshow(backproj,cmap=cm.gray)
    show()
    
    # Computing a controlnumbers to quantitatively verify correctness 
    m=1000
    test_s,test_phi,test_z,factors,test_x,test_y=read_random_test_numbers(
                                                     Nx,Nx,number_detectors,angles,1)
    mysum0=0
    mysum1=0
    mysum2=0
    for i in range(0,m): 
        mysum0+=factors[i]*img[test_x[i],test_y[i]]
        mysum1+=factors[i]*sino[test_s[i],test_phi[i]]
        mysum2+=factors[i]*backproj[test_x[i],test_y[i]]

    mysumtrue0=7.118201777449383       
    mysumtrue1=-0.5455533171777472
    mysumtrue2=0.7838917015784652
    assert(abs(mysum0-mysumtrue0)<0.000001),\
        "A control-sum for the original image did not match the expected value,\
        expected: "+str(mysumtrue0) +", received: "+str(mysum0)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error."  
    
    assert(abs(mysum1-mysumtrue1)<0.000001),\
        "A control-sum for the sinogram did not match the expected value,\
        expected: "+str(mysumtrue1) +", received: "+str(mysum1)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error."
	
    assert(abs(mysum2-mysumtrue2)<0.000001),\
        "A control-sum for the backprojection did not match the expected value,\
        expected: "+str(mysumtrue2) +", received: "+str(mysum2)+\
        ". Please consider the visual results to check whether this is \
        a numerical issue or a more fundamental error." 


# test
if __name__ == '__main__':
    pass
