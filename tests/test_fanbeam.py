import os
from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from gratopy import *
from scipy import misc
from PIL import Image
import matplotlib.image as mpimg

INTERACTIVE=False


def curdir(filename):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)

TESTBRAIN=curdir('brain.png')
TESTDUCK=curdir('duck.jpg')
TESTWALNUT=curdir('walnut.png')
TESTWALNUTSINOGRAM=curdir('walnut_sinogram.png')

ctx = None
queue = None


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
    """  Basic test simply computes forward and backprojection of fanbeam
    transform of two test images, to visually confirm the 
    correctness of the method.    """

    print("Projection test")
    
    # create PyopenCL context 
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test image
    Nx=225
    dtype=float32
    img=create_phantoms(queue,Nx,dtype)
    original=img.get()
    
    # define setting for projection
    number_detectors = 600
    angles=220
    PS = ProjectionSettings(queue, FANBEAM, img_shape=img.shape, 
        angles=angles, detector_width=400, R=752, RE=200, 
	n_detectors=number_detectors)

    
    sino_gpu = clarray.zeros(queue, (PS.n_detectors,PS.n_angles,2), 
        dtype=dtype, order='F')

    # test speed of implementation for forward projection
    a=time.perf_counter()
    for i in range(100):
        forwardprojection(img, PS, sino=sino_gpu)

    print ('Average time required Forward',(time.perf_counter()-a)/100)
    sino=sino_gpu.get()

    numberofangles=180
    angles = linspace(0,2*pi,numberofangles+1)[:-1] 

    a=time.perf_counter()
    for i in range(100):
        backprojection(sino_gpu, PS, img=img)
    print ('Average time required Backprojection',\
        (time.perf_counter()-a)/100)


    # plot results
    figure(1)
    imshow(hstack([original[:,:,0],original[:,:,1]]), cmap=cm.gray)
    title('original image')
    figure(2)
    imshow(hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), \
        cmap=cm.gray)
    
    title('Fanbeam transformed image')
    figure(3)
    imshow(hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    title('Backprojected image')
    show()

def test_weighting():
    """ Checks whether the mass of an image is correctly transported into
    the mass inside a projection. As the object has a larger shadow than 
    itself, the mass in the sinogram is roughly a multiplication of the 
    mass by the ratio of R to RE. Indicates that scaling of the transform 
    is suitable.
    """
    print("Weighting;")
    
    # create PyopenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # execute for different number of detectors, to ensure
    # resolution independence
    for number_detectors in [50,100,200,400,800,1600]:

        # consider full image
        Nx=400
        img=np.ones([Nx,Nx])
        angles=720

        # relevant quantities for scaling
        rescaling=1/40.*sqrt(2)
        detector_width=400*rescaling
        R=1200.*rescaling
        RE=200.*rescaling
        image_width=40.*rescaling

        # create projectionsetting
        PS = ProjectionSettings(queue, FANBEAM, img_shape=img.shape, 
	     angles=angles, detector_width=detector_width, R=R,RE= RE,
	    n_detectors=number_detectors,image_width=image_width)
        
        # forward and backprojection
        img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
        sino_gpu= forwardprojection(img_gpu, PS)
	
	# compute mass inside object and on the detector
        mass_in_image=np.sum(img_gpu.get())*PS.delta_x**2
        mass_in_projcetion=np.sum(sino_gpu.get())*(PS.delta_ratio*PS.delta_x)\
	    /angles

        print("Mass in original image",mass_in_image, "mass in projection",\
            mass_in_projcetion,"Ratio",mass_in_projcetion/mass_in_image,\
            "Ratio should be "+str(R/RE))
        
        assert( abs(mass_in_projcetion/mass_in_image/(R/RE)-1)<0.1),\
            "Due to the fan effect the object is enlarged on the detector,\
            roughly by the ratio of the distances, but this was not\
            satisfied in this test."

def test_adjointness():
    """ 
    Randomly creates images and sinograms to check whether forward 
    and backprojection are indeed adjoint to one another 
    (by considering corresponding dual pairings). 
    This is carried out for multiple experiments.
    """

    print("Adjointness:")

    # create PyopenCL context 
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    number_detectors=230
    img=np.zeros([400,400])
    angles=390
    midpoint_shift=[100,100]

    # define projection setting
    PS = ProjectionSettings(queue, FANBEAM, img.shape, angles, 
        n_detectors=number_detectors, detector_width=83,
	detector_shift=0.0, midpoint_shift=[0,0],
	R=900, RE=300, image_width=None, fullangle=True)

    # preliminary definitions
    Error=[]
    count=0
    eps=0.00001

    img2_gpu = clarray.zeros(queue, PS.img_shape, dtype=float32, \
        order='F')
    sino2_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=float32, \
        order='F')
    for i in range(100):
	
	# loop through a number of experiments
        img1_gpu = clarray.to_device(queue, \
	     require(np.random.random(PS.img_shape), float32, 'F'))
        sino1_gpu = clarray.to_device(queue, \
	     require(np.random.random(PS.sinogram_shape), float32, 'F'))
        
        # compute corresponding forward and backprojections
        forwardprojection(img1_gpu, PS, sino=sino2_gpu)
        backprojection(sino1_gpu, PS, img=img2_gpu)
        
	# extract suitable information
        sino1=sino1_gpu.get().flatten()
        sino2=sino2_gpu.get().flatten()
        img1=img1_gpu.get().flatten()
        img2=img2_gpu.get().flatten()

        # dual pairing in image domain
        a=np.dot(img1,img2)*PS.delta_x**2
        # dual pairing in sinogram domain
        b=np.dot(sino1,sino2)*(2*np.pi)/angles*(PS.delta_ratio*PS.delta_x)
        
        # check whether an error occurred
        if abs(a-b)/min(abs(a),abs(b))>eps:
            print (a,b,a/b)
            count+=1
            Error.append((a,b))

    print ('Adjointness: Number of Errors: '+str(count)+' out of 100\
        tests adjointness-errors were bigger than '+str(eps))
    assert(len(Error)<10),'A large number of experiments for adjointness\
        turned out negative, number of errors: '+str(count)+' out of 100\
        tests adjointness-errors were bigger than '+str(eps) 

def test_fullangle():
    """
    Illustrates the impact of the full-angle parameter, in particular
    showing artifacts resulting from incorrect use for the limited 
    angle setting.
    """
    
    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test phantom
    dtype=float32
    N = 1200
    img = create_phantoms(queue, N,dtype)
    
    # relevant quantities
    Ns=int(0.3*img.shape[0])
    shift=0
    (R, RE, Detector_width, image_width)=(5, 2, 6, 2)
    
    # angles cover only part of the angular range
    angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8
   
    # create two projecetionsettings, one with the correct "fullangle=False"
    # parameter for limited-angle situation, incorrectly using "fullangle=True"
    PScorrect = ProjectionSettings(queue, FANBEAM, img.shape, angles, Ns,
	                           image_width=image_width, R=R, RE=RE,
				   detector_width=Detector_width,
	                           detector_shift=shift, fullangle=False)
	
    PSincorrect = ProjectionSettings(queue, FANBEAM, img.shape, angles, Ns,
                                     image_width=image_width, R=R, RE=RE,
                                     detector_width=Detector_width, 
				     detector_shift=shift,fullangle=True)

    # show geometry of the problem
    PScorrect.show_geometry(np.pi/4, show=False)
     
    # forward and backprojection for the two settings
    sino_gpu_correct=forwardprojection(img, PScorrect)
    sino_gpu_incorrect=forwardprojection(img, PSincorrect)
    backprojected_correct=backprojection(sino_gpu_correct, PScorrect)
    backprojected_incorrect=backprojection(sino_gpu_correct, PSincorrect)

    # plot results
    figure(1)
    imshow(np.hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Sinograms with vs without fullangle")
    imshow(np.vstack([np.hstack([sino_gpu_correct.get()[:,:,0],\
        sino_gpu_correct.get()[:,:,1]]),\
	np.hstack([sino_gpu_incorrect.get()[:,:,0],\
	sino_gpu_incorrect.get()[:,:,1]])]), cmap=cm.gray)
    
    figure(3)
    title("Backprojection with vs without fullangle")
    imshow(np.vstack([np.hstack([backprojected_correct.get()[:,:,0],\
        backprojected_correct.get()[:,:,1]]),\
	np.hstack([backprojected_incorrect.get()[:,:,0],\
	backprojected_incorrect.get()[:,:,1]])]), cmap=cm.gray)
    
    show()

def test_midpointshift():
    """ 
    Illustrates how the sinogram changes if the midpoint of an 
    images is shifted away from the center of roation.
    """

    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create phantom for test
    dtype=float32
    N = 1200
    img = create_phantoms(queue, N,dtype)

    # relevant quantities
    (angles, R, RE, Detector_width, image_width, shift) = (360, 5, 3, 6, 2, 0)
    midpoint_shift=[0,0.5]
    Ns=int(0.5*img.shape[0])

    # define projectionsetting
    PS = ProjectionSettings(queue, FANBEAM, img.shape, angles, Ns,
                            image_width=image_width, R=R, RE=RE,
                            detector_width=Detector_width, detector_shift=shift,
                            midpoint_shift=midpoint_shift,fullangle=True)
    
    # plot the geometry from various angles
    figure(0)
    for k in range(0,16):
        PS.show_geometry(k*np.pi/8, axes=subplot(4,4,k+1))

    # compute forward and backprojection
    sino_gpu=forwardprojection(img, PS)    
    backprojected=backprojection(sino_gpu, PS)

    # plot results
    figure(1)
    imshow(np.hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Sinogram with shifted midpoint")
    imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), \
        cmap=cm.gray)
    figure(3)
    title("Backprojection with shifted midpoint")
    imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]),\
        cmap=cm.gray)
    show()

def test_landweber():
    """ 
    Executes the Landweber iteration for the `walnut dataset <https://arxiv.org/abs/1905.04787>`_  to compute an inversion of a 
    sinogram, testing the implementation.
    """
    print("Walnut Landweber reconstruction test")

    # create phantom for test
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # load and rescale Walnut image
    walnut=mpimg.imread(TESTWALNUT)
    walnut=walnut/np.mean(walnut)
    walnut[np.where(walnut<=1.5)]=0

    # geometric quantities
    dtype=float
    number_detectors=328
    (Detectorwidth, FOD, FDD, numberofangles) = (114.8, 110, 300, 120)
    geometry=[Detectorwidth,FDD,FOD,number_detectors]
    
    # create projectionsetting
    PS = ProjectionSettings(queue, FANBEAM, img_shape=walnut.shape,
                            angles = numberofangles, detector_width=Detectorwidth,
                            R=FDD, RE=FOD, n_detectors=number_detectors)

    # copy image to the gpu
    walnut_gpu=clarray.to_device(queue,require(walnut,dtype,'F'))
    
    # compute forward and backprojection
    sino=forwardprojection(walnut_gpu, PS)	
    walnutbp=backprojection(sino, PS)

    # plot results for forward and backprojection
    figure(1)
    imshow(walnut, cmap=cm.gray)
    title('original walnut')
    figure(2)
    imshow(sino.get(), cmap=cm.gray)
    title('Fanbeam transformed image')
    figure(3)
    imshow(walnutbp.get(), cmap=cm.gray)
    title('Backprojected image')

    # load sinogram of walnut and rescale
    sinonew=mpimg.imread(TESTWALNUTSINOGRAM)
    sinonew/=np.mean(sinonew)

    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110, 300)
    angles = linspace(0,2*pi,121)[:-1] + pi/2
    geometry=[Detectorwidth,FDD,FOD,number_detectors]
    img_shape=(600,600)
    
    # create projectionsetting
    PS = ProjectionSettings(queue, FANBEAM, img_shape=img_shape, angles=angles,
                            detector_width=Detectorwidth, R=FDD, RE=FOD,
                            n_detectors=number_detectors)
    
    my_phantom=phantom(queue,img_shape)
    my_phantom_sinogram=forwardprojection(my_phantom,PS)
    
    sino=np.zeros(PS.sinogram_shape+tuple([2]))    
    sino[:,:,0]=sinonew/np.max(sinonew)
    sino[:,:,1]=my_phantom_sinogram.get()/np.max(my_phantom_sinogram.get())
    
    walnut_gpu2new=clarray.to_device(queue,require(sino,dtype,'F'))
    
    # execute Landweber method
    ULW=landweber(walnut_gpu2new, PS, 20)

    # plot Landweber reconstruction
    figure(4)
    imshow(np.hstack([ULW.get()[:,:,0],ULW.get()[:,:,1]]),cmap=cm.gray)
    title("Landweber reconstruction")
    show()

def test_conjugate_gradients():
    """ 
    Executes the conjugate gradients iteration for the `walnut dataset`_  to compute an 
    inversion of a sinogram, testing the implementation.
    """
    print("Walnut conjugated_gradients reconstruction test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

  
    dtype=float32
    # load and rescale image
    sinonew=mpimg.imread(TESTWALNUTSINOGRAM)
    sinonew=np.array(sinonew,dtype=dtype)
    sinonew/=np.mean(sinonew)

    # geometric quantities
    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110, 300)
    angles = linspace(0,2*pi,121)[:-1] + pi/2
    geometry=[Detectorwidth,FDD,FOD,number_detectors]
    img_shape=(600,600)

    # create projectionsetting
    PS = ProjectionSettings(queue, FANBEAM, img_shape=img_shape, angles=angles,
                            detector_width=Detectorwidth, R=FDD, RE=FOD,
                            n_detectors=number_detectors)

    my_phantom=phantom(queue,img_shape)
    my_phantom_sinogram=forwardprojection(my_phantom,PS)


    sino=np.zeros(PS.sinogram_shape+tuple([2]))
    sino[:,:,0]=sinonew/np.max(sinonew)
    sino[:,:,1]=my_phantom_sinogram.get()/np.max(my_phantom_sinogram.get())
  
    # perform conjugate gradients algorithm
    walnut_gpu2new=clarray.to_device(queue,require(sino,dtype,'C'))
    UCG=conjugate_gradients(walnut_gpu2new, PS, 0.1,number_iterations=100)

    # plot results
    figure(4)
    imshow(np.hstack([UCG.get()[:,:,0],UCG.get()[:,:,1]]),cmap=cm.gray)
    title("Conjugate gradients reconstruction")
    show()


def test_total_variation():
    """ 
    Executes the total variation approach to compute an inversion of
    a sinogram from the `walnut dataset`_, testing the implementation.
    """
    print("Walnut total variation reconstruction test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    dtype=float32
    number_detectors=328
    (Detectorwidth, FOD, FDD, numberofangles) = (114.8, 110, 300, 120)
    img_shape=(328,328)

    # create projectionsetting
    PS = ProjectionSettings(queue, FANBEAM, img_shape=img_shape,
    	                    angles = numberofangles, 
			    detector_width=Detectorwidth,
    	                    R=FDD, RE=FOD, n_detectors=number_detectors)

    # load and rescale sinogram
    sinonew=mpimg.imread(TESTWALNUTSINOGRAM)
    sinonew=np.array(sinonew,dtype=dtype)
    sinonew/=np.mean(sinonew)
    sinonew+=+np.random.randn(sinonew.shape[0],sinonew.shape[1])*0.2
    
    # perform total_variation reconstruction
    walnut_gpu2new=clarray.to_device(queue,require(sinonew,dtype,'C'))

    UTV=total_variation_reconstruction(walnut_gpu2new,PS,mu=1000,
                                       number_iterations=5000,z_distance=0)
    
    sinoreprojected=forwardprojection(UTV,PS)
    
    # plot results
    figure(4)
    imshow(UTV.get(),cmap=cm.gray)

    title("total variation reconstruction")
    figure(5)
    imshow(sinoreprojected.get(),cmap=cm.gray)
    title("reprojected sinogram of solution")
    show()


def test_nonquadratic():
    """Illustrates the use of gratopy for non-quadratic images. """
    
    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create phantom and cut one side
    N = 1200
    img = create_phantoms(queue, N)
    N1=img.shape[0]
    N2=int(img.shape[0]*2/3.)
    img=cl.array.to_device(queue, img.get()[:,0:N2,:].copy())

    # geometric quantities
    (angles, R, RE, Detector_width, shift) = (360, 5, 3, 5, 0)
    image_width=None
    midpoint_shift=[0,0.]
    dtype=float32
    Ns=int(0.5*img.shape[0])

    # create projectionsetting
    PS = ProjectionSettings(queue, FANBEAM, img.shape, angles, Ns, 
        image_width=image_width,R=R, RE=RE, detector_width=Detector_width,
	detector_shift=shift, midpoint_shift=midpoint_shift, fullangle=True)

    # compute forward and backprojection
    sino_gpu=forwardprojection(img, PS)
    backprojected=backprojection(sino_gpu, PS)

    # show geometry of projectionsetting
    PS.show_geometry(1*np.pi/8, show=False)

    # plot results
    figure(1)
    title("original non square images")
    imshow(np.hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Fanbeam sinogram for non-square image")
    imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), 
        cmap=cm.gray)
    
    figure(3)
    title("backprojection for non-square image")
    imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]),
        cmap=cm.gray)
    show()

# test
if __name__ == '__main__':
    pass
