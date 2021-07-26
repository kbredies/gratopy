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

TESTWALNUT=curdir('walnut.png')
TESTWALNUTSINOGRAM=curdir('walnut_sinogram.png')
TESTRNG=curdir("rng.txt")

ctx = None
queue = None



def evaluate_control_numbers(data, dimensions,expected_result,precision,classified,name):
    [Nx,Ny,Ns,Na,Nz]=dimensions
    
    test_s,test_phi,test_z,factors,test_x,test_y=read_control_numbers(
                                                     Nx,Ny,Ns,Na,Nz)
    m=1000
    mysum=0
    if classified=="img":
        var1=test_x
        var2=test_y
        var3=test_z
    else:
        var1=test_s
        var2=test_phi
        var3=test_z
    
    if Nz==1:
        data=data.reshape(data.shape[0],data.shape[1],1)
    for i in range(0,m):    
        mysum+=factors[i]*data[var1[i],var2[i],var3[i]]
    
    assert(abs(mysum-expected_result)<precision),\
        "A control-sum for the "+name+ " did not match the expected value,"\
        +"expected: "+str(expected_result) +", received: "+str(mysum)+\
        ". Please consider the visual results to check whether this is "+\
        "a numerical issue or a more fundamental error."  




def create_control_numbers():
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
    
    myfile=open(TESTRNG,"w")
    
    for j in range(6):	
        for i in range(m):
             myfile.write(str(mylist[j][i])+"\n")
    myfile.close()
	


def read_control_numbers(Nx,Ny,Ns,Na, Nz=1):
    
    myfile=open(TESTRNG,"r")
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
    """  Basic projection test. Computes the forward and backprojection of 
    the fanbeam transform for two test images to visually confirm the 
    correctness of the method.    """

    print("Projection test")
    
    # create PyopenCL context 
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test image
    Nx=1200
    dtype=float32
    img=create_phantoms(queue,Nx,dtype)
    original=img.get()
    
    # define setting for projection
    number_detectors = 600
    angles=360
    PS = ProjectionSettings(queue, FANBEAM, img_shape=img.shape, 
        angles=angles, detector_width=400, R=752, RE=200, 
	n_detectors=number_detectors)

    
    sino_gpu = clarray.zeros(queue, (PS.n_detectors,PS.n_angles,2), 
        dtype=dtype, order='F')
	
    backprojected_gpu = clarray.zeros(queue, (PS.img_shape+tuple([2])), 
        dtype=dtype, order='F')


    # test speed of implementation for forward projection
    iterations=10
    a=time.perf_counter()
    for i in range(iterations):
        forwardprojection(img, PS, sino=sino_gpu)
    img.get()

    print ('Average time required Forward',(time.perf_counter()-a)/iterations)


    a=time.perf_counter()
    for i in range(iterations):
        backprojection(sino_gpu, PS, img=backprojected_gpu)
    sino_gpu.get()
    print ('Average time required Backprojection',\
        (time.perf_counter()-a)/iterations)
    
    sino=sino_gpu.get()
    backprojected=backprojected_gpu.get()


    # plot results
    figure(1)
    imshow(hstack([original[:,:,0],original[:,:,1]]), cmap=cm.gray)
    title('original image')
    figure(2)
    imshow(hstack([sino[:,:,0],sino[:,:,1]]), \
        cmap=cm.gray)
    
    title('Fanbeam transformed image')
    figure(3)
    imshow(hstack([backprojected[:,:,0],backprojected[:,:,1]]), cmap=cm.gray)
    title('Backprojected image')
    show()
    
    # Computing controlnumbers to quantitatively verify correctness 
    evaluate_control_numbers(img, (Nx,Nx,number_detectors,angles,2),
                expected_result=2949.3728,
		precision=0.001,classified="img",name="original image")

    evaluate_control_numbers(sino, (Nx,Nx,number_detectors,angles,2),
                expected_result=66998.337281,
		precision=0.001,classified="sino",name="sinogram")

    evaluate_control_numbers(backprojected, (Nx,Nx,number_detectors,angles,2),expected_result=1482240.72690,
		precision=0.001,classified="img",name="backprojected image")

    

def test_weighting():
    """ Mass preservation test. Checks whether the total mass of an image 
    is correctly transported into the total mass of a projection. 
    Due to the fan geometry the width of a projected object on the detector is 
    wider than the original object was, as the width of the fan grows linearly
    with the distance it travels. Consequently, also the total mass on the
    detector is rougly the multiplication of the total mass in the 
    object by the ratio of **R** and **RE**. This test indicates that 
    the scaling of the transform 
    is suitable. (???)
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
    Adjointness test. Creates random images 
    and sinograms to check whether forward and backprojection are indeed 
    adjoint to one another (by comparing the corresponding dual pairings). 
    This comparison is carried out for multiple experiments.
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
    assert (len(Error)<10), 'A large number of experiments for adjointness\
        turned out negative, number of errors: '+str(count)+' out of 100\
        tests adjointness-errors were bigger than '+str(eps) 

def test_fullangle():
    """
    Full-angle test. Tests and illustrates the impact of the fullangle 
    parameter, in particular showing artifacts resulting from the incorrect 
    use of the limited angle setting.
    """
    
    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test phantom
    dtype=float32
    N = 1200
    img_gpu = create_phantoms(queue, N,dtype)
    
    # relevant quantities
    Ns=int(0.3*N)
    shift=0
    (R, RE, Detector_width, image_width)=(5, 2, 6, 2)
    
    # angles cover only part of the angular range
    angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8
   
    # create two projecetionsettings, one with the correct "fullangle=False"
    # parameter for limited-angle situation, incorrectly using "fullangle=True"
    PScorrect = ProjectionSettings(queue, FANBEAM, img_gpu.shape, angles, Ns,
	                           image_width=image_width, R=R, RE=RE,
				   detector_width=Detector_width,
	                           detector_shift=shift, fullangle=False)
	
    PSincorrect = ProjectionSettings(queue, FANBEAM, img_gpu.shape, angles, Ns,
                                     image_width=image_width, R=R, RE=RE,
                                     detector_width=Detector_width, 
				     detector_shift=shift,fullangle=True)

    # show geometry of the problem
    PScorrect.show_geometry(np.pi/4, show=False)
     
    # forward and backprojection for the two settings
    sino_gpu_correct=forwardprojection(img_gpu, PScorrect)
    sino_gpu_incorrect=forwardprojection(img_gpu, PSincorrect)
    backprojected_gpu_correct=backprojection(sino_gpu_correct, PScorrect)
    backprojected_gpu_incorrect=backprojection(sino_gpu_correct, PSincorrect)

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
    
    # Computing controlnumbers to quantitatively verify correctness 
    evaluate_control_numbers(img, (N,N,Ns,len(angles),2),expected_result=2949.3738,
		precision=0.001,classified="img",name="original image")

    evaluate_control_numbers(sino_correct, (N,N,Ns,len(angles),2),
                expected_result=313.94908,precision=0.0001,classified="sino",
		name="sinogram with correct fullangle setting")

    evaluate_control_numbers(sino_incorrect, (N,N,Ns,len(angles),2),
		expected_result=313.94908,precision=0.0001,classified="sino",
		name="sinogram with incorrect fullangle setting")

    evaluate_control_numbers(backprojected_correct, (N,N,Ns,len(angles),2),
		expected_result=7044.6522, precision=0.0001,classified="img",
		name="backprojected image with correct fullangle setting")

    evaluate_control_numbers(backprojected_incorrect, (N,N,Ns,len(angles),2),
		expected_result=37105.70765,precision=0.001,classified="img",
		name="backprojected image with incorrect fullangle setting")


def test_midpointshift():
    """ 
    Shifted midpoint test.
    Tests and illustrates how the sinogram changes if the midpoint of an 
    images is shifted away from the center of rotation.
    """

    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create phantom for test
    dtype=float32
    N = 1200
    img_gpu = create_phantoms(queue, N,dtype)

    # relevant quantities
    (angles, R, RE, Detector_width, image_width, shift) = (360, 5, 3, 6, 2, 0)
    midpoint_shift=[0,0.5]
    Ns=int(0.5*N)

    # define projectionsetting
    PS = ProjectionSettings(queue, FANBEAM, img_gpu.shape, angles, Ns,
                            image_width=image_width, R=R, RE=RE,
                            detector_width=Detector_width, detector_shift=shift,
                            midpoint_shift=midpoint_shift,fullangle=True)
    
    # plot the geometry from various angles
    figure(0)
    for k in range(0,16):
        PS.show_geometry(k*np.pi/8, axes=subplot(4,4,k+1))

    # compute forward and backprojection
    sino_gpu=forwardprojection(img_gpu, PS)    
    backprojected_gpu=backprojection(sino_gpu, PS)
    
    img=img_gpu.get()
    sino=sino_gpu.get()
    backprojected=backprojected_gpu.get()

    # plot results
    figure(1)
    imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Sinogram with shifted midpoint")
    imshow(np.hstack([sino[:,:,0],sino[:,:,1]]), \
        cmap=cm.gray)
    figure(3)
    title("Backprojection with shifted midpoint")
    imshow(np.hstack([backprojected[:,:,0],backprojected[:,:,1]]),\
        cmap=cm.gray)
    show()
    
    # Computing controlnumbers to quantitatively verify correctness 
    evaluate_control_numbers(img, (N,N,Ns,angles,2),expected_result=2949.37386,
		precision=0.001,classified="img",name="original image")

    evaluate_control_numbers(sino, (N,N,Ns,angles,2),expected_result=699.5769,
		precision=0.001,classified="sino",name="sinogram")

    evaluate_control_numbers(backprojected, (N,N,Ns,angles,2),expected_result=14599.8994,
		precision=0.001,classified="img",name="backprojected image")


def test_landweber():
    """ 
    Landweber reconstruction test. Performs the Landweber iteration   
    to compute a reconstruction from a sinogram contained in 
    the walnut data set of [1]_, testing the implementation.

    .. [1] Keijo Hämäläinen and Lauri Harhanen and Aki Kallonen and 
           Antti Kujanpää and Esa Niemi and Samuli Siltanen.
           "Tomographic X-ray data of a walnut". 
           https://arxiv.org/abs/1502.04064
           **(???) Is this really the right reference?**
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
    ULW=landweber(walnut_gpu2new, PS, 20).get()

    # plot Landweber reconstruction
    figure(4)
    imshow(np.hstack([ULW[:,:,0],ULW[:,:,1]]),cmap=cm.gray)
    title("Landweber reconstruction")
    show()
    
    # Computing controlnumbers to quantitatively verify correctness
    [Nx,Ny]=img_shape
    evaluate_control_numbers(ULW, (Nx,Ny,number_detectors,len(angles),2),
                expected_result=0.782582206,
		precision=0.000001,classified="img",
		name=" Landweber-reconstruction")

    

def test_conjugate_gradients():
    """ 
    Conjugate gradients reconstruction test.
    Performs the conjugate gradients iteration   
    to compute a reconstruction from a sinogram contained in 
    the walnut data set of [1]_, testing the implementation.
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
    UCG=UCG.get()
    # plot results
    figure(4)
    imshow(np.hstack([UCG[:,:,0],UCG[:,:,1]]),cmap=cm.gray)
    title("Conjugate gradients reconstruction")
    show()

    # Compute control numbers to quantitatively verify correctness
    [Nx,Ny]=img_shape
    evaluate_control_numbers(UCG, (Nx,Ny,number_detectors,len(angles),2),
                expected_result=0.99194361,
		precision=0.000001,classified="img",
		name=" conjugate gradients reconstruction")


def test_total_variation():
    """ 
    Total variation reconstruction test.
    Performs the toolbox's total-variation-based approach   
    to compute a reconstruction from a sinogram contained in 
    the walnut data set of [1]_, testing the implementation.
    """
    print("Walnut total variation reconstruction test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    dtype=float32
    number_detectors=328
    (Detectorwidth, FOD, FDD, numberofangles) = (114.8, 110, 300, 120)
    angles = linspace(0,2*pi,121)[:-1] + pi/2
    img_shape=(328,328)

    # create projectionsetting
    PS = ProjectionSettings(queue, FANBEAM, img_shape=img_shape,
    	                    angles = angles, 
                            detector_width=Detectorwidth,
    	                    R=FDD, RE=FOD, n_detectors=number_detectors)

    # load and rescale sinogram
    sinonew=mpimg.imread(TESTWALNUTSINOGRAM)
    sinonew=np.array(sinonew,dtype=dtype)
    sinonew/=np.mean(sinonew)
    
    #Add noise 
    rng=np.random.default_rng(1)
    noise=rng.normal(0,1,sinonew.shape)*0.2    
    sinonew+=noise
    
    # perform total_variation reconstruction
    walnut_gpu2new=clarray.to_device(queue,require(sinonew,dtype,'C'))

    UTV=total_variation(walnut_gpu2new,PS,mu=1000,
                        number_iterations=4000,slice_thickness=0)
    
    sinoreprojected=forwardprojection(UTV,PS)
    
    UTV=UTV.get()
    
    # plot results
    figure(4)
    imshow(UTV,cmap=cm.gray)

    title("total variation reconstruction")
    figure(5)
    imshow(sinoreprojected.get(),cmap=cm.gray)
    title("reprojected sinogram of solution")
    show()
    
    # Computing controlnumbers to quantitatively verify correctness
    [Nx,Ny]=img_shape
    evaluate_control_numbers(UTV, (Nx,Ny,number_detectors,numberofangles,1),
                expected_result=0.93240546,
		precision=0.000001,classified="img",
		name="total-variation reconstruction")



def test_nonquadratic():
    """
    Nonquadratic image test. Tests and illustrates the projection 
    operator for non-quadratic images.
    """
    
    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create phantom and cut one side
    N = 1200
    img_gpu = create_phantoms(queue, N)
    N1=img_gpu.shape[0]
    N2=int(img_gpu.shape[0]*2/3.)
    img_gpu=cl.array.to_device(queue, img_gpu.get()[:,0:N2,:].copy())

    # geometric quantities
    (angles, R, RE, Detector_width, shift) = (360, 5, 3, 5, 0)
    image_width=None
    midpoint_shift=[0,0.]
    dtype=float32
    Ns=int(0.5*N1)

    # create projectionsetting
    PS = ProjectionSettings(queue, FANBEAM, img_gpu.shape, angles, Ns, 
        image_width=image_width,R=R, RE=RE, detector_width=Detector_width,
	detector_shift=shift, midpoint_shift=midpoint_shift, fullangle=True)

    # compute forward and backprojection
    sino_gpu=forwardprojection(img_gpu, PS)
    backprojected_gpu=backprojection(sino_gpu, PS)
    
    img=img_gpu.get()
    sino=sino_gpu.get()
    backprojected=backprojected_gpu.get()


    # show geometry of projectionsetting
    PS.show_geometry(1*np.pi/8, show=False)

    # plot results
    figure(1)
    title("original non square images")
    imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Fanbeam sinogram for non-square image")
    imshow(np.hstack([sino[:,:,0],sino[:,:,1]]), 
        cmap=cm.gray)
    
    figure(3)
    title("backprojection for non-square image")
    imshow(np.hstack([backprojected[:,:,0],backprojected[:,:,1]]),
        cmap=cm.gray)
    show()
    
    # Computing a controlnumbers to quantitatively verify correctness 
    evaluate_control_numbers(img, (N1,N2,Ns,angles,2),expected_result=999.4965,
		precision=0.001,classified="img",name="original image")

    evaluate_control_numbers(sino, (N1,N2,Ns,angles,2),expected_result=1547.6640,
		precision=0.001,classified="sino",name="sinogram")

    evaluate_control_numbers(backprojected, (N1,N2,Ns,angles,2),expected_result=16101.3542,
		precision=0.001,classified="img",name="backprojected image")



def test_extract_sparse_matrix():
    """
    Tests the create_sparse_matrix method to create a sparse matrix
    associated with the transform, and tests it by appling forward and
    backprojection by matrix multiplication.
    """

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
    PS = ProjectionSettings(queue, FANBEAM, img_shape=img.shape, 
               angles=angles, detector_width=400, R=752, RE=200, 
	       n_detectors=number_detectors)

     #Create corresponding sparse matrix
    sparsematrix=PS.create_sparse_matrix(dtype=dtype,order=order,outputfile=None)
    
    # Test image
    img=phantom(queue, Nx, dtype)
    img=img.get()
    img=img.reshape(Nx**2,order=order)
    
    # Compute forward and backprojection
    sino=sparsematrix*img
    backproj=sparsematrix.T*sino

    # reshape
    img=img.reshape(Nx,Nx,order=order)
    sino=sino.reshape(number_detectors,angles,order=order)
    backproj=backproj.reshape(Nx,Nx,order=order)


    figure(1)
    title("test image")
    imshow(img,cmap=cm.gray)


    figure(2)
    title("projection via sparse matrix")
    imshow(sino,cmap=cm.gray)
    
    figure(3)
    title("backprojection via sparse matrix")
    imshow(backproj,cmap=cm.gray)
    show()
    
    # Computing a controlnumbers to quantitatively verify correctness 
    evaluate_control_numbers(img, (Nx,Nx,number_detectors,angles,1),expected_result=7.1182017,
		precision=0.000001,classified="img",name="original image")

    evaluate_control_numbers(sino, (Nx,Nx,number_detectors,angles,1),expected_result=213.071169,
		precision=0.001,classified="sino",name="sinogram")

    evaluate_control_numbers(backproj, (Nx,Nx,number_detectors,angles,1),expected_result=2765.528,
		precision=0.01,classified="img",name="backprojected image")

    


# test
if __name__ == '__main__':
    pass
