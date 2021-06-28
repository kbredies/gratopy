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
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    #Create testimage
    Nx=225
    dtype=float32
    img=create_phantoms(queue,Nx,dtype)
    original=img.get()
    
    #Define setting for projection
    number_detectors = 600
    angles=220
    PS = ProjectionSettings(queue, FANBEAM, img_shape=img.shape, angles=angles,
                            detector_width=400, R=752, RE=200, n_detectors=number_detectors)

    
    sino_gpu = clarray.zeros(queue, (PS.n_detectors,PS.n_angles,2), dtype=dtype, order='F')

    #Test speed of implementation for forward projection
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
    print ('Average time required Backprojection',(time.perf_counter()-a)/100)


    #plott results
    figure(1)
    imshow(hstack([original[:,:,0],original[:,:,1]]), cmap=cm.gray)
    title('original image')
    figure(2)
    imshow(hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
    title('Fanbeam transformed image')
    figure(3)
    imshow(hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    title('Backprojected image')
    show()

def test_weighting():
    ###Weighting:  Checks wether the mass of an image is correctly transported into the mass inside a projection
    print("Weighting;")
    
    #Create PyopenCL context 
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    #Execute for different number of detectors, to ensure consideration is resolution independent.
    for number_detectors in [50,100,200,400,800,1600]:

        #Consider full image
        Nx=400
        img=np.ones([Nx,Nx])
        angles=720

        #relevant quantities for scaling
        rescaling=1/40.*sqrt(2)
        detector_width=400*rescaling
        R=1200.*rescaling
        RE=200.*rescaling
        image_width=40.*rescaling

        #create projectionsetting
        PS = ProjectionSettings(queue, FANBEAM, img_shape=img.shape, angles=angles,
                                detector_width=detector_width, R=R,RE= RE,
                                n_detectors=number_detectors,image_width=image_width)
        
        #forward and backprojection
        img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
        sino_gpu= forwardprojection(img_gpu, PS)
	
	#Compute mass inside object and on the detector
        mass_in_image=np.sum(img_gpu.get())*PS.delta_x**2
        mass_in_projcetion=np.sum(sino_gpu.get())*(PS.delta_ratio*PS.delta_x)/angles

        print("Mass in original image",mass_in_image, "mass in projection",mass_in_projcetion,"Ratio",mass_in_projcetion/mass_in_image,"Ratio should be "+str(R/RE))
        assert( abs(mass_in_projcetion/mass_in_image/(R/RE)-1)<0.1), "Due to the fan effect the object is enlarged on the detector, roughly by the ratio of the distances, but this was not satisfied in this test."

def test_adjointness():
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    ###Adjointness
    print("Adjointness:")

    number_detectors=230
    img=np.zeros([400,400])
    angles=390
    midpoint_shift=[100,100]

    PS = ProjectionSettings(queue, FANBEAM, img.shape, angles, n_detectors=number_detectors,
                            detector_width=83, detector_shift=0.0, midpoint_shift=[0,0],
                            R=900, RE=300, image_width=None, fullangle=True)

    delta_x=PS.delta_x
    delta_s_ratio=PS.delta_ratio


    Fehler=[]
    count=0
    eps=0.00001

    img2_gpu = clarray.zeros(queue, PS.img_shape, dtype=float32, order='F')
    sino2_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=float32, order='F')
    for i in range(100):
        img1_gpu = clarray.to_device(queue, require(np.random.random(PS.img_shape), float32, 'F'))
        sino1_gpu = clarray.to_device(queue, require(np.random.random(PS.sinogram_shape), float32, 'F'))
        forwardprojection(img1_gpu, PS, sino=sino2_gpu)
        backprojection(sino1_gpu, PS, img=img2_gpu)
        
        sino1=sino1_gpu.get().flatten()
        sino2=sino2_gpu.get().flatten()
        img1=img1_gpu.get().flatten()
        img2=img2_gpu.get().flatten()

        a=np.dot(img1,img2)*delta_x**2
        b=np.dot(sino1,sino2)*(2*np.pi)/angles*(delta_s_ratio*delta_x)
        if abs(a-b)/min(abs(a),abs(b))>eps:
            print (a,b,a/b)
            count+=1
            Fehler.append((a,b))

    print ('Number of Errors: '+str(count)+' out of 100 tests adjointness-errors were bigger than '+str(eps))

def test_fullangle():
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    ########
    N = 1200
    img = create_phantoms(queue, N)

    dtype=float32
    Ns=int(0.3*img.shape[0])
    shift=0

    angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8
    (R, RE, Detector_width, image_width)=(5, 2, 6, 2)
    PScorrect = ProjectionSettings(queue, FANBEAM, img.shape, angles, Ns,
                                   image_width=image_width, R=R, RE=RE,
                                   detector_width=Detector_width, detector_shift=shift,
                                   fullangle=False)
    PSincorrect = ProjectionSettings(queue, FANBEAM, img.shape, angles, Ns,
                                     image_width=image_width, R=R, RE=RE,
                                     detector_width=Detector_width, detector_shift=shift,
                                     fullangle=True)

    PScorrect.show_geometry(np.pi/4, show=False)
    img_gpu = img

    sino_gpu_correct=forwardprojection(img_gpu, PScorrect)
    sino_gpu_incorrect=forwardprojection(img_gpu, PSincorrect)
    backprojected_correct=backprojection(sino_gpu_correct, PScorrect)
    backprojected_incorrect=backprojection(sino_gpu_correct, PSincorrect)

    figure(1)
    imshow(np.hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Sinograms with vs without fullangle")
    imshow(np.vstack([np.hstack([sino_gpu_correct.get()[:,:,0],sino_gpu_correct.get()[:,:,1]]),np.hstack([sino_gpu_incorrect.get()[:,:,0],sino_gpu_incorrect.get()[:,:,1]])]), cmap=cm.gray)
    figure(3)
    title("Backprojection with vs without fullangle")
    imshow(np.vstack([np.hstack([backprojected_correct.get()[:,:,0],backprojected_correct.get()[:,:,1]]),np.hstack([backprojected_incorrect.get()[:,:,0],backprojected_incorrect.get()[:,:,1]])]), cmap=cm.gray)
    show()	 

def test_midpointshift():
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    ##Midpointshift
    N = 1200
    img = create_phantoms(queue, N)

    dtype=float32

    (angles, R, RE, Detector_width, image_width, shift) = (360, 5, 3, 6, 2, 0)
    midpoint_shift=[0,0.5]

    Ns=int(0.5*img.shape[0])

    PS = ProjectionSettings(queue, FANBEAM, img.shape, angles, Ns,
                            image_width=image_width, R=R, RE=RE,
                            detector_width=Detector_width, detector_shift=shift,
                            midpoint_shift=midpoint_shift,fullangle=True)

    figure(0)
    for k in range(0,16):
        PS.show_geometry(k*np.pi/8, axes=subplot(4,4,k+1))

    img_gpu = img
    sino_gpu=forwardprojection(img_gpu, PS)    
    backprojected=backprojection(sino_gpu, PS)

    figure(1)
    imshow(np.hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Sinogram with shifted midpoint")
    imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
    figure(3)
    title("Backprojection with shifted midpoint")

    imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]), cmap=cm.gray)
    show()			

def test_landweber():
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    ###Nuss Landweber
    print("Walnut Landweber reconstruction test")
    walnut=mpimg.imread(TESTWALNUT)
    walnut=walnut/np.mean(walnut)
    walnut[np.where(walnut<=1.5)]=0
    #walnut=scipy.misc.imresize(walnut,[328,328])
    walnut=np.array(Image.fromarray(walnut).resize([328,328]))

    dtype=float

    number_detectors=328
    (Detectorwidth, FOD, FDD, numberofangles) = (114.8, 110, 300, 120)
    geometry=[Detectorwidth,FDD,FOD,number_detectors]

    PS = ProjectionSettings(queue, FANBEAM, img_shape=walnut.shape,
                            angles = numberofangles, detector_width=Detectorwidth,
                            R=FDD, RE=FOD, n_detectors=number_detectors)

    walnut_gpu=clarray.to_device(queue,require(walnut,dtype,'F'))

    sino=forwardprojection(walnut_gpu, PS)	
    walnutbp=backprojection(sino, PS)

    figure(1)
    imshow(walnut, cmap=cm.gray)
    title('original walnut')
    figure(2)
    imshow(sino.get(), cmap=cm.gray)
    title('Fanbeam transformed image')
    figure(3)
    imshow(walnutbp.get(), cmap=cm.gray)
    title('Backprojected image')

    sinonew=mpimg.imread(TESTWALNUTSINOGRAM)
    #sinonew[np.where(sinonew<2000)]=0
    #sinonew=np.array(sinonew,dtype=dtype)
    sinonew/=np.mean(sinonew)

    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110, 300)
    angles = linspace(0,2*pi,121)[:-1] + pi/2
    geometry=[Detectorwidth,FDD,FOD,number_detectors]
    
    PS = ProjectionSettings(queue, FANBEAM, img_shape=(600,600), angles=angles,
                            detector_width=Detectorwidth, R=FDD, RE=FOD,
                            n_detectors=number_detectors)

    walnut_gpu2new=clarray.to_device(queue,require(sinonew,dtype,'F'))
    ULW=landweber(walnut_gpu2new, PS, 20)

    figure(4)
    imshow(ULW.get(),cmap=cm.gray)
    title("Landweber reconstruction")
    show()

    sinonew=[sinonew.T]

def test_conjugate_gradients():
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    print("Walnut conjugated_gradients reconstruction test")
    walnut=mpimg.imread(TESTWALNUT)
    walnut=walnut/np.mean(walnut)
    walnut[np.where(walnut<=1.5)]=0
    #walnut=scipy.misc.imresize(walnut,[328,328])
    walnut=np.array(Image.fromarray(walnut).resize([328,328]))

    dtype=float

    number_detectors=328
    (Detectorwidth, FOD, FDD, numberofangles) = (114.8, 110, 300, 120)
    geometry=[Detectorwidth,FDD,FOD,number_detectors]

    PS = ProjectionSettings(queue, FANBEAM, img_shape=walnut.shape,
                            angles = numberofangles, detector_width=Detectorwidth,
                            R=FDD, RE=FOD, n_detectors=number_detectors)

    walnut_gpu=clarray.to_device(queue,require(walnut,dtype,'F'))

    sino=forwardprojection(walnut_gpu, PS)	
    walnutbp=backprojection(sino, PS)

    figure(1)
    imshow(walnut, cmap=cm.gray)
    title('original walnut')
    figure(2)
    imshow(sino.get(), cmap=cm.gray)
    title('Fanbeam transformed image')
    figure(3)
    imshow(walnutbp.get(), cmap=cm.gray)
    title('Backprojected image')

    sinonew=mpimg.imread(TESTWALNUTSINOGRAM)
    #sinonew[np.where(sinonew<2000)]=0
    #sinonew=np.array(sinonew,dtype=dtype)
    sinonew/=np.mean(sinonew)

    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110, 300)
    angles = linspace(0,2*pi,121)[:-1] + pi/2
    geometry=[Detectorwidth,FDD,FOD,number_detectors]
    
    PS = ProjectionSettings(queue, FANBEAM, img_shape=(600,600), angles=angles,
                            detector_width=Detectorwidth, R=FDD, RE=FOD,
                            n_detectors=number_detectors)

    walnut_gpu2new=clarray.to_device(queue,require(sinonew,dtype,'C'))
    UCG=conjugate_gradients(walnut_gpu2new, PS, 0.1,number_iterations=100)

    figure(4)
    imshow(UCG.get(),cmap=cm.gray)
    title("Conjugate gradients reconstruction")
    show()

    sinonew=[sinonew.T]

def test_total_variation():
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    print("Walnut total variation reconstruction test")
    walnut=mpimg.imread(TESTWALNUT)
    walnut=walnut/np.mean(walnut)
    walnut[np.where(walnut<=1.5)]=0
    #walnut=scipy.misc.imresize(walnut,[328,328])
    walnut=np.array(Image.fromarray(walnut).resize([328,328]))

    dtype=float32

    number_detectors=328
    (Detectorwidth, FOD, FDD, numberofangles) = (114.8, 110, 300, 120)

    PS = ProjectionSettings(queue, FANBEAM, img_shape=walnut.shape,
    	                    angles = numberofangles, detector_width=Detectorwidth,
    	                    R=FDD, RE=FOD, n_detectors=number_detectors)

    sinonew=mpimg.imread(TESTWALNUTSINOGRAM)
    #sinonew[np.where(sinonew<2000)]=0
    sinonew=np.array(sinonew,dtype=dtype)
    sinonew/=np.mean(sinonew)

    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110, 300)

    geometry=[Detectorwidth,FDD,FOD,number_detectors]


    walnut_gpu2new=clarray.to_device(queue,require(sinonew,dtype,'C'))
    UTV=total_variation_reconstruction(walnut_gpu2new, PS,mu=10000,number_iterations=3000,z_distance=1)
    sinoreprojected=forwardprojection(UTV,PS)
    figure(4)
    imshow(UTV.get(),cmap=cm.gray)
	
    title("total variation reconstruction")
    figure(5)
    imshow(sinoreprojected.get(),cmap=cm.gray)
    title("reprojected sinogram")

    show()

    sinonew=[sinonew.T]




def test_nonquadratic():
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    ##Non-quadratic images
    N = 1200
    img = create_phantoms(queue, N)

    N1=img.shape[0]
    N2=int(img.shape[0]*2/3.)
    img=cl.array.to_device(queue, img.get()[:,0:N2,:].copy())

    (angles, R, RE, Detector_width, shift) = (360, 5, 3, 5, 0)
    image_width=None
    midpoint_shift=[0,0.]
    dtype=float32
    Ns=int(0.5*img.shape[0])

    PS = ProjectionSettings(queue, FANBEAM, img.shape, angles, Ns, image_width=image_width,
                            R=R, RE=RE, detector_width=Detector_width, detector_shift=shift,
                            midpoint_shift=midpoint_shift, fullangle=True)

    img_gpu = img
    sino_gpu=forwardprojection(img_gpu, PS)
    backprojected=backprojection(sino_gpu, PS)

    PS.show_geometry(1*np.pi/8, show=False)

    figure(1)
    title("original non square images")
    imshow(np.hstack([img.get()[:,:,0],img.get()[:,:,1]]), cmap=cm.gray)
    figure(2)
    title("Fanbeam sinogram for non-square image")
    imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
    figure(3)
    title("backprojection for non-square image")
    imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]), cmap=cm.gray)
    show()			


# test
if __name__ == '__main__':
    pass
