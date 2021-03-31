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
	
img=np.zeros([225,225,2])
A =  imread('Testfiles/brain.png')[:,:,0]
A/=np.max(A)
B=imread('Testfiles/Ente.jpeg')[:,:,0]
B=np.array(B,dtype=float)
B/=np.max(B)

img[:,:,0]=A
img[:,:,1] = B

img=np.array(img,dtype=float)
number_detectors = 600
angles=220

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


PS = projection_settings(queue,"fan",img_shape=img.shape,angles= angles,  detector_width=400, R=752, RE=200, n_detectors=number_detectors,data_type='single')

img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
sino_gpu = clarray.zeros(queue, (PS.n_detectors,PS.n_angles,2), dtype=float32, order='F')

a=time.clock()
for i in range(100):
	forwardprojection(sino_gpu,img_gpu,PS,wait_for=sino_gpu.events)

print ('Time Required Forward',(time.clock()-a)/100)
sino=sino_gpu.get()

numberofangles=180
angles = linspace(0,2*pi,numberofangles+1)[:-1] + pi

a=time.clock()
for i in range(100):
	backprojection(img_gpu,sino_gpu,PS,wait_for=img_gpu.events)
print ('Time Required Backprojection',(time.clock()-a)/100)
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






## Weighting
print("")
print("Weighting;")
for number_detectors in [50,100,200,400,800,1600]:
	Nx=400
	img=np.ones([Nx,Nx])
	angles=720

	rescaling=1/40.*sqrt(2)
	detector_width=400*rescaling
	R=1200*rescaling
	RE=200*rescaling
	image_width=40*rescaling

	PS = projection_settings(queue,"fan",img_shape=img.shape, angles= angles,  detector_width=detector_width, R=R,RE= RE, n_detectors=number_detectors,image_width=image_width,data_type='single')
	delta_x=PS.delta_x
	delta_xi_ratio=PS.delta_ratio
	

	img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
	sino_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=float32, order='F')
	forwardprojection(sino_gpu,img_gpu,PS)

	a=np.sum(img_gpu.get())*delta_x**2
	b=np.sum(sino_gpu.get())*(delta_xi_ratio*delta_x)/angles
	print("Mass in original image",a, "mass in projection",b,"Ratio",b/a,"Ratio should be 6")




###Adjointness	
number_detectors=230
img=np.zeros([400,400])
angles=390
midpoint_shift=[100,100]

PS=projection_settings(queue,"fan",img.shape, angles, n_detectors=number_detectors, 
				detector_width=83,detector_shift = 0.0, midpoint_shift=[0,0],
				R=900, RE=300,
				image_width=None, fullangle=True,data_type='single')

delta_x=PS.delta_x
delta_xi_ratio=PS.delta_ratio

print("")
print("Adjointness:")

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
	b=np.dot(sino1,sino2)*(2*np.pi)/angles *(delta_xi_ratio*delta_x)
	if abs(a-b)/min(abs(a),abs(b))>eps:
		print (a,b,a/b)
		count+=1
		Fehler.append((a,b))
print ('Number of Errors: '+str(count)+' out of 100 tests adjointness-errors were bigger than '+str(eps))

def rgb2gray(rgb):
    if len(rgb.shape)>2:
    	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
    	gray=rgb
    return gray
    
    
    
    
    
###Fullangle    
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

s_axis=[]
Resulting_Sino=[]

angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8

img=255-img
my_dtype=float32

p=2

Ns=int(0.3*img.shape[0])
shift=0

PScorrect=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=p,detector_shift=shift,fullangle=False,data_type=my_dtype)
PSincorrect=projection_settings(queue,"parallel",img.shape,angles,Ns,detector_width=p,detector_shift=shift,fullangle=True,data_type=my_dtype)

img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))

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
	
########		
angles=np.linspace(0,np.pi*3/4.,180)+np.pi/8
R=5
RE=2
Detector_width=6
image_width=2
PScorrect=projection_settings(queue,"fan",img.shape,angles,Ns,image_width=image_width,R=R,RE=RE,detector_width=Detector_width,detector_shift=shift,fullangle=False,data_type=my_dtype)
PSincorrect=projection_settings(queue,"fan",img.shape,angles,Ns,image_width=image_width,R=R,RE=RE,detector_width=Detector_width,detector_shift=shift,fullangle=True,data_type=my_dtype)

PScorrect.show_geometry(np.pi/4)
img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))

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
    
    

	
##Midpointshift
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
my_dtype=float32

	

angles=360
R=5
RE=3
Detector_width=6
image_width=2
shift=0
midpoint_shif=[0,0.5]

Ns=int(0.5*img.shape[0])

PS=projection_settings(queue,"fan",img.shape,angles,Ns,image_width=image_width,R=R,RE=RE,detector_width=Detector_width,detector_shift=shift,midpoint_shift=midpoint_shif,fullangle=True,data_type=my_dtype)

for k in range(0,16):
	PS.show_geometry(k*np.pi/8)

img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))

sino_gpu=forwardprojection(None,img_gpu,PS)


figure(1)
imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)

backprojected=backprojection(None,sino_gpu,PS)
figure(2)
title("Sinogram with shifted midpoint")
imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
figure(3)
title("Backprojection with shifted midpoint")

imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]), cmap=cm.gray)
show()			

###Nuss Landweber
print("")
print("Nuss Test")
Wallnut=mpimg.imread('Wallnut/Image.png')
Wallnut=Wallnut/np.mean(Wallnut)
Wallnut[np.where(Wallnut<=1.5)]=0
#Wallnut=scipy.misc.imresize(Wallnut,[328,328])
Wallnut=np.array(Image.fromarray(Wallnut).resize([328,328]))

dtype=float

number_detectors=328
Detectorwidth=114.8
FOD=110
FDD=300
numberofangles=120

geometry=[Detectorwidth,FDD,FOD,number_detectors]

PS = projection_settings(queue,"fan",img_shape=Wallnut.shape, angles = numberofangles,detector_width=Detectorwidth, R=FDD, RE=FOD, n_detectors=number_detectors,data_type=dtype)

Wallnut_gpu=clarray.to_device(queue,require(Wallnut,dtype,'F'))

sino=forwardprojection(None,Wallnut_gpu,PS)	

Wallnutbp=backprojection(None,sino,PS)

figure(1)
imshow(Wallnut, cmap=cm.gray)
title('original Wallnut')
figure(2)
imshow(sino.get(), cmap=cm.gray)
title('Fanbeam transformed image')
figure(3)
imshow(Wallnutbp.get(), cmap=cm.gray)
title('Backprojected image')
show()

sinonew=mpimg.imread('Wallnut/Sinogram.png')
#sinonew[np.where(sinonew<2000)]=0
#sinonew=np.array(sinonew,dtype=dtype)
sinonew/=np.mean(sinonew)

number_detectors=328
Detectorwidth=114.8
FOD=110
FDD=300
numberofangles=120
geometry=[Detectorwidth,FDD,FOD,number_detectors]
PS = projection_settings(queue,"fan",img_shape=(600,600), angles=numberofangles, detector_width=Detectorwidth, R=FDD, RE=FOD, n_detectors=number_detectors,data_type=dtype)

Wallnut_gpu2new=clarray.to_device(queue,require(sinonew,dtype,'F'))
ULW=Landweberiteration(Wallnut_gpu2new,PS,20)
imshow(ULW.get(),cmap=cm.gray)
title("Landweber reconstruction")
show()

sinonew=[sinonew.T]


##Non-quadratic images

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

N1=img.shape[0]
N2=int(img.shape[0]*2/3.)

img=np.array(img[:,0:N2,:])

img=255-img

angles=360
R=5
RE=3
Detector_width=5
image_width=None
shift=0
midpoint_shif=[0,0.]
my_dtype=float32
Ns=int(0.5*img.shape[0])

PS=projection_settings(queue,"fan",img.shape,angles,Ns,image_width=image_width,R=R,RE=RE,detector_width=Detector_width,detector_shift=shift,midpoint_shift=midpoint_shif,fullangle=True,data_type=my_dtype)

img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))

sino_gpu=forwardprojection(None,img_gpu,PS)
backprojected=backprojection(None,sino_gpu,PS)


PS.show_geometry(1*np.pi/8)

figure(1)
title("original non square images")
imshow(np.hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
figure(2)
title("Fanbeam sinogram for non-square image")
imshow(np.hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
figure(3)
title("backprojection for non-square image")
imshow(np.hstack([backprojected.get()[:,:,0],backprojected.get()[:,:,1]]), cmap=cm.gray)
show()			

PS=projection_settings(queue,"parallel",img.shape,angles,Ns,data_type=my_dtype)
img_gpu = clarray.to_device(queue, require(img, my_dtype, 'F'))
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

