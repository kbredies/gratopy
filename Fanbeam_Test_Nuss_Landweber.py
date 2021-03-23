from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from fanbeam_source import *


from scipy import misc				
Wallnut=misc.imread('Wallnut/Image.png')
Wallnut=Wallnut/np.mean(Wallnut)
Wallnut[np.where(Wallnut<=1.5)]=0
Wallnut=scipy.misc.imresize(Wallnut,[328,328])

dtype=float32


#	Wallnut=np.ones(Wallnut.shape)
Wallnut_gpu=clarray.to_device(queue,require(Wallnut,dtype,'F'))
number_detectors=328
Detectorwidth=114.8
FOD=110
FDD=300
numberofangles=120

geometry=[Detectorwidth,FDD,FOD,number_detectors]
#misc.imshow(Wallnut)
PS = projection_settings(queue,"fan",img_shape=Wallnut.shape, angles = numberofangles,detector_width=Detectorwidth, R=FDD, RE=FOD, n_detectors=number_detectors,data_type=dtype)
sino2_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=dtype, order='F')

fanbeam_richy_gpu(sino2_gpu,Wallnut_gpu,PS)	
#misc.imshow(sino2_gpu.get())
Wallnut_gpu2=clarray.to_device(queue,require(Wallnut,dtype,'F'))
fanbeam_richy_gpu_add(Wallnut_gpu2,sino2_gpu,PS)

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
#print 'asdf',np.max(sinonew)
sinonew[np.where(sinonew<2000)]=0
sinonew=np.array(sinonew,dtype=dtype)
sinonew/=np.mean(sinonew)
#sinonew=sino2_gpu.get()


#h=scipy.hstack([sino2_gpu.get()/np.mean(sino2_gpu.get()),sinonew])
#misc.imshow(h)
#misc.imsave('Comparison_sinogram.png',h)
#import pdb;pdb.set_trace()


number_detectors=328
Detectorwidth=114.8
FOD=110
FDD=300
numberofangles=120
geometry=[Detectorwidth,FDD,FOD,number_detectors]
#misc.imshow(Wallnut)
PS = projection_settings(queue,"fan",img_shape=(600,600), angles=numberofangles, detector_width=Detectorwidth, R=FDD, RE=FOD, n_detectors=number_detectors,data_type=dtype)

Wallnut_gpu2new=clarray.to_device(queue,require(sinonew,dtype,'F'))
ULW=Landweberiteration(Wallnut_gpu2new,PS,20)
misc.imshow(ULW.get())
#fanbeam_richy_gpu(Wallnut,U,f_struct_gpu_wallnut)
#misc.imshow(Wallnut.get())
#import pdb;pdb.set_trace()

sinonew=[sinonew.T]
import pdb; pdb.set_trace()
