from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from fanbeam_source import *
from PIL import Image

number_detectors=328
Detectorwidth=114.8
FOD=110
FDD=300
numberofangles=120

from scipy import misc
Wallnut=imread('Wallnut/Image.png')

Wallnut=Wallnut/np.mean(Wallnut)
Wallnut[np.where(Wallnut<=1.5)]=0
Wallnut=scipy.misc.imresize(Wallnut,[328,328])

data_type=float
#	Wallnut=np.ones(Wallnut.shape)
Wallnut_gpu=clarray.to_device(queue,require(Wallnut,data_type,'F'))

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

PS = projection_settings("fan",img_shape=Wallnut.shape, angles = numberofangles,detector_width=Detectorwidth, R=FDD, RE=FOD, n_detectors=number_detectors,data_type=data_type)
sino2_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=data_type, order='F')

fanbeam_richy_gpu(sino2_gpu,Wallnut_gpu,PS)	
#misc.imshow(sino2_gpu.get())
Wallnut_gpu2=clarray.to_device(queue,require(Wallnut,data_type,'F'))
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
