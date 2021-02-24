
from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc




###########
## GPU code

class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build()
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = Program(ctx, open("fanbeam").read())


def fanbeam_richy_gpu(sino, img, f_struct, wait_for=None):
    shape,sinogram_shape,ofs_buf,Geometryinfo = f_struct 
    return prg.fanbeam_richy(sino.queue, sino.shape, None,
                       sino.data, img.data, ofs_buf,
                       float32(Geometryinfo[0]), float32(Geometryinfo[1]), float32(Geometryinfo[7]),
                       wait_for=wait_for)
                       

def fanbeam_richy_gpu_add( img,sino, f_struct, wait_for=None):
    shape,sinogram_shape,ofs_buf,Geometryinfo = f_struct 
    return prg.fanbeam_richy_add(img.queue, img.shape, None,
                       img.data,sino.data, ofs_buf,
                       float32(Geometryinfo[7]),int32(Geometryinfo[2]),int32(sinogram_shape[1]),
                       wait_for=wait_for)
                       


def fanbeam_struct_richy_gpu( shape, angles, detector_width,
                   source_detector_dist, source_origin_dist,
                   n_detectors=None, detector_shift = 0.0,image_width=None):
	
	detector_width=float(detector_width)
	source_detector_dist=float(source_detector_dist)
	source_origin_dist=float(source_origin_dist)
	
	
	
	
	# choose equidistant angles in (0,2pi] if no specific angles are given.
	if isscalar(angles):
		angles = linspace(0,2*pi,angles+1)[:-1] + pi

	
	
	image_pixels = max(shape[0],shape[1])
	#set number of pixels same as image_resolution if not specified
	if n_detectors is None:
		nd = image_pixels
	else:
		nd = n_detectors

	assert isinstance(nd, int), "Number of detectors must be integer"

	#compute  midpoints vor orientation
	midpoint_domain = array([shape[0]-1, shape[1]-1])/2.0
	midpoint_detectors = (nd-1.0)/2.0

	
	#Ensure that indeed detector on the opposite side of the source
	#assert source_detector_dist>source_origin_dist, 'Origin not between detector and source'
	
	#Save relevant geometric inforrmation
	sinogram_shape = (nd, len(angles))
	Geometryinfo=zeros(8,dtype=float32,order='F')
	Geometryinfo[0]=(shape[0]-1)/2.
	Geometryinfo[1]=(shape[1]-1)/2.
	Geometryinfo[2]=sinogram_shape[0]
	Geometryinfo[3]=detector_width
	Geometryinfo[4]=source_detector_dist
	Geometryinfo[5]=source_origin_dist
	
	#to include detector shift, the midpoint changes accordingly (with detectorshift=detectorwidth/2 would correspond to the closest point to the source is the end of the detector
	Geometryinfo[7]=midpoint_detectors+detector_shift*nd/detector_width

	#In case no image_width is predetermined, image_width is chosen in a way that the (square) image is always contained inside the fan between source and detector
	if image_width==None:
		dd=(0.5*detector_width-abs(detector_shift))/source_detector_dist
		image_width = sqrt(2)*dd*source_origin_dist/sqrt(1+dd**2) # Projection to compute distance via projectionvector (1,dd) after normalization
	
	assert image_width<source_origin_dist , " the image is encloses the source"
		
	# adjust distances to pixel units, i.e. 1 unit corresponds to the length of one image pixel
	source_detector_dist *= image_pixels/float(image_width)
	source_origin_dist *= image_pixels/float(image_width)
	detector_width *= image_pixels/float(image_width)

	#import pdb; pdb.set_trace()
	# offset function parameters
	thetaX = -cos(angles)
	thetaY = sin(angles)
	
	#Direction vector of detector direction normed to the length of a single detector pixel
	XD=thetaX*detector_width/nd
	YD=thetaY*detector_width/nd

	#Direction vector leading to from source to origin (with proper length)
	Qx=-thetaY*source_origin_dist
	Qy=thetaX*source_origin_dist

	#Directionvector from origin to the detector center
	Dx0= thetaY*(source_detector_dist-source_origin_dist)
	Dy0= -thetaX*(source_detector_dist-source_origin_dist)

	Geometryinfo[6]=image_width			
	
	#Save relevant information
	ofs = zeros((8, len(angles)), dtype=float32, order='F')
	ofs[0,:] = XD; ofs[1,:] = YD
	ofs[2,:]=Qx; ofs[3,:]=Qy
	ofs[4,:]=Dx0; ofs[5]=Dy0
	
	#write to Buffer
	ofs_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, len(ofs.data))
	cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
	return (shape,sinogram_shape,ofs_buf,Geometryinfo)


def show_geometry(angle,f_struct):
	
	#Extract relevant geometric information
	sino_size=f_struct[3][2]
	detector_width=f_struct[3][3]
	source_detector_dist=f_struct[3][4]
	source_origin_dist=f_struct[3][5]
	image_width=f_struct[3][6]
	midpoint_det=f_struct[3][7]
	figure(0)
	#Plot all relevant sizes
	plot([-source_origin_dist, source_detector_dist-source_origin_dist],[0 ,0],"c")
	plot([source_detector_dist-source_origin_dist, source_detector_dist-source_origin_dist], [detector_width*midpoint_det/sino_size , detector_width*(midpoint_det/sino_size-1)],"k")
	
	plot([-source_origin_dist, source_detector_dist-source_origin_dist],[0,detector_width*midpoint_det/sino_size],"g")
	plot([-source_origin_dist, source_detector_dist-source_origin_dist],[0,detector_width*(midpoint_det/sino_size-1)],"g")

	x1=(cos(angle)+sin(angle))*image_width*0.5; x2=(cos(angle)-sin(angle))*image_width*0.5;
	
	plot([x1,x2,-x1,-x2,x1],[x2,-x1,-x2,x1,x2],"b")
	
	draw_circle=matplotlib.patches.Circle((0, 0), image_width/sqrt(2), color='r')
	
	gcf().gca().add_artist(draw_circle)	
	draw()

def Landweberiteration(sinogram,r_struct,number_iterations=100):
	sino2=clarray.to_device(queue, require(sinogram.get(), float32, 'F'))
	
	U=clarray.zeros(queue,r_struct[0],dtype=float32, order='F')
	Unew=clarray.zeros(queue,r_struct[0],dtype=float32, order='F')
	
	norm=radon_normest(queue,r_struct)
	#norm=10000
	w=float32(0.5/norm**2)
	
	for i in range(number_iterations):
	
		#import pdb; pdb.set_trace()
		sino2.events.append(fanbeam_richy_gpu(sino2,Unew,r_struct,wait_for=U.events+Unew.events))
		sino2=sino2-sinogram
		cl.enqueue_barrier(queue)
		U.events.append(fanbeam_richy_gpu_add(U,sino2,r_struct,wait_for=sino2.events))
		Unew=Unew-w*U	
		print (i,np.max(Unew.get()), np.linalg.norm(sino2.get()))
	return Unew

