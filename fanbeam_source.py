
from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray




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

raw_fanbeam_code=open("fanbeam").read()
raw_radon_code=open("Radontransform").read()
adjusted_code=raw_fanbeam_code.replace("my_variable_type","float")\
	+raw_fanbeam_code.replace("my_variable_type","double")\
	+raw_radon_code.replace("my_variable_type","float")\
	+raw_radon_code.replace("my_variable_type","double")

prg = Program(ctx,adjusted_code)


def forwardprojection(sino, img, projection_settings, wait_for=None):
	
	if projection_settings.geometry in ["FAN","FANBEAM"]:
		return fanbeam_richy_gpu(sino, img, projection_settings, wait_for=None)
	if projection_settings.geometry in ["RADON","PARALLEL"]:
		return radon(sino, img, projection_settings, wait_for=None)
	
def backprojection(img, sino, projection_settings, wait_for=None):
	
	if projection_settings.geometry in ["FAN","FANBEAM"]:
		return fanbeam_richy_gpu_add(img, sino, projection_settings, wait_for=None)
	if projection_settings.geometry in ["RADON","PARALLEL"]:
		return radon_ad(img, sino, projection_settings, wait_for=None)




						
def radon(sino, img, projection_settings, wait_for=None):
	(ofs_buf, shape, sinogram_shape,Geometryinformation) = projection_settings.struct


	assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
			+str(sino.dtype)+" and "+str(img.dtype))
			
	assert(sino.dtype==projection_settings.dtype),\
			("sinogram and projection_settings do not share common data type: "\
			+str(sino.dtype)+" and "+str(projection_settings.dtype)+\
			"this might be remidies by choosing the parameter data_type to\
			 be equal to float32 or float (double precission) in the \
			 initialization of the parameter_settings" )

	
	if projection_settings.data_type=="SINGLE":

		return prg.radon_float(sino.queue, sino.shape, None,
					sino.data, img.data, ofs_buf,Geometryinformation.data,
					wait_for=wait_for)
	if projection_settings.data_type=="DOUBLE":

		return prg.radon_double(sino.queue, sino.shape, None,
					sino.data, img.data, ofs_buf,
					Geometryinformation.data,
					wait_for=wait_for)


"""Starts the GPU backprojection code 
input	sino ... A pyopencl.array in which the sinogram for transformation will be saved.
		img ...	 A pyopencl.array in which the result for the adjoint radontransform is contained
		r_struct. ..	The r_struct corresponding the given topology, see radon_struct
output	An event for the queue to compute the adjoint radon transform of image saved into img w.r.t. r_struct geometry
"""
def radon_ad(img, sino, projection_settings, wait_for=None):
	(ofs_buf, shape, sinogram_shape,Geometryinformation) = projection_settings.struct
	
	assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
			+str(sino.dtype)+" and "+str(img.dtype))
			
	assert(sino.dtype==projection_settings.dtype),\
			("sinogram and projection_settings do not share common data type: "\
			+str(sino.dtype)+" and "+str(projection_settings.dtype)+\
			"this might be remidies by choosing the parameter data_type to\
			 be equal to float32 or float (double precission) in the \
			 initialization of the parameter_settings" )

	
	if projection_settings.data_type=="SINGLE":
		return prg.radon_ad_float(img.queue, img.shape, None,
						img.data, sino.data, ofs_buf,
						Geometryinformation.data,wait_for=wait_for)

	if projection_settings.data_type=="DOUBLE":
		return prg.radon_ad_double(img.queue, img.shape, None,
						img.data, sino.data, ofs_buf,
						Geometryinformation.data,wait_for=wait_for)



"""Creates the structure of radon geometry required for radontransform and its adjoint
Input:
		queue ... a queue object corresponding to a context in pyopencl
		shape ... the shape of the object (image) in pixels
		angles ... a list of angles considered
		n_detectors ... Number of detectors, i.e. resolution of the sinogram
		detector_with ... Width of one detector relatively to a pixel in the image (default 1.0)
		detector_shift ... global shift of ofsets (default 0)
Output:
		ofs_buf ... a buffer object with 4 x number of angles entries corresponding to the cos and sin divided by the detectorwidth, also offset depending on the angle and the inverse of the cos values
		shape ... The same shape as in the input.
		sinogram_shape ... The sinogram_shape is a list with first the number of detectors, then number of angles.
"""
def radon_struct(queue, shape, angles, n_detectors=None,
			 detector_width=1.0, image_width=2,detector_shift=0.0,data_type=float32,fullangle=True):
	if isscalar(angles):	
		angles = linspace(0,pi,angles+1)[:-1]
	if n_detectors is None:
		nd = int(ceil(hypot(shape[0],shape[1])))
	else:
		nd = n_detectors
	midpoint_domain = array([shape[0]-1, shape[1]-1])/2.0
	midpoint_detectors = (nd-1.0)/2.0

	X = cos(angles)/detector_width
	Y = sin(angles)/detector_width
	Xinv = 1.0/X

	# set near vertical lines to vertical
	mask = abs(Xinv) > 10*nd
	X[mask] = 0
	Y[mask] = sin(angles[mask]).round()/detector_width
	Xinv[mask] = 0

	offset = midpoint_detectors - X*midpoint_domain[0] \
			- Y*midpoint_domain[1] + detector_shift/detector_width


	#Angular weights
	angles_index = np.argsort(angles%(np.pi)) 
	angles_sorted=angles[angles_index]	%(np.pi)
	
	if fullangle==True:
		angles_sorted=np.array(hstack([-np.pi+angles_sorted[-1],angles_sorted,angles_sorted[0]+np.pi]))
	else:
		angles_sorted=np.array(hstack([2*angles_sorted[0]-angles_sorted[1],angles_sorted,2*angles_sorted[len(angles)-1]-angles_sorted[len(angles)-2]]))
	angles_diff= 0.5*(abs(angles_sorted[2:len(angles_sorted)]-angles_sorted[0:len(angles_sorted)-2]))
	angles_diff=angles_diff[angles_index]


	ofs = zeros((8, len(angles)), dtype=data_type, order='F')
	ofs[0,:] = X; ofs[1,:] = Y; ofs[2,:] = offset; ofs[3,:] = Xinv
	ofs[4,:]=angles_diff
	
	ofs_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
	cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
	
	
	
	[Nx,Ny]=shape
	[Ni,Nj]= [nd,len(angles)]
	delta_x=image_width/float(max(Nx,Ny))
	
	delta_xi=detector_width*delta_x
	Geometry_rescaled = np.array([delta_x,delta_xi,Nx,Ny,Ni,Nj])
	Geometry_rescaled =	clarray.to_device(queue, require(Geometry_rescaled, data_type, 'F'))

	import pdb;pdb.set_trace()
	sinogram_shape = (nd, len(angles))
	return (ofs_buf, shape, sinogram_shape, Geometry_rescaled)
 

"""Starts the GPU Radon transform code 
input	sino ... A pyopencl.array in which result will be saved.
		img ...	 A pyopencl.array in which the image for the radontransform is contained
		r_struct. ..	The r_struct corresponding the given topology (geometry), see radon_struct
output	An event for the queue to compute the radon transform of image saved into img w.r.t. r_struct geometry
"""		   


def fanbeam_richy_gpu(sino, img, projection_settings, wait_for=None):
	ofs_buf=projection_settings.ofs_buf; sdpd_buf=projection_settings.sdpd_buf
	Geometry_rescaled=projection_settings.Geometry_rescaled
	
	#shape,sinogram_shape,ofs_buf,sdpd_buf,Geometryinfo,Geometry_rescaled = f_struct 

	assert (sino.dtype==img.dtype),("sinogram and image do not share common data type: "\
		+str(sino.dtype)+" and "+str(img.dtype))
		
	assert(sino.dtype==projection_settings.dtype),\
		("sinogram and projection_settings do not share common data type: "\
		+str(sino.dtype)+" and "+str(projection_settings.dtype)\
		+"this might be remidies by choosing the parameter data_type to\
		 be equal to 'single' (for float32) or 'double' (float) in the initialization \
		 of the parameter_settings" )

	
	if projection_settings.data_type=='SINGLE':
		return prg.fanbeam_float(sino.queue, sino.shape, None,
						sino.data, img.data, ofs_buf,sdpd_buf,
						Geometry_rescaled.data,
						wait_for=wait_for)
	if projection_settings.data_type=='DOUBLE':
		return prg.fanbeam_double(sino.queue, sino.shape, None,
						sino.data, img.data, ofs_buf,sdpd_buf,
						Geometry_rescaled.data,
						wait_for=wait_for)


def fanbeam_richy_gpu_add( img,sino, projection_settings, wait_for=None):
	
	ofs_buf=projection_settings.ofs_buf; sdpd_buf=projection_settings.sdpd_buf
	Geometry_rescaled=projection_settings.Geometry_rescaled

	#shape,sinogram_shape,ofs_buf,sdpd_buf,Geometryinfo,Geometry_rescaled = f_struct 
	
	assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
			+str(sino.dtype)+" and "+str(img.dtype))
			
	assert(sino.dtype==projection_settings.dtype),\
			("sinogram and projection_settings do not share common data type: "\
			+str(sino.dtype)+" and "+str(projection_settings.dtype)+\
			"this might be remidies by choosing the parameter data_type to\
			 be equal to float32 or float (double precission) in the \
			 initialization of the parameter_settings" )

	if projection_settings.data_type=='SINGLE':
		return prg.fanbeam_add_float(img.queue, img.shape, None,
					   img.data,sino.data, ofs_buf,sdpd_buf,Geometry_rescaled.data,
					   wait_for=wait_for)

	if projection_settings.data_type=='DOUBLE':
		return prg.fanbeam_add_double(img.queue, img.shape, None,
					   img.data,sino.data, ofs_buf,sdpd_buf,Geometry_rescaled.data,
					   wait_for=wait_for)






def fanbeam_struct_gpu(queue, shape, angles, detector_width,
                   source_detector_dist, source_origin_dist,
                   n_detectors=None, detector_shift = 0.0,
                   image_width=None,midpointshift=[0,0], fullangle=True, 
                   data_type="float32"):
	
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
	midpoint_detectors = midpoint_detectors+detector_shift*nd/detector_width
	
	#Ensure that indeed detector on the opposite side of the source
	#assert source_detector_dist>source_origin_dist, 'Origin not between detector and source'
	
	#Save relevant geometric inforrmation
	sinogram_shape = (nd, len(angles))
	Geometryinfo=zeros(8,dtype=data_type,order='F')
	Geometryinfo[0]=(shape[0]-1)/2.
	Geometryinfo[1]=(shape[1]-1)/2.
	Geometryinfo[2]=sinogram_shape[0]
	Geometryinfo[3]=detector_width
	Geometryinfo[4]=source_detector_dist
	Geometryinfo[5]=source_origin_dist
	
	#to include detector shift, the midpoint changes accordingly (with detectorshift=detectorwidth/2 would correspond to the closest point to the source is the end of the detector
	Geometryinfo[7]=midpoint_detectors

	#In case no image_width is predetermined, image_width is chosen in a way that the (square) image is always contained inside the fan between source and detector
	if image_width==None:
		dd=(0.5*detector_width-abs(detector_shift))/source_detector_dist
		image_width = 2*dd*source_origin_dist/sqrt(1+dd**2) # Projection to compute distance via projectionvector (1,dd) after normalization, is equal to delta_x*N_x
	
	assert image_width<source_origin_dist , " the image is encloses the source"
	
	midpoint_x=midpointshift[0]*image_pixels/float(image_width)+(shape[0]-1)/2.
	midpoint_y=midpointshift[1]*image_pixels/float(image_width)+(shape[0]-1)/2.

		
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


	xi=(np.arange(0,nd)- midpoint_detectors)*detector_width/nd
	source_detectorpixel_distance= sqrt((xi)**2+source_detector_dist**2)
	source_detectorpixel_distance=np.array(source_detectorpixel_distance,dtype=data_type,order='F')
	sdpd = zeros(( 1,len(source_detectorpixel_distance)), dtype=data_type, order='F')
	sdpd[0,:]=source_detectorpixel_distance[:]
	sdpd_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, sdpd.nbytes)
	

	Geometryinfo[6]=image_width			
	
	
	#Angular weights
	angles_index = np.argsort(angles%(2*np.pi)) 
	angles_sorted=angles[angles_index]	%(2*np.pi)
	
	if fullangle==True:
		angles_sorted=np.array(hstack([-2*np.pi+angles_sorted[-1],angles_sorted,angles_sorted[0]+2*np.pi]))
	else:
		angles_sorted=np.array(hstack([2*angles_sorted[0]-angles_sorted[1],angles_sorted,2*angles_sorted[len(angles)-1]-angles_sorted[len(angles)-2]]))
	angles_diff= 0.5*(abs(angles_sorted[2:len(angles_sorted)]-angles_sorted[0:len(angles_sorted)-2]))
	angles_diff=angles_diff[angles_index]
	#import pdb;pdb.set_trace()
	
	#Save relevant information
	ofs = zeros((8, len(angles)), dtype=data_type, order='F')
	ofs[0,:] = XD; ofs[1,:] = YD
	ofs[2,:]=Qx; ofs[3,:]=Qy
	ofs[4,:]=Dx0; ofs[5]=Dy0
	ofs[6]=angles_diff
	#write to Buffer
	ofs_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
	
	#import pdb;pdb.set_trace()
	cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
	cl.enqueue_copy(queue, sdpd_buf, sdpd.data).wait()
	
	Geometry_rescaled=np.array([source_detector_dist,source_origin_dist,detector_width/nd, midpoint_x,midpoint_y,midpoint_detectors,shape[0],shape[1],sinogram_shape[0],sinogram_shape[1],image_width/float(max(shape))])
	Geometry_rescaled=	clarray.to_device(queue, require(Geometry_rescaled, data_type, 'F'))
	#import pdb;pdb.set_trace()
	
	return (shape,sinogram_shape,ofs_buf,sdpd_buf,Geometryinfo,Geometry_rescaled)

def show_geometry(angle,f_struct):
	
	#Extract relevant geometric information
	sino_size=f_struct[4][2]
	detector_width=f_struct[4][3]
	source_detector_dist=f_struct[4][4]
	source_origin_dist=f_struct[4][5]
	image_width=f_struct[4][6]
	midpoint_det=f_struct[4][7]
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
	show()
	
	

def normest(queue, projection_settings):
	
	img = clarray.to_device(queue, require((random.randn(*projection_settings.shape)), projection_settings.dtype, 'F'))
	
	sino = clarray.zeros(queue, projection_settings.sinogram_shape, dtype=projection_settings.dtype, order='F')

	V=(forwardprojection(sino, img, projection_settings, wait_for=img.events))
	
	for i in range(50):
		#normsqr = float(sum(img.get()**2)**0.5)
		normsqr = float(clarray.sum(img).get())
	
		img /= normsqr
		#import pdb; pdb.set_trace()
		sino.events.append( forwardprojection (sino, img, projection_settings, wait_for=img.events))
		img.events.append(backprojection(img, sino, projection_settings, wait_for=sino.events))
		
		if i%10==0:
			print('normest',i, normsqr)
	return sqrt(normsqr)


class projection_settings():
	def __init__(self, queue,geometry, img_shape, angles, n_detectors=None, 
					detector_width=1,detector_shift = 0.0, midpointshift=[0,0],
					R=None, RE=None,
					image_width=None, fullangle=True,data_type="SINGLE"):
		
		self.geometry=geometry.upper()

		
		if isscalar(img_shape):
			img_shape=[img_shape,img_shape]
		self.shape=tuple(img_shape)
		
		if isscalar(angles):
			if self.geometry in ["FAN","FANBEAM"]:
				angles = linspace(0,2*pi,angles+1)[:-1] + pi
			elif self.geometry in ["RADON","PARALLEL"]:
				angles = linspace(0,pi,angles+1)[:-1] 
		self.angles=angles
		
		if n_detectors is None:
			self.n_detectors = int(ceil(hypot(img_shape[0],img_shape[1])))
		else:
			self.n_detectors = n_detectors

		detector_width=float(detector_width)
		
		self.N_angles=len(angles)
		self.sinogram_shape=(self.n_detectors,self.N_angles)

		self.fullangle=fullangle
		
		self.detector_shift = detector_shift
		self.midpointshift=midpointshift
		
		self.detector_width=detector_width
		self.R=R
		self.RE=RE
				
		
		self.queue=queue
		
		
		if data_type==float32:
			self.data_type="SINGLE"
		elif data_type==float:
			self.data_type="DOUBLE"
		else:
			self.data_type=data_type.upper()
		assert(self.data_type.upper() in ["SINGLE","DOUBLE"]),\
				"Unknown data_type, choose 'single' or float32  for \
				float operations, or 'double' or float for double precision"
		
		

		if self.data_type.upper()=="SINGLE":
			self.dtype=float32
		elif self.data_type.upper()=="DOUBLE":
			self.dtype=float
			
	

		if self.geometry not in ["RADON","PARALLEL","FAN","FANBEAM"]:
			raise("unknown projection_type, projection_type must be 'parallel' or 'fan'")
		if self.geometry in ["FAN","FANBEAM"]:
			assert( (R!=None)*(R!=None)),"For the Fanbeam geometry \
				you need to set R (the normal distance from source to detector)\
				 and RE (distance from source to coordinate origin which is the \
				 rotation center) "
				 
			
			self.struct=fanbeam_struct_gpu(self.queue,self.shape, self.angles, 
					self.detector_width, R, self.RE,self.n_detectors, self.detector_shift,
					image_width,self.midpointshift, self.fullangle, self.dtype)
		
			self.image_width=self.struct[4][6]
			self.sdpd_buf=self.struct[3]
			
			
			self.ofs_buf=self.struct[2]
		
			self.Geometryinfo= self.struct[4]
			self.Geometry_rescaled=self.struct[5]
			
			
			self.delta_x=self.image_width/max(img_shape)
			self.delta_ratio=self.struct[5][2]
			self.delta_xi=self.delta_x*self.delta_ratio

	
		if self.geometry in ["RADON","PARALLEL"]:
			
			if image_width==None:
				image_width=2
			import pdb;pdb.set_trace()
			self.struct=radon_struct(self.queue,self.shape, self.angles, 
				n_detectors=self.n_detectors, detector_width=self.detector_width*max(img_shape)/n_detectors, image_width= image_width,detector_shift=self.detector_shift,
				fullangle=self.fullangle, data_type=self.dtype)
                
			self.ofs_buf=self.struct[0]
			self.delta_x=self.struct[3][0]		
			self.delta_xi=self.struct[3][1]
			self.delta_ratio=self.delta_xi/self.delta_x
			#self.image_width=self.delta_x*max(shape)
			self.Geometry_rescaled=self.struct[3]

		
	def show_geometry(self,angle):
		#Extract relevant geometric information
		n_detectors=self.n_detectors
		detector_width=self.detector_width
		source_detector_dist=self.R
		source_origin_dist=self.RE
		image_width=self.image_width
		midpoint_det=self.struct[4][7]
		figure(0)
		#Plot all relevant sizes
		plot([-source_origin_dist, source_detector_dist-source_origin_dist],[0 ,0],"c")
		plot([source_detector_dist-source_origin_dist, source_detector_dist-source_origin_dist], [detector_width*midpoint_det/n_detectors , detector_width*(midpoint_det/n_detectors-1)],"k")
		
		plot([-source_origin_dist, source_detector_dist-source_origin_dist],[0,detector_width*midpoint_det/n_detectors],"g")
		plot([-source_origin_dist, source_detector_dist-source_origin_dist],[0,detector_width*(midpoint_det/n_detectors-1)],"g")

		x1=(cos(angle)+sin(angle))*image_width*0.5; x2=(cos(angle)-sin(angle))*image_width*0.5;
		
		plot([x1,x2,-x1,-x2,x1],[x2,-x1,-x2,x1,x2],"b")
		
		draw_circle=matplotlib.patches.Circle((0, 0), image_width/sqrt(2), color='r')
		
		gcf().gca().add_artist(draw_circle)	
		draw_circle=matplotlib.patches.Circle((0, 0), image_width/2, color='g')
		
		gcf().gca().add_artist(draw_circle)	

		show()




def Landweberiteration(sinogram,projection_settings,number_iterations=100,w=1):
	sino2=clarray.to_device(queue, require(sinogram.get(), projection_settings.dtype, 'F'))
	
	U=clarray.zeros(queue,projection_settings.shape,dtype=projection_settings.dtype, order='F')
	Unew=clarray.zeros(queue,projection_settings.shape,dtype=projection_settings.dtype, order='F')
	
	norm=normest(queue,projection_settings)
	#norm=1000
	
	w=float32(w/norm**2)
	
	print("norm",norm)
	for i in range(number_iterations):
	
		#import pdb; pdb.set_trace()
		sino2.events.append(forwardprojection(sino2,Unew,projection_settings,wait_for=U.events+Unew.events))
		sino2=sino2-sinogram
		cl.enqueue_barrier(queue)
		U.events.append(backprojection(U,sino2,projection_settings,wait_for=sino2.events))
		Unew=Unew-w*U	
		print (i,np.max(Unew.get()), np.linalg.norm(sino2.get()))
	return Unew









