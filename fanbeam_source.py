
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

raw_code=open("fanbeam").read()
adjusted_code=raw_code.replace("my_variable_type","float")+raw_code.replace("my_variable_type","double")
prg = Program(ctx,adjusted_code)



def fanbeam_richy_gpu(sino, img, projection_settings, wait_for=None):
	ofs_buf=projection_settings.ofs_buf; sdpd_buf=projection_settings.sdpd_buf
	Geometry_rescaled=projection_settings.Geometry_rescaled
	
	#shape,sinogram_shape,ofs_buf,sdpd_buf,Geometryinfo,Geometry_rescaled = f_struct 

	assert (sino.dtype==img.dtype),("sinogram and image do not share common data type: "+str(sino.dtype)+" and "+str(img.dtype))
	assert(sino.dtype==projection_settings.data_type),("sinogram and projection_settings do not share common data type: "+str(sino.dtype)+" and "+str(projection_settings.data_type)+"this might be remidies by choosing the parameter data_type to be equal to float32 or float (double precission) in the initialization of the parameter_settings" )

	
	if projection_settings.data_type==float32:
		return prg.fanbeam_float(sino.queue, sino.shape, None,
						sino.data, img.data, ofs_buf,sdpd_buf,
						Geometry_rescaled.data,
						wait_for=wait_for)
	if projection_settings.data_type==float:
		return prg.fanbeam_double(sino.queue, sino.shape, None,
						sino.data, img.data, ofs_buf,sdpd_buf,
						Geometry_rescaled.data,
						wait_for=wait_for)
						
					   

def fanbeam_richy_gpu_add( img,sino, projection_settings, wait_for=None):
	
	ofs_buf=projection_settings.ofs_buf; sdpd_buf=projection_settings.sdpd_buf
	Geometry_rescaled=projection_settings.Geometry_rescaled

	#shape,sinogram_shape,ofs_buf,sdpd_buf,Geometryinfo,Geometry_rescaled = f_struct 
	
	assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "+str(sino.dtype)+" and "+str(img.dtype))
	assert(sino.dtype==projection_settings.data_type),("sinogram and projection_settings do not share common data type: "+str(sino.dtype)+" and "+str(projection_settings.data_type)+"this might be remidies by choosing the parameter data_type to be equal to float32 or float (double precission) in the initialization of the parameter_settings" )

	if projection_settings.data_type==float32:
		return prg.fanbeam_add_float(img.queue, img.shape, None,
					   img.data,sino.data, ofs_buf,sdpd_buf,Geometry_rescaled.data,
					   wait_for=wait_for)

	if projection_settings.data_type==float:
		return prg.fanbeam_add_double(img.queue, img.shape, None,
					   img.data,sino.data, ofs_buf,sdpd_buf,Geometry_rescaled.data,
					   wait_for=wait_for)



def fanbeam_struct_gpu( shape, angles, detector_width,
                   source_detector_dist, source_origin_dist,
                   n_detectors=None, detector_shift = 0.0,
                   image_width=None,midpointshift=[0,0], fullangle=True, data_type="float32"):
	
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
	
	

def Fanbeam_normest(queue, projection_settings):
	
	img = clarray.to_device(queue, require((random.randn(*projection_settings.shape)), float32, 'F'))
	
	sino = clarray.zeros(queue, projection_settings.sinogram_shape, dtype=float32, order='F')

	V=(fanbeam_richy_gpu(sino, img, projection_settings, wait_for=img.events))

	for i in range(50):
		#normsqr = float(sum(img.get()**2)**0.5)
		normsqr = float(clarray.sum(img).get())
		img /= normsqr
		#import pdb; pdb.set_trace()
		sino.events.append( fanbeam_richy_gpu (sino, img, projection_settings, wait_for=img.events))
		img.events.append(fanbeam_richy_gpu_add(img, sino, projection_settings, wait_for=sino.events))
		
		if i%10==0:
			print('normest',i, normsqr)
	return sqrt(normsqr)


class projection_settings():
	def __init__(self,img_shape, angles, n_detectors=None, geometry="parallel",
					detector_width=1,detector_shift = 0.0, midpointshift=[0,0],
					R=None, RE=None,
					image_width=None, fullangle=True,data_type=float):
		
		if isscalar(img_shape):
			img_shape=[img_shape,img_shape]
		self.shape=tuple(img_shape)
		
		if isscalar(angles):
			angles = linspace(0,2*pi,angles+1)[:-1] + pi		
		self.angles=angles

		self.image_width=sqrt(2)
		self.geometry=geometry
		if geometry not in ["parallel","fan"]:
			raise("unknown projection_type, projection_type must be 'parallel' or 'fan'")
		if geometry=="fan":
			self.struct=fanbeam_struct_gpu(self.shape, angles, detector_width,
                   R, RE,n_detectors, detector_shift,
                   image_width,midpointshift, fullangle,
                   data_type)
			
			self.image_width=self.struct[4][6]
	
		self.N_angles=len(angles)
		self.sinogram_shape=(n_detectors,self.N_angles)
		self.R=R
		self.RE=RE
		self.detector_width=detector_width
		
		self.ofs_buf=self.struct[2]
		self.sdpd_buf=self.struct[3]
		self.Geometryinfo= self.struct[4]
		self.Geometry_rescaled=self.struct[5]
		
		self.n_detectors=n_detectors
		
		self.detector_shift = detector_shift
		self.midpointshift=midpointshift
		self.fullangle=fullangle
		
		
		self.delta_x=self.image_width/max(img_shape)
		self.delta_ratio=self.struct[5][2]
		self.delt_xi=self.delta_x*self.delta_ratio
				
		self.data_type=data_type
		
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
	sino2=clarray.to_device(queue, require(sinogram.get(), float32, 'F'))
	
	U=clarray.zeros(queue,projection_settings.shape,dtype=float32, order='F')
	Unew=clarray.zeros(queue,projection_settings.shape,dtype=float32, order='F')
	
	norm=Fanbeam_normest(queue,projection_settings)
	#norm=1000
	
	w=float32(w/norm**2)
	
	print("norm",norm)
	for i in range(number_iterations):
	
		#import pdb; pdb.set_trace()
		sino2.events.append(fanbeam_richy_gpu(sino2,Unew,projection_settings,wait_for=U.events+Unew.events))
		sino2=sino2-sinogram
		cl.enqueue_barrier(queue)
		U.events.append(fanbeam_richy_gpu_add(U,sino2,projection_settings,wait_for=sino2.events))
		Unew=Unew-w*U	
		print (i,np.max(Unew.get()), np.linalg.norm(sino2.get()))
	return Unew









