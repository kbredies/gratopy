from numpy import *
import numpy as np
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray




###########
## Programm created from the gpu_code 
class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build()
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel



"""Starts the GPU forwardprojection code
Computes the forwardprojection (i.e. Radon transform or Fanbeam transform) 
of an given  image based on the given projection_settings 
input	sino ... A pyopencl.array in which the result of transformation will be saved. 
				 If equald 'None' is given, a new array will be created in which the result is saved
		img ...	 A pyopencl.array containing the image to be transformed
		projection_settings...	Instance of the projection_settings class
				 representing the geometric setting in Question
output	sino ... An pyopencl.array representing the transformation (i.e. the sinogram)
				 In case sino is given as input, this will be the identical array 
				 but the values save therein will be changed 
				 (or are in the waitinglist to be changed once wait_for limitations have passed) 
"""
def forwardprojection(sino,img, projection_settings, wait_for=None):
	
	my_function={0:radon,1:fanbeam}
	
	if img.flags.c_contiguous:
		img=img.T
	
	#Initialize new sinogram if no sinogram is yet given
	if sino is None:
		z_dimension=tuple()
		if len(img.shape)>2:
			z_dimension=[img.shape[2]]
		sino = clarray.zeros(projection_settings.queue, projection_settings.sinogram_shape+tuple(z_dimension), dtype=img.dtype, order='F')	

	#execute corresponding Projection operation
	myevent=my_function[projection_settings.is_fan](sino, img, projection_settings, wait_for=wait_for)
	return sino


"""Starts the GPU backprojection code
Computes the backprojection (i.e. Radon backprojection or Fanbeam backprojection) 
of an given sinogram based on the given projection_settings 
input	img ... A pyopencl.array in which the result of backprojection will be saved. 
				 If equal 'None' is given, a new array will be created in which the result is saved
		sino ... A pyopencl.array containing the image to be transformed
		projection_settings...	Instance of the projection_settings class
				 representing the geometric setting in question
		wait_for... events to be waited for before transform is executed, see pyopencl wait_for
output	img ... An pyopencl.array representing the backprojection 
				 In case img is given as input, this will be the identical array 
				 but the values save therein will be changed 
				 (or are in the waitinglist to be changed once wait_for limitations have passed) 
"""	
def backprojection(img, sino, projection_settings, wait_for=None):
	
	my_function={0:radon_ad,1:fanbeam_add}
	
	
	if sino.flags.c_contiguous:
		sino=sino.T
	
	#Initialize new img (to save backprojection in) if none is yet given
	if img is None:
		z_dimension=tuple()
		if len(sino.shape)>2:
			z_dimension=[sino.shape[2]]
		img = clarray.zeros(projection_settings.queue, projection_settings.shape+tuple(z_dimension), dtype=sino.dtype, order='F')	

	#execute corresponding backprojection operation
	myevent=my_function[projection_settings.is_fan](img, sino, projection_settings, wait_for=wait_for)
	return img




"""Starts the GPU Radontransform code
Computes the  Radon transform  
of an given  image based on the given projection_settings 
input	sino ... A pyopencl.array in which the result of transformation will be saved. 
		img ...	 A pyopencl.array containing the image to be transformed
		projection_settings...	Instance of the projection_settings class
				 representing the geometric setting in question
		wait_for... events to be waited for before transform is executed, see pyopencl wait_for
output	sino ... An event for the  the transformation (i.e. the sinogram)
				 and the resultun transform is written into sino 
				 (or are in the waitinglist to be changed once wait_for limitations have passed) 
"""			
def radon(sino, img, projection_settings, wait_for=None):
	
	my_function={np.dtype('float32'):projection_settings.prg.radon_float,
			np.dtype('float'):projection_settings.prg.radon_double}

	#ensure that all relevant arrays have common data_type
	assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
			+str(sino.dtype)+" and "+str(img.dtype))
	
	my_data_type=sino.dtype
	updload_bufs(projection_settings,my_data_type)	
	ofs_buf=projection_settings.ofs_buf[my_data_type] 
	geometry_information=projection_settings.geometry_information[my_data_type]

	
	#Choose function with approrpiate dtype 
	myevent=my_function[my_data_type](sino.queue, sino.shape, None,
					sino.data, img.data, ofs_buf,
					geometry_information,
					wait_for=wait_for)

	
	sino.add_event(myevent)
	return myevent
	
"""Starts the GPU backprojection code 
input	img ... A pyopencl.array in which the resulting backprojection will be saved.
		sino ...	 A pyopencl.array in which the sinogram for the adjoint radontransform is contained
		projection_settings...	Instance of the projection_settings class
				 representing the geometric setting in question
		wait_for... events to be waited for before transform is executed, see pyopencl wait_for
output	An event for the queue to compute the adjoint radon transform of image saved into img w.r.t. projection_settings
		and write the result into img
"""
def radon_ad(img, sino, projection_settings, wait_for=None):
	my_function={np.dtype('float32'):projection_settings.prg.radon_ad_float,
			np.dtype('float'):projection_settings.prg.radon_ad_double}

	#ensure that all relevant arrays have common data_type
	assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
			+str(sino.dtype)+" and "+str(img.dtype))
			

	my_data_type=sino.dtype
	updload_bufs(projection_settings,my_data_type)	
	ofs_buf=projection_settings.ofs_buf[my_data_type] 
	geometry_information=projection_settings.geometry_information[my_data_type]


	#Choose function with approrpiate dtype 
	
	myevent=my_function[my_data_type](img.queue, img.shape, None,
						img.data, sino.data, ofs_buf,
						geometry_information,wait_for=wait_for)
	img.add_event(myevent)
	return myevent


"""Creates the structure of radon geometry required for radontransform and its adjoint
Input:
		queue ... a queue object corresponding to a context in pyopencl
		shape ... the shape of the object (image) in pixels
		angles ... a list of angles considered
		n_detectors ... Number of detectors, i.e. resolution of the projections
		detector_width ...  length of the detector(default=2)
		image_width ...    diameter of the object in question (side-length of the image square)
		detector_shift ... global shift of detector_ofsets (default 0)
		data_type	...	shall the relevant information be saved in float32 or float
		fullangle ... 	Bool, True if the given angles incorporate an partion of [0,2pi]
						false otherwise
Output:
		ofs_buf ... a buffer object with 4 x number of angles entries corresponding to the cos and sin divided by the detectorwidth, also offset depending on the angle and the inverse of the cos values
		shape ... The same shape as in the input.
		sinogram_shape ... The sinogram_shape is a list with first the number of detectors, then number of angles.
		Geometry_info  ...  pyopencl array containing the geometric
							information [delta_x,delta_s,Nx,Ny,Ni,Nj]
"""
def radon_struct(queue, shape, angles, n_detectors=None,
			 detector_width=2.0, image_width=2.0,detector_shift=0.0,fullangle=True):
	
	#relative_detector_pixel_width is delta_s/delta_x
	relative_detector_pixel_width=detector_width/float(image_width)*max(shape)/n_detectors
	
	
	#When angles are None, understand as number of angles discretizing [0,pi]
	if isscalar(angles):	
		angles = linspace(0,pi,angles+1)[:-1]
	
	#Choosing the number of detectors as the half of the diagonal through the the image (in image_pixel scale)
	if n_detectors is None:
		nd = int(ceil(hypot(shape[0],shape[1])))
	else:
		nd = n_detectors
	
	
	#Extract angle information
	if isinstance(angles[0], list) or  isinstance(angles[0], np.ndarray):
		n_angles=0
		angles_new=[]
		angles_section=[0]
		count=0
		for j in range(len(angles)):
			n_angles+=len(angles[j])
			for k in range(len(angles[j])): 
				angles_new.append(angles[j][k])
				count+=1
			angles_section.append(count)
		angles=np.array(angles_new)
	else:
		n_angles=len(angles)
		angles_section=[0,n_angles]
	
	sinogram_shape = (nd, n_angles)
	
	#Angular weights (resolution associated to angles)
	#If fullangle is activated, the angles partion [0,pi] completely and choose the first /last width appropriately
	if fullangle==True:
		angles_index = np.argsort(angles%(np.pi)) 
		angles_sorted=angles[angles_index]	%(np.pi)
		angles_sorted=np.array(hstack([-np.pi+angles_sorted[-1],angles_sorted,angles_sorted[0]+np.pi]))
		angles_diff= 0.5*(abs(angles_sorted[2:len(angles_sorted)]-angles_sorted[0:len(angles_sorted)-2]))
		angles_diff=np.array(angles_diff)
		angles_diff=angles_diff[angles_index]
	else:##Act as though first/last angles width is equal to the distance from the second/second to last angle  
		angles_diff=[]
		for j in range(len(angles_section)-1):
			current_angles=angles[angles_section[j]:angles_section[j+1]]
			current_angles_index = np.argsort(current_angles%(np.pi)) 
			current_angles=current_angles[current_angles_index]	%(np.pi)

			angles_sorted_temp=np.array(hstack([2*current_angles[0]-current_angles[1],
				current_angles,2*current_angles[len(current_angles)-1]-current_angles[len(current_angles)-2]]))
			
			angles_diff_temp= 0.5*(abs(angles_sorted_temp[2:len(angles_sorted_temp)]-angles_sorted_temp[0:len(angles_sorted_temp)-2]))
			angles_diff+=list(angles_diff_temp[current_angles_index])
	
	#Compute the midpoints of geometries
	midpoint_domain = array([shape[0]-1, shape[1]-1])/2.0
	midpoint_detectors = (nd-1.0)/2.0

	# direction vectors and inverse in x 
	X = cos(angles)/relative_detector_pixel_width
	Y = sin(angles)/relative_detector_pixel_width
	Xinv = 1.0/X

	# set near vertical lines to vertical
	mask = abs(Xinv) > 10*nd
	X[mask] = 0
	Y[mask] = sin(angles[mask]).round()/relative_detector_pixel_width
	Xinv[mask] = 0


	#X*x+Y*y=detectorposition, ofs is error in midpoint of the image (in shifted detector setting)
	offset = midpoint_detectors - X*midpoint_domain[0] \
			- Y*midpoint_domain[1] + detector_shift/detector_width*nd


	#Angular weights
	angles_index = np.argsort(angles%(np.pi)) 
	angles_sorted=angles[angles_index]	%(np.pi)
	
	#Also write basic information to gpu 
	[Nx,Ny]=shape
	[Ni,Nj]= [nd,len(angles)]
	delta_x=image_width/float(max(Nx,Ny))
	delta_s=float(detector_width)/nd

	
	Geo_dict={}
	Ofs_dict={}
	for my_data_type in [np.dtype('float64'),np.dtype('float32')]:
		#Save angular information into the ofs buffer
		ofs = zeros((8, len(angles)), dtype=my_data_type, order='F')
		ofs[0,:] = X; ofs[1,:] = Y; ofs[2,:] = offset; ofs[3,:] = Xinv
		ofs[4,:]=angles_diff
		Ofs_dict[my_data_type]=ofs
		
		geometry_info = np.array([delta_x,delta_s,Nx,Ny,Ni,Nj],dtype=my_data_type,order='F')
		Geo_dict[my_data_type]=geometry_info
	
	

	sinogram_shape = (nd, len(angles))
	return (Ofs_dict, shape, sinogram_shape, Geo_dict,angles_diff)
 

"""Starts the GPU Radon transform code 
input	sino ... A pyopencl.array in which result will be saved.
		img ...	 A pyopencl.array in which the image for the radontransform is contained
		r_struct. ..	The r_struct corresponding the given topology (geometry), see radon_struct
output	An event for the queue to compute the radon transform of image saved into img w.r.t. r_struct geometry
"""		   
def fanbeam(sino, img, projection_settings, wait_for=None):
	
	my_function={np.dtype('float32'):projection_settings.prg.fanbeam_float,
			np.dtype('float'):projection_settings.prg.fanbeam_double}
	
	#ensure that all relevant arrays have common data_type
	assert (sino.dtype==img.dtype),("sinogram and image do not share common data type: "\
		+str(sino.dtype)+" and "+str(img.dtype))	
		 
	my_data_type=sino.dtype

	updload_bufs(projection_settings,my_data_type)	
	ofs_buf=projection_settings.ofs_buf[my_data_type]; 
	sdpd_buf=projection_settings.sdpd_buf[my_data_type]
	geometry_information=projection_settings.geometry_information[my_data_type]
	
	#Choose function with approrpiate dtype
	myevent=my_function[my_data_type](sino.queue, sino.shape, None,
						sino.data, img.data, ofs_buf,sdpd_buf,
						geometry_information,
						wait_for=wait_for)
	sino.add_event(myevent)
	return myevent

def fanbeam_add( img,sino, projection_settings, wait_for=None):
	
	my_function={np.dtype('float32'):projection_settings.prg.fanbeam_add_float,
			np.dtype('float'):projection_settings.prg.fanbeam_add_double}

	#ensure that all relevant arrays have common data_type
	assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
			+str(sino.dtype)+" and "+str(img.dtype))
			

	my_data_type=sino.dtype
	
	updload_bufs(projection_settings,my_data_type)	
	ofs_buf=projection_settings.ofs_buf[my_data_type]; 
	sdpd_buf=projection_settings.sdpd_buf[my_data_type]
	geometry_information=projection_settings.geometry_information[my_data_type]

	
	
	myevent = my_function[my_data_type](img.queue, img.shape, None,
					   img.data,sino.data, ofs_buf,sdpd_buf,geometry_information,
					   wait_for=wait_for)
					   
	img.add_event(myevent)
	return myevent



"""Creates the structure of fanbeam geometry required for fanbeamtransform and its adjoint
Input:
		queue ... a queue object corresponding to a context in pyopencl
		shape ... the shape of the object (image) in pixels
		angles ... a list of angles considered
		detector_width ...  length of the detector
		source_detector_dist...  Distance (orthonormal) from the source to the detectorline
		source_origin_dist ...	 Distance from source to origin (which is center of rotation)
		n_detectors ... Number of detectors, i.e. resolution of the projections
		detector_shift ... global shift of detector_ofsets (default 0)
		image_width ...    diameter of the object in question (side-length of the image square)
							if it is chosen as None, a suitable size will be chosen (and returned)
		midpoint_shift... shifts the center of the image (then not the center of rotation, which remains the origin)
		data_type	...	shall the relevant information be saved in float32 or float
		fullangle ... 	Bool, True if the given angles incorporate an partion of [0,2pi]
						false otherwise
Output:
shape,sinogram_shape,ofs_buf,sdpd_buf,image_width,Geometry_info
		shape ... The same shape as in the input.
		sinogram_shape ... The sinogram_shape is a list with first the number of detectors, then number of angles.
		ofs_buf ... a buffer object with 4 x number of angles entries corresponding to the cos and sin divided by the detectorwidth, also offset depending on the angle and the inverse of the cos values
		sdpd_buf... a bufer representing sqrt(xi^2+R^2), i.e. weighting term
		image_width...	a scalar represnting the image_width (see input image_width)
		Geometry_info  ...  pyopencl array containing the geometric
							information [R,RE,delta_s(rescaled to delta_x), midpoint_x,midpoint_y,midpoint_detectors,Nx,Ny,Nxi,Nphi,delta_x]
"""
def fanbeam_struct_gpu(queue, shape, angles, detector_width,
                   source_detector_dist, source_origin_dist,
                   n_detectors=None, detector_shift = 0.0,
                   image_width=None,midpointshift=[0,0], fullangle=True):
	
	detector_width=float(detector_width)
	source_detector_dist=float(source_detector_dist)
	source_origin_dist=float(source_origin_dist)
	midpointshift=np.array(midpointshift)
	
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
	
	#Extract angle information
	if isinstance(angles[0], list) or  isinstance(angles[0], np.ndarray):
		n_angles=0
		angles_new=[]
		angles_section=[0]
		count=0
		for j in range(len(angles)):
			n_angles+=len(angles[j])
			for k in range(len(angles[j])): 
				angles_new.append(angles[j][k])
				count+=1
			angles_section.append(count)
		angles=np.array(angles_new)
	else:
		n_angles=len(angles)
		angles_section=[0,n_angles]
	
	sinogram_shape = (nd, n_angles)
	
	#Angular weights (resolution associated to angles)
	#If fullangle is activated, the angles partion [0,2pi] completely and choose the first /last width appropriately
	if fullangle==True:
		angles_index = np.argsort(angles%(2*np.pi)) 
		angles_sorted=angles[angles_index]	%(2*np.pi)
		angles_sorted=np.array(hstack([-2*np.pi+angles_sorted[-1],angles_sorted,angles_sorted[0]+2*np.pi]))
		angles_diff= 0.5*(abs(angles_sorted[2:len(angles_sorted)]-angles_sorted[0:len(angles_sorted)-2]))
		angles_diff=np.array(angles_diff)
		angles_diff=angles_diff[angles_index]
	else:##Act as though first/last angles width is equal to the distance from the second/second to last angle  
		angles_diff=[]
		for j in range(len(angles_section)-1):
			current_angles=angles[angles_section[j]:angles_section[j+1]]
			current_angles_index = np.argsort(current_angles%(2*np.pi)) 
			current_angles=current_angles[current_angles_index]	%(2*np.pi)

			angles_sorted_temp=np.array(hstack([2*current_angles[0]-current_angles[1],
				current_angles,2*current_angles[len(current_angles)-1]-current_angles[len(current_angles)-2]]))
			
			angles_diff_temp= 0.5*(abs(angles_sorted_temp[2:len(angles_sorted_temp)]-angles_sorted_temp[0:len(angles_sorted_temp)-2]))
			angles_diff+=list(angles_diff_temp[current_angles_index])
		
	
	

	#compute  midpoints foror orientation
	midpoint_domain = array([shape[0]-1, shape[1]-1])/2.0
	midpoint_detectors = (nd-1.0)/2.0
	midpoint_detectors = midpoint_detectors+detector_shift*nd/detector_width
	
	#Ensure that indeed detector on the opposite side of the source
	assert source_detector_dist>source_origin_dist, 'Origin not between detector and source'
	
	#In case no image_width is predetermined, image_width is chosen in a way that the (square) image is always contained inside the fan between source and detector
	if image_width==None:
		dd=(0.5*detector_width-abs(detector_shift))/source_detector_dist
		image_width = 2*dd*source_origin_dist/sqrt(1+dd**2) # Projection to compute distance via projectionvector (1,dd) after normalization, is equal to delta_x*N_x
	
	#Ensure that source is outside the image (otherwise fanbeam is not continuous in classical L2)
	assert image_width*0.5*sqrt(1+(min(shape)/max(shape))**2)+np.linalg.norm(midpointshift)<source_origin_dist , " the image is encloses the source"
	
	#Determine midpoint (in scaling 1 = 1 pixelwidth,i.e. index of center) 
	midpoint_x=midpointshift[0]*image_pixels/float(image_width)+(shape[0]-1)/2.
	midpoint_y=midpointshift[1]*image_pixels/float(image_width)+(shape[1]-1)/2.

		
	# adjust distances to pixel units, i.e. 1 unit corresponds to the length of one image pixel
	source_detector_dist *= image_pixels/float(image_width)
	source_origin_dist *= image_pixels/float(image_width)
	detector_width *= image_pixels/float(image_width)

	# unit vector associated to the angle (vector showing along the detector)
	thetaX = -cos(angles)
	thetaY = sin(angles)
	
	#Direction vector of detector direction normed to the length of a single detector pixel (i.e. delta_s (in the scale of delta_x=1))
	XD=thetaX*detector_width/nd
	YD=thetaY*detector_width/nd

	#Direction vector leading to from source to origin (with proper length)
	Qx=-thetaY*source_origin_dist
	Qy=thetaX*source_origin_dist

	#Direction vector from origin to the detector center
	Dx0= thetaY*(source_detector_dist-source_origin_dist)
	Dy0= -thetaX*(source_detector_dist-source_origin_dist)

	
	#Save relevant information	
	Ofs_dict={}
	Sdpd_dict={}
	Geo_dict={}
	for my_data_type in [np.dtype('float64'),np.dtype('float32')]:
		#Angular Information
		ofs = zeros((8, len(angles)), dtype=my_data_type, order='F')
		ofs[0,:] = XD; ofs[1,:] = YD
		ofs[2,:]=Qx; ofs[3,:]=Qy
		ofs[4,:]=Dx0; ofs[5]=Dy0
		ofs[6]=angles_diff
		Ofs_dict[my_data_type]=ofs
		
		
		# Determine source detectorpixel-distance (=sqrt(R+xi**2)) for scaling
		xi=(np.arange(0,nd)- midpoint_detectors)*detector_width/nd
		source_detectorpixel_distance= sqrt((xi)**2+source_detector_dist**2)
		source_detectorpixel_distance=np.array(source_detectorpixel_distance,dtype=my_data_type,order='F')
		sdpd = zeros(( 1,len(source_detectorpixel_distance)), dtype=my_data_type, order='F')
		sdpd[0,:]=source_detectorpixel_distance[:]
		Sdpd_dict[my_data_type]=sdpd
		
		#collect various geometric information necessary for computations
		geometry_info=np.array([source_detector_dist,source_origin_dist,detector_width/nd,
			midpoint_x,midpoint_y,midpoint_detectors,shape[0],shape[1],sinogram_shape[0],
			sinogram_shape[1],image_width/float(max(shape))],dtype=my_data_type,order='F')
		Geo_dict[my_data_type]=geometry_info
		
	
	
	return (shape,sinogram_shape,Ofs_dict,Sdpd_dict,image_width,Geo_dict,angles_diff)


"""Determine the operator norm of the Projectionmethod
Uses poweriteration (with default 50 iterations) to determine the biggest eigenvalue, i.e. the spectralnorm (dual to L^2) 
input 	
		Projection_setting... from the Projection_settings class containing the projection information
		number_of_iterations ... integer representing the number of iterations to terminate after (default 50)
Output	
		norm ... the operator_norm of the operator
"""
def normest( projection_settings,number_of_iterations=50,my_dtype='float32'):
	queue=projection_settings.queue
	
	#random starting point
	img = clarray.to_device(queue, require((random.randn(*projection_settings.shape)), my_dtype, 'F'))
	sino=forwardprojection(None, img, projection_settings, wait_for=img.events)
	
	#power_iteration
	for i in range(number_of_iterations):
		normsqr = float(clarray.sum(img).get())
	
		img /= normsqr
		forwardprojection (sino, img, projection_settings, wait_for=img.events)
		backprojection(img, sino, projection_settings, wait_for=sino.events)		
	return sqrt(normsqr)


"""Class saving all relevant projection information, which is always used to compute projections
atributes:
			queue ... 	a pyopencl.queue associated to the context in question
			geometry... a strinc containing "PARALLEL"/"Radon" OR "FAN"/"FANBEAM" (i.e. which type of projection)
			self.prg... Instance of programm class containing the programs for the execution of the projection-methods 
			shape ... tuple representing the number of pixels (of the image) in x and y direction
			sinogram_shape... tuple representing the number of detectors and number of angles considered
			angles		... list the angles for which the projections are considered
			n_detector	... int the number of detectors used
			n_agles	... int number of angles used
			R	... float distance (orthogonal from source to detector-line
			RE	... float distance from source to origin (which is center of rotation)
			image_width ...	float the size of the image (i.e. the (larger) side length of the rectangle image)
			detector_width... float the length of the detector used 
			detector_shift ... float shift of the detector
			midpoint_shift ... list of length 2 shifting the image of center by these coordinates
			fullangel ...	bool representing which wether the entirety of the angular domain is partioned by offered angles
			data_type...	string representing "SINGLE" or "DOUBLE" for single or double  precision
			d_type...	np.dtype representing the precision in terms of the data_type, i.e. float32 for single, and float for double  
			struct...  the struct associated to this geometry, see radon_struct and fanbeam_struct
			ofs...		the ofs_buf buffer from the struct
			sdpd...		the sdpd_buf buffer from the struct
			delta_x... discretization parameter in image_resolution
			delta_s... discretization parameter in detector_resolution
			delta_ratio ... delta_s/delta_x
methods:
			show_geometry ... shows geometry visually for a given angle
"""
class projection_settings():
	def __init__(self, queue,geometry, img_shape, angles, n_detectors=None, 
					detector_width=2.0,detector_shift = 0.0, midpoint_shift=[0,0],
					R=None, RE=None,
					image_width=None, fullangle=True):
		
		self.geometry=geometry.upper()
		self.queue=queue
		
		
		raw_fanbeam_code=open("fanbeam").read()
		raw_radon_code=open("Radontransform").read()
		adjusted_code=raw_fanbeam_code.replace("my_variable_type","float")\
			+raw_fanbeam_code.replace("my_variable_type","double")\
			+raw_radon_code.replace("my_variable_type","float")\
			+raw_radon_code.replace("my_variable_type","double")

		self.prg = Program(queue.context,adjusted_code)
		self.image_width=image_width
		
		if isscalar(img_shape):
			img_shape=[img_shape,img_shape]
		
		if len(img_shape)>2:
			img_shape=[img_shape[0],img_shape[1]]
		self.shape=tuple(img_shape)
		
		
		if self.geometry not in ["RADON","PARALLEL","FAN","FANBEAM"]:
			raise("unknown projection_type, projection_type must be 'parallel' or 'fan'")
		if self.geometry in ["RADON","PARALLEL"]:
			self.is_parallel=1
			self.is_fan=0
		if self.geometry in ["FAN","FANBEAM"]:
			self.is_parallel=0
			self.is_fan=1


		
		if isscalar(angles):
			if self.is_fan:
				angles = linspace(0,2*pi,angles+1)[:-1] + pi
			elif self.is_parallel:
				angles = linspace(0,pi,angles+1)[:-1] 
		angles=angles
		
		if n_detectors is None:
			self.n_detectors = int(ceil(hypot(img_shape[0],img_shape[1])))
		else:
			self.n_detectors = n_detectors

		detector_width=float(detector_width)
		
		
		
		if isinstance(angles[0], list)or isinstance(angles[0],np.ndarray):
			self.n_angles=0
			for j in range(len(angles)):
				self.n_angles+=len(angles[j])
		else:
			self.n_angles=len(angles)
		self.angles=angles
		self.sinogram_shape=(self.n_detectors,self.n_angles)

		self.fullangle=fullangle
		
		self.detector_shift = detector_shift
		self.midpoint_shift=midpoint_shift
		
		self.detector_width=detector_width
		self.R=R
		self.RE=RE			
				
			
		if self.is_fan:
			assert( (R!=None)*(RE!=None)),"For the Fanbeam geometry \
				you need to set R (the normal distance from source to detector)\
				 and RE (distance from source to coordinate origin which is the \
				 rotation center) "
				 
			
			self.struct=fanbeam_struct_gpu(self.queue,self.shape, self.angles, 
					self.detector_width, R, self.RE,self.n_detectors, self.detector_shift,
					image_width,self.midpoint_shift, self.fullangle)
			
			self.buf_upload={np.dtype('float64'):0,np.dtype('float32'):0}
			
			self.ofs_buf=self.struct[2]
			self.sdpd_buf=self.struct[3]
			self.image_width=self.struct[4]
			self.geometry_information=self.struct[5]
			
			self.angle_weights=self.struct[6]
			
			self.delta_x=self.image_width/max(img_shape)			
			self.delta_s=detector_width/n_detectors
			self.delta_ratio=self.delta_s/self.delta_x
			
	
		if self.is_parallel:
			if image_width==None:
				self.image_width=2.
			
			self.struct=radon_struct(self.queue,self.shape, self.angles, 
				n_detectors=self.n_detectors,
				detector_width=self.detector_width, image_width= self.image_width,
				detector_shift=self.detector_shift,fullangle=self.fullangle, )
                
			self.ofs_buf=self.struct[0]
			self.buf_upload={np.dtype('float64'):0,np.dtype('float32'):0}
			

			
			self.delta_x=self.image_width/max(self.shape)
			self.delta_s=self.detector_width/self.n_detectors
			self.delta_ratio=self.delta_s/self.delta_x	
			
			
			self.geometry_information=self.struct[3]
			self.angle_weights=self.struct[4]


		
	def show_geometry(self,angle):

		figure(0)
		if self.is_fan:
			detector_width=self.detector_width
			source_detector_dist=self.R
			source_origin_dist=self.RE
			image_width=self.image_width
			midpoint_shift=self.midpoint_shift
			
			maxsize=max(self.RE,sqrt((self.R-self.RE)**2+detector_width**2/4.))
			
			
			angle=-angle
			A=np.array([[cos(angle),sin(angle)],[-sin(angle),cos(angle)]])	
			figure(0)
			#Plot all relevant sizes			
			
			
			sourceposition=[-source_origin_dist,0]
			upper_detector=[source_detector_dist-source_origin_dist,detector_width*0.5+self.detector_shift]
			lower_detector=[source_detector_dist-source_origin_dist,-detector_width*0.5+self.detector_shift]
			central_detector=[source_detector_dist-source_origin_dist,0]
			
			sourceposition=np.dot(A,sourceposition)
			upper_detector=np.dot(A,upper_detector)
			lower_detector=np.dot(A,lower_detector)
			central_detector=np.dot(A,central_detector)
			
			
			plot([upper_detector[0], lower_detector[0]], [upper_detector[1], lower_detector[1]],"k")
			
			plot([sourceposition[0], upper_detector[0]],[sourceposition[1],upper_detector[1]],"g")
			plot([sourceposition[0], lower_detector[0]],[sourceposition[1],lower_detector[1]],"g")
			
			plot([sourceposition[0], central_detector[0]],[sourceposition[1], central_detector[1]],"g")			
			

			#plot(x[0]+midpoint_rotation[0],x[1]+midpoint_rotation[1],"b")
			
			draw_circle=matplotlib.patches.Circle(midpoint_shift, image_width/2*sqrt(1+(min(self.shape)/max(self.shape))**2), color='r')
			gcf().gca().add_artist(draw_circle)
			
			color=(1,1,0)	
			rect = Rectangle(midpoint_shift-0.5*np.array([image_width*self.shape[0]/np.max(self.shape),image_width*self.shape[1]/np.max(self.shape)]), image_width*self.shape[0]/np.max(self.shape),image_width*self.shape[1]/np.max(self.shape),facecolor=color, edgecolor=color)
			gcf().gca().add_artist(rect)	
			
			draw_circle=matplotlib.patches.Circle(midpoint_shift, image_width/2, color='b')			
			gcf().gca().add_artist(draw_circle)	
			
			draw_circle=matplotlib.patches.Circle((0,0), image_width/10, color='k')			
			gcf().gca().add_artist(draw_circle)	
			xlim([-maxsize,maxsize])
			ylim([-maxsize,maxsize])
			
			
		if self.is_parallel:
			
			detector_width=self.detector_width
			image_width=self.image_width
			
			angle=-angle
			A=np.array([[cos(angle),sin(angle)],[-sin(angle),cos(angle)]])	
			
			center_source=[-image_width,self.detector_shift]
			center_detector=[image_width,self.detector_shift]
			
			upper_source=[-image_width,self.detector_shift+0.5*detector_width]
			lower_source=[-image_width,self.detector_shift-0.5*detector_width]
			
			upper_detector=[image_width,self.detector_shift+0.5*detector_width]
			lower_detector=[image_width,self.detector_shift-0.5*detector_width]
			
			center_source=np.dot(A,center_source)
			center_detector=np.dot(A,center_detector)
			upper_source=np.dot(A,upper_source)
			lower_source=np.dot(A,lower_source)
			upper_detector=np.dot(A,upper_detector)
			lower_detector=np.dot(A,lower_detector)
						
			plot([center_source[0],center_detector[0]],[center_source[1] ,center_detector[1]],"g")
			
			plot([lower_source[0],lower_detector[0]],[lower_source[1],lower_detector[1]],"g")
			plot([upper_source[0],upper_detector[0]],[upper_source[1],upper_detector[1]],"g")
			
			plot([lower_detector[0],upper_detector[0]],[lower_detector[1],upper_detector[1]],"k")
			
			draw_circle=matplotlib.patches.Circle((0, 0), image_width/sqrt(2), color='r')
			
			gcf().gca().add_artist(draw_circle)	

			color=(1,1,0)
			draw_rectangle=matplotlib.patches.Rectangle(-0.5*np.array([image_width*self.shape[0]/np.max(self.shape),image_width*self.shape[1]/np.max(self.shape)]), image_width*self.shape[0]/np.max(self.shape),image_width*self.shape[1]/np.max(self.shape),facecolor=color, edgecolor=color)

			gcf().gca().add_artist(draw_rectangle)					

			draw_circle=matplotlib.patches.Circle((0, 0), image_width/2, color='b')
			
			gcf().gca().add_artist(draw_circle)	
			draw_circle=matplotlib.patches.Circle((0,0), image_width/10, color='k')			
			gcf().gca().add_artist(draw_circle)	
			
			maxsize=sqrt(image_width**2+detector_width**2)
			xlim([-maxsize,maxsize])
			ylim([-maxsize,maxsize])
		show()


"""Executes Landweberiteration for projectionmethod
Input:		sinogram 	... pyopencl.array of sinogram for which to compute the inverse projection transform
			number_iterations... int number of iterations to execute
			w ...	float representing the relaxation parameter (w<1 garanties convergence)
"""
def Landweberiteration(sinogram,PS,number_iterations=100,w=1):
	norm_estimate=normest(PS)
	w=sinogram.dtype.type(w/norm_estimate**2)	

	sinonew=sinogram+0.
	U=w*backprojection(None,sinonew,PS)
	Unew=clarray.zeros(PS.queue,U.shape,dtype=sinogram.dtype, order='F')
	
	for i in range(number_iterations):
		sinonew=forwardprojection(sinonew,Unew,PS,wait_for=Unew.events)-sinogram
		Unew=Unew-w*backprojection(U,sinonew,PS,wait_for=sinonew.events)	
	return Unew



def updload_bufs(projection_settings,mydtype):
	if projection_settings.buf_upload[mydtype]==0:
		ofs=projection_settings.ofs_buf[mydtype]
		ofs_buf = cl.Buffer(projection_settings.queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
		cl.enqueue_copy(projection_settings.queue, ofs_buf, ofs.data).wait()
		
		geometry_information=projection_settings.geometry_information[mydtype]
		geometry_buf=cl.Buffer(projection_settings.queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
		cl.enqueue_copy(projection_settings.queue, geometry_buf, geometry_information.data).wait()
		
		if projection_settings.geometry=="FAN":
			sdpd=projection_settings.sdpd_buf[mydtype]
			sdpd_buf = cl.Buffer(projection_settings.queue.context, cl.mem_flags.READ_ONLY, sdpd.nbytes)	
			cl.enqueue_copy(projection_settings.queue, sdpd_buf, sdpd.data).wait()
			projection_settings.sdpd_buf[mydtype]=sdpd_buf

		projection_settings.ofs_buf[mydtype]=ofs_buf
		projection_settings.geometry_information[mydtype]=geometry_buf
		projection_settings.buf_upload[mydtype]=1




def CG_iteration(sinogram,PS,epsilon,x0,number_iterations=100,relaunch=True):
	x=x0
	d=sinogram-forwardprojection(None,x,PS,wait_for=x.events)
	p=backprojection(None,d,PS,wait_for=d.events)
	q=clarray.empty_like(d,PS.queue)
	snew=p+0
		
	angle_weights=clarray.reshape(PS.angle_weights,[1,len(PS.angle_weights)])
	angle_weights=np.ones(sinogram.shape)*angle_weights
	angle_weights=clarray.to_device(PS.queue,require(angle_weights,sinogram.dtype,'F'))

	for k in range(0,number_iterations):
		residual=np.sum(clarray.vdot(snew,snew).get())**0.5
	#	print(str(k)+"-th iteration with risidual",residual, "relative orthogonality of the residuals",clarray.vdot(snew,sold)/(clarray.vdot(snew,snew)**0.5*clarray.vdot(sold,sold)**0.5))
		if  residual<epsilon:
			break
		
		sold=snew+0.	
		forwardprojection(q,p,PS,wait_for=p.events)
		alpha=x.dtype.type( PS.delta_x**2/(PS.delta_s)*(clarray.vdot(sold,sold)/clarray.vdot(q*angle_weights,q)).get())
		x=x+alpha*p;		d=d-alpha*q;
		backprojection(snew,d,PS ,wait_for=d.events)
		beta= (clarray.vdot(snew,snew)/clarray.vdot(sold,sold)).get()
		p=snew+beta*p		
		
		if beta>1 and relaunch==True:
			print("relaunch at", k)
			d=sinogram-forwardprojection(None,x,PS,wait_for=x.events)
			p=backprojection(None,d,PS,wait_for=d.events)
			q=clarray.empty_like(d,PS.queue)			
			snew=p+0.		
	return x

