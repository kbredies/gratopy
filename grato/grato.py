import os
from numpy import *
import numpy as np
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray


CL_FILES = ["radon.cl", "fanbeam.cl"]

PARALLEL = 1
RADON = 1
FANBEAM = 2
FAN = 2


###########
## Programm created from the gpu_code 
class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build()
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel

def forwardprojection(img, projection_settings, sino=None, wait_for=[]):
    """Performs the forward projection (either for the Radon or the
    fanbeam transform) of a given image using the given projection
    settings.

    **Parameters**
        img : pyopencl.Array 
            The image to be transformed, dimensions Nx x Ny x Nz

        projection_settings : grato.ProjectionSettings
            The instance of the *ProjectionSettings* class that contains
            transform type and geometry parameters.

        sino : pyopencl.Array, optional
            The array in which the result of transformation will be saved. 
            dimensions are Ns x Na x Nz
            If *None*, a new array will be created.

        wait_for : pyopencl.Event list, optional
            The events to wait for before performing the computation.

    **Returns**
        sino : pyopencl.Array
            The projection image, coincides with input sino if given.

    **Notes**
        The forward projection can be performed for single or double
        precision arrays. The data type of *img* and *sino* have to coincide.
        It respects any combination of *C* and *F* contiguous arrays.
        The OpenCL events associated with the transform will be appended
        to the output *sino*. In case the output array is created, it will
        use the allocator of *img*.

"""
    
    # initialize new sinogram if no sinogram is yet given
    if sino is None:
        z_dimension=tuple()
        if len(img.shape)>2:
            z_dimension=(img.shape[2],)
        sino = clarray.zeros(projection_settings.queue, projection_settings.sinogram_shape+z_dimension,
				dtype=img.dtype, order={0:'F',1:'C'}[img.flags.c_contiguous], allocator=img.allocator)    

    # perform projection operation
    function = projection_settings.forwardprojection
    function(sino, img, projection_settings, wait_for=wait_for)
    return sino


def backprojection(sino, projection_settings, img=None, wait_for=[]):
    """Performs the backprojection (either for the Radon or the
    fanbeam transform) of a given sinogram  using the given projection
    settings.
    
	**Parameters**
		sino : pyopencl.Array
			 sinogram to be backprojected, dimensions are Ns x Na x Nz
		
		projection_settings : grato.ProjectionSettings
            The instance of the *ProjectionSettings* class that contains
            transform type and geometry parameters.
		
		img : pyopencl.Array
			The array in which the result of backprojection will be saved.
			dimensions are Ns x Na x Nz 
			If equal *None* is given, a new array will be created.

        wait_for : pyopencl.Event list, optional
            The events to wait for before performing the computation.

	**Returns**	
		img:  pyopencl.Array 
			backprojection of sinogram, coincides with input img if given. 
	**Notes**
		The forward projection can be performed for single or double
		precision arrays. The data type of *img* and *sino* have to coincide.
		It respects any combination of *C* and *F* contiguous arrays.
		The OpenCL events associated with the transform will be appended
		to the output *img*. In case the output array is created, it will
		use the allocator of *sino*.
		
    """ 
        
    # initialize new img (to save backprojection in) if none is yet given
    if img is None:
        z_dimension=tuple()
        if len(sino.shape)>2:
            z_dimension=(sino.shape[2],)
        img = clarray.zeros(projection_settings.queue, projection_settings.shape+z_dimension, dtype=sino.dtype, order={0:'F',1:'C'}[sino.flags.c_contiguous], allocator=sino.allocator)    

    #execute corresponding backprojection operation
    function = projection_settings.backprojection
    function(img, sino, projection_settings, wait_for=wait_for)
    return img


def radon(sino, img, projection_settings, wait_for=[]):
    """Performs the Radon transform of a given image using on the
     given projection_settings 
     
     **Parameters**
     
		sino: pyopencl.Array
			sinogram to be computed (Ns x Na x Nz)

		img: pyopencl.Array
			img to be transformed (Nx x Ny x Nz)

		projection_settings : grato.ProjectionSettings
            The instance of the *ProjectionSettings* class that contains
            transform type and geometry parameters. Must be of parallel type

        wait_for : pyopencl.Event list, optional
            The events to wait for before performing the computation.
	
	**Returns**
		event corresponding to the computation of the Radon transform
		(which is also added to the sino events).
		
    """         
    
    #Ensure correct geometry is used
    assert(projection_settings.is_parallel), ("the projection_settings ist\
		not parallel setting, make sure to use geometry='parallel'  in the \
		projection_setting ")
	

    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
            +str(sino.dtype)+" and "+str(img.dtype))
    
    dtype=sino.dtype
    projection_settings.ensure_dtype(dtype)
    ofs_buf=projection_settings.ofs_buf[dtype] 
    geometry_information=projection_settings.geometry_information[dtype]

    #Choose function with approrpiate dtype
    function = projection_settings.functions[(dtype,sino.flags.c_contiguous,img.flags.c_contiguous)]
    myevent=function(sino.queue, sino.shape, None,
                     sino.data, img.data, ofs_buf,
                     geometry_information,
                     wait_for=img.events+sino.events+wait_for)

    sino.add_event(myevent)
    return myevent
    
def radon_ad(img, sino, projection_settings, wait_for=[]):
    """Performs the Radon backprojection transform of a given image using on the
     given projection_settings 
     
     **Parameters**
      	img: pyopencl.Array
    		img to be computed (Nx x Ny x Nz)
			
    	sino: pyopencl.Array
    		sino to be transformed (Ns x Na x Nz)
			
    	projection_settings : grato.ProjectionSettings
            The instance of the *ProjectionSettings* class that contains
            transform type and geometry parameters. Must be of parallel type
 
        wait_for : pyopencl.Event list, optional
            The events to wait for before performing the computation.
	
    **Returns**
    	event corresponding to the computation of the backprojection
    	(which is also added to the img events).
    	
    """         

	#Ensure correct geometry is used
    assert(projection_settings.is_parallel), ("the projection_settings ist\
    	not parallel setting, make sure to use geometry='parallel'  in the \
    	projection_setting ")

    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
            +str(sino.dtype)+" and "+str(img.dtype))
            
    dtype=sino.dtype
    projection_settings.ensure_dtype(dtype)   
    ofs_buf=projection_settings.ofs_buf[dtype] 
    geometry_information=projection_settings.geometry_information[dtype]

    #Choose function with approrpiate dtype 
    function = projection_settings.functions_ad[(dtype,img.flags.c_contiguous,sino.flags.c_contiguous)]
    myevent = function(img.queue, img.shape, None,
                        img.data, sino.data, ofs_buf,
                        geometry_information,wait_for=img.events+sino.events+wait_for)
    img.add_event(myevent)
    return myevent


def radon_struct(queue, shape, angles, n_detectors=None,
             detector_width=2.0, image_width=2.0,detector_shift=0.0,fullangle=True):
    """Creates the structure of radon geometry required for radontransform and its adjoint
    
    **Parameters**
    
        queue: pyopencl.queue
			queue object corresponding to a context in pyopencl
			
        shape: tuple of two integers
			the Nx,Ny number of pixels of the object (image) in pixels
        
        angles: integer or list[float], list[list[float]]  
			number of angles (uniform in [0,pi[, or list of angles considered,
			or a list of list of angles considered (last used when for multiple
			limited angle sets, see *fullangle*.)
        
        n_detectors:  integer, optional 
			Number of detectors considered, i.e., resolution of the projections
			(default None chooses n_detectors =sqrt(Nx+Ny))
        
        detector_width: float , optional  
			physical length of the detector (default=2.)
        
        image_width: float, optional
            physical diameter of the object in question (side-length of the image square)
            (default=2.)
        
        detector_shift:  float, optional
			physical size of shift of detector and corresponding detector-pixel positions
			(default=0.0)
        
        fullangle: Boolean, optional
			 Boolean representing wether the entire [0,pi) is represented by the angles
			 (Default = True). Impacts the weighting in the backprojection. If the given angles
			 only represent a discretization of a real subset of [0,pi[ ,
			 i.e., limited angle situation.

	**Results**
        Ofs_Dict: dict[np.dtype --> numpy.array]
			dictionary containing the relevant angular information in 
			numpy.array form for respective data-type numpy.dtype(float32)
			or numpy.dtype(float64)
			Arrays have dimension 8 x Na with 
			0 ... weighted cos()
			1 ... weighted sin()
			2 ... detector offset
			3 ... inverse of cos
			4 ... Angular weights			
        shape: tuple of integers
			(Nx,Ny) pixels in the image 
        sinogram_shape: (tuple) 
			(Ns,Na) pixels in the sinogram.
        Geo_Dict: [np.dtype --> numpy.array]
			array containing information [delta_x,delta_s,Nx,Ny,Ni,Nj]
		angles_diff: numpy.array
			same values as in ofs_dict[4] representing the weight 
			associated to the angles (i.e. size of detector-pixels)
			
    """
    
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
        angles_sorted=angles[angles_index]  %(np.pi)
        angles_sorted=np.array(hstack([-np.pi+angles_sorted[-1],angles_sorted,angles_sorted[0]+np.pi]))
        angles_diff= 0.5*(abs(angles_sorted[2:len(angles_sorted)]-angles_sorted[0:len(angles_sorted)-2]))
        angles_diff=np.array(angles_diff)
        angles_diff=angles_diff[angles_index]
    else:##Act as though first/last angles width is equal to the distance from the second/second to last angle  
        angles_diff=[]
        for j in range(len(angles_section)-1):
            current_angles=angles[angles_section[j]:angles_section[j+1]]
            current_angles_index = np.argsort(current_angles%(np.pi)) 
            current_angles=current_angles[current_angles_index] %(np.pi)

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
    angles_sorted=angles[angles_index]  %(np.pi)
    
    #Also write basic information to gpu 
    [Nx,Ny]=shape
    [Ni,Nj]= [nd,len(angles)]
    delta_x=image_width/float(max(Nx,Ny))
    delta_s=float(detector_width)/nd

    
    Geo_dict={}
    Ofs_dict={}
    for dtype in [np.dtype('float64'),np.dtype('float32')]:
        #Save angular information into the ofs buffer
        ofs = zeros((8, len(angles)), dtype=dtype, order='F')
        ofs[0,:] = X; ofs[1,:] = Y; ofs[2,:] = offset; ofs[3,:] = Xinv
        ofs[4,:]=angles_diff
        Ofs_dict[dtype]=ofs
        
        geometry_info = np.array([delta_x,delta_s,Nx,Ny,Ni,Nj],dtype=dtype,order='F')
        Geo_dict[dtype]=geometry_info
    
    

    sinogram_shape = (nd, len(angles))
    return (Ofs_dict, shape, sinogram_shape, Geo_dict,angles_diff)
 

def fanbeam(sino, img, projection_settings, wait_for=[]):
    """Performs the Fanbeam transform of a given image using on the
     given projection_settings 
     
     **Parameters**     
		sino: pyopencl.Array
			sinogram to be computed (Ns x Na x Nz)

		img: pyopencl.Array
			img to be transformed (Nx x Ny x Nz)

		projection_settings : grato.ProjectionSettings
            The instance of the *ProjectionSettings* class that contains
            transform type and geometry parameters. Must be of parallel type

        wait_for : pyopencl.Event list, optional
            The events to wait for before performing the computation.
	
	**Returns**
		event corresponding to the computation of the fanbeam transform
		(which is also added to the sino events).
		
    """   
	  
	#Ensure correct geometry is used       
    assert(projection_settings.is_fan), ("the projection_settings ist\
		not parallel setting, make sure to use geometry='fan'  in the \
		projection_setting ")

    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype),("sinogram and image do not share common data type: "\
        +str(sino.dtype)+" and "+str(img.dtype))    
         
    dtype=sino.dtype

    projection_settings.ensure_dtype(dtype)   
    ofs_buf=projection_settings.ofs_buf[dtype]
    sdpd_buf=projection_settings.sdpd_buf[dtype]
    geometry_information=projection_settings.geometry_information[dtype]
    
    #Choose function with approrpiate dtype
    function = projection_settings.functions[(dtype,sino.flags.c_contiguous,img.flags.c_contiguous)]
    myevent=function(sino.queue, sino.shape, None,
                        sino.data, img.data, ofs_buf, sdpd_buf,
                        geometry_information,
                        wait_for=img.events+sino.events+wait_for)
    sino.add_event(myevent)
    return myevent

def fanbeam_ad(img, sino, projection_settings, wait_for=[]):
    """Performs the fanbeam backprojection of a given image using on the
    given projection_settings 
     
    **Parameters**
        img: pyopencl.Array
            img to be computed (Nx x Ny x Nz)
			
        sino: pyopencl.Array
            sino to be transformed (Ns x Na x Nz)
			
        projection_settings : grato.ProjectionSettings
            The instance of the *ProjectionSettings* class that contains
            transform type and geometry parameters. Must be of parallel type

        wait_for : pyopencl.Event list, optional
            The events to wait for before performing the computation.
	
    **Returns**
        event corresponding to the computation of the backprojection
        (which is also added to the img events).
        
    """         
    
    
    #Ensure correct geometry is used
    assert(projection_settings.is_fan), ("the projection_settings ist \
    not parallel setting, make sure to use geometry='fan'  in the \
    projection_setting ")

    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
            +str(sino.dtype)+" and "+str(img.dtype))
            
    dtype=sino.dtype
    
    projection_settings.ensure_dtype(dtype)   
    ofs_buf=projection_settings.ofs_buf[dtype]; 
    sdpd_buf=projection_settings.sdpd_buf[dtype]
    geometry_information=projection_settings.geometry_information[dtype]

    function = projection_settings.functions_ad[(dtype,img.flags.c_contiguous,sino.flags.c_contiguous)]
    myevent = function(img.queue, img.shape, None,
                       img.data,sino.data, ofs_buf,sdpd_buf,geometry_information,
                       wait_for=img.events+sino.events+wait_for)
    img.add_event(myevent)
    return myevent

def fanbeam_struct_gpu(queue, shape, angles, detector_width,
                   source_detector_dist, source_origin_dist,
                   n_detectors=None, detector_shift = 0.0,
                   image_width=None,midpointshift=[0,0], fullangle=True):   
    """Creates the structure of fanbeam geometry required for fanbeam transform and its adjoint
    **Parameters**
    
        queue: pyopencl.queue
			queue object corresponding to a context in pyopencl
			
        shape: tuple of two integers
			the Nx,Ny number of pixels of the object (image) in pixels
        
        angles: integer or list[float], list[list[float]]  
			number of angles (uniform in [0,pi[, or list of angles considered,
			or a list of list of angles considered (last used when for multiple
			limited angle sets, see *fullangle*.)
		
		detector_width: float , optional  
			physical length of the detector (default=2.)

        
        source_detector_dist: float
    		the physical (normal) distance from source to the detector 
        
        source_origin_dist: float
			Physical distance for source to the origin (center of rotation)
			
        
        n_detectors:  integer, optional 
			Number of detectors considered, i.e., resolution of the projections
			(default None chooses n_detectors =sqrt(Nx+Ny))
		
        detector_shift:  float, optional
			physical size of shift of detector and corresponding detector-pixel positions
			(default=0.0)
        
        image_width: float, optional
            Physical diameter of the object in question (side-length of the image square)
            (default=*None*) if not given, the algorithm determins an image_width
            such that a every ball (inside the image) is completely captured 
            by the observed projections.
        
        midpoint_shift: list[float] of length 2
			shifts the image midpoint away from the center of rotation(default [0,0]
       
        fullangle: Boolean, optional
			 Boolean representing wether the entire [0,pi) is represented by the angles
			 (Default = True). Impacts the weighting in the backprojection. If the given angles
			 only represent a discretization of a real subset of [0,pi[ ,
			 i.e., limited angle situation.
	**Results**
        shape: tuple of integers
			(Nx,Ny) pixels in the image 
        sinogram_shape: (tuple) 
			(Ns,Na) pixels in the sinogram.
        Ofs_Dict: dict[np.dtype --> numpy.array]
			dictionary containing the relevant angular information in 
			numpy.array form for respective data-type numpy.dtype(float32)
			or numpy.dtype(float64)
			Arrays have dimension 8 x Na with 
			0 1 ... vector along detector-direction with length delta_s
			2 3	... vector from source vector connecting source to center of rotation
			4 5 ... vector connecting the origin to the detectorline
			6   ... Angular weights		SDPD_Dict[np.dtype --> numpy.array]
			dictionary containing numpy.array with regard to dtype 
			reprsenting the values sqrt( s^2+R^2) for the weighting in 
			the Fanbeam transform (weighted by delta_ratio)
        image_width: float
			the observed image_width, is equal to the input if given,
			or the determined suitable image_width if *None* is give
			(see parameter image_width) 
			
        
        Geo_Dict: [np.dtype --> numpy.array]
			array containing information [delta_x,delta_s,Nx,Ny,Ni,Nj]
		angles_diff: numpy.array
			same values as in ofs_dict[4] representing the weight 
			associated to the angles (i.e. size of detector-pixels)
			
    """

					   

    
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
        angles_sorted=angles[angles_index]  %(2*np.pi)
        angles_sorted=np.array(hstack([-2*np.pi+angles_sorted[-1],angles_sorted,angles_sorted[0]+2*np.pi]))
        angles_diff= 0.5*(abs(angles_sorted[2:len(angles_sorted)]-angles_sorted[0:len(angles_sorted)-2]))
        angles_diff=np.array(angles_diff)
        angles_diff=angles_diff[angles_index]
    else:##Act as though first/last angles width is equal to the distance from the second/second to last angle  
        angles_diff=[]
        for j in range(len(angles_section)-1):
            current_angles=angles[angles_section[j]:angles_section[j+1]]
            current_angles_index = np.argsort(current_angles%(2*np.pi)) 
            current_angles=current_angles[current_angles_index] %(2*np.pi)

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
    for dtype in [np.dtype('float64'),np.dtype('float32')]:
        #Angular Information
        ofs = zeros((8, len(angles)), dtype=dtype, order='F')
        ofs[0,:] = XD; ofs[1,:] = YD
        ofs[2,:]=Qx; ofs[3,:]=Qy
        ofs[4,:]=Dx0; ofs[5]=Dy0
        ofs[6]=angles_diff
        Ofs_dict[dtype]=ofs
        
        
        # Determine source detectorpixel-distance (=sqrt(R+xi**2)) for scaling
        xi=(np.arange(0,nd)- midpoint_detectors)*detector_width/nd
        source_detectorpixel_distance= sqrt((xi)**2+source_detector_dist**2)
        source_detectorpixel_distance=np.array(source_detectorpixel_distance,dtype=dtype,order='F')
        sdpd = zeros(( 1,len(source_detectorpixel_distance)), dtype=dtype, order='F')
        sdpd[0,:]=source_detectorpixel_distance[:]
        Sdpd_dict[dtype]=sdpd
        
        #collect various geometric information necessary for computations
        geometry_info=np.array([source_detector_dist,source_origin_dist,detector_width/nd,
            midpoint_x,midpoint_y,midpoint_detectors,shape[0],shape[1],sinogram_shape[0],
            sinogram_shape[1],image_width/float(max(shape))],dtype=dtype,order='F')
        Geo_dict[dtype]=geometry_info
        
    return (shape,sinogram_shape,Ofs_dict,Sdpd_dict,image_width,Geo_dict,angles_diff)





def create_code():    
    """
    reads the c code associated with the opencl functions
	
	
    **Returns**
    	total_code: string
			string associated with the opencl code
    """
    total_code=""
    for file in CL_FILES:
        textfile=open(os.path.join(os.path.abspath(os.path.dirname(__file__)), file))
        code_template=textfile.read()
        textfile.close()
        
        for dtype in ["float","double"]:
            for order1 in ["f","c"]:
                for order2 in ["f","c"]:
                    total_code+=code_template.replace("\my_variable_type",dtype).replace("\order1",order1).replace("\order2",order2)
                    
    return total_code

def upload_bufs(projection_settings, dtype):
    """Uploads arrays  (ofs,sdpd,geometry_information) of a projection_setting 
    for a given datatype numpy.dtype(float32), or numpy.dtype(float32)
    (in case they were not uploaded before) to the OpenCL device
	
    **Parameters**

    	projection_settings : grato.ProjectionSettings
            The instance of the *ProjectionSettings* class that contains
            transform type and geometry parameters.

        dtype: numpy.dtype
    		The data_type for which corresponding array shall be uploaded
    **Note**
    	The function has no returns, but internally updates the arrays
    	inside of the ofs_buf,sdpd_buf, and geometry_information (dictionary)
    	attribute from numpy.array to pyopencl.Buffer.
    	
    """
	
    ofs=projection_settings.ofs_buf[dtype]
    ofs_buf = cl.Buffer(projection_settings.queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
    cl.enqueue_copy(projection_settings.queue, ofs_buf, ofs.data).wait()
        
    geometry_information=projection_settings.geometry_information[dtype]
    geometry_buf=cl.Buffer(projection_settings.queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
    cl.enqueue_copy(projection_settings.queue, geometry_buf, geometry_information.data).wait()
        
    if projection_settings.is_fan:
        sdpd=projection_settings.sdpd_buf[dtype]
        sdpd_buf = cl.Buffer(projection_settings.queue.context, cl.mem_flags.READ_ONLY, sdpd.nbytes)    
        cl.enqueue_copy(projection_settings.queue, sdpd_buf, sdpd.data).wait()
        projection_settings.sdpd_buf[dtype]=sdpd_buf

    projection_settings.ofs_buf[dtype]=ofs_buf
    projection_settings.geometry_information[dtype]=geometry_buf

class ProjectionSettings():
    """Class saving all relevant projection information, which is always used to compute projections

    **atributes**:
        queue: pyopencl.queue
			queue associated to the OpenCL-context in question
        geometry: int
			Number 1 representing parallel beam setting, 2 fanbeam setting
		prg: grato.Program
			programm containing the opencl kernels for parallel execution
        shape: tuple of two integers
            (Nx,Ny) dimensions of the image
        sinogram_shape: tuple of two integers
		    (Ns,Na) dimension of the sinogram (i.e., number of detectors and angles)
        angles: list[float]
             List of all angles considered for the projection
        n_detector: int
             number of detectors used
        n_angles: int
             number of angles used
        R: float
            the physical (normal) distance from source to the detector
        RE: float
			Physical distance for source to the origin (center of rotation)
        image_width: float 
			the size of the image (i.e. the (larger) side length of the rectangle image)
			i.e. the diameter of the object captured by image
			(for parallel-beam default value 2, for fanbeam chosen suitably to capture all rays)
        detector_width: float
			physical length of the detector (default=2.) 
        detector_shift: float 
			physical shift of the detector and correspond detector offsets
			(default=0)
        midpoint_shift: list[array] length 2
			vector shifting the image away from the center of rotation
        fullangle: Boolean, optional
			 Boolean representing wether the entire [0,pi[ (or [0,2pi[ for fanbeam)
			 is represented by the given angles, i.e., not a limited angle setting
			 (Default = True). Impacts the weighting in the backprojection.
			 fullangle should be set false if the given angles
			 only represent a discretization of a strict subset of [0,pi[ (or [0,2pi[
         
        Ofs_Dict: dict[np.dtype --> {numpy.array or pyopencl.array}] 
			dictionary containing the relevant angular information in 
			array form for respective data-type numpy.dtype(float32)
			or numpy.dtype(float64), switching from numpy to cl.array
			when the information is uploaded to the OpenCL device
			(which is done automatically when required)
			*Radon transformation*
				Arrays have dimension 8 x Na with 
				0 ... weighted cos()
				1 ... weighted sin()
				2 ... detector offset
				3 ... inverse of cos
				4 ... Angular weights
			*Fanbeam transform*
				Arrays have dimension 8 x Na with 
				0 1 ... vector along detector-direction with length delta_s
				2 3	... vector from source vector connecting source to center of rotation
				4 5 ... vector connecting the origin to the detectorline
				6   ... Angular weights

		SDPD_Dict[np.dtype --> {numpy.array or pyopencl.array} ]
			dictionary containing arrays with regard to dtype 
			reprsenting the values sqrt( s^2+R^2) for the weighting in 
			the Fanbeam transform (weighted by delta_ratio)
			For parallel beam setting this atribute is not set.

        delta_x: float
            physical length/width of an image pixel
        delta_s: float
			physical length of a detector-pixel
        delta_ratio: float
            ratio delta_s/delta_x
    **methods**:
            show_geometry: shows geometry visually for a given angle
            
            """

    def __init__(self, queue,geometry, img_shape, angles, n_detectors=None, 
                    detector_width=2.0, detector_shift=0.0,
                    midpoint_shift=[0,0], R=None, RE=None,
                    image_width=None, fullangle=True):        
        """
        initialize ProjectionSettings variable
        
        **Parameters**
			queue:pyopencl.queue
				queue associated to the OpenCL-context in question
			geometry: int
				1 for Radon transform or 2 for fanbeam Transform
				Grato defines RADON with the value 1 and FANBEAM with 2,
				i.e., one can give as argument the Variable RADON or FANBEAM
			img_shape: (tuple of two integers)
				see class *ProjectionSettings*
			angles: integer or list[float], list[list[float]]  
				number of angles (uniform in [0,pi[, or list of angles considered,
				or a list of list of angles considered (last used when for multiple
				limited angle sets, see *fullangle*.)
			n_detectors: int
				number of detector-pixels considered
			detector_width: float
				see class *ProjectionSettings*
			detector_shift: float 
				see class *ProjectionSettings*
			midpoint_shift: list of two floats
				see class *ProjectionSettings*
			R: float
				see class *ProjectionSettings*
			RE: float
				see class *ProjectionSettings*
			image_width: float
				see class *ProjectionSettings*
			fullangle: boolean
				see class *ProjectionSettings*
		**Returns**
			self: grato.ProjectionSettings
				ProjectionSettings with the corresponding atributes
		"""	
			
     
        
        self.geometry=geometry
        self.queue=queue
        
        self.adjusted_code=create_code()

        self.prg = Program(queue.context,self.adjusted_code)
        self.image_width=image_width
        
        if isscalar(img_shape):
            img_shape=(img_shape, img_shape)
        
        if len(img_shape)>2:
            img_shape=img_shape[0:2]
        self.shape=img_shape
                
        if self.geometry not in [RADON, PARALLEL, FAN, FANBEAM]:
            raise("unknown projection_type, projection_type must be PARALLEL or FAN")
        if self.geometry in [RADON, PARALLEL]:
            self.is_parallel=True
            self.is_fan=False
            
            self.forwardprojection = radon
            self.backprojection = radon_ad
            
            float32 = np.dtype('float32')
            float64 = np.dtype('float64')
            self.functions={(float32,0,0):self.prg.radon_float_ff,
                            (float32,1,0):self.prg.radon_float_cf,
                            (float32,0,1):self.prg.radon_float_fc,
                            (float32,1,1):self.prg.radon_float_cc,
                            (float64,0,0):self.prg.radon_double_ff,
                            (float64,1,0):self.prg.radon_double_cf,
                            (float64,0,1):self.prg.radon_double_fc,
                            (float64,1,1):self.prg.radon_double_cc}
            self.functions_ad={(float32,0,0):self.prg.radon_ad_float_ff,
                               (float32,1,0):self.prg.radon_ad_float_cf,
                               (float32,0,1):self.prg.radon_ad_float_fc,
                               (float32,1,1):self.prg.radon_ad_float_cc,
                               (float64,0,0):self.prg.radon_ad_double_ff,
                               (float64,1,0):self.prg.radon_ad_double_cf,
                               (float64,0,1):self.prg.radon_ad_double_fc,
                               (float64,1,1):self.prg.radon_ad_double_cc}

            
        if self.geometry in [FAN, FANBEAM]:
            self.is_parallel=False
            self.is_fan=True

            self.forwardprojection = fanbeam
            self.backprojection = fanbeam_ad

            float32 = np.dtype('float32')
            float64 = np.dtype('float64')
            self.functions = {(float32,0,0):self.prg.fanbeam_float_ff,
                              (float32,1,0):self.prg.fanbeam_float_cf,
                              (float32,0,1):self.prg.fanbeam_float_fc,
                              (float32,1,1):self.prg.fanbeam_float_cc,
                              (float64,0,0):self.prg.fanbeam_double_ff,
                              (float64,1,0):self.prg.fanbeam_double_cf,
                              (float64,0,1):self.prg.fanbeam_double_fc,
                              (float64,1,1):self.prg.fanbeam_double_cc}
            self.functions_ad = {(float32,0,0):self.prg.fanbeam_ad_float_ff,
                                 (float32,1,0):self.prg.fanbeam_ad_float_cf,
                                 (float32,0,1):self.prg.fanbeam_ad_float_fc,
                                 (float32,1,1):self.prg.fanbeam_ad_float_cc,
                                 (float64,0,0):self.prg.fanbeam_ad_double_ff,
                                 (float64,1,0):self.prg.fanbeam_ad_double_cf,
                                 (float64,0,1):self.prg.fanbeam_ad_double_fc,
                                 (float64,1,1):self.prg.fanbeam_ad_double_cc}

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
        
        if isinstance(angles[0], list) or isinstance(angles[0],np.ndarray):
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

        self.buf_upload={}
                            
        if self.is_fan:
            assert( (R!=None)*(RE!=None)),"For the Fanbeam geometry \
                you need to set R (the normal distance from source to detector)\
                 and RE (distance from source to coordinate origin which is the \
                 rotation center) "
                             
            struct=fanbeam_struct_gpu(self.queue,self.shape, self.angles, 
                    self.detector_width, R, self.RE,self.n_detectors, self.detector_shift,
                    image_width,self.midpoint_shift, self.fullangle)
            
            self.ofs_buf=struct[2]
            self.sdpd_buf=struct[3]
            self.image_width=struct[4]
            self.geometry_information=struct[5]
            
            self.angle_weights=struct[6]
            
            self.delta_x=self.image_width/max(img_shape)            
            self.delta_s=detector_width/n_detectors
            self.delta_ratio=self.delta_s/self.delta_x
            
    
        if self.is_parallel:
            if image_width==None:
                self.image_width=2.
            
            struct=radon_struct(self.queue,self.shape, self.angles, 
                n_detectors=self.n_detectors,
                detector_width=self.detector_width, image_width= self.image_width,
                detector_shift=self.detector_shift,fullangle=self.fullangle, )
                
            self.ofs_buf=struct[0]
                        
            self.delta_x=self.image_width/max(self.shape)
            self.delta_s=self.detector_width/self.n_detectors
            self.delta_ratio=self.delta_s/self.delta_x  
            
            self.geometry_information=struct[3]
            self.angle_weights=struct[4]

    def ensure_dtype(self, dtype):
        if not dtype in self.buf_upload:
            upload_bufs(self, dtype)
            self.buf_upload[dtype] = 1

    def show_geometry(self,angle=0):
        """ 
        Depict geometry associated with to the ProjectionSetting
		
        **Parameters**
        	angle: float
        		angle in Radian from which projection is considered
		
        """
		
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



    def normest(self, number_of_iterations=50, dtype=np.dtype('float32'),
              allocator=None):
        """
        Determine the operator norm of the projectionmethod via  poweriteration
        
        **Parameters**
        number_of_iterations: integer
		    number of iterations for the poweriteration to be used (default 50)
		dtype: np.dtype
			dtye for which to compute the poweriteration (though this should not
			significantly impact) the result
        **Results
		    norm: float
			    an estimate of the operator norm (spectral norm/ matrix norm)
				
		"""

        queue=self.queue
    
        #random starting point
        img = clarray.to_device(queue, require((random.randn(*self.shape)), dtype, 'F'), allocator=allocator)
        sino=forwardprojection(img, self)
    
        #power_iteration
        for i in range(number_of_iterations):
            normsqr = float(clarray.sum(img).get())
    
            img /= normsqr
            forwardprojection(img, self, sino=sino)
            backprojection(sino, self, img=img) 
        return sqrt(normsqr)

        
def landweber(sinogram, PS, number_iterations=100, w=1):
    """Executes Landweberiteration for projectionmethod
    
    **Parameters**    
        sinogram: pyopencl.array
            sinogram for which to compute the inverse projection transform
        PS: grato.ProjectionSetting
            The instance of the *ProjectionSettings* class that contains
            transform type and geometry parameters.
        number_iterations: int
              number of iterations to execute for the Landweber iteration
        w: float
            relaxation parameter (w<1 garanties convergence)
      
    **Returns**
        Unew: pyopencl.array:
		    Resulting reconstruction of Landweber iteration.
    
        
      """
    norm_estimate=PS.normest(allocator=sinogram.allocator)
    w=sinogram.dtype.type(w/norm_estimate**2)   

    sinonew=sinogram.copy()
    U=w*backprojection(sinonew, PS)
    Unew=clarray.zeros(PS.queue, U.shape, dtype=sinogram.dtype, order='F',
                       allocator=sinogram.allocator)
    
    for i in range(number_iterations):
        sinonew=forwardprojection(Unew, PS, sino=sinonew)-sinogram
        Unew=Unew-w*backprojection(sinonew, PS, img=U) 
    return Unew

def cg(sinogram, PS, x0,epsilon=0.0, number_iterations=100, restart=True):
    """Executes conjugated gradients for projectionmethod
    
    **Parameters**    
        sinogram: pyopencl.array
            sinogram for which to compute the inverse projection transform
        PS: grato.ProjectionSetting
            The instance of the *ProjectionSettings* class that contains
            transform type and geometry parameters.
        x0: pyopencl.array
            startpoint for iteration
        epsilon: float
            stopping criteria when resiudal<epsilon
        number_iterations: int
              number of iterations to execute for the Landweber iteration
        restart: boolean
            relaunch iteration to avoid numerical issues.
      
    **Returns**
        Unew: pyopencl.array:
		    Resulting reconstruction of conjugated gradients iteration.
    
        
      """
	
    x=x0
    d=sinogram-forwardprojection(x, PS)
    p=backprojection(d, PS)
    q=clarray.empty_like(d, PS.queue)
    snew=p+0
        
    angle_weights=clarray.reshape(PS.angle_weights,[1,len(PS.angle_weights)])
    angle_weights=np.ones(sinogram.shape)*angle_weights
    angle_weights=clarray.to_device(PS.queue, require(angle_weights, sinogram.dtype, 'F'),
                                    allocator=sinogram.allocator)

    for k in range(0,number_iterations):
        residual=np.sum(clarray.vdot(snew,snew).get())**0.5
    #   print(str(k)+"-th iteration with risidual",residual, "relative orthogonality of the residuals",clarray.vdot(snew,sold)/(clarray.vdot(snew,snew)**0.5*clarray.vdot(sold,sold)**0.5))
        if  residual<epsilon:
            break
        
        sold=snew+0.    
        forwardprojection(p, PS, sino=q)
        alpha=x.dtype.type(PS.delta_x**2/(PS.delta_s)*(clarray.vdot(sold,sold)/clarray.vdot(q*angle_weights,q)).get())
        x=x+alpha*p
        d=d-alpha*q
        backprojection(d, PS, img=snew)
        beta= (clarray.vdot(snew,snew)/clarray.vdot(sold,sold)).get()
        p=snew+beta*p       
        
        if beta>1 and restart==True:
            print("restart at", k)
            d=sinogram-forwardprojection(x, PS)
            p=backprojection(d, PS)
            q=clarray.empty_like(d, PS.queue)            
            snew=p+0.
    return x

