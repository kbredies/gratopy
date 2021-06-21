import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
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

def forwardprojection(img, projectionsettings, sino=None, wait_for=[]):
    """Performs the forward projection (either for the Radon or the
    fanbeam transform) of a given image using the given projection
    settings.

    :param img: The image to be transformed.
    :type img: :class:`pyopencl.Array` with dimensions compatible to projectionsettings
    :param projectionsettings: The instance of the *ProjectionSettings* 
        class that contains
        transform type and geometry parameters.
    :type projectionsettings: `grato.ProjectionSettings`
    :param sino: The array in which the result of transformation will be saved. 
        If *None*, a new array will be created and returned.
    :type sino: :class:`pyopencl.Array` with dimensions compatible to projectionsettings or None, optional
    :param wait_for: 
        The events to wait for before performing the computation in order to avoid race conditions, see `pyopencl.wait_for`
    :type wait_for: :class:`pyopencl.Event` list or None, optional
    :return: The sinogram created by the projection of the image. If the sinogram input is given, the same pyopencl array will be returned, though the values in its data are overwritten.
    :rtype: :class:`pyopencl.Array`

    The forward projection can be performed for single or double
    precision arrays. The dtype (precision) of *img* and *sino* (if given) have to coincide and the output will be of the same precision.
    It respects any combination of *C* and *F* contiguous arrays where output will be of the same contiguity as img if no sino is given.
    The OpenCL events associated with the transform will be appended
    to the output *sino*. In case the output array is created, it will
    use the allocator of *img*.
    """
    
    # initialize new sinogram if no sinogram is yet given
    if sino is None:
        z_dimension=tuple()
        if len(img.shape)>2:
            z_dimension=(img.shape[2],)
        sino = clarray.zeros(projectionsettings.queue, projectionsettings.sinogram_shape+z_dimension, dtype=img.dtype, order={0:'F',1:'C'}[img.flags.c_contiguous], allocator=img.allocator)    

    # perform projection operation
    function = projectionsettings.forwardprojection
    function(sino, img, projectionsettings, wait_for=wait_for)
    return sino


def backprojection(sino, projectionsettings, img=None, wait_for=[]):
    """
    Performs the backprojection (either for the Radon or the
    fanbeam transform) of a given sinogram  using the given projection 
    settings.
		    
    :param  sino: sinogram to be backprojected.
    :type sino: :class:`pyopencl.Array` with dimensions are (Ns, Na, Nz) compatible to projectionsettings

    :param projectionsettings: The instance of the *ProjectionSettings* 
            class that contains
            transform type and geometry parameters.
    :type projectionsettings: :class:`grato.ProjectionSettings`
		
    :param  img: The array in which the result of backprojection will be saved.
        If equal *None* is given, a new array will be created.
    :type img: :class:`pyoepncl.Array`  (with dimensions Ns x Na x Nz) compatible to projectionsettings or None
    
    :param wait_for: 
        The events to wait for before performing the computation in order to avoid race conditions.
    :type wait_for: :class:`pyopencl.Event` list or None.
    
    :return: The image associated to the backprojected sinogram, coinciding with the *img* input if given, though the values are overwriten.
    :rtype:  `pyopencl.Array` 

    The backprojection can be performed for single or double
    precision arrays. The data type of *img* and *sino* have to coincide, if no img is given the output precision coincides with sino's.
    It respects any combination of *C* and *F* contiguous arrays, where if no img is given the results contiguity coincides with sino's.
    The OpenCL events associated with the transform will be appended
    to the output *sino*. In case the output array is created, it will
    use the allocator of *img*.
		  
    """        
        
    # initialize new img (to save backprojection in) if none is yet given
    if img is None:
        z_dimension=tuple()
        if len(sino.shape)>2:
            z_dimension=(sino.shape[2],)
        img = clarray.zeros(projectionsettings.queue, projectionsettings.img_shape+z_dimension, dtype=sino.dtype, order={0:'F',1:'C'}[sino.flags.c_contiguous], allocator=sino.allocator)    

    #execute corresponding backprojection operation
    function = projectionsettings.backprojection
    function(img, sino, projectionsettings, wait_for=wait_for)
    return img


def radon(sino, img, projectionsettings, wait_for=[]):
    """
    Performs the Radon transform of a given image using on the given projectionsettings 

    :param sino: The sinogram to be computed (in which results are saved).
    :type sino: pyopencl.Array
    
    :param img: The image to be transformed.
    :type img: pyopencl.Array 

    :param projectionsettings: Contains transform type and geometry parameters.
    :type projectionsettings: float
 
     :param wait_for: 
        The events to wait for before performing the computation.
    :type wait_for: `pyopencl.Event` list or None

    :return: Event associated to computation of Radon transform 
        (which was added to the events of *sino*).
    :rtype:  `pyopencl.Event`
    
    """         

    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
            +str(sino.dtype)+" and "+str(img.dtype))
    
    dtype=sino.dtype
    projectionsettings.ensure_dtype(dtype)
    ofs_buf=projectionsettings.ofs_buf[dtype] 
    geometry_information=projectionsettings.geometry_information[dtype]

    #Choose function with approrpiate dtype
    function = projectionsettings.functions[(dtype,sino.flags.c_contiguous,img.flags.c_contiguous)]
    myevent=function(sino.queue, sino.shape, None,
                     sino.data, img.data, ofs_buf,
                     geometry_information,
                     wait_for=img.events+sino.events+wait_for)

    sino.add_event(myevent)
    return myevent
    
def radon_ad(img, sino, projectionsettings, wait_for=[]):
    """
    Performs the Radon backprojection  of a given sinogram using on the given projectionsettings 

    :param img: The image to be computed (in which results are saved).
    :type img: `pyopencl.Array`.
    
    :param sino: The sinogram to be transformed .
    :type sino: `pyopencl.Array`
    :param projectionsettings: The instance of the *ProjectionSettings* 
        class that contains
        transform type and geometry parameters.
    :type projectionsettings: `grato.ProjectionSettings`
 
     :param wait_for: 
        The events to wait for before performing the computation.
    :type wait_for: `pyopencl.Event` list or None

    :return: Event associated to computation of Radon backprojection 
        (which was added to the events of *img*).
    :rtype: `pyopencl.Event`
    
    """    
    
    my_function={(np.dtype('float32'),0,0):projectionsettings.prg.radon_ad_float_ff,
            (np.dtype('float32'),1,0):projectionsettings.prg.radon_ad_float_cf,
            (np.dtype('float32'),0,1):projectionsettings.prg.radon_ad_float_fc,
            (np.dtype('float32'),1,1):projectionsettings.prg.radon_ad_float_cc,
            (np.dtype('float'),0,0):projectionsettings.prg.radon_ad_double_ff,
            (np.dtype('float'),1,0):projectionsettings.prg.radon_ad_double_cf,
            (np.dtype('float'),0,1):projectionsettings.prg.radon_ad_double_fc,
            (np.dtype('float'),1,1):projectionsettings.prg.radon_ad_double_cc}


    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
            +str(sino.dtype)+" and "+str(img.dtype))
            
    dtype=sino.dtype
    projectionsettings.ensure_dtype(dtype)   
    ofs_buf=projectionsettings.ofs_buf[dtype] 
    geometry_information=projectionsettings.geometry_information[dtype]

    #Choose function with approrpiate dtype 
    function = projectionsettings.functions_ad[(dtype,img.flags.c_contiguous,sino.flags.c_contiguous)]
    myevent = function(img.queue, img.shape, None,
                        img.data, sino.data, ofs_buf,
                        geometry_information,wait_for=img.events+sino.events+wait_for)
    img.add_event(myevent)
    return myevent


def radon_struct(queue, img_shape, angles, n_detectors=None,
             detector_width=2.0, image_width=2.0,detector_shift=0.0,fullangle=True):
    """
    Creates the structure of radon geometry required for radontransform and its adjoint
	    
    :param queue: Queue object corresponding to a context in pyopencl to execute computations in
    :type queue: :class:`pyopencl.queue`
	
    :param img_shape: (Nx,Ny) the number of pixels of the image 
    :type img_shape: `tuple` of two integers 
			             
    :angles: number of angles (uniform in [0,pi[, or list of angles considered,
        or a list of list of angles considered (last used when for multiple
        limited angle sets, see *fullangle*.)
    :type angles: :class: integer or list[float], list[list[float]]  
			         
    :param n_detectors: Number of detectors considered, i.e., resolution of the projections.
        default None chooses n_detectors =sqrt(Nx+Ny)  
    :type n_detectors:  int or None, optional
			         
    :param detector_width: Physical length of the detector. default=2.0   
    :type detector_width: float, optional
			          
    :param image_width:  Physical diameter of the object in question (side-length of the image square)
        (default=2.)
    :type image_width: float, optional
			           		        
    :param detector_shift: Physical size of shift of detector and corresponding detector-pixel positions.
        (default=0.0)
    :type detector_shift: float, optional
			        
    :param fullangle:  True if entire [0,pi) is represented by the angles, False for a limited angle setting.
        Impacts the weighting in the backprojection. If the given angles,
        only represent a discretization of a real subset of [0,pi[,
        i.e., limited angle situation. 
    :type fullangle:  Boolean, optional.
 	

    :return:
        Ofs_Dict: dict[np.dtype --> numpy.array]
        Dictionary containing the relevant angular information in 
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
 
         sinogram_shape: (tuple) (Ns,Na) pixels in the sinogram.
 
        Geo_Dict: [np.dtype --> numpy.array]
        array containing information [delta_x,delta_s,Nx,Ny,Ni,Nj]
 
        angles_diff: numpy.array
        same values as in ofs_dict[4] representing the weight 
        associated to the angles (i.e. size of detector-pixels)
    :rtype: List (Ofs_Dict,shape,Geo_Dict,angles_diff)
    
    """
        
    #relative_detector_pixel_width is delta_s/delta_x
    relative_detector_pixel_width=detector_width/float(image_width)*max(img_shape)/n_detectors
    
    #When angles are None, understand as number of angles discretizing [0,pi]
    if isscalar(angles):    
        angles = linspace(0,pi,angles+1)[:-1]
    
    #Choosing the number of detectors as the half of the diagonal through the the image (in image_pixel scale)
    if n_detectors is None:
        nd = int(ceil(hypot(img_shape[0],img_shape[1])))
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
    midpoint_domain = array([img_shape[0]-1, img_shape[1]-1])/2.0
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
    [Nx,Ny]=img_shape
    [Ni,Nj]= [nd,len(angles)]
    delta_x=image_width/float(max(Nx,Ny))
    delta_s=float(detector_width)/nd

    
    geo_dict={}
    ofs_dict={}
    for dtype in [np.dtype('float64'),np.dtype('float32')]:
        #Save angular information into the ofs buffer
        ofs = zeros((8, len(angles)), dtype=dtype, order='F')
        ofs[0,:] = X; ofs[1,:] = Y; ofs[2,:] = offset; ofs[3,:] = Xinv
        ofs[4,:]=angles_diff
        ofs_dict[dtype]=ofs
        
        geometry_info = np.array([delta_x,delta_s,Nx,Ny,Ni,Nj],dtype=dtype,order='F')
        geo_dict[dtype]=geometry_info
    
    

    sinogram_shape = (nd, len(angles))
    return (ofs_dict, img_shape, sinogram_shape, geo_dict,angles_diff)
 

def fanbeam(sino, img, projectionsettings, wait_for=[]):
    """Performs the fanbeam transform of a given image using on the given projectionsettings 

    :param sino: The sinogram to be computed (in which results are saved).
    :type sino: `pyopencl.Array`.
    
    :param img: The image to be transformed .
    :type img: `pyopencl.Array`

    :param projectionsettings: The instance of the *ProjectionSettings* 
        class that contains transform type and geometry parameters.
    :type projectionsettings: `grato.ProjectionSettings`
 
    :param wait_for: 
        The events to wait for before performing the computation.
    :type wait_for: `pyopencl.Event` list or None

    :return: Event associated to computation of fanbeam transform 
        (which was added to the events of *sino*).
    :rtype: `pyopencl.Event`
    """           
    
    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype),("sinogram and image do not share common data type: "\
        +str(sino.dtype)+" and "+str(img.dtype))    
         
    dtype=sino.dtype

    projectionsettings.ensure_dtype(dtype)   
    ofs_buf=projectionsettings.ofs_buf[dtype]
    sdpd_buf=projectionsettings.sdpd_buf[dtype]
    geometry_information=projectionsettings.geometry_information[dtype]
    
    #Choose function with approrpiate dtype
    function = projectionsettings.functions[(dtype,sino.flags.c_contiguous,img.flags.c_contiguous)]
    myevent=function(sino.queue, sino.shape, None,
                        sino.data, img.data, ofs_buf, sdpd_buf,
                        geometry_information,
                        wait_for=img.events+sino.events+wait_for)
    sino.add_event(myevent)
    return myevent

def fanbeam_ad(img, sino, projectionsettings, wait_for=[]):
    """Performs the fanbeam backprojection  of a given sinogram using on the given projectionsettings 

    :param img: The image to be computed (in which results are saved).
    :type img: `pyopencl.Array` (with dimensions Nx x Ny x Nz).
    
    :param sino: The sinogram to be transformed .
    :type sino: `pyopencl.Array` (with dimensions Ns x Na x Nz)

    :param projectionsettings: The instance of the *ProjectionSettings* 
        class that contains
        transform type and geometry parameters.
    :type projectionsettings: `grato.ProjectionSettings`
 
     :param wait_for: 
        The events to wait for before performing the computation.
    :type wait_for: `pyopencl.Event` list or None

    :return: Event associated to computation of fanbeam backprojection 
        (which was added to the events of *img*).
    :rtype: `pyopencl.Event`
    """
        
    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype), ("sinogram and image do not share common data type: "\
            +str(sino.dtype)+" and "+str(img.dtype))
            
    dtype=sino.dtype
    
    projectionsettings.ensure_dtype(dtype)   
    ofs_buf=projectionsettings.ofs_buf[dtype]; 
    sdpd_buf=projectionsettings.sdpd_buf[dtype]
    geometry_information=projectionsettings.geometry_information[dtype]

    function = projectionsettings.functions_ad[(dtype,img.flags.c_contiguous,sino.flags.c_contiguous)]
    myevent = function(img.queue, img.shape, None,
                       img.data,sino.data, ofs_buf,sdpd_buf,geometry_information,
                       wait_for=img.events+sino.events+wait_for)
    img.add_event(myevent)
    return myevent

def fanbeam_struct(queue, img_shape, angles, detector_width,
                   source_detector_dist, source_origin_dist,
                   n_detectors=None, detector_shift = 0.0,
                   image_width=None, midpointshift=[0,0], fullangle=True):
    """Creates the structure of fanbeam geometry required for fanbeamtransform and its adjoint
    
    :param queue: Queue object corresponding to a context in pyopencl
    :type queue:  pyopencl.Queue
    
    :param img_shape: (Nx,Ny) the number of pixels of the image 
    :type img_shape: tuple of two integers 
			             
    :param angles: number of angles (uniform in [0,pi[, or list of angles considered,
        or a list of list of angles considered (last used when for multiple
        limited angle sets, see *fullangle*.)
    :type angles:  integer or list[float], list[list[float]]  
    
    :param detector_width: Physical length of the detector.
    :type detector_width: float
     
    :param source_detector_dist: The physical distance (orthonormal) from the source to the detectorline
    :type source_detector_dist:  float
     
    :param source_origin_dist: Distance from source to origin (which is center of rotation)
    :type source_origin_dist: float
			         
    :param n_detectors:  Number of detectors considered, i.e., resolution of the projections.
        default None chooses n_detectors =sqrt(Nx+Ny)  
    :type n_detectors: integer or None optional
			         

    :param detector_shift: Physical size of shift of detector and corresponding detector-pixel positions.
        (default=0.0)
    :type detector_shift: float, optional
			          
    :param image_width:  physical diameter of the object in question (side-length of the image square)
        (default=2.)
    :type image_width: float, optional
			           		        
    :param midpointshift: Physical shift of image away from center of rotation.
    :type midpointshift:  list of length two 

    :param fullangle:  Boolean representing wether the entire [0,pi) is represented by the angles
        (Default = True). Impacts the weighting in the backprojection. If the given angles
        only represent a discretization of a real subset of [0,pi[ ,
        i.e., limited angle situation.
        
    :return: 
        img_shape: tuple of integers
        (Nx,Ny) pixels in the image 
        
        sinogram_shape: (tuple) 
        (Ns,Na) pixels in the sinogram.
        
        Ofs_Dict- dict[np.dtype --> numpy.array]
        dictionary containing the relevant angular information in 
        numpy.array form for respective data-type numpy.dtype(float32)
        or numpy.dtype(float64)
        Arrays have dimension 8 x Na with 
        0 1 ... vector along detector-direction with length delta_s
        2 3   ... vector from source vector connecting source to center of rotation
        4 5 ... vector connecting the origin to the detectorline
        6   ... Angular weights      SDPD_Dict[np.dtype --> numpy.array]
        
        sdpd dictionary containing numpy.array with regard to dtype 
        reprsenting the values sqrt( s^2+R^2) for the weighting in 
        the Fanbeam transform (weighted by delta_ratio)
        
        image_width: the observed image_width, is equal to the input if given,
        or the determined suitable image_width if *None* is give
        (see parameter image_width)         
        
        geo_dict: numpy array (source_detector_dist, source_origin_dist,width of a detector_pixel,
        midpoint_x,midpoint_y,midpoint_detectors, img_shape[0],img_shape[1], sinogram_shape[0],
        sinogram_shape[1],width of a pixel])
        
        angles_diff: numpy.array
        same values as in ofs_dict[4] representing the weight 
        associated to the angles (i.e. size of detector-pixels)
        
    :rtype: List(img_shape,sinogram_shape,Ofs_Dict)
    """	
    
    detector_width=float(detector_width)
    source_detector_dist=float(source_detector_dist)
    source_origin_dist=float(source_origin_dist)
    midpointshift=np.array(midpointshift)
    
    # choose equidistant angles in (0,2pi] if no specific angles are given.
    if isscalar(angles):
        angles = linspace(0,2*pi,angles+1)[:-1] + pi

    image_pixels = max(img_shape[0],img_shape[1])

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
    midpoint_domain = array([img_shape[0]-1, img_shape[1]-1])/2.0
    midpoint_detectors = (nd-1.0)/2.0
    midpoint_detectors = midpoint_detectors+detector_shift*nd/detector_width
    
    #Ensure that indeed detector on the opposite side of the source
    assert source_detector_dist>source_origin_dist, 'Origin not between detector and source'
    
    #In case no image_width is predetermined, image_width is chosen in a way that the (square) image is always contained inside the fan between source and detector
    if image_width==None:
        dd=(0.5*detector_width-abs(detector_shift))/source_detector_dist
        image_width = 2*dd*source_origin_dist/sqrt(1+dd**2) # Projection to compute distance via projectionvector (1,dd) after normalization, is equal to delta_x*N_x
    
    #Ensure that source is outside the image (otherwise fanbeam is not continuous in classical L2)
    assert image_width*0.5*sqrt(1+(min(img_shape)/max(img_shape))**2)+np.linalg.norm(midpointshift)<source_origin_dist , " the image is encloses the source"
    
    #Determine midpoint (in scaling 1 = 1 pixelwidth,i.e. index of center) 
    midpoint_x=midpointshift[0]*image_pixels/float(image_width)+(img_shape[0]-1)/2.
    midpoint_y=midpointshift[1]*image_pixels/float(image_width)+(img_shape[1]-1)/2.

        
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
    ofs_dict={}
    sdpd_dict={}
    geo_dict={}
    for dtype in [np.dtype('float64'),np.dtype('float32')]:
        #Angular Information
        ofs = zeros((8, len(angles)), dtype=dtype, order='F')
        ofs[0,:] = XD; ofs[1,:] = YD
        ofs[2,:]=Qx; ofs[3,:]=Qy
        ofs[4,:]=Dx0; ofs[5]=Dy0
        ofs[6]=angles_diff
        ofs_dict[dtype]=ofs
        
        
        # Determine source detectorpixel-distance (=sqrt(R+xi**2)) for scaling
        xi=(np.arange(0,nd)- midpoint_detectors)*detector_width/nd
        source_detectorpixel_distance= sqrt((xi)**2+source_detector_dist**2)
        source_detectorpixel_distance=np.array(source_detectorpixel_distance,dtype=dtype,order='F')
        sdpd = zeros(( 1,len(source_detectorpixel_distance)), dtype=dtype, order='F')
        sdpd[0,:]=source_detectorpixel_distance[:]
        sdpd_dict[dtype]=sdpd
        
        #collect various geometric information necessary for computations
        geometry_info=np.array([source_detector_dist,source_origin_dist,detector_width/nd,
            midpoint_x,midpoint_y,midpoint_detectors,img_shape[0],img_shape[1],sinogram_shape[0],
            sinogram_shape[1],image_width/float(max(img_shape))],dtype=dtype,order='F')
        geo_dict[dtype]=geometry_info
        
    return (img_shape,sinogram_shape,ofs_dict,sdpd_dict,image_width,geo_dict,angles_diff)


def create_code():
    """ Reads c-code for opencl execution from the grato toolbox
	
    :return: code for the grat toolbox
    :rtype: :class: str
	
	
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

def upload_bufs(projectionsettings, dtype):
    ofs=projectionsettings.ofs_buf[dtype]
    ofs_buf = cl.Buffer(projectionsettings.queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
    cl.enqueue_copy(projectionsettings.queue, ofs_buf, ofs.data).wait()
        
    geometry_information=projectionsettings.geometry_information[dtype]
    geometry_buf=cl.Buffer(projectionsettings.queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
    cl.enqueue_copy(projectionsettings.queue, geometry_buf, geometry_information.data).wait()
        
    if projectionsettings.is_fan:
        sdpd=projectionsettings.sdpd_buf[dtype]
        sdpd_buf = cl.Buffer(projectionsettings.queue.context, cl.mem_flags.READ_ONLY, sdpd.nbytes)    
        cl.enqueue_copy(projectionsettings.queue, sdpd_buf, sdpd.data).wait()
        projectionsettings.sdpd_buf[dtype]=sdpd_buf

    projectionsettings.ofs_buf[dtype]=ofs_buf
    projectionsettings.geometry_information[dtype]=geometry_buf

class ProjectionSettings():
 
    """Class saving all relevant information concerning the projection geometry, and is thus a cornerstone of gratopy used in virtually all functions.
        
  	
    :param queue: Commandqueue associated with the context in question.
    :type queue: `pyoencl.CommandQueue`  	
  	
    :param geometry: Represents wether parallel beam or fanbeam setting
    	    Number 1 representing parallel beam setting, 2 fanbeam setting
    	    grato sets the Variable RADON and FANBEAM to the values 1 and two respectively.
    :type geometry: `int`
    	      
    :param img_shape: (Nx,Ny) representing the number of pixels of the image in x and y direction. It is assumed that the center of rotation is in the middle.
    :type img_shape: tuple of length two
	
	 
    :param angles:  Integer for the number of uniformly distributed angles in the angular range [0,pi[, [0,2pi[ for Radon and fanbeam transform respectively. List containing all angles considered for the projection. More advanced list of lists of angles for multiple limited angle segments.
    :type angles: Int, List[float] or List[List[float]]
    :param n_detector: Ns the number of detectors used. When none is given Ns will be chosen as sqrt(Nx^2+Ny^2).
    :type n_detector:  int, optional
    :param detector_width: Physical length of the detector (default=2.) In particular the ratio of detector_width to the image_width is as well as to R and RE are of relevance.
    :type detector_width: float, optional
    :param detector_shift:   Physical shift of the detector and correspond detector offsets. If not given no shift is applied.
    :type detector_shift: List[float], optional
    :param midpoint_shift: Vector of length two representing the  shift of the image away from center of rotation. If not given no shift is applied.
    :type midpoint_shift:  List , optional
    :param R:  the physical distance from source to the detector line. Only relevant for the fanbeam setting.
    :type R: float, necessary for fanbeam
    :param RE: physical distance for source to the origin (center of rotation). Only relevant for the fanbeam setting.
    :type RE: float, necessary for fanbeam
    :param image_width: the size of the image (more precisely  the larger side length of the rectangle image)
        i.e. the diameter of the circular object captured by image. 
        (for parallel-beam default value 2, for fanbeam chosen suitably to capture all rays). For the parallel beam setting chosing image_width = detector_width results in the standard Radon transform with each projection touching the entire object, while img_with=2 detector_width results in each projection capturing only half of the image.
    :type image_width: float, optional
    
    :param fullangel: Representing wether the entire [0,pi[ (or [0,2pi[ for fanbeam)
        is represented by the given angles, i.e., not a limited angle setting
        (Default = True). Impacts the weighting in the backprojection.
        fullangle should be set false if the given angles
        only represent a discretization of a strict subset of [0,pi[ (or [0,2pi[.
    :type fullangel: boolean, optoinal
    
    :ivar is_parallel: True if the setting is for parallel beam, False otherwise.
    :vartype is_parallel: Boolean
    
    :ivar is_fan: True if the setting is for fanbeam, False otherwise.
    :vartype is_fan: Boolean
    
    :ivar angles: List of all relevant angles for the setting. 
    :vartype angles: List
    :ivar n_angles:   Na number of angles used.
    :vartype n_angles: int 
    
    
    :ivar sinogram_shape:  (Ns,Na) representing the number of detectors and number of angles considered.
    :vartype sinogram_shape: tuple of length two
    :ivar delta_x: representing the width/lengtho of an image pixel
    :vartype delta_x:  float
    :ivar delta_s:  representing the size of an detector pixel 
    :vartype delta_s:  float
    :ivar delta_ratio:  representing the ratio delta_s/delta_x
    :vartype delta_ratio:  float
    :ivar angle_weights:    representing the angular discretization
        width for each angle, which can be used to weight the Projections. In the fullangle case these sum up to [0,pi[ or [0,2pi[ respectively.
    :vartype angle_weights: List 
    :ivar prg:   Program containing the kernels to execute gratopy.
    :vartype prg:  grato.Programm
    
    :ivar struct: Contains various information, in particular in pyopencl.Arrays containing the angular information necessary for computations.
    :vartype struct: List, see r_struct and fanbeam_struct returns
    
    """
        
        
     # :ivar ofs_buf:   
        # dictionary containing the relevant angular information in 
        # array form for respective data-type numpy.dtype(float32)
        # or numpy.dtype(float64), switching from numpy to cl.array
        # when the information is uploaded to the OpenCL device
        # (which is done automatically when required)
        # *Radon transformation*
        # Arrays have dimension 8 x Na with 
        # 0 ... weighted cos()
        # 1 ... weighted sin()
        # 2 ... detector offset
        # 3 ... inverse of cos
        # 4 ... Angular weights
        # *Fanbeam transform*
        # Arrays have dimension 8 x Na with 
        # 0 1 ... vector along detector-direction with length delta_s
        # 2 3   ... vector from source vector connecting source to center of rotation
        # 4 5 ... vector connecting the origin to the detectorline
        # 6   ... Angular weights
    # :vartype ofs_buf: dictionary  from  numpy.dtype  to  numpy.Array or  pyopencl.
    # :ivar sdpd: representing the weight sqrt(delta_s^2+R^2) required for the computation of fanbeam transform
    # :vartype sdpd: dictionary  from  numpy.dtype  to  numpy.Array or  pyopencl.        
       
    
    def __init__(self, queue,geometry, img_shape, angles, n_detectors=None, 
                    detector_width=2.0, detector_shift=0.0,
                    midpoint_shift=[0,0], R=None, RE=None,
                    image_width=None, fullangle=True):
        """Initialize a ProjectionSettings instance.
        
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
        self.img_shape=img_shape
                
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
                             
            self.struct=fanbeam_struct(self.queue,self.img_shape, self.angles, 
                    self.detector_width, R, self.RE,self.n_detectors, self.detector_shift,
                    image_width,self.midpoint_shift, self.fullangle)
            
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
            
            self.struct=radon_struct(self.queue,self.img_shape, self.angles, 
                n_detectors=self.n_detectors,
                detector_width=self.detector_width, image_width= self.image_width,
                detector_shift=self.detector_shift,fullangle=self.fullangle, )
                
            self.ofs_buf=self.struct[0]
                        
            self.delta_x=self.image_width/max(self.img_shape)
            self.delta_s=self.detector_width/self.n_detectors
            self.delta_ratio=self.delta_s/self.delta_x  
            
            self.geometry_information=self.struct[3]
            self.angle_weights=self.struct[4]

    def ensure_dtype(self, dtype):
        if not dtype in self.buf_upload:
            upload_bufs(self, dtype)
            self.buf_upload[dtype] = 1

    def show_geometry(self, angle, figure=None, axes=None, show=True):
        """ Visualize geometry associated with the projection settings. This can be useful in checking that indeed the correct input for the desired geometry were entered.
        
        :param angle: An angle in Radian from which the projection is considered.
        :type angle: float
        :param figure: Figure in which to plot-
        :type figure: matplotlib.pyplot.figure
        
        :param axes: Axes to plot in.
        :type axes: matplotlib.pyplot.axes
        
        :param show: True if the resulting plot shall be shown, False otherwise. If True, figures with the corresponding plots will be shown, though this will also show other pyplot figures not yet shown, and depent on the backend used the execution might stopp until the figures are closed. Alternatively you can at a later point use *pyplot.show()* to show the figures.
        :type show: Boolean
        """

        if (figure is None) and (axes is None):
            figure = plt.figure(0)
        if (axes is None):
            fig_axes = figure.get_axes()
            if len(fig_axes) == 0:
                axes = figure.add_subplot()
            else:
                axes = fig_axes[0]
        axes.clear()
        
        if self.is_fan:
            detector_width=self.detector_width
            source_detector_dist=self.R
            source_origin_dist=self.RE
            image_width=self.image_width
            midpoint_shift=self.midpoint_shift
            
            maxsize=max(self.RE,sqrt((self.R-self.RE)**2+detector_width**2/4.))
            
            
            angle=-angle
            A=np.array([[cos(angle),sin(angle)],[-sin(angle),cos(angle)]])  
            #Plot all relevant sizes            
            
            
            sourceposition=[-source_origin_dist,0]
            upper_detector=[source_detector_dist-source_origin_dist,detector_width*0.5+self.detector_shift]
            lower_detector=[source_detector_dist-source_origin_dist,-detector_width*0.5+self.detector_shift]
            central_detector=[source_detector_dist-source_origin_dist,0]
            
            sourceposition=np.dot(A,sourceposition)
            upper_detector=np.dot(A,upper_detector)
            lower_detector=np.dot(A,lower_detector)
            central_detector=np.dot(A,central_detector)
            
            
            axes.plot([upper_detector[0], lower_detector[0]], [upper_detector[1], lower_detector[1]],"k")
            
            axes.plot([sourceposition[0], upper_detector[0]],[sourceposition[1],upper_detector[1]],"g")
            axes.plot([sourceposition[0], lower_detector[0]],[sourceposition[1],lower_detector[1]],"g")
            
            axes.plot([sourceposition[0], central_detector[0]],[sourceposition[1], central_detector[1]],"g")         
            

            #plot(x[0]+midpoint_rotation[0],x[1]+midpoint_rotation[1],"b")
            
            draw_circle=matplotlib.patches.Circle(midpoint_shift, image_width/2*sqrt(1+(min(self.img_shape)/max(self.img_shape))**2), color='r')
            axes.add_artist(draw_circle)
            
            color=(1,1,0)   
            rect = plt.Rectangle(midpoint_shift-0.5*np.array([image_width*self.img_shape[0]/np.max(self.img_shape),image_width*self.img_shape[1]/np.max(self.img_shape)]), image_width*self.img_shape[0]/np.max(self.img_shape),image_width*self.img_shape[1]/np.max(self.img_shape),facecolor=color, edgecolor=color)
            axes.add_artist(rect)    
            
            draw_circle=matplotlib.patches.Circle(midpoint_shift, image_width/2, color='b')         
            axes.add_artist(draw_circle) 
            
            draw_circle=matplotlib.patches.Circle((0,0), image_width/10, color='k')         
            axes.add_artist(draw_circle) 
            axes.set_xlim([-maxsize,maxsize])
            axes.set_ylim([-maxsize,maxsize])
            
            
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
                        
            axes.plot([center_source[0],center_detector[0]],[center_source[1] ,center_detector[1]],"g")
            
            axes.plot([lower_source[0],lower_detector[0]],[lower_source[1],lower_detector[1]],"g")
            axes.plot([upper_source[0],upper_detector[0]],[upper_source[1],upper_detector[1]],"g")
            
            axes.plot([lower_detector[0],upper_detector[0]],[lower_detector[1],upper_detector[1]],"k")
            
            draw_circle=matplotlib.patches.Circle((0, 0), image_width/sqrt(2), color='r')
            
            axes.add_artist(draw_circle) 

            color=(1,1,0)
            draw_rectangle=matplotlib.patches.Rectangle(-0.5*np.array([image_width*self.img_shape[0]/np.max(self.img_shape),image_width*self.img_shape[1]/np.max(self.img_shape)]), image_width*self.img_shape[0]/np.max(self.img_shape),image_width*self.img_shape[1]/np.max(self.img_shape),facecolor=color, edgecolor=color)

            axes.add_artist(draw_rectangle)

            draw_circle=matplotlib.patches.Circle((0, 0), image_width/2, color='b')
            
            axes.add_artist(draw_circle) 
            draw_circle=matplotlib.patches.Circle((0,0), image_width/10, color='k')         
            axes.add_artist(draw_circle) 
            
            maxsize=sqrt(image_width**2+detector_width**2)
            axes.set_xlim([-maxsize,maxsize])
            axes.set_ylim([-maxsize,maxsize])
        if show and (figure is not None):
            figure.show()
            plt.show()



def normest(projectionsettings, number_of_iterations=50, dtype='float32',
            allocator=None):
    """
    Determine the operator norm of the projectionmethod via poweriteration. This the norm with respect to the standard l^2 norms as sum of squares.
   
    :param projectionsettings: The instance of the *ProjectionSettings* 
        class that contains transform type and geometry parameters.
    :type projectionsetting: :class: `grato.ProjectionSettings`
    :param number_of_iterations:  The number of iterations to terminate after (default 50)
    :type number_of_iterations: :class: integer 
    :param dtype:  Precision for which to apply the projection operator
        (which is not supposed to impact the estimate significantly)
    :type dtype: `numpy.dtype`
         
    :return: the operator_norm estimate of the projection  operator
    :rtype: float    
    """    
    queue=projectionsettings.queue
    
    #random starting point
    img = clarray.to_device(queue, require((random.randn(*projectionsettings.img_shape)), dtype, 'F'), allocator=allocator)
    sino=forwardprojection(img, projectionsettings)
    
    #power_iteration
    for i in range(number_of_iterations):
        normsqr = float(clarray.sum(img).get())
    
        img /= normsqr
        forwardprojection(img, projectionsettings, sino=sino)
        backprojection(sino, projectionsettings, img=img) 
    return sqrt(normsqr)

        
def landweber(sinogram, projectionsettings, number_iterations=100, w=1):
    """
    Executes Landweberiteration for projectionmethod to approximate an solution to the  projection inversion problem.

    :param sinogram: Sinogram data to inverte 
    :type sinogram: :class:`pyopencl.Array`
					
    :param projectionsettings: The instance of the *ProjectionSettings* 
        class that contains transform type and geometry parameters.
    :type projectionsettings: `grato.ProjectionSettings`
	
    :param number_iterations: Number of iterations to execute for the Landweber iteration
    :type number_iterations: int

    :param w: Relaxation parameter (default w=1, w<1 garanties convergence)
    :type w:  float
				
    :return: Reconstruction gained via Landweber iteration
    :rtype: :class: `pyopencl.Array`

    """    

    
    norm_estimate=normest(projectionsettings, allocator=sinogram.allocator)
    w=sinogram.dtype.type(w/norm_estimate**2)   

    sinonew=sinogram.copy()
    U=w*backprojection(sinonew, projectionsettings)
    Unew=clarray.zeros(projectionsettings.queue, U.shape, dtype=sinogram.dtype, order='F',
                       allocator=sinogram.allocator)
    
    for i in range(number_iterations):
        sinonew=forwardprojection(Unew, projectionsettings, sino=sinonew)-sinogram
        Unew=Unew-w*backprojection(sinonew, projectionsettings, img=U) 
    return Unew

def conjugate_gradients(sinogram, projectionsettings, epsilon, x0, number_iterations=100, restart=True):
    """
    Executes conjugate gradients methods for projectionmethods in order toapproximate the solution of the projection inversion problem.

    :param sinogram: Sinogram data to inverte.
    :type sinogram: pyopencl.Array
			            
    :param projectionsetting: The instance of the *ProjectionSettings* 
        class that contains transform type and geometry parameters.
    :type projectionsetting: `grato.ProjectionSettings`		            

    :param epsilon: Stopping criteria when residual<epsilon.
    :type epsilon: float
    
    :param x0: Startpoint for iteration (np.zeros by default).
    :type x0: pyopencl.Array

    :param number_iterations: Number of iterations to be executed.
    :type number_iterations: :class: float

    :param restart: The algorithm is relaunched when sanity check fails (for numerical reasons).
    :type restart: Boolean		      
   
    :return: Reconstruction gained via conjugate gradients.
    :rtype:  `pyopencl.Array`

    """
    
    x=x0
    d=sinogram-forwardprojection(x, projectionsettings)
    p=backprojection(d, projectionsettings)
    q=clarray.empty_like(d, projectionsettings.queue)
    snew=p+0
        
    angle_weights=clarray.reshape(projectionsettings.angle_weights,[1,len(projectionsettings.angle_weights)])
    angle_weights=np.ones(sinogram.shape)*angle_weights
    angle_weights=clarray.to_device(projectionsettings.queue, require(angle_weights, sinogram.dtype, 'F'),
                                    allocator=sinogram.allocator)

    for k in range(0,number_iterations):
        residual=np.sum(clarray.vdot(snew,snew).get())**0.5
    #   print(str(k)+"-th iteration with risidual",residual, "relative orthogonality of the residuals",clarray.vdot(snew,sold)/(clarray.vdot(snew,snew)**0.5*clarray.vdot(sold,sold)**0.5))
        if  residual<epsilon:
            break
        
        sold=snew+0.    
        forwardprojection(p, projectionsettings, sino=q)
        alpha=x.dtype.type(projectionsettings.delta_x**2/(projectionsettings.delta_s)*(clarray.vdot(sold,sold)/clarray.vdot(q*angle_weights,q)).get())
        x=x+alpha*p
        d=d-alpha*q
        backprojection(d, projectionsettings, img=snew)
        beta= (clarray.vdot(snew,snew)/clarray.vdot(sold,sold)).get()
        p=snew+beta*p       
        
        if beta>1 and restart==True:
            print("restart at", k)
            d=sinogram-forwardprojection(x, projectionsettings)
            p=backprojection(d, projectionsettings)
            q=clarray.empty_like(d, projectionsettings.queue)            
            snew=p+0.
    return x

