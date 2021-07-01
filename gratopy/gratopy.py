import sys
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pyopencl as cl
import pyopencl.array as clarray

CL_FILES = ["radon.cl", "fanbeam.cl","total_variation.cl"]

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

def forwardprojection(img, projectionsetting, sino=None, wait_for=[]):
    """
    Performs the forward projection (either for the Radon or the
    fanbeam transform) of a given image using the given projection
    settings.

    :param img: The image to be transformed.
    :type img: :class:`pyopencl.Array` with 
        `compatible  <getting_started.html>`_ dimensions
    :param projectionsetting: The settings in which the forward 
        transform is to be executed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`
    :param sino: The array in which the result of transformation
        will be saved. If *None* (per default) is given, a new array 
        will be created and returned.
    :type sino: :class:`pyopencl.Array` with `compatible`_ dimensions, 
        default None
    :param wait_for: The events to wait for before performing the 
        computation in order to avoid race conditions, see 
        `PyopenCL.events  <https://documen.tician.de/pyopencl/runtime_queue.html>`_.
    :type wait_for: :class:`list[pyopencl.Event]`, default []
    
    :return: The sinogram created by the projection of the image. 
        If the sinogram input is given, the same pyopencl array will be 
        returned, though the values in its data are overwritten.
    :rtype: :class:`pyopencl.Array`

    The forward projection can be performed for single or double
    precision arrays. The dtype (precision) of *img* and *sino* (if given)
    have to coincide and the output will be of the same precision.
    It respects any combination of *C* and *F* contiguous arrays where
    output will be of the same contiguity as img if no sino is given.
    The OpenCL events associated with the transform will be added to the
    output's events. In case the output array is created, it will use the
    allocator of img. If the image and sinogram have a third dimension 
    (z-direction) the operator is applied slicewise.
    
    """
    
    # initialize new sinogram if no sinogram is yet given
    if sino is None:
        z_dimension=tuple()
        if len(img.shape)>2:
            z_dimension=(img.shape[2],)
        sino = clarray.zeros(projectionsetting.queue, 
            projectionsetting.sinogram_shape+z_dimension, dtype=img.dtype, 
            order={0:'F',1:'C'}[img.flags.c_contiguous], 
                allocator=img.allocator)    

    # perform projection operation
    function = projectionsetting.forwardprojection
    function(sino, img, projectionsetting, wait_for=wait_for)
    return sino


def backprojection(sino, projectionsetting, img=None, wait_for=[]):
    """
    Performs the backprojection (either for the Radon or the
    fanbeam transform) of a given sinogram  using the given projection 
    settings.
		    
    :param sino: sinogram to be backprojected.
    :type sino: :class:`pyopencl.Array` with `compatible`_ dimensions
    

    :param projectionsetting: The settings in which the backprojection 
        is to be executed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`
		
    :param  img: The array in which the result of backprojection will 
        be saved.
        If None is given, a new array will be created and returned.
    :type img: :class:`pyoepncl.Array`  with `compatible`_ dimensions,
        default None
    
    :param wait_for: The events to wait for before performing the 
        computation in order to avoid race conditions, see 
        `PyopenCL.events  <https://documen.tician.de/pyopencl/runtime_queue.html>`_.
    :type wait_for: :class:`list[pyopencl.Event]`, default []
        
    :return: The image associated to the backprojected sinogram,
        coinciding with the **img** input if given, though the values 
        are overwriten.
    :rtype:  :class:`pyopencl.Array` 

    The backprojection can be performed for single or double
    precision arrays. The dtype (precision) of *img* and *sino* have
    to coincide, if no img is given the output precision coincides
    with sino's. It respects any combination of *C* and *F* contiguous 
    arrays, where if no img is given the results contiguity coincides
    with sino's. The OpenCL events associated with the transform will be
    added to the output's events.
    In case the output array is created, it will
    use the allocator of sino. If the sinogram and image have a third 
    dimension (z-direction) the operator is applied slicewise.
		  
    """        
        
    # initialize new img (to save backprojection in) if none is yet given
    if img is None:
        z_dimension=tuple()
        if len(sino.shape)>2:
            z_dimension=(sino.shape[2],)
        img = clarray.zeros(projectionsetting.queue,
            projectionsetting.img_shape+z_dimension,
            dtype=sino.dtype, order={0:'F',1:'C'}[sino.flags.c_contiguous],
            allocator=sino.allocator)    

    #execute corresponding backprojection operation
    function = projectionsetting.backprojection
    function(img, sino, projectionsetting, wait_for=wait_for)
    return img


def radon(sino, img, projectionsetting, wait_for=[]):
    """
    Performs the Radon transform of a given image using on the 
    given projectionsetting.

    :param sino: The sinogram to be computed (in which results are saved).
    :type sino: :class:`pyopencl.Array`
    
    :param img: The image to be transformed.
    :type img: :class:`pyopencl.Array` 
    :param projectionsetting: The settings in which the backprojection 
        is to be executed. 
    :type projectionsetting: :class:`gratopy.ProjectionSettings` 
 
    :param wait_for: The events to wait for before performing the computation 
        in order to avoid race conditions, see
        `PyopenCL.events  <https://documen.tician.de/pyopencl/runtime_queue.html>`_.
    :type wait_for: :class:`list[pyopencl.Event]`, default []

    :return: Event associated to computation of Radon transform 
        (which was also added to the events of sino).
    :rtype:  :class:`pyopencl.Event`
    
    """         

    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype), ("sinogram and image do not share\
        common data type: "+str(sino.dtype)+" and "+str(img.dtype))
    
    dtype=sino.dtype
    projectionsetting.ensure_dtype(dtype)
    ofs_buf=projectionsetting.ofs_buf[dtype] 
    geometry_information=projectionsetting.geometry_information[dtype]

    #Choose function with approrpiate dtype
    function = projectionsetting.functions[(dtype,\
        sino.flags.c_contiguous,img.flags.c_contiguous)]
    myevent=function(sino.queue, sino.shape, None,
                     sino.data, img.data, ofs_buf,
                     geometry_information,
                     wait_for=img.events+sino.events+wait_for)

    sino.add_event(myevent)
    return myevent
    
def radon_ad(img, sino, projectionsetting, wait_for=[]):
    """
    Performs the Radon backprojection  of a given sinogram using on 
    the given projectionsetting.

    :param img: The image to be computed (in which results are saved).
    :type img: :class:`pyopencl.Array`
    
    :param sino: The sinogram to be transformed .
    :type sino: :class:`pyopencl.Array`
    :param projectionsetting: The settings in which the backprojection 
        is to be executed. 
    :type projectionsetting: :class:`gratopy.ProjectionSettings` 
  
    :param wait_for: The events to wait for before performing the 
        computation in order to avoid race conditions, see
        `PyopenCL.events  <https://documen.tician.de/pyopencl/runtime_queue.html>`_.
    :type wait_for: :class:`list[pyopencl.Event]`, default []

    :return: Event associated to computation of Radon backprojection 
        (which was added to the events of *img*).
    :rtype: :class:`pyopencl.Event`
    
    """    
    
    my_function={(np.dtype('float32'),0,0):projectionsetting.prg.radon_ad_float_ff,
            (np.dtype('float32'),1,0):projectionsetting.prg.radon_ad_float_cf,
            (np.dtype('float32'),0,1):projectionsetting.prg.radon_ad_float_fc,
            (np.dtype('float32'),1,1):projectionsetting.prg.radon_ad_float_cc,
            (np.dtype('float'),0,0):projectionsetting.prg.radon_ad_double_ff,
            (np.dtype('float'),1,0):projectionsetting.prg.radon_ad_double_cf,
            (np.dtype('float'),0,1):projectionsetting.prg.radon_ad_double_fc,
            (np.dtype('float'),1,1):projectionsetting.prg.radon_ad_double_cc}


    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype), ("sinogram and image do not share \
        common data type: "+str(sino.dtype)+" and "+str(img.dtype))
            
    dtype=sino.dtype
    projectionsetting.ensure_dtype(dtype)   
    ofs_buf=projectionsetting.ofs_buf[dtype] 
    geometry_information=projectionsetting.geometry_information[dtype]

    #Choose function with approrpiate dtype 
    function = projectionsetting.functions_ad[(dtype,\
        img.flags.c_contiguous,sino.flags.c_contiguous)]
    myevent = function(img.queue, img.shape, None,
                        img.data, sino.data, ofs_buf,
                        geometry_information,wait_for=\
                        img.events+sino.events+wait_for)
    img.add_event(myevent)
    return myevent


def radon_struct(queue, img_shape, angles, n_detectors=None,
             detector_width=2.0, image_width=2.0,
             detector_shift=0.0,fullangle=True):
    """
    Creates the structure of radon geometry required for radontransform
    and its adjoint.
	    
    :param queue: Opencl CommandQueue in which context the 
        computations are to be executed.
    :type queue: :class:`pyoencl.CommandQueue`  	
	
    :param img_shape:  The number of pixels of the image in x- and 
        y-direction respectively, i.e., image resolution. It is assumed
        that the center of rotation is in the middle
        of the grid of quadratic pixels.
    :type img_shape: :class:`tuple` :math:`(N_x,N_y)`			             
    
    :param angles:  Determines which angles are considered for the
        projection. Integer :math:`N_a` for the number of uniformly 
        distributed angles in the angular range :math:`[0,\pi[`. 
        List containing all angles considered for the projection. 
        More advanced list of lists of angles for multiple limited 
        angle segments, see :class:`fullangle` parameter.
    :type angles: :class:`int`, :class:`list[float]` or 
        :class:`list[list[float]]`
			         
    :param n_detector: :math:`N_s` the number of (equi-spaced) detectors
        pixels considered, i.e., detector
        resolution. When None is given :math:`N_s` will be chosen as
        :math:`\sqrt{N_x^2+N_y^2}`.
    :type n_detector:  :class:`int`, default None
			         
    :param detector_width: Physical length of the detector. 
    :type detector_width: :class:`float`, default 2.0
			          
    :param image_width: Size of the image (more precisely  the larger 
        side length of the rectangle image), i.e., the diameter of the 
        circular object captured by image. For the parallel beam setting 
        chosing :math:`image_width = detector_width` results in 
        the standard Radon transform with each projection touching the 
        entire object, while img_with=2 detector_width results in each
        projection capturing only half of the image.
    :type image_width: :class:`float`, default  2.0
			           		        
    :param detector_shift:   Physical shift of the detector along 
        the detector-line
        and corresponding detector pixel offsets, i.e, shifting the 
        position of the detector pixels. If not given no shift
        is applied, i.e., the detector reaches from
        :math:`[-detector_width/2,detector_width/2]`.
    :type detector_shift: :class:`list[float]`, default 0.0
			        
    :param fullangle:  True if entire :math:`[0,\pi[` is represented by 
        the angles, False for a limited angle setting.
        Impacts the weighting in the backprojection. If the given angles,
        only represent a discretization of a real subset of :math:`[0,\pi[`,
        i.e., limited angle situation. 
    :type fullangle:  :class:`bool`, default True.
 	

    :return: Ofs_Dict: dict[np.dtype --> numpy.array]
        Dictionary containing the relevant angular information in 
        numpy.array form for respective data-type numpy.dtype(float32)
        or numpy.dtype(float64)
        
        Arrays have dimension :math:`(8, N_a)` with:
        
        * 0 ... weighted cos()
        * 1 ... weighted sin()
        * 2 ... detector offset  
        * 3 ... inverse of cos
        * 4 ... Angular weights			
 
        shape: tuple of integers :math:`(N_x,N_y)` resolution 
        of the image 
 
        sinogram_shape: (tuple) :math:`(N_s,N_a)` resolution of the sinogram.
 
        Geo_Dict: [np.dtype --> numpy.array]
        array containing information [:math:`\delta_x,\delta_s,N_x,N_y,N_s,N_a`]
 
        angles_diff: numpy.array
        same values as in ofs_dict[4] representing the weight 
        associated to the angles (i.e., size of angle-pixels)
    :rtype: mutliple (Ofs_Dict,shape,Geo_Dict,angles_diff) with 
        (:class:`dict`, :class:`tuple`, :class:`dict`, :class:`list`)
    
    """
        
    #relative_detector_pixel_width is delta_s/delta_x
    relative_detector_pixel_width=detector_width/float(image_width)\
        *max(img_shape)/n_detectors
    
    #When angles are None, understand as number of angles 
    #discretizing [0,pi]
    if isscalar(angles):    
        angles = linspace(0,pi,angles+1)[:-1]
    
    #Choosing the number of detectors as the half of the diagonal
    #through the the image (in image_pixel scale)
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
    #If fullangle is activated, the angles partion [0,pi] completely and
    # choose the first /last width appropriately
    if fullangle==True:
        angles_index = np.argsort(angles%(np.pi)) 
        angles_sorted=angles[angles_index]  %(np.pi)
        angles_sorted=np.array(hstack([-np.pi+angles_sorted[-1],\
            angles_sorted,angles_sorted[0]+np.pi]))
        angles_diff= 0.5*(abs(angles_sorted[2:len(angles_sorted)]\
            -angles_sorted[0:len(angles_sorted)-2]))
        angles_diff=np.array(angles_diff)
        angles_diff=angles_diff[angles_index]
    else:##Act as though first/last angles width is equal to the
		# distance from the second/second to last angle  
        angles_diff=[]
        for j in range(len(angles_section)-1):
            current_angles=angles[angles_section[j]:angles_section[j+1]]
            current_angles_index = np.argsort(current_angles%(np.pi)) 
            current_angles=current_angles[current_angles_index] %(np.pi)

            angles_sorted_temp=np.array(hstack([2*current_angles[0]\
                -current_angles[1],current_angles,\
                2*current_angles[len(current_angles)-1]\
                -current_angles[len(current_angles)-2]]))
            
            angles_diff_temp= 0.5*(abs(angles_sorted_temp\
                [2:len(angles_sorted_temp)]\
                -angles_sorted_temp[0:len(angles_sorted_temp)-2]))
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


    #X*x+Y*y=detectorposition, ofs is error in midpoint of 
    #the image (in shifted detector setting)
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
        
        geometry_info = np.array([delta_x,delta_s,Nx,Ny,Ni,Nj],\
            dtype=dtype,order='F')
        geo_dict[dtype]=geometry_info
    
    

    sinogram_shape = (nd, len(angles))
    return (ofs_dict, img_shape, sinogram_shape, geo_dict,angles_diff)
 

def fanbeam(sino, img, projectionsetting, wait_for=[]):
    """Performs the fanbeam transform of a given image 
    using on the given projectionsetting.

    :param sino: The sinogram to be computed (in which results are saved).
    :type sino: :class:`pyopencl.Array`
    
    :param img: The image to be transformed .
    :type img: :class:`pyopencl.Array`

    :param projectionsetting: The settings in which the 
        fanbeam transform is to be executed. 
    :type projectionsetting: :class:`gratopy.ProjectionSettings` 
  
    :param wait_for: The events to wait for before performing 
        the computation in order to avoid race conditions, see
        `PyopenCL.events  <https://documen.tician.de/pyopencl/runtime_queue.html>`_.
    :type wait_for: :class:`list[pyopencl.Event]`, default []

    :return: Event associated to computation of fanbeam transform 
        (which was added to the events of *sino*).
    :rtype: `pyopencl.Event`
    """           
    
    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype),("sinogram and image do not share \
        common data type: "+str(sino.dtype)+" and "+str(img.dtype))    
         
    dtype=sino.dtype

    projectionsetting.ensure_dtype(dtype)   
    ofs_buf=projectionsetting.ofs_buf[dtype]
    sdpd_buf=projectionsetting.sdpd_buf[dtype]
    geometry_information=projectionsetting.geometry_information[dtype]
    
    #Choose function with approrpiate dtype
    function = projectionsetting.functions[(dtype,sino.flags.c_contiguous,\
        img.flags.c_contiguous)]
    myevent=function(sino.queue, sino.shape, None,
                        sino.data, img.data, ofs_buf, sdpd_buf,
                        geometry_information,
                        wait_for=img.events+sino.events+wait_for)
    sino.add_event(myevent)
    return myevent

def fanbeam_ad(img, sino, projectionsetting, wait_for=[]):
    """Performs the fanbeam backprojection  of a given sinogram using 
    on the given projectionsetting 

    :param img: The image to be computed (in which results are saved).
    :type img: :class:`pyopencl.Array`
    
    :param sino: The sinogram to be transformed .
    :type sino: :class:`pyopencl.Array` 

    :param projectionsetting: The settings in which the fanbeam 
        backprojection is to be executed. 
    :type projectionsetting: :class:`gratopy.ProjectionSettings` 
  
    :param wait_for: The events to wait for before performing 
        the computation in order to avoid race conditions, see
        `PyopenCL.events  <https://documen.tician.de/pyopencl/runtime_queue.html>`_.
    :type wait_for: :class:`list[pyopencl.Event]`, default []

    :return: Event associated to computation of fanbeam backprojection 
        (which was added to the events of *img*).
    :rtype: `pyopencl.Event`
    """
        
    #ensure that all relevant arrays have common data_type
    assert (sino.dtype==img.dtype), ("sinogram and image do not share\
        common data type: "+str(sino.dtype)+" and "+str(img.dtype))
            
    dtype=sino.dtype
    
    projectionsetting.ensure_dtype(dtype)   
    ofs_buf=projectionsetting.ofs_buf[dtype]; 
    sdpd_buf=projectionsetting.sdpd_buf[dtype]
    geometry_information=projectionsetting.geometry_information[dtype]

    function = projectionsetting.functions_ad[(dtype,\
        img.flags.c_contiguous,sino.flags.c_contiguous)]
    myevent = function(img.queue, img.shape, None,
                       img.data,sino.data, ofs_buf,sdpd_buf,
                       geometry_information,
                       wait_for=img.events+sino.events+wait_for)
    img.add_event(myevent)
    return myevent

def fanbeam_struct(queue, img_shape, angles, detector_width,
                   source_detector_dist, source_origin_dist,
                   n_detectors=None, detector_shift = 0.0,
                   image_width=None, midpointshift=[0,0], fullangle=True):
    """Creates the structure of fanbeam geometry required for 
    fanbeamtransform and its adjoint
    
    :param queue: Opencl CommandQueue in which context the computations 
        are to be executed.
    :type queue: :class:`pyoencl.CommandQueue`  	
	
    :param img_shape:  The number of pixels of the image in x- and 
        y-direction respectively, i.e., image resolution.
        It is assumed that the center of rotation is in the middle
        of the grid of quadratic pixels.
    :type img_shape: :class:`tuple` :math:`(N_x,N_y)`			             
    
    :param angles:  Determines which angles are considered for the 
        projection. Integer :math:`N_a` for the number of uniformly 
        distributed angles in the angular range :math:`\ [0,2\pi[`.  
        List containing all angles considered for the projection. 
        More advanced list of lists of angles for multiple limited angle 
        segments, see :class:`fullangle` parameter.
    :type angles: :class:`int`, :class:`list[float]` 
        or :class:`list[list[float]]`
    :param detector_width: Physical length of the detector.
    :type detector_width: :class:`float` 
    :param source_detector_dist:  Physical (orthogonal) distance from 
        the source to the detector line (R). 
    :type source_detector_dist: :class:`float`
    :param source_origin_dist: Physical distance for source to the 
        origin (center of rotation) (RE). 
    :type source_origin_dist: :class:`float`     
			         
    :param n_detector: :math:`N_s` the number of (equi-spaced) detectors
         pixels considered, i.e., detector resolution.
         When none is given :math:`N_s` will be chosen as 
         :math:`\sqrt{N_x^2+N_y^2}`.
    :type n_detector:  :class:`int`, default None
			         

    :param detector_shift:   Physical shift of the detector along the 
        detector-line and corresponding detector pixel offsets 
        , i.e, shifting the position of the detector pixels.
        If not given no shift is applied, i.e. the detector reaches from
        [-detector_width/2,detector_width/2].
    :type detector_shift: :class:`list[float]`, default 0.0
			          
    :param image_width: Physical size of the image (more precisely  the 
        larger side length of the rectangle image),
        i.e., the diameter of the circular object captured by image. 
        The image_width is chosen suitably to capture all rays if 
        no image_width is set.
    :type image_width: float, default None corresponds  automatic choice
			           		        
    :param midpoint_shift: Vector of length two representing the  
        shift of the image away from center of rotation. 
        If not given no shift is applied.
    :type midpoint_shift:  :class:`list` , default [0.0,0.0]

    :param fullangle:  True if entire :math:`[0,2\pi[` is represented by 
        the angles, False for a limited angle setting.
        Impacts the weighting in the backprojection. If the given angles,
        only represent a discretization of a real subset of
        :math:`[0,2\pi[`, i.e., limited angle situation.
    :type fullangel: :class:`bool`, default True
        
    :return: 
        img_shape: :class:`tuple`  :math:`(N_x,N_y)` resolution 
        of the image 
        
        sinogram_shape: :class:`tuple`  :math:`(N_s,N_a)` 
        resolution of the sinogram.
        
        Ofs_Dict- dict[np.dtype --> numpy.array]
        Dictionary containing the relevant angular information in 
        numpy.array form for respective data-type numpy.dtype(float32)
        or numpy.dtype(float64)
        Arrays have dimension :math:`(8,  N_a)` with: 
        
        * 0 1 ... vector along detector-direction with length delta_s
        * 2 3 ... vector from source vector connecting source to center of rotation
        * 4 5 ... vector connecting the origin to the detectorline
        * 6 ... Angular weights  
            
        
        Sdpd_Dict: dictionary containing numpy.array with regard to  
        dtype reprsenting the values :math:`\sqrt{(s^2+R^2)}` for  
        the weighting in the fanbeam transform (weighted by delta_ratio).
        
        image_width: Physical width of the image, is equal to the input
        if given, or the determined suitable image_width if *None* 
        is give (see parameter image_width).         
        
        geo_dict: numpy array (source_detector_dist, source_origin_dist,
        width of a detector_pixel, midpoint_x,midpoint_y,midpoint_detectors,
        img_shape[0],img_shape[1], sinogram_shape[0],
        sinogram_shape[1],width of a pixel]).
        
        angles_diff: numpy.array
        same values as in ofs_dict[4] representing the weight 
        associated to the angles (i.e. size of detector-pixels).
        
    :rtype: multiple (img_shape,sinogram_shape,Ofs_Dict,Sdpd_Dict,
        image_width, geo_dict,angles_diff) with (:class:`tuple`, 
        :class:`tuple`, :class:`dict`,
        :class:`dict`, :class:`float`, :class:`dict`, :class:`list`)
    """	
    
    detector_width=float(detector_width)
    source_detector_dist=float(source_detector_dist)
    source_origin_dist=float(source_origin_dist)
    midpointshift=np.array(midpointshift)
    
    # choose equidistant angles in (0,2pi] if no specific angles are 
    # given.
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
    #If fullangle is activated, the angles partion [0,2pi] completely and
    # choose the first /last width appropriately
    if fullangle==True:
        angles_index = np.argsort(angles%(2*np.pi)) 
        angles_sorted=angles[angles_index]  %(2*np.pi)
        angles_sorted=np.array(hstack([-2*np.pi+angles_sorted[-1],
            angles_sorted,angles_sorted[0]+2*np.pi]))
        angles_diff= 0.5*(abs(angles_sorted[2:len(angles_sorted)]
            -angles_sorted[0:len(angles_sorted)-2]))
        angles_diff=np.array(angles_diff)
        angles_diff=angles_diff[angles_index]
    else:##Act as though first/last angles width is equal to the distance 
		 #from the second/second to last angle  
        angles_diff=[]
        for j in range(len(angles_section)-1):
            current_angles=angles[angles_section[j]:angles_section[j+1]]
            current_angles_index = np.argsort(current_angles%(2*np.pi)) 
            current_angles=current_angles[current_angles_index] %(2*np.pi)

            angles_sorted_temp=np.array(hstack([2*current_angles[0]\
                -current_angles[1],
                current_angles,2*current_angles[len(current_angles)-1]
                -current_angles[len(current_angles)-2]]))
            
            angles_diff_temp= 0.5*(abs(angles_sorted_temp\
                [2:len(angles_sorted_temp)]
                -angles_sorted_temp[0:len(angles_sorted_temp)-2]))
            angles_diff+=list(angles_diff_temp[current_angles_index])
        
    
    

    #compute  midpoints foror orientation
    midpoint_domain = array([img_shape[0]-1, img_shape[1]-1])/2.0
    midpoint_detectors = (nd-1.0)/2.0
    midpoint_detectors = midpoint_detectors+detector_shift*nd\
        /detector_width
    
    #Ensure that indeed detector on the opposite side of the source
    assert source_detector_dist>source_origin_dist, 'Origin not \
        between detector and source'
    
    #In case no image_width is predetermined, image_width is chosen in 
    #a way that the (square) image is always contained inside 
    #the fan between source and detector
    if image_width==None:
        dd=(0.5*detector_width-abs(detector_shift))/source_detector_dist
        image_width = 2*dd*source_origin_dist/sqrt(1+dd**2) 
        # Projection to compute distance
        # via projectionvector (1,dd) after normalization, 
        #is equal to delta_x*N_x
    
    #Ensure that source is outside the image (otherwise fanbeam is not
    # continuous in classical L2)
    assert image_width*0.5*sqrt(1+(min(img_shape)/max(img_shape))**2)\
        +np.linalg.norm(midpointshift)<source_origin_dist ,\
        " the image is encloses the source"
    
    #Determine midpoint (in scaling 1 = 1 pixelwidth,
    # i.e., index of center) 
    midpoint_x=midpointshift[0]*image_pixels/\
        float(image_width)+(img_shape[0]-1)/2.
    midpoint_y=midpointshift[1]*image_pixels/\
        float(image_width)+(img_shape[1]-1)/2.

        
    # adjust distances to pixel units, i.e. 1 unit corresponds 
    # to the length of one image pixel
    source_detector_dist *= image_pixels/float(image_width)
    source_origin_dist *= image_pixels/float(image_width)
    detector_width *= image_pixels/float(image_width)

    # unit vector associated to the angle 
    #(vector showing along the detector)
    thetaX = -cos(angles)
    thetaY = sin(angles)
    
    #Direction vector of detector direction normed to the length of a
    #single detector pixel (i.e. delta_s (in the scale of delta_x=1))
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
        
        
        # Determine source detectorpixel-distance (=sqrt(R+xi**2)) 
        #for scaling
        xi=(np.arange(0,nd)- midpoint_detectors)*detector_width/nd
        source_detectorpixel_distance= sqrt((xi)**2\
            +source_detector_dist**2)
        source_detectorpixel_distance=np.array(\
            source_detectorpixel_distance,dtype=dtype,order='F')
        sdpd = zeros(( 1,len(source_detectorpixel_distance)),\
            dtype=dtype, order='F')
        sdpd[0,:]=source_detectorpixel_distance[:]
        sdpd_dict[dtype]=sdpd
        
        #collect various geometric information necessary for computations
        geometry_info=np.array([source_detector_dist,
            source_origin_dist,detector_width/nd,
            midpoint_x,midpoint_y,midpoint_detectors,img_shape[0], 
            img_shape[1],sinogram_shape[0],
            sinogram_shape[1],image_width/float(max(img_shape))],
            dtype=dtype,order='F')
            
        geo_dict[dtype]=geometry_info
        
    return (img_shape,sinogram_shape,ofs_dict,sdpd_dict,\
        image_width,geo_dict,angles_diff)


def create_code():
    """ 
    Reads c-code for opencl execution from the gratopy toolbox.
	
    :return: code for the gratopy toolbox
    :rtype:  :class:`str`
	
	
    """    
    
    total_code=""
    for file in CL_FILES:
        textfile=open(os.path.join(os.path.abspath(\
            os.path.dirname(__file__)), file))
        code_template=textfile.read()
        textfile.close()
        
        for dtype in ["float","double"]:
            for order1 in ["f","c"]:
                for order2 in ["f","c"]:
                    total_code+=code_template.replace(\
                        "\my_variable_type",dtype)\
                        .replace("\order1",order1)\
                        .replace("\order2",order2)
                    
    return total_code

def upload_bufs(projectionsetting, dtype):
    ofs=projectionsetting.ofs_buf[dtype]
    ofs_buf = cl.Buffer(projectionsetting.queue.context, \
        cl.mem_flags.READ_ONLY, ofs.nbytes)
    cl.enqueue_copy(projectionsetting.queue, ofs_buf, ofs.data).wait()
        
    geometry_information=projectionsetting.geometry_information[dtype]
    geometry_buf=cl.Buffer(projectionsetting.queue.context, \
        cl.mem_flags.READ_ONLY, ofs.nbytes)
    cl.enqueue_copy(projectionsetting.queue, geometry_buf, \
        geometry_information.data).wait()
        
    if projectionsetting.is_fan:
        sdpd=projectionsetting.sdpd_buf[dtype]
        sdpd_buf = cl.Buffer(projectionsetting.queue.context, \
            cl.mem_flags.READ_ONLY, sdpd.nbytes)    
        cl.enqueue_copy(projectionsetting.queue, sdpd_buf, sdpd.data)\
            .wait()
        projectionsetting.sdpd_buf[dtype]=sdpd_buf

    projectionsetting.ofs_buf[dtype]=ofs_buf
    projectionsetting.geometry_information[dtype]=geometry_buf

class ProjectionSettings():
    """Class saving all relevant information concerning 
    the projection geometry, and is thus a cornerstone of gratopy used 
    in virtually all functions.
        
  	
    :param queue: Opencl CommandQueue in which context the computations 
        are to be executed.
    :type queue: :class:`pyoencl.CommandQueue`  	
  	
    :param geometry: Represents whether parallel beam or fanbeam setting
        is considered. Number 1 representing parallel beam setting, 
        2 fanbeam setting.
        Alternatively gratopy.RADON and gratopy.FANBEAM are set to these
        values and can be used.
    	    
    :type geometry: :class:`int`
    	      
    :param img_shape:  The number of pixels of the image in x- and 
        y-direction respectively, i.e.,
        image resolution. It is assumed that the center of rotation  
        is in the middle of the grid of quadratic pixels.
    :type img_shape: :class:`tuple` :math:`(N_x,N_y)`
    
    :param angles:  Determines which angles are considered for 
        the projection. Integer :math:`N_a` for the number
        of uniformly distributed angles in the angular range
        :math:`[0,\pi[,\ [0,2\pi[`
        for Radon and fanbeam transform respectively. Alternatively,
        a list containing all angles 
        considered for the projection can be given. 
        More advanced, list of lists of angles for multiple limited 
        angle segments can be given,
        see :class:`fullangle` parameter.
    :type angles: :class:`int`, :class:`list[float]` 
        or :class:`list[list[float]]` 
    :param n_detector: :math:`N_s` the number of (equi-spaced) detectors
        pixels considered. When none is given :math:`N_s`
        will be chosen as :math:`\sqrt{N_x^2+N_y^2}`.
    :type n_detector:  :class:`int`, default None
    :param detector_width: Physical length of the detector.
    :type detector_width: :class:`float`, default 2.0

    :param image_width: Physical size of the image 
        (more precisely  the sidelength 
        of the larger side of the rectangle image),
        i.e., the diameter of the circular object captured by image. 
        For fanbeam the image_width is chosen suitably so the 
        projections captures  exactly the image if no image_width is set.
        For the parallel beam setting chosing 
        image_width = detector_width results in 
        the standard Radon transform with each projection touching 
        the entire object, while img_with=2 detector_width results in 
        each projection capturing only 
        half of the image.
    :type image_width: :class:`float`, default None corresponds 
        to 2.0 for parallel-beam, choosen adaptively for fanbeam

    :param R:  Physical (orthogonal) distance from the source 
        to the detector line. Has no impact for parallel beam setting.
    :type R: :class:`float`, **must be set for fanbeam situation**
    :param RE: Physical distance from source to the origin 
        (center of rotation).
        Has no impact for parallel beam setting.
    :type RE: :class:`float`, **must be set for fanbeam situation**


    :param detector_shift:   Physical shift of the detector along 
        the detector-line and corresponding detector pixel offsets,
        i.e, shifting the position of the detector pixels.
        If not given no shift is applied, i.e., the detector reaches from
        [-detector_width/2,detector_width/2].
    :type detector_shift: :class:`list[float]`, default 0.0
    :param midpoint_shift: Vector of length two representing the  shift 
        of the image away from center of rotation. If not given no shift 
        is applied.
    :type midpoint_shift:  :class:`list` , default [0.0,0.0]
    :param fullangel: True if entire angular range (:math:`[0,\pi[` 
        for parallel, :math:`[0,2\pi[` for fan) is represented by
        the set :class:`angles`. False thus indicates a limited
        angle setting, i.e., the angles only represent
        a discretization of a strict subset of the angular range. 
        This impacts the weights in the backprojection. 
    :type fullangel: :class:`bool`, default True
 
    These input parameters create attributes of the same name in 
    an instance of :class:`ProjectionSettings`, though the corresponding 
    values might be slightly restructured by internal processes.
    Further useful attributes are listed below.
    
    :ivar is_parallel: True if the setting is for parallel beam, 
        False otherwise.
    :vartype is_parallel: :class:`bool`
    
    :ivar is_fan: True if the setting is for fanbeam, False otherwise.
    :vartype is_fan: :class:`bool`
    
    :ivar angles: List of all relevant angles for the setting. 
    :vartype angles: :class:`list[float]`
    :ivar n_angles: Number of angles :math:`N_a` used.
    :vartype n_angles: :class:`int`
    
    :ivar sinogram_shape: Represents the number of detectors 
        (n_detectors) and
        number of angles (n_angles) considered.
    :vartype sinogram_shape: :class:`tuple` :math:`(N_s,N_a)`
    :ivar delta_x: 	Physical width and height :math:`\delta_x` of 
        the image pixels.
    :vartype delta_x:  :class:`float`
    :ivar delta_s:  Physical width :math:`\delta_s` of a detector pixel. 
    :vartype delta_s:  :class:`float`
    :ivar delta_ratio:  Ratio :math:`{\delta_s}/{\delta_x}`, 
        i.e. the detector 
        pixel width relative to unit image pixels.
    :vartype delta_ratio:  :class:`float`
    :ivar angle_weights:    representing the angular discretization
        width for each angle, which can be used to weight the projections. 
        In the fullangle case these sum up to 
        :math:`[0,\pi[` or :math:`[0,2\pi[` respectively.
    :vartype angle_weights: :class:`list[float]` 
    :ivar prg:   Program containing the kernels to execute gratopy,
        for the corresponding code see :class:`gratopy.create_code`
    :vartype prg:  :class:`gratopy.Programm`
    
    :ivar struct: Contains various information, in particular 
        pyopencl.Arrays 
        containing the angular information necessary for computations.
    :vartype struct: list, see r_struct and fanbeam_struct returns
    
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
    # :ivar sdpd: representing the weight sqrt(delta_s^2+R^2) required for the
    # computation of fanbeam transform
    # :vartype sdpd: dictionary  from  numpy.dtype  to  numpy.Array or  pyopencl.        
       
    
    def __init__(self, queue,geometry, img_shape, angles, 
                    n_detectors=None, detector_width=2.0,
                    image_width=None, R=None, RE=None, 
                    detector_shift=0.0,midpoint_shift=[0,0],
                    fullangle=True):
                    
        """Initialize a ProjectionSettings instance.
        
        **Parameters**
            queue:pyopencl.queue
                queue associated to the OpenCL-context in question
            geometry: int
                1 for Radon transform or 2 for fanbeam Transform
                Gratopy defines RADON with the value 1 and 
                FANBEAM with 2,
                i.e., one can give as argument the Variable 
                RADON or FANBEAM
            img_shape: (tuple of two integers)
                see :class:`ProjectionSettings`
            angles: integer or list[float], list[list[float]]  
                number of angles (uniform in [0,pi[, or list of 
                angles considered,
                or a list of list of angles considered 
                (last used when for multiple
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
            fullangle: bool
                see class *ProjectionSettings*
        **Returns**
            self: gratopy.ProjectionSettings
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
            raise("unknown projection_type, projection_type \
                must be PARALLEL or FAN")
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
                you need to set R (the normal distance from source to \
                detector) and RE (distance from source to coordinate \
                origin which is the rotation center) "
                             
            self.struct=fanbeam_struct(self.queue,self.img_shape,
                self.angles, self.detector_width, R, self.RE,
                self.n_detectors, self.detector_shift, image_width,
                self.midpoint_shift, self.fullangle)
            
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
            
            self.struct=radon_struct(self.queue,self.img_shape,  
                self.angles, n_detectors=self.n_detectors,
                detector_width=self.detector_width, 
                image_width= self.image_width,
                detector_shift=self.detector_shift,
                fullangle=self.fullangle )
                
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
        """ Visualize geometry associated with the projection settings. 
        This can be useful in checking that indeed the correct input
        for the desired geometry was entered.
        
        :param angle: An angle in Radian from which the projection 
            is considered.
        :type angle: :class:`float`
        
        :param figure: Figure in which to plot. If neither figure nor  
            axes are given, a new figure (figure(0)) will be created.
        :type figure: :class:`matplotlib.pyplot.figure`, default None
        
        :param axes: Axes to plot in. If None is given, a new axes inside 
            the figure is created.
        :type axes: :class:`matplotlib.pyplot.axes`, default None
        
        :param show: True if the resulting plot shall be shown right away, 
            False otherwise. 
            Alternatively you can  use the *show()* method at a later  
            point to show the figures.
        :type show: :class:`bool`, default True
        
        :return: Figure and axes in which the graphic was plotted.
        
        :rtype: mutliple (:class:`matplotlib.pyplot.figure`, 
            :class:`matplotlib.pyplot.axes`)
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
            
            maxsize=max(self.RE,sqrt((self.R-self.RE)**2+\
                detector_width**2/4.))
            
            
            angle=-angle
            A=np.array([[cos(angle),sin(angle)],[-sin(angle),cos(angle)]])  
            #Plot all relevant sizes            axes
            
            
            sourceposition=[-source_origin_dist,0]
            upper_detector=[source_detector_dist-source_origin_dist,\
                detector_width*0.5+self.detector_shift]
            lower_detector=[source_detector_dist-source_origin_dist,\
                -detector_width*0.5+self.detector_shift]
            central_detector=[source_detector_dist-source_origin_dist,0]
            
            sourceposition=np.dot(A,sourceposition)
            upper_detector=np.dot(A,upper_detector)
            lower_detector=np.dot(A,lower_detector)
            central_detector=np.dot(A,central_detector)
            
            
            axes.plot([upper_detector[0], lower_detector[0]], \
                [upper_detector[1], lower_detector[1]],"k")
            
            axes.plot([sourceposition[0], upper_detector[0]],\
                [sourceposition[1],upper_detector[1]],"g")
            axes.plot([sourceposition[0], lower_detector[0]],\
                [sourceposition[1],lower_detector[1]],"g")
            
            axes.plot([sourceposition[0], central_detector[0]],\
                [sourceposition[1], central_detector[1]],"g")         
            

            #plot(x[0]+midpoint_rotation[0],x[1]+midpoint_rotation[1],"b")
            
            draw_circle=matplotlib.patches.Circle(midpoint_shift,
                image_width/2*sqrt(1+(min(self.img_shape)/\
                max(self.img_shape))**2), color='r')
            axes.add_artist(draw_circle)
            
            color=(1,1,0)   
            rect = plt.Rectangle(midpoint_shift-0.5*\
                np.array([image_width*self.img_shape[0]/
                np.max(self.img_shape),image_width*self.img_shape[1]/\
                np.max(self.img_shape)]), 
                image_width*self.img_shape[0]/np.max(self.img_shape),\
                image_width*self.img_shape[1]/np.max(self.img_shape),
                facecolor=color, edgecolor=color)
            axes.add_artist(rect)    
            
            draw_circle=matplotlib.patches.Circle(midpoint_shift,\
                image_width/2, color='b')         
            axes.add_artist(draw_circle) 
            
            draw_circle=matplotlib.patches.Circle((0,0), image_width/10,\
                color='k')         
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
            
            upper_source=[-image_width,\
                self.detector_shift+0.5*detector_width]
            lower_source=[-image_width,\
                self.detector_shift-0.5*detector_width]
            
            upper_detector=[image_width,\
                self.detector_shift+0.5*detector_width]
            lower_detector=[image_width,\
                self.detector_shift-0.5*detector_width]
            
            center_source=np.dot(A,center_source)
            center_detector=np.dot(A,center_detector)
            upper_source=np.dot(A,upper_source)
            lower_source=np.dot(A,lower_source)
            upper_detector=np.dot(A,upper_detector)
            lower_detector=np.dot(A,lower_detector)
                        
            axes.plot([center_source[0],center_detector[0]],\
                [center_source[1] ,center_detector[1]],"g")
            
            axes.plot([lower_source[0],lower_detector[0]],\
                [lower_source[1],lower_detector[1]],"g")
            axes.plot([upper_source[0],upper_detector[0]],\
                [upper_source[1],upper_detector[1]],"g")
            
            axes.plot([lower_detector[0],upper_detector[0]],\
                [lower_detector[1],upper_detector[1]],"k")
            
            draw_circle=matplotlib.patches.Circle((0, 0),\
                image_width/sqrt(2), color='r')
            
            axes.add_artist(draw_circle) 

            color=(1,1,0)
            draw_rectangle=matplotlib.patches.Rectangle(-0.5*np.array(\
                [image_width*self.img_shape[0]/np.max(self.img_shape),\
                image_width*self.img_shape[1]/np.max(self.img_shape)]), \
                image_width*self.img_shape[0]/np.max(self.img_shape),\
                image_width*self.img_shape[1]/np.max(self.img_shape),\
                facecolor=color, edgecolor=color)

            axes.add_artist(draw_rectangle)

            draw_circle=matplotlib.patches.Circle((0, 0), image_width/2, \
                color='b')
            
            axes.add_artist(draw_circle) 
            draw_circle=matplotlib.patches.Circle((0,0), image_width/10,\
                color='k')
                         
            axes.add_artist(draw_circle) 
            
            maxsize=sqrt(image_width**2+detector_width**2)
            axes.set_xlim([-maxsize,maxsize])
            axes.set_ylim([-maxsize,maxsize])
        if show and (figure is not None):
            figure.show()
#            plt.show()
        return figure, axes



def normest(projectionsetting, number_of_iterations=50, dtype='float32',
            allocator=None):
    """
    Determine the operator norm of the projection method via power 
    iteration. This the norm with respect to the standard :math:`l^2`
    norms as sum of squares. This is not a solver itself, but
    can proof useful for many iterative methods, as convergence of such
    methods usually depends on 
    the choosing a parameter suitably with respect to the operator norm. 
   
    :param projectionsetting: The settings in which to 
        compute operatornorm.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param number_of_iterations:  The number of iterations to 
        terminate after.
    :type number_of_iterations: :class:`int`, default 50 
    :param dtype:  Precision for which to apply the projection operator
        (which is not supposed to impact the estimate significantly).
    :type dtype: :class:`numpy.dtype`, default numpy.dtype(float32)
         
    :return: An estimate of the operator norm for the projection  operator.
    :rtype: :class:`float`    
    """    
    queue=projectionsetting.queue
    
    #random starting point
    img = clarray.to_device(queue, require((random.randn(\
        *projectionsetting.img_shape)), dtype, 'F')
        , allocator=allocator)
    sino=forwardprojection(img, projectionsetting)
    
    #power_iteration
    for i in range(number_of_iterations):
        normsqr = float(clarray.sum(img).get())
    
        img /= normsqr
        forwardprojection(img, projectionsetting, sino=sino)
        backprojection(sino, projectionsetting, img=img) 
    return sqrt(normsqr)

        
def landweber(sino, projectionsetting, number_iterations=100, w=1):
    """ 
    Executes Landweber iteration for projection methods to approximate 
    a solution to the
    projection inversion problem. This method is also known as SIRT to
    some communities.

    :param sino: Sinogram data to inverte.
    :type sino: :class:`pyopencl.Array`
					
    :param projectionsetting: The settings in which the projection 
        inversion is considered.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`
	
    :param number_iterations: Number of Landweber iteration to be executed.
    :type number_iterations: :class:`int`, default 100

    :param w: Relaxation parameter weighted by the norm of the projection
        operator (w<1 garanties convergence).
    :type w:  :class:`float`, default 1
				
    :return: Reconstruction from given sinogram gained via Landweber 
        iteration.
    :rtype: :class:`pyopencl.Array`

    """    

    
    norm_estimate=normest(projectionsetting, allocator=sino.allocator)
    print ("norm",norm_estimate)
    w=sino.dtype.type(w/norm_estimate**2)   

    sinonew=sino.copy()
    U=w*backprojection(sinonew, projectionsetting)
    Unew=clarray.zeros(projectionsetting.queue, U.shape, dtype=sino.dtype,
        order='F', allocator=sino.allocator)
    
    for i in range(number_iterations):
        print(np.linalg.norm(sinonew.get()))
        sinonew=forwardprojection(Unew, projectionsetting, sino=sinonew)\
            -sino
            
        Unew=Unew-w*backprojection(sinonew, projectionsetting, img=U) 
    return Unew

def conjugate_gradients(sino, projectionsetting, epsilon=0.01, 
    number_iterations=20, x0=None, restart=True):
    """
    Executes conjugate gradients iteration for projection methods in 
    order to approximate the solution of the projection inversion problem.

    :param sino: Sinogram data to inverte.
    :type sino: :class:`pyopencl.Array`
			            
    :param projectionsetting: The settings in which the projection 
        inversion is considered.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`		            

    :param epsilon: Stopping criteria when relative residual<epsilon.
    :type epsilon: :class:`float`, default 0.01
    
    :param number_iterations: Number of iterations to be executed.
    :type number_iterations: :class:`float`, default 20
    
    :param x0: Startpoint for iteration (zeros by default).
    :type x0: :class:`pyopencl.Array`, default None

    :param restart: The algorithm is relaunched when sanity check fails 
        (for numerical reasons).
    :type restart: :class:`bool`		      
   
    :return: Reconstruction gained via conjugate gradients iteration.
    :rtype:  :class:`pyopencl.Array`

    """
    
    if x0==None:
        dimensions=projectionsetting.img_shape
        if len(sino.shape)>2:
            dimensions+tuple([sino.shape[2]])
        x0=clarray.zeros(projectionsetting.queue,dimensions,
            sino.dtype,order={0:'F',1:'C'}[sino.flags.c_contiguous])
    assert(x0.flags.c_contiguous==sino.flags.c_contiguous),\
        ("Error, please make sure both the data sino and your \
        guess x0 have the same contiguity")
    x=x0
    
    d=sino-forwardprojection(x, projectionsetting,wait_for=x.events)
    p=backprojection(d, projectionsetting,wait_for=d.events)
    q=clarray.empty_like(d, projectionsetting.queue)
    snew=backprojection(d, projectionsetting,wait_for=d.events)
    sold=backprojection(d, projectionsetting,wait_for=d.events)
        
    angle_weights=clarray.reshape(projectionsetting.angle_weights,\
        [1,len(projectionsetting.angle_weights)])
        
    angle_weights=np.ones(sino.shape)*angle_weights
    angle_weights=clarray.to_device(projectionsetting.queue,\
        require(np.array(angle_weights,
        order={0:'F',1:'C'}[sino.flags.c_contiguous]), sino.dtype),\
        allocator=sino.allocator)


    for k in range(0,number_iterations):    
		
        forwardprojection(p, projectionsetting, sino=q,wait_for=p.events)
        alpha=x.dtype.type(projectionsetting.delta_x**2/\
            (projectionsetting.delta_s) *(clarray.vdot(sold,sold)\
           /clarray.vdot(q*angle_weights,q)).get())
           
        x=x+alpha*p
        d=d-alpha*q
        backprojection(d, projectionsetting, img=snew,wait_for=d.events)
        beta= (clarray.vdot(snew,snew)/clarray.vdot(sold,sold)).get()
        sold=snew+0.
        p=beta*p+snew
        residual=np.sum(clarray.vdot(snew,snew).get())**0.5/\
            np.sum(clarray.vdot(sino,sino).get())**0.5
        if  residual<epsilon:
            break

        if beta>1 and restart==True:
            #print("restart at", k)
            d=sino-forwardprojection(x, projectionsetting,\
                wait_for=x.events)
                
            p=backprojection(d, projectionsetting,wait_for=d.events)
            q=clarray.empty_like(d, projectionsetting.queue)            
            snew=backprojection(d, projectionsetting,wait_for=d.events)
            residual=np.sum(clarray.vdot(snew,snew).get())**0.5/\
                np.sum(clarray.vdot(sino,sino).get())**0.5

    return x

def total_variation_reconstruction(sino, projectionsetting,mu,
    number_iterations=1000,z_distance=1):
    """
    Executes primal-dual algorithm projection methods to solve 
    :math:`\min_{u} \mu\|\mathcal{P}u-f\|_{L^2}^2+{TV}(u)` 
    for :math:`\mathcal{P}` the projection operator in question.
    This is an approximation approach for the projection inversion approach

    :param sino: Sinogram data to inverte.
    :type sino: :class:`pyopencl.Array`
			            
    :param projectionsetting: The settings in which the projection 
        inversion is considered.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`		            

    :param mu: Weight parameter, the smaller the stronger the 
        applied regularization.
    :type epsilon: :class:`float`
    
    :param number_iterations: Number of iterations to be executed.
    :type number_iterations: :class:`float`, default 1000
    
    :param z_distance: When 3-dimensional datasets are considered,  
        regularization is also applied in z-dimension,
        but we allow unisotropic discretizationsize in z-direction.
        The parameter represents the ratio of the z-height to the  
        xy-pixel width and length. If no coupling in z-direction is 
        desired, choose z_distance=0.
    :type z_distance: :class:`float`, default 1 i.e. isotropic pixels
   
    :return: Reconstruction gained via primal dual iteration for the 
        total variation penalized inversion problem.
    :rtype:  :class:`pyopencl.Array`

    """
    #Establish queue and context
    
    
    
    #preliminary definitions and parameters
    queue=projectionsetting.queue
    ctx=queue.context    
    my_dtype=sino.dtype
    my_dimensions=projectionsetting.img_shape
    my_order=order={0:'F',1:'C'}[sino.flags.c_contiguous]
    
    img_shape=projectionsetting.img_shape
    if len(sino.shape)==2:
        z_distance=0
    else:
        img_shape=img_shape+tuple([sino.shape[2]])
    extended_img_shape=tuple([4])+img_shape

    ##Definitions of suitable kernel functions for primal and dual updates
    # update dual variable to dataterm
    def update_lambda(lamb, Ku, f, sigma,mu, normest, wait_for=None):
        myfunction={(np.dtype("float32"),0):projectionsetting.prg.update_lambda_L2_float_ff,
		    (np.dtype("float32"),1):projectionsetting.prg.update_lambda_L2_float_cc,
		    (np.dtype("float"),0):projectionsetting.prg.update_lambda_L2_double_ff,
		    (np.dtype("float"),1):projectionsetting.prg.update_lambda_L2_double_cc}
		
        return myfunction[lamb.dtype,lamb.flags.c_contiguous](lamb.queue,
            lamb.shape, None,lamb.data, Ku.data, f.data,
            float32(sigma/normest), float32(mu), wait_for=wait_for)

    #update v the dual of gradient of u
    def update_v(v, u,  sigma, z_distance, wait_for=None):
        myfunction={(np.dtype("float32"),0):projectionsetting.prg.update_v_float_ff,
        (np.dtype("float32"),1):projectionsetting.prg.update_v_float_cc,
        (np.dtype("float"),0):projectionsetting.prg.update_v_double_ff,
        (np.dtype("float"),1):projectionsetting.prg.update_v_double_cc}	
		
        return myfunction[v.dtype,v.flags.c_contiguous](v.queue, u.shape, None,
            v.data, u.data, float32(sigma),float32(z_distance), 
            wait_for=wait_for)

    #update primal variable u (the image)
    def update_u(u, u_, v, Kstarlambda, tau, normest,z_distance,wait_for=None):
        myfunction={(np.dtype("float32"),0):projectionsetting.prg.update_u_float_ff,
        (np.dtype("float32"),1):projectionsetting.prg.update_u_float_cc,
        (np.dtype("float"),0):projectionsetting.prg.update_u_double_ff,
        (np.dtype("float"),1):projectionsetting.prg.update_u_double_cc}
		
        return myfunction[u.dtype,u.flags.c_contiguous](u.queue, u.shape, None,
            u.data, u_.data,v.data, Kstarlambda.data, float32(tau),
            float32(1.0/normest),float32(z_distance), wait_for=wait_for)
    
    #Compute the norm of v and project (dual update)
    def update_NormV(V,normV,wait_for=None):
        myfunction={(np.dtype("float32"),0):projectionsetting.prg.update_NormV_unchor_float_ff,
        (np.dtype("float32"),1):projectionsetting.prg.update_NormV_unchor_float_cc,
        (np.dtype("float"),0):projectionsetting.prg.update_NormV_unchor_double_ff,
        (np.dtype("float"),1):projectionsetting.prg.update_NormV_unchor_double_cc}
        return myfunction[V.dtype,V.flags.c_contiguous](V.queue, V.shape[1:], None, 
            V.data,normV.data, wait_for=wait_for)

    # update of the extra gradient
    update_extra = {np.dtype(float32):cl.elementwise.ElementwiseKernel(
        ctx, 'float *u_, float *u', 'u[i] = 2.0f*u_[i] - u[i]'),
        np.dtype(float):cl.elementwise.ElementwiseKernel(ctx, 
        'double *u_, double *u', 'u[i] = 2.0f*u_[i] - u[i]')}[sino.dtype]

	
	#Initialising Variables for the iteration
    U=clarray.zeros(queue, img_shape, dtype=my_dtype, order=my_order)
    U_=clarray.zeros(queue, img_shape, dtype=my_dtype, order=my_order)
    V=clarray.zeros(queue, extended_img_shape, dtype=my_dtype, \
        order=my_order)
        
    Lamb=clarray.zeros(queue,sino.shape,dtype=my_dtype, order=my_order)
    KU=clarray.zeros(queue, sino.shape, dtype=my_dtype, order=my_order)
    KSTARlambda=clarray.zeros(queue, img_shape, dtype=my_dtype, \
        order=my_order)
        
    normV=clarray.zeros(queue, img_shape, dtype=my_dtype, order=my_order)


	#Computing estimates for Parameter
    norm_estimate = normest(projectionsetting)
    Lsqr = 17.0
    sigma = 1.0/sqrt(Lsqr)
    tau = 1.0/sqrt(Lsqr)
    mu=mu/(sigma+mu)
	

    #Primal Dual Iterations
    for i in range(number_iterations):		
		
        #Dual Update
        V.add_event(update_v(V, U_, sigma,z_distance,
        				 wait_for=U_.events))

        normV.add_event(update_NormV(V,normV,wait_for=V.events))	

        forwardprojection(U_,projectionsetting,KU,wait_for=U_.events)
   
        Lamb.add_event(update_lambda(Lamb, KU, sino, sigma,mu, norm_estimate,
        						 wait_for=KU.events + sino.events))
		
        #Primal Update
        backprojection(Lamb,projectionsetting,KSTARlambda,\
            wait_for=Lamb.events)	
        
        U_.add_event(update_u(U_, U, V, KSTARlambda, tau, norm_estimate,\
            z_distance, wait_for=[]))
		
		#Extragradient update 
        U.add_event(update_extra(U_, U, wait_for=U.events + U_.events))	
			
        (U, U_) = (U_, U)

        sys.stdout.write('\rProgress at {:3.0%}'.\
            format(float(i)/number_iterations))
		
    return U

    
