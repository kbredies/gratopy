import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pyopencl as cl
import pyopencl.array as clarray
import scipy
import scipy.sparse

# Source files for opencl kernels
CL_FILES1 = ["radon.cl", "fanbeam.cl"]
CL_FILES2 = ["total_variation.cl", "utilities.cl"]

# Class attribute corresponding to which geometry to consider
PARALLEL = 1
RADON = 1
FANBEAM = 2
FAN = 2


###########
# Programm created from the gpu_code
class Program(object):
    def __init__(self, ctx, code):

        # activate warnings
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        # build opencl code
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build()
        # add the kernels functions to the local dictionary
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
            self.__dict__[kernel.function_name] = kernel


def check_compatibility(img, sino, projectionsetting):
    """
    Ensures, that img, sino, and projectionsetting have compatible
    dimensions and types.
    """
    assert (sino.dtype == img.dtype), ("sinogram and image do not share"
        + "common data type: " + str(sino.dtype)+" and "+str(img.dtype))

    assert (sino.shape[0:2] == projectionsetting.sinogram_shape), (
        "The dimensions of the sinogram" + str(sino.shape)
        + " do not match the projectionsetting's "
        + str(projectionsetting.sinogram_shape))

    assert (sino.shape[0:2] == projectionsetting.sinogram_shape), (
        "The dimensions of the image " + str(img.shape)
        + " do not match the projectionsetting's "
        + str(projectionsetting.img_shape))

    if len(sino.shape) > 2:
        if sino.shape[2] > 1:
            assert(len(img.shape) > 2), (" The sinogram has a third dimension"
                + "but the image does not.")
            assert(sino.shape[2] == img.shape[2]), ("The third dimension"
                + "(z-direction) of the sinogram is" + str(sino.shape[2])
                + " and the image's is" + str(img.shape[2])
                + ", they do not coincide.")

    if len(img.shape) > 2:
        if img.shape[2] > 1:
            assert(len(sino.shape) > 2), (" The sinogram has a third dimension"
                + "but the image does not.")


def forwardprojection(img, projectionsetting, sino=None, wait_for=[]):
    """
    Performs the forward projection (either for the Radon or the
    fanbeam transform) of a given image using the given projection
    settings.

    :param img: The image to be transformed.
    :type img: :class:`pyopencl.array.Array` with
        :ref:`compatible  <compatible>` dimensions
    :param projectionsetting: The geometry settings for which the forward
        transform is computed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`
    :param sino: The array in which the result of transformation
        is saved. If :obj:`None` (per default) is given, a new array
        will be created and returned.
    :type sino: :class:`pyopencl.array.Array` with
        :ref:`compatible <compatible-sino>` dimensions,
        default :obj:`None`
    :param wait_for: The events to wait for before performing the
        computation in order to avoid, e.g., race conditions, see
        :class:`pyopencl.Event`.
    :type wait_for: :class:`list[pyopencl.Event]`, default :attr:`[]`

    :return: The sinogram associated with the projection of the image.
        If the **sino** is not :obj:`None`, the same :mod:`pyopencl` array
        is returned with the values in its data overwritten.
    :rtype: :class:`pyopencl.array.Array`

    The forward projection can be performed for single or double
    precision arrays. The dtype (precision) of **img** and **sino** (if given)
    have to coincide and the output will be of the same precision.
    It respects any combination of *C* and *F* contiguous arrays where
    output will be of the same contiguity as img if no sino is given.
    The OpenCL events associated with the transform will be added to the
    output's events. In case the output array is created, it will use the
    allocator of **img**. If the image and sinogram have a third dimension
    (z-direction) the operator is applied slicewise.

    """

    # initialize new sinogram if no sinogram is yet given
    if sino is None:
        z_dimension = tuple()
        if len(img.shape) > 2:
            z_dimension = (img.shape[2],)
        # create sinogram with same basic properties as img
        sino = clarray.zeros(projectionsetting.queue,
                             projectionsetting.sinogram_shape+z_dimension,
                             dtype=img.dtype,
                             order={0: 'F', 1: 'C'}[img.flags.c_contiguous],
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

    :param sino: Sinogram to be backprojected.
    :type sino: :class:`pyopencl.array.Array` with
        :ref:`compatible <compatible-sino>` dimensions

    :param projectionsetting: The geometry settings for which the forward
        transform is computed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param  img: The array in which the result of backprojection is saved.
        If :obj:`None` is given, a new array will be created and returned.
    :type img: :class:`pyopencl.array.Array`  with
        :ref:`compatible <compatible>` dimensions,
        default :obj:`None`

    :param wait_for: The events to wait for before performing the
        computation in order to avoid, e.g., race conditions, see
        :class:`pyopencl.Event`.
    :type wait_for: :class:`list[pyopencl.Event]`, default :attr:`[]`

    :return: The image associated with the backprojected sinogram,
        coinciding with the **img** if not :obj:`None`, with the values
        in its data overwritten.
    :rtype:  :class:`pyopencl.array.Array`

    The backprojection can be performed for single or double
    precision arrays. The dtype (precision) of **img** and **sino** have
    to coincide. If no **img** is given, the output precision coincides
    with **sino**'s. The operation respects any combination of
    *C* and *F* contiguous
    arrays, where if **img** is :obj:`None`, the result's contiguity coincides
    with **sino**'s. The OpenCL events associated with the transform will be
    added to the output's events.
    In case the output array is created, it will
    use the allocator of **sino**. If the sinogram and image have a third
    dimension (z-direction), the operator is applied slicewise.
    """

    # initialize new img (to save backprojection in) if none is yet given
    if img is None:
        z_dimension = tuple()
        if len(sino.shape) > 2:
            z_dimension = (sino.shape[2],)
        # create img for backprojection with same basic properties as sino
        img = clarray.zeros(projectionsetting.queue,
                            projectionsetting.img_shape+z_dimension,
                            dtype=sino.dtype, order={0: 'F', 1: 'C'}[
                                sino.flags.c_contiguous],
                            allocator=sino.allocator)

    # execute corresponding backprojection operation
    function = projectionsetting.backprojection
    function(img, sino, projectionsetting, wait_for=wait_for)
    return img


def radon(sino, img, projectionsetting, wait_for=[]):
    """
    Performs the Radon transform of a given image using the
    given **projectionsetting**.

    :param sino: The array in which the resulting sinogram is written.
    :type sino: :class:`pyopencl.array.Array`

    :param img: The image to transform.
    :type img: :class:`pyopencl.array.Array`

    :param projectionsetting: The geometry settings for which the
        Radon transform is performed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param wait_for: The events to wait for before performing the computation
        in order to avoid, e.g., race conditions, see :class:`pyopencl.Event`.
    :type wait_for: :class:`list[pyopencl.Event]`, default []

    :return: Event associated with the computation of the
        Radon transform (which is also added to the events of **sino**).
    :rtype:  :class:`pyopencl.Event`
    """

    # ensure that all relevant arrays have common data_type and
    # compatible dimensions
    check_compatibility(img, sino, projectionsetting)

    # select additional information of suitable data-type,
    # upload via ensure_dtype in case not yet uploaded to gpu
    dtype = sino.dtype
    projectionsetting.ensure_dtype(dtype)
    ofs_buf = projectionsetting.ofs_buf[dtype]
    geometry_information = projectionsetting.geometry_information[dtype]

    # choose function with approrpiate dtype
    function = projectionsetting.functions[(dtype,
                                            sino.flags.c_contiguous,
                                            img.flags.c_contiguous)]

    # execute corresponding function and add event to sinogram
    myevent = function(sino.queue, sino.shape, None,
                       sino.data, img.data, ofs_buf,
                       geometry_information,
                       wait_for=img.events+sino.events+wait_for)

    sino.add_event(myevent)
    return myevent


def radon_ad(img, sino, projectionsetting, wait_for=[]):
    """
    Performs the Radon backprojection of a given sinogram using
    the given **projectionsetting**.

    :param img: The array in which the resulting backprojection is
        written.
    :type img: :class:`pyopencl.array.Array`

    :param sino: The sinogram to transform.
    :type sino: :class:`pyopencl.array.Array`

    :param projectionsetting: The geometry settings for which the
        Radon backprojection is performed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param wait_for: The events to wait for before performing the computation
        in order to avoid, e.g., race conditions, see :class:`pyopencl.Event`.
    :type wait_for: :class:`list[pyopencl.Event]`, default []

    :return: Event associated with the computation of the Radon
        backprojection
        (which is also added to the events of **img**).
    :rtype: :class:`pyopencl.Event`
    """

    # ensure that all relevant arrays have common data_type and
    # compatible dimensions
    check_compatibility(img, sino, projectionsetting)

    # select additional information of suitable data-type,
    # upload via ensure_dtype in case not yet uploaded to gpu
    dtype = sino.dtype
    projectionsetting.ensure_dtype(dtype)
    ofs_buf = projectionsetting.ofs_buf[dtype]
    geometry_information = projectionsetting.geometry_information[dtype]

    # choose function with approrpiate dtype
    function = projectionsetting.functions_ad[(dtype,
                                               img.flags.c_contiguous,
                                               sino.flags.c_contiguous)]

    # execute corresponding function and add event to image
    myevent = function(img.queue, img.shape, None,
                       img.data, sino.data, ofs_buf,
                       geometry_information,
                       wait_for=img.events+sino.events+wait_for)
    img.add_event(myevent)
    return myevent


def radon_struct(queue, img_shape, angles, n_detectors=None,
                 detector_width=2.0, image_width=2.0, midpoint_shift=[0, 0],
                 detector_shift=0.0, fullangle=True, angular_range=[]):
    """
    Creates the structure storing geometry information required for
    the Radon transform and its adjoint.

    :param queue: OpenCL command queue in which context the
        computations are to be performed.
    :type queue: :class:`pyopencl.CommandQueue`

    :param img_shape:  The number of pixels of the image in x- and
        y-direction respectively, i.e., the image size.
        It is assumed that by default, the center of rotation is in
        the middle of the grid of quadratic pixels. The midpoint can,
        however, be shifted, see the **midpoint_shift** parameter.
    :type img_shape: :class:`tuple` :math:`(N_x,N_y)`

    :param angles:  Determines which angles are considered for the
        projection. Either the integer :math:`N_a` representing the
        number of uniformly distributed angles in the angular range
        :math:`[0,\\pi[`, a list containing all angles considered for
        the projection, or a list of lists containing angles for
        multiple limited angle sections, also see the
        **fullangle** parameter.
    :type angles: :class:`int`, :class:`list[float]`,
        :class:`list[list[float]]`, :class:`list[(float,float,int)]`
        or :class:`list[(float,float,list[float])]`

    :param n_detectors: The number :math:`N_s` of considered (equi-spaced)
        detectors. If :obj:`None`, :math:`N_s` will be chosen as
        :math:`\\sqrt{N_x^2+N_y^2}`.
    :type n_detectors:  :class:`int` or :obj:`None`, default :obj:`None`

    :param detector_width: Physical length of the detector line.
    :type detector_width: :class:`float`, default 2.0

    :param image_width: Size of the image indicated by the length of
        the longer side of the rectangular image domain.
        Choosing
        **image_width** = **detector_width** results in
        the standard Radon transform with each projection touching the
        entire object, while **img_width** = 2 **detector_width** results
        in each projection capturing only half of the image.
    :type image_width: :class:`float`, default 2.0

    :param midpoint_shift: Two-dimensional vector representing the
        shift of the image away from center of rotation.
        Defaults to the application of no shift.
    :type midpoint_shift:  :class:`list[float]`, default [0.0, 0.0]

    :param detector_shift: Physical shift of the detector along
        the detector line in detector pixel offsets. Defaults to
        the application of no shift, i.e., the detector reaches from
        [- **detector_width**/2, **detector_width**/2].
    :type detector_shift: :class:`list[float]`, default 0.0

    :param fullangle: If :obj:`True`, the angles are interpreted to
        represent the whole interval :math:`[0,\\pi[`.
        If :obj:`False`, a limited angle setting is considered, i.e.,
        the given angles represent a discretization of a
        proper subset of :math:`[0,\\pi[`.
        Affects the weights in the backprojection.
    :type fullangle:  :class:`bool`, default :attr:`True`

    :param angular_range: Only relevant if fullangle=False, can be used to
        determine the angular range of the different angle sections
        (which impacts the angle_weights). List of tuples with lower and upper
        bound of the angular-range of the section in question. If instead a
        empty tuple is given, the algorithm will try to choose suitable ranges
        itself, similarly missing tuples will be filled with automatic choices.
    :type angular_range: list[tuple(a,b)], default = []

    :return:
        Tuple (**ofs_dict**, **img_shape**, **sinogram_shape**,
        **geo_dict**, **angles_diff**).

    :var ofs_dict:
        Dictionary containing the relevant angular information as
        :class:`numpy.ndarray` for the data types :attr:`numpy.float32`
        and :attr:`numpy.float64`.
        The arrays have dimension :math:`(8, N_a)` with columns:

        +---+-------------------+
        | 0 | weighted cosine   |
        +---+-------------------+
        | 1 | weighted sine     |
        +---+-------------------+
        | 2 | detector offset   |
        +---+-------------------+
        | 3 | inverse of cosine |
        +---+-------------------+
        | 4 | angular weight    |
        +---+-------------------+

        The remaining columns are unused.
    :vartype ofs_dict: :class:`dict{numpy.dtype: numpy.ndarray}`

    :var shape:
        Tuple of integers :math:`(N_x,N_y)` representing the size
        of the image.
    :vartype shape: :class:`tuple`

    :var sinogram_shape:
        Tuple of integers :math:`(N_s,N_a)` representing the size
        of the sinogram.
    :vartype sinogram_shape: :class:`tuple`

    :var geo_dict:
        Dictionary mapping the allowed data types 
        to an array containing the values
        [:math:`\\delta_x, \\delta_s, N_x, N_y, N_s, N_a`].
    :vartype geo_dict: :class:`dict{numpy.dtype: numpy.ndarray}`

    :var angles_diff_buf: Dictionary containing the
        same values as in **ofs_dict** [4] representing the weights
        associated with the angles (i.e., the length of sinogram
        pixels in the angular direction).
    :vartype angles_diff: :class:`dict{numpy.dtype: numpy.ndarray}`

    :var angular_range: See inputvariable of the same name, though the
        automatic choices will be added.
    :vartype angular_range: list[tuple]

    """

    # relative_detector_pixel_width is delta_s/delta_x
    relative_detector_pixel_width = detector_width/float(image_width)\
        * max(img_shape)/n_detectors

    # When angles are None, understand as number of angles
    # discretizing [0,pi]
    if np.isscalar(angles):
        angles = np.linspace(0, np.pi, angles+1)[:-1]

    # Choosing the number of detectors as the the diagonal
    # through the the image (in image_pixel scale) if none is given
    if n_detectors is None:
        nd = int(np.ceil(np.hypot(img_shape[0], img_shape[1])))
    else:
        nd = n_detectors

    # Warn that fullangle == True, the angular_range will be ignored
    if (fullangle and angular_range):
        print("WARNING, fullangle is active, and hence the given angular_range"
              + " will be ignored!")

    # Extract angle information when angles has multiple sections
    # (list of lists)
    if isinstance(angles[0], list) or isinstance(angles[0], np.ndarray):
        n_angles = 0
        angles_new = []
        angles_section = [0]
        count = 0
        # write all angles into a single array and remember
        # where new sections begin
        for j in range(len(angles)):
            n_angles += len(angles[j])
            for k in range(len(angles[j])):
                angles_new.append(angles[j][k])
                count += 1
            angles_section.append(count)
        angles = np.array(angles_new)
    else:
        n_angles = len(angles)
        angles_section = [0, n_angles]
        angles = np.array(angles)
    sinogram_shape = (nd, n_angles)

    # Angular weights (resolution associated to angles)
    # If fullangle is activated, the angles partion [0,pi] completely and
    # choose the first /last width appropriately
    if fullangle:
        angles_index = np.argsort(angles % (np.pi))
        angles_sorted = angles[angles_index] % (np.pi)

        # add first angle at end and last angle at beginning
        # to create full circle, and then compute suitable difference
        angles_sorted = np.array(np.hstack([-np.pi+angles_sorted[-1],
                                            angles_sorted, angles_sorted[0]
                                            + np.pi]))
        angles_diff = 0.5*(abs(angles_sorted[2:len(angles_sorted)]
                               - angles_sorted[0:len(angles_sorted)-2]))
        angles_diff = np.array(angles_diff)
        angles_diff = angles_diff[angles_index]

        # Special case when an angle appears twice (particular when angles in
        # [0,2pi] are considered instead of [0,pi] and mod pi has same angles)
        for i in range(len(angles)-1):
            if abs(angles_sorted[i+1] - angles_sorted[i+2]) < 0.00001:
                val = (angles_diff[angles_index[i]]
                       + angles_diff[angles_index[i+1]])*0.5
                angles_diff[angles_index[i+1]] = val
                angles_diff[angles_index[i]] = val

    else:  # limited angle setting is considered, then the given angular_range
           # is considered and if none given an automatism is used

        # Make sure that angular_range has suitable form
        if isinstance(angular_range, tuple):
            angular_range2 = []
            angular_range2.append(angular_range)
            angular_range = angular_range2

        assert isinstance(angular_range, list), "Expected that angular "\
                                                + "range to be a list of "\
                                                + "tuples, but recieved "\
                                                + str(type(angular_range))
        # Case not a sufficient amount of angular_range information is given
        # Add empty tuples corresponding to automatisms
        if len(angular_range) < len(angles_section)-1:
            print("Fullangle"
                  + "=False requires as many angular_range entries "
                  + "(list of tuples) as angle_sections are given, but "
                  + str(len(angular_range))
                  + " angular ranges given and " + str(len(angles_section)-1)
                  + " angle sections! The missing angular_ranges will be "
                  + "added automatically (This can lead to issues for angles "
                  + "set with a single angle). In particular it leads to the "
                  + "two outermost angles. It might be adviseable to check the"
                  + " resulting angle_weights to ensure suitable "
                  + "weights were chosen")
            for i in range(len(angular_range), len(angles_section)-1):
                angular_range.append(tuple())

        # go through all angle_sections to compute the angular widths
        angles_diff = []
        for j in range(len(angles_section)-1):
            # get angles of current angles_section
            current_angles = angles[angles_section[j]:angles_section[j+1]]
            current_angles_index = np.argsort(current_angles % (np.pi))
            current_angles = current_angles[current_angles_index] % (np.pi)

            # Consider corresponding angular range, choose automatically
            # if necessary
            current_boundaries = angular_range[j]
            if current_boundaries == tuple():
                # choose boundaries with half the distance from first/last to
                # second/second to last
                current_boundaries = (1.5*current_angles[0]
                                      - 0.5*current_angles[1],
                                      1.5*current_angles[-1]
                                      - 0.5*current_angles[-2])
                angular_range[j] = current_boundaries

            # compute angular difference
            angles_sorted_temp = np.array(np.hstack([2*current_boundaries[0]
                                                    - current_angles[0],
                                                     current_angles,
                                                     2*current_boundaries[1]
                                                     - current_angles[-1]]))
            angles_diff_temp = 0.5*(abs(angles_sorted_temp
                                        [2:len(angles_sorted_temp)]
                                        - angles_sorted_temp
                                        [0:len(angles_sorted_temp)-2]))

            # Special case when an angle appears twice (particular when
            # angles in [0,2pi[ are considered instead of [0,pi[ and
            # modulo pi has same angles)
            for i in range(len(current_angles)-1):
                if abs(angles_sorted_temp[i+1]
                        - angles_sorted_temp[i+2]) < 0.00001:
                    val = (angles_diff_temp[i]+angles_diff_temp[i+1])*0.5
                    angles_diff_temp[[i]] = val
                    angles_diff_temp[[i+1]] = val

            angles_diff += list(angles_diff_temp[current_angles_index])

    # compute basic resolutions
    delta_x = image_width/float(max(img_shape))
    delta_s = float(detector_width)/nd

    # Compute the midpoints of geometries
    midpoint_domain = np.array([img_shape[0]-1, img_shape[1]-1])/2.0 +\
        np.array(midpoint_shift)/delta_x
    midpoint_detectors = (nd-1.0)/2.0

    # Vector in projection-direction (from source toward detector)
    X = np.cos(angles)/relative_detector_pixel_width
    Y = np.sin(angles)/relative_detector_pixel_width
    Xinv = 1.0/X

    # set near vertical lines to horizontal
    mask = np.where(abs(X) <= abs(Y))
    Xinv[mask] = 1.0/Y[mask]

    # X*x+Y*y=detectorposition, offset is error in midpoint of
    # the sinogram (in shifted detector setting)
    offset = midpoint_detectors - X*midpoint_domain[0]\
        - Y*midpoint_domain[1] + detector_shift/delta_s

    # Save for datatype float64 and float32 the relevant additional information
    # required for the computations
    geo_dict = {}
    ofs_dict = {}
    angle_diff_dict = {}
    for dtype in [np.dtype('float64'), np.dtype('float32')]:
        # save angular information into the ofs buffer
        ofs = np.zeros((8, len(angles)), dtype=dtype, order='F')
        ofs[0, :] = X
        ofs[1, :] = Y
        ofs[2, :] = offset
        ofs[3, :] = Xinv
        ofs[4, :] = angles_diff
        ofs_dict[dtype] = ofs

        angle_diff_dict[dtype] = np.array(angles_diff, dtype=dtype)

        geometry_info = np.array([delta_x, delta_s, img_shape[0], img_shape[1],
                                  nd, n_angles], dtype=dtype, order='F')
        geo_dict[dtype] = geometry_info

    return (ofs_dict, img_shape, sinogram_shape, geo_dict,
            angle_diff_dict, angular_range)


def fanbeam(sino, img, projectionsetting, wait_for=[]):
    """
    Performs the fanbeam transform of a given image using the
    given **projectionsetting**.

    :param sino: The array in which the resulting sinogram is written.
    :type sino: :class:`pyopencl.array.Array`

    :param img: The image to transform.
    :type img: :class:`pyopencl.array.Array`

    :param projectionsetting: The geometry settings for which the
        fanbeam transform is performed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param wait_for: The events to wait for before performing the computation
        in order to avoid, e.g., race conditions, see :class:`pyopencl.Event`.
    :type wait_for: :class:`list[pyopencl.Event]`, default []

    :return: Event associated with the computation of the
        fanbeam transform (which is also added to the events of **sino**).
    :rtype:  :class:`pyopencl.Event`
    """

    # ensure that all relevant arrays have common data_type and
    # compatible dimensions
    check_compatibility(img, sino, projectionsetting)

    # select additional information of suitable data-type,
    # upload via ensure_dtype in case not yet uploaded to gpu
    dtype = sino.dtype
    projectionsetting.ensure_dtype(dtype)
    ofs_buf = projectionsetting.ofs_buf[dtype]
    sdpd_buf = projectionsetting.sdpd_buf[dtype]
    geometry_information = projectionsetting.geometry_information[dtype]

    # choose function with approrpiate dtype
    function = projectionsetting.functions[(dtype, sino.flags.c_contiguous,
                                            img.flags.c_contiguous)]

    # execute corresponding function and add event to sinogram
    myevent = function(sino.queue, sino.shape, None,
                       sino.data, img.data, ofs_buf, sdpd_buf,
                       geometry_information,
                       wait_for=img.events+sino.events+wait_for)
    sino.add_event(myevent)
    return myevent


def fanbeam_ad(img, sino, projectionsetting, wait_for=[]):
    """
    Performs the fanbeam backprojection of a given sinogram using
    the given **projectionsetting**.

    :param img: The array in which the resulting backprojection is
        written.
    :type img: :class:`pyopencl.array.Array`

    :param sino: The sinogram to transform.
    :type sino: :class:`pyopencl.array.Array`

    :param projectionsetting: The geometry settings for which the
        fanbeam backprojection is performed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param wait_for: The events to wait for before performing the computation
        in order to avoid, e.g., race conditions, see :class:`pyopencl.Event`.
    :type wait_for: :class:`list[pyopencl.Event]`, default []

    :return: Event associated with the computation of the fanbeam
        backprojection
        (which is also added to the events of **img**).
    :rtype: :class:`pyopencl.Event`
    """

    # ensure that all relevant arrays have common data_type and
    # compatible dimensions
    check_compatibility(img, sino, projectionsetting)

    # select additional information of suitable data-type,
    # upload via ensure_dtype in case not yet uploaded to gpu
    dtype = sino.dtype
    projectionsetting.ensure_dtype(dtype)
    ofs_buf = projectionsetting.ofs_buf[dtype]
    sdpd_buf = projectionsetting.sdpd_buf[dtype]
    geometry_information = projectionsetting.geometry_information[dtype]

    function = projectionsetting.functions_ad[(dtype,
                                               img.flags.c_contiguous,
                                               sino.flags.c_contiguous)]

    # execute corresponding function and add event to sinogram
    myevent = function(img.queue, img.shape, None,
                       img.data, sino.data, ofs_buf, sdpd_buf,
                       geometry_information,
                       wait_for=img.events+sino.events+wait_for)
    img.add_event(myevent)
    return myevent


def fanbeam_struct(queue, img_shape, angles, detector_width,
                   source_detector_dist, source_origin_dist,
                   n_detectors=None, detector_shift=0.0,
                   image_width=None, midpoint_shift=[0, 0],
                   fullangle=True, reverse_detector=False,
                   angular_range=[]):
    """
    Creates the structure storing geometry information required for
    the fanbeam transform and its adjoint.

    :param queue: OpenCL command queue in which context the
        computations are to be performed.
    :type queue: :class:`pyopencl.CommandQueue`

    :param img_shape:  The number of pixels of the image in x- and
        y-direction respectively, i.e., the image size.
        It is assumed that by default, the center of rotation is in
        the middle of the grid of quadratic pixels. The midpoint can,
        however, be shifted, see the **midpoint_shift** parameter.
    :type img_shape: :class:`tuple` :math:`(N_x,N_y)`

    :param angles:  Determines which angles are considered for the
        projection. Either the integer :math:`N_a` representing the
        number of uniformly distributed angles in the angular range
        :math:`[0,2\\pi[`, a list containing all angles considered for
        the projection, or a list of lists containing angles for
        multiple limited angle sections, also see the
        **fullangle** parameter.
    :type angles: :class:`int`, :class:`list[float]` or
        :class:`list[list[float]]`

    :param detector_width: Physical length of the detector line.
    :type detector_width: :class:`float`, default 2.0

    :param source_detector_dist:  Physical (orthogonal) distance **R** from
        the source to the detector line.
    :type source_detector_dist: :class:`float`

    :param source_origin_dist: Physical distance **RE** from the source to the
        origin (center of rotation).
    :type source_origin_dist: :class:`float`

    :param n_detectors: The number :math:`N_s` of considered (equi-spaced)
        detectors. If :obj:`None`, :math:`N_s` will be chosen as
        :math:`\\sqrt{N_x^2+N_y^2}`.
    :type n_detectors:  :class:`int` or :obj:`None`, default :obj:`None`

    :param detector_shift: Physical shift of the detector along
        the detector line in detector pixel offsets. Defaults to
        the application of no shift, i.e., the detector reaches from
        [- **detector_width**/2, **detector_width**/2].
    :type detector_shift: :class:`list[float]`, default 0.0

    :param image_width: Size of the image indicated by the length of
        the longer side of the rectangular image domain.
        If :obj:`None`, **image_width** is chosen to capture just
        all rays.
    :type image_width: :class:`float`, default :obj:`None`

    :param midpoint_shift: Two-dimensional vector representing the
        shift of the image away from center of rotation.
        Defaults to the application of no shift.
    :type midpoint_shift:  :class:`list[float]`, default [0.0, 0.0]

    :param fullangle: If :obj:`True`, the angles are interpreted to
        represent the whole interval :math:`[0,2\\pi[`.
        If :obj:`False`, a limited angle setting is considered, i.e.,
        the given angles represent a discretization of a
        proper subset of :math:`[0,2\\pi[`.
        Affects the weights in the backprojection.
    :type fullangle:  :class:`bool`, default :attr:`True`

    :param reverse_detector: When :attr:`True`, the detector direction
        is flipped.
    :type reverse_detector: :class:`bool`, default :attr:`False`

    :param angular_range: Only relevant if fullangle=False, can be used to
        determine the angular range of the different angle sections
        (which impacts the angle_weights). List of tuples with lower and upper
        bound of the angular-range of the section in question. If instead a
        empty tuple is given, the algorithm will try to choose suitable ranges
        itself, similarly missing tuples will be filled with automatic choices.
    :type angular_range: list[tuple(a,b)], default = []

    :return:
        Tuple (**img_shape**, **sinogram_shape**, **ofs_dict**,
        **sdpd_dict**, **image_width**, **geo_dict**, **angles_diff**,
        **angular_range**).

    :var img_shape:
        Tuple of integers :math:`(N_x,N_y)` representing the size
        of the image.
    :vartype img_shape: :class:`tuple`

    :var sinogram_shape:
        Tuple of integers :math:`(N_s,N_a)` representing the size
        of the sinogram.
    :vartype sinogram_shape: :class:`tuple`

    :var ofs_dict:
        Dictionary containing the relevant angular information as
        :class:`numpy.ndarray` for the data types :attr:`numpy.float32`
        and :attr:`numpy.float64`.
        The arrays have dimension :math:`(8, N_a)` with columns:

        +-----+--------------------------------------------+
        | 0 1 | vector of length :math:`\\delta_s`          |
        |     | pointing in positive detector direction    |
        +-----+--------------------------------------------+
        | 2 3 | vector connecting source and center of     |
        |     | rotation                                   |
        +-----+--------------------------------------------+
        | 4 5 | vector connection the origin and its       |
        |     | projection onto the detector line          |
        +-----+--------------------------------------------+
        | 6   | angular weight                             |
        +-----+--------------------------------------------+

        The remaining column is unused.
    :vartype ofs_dict: :class:`dict{numpy.dtype: numpy.ndarray}`

    :var sdpd_dict: Dictionary mapping :attr:`numpy.float32`
        and :attr:`numpy.float64` to a :class:`numpy.ndarray`
        representing the values :math:`\\sqrt{(s^2+R^2)}` for
        the weighting in the fanbeam transform (weighted by 
        **delta_ratio**, i.e., :math:`\\delta_s/\\delta_x`).
    :vartype sdpd_dict: :class:`dict{numpy.dtype: numpy.ndarray}`

    :var image_width: Physical size of the image. Equal to the input
        parameter if given, or to the determined image size if
        **image_width** is :obj:`None` (see parameter
        **image_width**).
    :vartype image:width: :class:`float`

    :var geo_dict:
        Dictionary mapping the allowed data types to an
        array containing the values
        [source detector distance, source origin distance,
        width of a detector_pixel, image midpoint x-coordinate,
        image midpoint y-coordinate, detector line midpoint,
        img_shape[0], img_shape[1], sinogram_shape[0],
        sinogram_shape[1], width of a pixel].
    :vartype geo_dict: :class:`dict{numpy.dtype: numpy.ndarray}`

    :var angles_diff:
        Dictionary containing the
        same values as in **ofs_dict** [6] representing the weights
        associated with the angles (i.e., the length of sinogram
        pixels in the angular direction).
    :vartype angles_diff: :class:`dict{numpy.dtype: numpy.ndarray}`

    :var angular_range: See inputvariable of the same name, though the
        automatic choices will be added.
    :vartype angular_range: list[tuple]
    """

    # ensure physical quantites are suitable
    detector_width = float(detector_width)
    source_detector_dist = float(source_detector_dist)
    source_origin_dist = float(source_origin_dist)
    midpointshift = np.array(midpoint_shift)

    # Warn that fullangle == True, the angular_range will be ignored
    if (fullangle and angular_range):
        print("WARNING, fullangle is active, and hence the given angular_range"
              + " will be ignored!")

    # choose equidistant angles in [0,2pi[ if no specific angles are given.
    if np.isscalar(angles):
        angles = np.linspace(0, 2*np.pi, angles+1)[:-1]

    image_pixels = max(img_shape[0], img_shape[1])

    # Choosing the number of detectors as the the diagonal
    # through the the image (in image_pixel scale) if none is given
    if n_detectors is None:
        nd = int(np.ceil(np.hypot(img_shape[0], img_shape[1])))
    else:
        nd = n_detectors
    assert isinstance(nd, int), "Number of detectors must be integer"

    # Extract angle information when angles has multiple sections
    # (list of lists)
    if isinstance(angles[0], list) or isinstance(angles[0], np.ndarray):
        n_angles = 0
        angles_new = []
        angles_section = [0]
        count = 0

        # write all angles into a single array and remember
        # where new sections begin
        for j in range(len(angles)):
            n_angles += len(angles[j])
            for k in range(len(angles[j])):
                angles_new.append(angles[j][k])
                count += 1
            angles_section.append(count)
        angles = np.array(angles_new)
    else:
        n_angles = len(angles)
        angles_section = [0, n_angles]
        angles = np.array(angles)

    sinogram_shape = (nd, n_angles)

    # Angular weights (resolution associated to angles)
    # If fullangle is activated, the angles partion [0,2pi] completely and
    # choose the first /last width appropriately
    if fullangle:
        angles_index = np.argsort(angles % (2*np.pi))
        angles_sorted = angles[angles_index] % (2*np.pi)
        angles_sorted = np.array(np.hstack([-2*np.pi+angles_sorted[-1],
                                           angles_sorted, angles_sorted[0]
                                           + 2*np.pi]))

        # add first angle at end and last angle at beginning
        # to create full circle, and then compute suitable difference
        angles_diff = 0.5*(abs(angles_sorted[2:len(angles_sorted)]
                               - angles_sorted[0:len(angles_sorted)-2]))
        angles_diff = np.array(angles_diff)
        angles_diff = angles_diff[angles_index]
    else:  # limited angle setting is considered, then the given angular_range
           # is considered and if none given an automatism is used

        # Make sure that angular_range has suitable form
        if isinstance(angular_range, tuple):
            angular_range2 = []
            angular_range2.append(angular_range)
            angular_range = angular_range2

        assert isinstance(angular_range, list), "Expected that angular "\
                                                + "range to be a list of "\
                                                +  "tuples, but recieved "\
                                                + str(type(angular_range))

        # Case not a sufficient amount of angular_range information is given
        # Add empty tuples corresponding to automatisms
        if len(angular_range) < len(angles_section)-1:
            print("Fullangle"
                  + "=False requires as many angular_range entries "
                  + "(list of tuples) as angle_sections are given, but "
                  + str(len(angular_range))
                  + " angular ranges given and " + str(len(angles_section)-1)
                  + " angle sections! The missing angular_ranges will be "
                  + "added automatically (This can lead to issues for angles "
                  + "set with a single angle). In particular it leads to the "
                  + "two outermost angles. It might be adviseable to check the "
                  + "resulting angle_weights to ensure suitable "
                  + "weights were chosen")
            for i in range(len(angular_range),len(angles_section)-1):
                angular_range.append(tuple())

        angles_diff = []
        # go through all angle_sections to compute the angular widths
        for j in range(len(angles_section)-1):
            # get angles of current angles_section
            current_angles = angles[angles_section[j]:angles_section[j+1]]
            current_angles_index = np.argsort(current_angles % (2*np.pi))
            current_angles = current_angles[current_angles_index] % (2*np.pi)

            # Consider corresponding angular range, choose automatically
            # if necessary
            current_boundaries = angular_range[j]
            if current_boundaries == tuple():
                current_boundaries = (1.5*current_angles[0]
                                      - 0.5*current_angles[1],
                                      1.5*current_angles[-1]
                                      - 0.5*current_angles[-2])
                angular_range[j] = current_boundaries

            # compute angular difference
            angles_sorted_temp = np.array(np.hstack([2*current_boundaries[0]
                                                    - current_angles[0],
                                                     current_angles,
                                                     2*current_boundaries[1]
                                                     - current_angles[-1]]))

            angles_diff_temp = 0.5*(abs(angles_sorted_temp
                                        [2:len(angles_sorted_temp)]
                                        - angles_sorted_temp
                                        [0:len(angles_sorted_temp)-2]))
            angles_diff += list(angles_diff_temp[current_angles_index])

    # compute midpoints of geometries
    midpoint_detectors = (nd-1.0)/2.0
    midpoint_detectors = midpoint_detectors+detector_shift*nd\
        / detector_width

    # ensure that indeed detector on the opposite side of the source
    assert (source_detector_dist > source_origin_dist), ('The origin is not '
        + 'between detector and source')

    # In case no image_width is predetermined, image_width is chosen in
    # a way that the (square) image is always contained inside
    # the fan between source and detector
    if image_width is None:
        # distance from image_center to the outermost rays between
        # source and detector via projection to compute distance via
        # projectionvector (1,dd) after normalization,
        # is equal to delta_x*N_x
        dd = (0.5*detector_width-abs(detector_shift))/source_detector_dist
        image_width = 2*dd*source_origin_dist/np.sqrt(1+dd**2)
        assert image_width > 0, "The automatism for choosing the image_width"\
            + " failed as the image can never"\
            + " be fully contained in the fans from all directions "\
            + "(most likely due to detector_shift being "\
            + "larger than half the detector_width)! "\
            + "Please set the image_width parameter by hand!"

    # ensure that source is outside the image domain
    # (otherwise fanbeam is not continuous in classical L2)
    assert (image_width*0.5*np.sqrt(1+(min(img_shape)/max(img_shape))**2)
        + np.linalg.norm(midpointshift) < source_origin_dist), (
        'The source is not outside the image domain')

    # Determine midpoint (in scaling 1 = 1 pixelwidth,
    # i.e., index of center)
    midpoint_x = (midpointshift[0]*image_pixels
                  / float(image_width)+(img_shape[0]-1)/2.)
    midpoint_y = (midpointshift[1]*image_pixels
                  / float(image_width)+(img_shape[1]-1)/2.)

    # adjust distances to pixel units, i.e. 1 unit corresponds
    # to the length of one image pixel
    source_detector_dist *= image_pixels/float(image_width)
    source_origin_dist *= image_pixels/float(image_width)
    detector_width *= image_pixels/float(image_width)

    # unit vector associated to the angle
    # (vector showing from source to detector)
    detector_orientation = {True: -1, False: 1}[reverse_detector]
    thetaX = np.cos(angles+np.pi*0.5)*detector_orientation
    thetaY = np.sin(angles+np.pi*0.5)*detector_orientation

    # Direction vector along the detector line normed to the length of a
    # single detector pixel (i.e. delta_s (in the scale of delta_x=1))
    XD = thetaY*detector_width/nd
    YD = -thetaX*detector_width/nd

    # Direction vector leading to source from origin (with proper length RE)
    Qx = -thetaX*source_origin_dist
    Qy = -thetaY*source_origin_dist

    # Direction vector from origin to the detector
    # (projection of center onto detector)
    Dx0 = thetaX*(source_detector_dist-source_origin_dist)
    Dy0 = thetaY*(source_detector_dist-source_origin_dist)

    # Save for datatype float64 and float32 the relevant additional information
    # required for the computations
    ofs_dict = {}
    sdpd_dict = {}
    geo_dict = {}
    angle_diff_dict = {}
    for dtype in [np.dtype('float64'), np.dtype('float32')]:
        # save angular information into the ofs buffer
        ofs = np.zeros((8, len(angles)), dtype=dtype, order='F')
        ofs[0, :] = XD
        ofs[1, :] = YD
        ofs[2, :] = Qx
        ofs[3, :] = Qy
        ofs[4, :] = Dx0
        ofs[5] = Dy0
        ofs[6] = angles_diff
        ofs_dict[dtype] = ofs

        # determine source detectorpixel-distance (=sqrt(R+xi**2))
        # for scaling
        xi = (np.arange(0, nd) - midpoint_detectors)*detector_width/nd
        source_detectorpixel_distance = np.sqrt((xi)**2
                                                + source_detector_dist**2)
        source_detectorpixel_distance = np.array(
            source_detectorpixel_distance, dtype=dtype, order='F')
        sdpd = np.zeros((1, len(source_detectorpixel_distance)),
                        dtype=dtype, order='F')
        sdpd[0, :] = source_detectorpixel_distance[:]
        sdpd_dict[dtype] = sdpd

        # collect various geometric information necessary for computations
        geometry_info = np.array([source_detector_dist,
                                 source_origin_dist, detector_width/nd,
                                 midpoint_x, midpoint_y, midpoint_detectors,
                                 img_shape[0], img_shape[1], sinogram_shape[0],
                                  sinogram_shape[1],
                                  image_width/float(max(img_shape))],
                                 dtype=dtype, order='F')

        geo_dict[dtype] = geometry_info

        angle_diff_dict[dtype] = np.array(angles_diff, dtype=dtype)

    return (img_shape, sinogram_shape, ofs_dict, sdpd_dict,
            image_width, geo_dict, angle_diff_dict, angular_range)


def create_code():
    """
    Reads and creates CL code containing all OpenCL kernels
    of the gratopy toolbox.

    :return: The toolbox's CL code.
    :rtype:  :class:`str`
    """

    total_code = ""
    # go through all the source files
    for file in CL_FILES1:
        textfile = open(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), file))
        code_template = textfile.read()
        textfile.close()

        # go through all possible dtypes and contiguities and replace
        # the placeholders suitably
        for dtype in ["float", "double"]:
            for order1 in ["f", "c"]:
                for order2 in ["f", "c"]:
                    total_code += code_template.replace(
                        "\\my_variable_type", dtype)\
                        .replace("\\order1", order1)\
                        .replace("\\order2", order2)

    # go through all the source files
    for file in CL_FILES2:
        textfile = open(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), file))
        code_template = textfile.read()
        textfile.close()

        # go through all possible dtypes and contiguities and replace
        # the placeholders suitably
        for dtype in ["float", "double"]:
            for order1 in ["f", "c"]:
                total_code += code_template.replace(
                    "\\my_variable_type", dtype).replace("\\order1", order1)

    return total_code


def upload_bufs(projectionsetting, dtype):
    """
    Loads the buffers from projectionsetting of desired type onto the gpu,
    i.e., change the np.arrays to buffers and save in corresponding
    dictionaries of buffers.
    """
    # upload ofs_buffer
    ofs = projectionsetting.ofs_buf[dtype]
    ofs_buf = cl.Buffer(projectionsetting.queue.context,
                        cl.mem_flags.READ_ONLY, ofs.nbytes)
    cl.enqueue_copy(projectionsetting.queue, ofs_buf, ofs.data).wait()

    # upload angle_weights
    angle_weights = projectionsetting.angle_weights_buf[dtype]
    angle_weights_buf = cl.Buffer(projectionsetting.queue.context,
                                  cl.mem_flags.READ_ONLY, angle_weights.nbytes)
    cl.enqueue_copy(projectionsetting.queue, angle_weights_buf,
                    angle_weights.data).wait()

    # upload geometric information
    geometry_information = projectionsetting.geometry_information[dtype]
    geometry_buf = cl.Buffer(projectionsetting.queue.context,
                             cl.mem_flags.READ_ONLY,
                             geometry_information.nbytes)
    cl.enqueue_copy(projectionsetting.queue, geometry_buf,
                    geometry_information.data).wait()

    # upload sdpd in case of fanbeam geometry
    if projectionsetting.is_fan:
        sdpd = projectionsetting.sdpd_buf[dtype]
        sdpd_buf = cl.Buffer(projectionsetting.queue.context,
                             cl.mem_flags.READ_ONLY, sdpd.nbytes)
        cl.enqueue_copy(projectionsetting.queue, sdpd_buf, sdpd.data).wait()
        projectionsetting.sdpd_buf[dtype] = sdpd_buf

    # update buffer dictionaries
    projectionsetting.ofs_buf[dtype] = ofs_buf
    projectionsetting.geometry_information[dtype] = geometry_buf
    projectionsetting.angle_weights_buf[dtype] = angle_weights_buf


class ProjectionSettings():
    """ Creates and stores all relevant information concerning
    the projection geometry. Serves as a parameter for virtually all
    gratopy's functions.

    :param queue: OpenCL command queue to which the computations
        are to be associated.
    :type queue: :class:`pyopencl.CommandQueue`

    :param geometry: Represents whether parallel beam (:const:`gratopy.RADON`)
        or fanbeam geometry (:const:`gratopy.FANBEAM`)
        is considered.

    :type geometry: :class:`int`

    :param img_shape:  The number of pixels of the image in x- and
        y-direction respectively, i.e., the image dimension.
        It is assumed that by default, the center of rotation is in
        the middle of the grid of quadratic pixels. The midpoint can,
        however, be shifted, see the **midpoint_shift** parameter.
    :type img_shape: :class:`tuple` :math:`(N_x,N_y)`

    :param angles:  Determines which angles are considered for
        the projection. An integer is interpreted as the number :math:`N_a`
        of uniformly distributed angles in the angular range
        :math:`[0,\\pi[`, :math:`[0,2\\pi[`
        for Radon and fanbeam transform, respectively. Alternatively,
        a list containing all angles
        considered for the projection can be given.
        Also, a list of lists of angles for multiple limited
        angle section can be given,
        see the **fullangle** parameter.
    :type angles: :class:`int`, :class:`list[float]`
        or :class:`list[list[float]]`
    :param n_detectors: The number :math:`N_s` of (equi-spaced) detectors
        pixels considered. When :obj:`None`, :math:`N_s`
        will be chosen as :math:`\\sqrt{N_x^2+N_y^2}`.
    :type n_detectors:  :class:`int`, default :obj:`None`
    :param detector_width: Physical length of the detector.
    :type detector_width: :class:`float`, default 2.0

    :param image_width: Physical size of the image
        indicated by the length of
        the longer side of the rectangular image domain.
        For parallel beam geometry, when :obj:`None`,
        **image_width** is chosen as 2.0.
        For fanbeam geometry, when :obj:`None`, **image_width** is chosen
        such that the projections exactly capture the image domain.
        To illustrate, chosing **image_width** = **detector_width** results
        in  the standard Radon transform with each projection touching
        the entire object, while **img_width** = 2 **detector_width**
        results in each projection capturing only
        half of the image.
    :type image_width: :class:`float`, default :obj:`None`

    :param R:  Physical (orthogonal) distance from source
        to detector line. Has no impact for parallel beam geometry.
    :type R: :class:`float`, **must be set for fanbeam geometry**

    :param RE: Physical distance from source to origin
        (center of rotation).
        Has no impact for parallel beam geometry.
    :type RE: :class:`float`, **must be set for fanbeam geometry**

    :param detector_shift:   Physical shift of all detector pixels
        along the detector line.
        Defaults to the application of no shift, i.e.,
        the detector pixels span the range
        [-**detector_width**/2, **detector_width**/2].
    :type detector_shift: :class:`list[float]`, default 0.0

    :param midpoint_shift: Two-dimensional vector representing the
        shift of the image away from center of rotation.
        Defaults to the application of no shift.
    :type midpoint_shift:  :class:`list`, default [0.0, 0.0]

    :param fullangle:
        Indicates whether the entire angular range is represented by
        **angles**. If :obj:`True`,
        the entire angular range (:math:`[0,\\pi[`
        for parallel beam, :math:`[0,2\\pi[` for fanbeam geometry)
        is represented. :obj:`False` indicates a limited
        angle setting, i.e., the angles only represent
        a discretization of a proper subset of the angular range.
        This impacts the weights in the backprojection.
    :type fullangle: :class:`bool`, default :obj:`True`

    :param reverse_detector: When :attr:`True`, the detector direction
        is flipped in case of fanbeam geometry, i.e., the positive and
        negative detector positions are swapped.
        This parameter has no effect for parallel geometry.
    :type reverse_detector: :class:`bool`, default :attr:`False`

    :param angular_range: Only relevant if fullangle=False, can be used to
        determine the angular range of the different angle sections
        (which impacts the angle_weights). List of tuples with lower and upper
        bound of the angular-range of the section in question. If instead a
        empty tuple is given, the algorithm will try to choose suitable ranges
        itself, similarly missing tuples will be filled with automatic choices.
    :type angular_range: list[tuple(a,b)], default = []



    These input parameters create attributes of the same name in
    an instance of :class:`ProjectionSettings`, though the corresponding
    values might be slightly restructured by internal processes.
    Further useful attributes are listed below. It is advised not to set
    these attributes directly but rather to choose suitable input
    parameters for the initialization.

    :ivar is_parallel: :obj:`True` if the geometry is for parallel beams,
        :obj:`False` otherwise.
    :vartype is_parallel: :class:`bool`

    :ivar is_fan: :obj:`True` if the geometry is for fanbeam geometry,
        :obj:`False` otherwise.
    :vartype is_fan: :class:`bool`

    :ivar angles: List of all computed projection angles.
    :vartype angles: :class:`list[float]`

    :ivar n_angles: Number of all angles :math:`N_a`.
    :vartype n_angles: :class:`int`

    :ivar sinogram_shape: Represents the number of considered
        detectors (**n_detectors**) and angles (**n_angles**).
    :vartype sinogram_shape: :class:`tuple` :math:`(N_s,N_a)`

    :ivar delta_x: 	Physical width and height :math:`\\delta_x` of
        the image pixels.
    :vartype delta_x:  :class:`float`

    :ivar delta_s:  Physical width :math:`\\delta_s` of a detector pixel.
    :vartype delta_s:  :class:`float`

    :ivar delta_ratio:  Ratio :math:`{\\delta_s}/{\\delta_x}`,
        i.e. the detector
        pixel width relative to unit image pixels.
    :vartype delta_ratio:  :class:`float`

    :ivar angle_weights: Represents the angular discretization
        width for each angle which are used to weight the projections.
        In the fullangle case, these sum up to
        :math:`\\pi` and :math:`2\\pi` for parallel beam and
        fanbeam geometry respectively
        (or more specific angular when angular_ranges is set).
    :vartype angle_weights: :class:`numpy.ndarray`

    :ivar prg:  OpenCL program containing the gratopy OpenCL kernels.
        For the corresponding code, see :class:`gratopy.create_code`
    :vartype prg:  :class:`gratopy.Program`

    :ivar struct: Various data used in the projection operator.
        Contains in particular a
        :class:`pyopencl.array.Array`
        with the angular information necessary for computations.
    :vartype struct: :class:`list`, see :func:`radon_struct` and
        :func:`fanbeam_struct` returns
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
    # 2 3   ... vector from source vector connecting source to
    #    center of rotation
    # 4 5 ... vector connecting the origin to the detectorline
    # 6   ... Angular weights
    # :vartype ofs_buf: dictionary  from  numpy.dtype  to  numpy.Array or
    #    pyopencl.
    # :ivar sdpd: representing the weight sqrt(delta_s^2+R^2) required for
    # the computation of fanbeam transform
    # :vartype sdpd: dictionary  from  numpy.dtype  to  numpy.Array or
    #   pyopencl.

    def __init__(self, queue, geometry, img_shape, angles,
                 n_detectors=None, detector_width=2.0,
                 image_width=None, R=None, RE=None,
                 detector_shift=0.0, midpoint_shift=[0., 0.],
                 fullangle=True,  angular_range=[], reverse_detector=False):
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
        self.geometry = geometry
        self.queue = queue

        # build program containing opencl code
        self.adjusted_code = create_code()
        self.prg = Program(queue.context, self.adjusted_code)

        if np.isscalar(img_shape):
            img_shape = (img_shape, img_shape)

        # only planar img_shape is of relevance
        if len(img_shape) > 2:
            img_shape = img_shape[0:2]
        self.img_shape = img_shape

        self.image_width = image_width

        # Check that given geometry is indeed available
        if self.geometry not in [RADON, PARALLEL, FAN, FANBEAM]:
            raise ValueError("unknown projection_type, projection_type "
                             + "must be PARALLEL or FAN")

        if self.geometry in [RADON, PARALLEL]:
            self.is_parallel = True
            self.is_fan = False

            # set relevant forward and backprojection functions
            self.forwardprojection = radon
            self.backprojection = radon_ad

            # The kernel-functions according to the possible data types
            float32 = np.dtype('float32')
            float64 = np.dtype('float64')
            self.functions = {(float32, 0, 0): self.prg.radon_float_ff,
                              (float32, 1, 0): self.prg.radon_float_cf,
                              (float32, 0, 1): self.prg.radon_float_fc,
                              (float32, 1, 1): self.prg.radon_float_cc,
                              (float64, 0, 0): self.prg.radon_double_ff,
                              (float64, 1, 0): self.prg.radon_double_cf,
                              (float64, 0, 1): self.prg.radon_double_fc,
                              (float64, 1, 1): self.prg.radon_double_cc}
            self.functions_ad = {(float32, 0, 0): self.prg.radon_ad_float_ff,
                                 (float32, 1, 0): self.prg.radon_ad_float_cf,
                                 (float32, 0, 1): self.prg.radon_ad_float_fc,
                                 (float32, 1, 1): self.prg.radon_ad_float_cc,
                                 (float64, 0, 0): self.prg.radon_ad_double_ff,
                                 (float64, 1, 0): self.prg.radon_ad_double_cf,
                                 (float64, 0, 1): self.prg.radon_ad_double_fc,
                                 (float64, 1, 1): self.prg.radon_ad_double_cc}

        if self.geometry in [FAN, FANBEAM]:
            self.is_parallel = False
            self.is_fan = True

            # set relevant forward and backprojection functions
            self.forwardprojection = fanbeam
            self.backprojection = fanbeam_ad

            # The kernel-functions according to the possible data types
            float32 = np.dtype('float32')
            float64 = np.dtype('float64')
            self.functions = {(float32, 0, 0): self.prg.fanbeam_float_ff,
                              (float32, 1, 0): self.prg.fanbeam_float_cf,
                              (float32, 0, 1): self.prg.fanbeam_float_fc,
                              (float32, 1, 1): self.prg.fanbeam_float_cc,
                              (float64, 0, 0): self.prg.fanbeam_double_ff,
                              (float64, 1, 0): self.prg.fanbeam_double_cf,
                              (float64, 0, 1): self.prg.fanbeam_double_fc,
                              (float64, 1, 1): self.prg.fanbeam_double_cc}
            self.functions_ad = {
                (float32, 0, 0): self.prg.fanbeam_ad_float_ff,
                (float32, 1, 0): self.prg.fanbeam_ad_float_cf,
                (float32, 0, 1): self.prg.fanbeam_ad_float_fc,
                (float32, 1, 1): self.prg.fanbeam_ad_float_cc,
                (float64, 0, 0): self.prg.fanbeam_ad_double_ff,
                (float64, 1, 0): self.prg.fanbeam_ad_double_cf,
                (float64, 0, 1): self.prg.fanbeam_ad_double_fc,
                (float64, 1, 1): self.prg.fanbeam_ad_double_cc}

        # angles are chosen automatically as equidistant in suitable
        # angular-range when scalar angle is given
        if np.isscalar(angles):
            if self.is_fan:
                angles = np.linspace(0, 2*np.pi, angles+1)[:-1]
            elif self.is_parallel:
                angles = np.linspace(0, np.pi, angles+1)[:-1]
        angles = angles

        # Choosing n_detectors by default as diagonal of image pixels
        if n_detectors is None:
            self.n_detectors = int(np.ceil(np.hypot(img_shape[0],
                                                    img_shape[1])))
        else:
            self.n_detectors = n_detectors
        detector_width = float(detector_width)

        # extract information in case of multiple angle_section
        if isinstance(angles[0], list) or isinstance(angles[0], np.ndarray):
            self.n_angles = 0
            for j in range(len(angles)):
                self.n_angles += len(angles[j])
        else:
            self.n_angles = len(angles)
        self.angles = angles
        self.angular_range=angular_range
        self.sinogram_shape = (self.n_detectors, self.n_angles)

        self.fullangle = fullangle

        self.detector_shift = detector_shift
        self.midpoint_shift = midpoint_shift

        self.detector_width = detector_width
        self.R = R
        self.RE = RE

        self.buf_upload = {}

        self.reverse_detector = reverse_detector
        if ((self.reverse_detector) and (self.is_parallel)):
            print("WARNING, the reverse_detector argument has no impact"
                  + " on the parallel beam setting. To reverse the angles,"
                  + " the angles parameter can be translated by np.pi")

        if self.is_fan:
            parameters_available = not ((R is None) or (RE is None))

            assert parameters_available, ("For the Fanbeam geometry "
                                          + "you need to set R (the normal "
                                          + "distance from source to detector)"
                                          + " and RE (distance from source to "
                                          + "coordinate origin which is the "
                                          + "rotation center)")

            self.struct = fanbeam_struct(self.queue, self.img_shape,
                                         self.angles, self.detector_width,
                                         R, self.RE, self.n_detectors,
                                         self.detector_shift, image_width,
                                         self.midpoint_shift, self.fullangle,
                                         self.reverse_detector,
                                         self.angular_range)

            self.ofs_buf = self.struct[2]
            self.sdpd_buf = self.struct[3]
            self.image_width = self.struct[4]
            self.geometry_information = self.struct[5]

            self.angle_weights_buf = self.struct[6]
            self.angle_weights = self.angle_weights_buf[
                                            np.dtype("float")].copy()

            self.angular_range = self.struct[7]

            self.delta_x = self.image_width/max(img_shape)
            self.delta_s = detector_width/n_detectors
            self.delta_ratio = self.delta_s/self.delta_x

        if self.is_parallel:
            if image_width is None:
                self.image_width = 2.

            self.struct = radon_struct(self.queue, self.img_shape,
                                       self.angles,
                                       n_detectors=self.n_detectors,
                                       detector_width=self.detector_width,
                                       image_width=self.image_width,
                                       midpoint_shift=self.midpoint_shift,
                                       detector_shift=self.detector_shift,
                                       fullangle=self.fullangle,
                                       angular_range=self.angular_range)

            self.ofs_buf = self.struct[0]

            self.delta_x = self.image_width/max(self.img_shape)
            self.delta_s = self.detector_width/self.n_detectors
            self.delta_ratio = self.delta_s/self.delta_x

            self.geometry_information = self.struct[3]
            self.angle_weights_buf = self.struct[4]
            self.angle_weights = self.angle_weights_buf[
                                            np.dtype("float")].copy()
            self.angular_range = self.struct[5]

    def ensure_dtype(self, dtype):
        if dtype not in self.buf_upload:
            upload_bufs(self, dtype)
            self.buf_upload[dtype] = 1

    def show_geometry(self, angle, figure=None, axes=None, show=True,
                      img=None):
        """ Visualize the geometry associated with the projection settings.
        This can be useful in checking that indeed, the correct input
        for the desired geometry was given.

        :param angle: The angle for which the projection
            is considered.
        :type angle: :class:`float`

        :param figure: Figure in which to plot. If neither **figure** nor
            **axes** are given, a new figure (``figure(0)``) will be created.
        :type figure: :class:`matplotlib.figure.Figure`, default :obj:`None`

        :param axes: Axes to plot into. If :obj:`None`, a new
            axes inside the figure is created.
        :type axes: :class:`matplotlib.axes.Axes`, default :obj:`None`

        :param show:
            Determines whether the resulting plot is immediately
            shown (:obj:`True`).
            If :obj:`False`, :func:`matplotlib.pyplot.show` can be used
            at a later point to show the figure.
        :type show: :class:`bool`, default :obj:`True`

        :return: Figure and axes in which the geometry visualization
            is plotted.

        :rtype: (:class:`matplotlib.figure.Figure`,
            :class:`matplotlib.axes.Axes`)
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
            detector_width = self.detector_width
            source_detector_dist = self.R
            source_origin_dist = self.RE
            image_width = self.image_width
            midpoint_shift = self.midpoint_shift
            midpoint_shift = [midpoint_shift[1], midpoint_shift[0]]

            maxsize = max(self.RE, np.sqrt((self.R-self.RE)**2
                                           + detector_width**2/4.))

            angle = -angle
            A = np.array([[np.cos(angle), np.sin(angle)],
                          [-np.sin(angle), np.cos(angle)]])
            # Plot all relevant sizes            axes

            sourceposition = [-source_origin_dist, 0]
            upper_detector = [source_detector_dist-source_origin_dist,
                              detector_width*0.5+self.detector_shift]
            lower_detector = [source_detector_dist-source_origin_dist,
                              - detector_width*0.5+self.detector_shift]
            central_detector = [
               source_detector_dist-source_origin_dist, 0]

            sourceposition = np.dot(A, sourceposition)
            upper_detector = np.dot(A, upper_detector)
            lower_detector = np.dot(A, lower_detector)
            central_detector = np.dot(A, central_detector)

            axes.plot([upper_detector[0], lower_detector[0]],
                      [upper_detector[1], lower_detector[1]], "k")

            axes.plot([sourceposition[0], upper_detector[0]],
                      [sourceposition[1], upper_detector[1]], "g")
            axes.plot([sourceposition[0], lower_detector[0]],
                      [sourceposition[1], lower_detector[1]], "g")

            axes.plot([sourceposition[0], central_detector[0]],
                      [sourceposition[1], central_detector[1]], "g")

            # plot(x[0]+midpoint_rotation[0],x[1]+midpoint_rotation[1],"b")

            draw_circle = matplotlib.patches.Circle(
                midpoint_shift,
                image_width/2 * np.sqrt(1 + (min(self.img_shape)
                                        / max(self.img_shape))**2),
                color='r')

            axes.add_artist(draw_circle)

            color = (1, 1, 0)
            rect = plt.Rectangle(midpoint_shift-0.5
                                 * np.array([image_width*self.img_shape[1]
                                             / np.max(self.img_shape),
                                             image_width*self.img_shape[0]
                                             / np.max(self.img_shape)]),
                                 image_width
                                 * self.img_shape[1]
                                 / np.max(self.img_shape),
                                 image_width
                                 * self.img_shape[0]
                                 / np.max(self.img_shape),
                                 facecolor=color, edgecolor=color)

            axes.add_artist(rect)

            draw_circle = matplotlib.patches.Circle(midpoint_shift,
                                                    image_width/2, color='b')
            axes.add_artist(draw_circle)

            draw_circle = matplotlib.patches.Circle((0, 0), image_width/10,
                                                    color='k')
            axes.add_artist(draw_circle)
            axes.set_xlim([-maxsize, maxsize])
            axes.set_ylim([-maxsize, maxsize])

        if self.is_parallel:
            detector_width = self.detector_width
            image_width = self.image_width
            midpoint_shift = self.midpoint_shift
            midpoint_shift = [midpoint_shift[1], midpoint_shift[0]]

            angle = -angle
            A = np.array([[np.cos(angle), np.sin(angle)],
                          [-np.sin(angle), np.cos(angle)]])

            center_source = [-image_width, self.detector_shift]
            center_detector = [image_width, self.detector_shift]

            upper_source = [-image_width,
                            self.detector_shift+0.5*detector_width]
            lower_source = [-image_width,
                            self.detector_shift-0.5*detector_width]

            upper_detector = [image_width,
                              self.detector_shift+0.5*detector_width]
            lower_detector = [image_width,
                              self.detector_shift-0.5*detector_width]

            center_source = np.dot(A, center_source)
            center_detector = np.dot(A, center_detector)
            upper_source = np.dot(A, upper_source)
            lower_source = np.dot(A, lower_source)
            upper_detector = np.dot(A, upper_detector)
            lower_detector = np.dot(A, lower_detector)

            axes.plot([center_source[0], center_detector[0]],
                      [center_source[1], center_detector[1]], "g")

            axes.plot([lower_source[0], lower_detector[0]],
                      [lower_source[1], lower_detector[1]], "g")
            axes.plot([upper_source[0], upper_detector[0]],
                      [upper_source[1], upper_detector[1]], "g")

            axes.plot([lower_detector[0], upper_detector[0]],
                      [lower_detector[1], upper_detector[1]], "k")

            draw_circle = matplotlib.patches.Circle(midpoint_shift,
                                                    image_width/np.sqrt(2),
                                                    color='r')

            axes.add_artist(draw_circle)

            color = (1, 1, 0)
            draw_rectangle = matplotlib.patches.Rectangle(
                                midpoint_shift
                                - 0.5*np.array([image_width*self.img_shape[1]
                                                / np.max(self.img_shape),
                                                image_width
                                                * self.img_shape[0]
                                                / np.max(self.img_shape)]),
                                image_width * self.img_shape[1]
                                / np.max(self.img_shape),
                                image_width * self.img_shape[0]
                                / np.max(self.img_shape),
                                facecolor=color,
                                edgecolor=color)

            axes.add_artist(draw_rectangle)

            draw_circle = matplotlib.patches.Circle(midpoint_shift,
                                                    image_width/2, color='b')

            axes.add_artist(draw_circle)
            draw_circle = matplotlib.patches.Circle((0, 0), image_width/10,
                                                    color='k')

            axes.add_artist(draw_circle)

            maxsize = np.sqrt(image_width**2+detector_width**2)
            axes.set_xlim([-maxsize, maxsize])
            axes.set_ylim([-maxsize, maxsize])

        if img:
            figure.figimage(img, 0, 0)
        if show and (figure is not None):
            figure.show()
        return figure, axes

    def create_sparse_matrix(self, dtype=np.dtype('float32'), order='F'):
        """
        Creates a sparse matrix representation of the associated forward
        operator.

        :param dtype: Precision to compute the sparse representation in.
        :type dtype: :class:`numpy.dtype`, default :attr:`numpy.float32`

        :param order: Contiguity of the image and sinogram array
            to the transform, can be ``F`` or ``C``.
        :type order: :class:`str`, default ``F``

        :return: Sparse matrix corresponding to the
            forward projection.
        :rtype: :class:`scipy.sparse.coo_matrix`

        Note that for high resolution projection operators,
        this may require infeasibly much time and memory.

        """

        dtype = np.dtype(dtype)
        if self.is_parallel:
            functions = {
                (np.dtype("float32"), 0): self.prg.single_line_radon_float_ff,
                (np.dtype("float32"), 1): self.prg.single_line_radon_float_cc,
                (np.dtype("float64"), 0): self.prg.single_line_radon_double_ff,
                (np.dtype("float64"), 1): self.prg.single_line_radon_double_cc
                }
        elif self.is_fan:
            functions = {
                (np.dtype("float32"), 0): self.prg.single_line_fan_float_ff,
                (np.dtype("float32"), 1): self.prg.single_line_fan_float_cc,
                (np.dtype("float64"), 0): self.prg.single_line_fan_double_ff,
                (np.dtype("float64"), 1): self.prg.single_line_fan_double_cc
                }

        function = functions[(np.dtype(dtype), order == 'C')]

        self.ensure_dtype(dtype)
        ofs_buf = self.ofs_buf[dtype]
        if self.is_fan:
            sdpd_buf = self.sdpd_buf[dtype]

        geometry_information = self.geometry_information[dtype]

        if self.is_parallel:
            def projection_from_single_pixel(x, y, sino=None, wait_for=[]):
                myevent = function(sino.queue, sino.shape, None,
                                   sino.data, np.int32(
                                       x), np.int32(y), ofs_buf,
                                   geometry_information,
                                   wait_for=sino.events+wait_for)
                sino.add_event(myevent)
        else:
            def projection_from_single_pixel(x, y, sino=None, wait_for=[]):
                myevent = function(sino.queue, sino.shape, None,
                                   sino.data, np.int32(x), np.int32(
                                       y), ofs_buf, sdpd_buf,
                                   geometry_information,
                                   wait_for=sino.events+wait_for)
                sino.add_event(myevent)

        epsilon = 0
        if order == "F":
            def pos_1(x, y):
                return x+Nx*y

            def pos_2(s, phi):
                return s+Ns*phi
        elif order == "C":
            def pos_1(x, y):
                return x*Ny+y

            def pos_2(s, phi):
                return s*Na+phi
        else:
            print("Order (contiguity) not recognized, suitable choices are"
                  + "'F' or 'C'")
            raise

        mylist = []

        Nx = self.img_shape[0]
        Ny = self.img_shape[1]
        Ns = self.sinogram_shape[0]
        Na = self.sinogram_shape[1]

        img = clarray.zeros(self.queue, self.img_shape, dtype=dtype,
                            order=order)

        rows = []
        cols = []
        vals = []
        sino = forwardprojection(img, self)

        for x in range(Nx):
            if x % int(Nx/100.) == 0:
                sys.stdout.write('\rProgress at {:3.0%}'
                                 .format(float(x)/Nx))

            for y in range(Ny):
                projection_from_single_pixel(x, y, sino)
                sinonew = sino.get()
                pos = pos_1(x, y)

                index = np.where(sinonew > epsilon)
                for i in range(len(index[0])):
                    s = index[0][i]
                    phi = index[1][i]
                    pos2 = pos_2(s, phi)
                    mylist.append(str(pos2)+" "+str(pos)+" "
                                  + str(sinonew[s, phi])+"\n")
                    rows.append(pos2)
                    cols.append(pos)
                    vals.append(sinonew[s, phi])

        print("\rSparse matrix creation complete")

        sparsematrix = scipy.sparse.coo_matrix((vals, (rows, cols)),
                                               shape=(Ns*Na, Nx*Ny))

        return sparsematrix


def weight_sinogram(sino, projectionsetting, sino_out=None, divide=False,
                    wait_for=[]):
    """
    Performs an angular rescaling of a given sinogram via multiplication 
    (or division) with the projection's angle weights (size of projections in 
    angle dimension, see attributes of :class:`ProjectionSettings`)
    to the respective projections.
    This can be useful, e.g., for computing norms or dual
    pairings in the appropriate Hilbert space.

    :param sino: The sinogram whose rescaling is computed.
        This array itself remains unchanged unless the same array is given
        as **sino_out**.
    :type img: :class:`pyopencl.array.Array`
    :param projectionsetting: The geometry settings for which the rescaling
        is computed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`
    :param sino_out: The array in which the result of rescaling
        is saved. If :obj:`None` (per default) is given, a new array
        will be created and returned. When giving the same array as 
        **sino**, the values in **sino** will be overwritten.
    :type sino: :class:`pyopencl.array.Array` default :obj:`None`

    :param divide: Determines whether the sinogram is multiplied
        or divided by the angular weights. If :obj:`True`, 
        a division is performed, otherwise, the weights are multiplied.
    :type divide: :class:`bool`, default :obj:`False`

    :param wait_for: The events to wait for before performing the
        computation in order to avoid, e.g., race conditions, see
        :class:`pyopencl.Event`.
    :type wait_for: :class:`list[pyopencl.Event]`, default :attr:`[]`

    :return: The weighted sinogram.
        If **sino_out** is not :obj:`None`, it
        is returned with the values in its data overwritten.
    :rtype: :class:`pyopencl.array.Array`
    """

    dtype = sino.dtype
    my_order = {0: 'F', 1: 'C'}[sino.flags.c_contiguous]

    if sino_out is None:
        sino_out = clarray.zeros(sino.queue, sino.shape, dtype=dtype,
                                 order=my_order)
    if divide is False:
        functions = {
            (np.dtype("float32"), "C"): projectionsetting.prg.multiply_float_c,
            (np.dtype("float32"), "F"): projectionsetting.prg.multiply_float_f,
            (np.dtype("float64"), "C"): projectionsetting.prg.multiply_double_c,
            (np.dtype("float64"), "F"): projectionsetting.prg.multiply_double_f
            }
    elif divide is True:
        functions = {
            (np.dtype("float32"), "C"): projectionsetting.prg.divide_float_c,
            (np.dtype("float32"), "F"): projectionsetting.prg.divide_float_f,
            (np.dtype("float64"), "C"): projectionsetting.prg.divide_double_c,
            (np.dtype("float64"), "F"): projectionsetting.prg.divide_double_f
            }
    function = functions[dtype, my_order]

    projectionsetting.ensure_dtype(dtype)

    myevent = function(sino.queue, sino.shape, None, sino.data,
                       projectionsetting.angle_weights_buf[dtype],
                       sino_out.data,
                       wait_for=wait_for+sino_out.events+sino.events)
    sino_out.add_event(myevent)
    return sino_out


def equ_mul_add(rhs, a, x, projectionsetting, wait_for=[]):
    dtype = x.dtype
    function = {np.dtype("float32"): projectionsetting.prg.equ_mul_add_float_c,
                np.dtype("float64"): projectionsetting.prg.equ_mul_add_double_c
                }[dtype]
    function(x.queue, [x.size], None, rhs.data, a, x.data,
             wait_for=x.events+rhs.events+wait_for)
    return rhs


def mul_add_add(rhs, a, x, y, projectionsetting, wait_for=[]):
    dtype = x.dtype
    function = {np.dtype("float32"): projectionsetting.prg.mul_add_add_float_c,
                np.dtype("float64"): projectionsetting.prg.mul_add_add_double_c
                }[dtype]
    function(x.queue, [x.size], None, rhs.data, a, x.data, y.data,
             wait_for=x.events+y.events+rhs.events+wait_for)
    return rhs


def normest(projectionsetting, number_iterations=50, dtype='float32',
            allocator=None):
    """
    Estimate the spectral norm of the projection operator via power
    iteration, i.e., the operator norm with respect to the standard
    Euclidean or :math:`\\ell^2`
    norms. Useful for iterative methods that require such an estimate,
    e.g., :func:`landweber` or :func:`total_variation`.

    :param projectionsetting: The geometry settings for which the projection
        is considered.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param number_iterations:  The number of iterations to
        terminate after.
    :type number_iterations: :class:`int`, default 50
    :param dtype:  Precision for which to apply the projection operator
        (which is not supposed to impact the estimate significantly).
    :type dtype: :class:`numpy.dtype`, default :attr:`numpy.float32`

    :return: An estimate of the spectral norm for the projection operator.
    :rtype: :class:`float`

    """

    queue = projectionsetting.queue

    # random starting point
    img = clarray.to_device(queue, np.require((np.random.randn(
                    *projectionsetting.img_shape)), dtype, 'F'),
                    allocator=allocator)
    sino = forwardprojection(img, projectionsetting)

    # power_iteration
    for i in range(number_iterations):
        normsqr = float(clarray.sum(img).get())
        img /= normsqr
        forwardprojection(img, projectionsetting, sino=sino)
        backprojection(sino, projectionsetting, img=img)
    return np.sqrt(normsqr)


def landweber(sino, projectionsetting, number_iterations=100, w=1):
    """
    Performs a Landweber iteration [L1951]_ to approximate
    a solution to the image reconstruction problem associated
    with a projection and sinogram. This method is also known as SIRT.

    :param sino: Sinogram data to reconstruct from.
    :type sino: :class:`pyopencl.array.Array`

    :param projectionsetting: The geometry settings for which the projection
        is considered.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param number_iterations: Number of iteration steps to be performed.
    :type number_iterations: :class:`int`, default 100

    :param w: Relaxation parameter weighted by the norm of the projection
        operator (w<1 guarantees convergence).
    :type w:  :class:`float`, default 1

    :return: Reconstruction from given sinogram gained via Landweber
        iteration.
    :rtype: :class:`pyopencl.array.Array`

    .. [L1951] Landweber, L. "An iteration formula for Fredholm integral
               equations of the first kind." Amer. J. Math. 73, 615–624
               (1951). https://doi.org/10.2307/2372313
    """
    # Order to consider solution in
    my_order = {0: 'F', 1: 'C'}[sino.flags.c_contiguous]

    # Set relaxation parameter
    norm_estimate = normest(
                projectionsetting, allocator=sino.allocator)
    w = sino.dtype.type(w/norm_estimate**2)

    sinonew = sino.copy()

    U = w*backprojection(sinonew, projectionsetting)
    Unew = clarray.zeros(projectionsetting.queue, U.shape, dtype=sino.dtype,
                         order=my_order, allocator=sino.allocator)

    # Poweriteration
    for i in range(number_iterations):
        sys.stdout.write('\rProgress at {:3.0%}'
                         .format(float(i)/number_iterations))

        forwardprojection(Unew, projectionsetting, sino=sinonew)
        sinonew -= sino
        # Unew = Unew-w*backprojection(sinonew, projectionsetting, img=U)
        equ_mul_add(Unew, -w,
                    backprojection(sinonew, projectionsetting, img=U),
                    projectionsetting, wait_for=[])
    print('\rLandweber reconstruction complete')
    return Unew


def conjugate_gradients(sino, projectionsetting, number_iterations=20,
                        epsilon=0.0, x0=None):
    """
    Performs a conjugate gradients iteration [HS1952]_ to approximate
    a solution to the image reconstruction problem associated
    with a projection and sinogram.

    :param sino: Sinogram data to invert.
    :type sino: :class:`pyopencl.array.Array`

    :param projectionsetting: The geometry settings for which the projection
        is considered.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param number_iterations: Maximal number of iteration steps
        to be performed.
    :type number_iterations: :class:`float`, default 20

    :param x0: Initial guess for iteration (defaults to zeros if
        :obj:`None`).
    :type x0: :class:`pyopencl.array.Array`, default :obj:`None`

    :param epsilon: Tolerance parameter, the iteration stops if
        relative residual<epsilon.
    :type epsilon: :class:`float`, default 0.00

    :return: Reconstruction gained via conjugate gradients iteration.
    :rtype:  :class:`pyopencl.array.Array`

    .. [HS1952] Hestenes, M. R., Stiefel, E. "Methods of Conjugate Gradients
                for Solving Linear Systems." Journal of Research of
                the National
                Bureau of Standards, 49:409–436 (1952).
                https://doi.org/10.6028/jres.049.044
    """

    # Determine suitable dimensions
    dimensions = projectionsetting.img_shape
    dimensions2 = (1, projectionsetting.n_angles)
    if len(sino.shape) > 2:
        dimensions = dimensions+tuple([sino.shape[2]])
        dimensions2 = dimensions2+tuple([1])

    order = order = {0: 'F', 1: 'C'}
    if x0 is None:
        x0 = clarray.zeros(projectionsetting.queue, dimensions,
                           sino.dtype, order[sino.flags.c_contiguous])

    if (x0.flags.c_contiguous != sino.flags.c_contiguous):
        raise ValueError("The data sino and initial "
                         + "guess x0 must have the same contiguity!")
    x = x0.copy()

    d = sino-forwardprojection(x, projectionsetting)
    p = backprojection(d, projectionsetting)
    q = clarray.empty_like(d, projectionsetting.queue)
    q_rescaled=q.copy()
    snew = backprojection(d, projectionsetting)
    sold = snew.copy()

    order = {0: 'F', 1: 'C'}[sino.flags.c_contiguous]

    for k in range(0, number_iterations):
        sys.stdout.write('\rProgress at {:3.0%}'
                         .format(float(k)/number_iterations))

        forwardprojection(p, projectionsetting, sino=q)
        alpha = x.dtype.type(projectionsetting.delta_x**2
                             / (projectionsetting.delta_s)
                             * (clarray.vdot(sold, sold)
                                / clarray.vdot(weight_sinogram(q,
                                                projectionsetting, q_rescaled)
                                               , q)).get())

        equ_mul_add(x, +alpha, p, projectionsetting)  # x += alpha*p
        equ_mul_add(d, -alpha, q, projectionsetting)  # d -= alpha*q
        backprojection(d, projectionsetting, img=snew)
        beta = (clarray.vdot(snew, snew)
                / clarray.vdot(sold, sold)).get()
        (sold, snew) = (snew, sold)
        mul_add_add(p, beta, p, sold, projectionsetting)  # p = beta*p+sold
        residue = np.sqrt(np.sum(clarray.vdot(sold, sold).get())
                          / np.sum(clarray.vdot(sino, sino).get()))
        if residue < epsilon:
            print('\rProgress aborted prematurely as desired'
                             + 'precision is reached')
            break

    print("\rCG reconstruction complete")

    return x


def total_variation(sino, projectionsetting, mu,
                    number_iterations=1000, slice_thickness=1,
                    stepsize_weighting=10.):
    """
    Peforms a primal-dual algorithm [CP2011]_ to solve a total-variation
    regularized reconstruction problem associated with a given
    projection operator and sinogram. This corresponds to the approximate
    solution of
    :math:`\\min_{u} {\\frac\\mu2}\\|\\mathcal{P}u-f\\|_{L^2}^2+\\mathrm{TV}(u)`
    for :math:`\\mathcal{P}` the projection operator, :math:`f` the sinogram
    and :math:`\\mu` a positive regluarization parameter (i.e.,
    an :math:`L^2-\\mathrm{TV}` reconstruction approach).

    :param sino: Sinogram data to invert.
    :type sino: :class:`pyopencl.array.Array`

    :param projectionsetting: The geometry settings for which the projection
        is considered.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`

    :param mu: Regularization parameter, the smaller the stronger the
        applied regularization.
    :type epsilon: :class:`float`

    :param number_iterations: Number of iterations to be performed.
    :type number_iterations: :class:`float`, default 1000

    :param slice_thickness: When 3-dimensional data sets are considered,
        regularization is also applied across slices.
        This parameter represents the ratio of the slice thickness to the
        length of one pixel within a slice. The choice
        **slice_thickness** =0
        results in no coupling across slices.
    :type slice_thickness: :class:`float`, default 1.0, i.e., isotropic voxels

    :param stepsize_weighting: Allows to weight the primal-dual algorithm's
        step sizes :math:`\\sigma` and :math:`\\tau`
        (with :math:`\\sigma\\tau\\|\\mathcal{P}\\|^2\\leq 1`)
        by multiplication and division, respectively,
        with the given value.
    :type stepsize_weighting: :class:`float`, default 10.0


    :return: Reconstruction gained via primal-dual iteration for the
        total-variation regularized reconstruction problem.
    :rtype:  :class:`pyopencl.array.Array`

    .. [CP2011] Chambolle, A., Pock, T. "A First-Order Primal-Dual Algorithm
                for Convex Problems with Applications to Imaging." J Math
                Imaging Vis 40, 120–145 (2011).
                https://doi.org/10.1007/s10851-010-0251-1
    """
    # Establish queue and context

    # preliminary definitions and parameters
    queue = projectionsetting.queue
    ctx = queue.context
    my_dtype = sino.dtype
    my_order = {0: 'F', 1: 'C'}[sino.flags.c_contiguous]

    img_shape = projectionsetting.img_shape
    if len(sino.shape) == 2:
        slice_thickness = 0
    else:
        img_shape = img_shape+tuple([sino.shape[2]])
    extended_img_shape = tuple([4])+img_shape

    # Definitions of suitable kernel functions for primal and dual updates

    # update dual variable to data term
    update_lambda_ = {
            (np.dtype("float32"), 0):
                projectionsetting.prg.update_lambda_L2_float_f,
            (np.dtype("float32"), 1):
                projectionsetting.prg.update_lambda_L2_float_c,
            (np.dtype("float"), 0):
                projectionsetting.prg.update_lambda_L2_double_f,
            (np.dtype("float"), 1):
                projectionsetting.prg.update_lambda_L2_double_c}

    update_lambda = lambda lamb, Ku, f, sigma, mu, normest, wait_for=[]: \
        lamb.add_event(update_lambda_[lamb.dtype, lamb.flags.c_contiguous]
                       (lamb.queue, lamb.shape, None, lamb.data, Ku.data,
                        f.data, np.float32(
                                sigma/normest), np.float32(mu),
                        wait_for=lamb.events + Ku.events + f.events+wait_for))

    # Update v the dual of gradient of u
    update_v_ = {
            (np.dtype("float32"), 0): projectionsetting.prg.update_v_float_f,
            (np.dtype("float32"), 1): projectionsetting.prg.update_v_float_c,
            (np.dtype("float"), 0): projectionsetting.prg.update_v_double_f,
            (np.dtype("float"), 1): projectionsetting.prg.update_v_double_c}
    update_v = lambda v, u, sigma, slice_thickness, wait_for=[]: \
        v.add_event(update_v_[v.dtype, v.flags.c_contiguous]
                    (v.queue, u.shape, None, v.data, u.data, np.float32(sigma),
                         np.float32(slice_thickness),
                         wait_for=v.events+u.events+wait_for))

    # Update primal variable u (the image)
    update_u_ = {
            (np.dtype("float32"), 0): projectionsetting.prg.update_u_float_f,
            (np.dtype("float32"), 1): projectionsetting.prg.update_u_float_c,
            (np.dtype("float"), 0): projectionsetting.prg.update_u_double_f,
            (np.dtype("float"), 1): projectionsetting.prg.update_u_double_c}
    update_u = lambda u, u_, v, Kstarlambda, tau, normest, slice_thickness, \
        wait_for=[]: \
        u_.add_event(update_u_[u.dtype, u.flags.c_contiguous]
                     (u.queue, u.shape, None, u.data, u_.data, v.data,
                      Kstarlambda.data, np.float32(
                              tau), np.float32(1.0/normest),
                      np.float32(slice_thickness),
                      wait_for=u.events+u_.events+v.events+wait_for))

    # Compute the norm of v and project (dual update)
    update_NormV_ = {
            (np.dtype("float32"), 0):
            projectionsetting.prg.update_NormV_unchor_float_f,
            (np.dtype("float32"), 1):
            projectionsetting.prg.update_NormV_unchor_float_c,
            (np.dtype("float"), 0):
            projectionsetting.prg.update_NormV_unchor_double_f,
            (np.dtype("float"), 1):
            projectionsetting.prg.update_NormV_unchor_double_c}

    update_NormV = lambda V, normV, wait_for=[]:\
        normV.add_event(update_NormV_[V.dtype, V.flags.c_contiguous]
                        (V.queue, V.shape[1:], None, V.data, normV.data,
                         wait_for=V.events+normV.events+wait_for))

    # update of the extra gradient
    update_extra_ = {np.dtype("float32"): cl.elementwise.ElementwiseKernel(
            ctx, 'float *u_, float *u', 'u[i] = 2.0f*u_[i] - u[i]'),
            np.dtype("float"): cl.elementwise.ElementwiseKernel
            (ctx, 'double *u_, double *u', 'u[i] = 2.0f*u_[i] - u[i]')}
    update_extra_ = update_extra_[sino.dtype]

    def update_extra(u_, u): return \
        u.add_event(update_extra_(
                u_, u, wait_for=u.events + u_.events))

    # Initialize variables for the iteration
    U = clarray.zeros(queue, img_shape,
                      dtype=my_dtype, order=my_order)
    U_ = clarray.zeros(queue, img_shape,
                       dtype=my_dtype, order=my_order)
    V = clarray.zeros(queue, extended_img_shape, dtype=my_dtype,
                      order=my_order)

    Lamb = clarray.zeros(queue, sino.shape,
                         dtype=my_dtype, order=my_order)
    KU = clarray.zeros(queue, sino.shape,
                       dtype=my_dtype, order=my_order)
    KSTARlambda = clarray.zeros(queue, img_shape, dtype=my_dtype,
                                order=my_order)
    normV = clarray.zeros(
            queue, img_shape, dtype=my_dtype, order=my_order)

    # Compute estimates for step-size parameters
    norm_estimate = normest(projectionsetting)

    # Estimation of the operator [Grad,Proj] when Proj possesses norm 1
    Lsqr = 13
    sigma = 1.0/np.sqrt(Lsqr)*stepsize_weighting
    tau = 1.0/np.sqrt(Lsqr)/stepsize_weighting

    # Modifie mu for internal uses
    assert mu > 0, "Regularization parameter mu must be positive"
    # To counteract normalizing the operator, and delta_x comes Frobeniusnorm
    # the gradient
    mu *= norm_estimate**2*projectionsetting.delta_x
    mu = mu/(sigma+mu)

    # Primal-dual iteration
    for i in range(number_iterations):
        # Dual update
        update_v(V, U_, sigma, slice_thickness)
        update_NormV(V, normV)
        forwardprojection(U_, projectionsetting, sino=KU)
        update_lambda(Lamb, KU, sino, sigma, mu, norm_estimate)

        # Primal update
        backprojection(Lamb, projectionsetting, img=KSTARlambda)
        update_u(U_, U, V, KSTARlambda, tau, norm_estimate,
                 slice_thickness)

        # Extragradient update
        update_extra(U_, U)
        (U, U_) = (U_, U)

        sys.stdout.write('\rProgress at {:3.0%}'
                         .format(float(i)/number_iterations))

    print("\rTV reconstruction complete")
    return U
