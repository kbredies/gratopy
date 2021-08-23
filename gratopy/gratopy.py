# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021 Kristian Bredies (kristian.bredies@uni-graz.at)
#                       Richard Huber (richard.huber@uni-graz.at)
#
#    This file is part of gratopy (https://github.com/kbredies/gratopy).
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# unofficial Python2 compatibility
from __future__ import division, print_function

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pyopencl as cl
import pyopencl.array as clarray
import scipy
import scipy.sparse

# Version number
VERSION = '0.1.0'

# Source files for opencl kernels
CL_FILES1 = ["radon.cl", "fanbeam.cl"]
CL_FILES2 = ["total_variation.cl", "utilities.cl"]

# Class attribute corresponding to which geometry to consider
PARALLEL = 1
RADON = 1
FANBEAM = 2
FAN = 2


###########
# Program created from the gpu_code
class Program(object):
    def __init__(self, ctx, code):

        # activate warnings
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        # build OpenCL code
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
    assert (sino.dtype == img.dtype),\
        ("sinogram and image do not share"
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
            assert(len(img.shape) > 2),\
                (" The sinogram has a third dimension"
                 + "but the image does not.")
            assert(sino.shape[2] == img.shape[2]),\
                ("The third dimension"
                 + "(z-direction) of the sinogram is" + str(sino.shape[2])
                 + " and the image's is" + str(img.shape[2])
                 + ", they do not coincide.")

    if len(img.shape) > 2:
        if img.shape[2] > 1:
            assert(len(sino.shape) > 2),\
                (" The sinogram has a third dimension"
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
        :class:`pyopencl.Event`. This program will always wait for img.events
        and sino.events (so you need not add them to wait_for).
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
        :class:`pyopencl.Event`. This program will always wait for img.events
        and sino.events (so you need not add them to wait_for).
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
        This program will always wait for img.events
        and sino.events (so you need not add them to wait_for).
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

    # choose function with appropriate dtype
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
        This program will always wait for img.events
        and sino.events (so you need not add them to wait_for).
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

    # choose function with appropriate dtype
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


def radon_struct(queue, img_shape, angles, angle_weights, n_detectors=None,
                 detector_width=2.0, image_width=2.0,
                 midpoint_shift=[0, 0], detector_shift=0.0):
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
        projection.
    :type angles: :class:`numpy.ndarray`

    :param angle_weights: The weights associated to the angles, e.g.,
        how much of the angular range is covered by this angle.
        This impacts the weighting of rays for the backprojection.
    :type angle_weights: :class:`numpy.ndarray`

    :param n_detectors: The number :math:`N_s` of considered (equi-spaced)
        detectors. If :obj:`None`, :math:`N_s` will be chosen as
        :math:`\\sqrt{N_x^2+N_y^2}`.
    :type n_detectors:  :class:`int`, default :obj:`None`

    :param detector_width: Physical length of the detector line.
    :type detector_width: :class:`float`, default 2.0

    :param image_width: Physical size of the image indicated by the length of
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


    :return: Struct dictionary with the following variables as entries,
        where the keys are strings of the same names\\:
    :rtype: :class:`dict`

    :var ofs_dict:
        Dictionary containing the relevant angular information as
        :class:`numpy.ndarray` for the data types :attr:`numpy.float32`
        and :attr:`numpy.float64`.
        The arrays have dimension :math:`(8, N_a)` with columns:

        +---+------------------------+
        | 0 | weighted cosine        |
        +---+------------------------+
        | 1 | weighted sine          |
        +---+------------------------+
        | 2 | detector offset        |
        +---+------------------------+
        | 3 | inverse of cosine/sine |
        +---+------------------------+
        | 4 | angular weight         |
        +---+------------------------+
        | 5 | flipped                |
        +---+------------------------+

        The remaining columns are unused.
        The value **flipped** indicates whether the x and y
        axis are flipped (1) or not (0), which is done for
        reasons of numerical stability.
        The 4th entry contains the inverse of sine if the axes
        are flipped and the inverse of cosine otherwise.
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

    """

    # relative_detector_pixel_width is delta_s/delta_x
    relative_detector_pixel_width = detector_width/float(image_width)\
        * max(img_shape)/n_detectors

    # Choosing the number of detectors as the the diagonal
    # through the the image (in image_pixel scale) if none is given
    if n_detectors is None:
        nd = int(np.ceil(np.hypot(img_shape[0], img_shape[1])))
    else:
        nd = n_detectors

    # compute basic resolutions
    delta_x = image_width/float(max(img_shape))
    delta_s = float(detector_width)/nd

    # Compute the midpoints of geometries
    midpoint_domain = np.array([img_shape[0]-1, img_shape[1]-1])/2.0 -\
        np.array(midpoint_shift)/delta_x
    midpoint_detectors = (nd-1.0)/2.0

    # Vector in projection-direction (from source toward detector)
    X = np.cos(angles-np.pi*0.5)/relative_detector_pixel_width
    Y = np.sin(angles-np.pi*0.5)/relative_detector_pixel_width

    Xinv = np.zeros(X.size)
    Xinv = 1.0/X

    # set near vertical lines to horizontal
    mask = np.where(abs(X) < abs(Y))
    Xinv[mask] = 1.0/Y[mask]
    reverse_mask = np.zeros(Xinv.shape)
    reverse_mask[mask] = 1.
    # X*x+Y*y=detectorposition, offset is error in midpoint of
    # the sinogram (in shifted detector setting)
    offset = midpoint_detectors - X*midpoint_domain[0]\
        - Y*midpoint_domain[1] - detector_shift/delta_s

    # Save for datatype float64 and float32 the relevant additional information
    # required for the computations
    geo_dict = {}
    ofs_dict = {}
    angle_diff_dict = {}
    n_angles = len(angles)
    sinogram_shape = (nd, n_angles)
    for dtype in [np.dtype('float64'), np.dtype('float32')]:
        # save angular information into the ofs buffer
        ofs = np.zeros((8, len(angles)), dtype=dtype, order='F')
        ofs[0, :] = X
        ofs[1, :] = Y
        ofs[2, :] = offset
        ofs[3, :] = Xinv
        ofs[4, :] = angle_weights
        ofs[5, :] = reverse_mask
        ofs_dict[dtype] = ofs

        angle_diff_dict[dtype] = np.array(angle_weights, dtype=dtype)

        geometry_info = np.array([delta_x, delta_s, img_shape[0], img_shape[1],
                                  nd, n_angles], dtype=dtype, order='F')
        geo_dict[dtype] = geometry_info

    struct = {"ofs_dict": ofs_dict, "img_shape": img_shape,
              "sinogram_shape": sinogram_shape, "geo_dict": geo_dict,
              "angle_diff_dict": angle_diff_dict}
    return struct


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
        This program will always wait for img.events
        and sino.events (so you need not add them to wait_for).
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

    # choose function with appropriate dtype
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
        This program will always wait for img.events
        and sino.events (so you need not add them to wait_for).
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
                   angle_weights, n_detectors=None,
                   detector_shift=0.0, image_width=None,
                   midpoint_shift=[0, 0], reverse_detector=False):
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
        projection.
    :type angles: :class:`numpy.ndarray`

    :param detector_width: Physical length of the detector line.
    :type detector_width: :class:`float`, default 2.0

    :param source_detector_dist:  Physical (orthogonal) distance **R** from
        the source to the detector line.
    :type source_detector_dist: :class:`float`

    :param source_origin_dist: Physical distance **RE** from the source to the
        origin (center of rotation).
    :type source_origin_dist: :class:`float`

    :param angle_weights: The weights associated to the angles, e.g.,
        how much of the angular range is covered by this angle.
        This impacts the weighting of rays for the backprojection.
    :type angle_weights: :class:`numpy.ndarray`

    :param n_detectors: The number :math:`N_s` of considered (equi-spaced)
        detectors. If :obj:`None`, :math:`N_s` will be chosen as
        :math:`\\sqrt{N_x^2+N_y^2}`.
    :type n_detectors:  :class:`int` or :obj:`None`, default :obj:`None`

    :param detector_shift: Physical shift of the detector along
        the detector line in detector pixel offsets. Defaults to
        the application of no shift, i.e., the detector reaches from
        [- **detector_width**/2, **detector_width**/2].
    :type detector_shift: :class:`list[float]`, default 0.0

    :param image_width: Physical size of the image indicated by the length of
        the longer side of the rectangular image domain.
        If :obj:`None`, **image_width** is chosen to capture just
        all rays.
    :type image_width: :class:`float`, default :obj:`None`

    :param midpoint_shift: Two-dimensional vector representing the
        shift of the image away from center of rotation.
        Defaults to the application of no shift.
    :type midpoint_shift:  :class:`list[float]`, default [0.0, 0.0]

    :param reverse_detector: When :obj:`True`, the detector direction
        is flipped.
    :type reverse_detector: :class:`bool`, default :obj:`False`


    :return: Struct dictionary with the following variables as entries,
        where the keys are strings of the same names\\:
    :rtype: :class:`dict`

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
        [source detector distance/:math:`\\delta_x`,
        source origin distance/:math:`\\delta_x`,
        width of a detector_pixel relative to width of image_pixels
        i.e. :math:`\\delta_s`/:math:`\\delta_x`,
        image midpoint x-coordinate (in pixels),
        image midpoint y-coordinate (in pixels),
        detector line midpoint (in detector-pixels),
        :math:`N_x`, :math:`N_y`, :math:`N_s`,
        :math:`N_a`, width of an image pixel (:math:`\\delta_x`)].
    :vartype geo_dict: :class:`dict{numpy.dtype: numpy.ndarray}`

    :var angles_diff:
        Dictionary containing the
        same values as in **ofs_dict** [6] representing the weights
        associated with the angles (i.e., the length of sinogram
        pixels in the angular direction).
    :vartype angles_diff: :class:`dict{numpy.dtype: numpy.ndarray}`
    """

    # ensure physical quantities are suitable
    detector_width = float(detector_width)
    source_detector_dist = float(source_detector_dist)
    source_origin_dist = float(source_origin_dist)
    midpointshift = np.array(midpoint_shift)

    image_pixels = max(img_shape[0], img_shape[1])

    # Choosing the number of detectors as the the diagonal
    # through the the image (in image_pixel scale) if none is given
    if n_detectors is None:
        nd = int(np.ceil(np.hypot(img_shape[0], img_shape[1])))
    else:
        nd = n_detectors
    assert isinstance(nd, int), "Number of detectors must be integer"

    # compute midpoints of geometries
    midpoint_detectors = (nd-1.0)/2.0
    midpoint_detectors = midpoint_detectors-detector_shift*nd\
        / detector_width

    # ensure that indeed detector on the opposite side of the source
    assert (source_detector_dist > source_origin_dist),\
        ('The origin is not '
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
            + np.linalg.norm(midpointshift) < source_origin_dist),\
        ('The source is not outside the image domain')

    # Determine midpoint (in scaling 1 = 1 pixel width,
    # i.e., index of center)
    midpoint_x = (img_shape[0]-1)*0.5 - (midpointshift[0]*image_pixels
                                         / float(image_width))
    midpoint_y = (img_shape[1]-1)*0.5 - (midpointshift[1]*image_pixels
                                         / float(image_width))

    # adjust distances to pixel units, i.e. 1 unit corresponds
    # to the length of one image pixel
    source_detector_dist *= image_pixels/float(image_width)
    source_origin_dist *= image_pixels/float(image_width)
    detector_width *= image_pixels/float(image_width)

    # unit vector associated to the angle
    # (vector showing from source to detector)
    thetaX = np.cos(angles)
    thetaY = np.sin(angles)

    # Direction vector along the detector line normed to the length of a
    # single detector pixel (i.e. delta_s (in the scale of delta_x=1))
    detector_orientation = {True: -1, False: 1}[reverse_detector]
    XD = thetaY*detector_width/nd*detector_orientation
    YD = -thetaX*detector_width/nd*detector_orientation

    # Direction vector leading to source from origin (with proper length RE)
    Qx = -thetaX*source_origin_dist
    Qy = -thetaY*source_origin_dist

    # Direction vector from origin to the detector
    # (projection of center onto detector)
    Dx0 = thetaX*(source_detector_dist-source_origin_dist)
    Dy0 = thetaY*(source_detector_dist-source_origin_dist)

    sinogram_shape = (nd, len(angles))

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
        ofs[6] = angle_weights
        ofs_dict[dtype] = ofs

        # determine source detector-pixel distance (=sqrt(R+xi**2))
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
                                  img_shape[0], img_shape[1],
                                  sinogram_shape[0], sinogram_shape[1],
                                  image_width/float(max(img_shape))],
                                 dtype=dtype, order='F')

        geo_dict[dtype] = geometry_info

        angle_diff_dict[dtype] = np.array(angle_weights, dtype=dtype)

    struct = {"img_shape": img_shape, "sinogram_shape": sinogram_shape,
              "ofs_dict": ofs_dict, "sdpd_dict": sdpd_dict,
              "image_width": image_width, "geo_dict": geo_dict,
              "angle_diff_dict": angle_diff_dict}
    return struct


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


def read_angles(angles, angle_weights, projectionsetting):
    """
    Interprets angle set and computes (if necessary) the
    angle_weights suitably.
    """
    if np.isscalar(angles):
        na = abs(angles)
        my_reverse = False
        if angles < 0:
            my_reverse = True
        # dependent on the geometry, create angles of full / half circle
        if projectionsetting.is_fan:
            angles = np.linspace(0, 2*np.pi, abs(angles)+1)[:-1]
            if angle_weights is None:
                angles_diff = np.ones(len(angles))*(2*np.pi/len(angles))
        if projectionsetting.is_parallel:
            angles = np.linspace(0, np.pi, abs(angles)+1)[:-1]
            if angle_weights is None:
                angles_diff = np.ones(len(angles))*(np.pi/len(angles))
        if my_reverse:
            angles = np.flip(angles)
            if angle_weights is None:
                angles_diff = np.flip(angles_diff)
    # In case a list of angles is given, also transform them to
    # list of list/array
    elif (isinstance(angles, (list, np.ndarray)) and np.isscalar(angles[0])):
        na = len(angles)
        angles = np.array(angles)
        if projectionsetting.is_fan:
            # sort angles
            angles_index = np.argsort(angles % (2*np.pi))
            angles_sorted = angles[angles_index] % (2*np.pi)
            angles_extended = np.array(np.hstack([-2*np.pi+angles_sorted[-1],
                                                  angles_sorted,
                                                  angles_sorted[0] + 2*np.pi]))
        if projectionsetting.is_parallel:
            # sort angles
            angles_index = np.argsort(angles % (np.pi))
            angles_sorted = angles[angles_index] % (np.pi)
            angles_extended = np.array(np.hstack([-np.pi+angles_sorted[-1],
                                                  angles_sorted,
                                                  angles_sorted[0] + np.pi]))

        # add first angle at end and last angle at beginning
        # to create full circle, and then compute suitable difference
        angles_diff = 0.5*(abs(angles_extended[2:na+2]
                               - angles_extended[0:na]))

        # Correct for multiple occurrence of angles, for example
        # angles in [0,2pi] are considered instead of [0,pi]
        # and mod pi has same value)
        tol = 0.000001
        na = len(angles_sorted)
        i = 0
        while i < na-1:
            count = 1
            sum = angles_diff[i]
            while abs(angles_sorted[i] - angles_sorted[i+count]) < tol:
                sum += angles_diff[i+count]
                count += 1
                if i+count > na-1:
                    break

            val = sum/count
            for j in range(i, i+count):
                angles_diff[j] = val
            i += count

        angles_diff_temp2 = np.zeros(angles_diff.shape)
        angles_diff_temp2[angles_index] = angles_diff

        angles_diff2 = np.zeros(angles_diff.shape)
        angles_diff2[angles_index] = angles_diff
        angles_diff = angles_diff2

    # Go through list of angle informations
    # (the previous cases both lead to this as well)
    elif isinstance(angles[0], tuple) or isinstance(angles, tuple):
        if isinstance(angles, tuple):
            angles = [angles]
        na = len(angles)
        for j in range(na):
            assert(isinstance(angles[j], tuple)),\
                ("When giving angles via tuples for limited angle setting "
                 + " all subsets must be given in tuple form!")
            assert(len(angles[j]) == 3),\
                ("When tuples are given for the limited angle setting,"
                 + " also angular bounds need to be given, i.e., "
                 + "tuple consists of number of angles or angles themselves "
                 + "as first entry and lower bound of  angular range as "
                 + "second, upper bound as third entry!")
        angles_new = []
        angles_diff = []
        for j in range(len(angles)):
            if isinstance(angles[j][0], int):
                # separate angular range (a,b) into na angles
                na = abs(angles[j][0])
                lower_bound = angles[j][1]
                upper_bound = angles[j][2]
                delta = (upper_bound-lower_bound) / (na)*0.5
                angles_current = np.linspace(lower_bound+delta,
                                             upper_bound-delta, na)
                if angles[j][0] < 0:
                    angles_current = np.flip(angles_current)
            # case where a list of angles is given in the tuple
            if isinstance(angles[j][0], (list, np.ndarray)):
                angles_current = np.array(angles[j][0])
                # try to access the angular_range information if given
                lower_bound = angles[j][1]
                upper_bound = angles[j][2]

            # update angles with currently extracted angles
            angles_new += list(angles_current)

            # compute angle_weights in fullangle setting
            if angle_weights is None:
                # reorder angles
                angles_sorted = np.sort(angles_current)
                angles_index = np.argsort(angles_current)

                # add suitable new angles, at front and beg
                angles_extended = np.array(np.hstack
                                           ([2*lower_bound - angles_sorted[0],
                                            angles_sorted,
                                            2*upper_bound - angles_sorted[-1]])
                                           )

                # compute differences to neighboring angles to compute the
                # angle_weights
                angles_diff_temp = 0.5*(abs(angles_extended
                                            [2:len(angles_extended)]
                                            - angles_extended
                                            [0:len(angles_extended)-2]))

                # Correct for multiple occurrence of angles, for example
                # angles in [0,2pi] are considered instead of [0,pi]
                # and mod pi has same value)
                tol = 0.000001
                na = len(angles_sorted)
                i = 0
                while i < na-1:
                    count = 1
                    sum = angles_diff_temp[i]
                    while abs(angles_sorted[i] - angles_sorted[i+count]) < tol:
                        sum += angles_diff_temp[i+count]
                        count += 1
                        if i+count > na-1:
                            break

                    val = sum/count
                    for j in range(i, i+count):
                        angles_diff_temp[j] = val
                    i += count

                angles_diff_temp2 = np.zeros(angles_diff_temp.shape)
                angles_diff_temp2[angles_index] = angles_diff_temp
                # update angle_diff with newly computed angle diffs
                angles_diff += list(angles_diff_temp2)

        # write angles_new and angles_diff as np.arrays (instead of lists)
        angles = np.array(angles_new)
        na = len(angles)

    if angle_weights is None:
        angle_weights = np.array(angles_diff)
    elif np.isscalar(angle_weights):
        angle_weights = np.ones(na) * angle_weights
    elif isinstance(angle_weights, (list, np.ndarray)):
        assert(na == len(angle_weights)),\
            ("The angle_weights given by the user do not have the same "
             + "length as the number of angles considered")
        angle_weights = np.array(angle_weights)
    return angles, angle_weights


class ProjectionSettings():
    """ Creates and stores all relevant information concerning
    the projection geometry. Serves as a parameter for virtually all
    gratopy's functions.

    :param queue: The OpenCL command queue with which the computations
        are associated.
    :type queue: :class:`pyopencl.CommandQueue`

    :param geometry: Determines whether parallel beam (:const:`gratopy.RADON`)
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
        for Radon and fanbeam transform, respectively, where for negative
        integers the same angles according to its modulus but with reversed
        order are generated.
        Alternatively, the angles can be given explicitly as a :class:`list` or
        :class:`numpy.ndarray`. These two options also imply a full angle
        setting (as opposed to limited angle setting).

        A limited angle setting can be specified in two ways.
        First, a list of angular range sections can be passed as input.
        An angular range section is a :class:`tuple` with either
        an integer or a list/array of angles (first element) together with a
        pair specifying the lower and upper bound of the angular range
        interval (second and third element), i.e., of type
        :class:`tuple(int, float, float)`
        or :class:`tuple(list[float], float, float)`.
        If the first element is an integer, the angular interval will be
        uniformly partitioned into the modulus number of angles
        (note that the first
        and last angles are not the lower/upper bounds to ensure
        uniform angle weights) again in increasing or decreasing order,
        depending on the sign.
        Otherwise, a list or array
        specifying the individual angles is expected.
        In particular, multiple angular sections can be specified,
        by passing a list of angular range sections.

        Alternatively, one can use a list of angles
        and set **angle_weigths** (see below) manually
        to suitable values by passing a scalar, a list or an array.
    :type angles: :class:`int`, :class:`list[float]` / :class:`numpy.ndarray`,
        :class:`list[tuple(int/list[float]/numpy.ndarray, float, float)]`

    :param n_detectors: The number :math:`N_s` of (equi-spaced) detector
        pixels considered. When :obj:`None`, :math:`N_s`
        will be chosen as :math:`\\sqrt{N_x^2+N_y^2}`.
    :type n_detectors:  :class:`int`, default :obj:`None`

    :param angle_weights: The weights :math:`(\\Delta_a)_a`
        associated with the angles,
        which influences the weighting of the rays for the backprojection.
        See :ref:`adjointness` for a more detailed description.
        If :obj:`None` (by default), the weights are computed
        automatically based on the **angles** parameter.
        In the full angle setting, this automatism considers a partition
        of the half circle for parallel beam
        and the full circle for fanbeam geometry based on the
        given angles and sets the angle weight to the average of
        the distances from of an angle to its two neighbors
        (in the sense of a circle).
        Similarly, in the limited angle case, each angle section is
        partitioned by the angles associated with this section and
        the weights are chosen taking additionally
        the boundary of the section into
        account.
        In case of a scalar input, this scalar will be used as the
        (constant) angle weight for all angles.
        Further, all angle weights can directly be set by passing
        an input of type :class:`list[float]` or :class:`numpy.ndarray` of
        suitable length.
    :type angle_weights: :obj:`None`, :class:`float`,
        :class:`list[float]` or :class:`numpy.ndarray`,
        default :obj:`None`

    :param detector_width: Physical length of the detector. For standard
        Radon transformation this can usually remain fixed at the default value
        (together with image\\_width).
    :type detector_width: :class:`float`, default 2.0

    :param image_width: Physical size of the image
        indicated by the length of
        the longer side of the rectangular image domain.
        For parallel beam geometry, when :obj:`None`,
        **image_width** is chosen as 2.0.
        For fanbeam geometry, when :obj:`None`, **image_width** is chosen
        such that the projections exactly capture the image domain.
        To illustrate, choosing **image_width** = **detector_width** results
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
        Defaults to the application of no shift, i.e., the center
        of rotation is also the center of the image.
    :type midpoint_shift:  :class:`list`, default [0.0, 0.0]

    :param reverse_detector: When :attr:`True`, the detector direction
        is flipped in case of fanbeam geometry, i.e., the positive and
        negative detector positions are swapped.
        This parameter has no effect for parallel geometry. When
        activated together with swapping the sign of the angles,
        this has the same effect for projection as mirroring the image.
    :type reverse_detector: :class:`bool`, default :attr:`False`


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

    :ivar angles: Angles from which projections are considered.
    :vartype angles: :class:`numpy.ndarray`

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
        width for each angle which are used to weight the projections, see
        parameter angle_weights above. When none was given as input,
        the angle_weights chosen by the automatism will be written to this
        variable.

    :vartype angle_weights: :class:`numpy.ndarray`

    :ivar prg:  OpenCL program containing the gratopy OpenCL kernels.
        For the corresponding code, see :class:`gratopy.create_code`
    :vartype prg:  :class:`gratopy.Program`

    :ivar struct: Data used in the projection operator.
        Contains in particular dictionaries of
        :class:`numpy.ndarray` associated to precision single and double
        with the angular information necessary for computations.
    :vartype struct: :class:`dict` see :func:`radon_struct` and
        :func:`fanbeam_struct` returns
    """

    def __init__(self, queue, geometry, img_shape, angles,
                 n_detectors=None, angle_weights=None, detector_width=2.0,
                 image_width=None, R=None, RE=None, detector_shift=0.0,
                 midpoint_shift=[0., 0.],
                 reverse_detector=False):

        self.geometry = geometry
        self.queue = queue

        # build program containing OpenCL code
        self.adjusted_code = create_code()
        self.prg = Program(queue.context, self.adjusted_code)

        if np.isscalar(img_shape):
            img_shape = (img_shape, img_shape)

        # only planar img_shape is of relevance
        if len(img_shape) > 2:
            img_shape = img_shape[0:2]
        self.img_shape = img_shape

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

        # extract suitable angles information
        angles, angles_diff = read_angles(angles, angle_weights, self)
        self.n_angles = len(angles)
        self.angle_weights = angles_diff
        self.angles = angles

        if n_detectors is None:
            self.n_detectors = int(np.ceil(np.hypot(img_shape[0],
                                                    img_shape[1])))
        else:
            self.n_detectors = n_detectors

        self.sinogram_shape = (self.n_detectors, self.n_angles)

        # add various values as attributes
        self.image_width = image_width
        self.angles = angles
        self.sinogram_shape = (self.n_detectors, self.n_angles)
        self.detector_shift = detector_shift
        self.midpoint_shift = midpoint_shift
        self.detector_width = detector_width
        self.R = R
        self.RE = RE
        self.reverse_detector = reverse_detector

        # Dictionary which will be used to indicate whether the buffers
        # in corresponding dtype are uploaded to the gpu
        self.buf_upload = {}
        # warning that reverse_detector has no impact on parallel beam setting
        if ((self.reverse_detector) and (self.is_parallel)):
            print("WARNING, the reverse_detector argument has no impact"
                  + " on the parallel beam setting. To reverse the angles,"
                  + " the angles parameter can be translated by np.pi")

        if self.is_fan:
            # assert that R and RE are set for fanbeam
            parameters_available = not ((R is None) or (RE is None))
            assert parameters_available, ("For the Fanbeam geometry "
                                          + "you need to set R (the normal "
                                          + "distance from source to detector)"
                                          + " and RE (distance from source to "
                                          + "coordinate origin which is the "
                                          + "rotation center)")

            # create fanbeam_struct
            self.struct = fanbeam_struct(queue=self.queue,
                                         img_shape=self.img_shape,
                                         angles=self.angles,
                                         angle_weights=self.angle_weights,
                                         detector_width=self.detector_width,
                                         source_detector_dist=self.R,
                                         source_origin_dist=self.RE,
                                         n_detectors=self.n_detectors,
                                         detector_shift=self.detector_shift,
                                         image_width=image_width,
                                         midpoint_shift=self.midpoint_shift,
                                         reverse_detector=self.reverse_detector
                                         )
            # extract relevant information from struct and write as attribute
            self.ofs_buf = self.struct["ofs_dict"]
            self.sdpd_buf = self.struct["sdpd_dict"]
            self.image_width = self.struct["image_width"]
            self.geometry_information = self.struct["geo_dict"]
            self.angle_weights_buf = self.struct["angle_diff_dict"]
            self.angle_weights = self.angle_weights_buf[
                                            np.dtype("float")].copy()
            self.delta_x = float(self.image_width)/max(img_shape)
            self.delta_s = float(detector_width)/n_detectors
            self.delta_ratio = self.delta_s/self.delta_x

        if self.is_parallel:
            # if image_width is not given, set by default to 2
            if image_width is None:
                self.image_width = 2.
            else:
                self.image_width = float(self.image_width)

            # create radon_struct
            self.struct = radon_struct(queue=self.queue,
                                       img_shape=self.img_shape,
                                       angles=self.angles,
                                       angle_weights=self.angle_weights,
                                       n_detectors=self.n_detectors,
                                       detector_width=self.detector_width,
                                       image_width=self.image_width,
                                       midpoint_shift=self.midpoint_shift,
                                       detector_shift=self.detector_shift,
                                       )

            # extract relevant information from struct and write as attribute
            self.ofs_buf = self.struct["ofs_dict"]
            self.delta_x = self.image_width/max(self.img_shape)
            self.delta_s = self.detector_width/self.n_detectors
            self.delta_ratio = self.delta_s/self.delta_x
            self.geometry_information = self.struct["geo_dict"]
            self.angle_weights_buf = self.struct["angle_diff_dict"]
            self.angle_weights = self.angle_weights_buf[
                                            np.dtype("float")].copy()

    def ensure_dtype(self, dtype):
        """
        Uploads buffers for ProjectionSetting
        with given dtype to the gpu (so they are ready to be used by the
        projection operators), in case they were not yet uploaded.
        """
        if dtype not in self.buf_upload:
            upload_bufs(self, dtype)
            self.buf_upload[dtype] = 1

    def set_angle_weights(self, angle_weights):
        """
        Allows to set the angle_weights in the projection-setting to
        arbitrary values.

        :param angle_weights: The array with angle_weights to set to.
        :type angle_weights: :class:`numpy.ndarray`

        """

        if self.is_parallel:
            self.struct = radon_struct(self.queue, self.img_shape,
                                       self.angles,
                                       angle_weights=angle_weights,
                                       n_detectors=self.n_detectors,
                                       detector_width=self.detector_width,
                                       image_width=self.image_width,
                                       midpoint_shift=self.midpoint_shift,
                                       detector_shift=self.detector_shift,
                                       )

        if self.is_fan:
            self.struct = fanbeam_struct(self.queue, self.img_shape,
                                         self.angles,
                                         angle_weights=angle_weights,
                                         detector_width=self.detector_width,
                                         source_detector_dist=self.R,
                                         source_origin_dist=self.RE,
                                         n_detectors=self.n_detectors,
                                         detector_shift=self.detector_shift,
                                         image_width=self.image_width,
                                         midpoint_shift=self.midpoint_shift,
                                         reverse_detector=self.reverse_detector
                                         )
        self.ofs_buf = self.struct["ofs_dict"]
        self.angle_weights_buf = self.struct["angle_diff_dict"]
        self.angle_weights = self.angle_weights_buf[
                                        np.dtype("float")].copy()

        # Make sure buffers are uploaded if necessary
        for dtype in [np.dtype("float32"), np.dtype("float64")]:
            if dtype in self.buf_upload:
                ofs = self.ofs_buf[dtype]
                ofs_buf = cl.Buffer(self.queue.context,
                                    cl.mem_flags.READ_ONLY, ofs.nbytes)
                cl.enqueue_copy(self.queue, ofs_buf, ofs.data).wait()

                # upload angle_weights
                angle_weights = self.angle_weights_buf[dtype]
                angle_weights_buf = cl.Buffer(self.queue.context,
                                              cl.mem_flags.READ_ONLY,
                                              angle_weights.nbytes)
                cl.enqueue_copy(self.queue, angle_weights_buf,
                                angle_weights.data).wait()

                self.ofs_buf[dtype] = ofs_buf
                self.angle_weights_buf[dtype] = angle_weights_buf

    def show_geometry(self, angle, figure=None, axes=None, show=True):
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

        :rtype: tuple(:class:`matplotlib.figure.Figure`,
            :class:`matplotlib.axes.Axes`)
        """

        # Create figure if neither figure nor axes is given
        if (figure is None) and (axes is None):
            figure = plt.figure(0)

        # create axes if non are given beforehand in the figure
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

            # switch around for x,y directions (for xy vs indices xy)
            midpoint_shift = [midpoint_shift[0], midpoint_shift[1]]

            # suitable axis-bounds
            maxsize = max(self.RE, np.sqrt((self.R-self.RE)**2
                                           + detector_width**2/4.))

            # Rotation matrix
            angle = -angle
            A = np.array([[np.cos(angle), np.sin(angle)],
                          [-np.sin(angle), np.cos(angle)]])

            # relevant positions for geometry with angle = 0
            sourceposition = [-source_origin_dist, 0]
            upper_detector = [source_detector_dist-source_origin_dist,
                              detector_width*0.5-self.detector_shift]
            lower_detector = [source_detector_dist-source_origin_dist,
                              - detector_width*0.5-self.detector_shift]
            central_detector = [
               source_detector_dist-source_origin_dist, 0]

            # Rotate these positions around
            sourceposition = np.dot(A, sourceposition)
            upper_detector = np.dot(A, upper_detector)
            lower_detector = np.dot(A, lower_detector)
            central_detector = np.dot(A, central_detector)

            # Connect upper and lower detector edge to create detector line
            axes.plot([upper_detector[0], lower_detector[0]],
                      [upper_detector[1], lower_detector[1]], "k")
            # Connect source position with upper and lower detector edge
            axes.plot([sourceposition[0], upper_detector[0]],
                      [sourceposition[1], upper_detector[1]], "g")
            axes.plot([sourceposition[0], lower_detector[0]],
                      [sourceposition[1], lower_detector[1]], "g")

            # connect source with central detector_positoin
            axes.plot([sourceposition[0], central_detector[0]],
                      [sourceposition[1], central_detector[1]], "g")

            # draw outer circle representing all touched by the image
            draw_circle = matplotlib.patches.Circle(
                midpoint_shift,
                image_width/2 * np.sqrt(1 + (min(self.img_shape)
                                        / max(self.img_shape))**2),
                color='r')
            axes.add_artist(draw_circle)

            # draw rectangle representing the image
            color = (1, 1, 0)
            rect = plt.Rectangle(midpoint_shift-0.5
                                 * np.array([image_width*self.img_shape[0]
                                             / np.max(self.img_shape),
                                             image_width*self.img_shape[1]
                                             / np.max(self.img_shape)]),
                                 image_width
                                 * self.img_shape[0]
                                 / np.max(self.img_shape),
                                 image_width
                                 * self.img_shape[1]
                                 / np.max(self.img_shape),
                                 facecolor=color, edgecolor=color)
            axes.add_artist(rect)

            # draw inner circle representing the object to be observed
            draw_circle = matplotlib.patches.Circle(midpoint_shift,
                                                    image_width/2, color='b')
            axes.add_artist(draw_circle)

            # draw small circle representing the center of rotation
            draw_circle = matplotlib.patches.Circle((0, 0), image_width/10,
                                                    color='k')
            axes.add_artist(draw_circle)

        if self.is_parallel:
            detector_width = self.detector_width
            image_width = self.image_width
            midpoint_shift = self.midpoint_shift

            # switch around for x,y directions (for xy vs indices xy)
            midpoint_shift = [midpoint_shift[0], midpoint_shift[1]]

            # Rotation matrix
            angle = -angle
            A = np.array([[np.cos(angle), np.sin(angle)],
                          [-np.sin(angle), np.cos(angle)]])

            # relevant positions for geometry with angle = 0
            center_source = [-image_width, -self.detector_shift]
            center_detector = [image_width, -self.detector_shift]
            upper_source = [-image_width,
                            -self.detector_shift+0.5*detector_width]
            lower_source = [-image_width,
                            -self.detector_shift-0.5*detector_width]
            upper_detector = [image_width,
                              -self.detector_shift+0.5*detector_width]
            lower_detector = [image_width,
                              -self.detector_shift-0.5*detector_width]

            # Rotate these positions around
            center_source = np.dot(A, center_source)
            center_detector = np.dot(A, center_detector)
            upper_source = np.dot(A, upper_source)
            lower_source = np.dot(A, lower_source)
            upper_detector = np.dot(A, upper_detector)
            lower_detector = np.dot(A, lower_detector)

            # Connect center of source to center of detector
            axes.plot([center_source[0], center_detector[0]],
                      [center_source[1], center_detector[1]], "g")

            # connect lower/upper edges of detector and source
            axes.plot([lower_source[0], lower_detector[0]],
                      [lower_source[1], lower_detector[1]], "g")
            axes.plot([upper_source[0], upper_detector[0]],
                      [upper_source[1], upper_detector[1]], "g")

            # connect lower with upper detector edge creating the detector line
            axes.plot([lower_detector[0], upper_detector[0]],
                      [lower_detector[1], upper_detector[1]], "k")

            # draw outer circle representing all the regions
            # touched by the image
            draw_circle = matplotlib.patches.Circle(midpoint_shift,
                                                    image_width/np.sqrt(2),
                                                    color='r')
            axes.add_artist(draw_circle)

            # draw rectangle representing the image-area
            color = (1, 1, 0)
            draw_rectangle = matplotlib.patches.Rectangle(
                                midpoint_shift
                                - 0.5*np.array([image_width*self.img_shape[0]
                                                / np.max(self.img_shape),
                                                image_width
                                                * self.img_shape[1]
                                                / np.max(self.img_shape)]),
                                image_width * self.img_shape[0]
                                / np.max(self.img_shape),
                                image_width * self.img_shape[1]
                                / np.max(self.img_shape),
                                facecolor=color,
                                edgecolor=color)
            axes.add_artist(draw_rectangle)

            # draw inner circle representing the object to be observed
            draw_circle = matplotlib.patches.Circle(midpoint_shift,
                                                    image_width/2, color='b')
            axes.add_artist(draw_circle)

            # draw small circle in the center of rotation
            draw_circle = matplotlib.patches.Circle((0, 0), image_width/10,
                                                    color='k')
            axes.add_artist(draw_circle)

            # set suitable axis-limits
            maxsize = np.sqrt(image_width**2+detector_width**2)

        axes.set_xlim([-maxsize, maxsize])
        axes.set_ylim([-maxsize, maxsize])
        axes.set_xlabel("x-direction")
        axes.set_ylabel("y-direction")

        # show plot if show parameter is set
        if show and (figure is not None):
            figure.show()
        return figure, axes

    def create_sparse_matrix(self, dtype=np.dtype('float32'), order='F'):
        """
        Creates a sparse matrix representation of the associated forward
        projection.

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

        # Suitable kernels dependent on data types
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
        # choose relevant function
        dtype = np.dtype(dtype)
        function = functions[(np.dtype(dtype), order == 'C')]

        # ensure buffers with suitable dtype are uploaded
        self.ensure_dtype(dtype)

        # get corresponding buffers
        ofs_buf = self.ofs_buf[dtype]
        if self.is_fan:
            sdpd_buf = self.sdpd_buf[dtype]
        geometry_information = self.geometry_information[dtype]

        # define application of the projection function
        # (arguments depend on geometry)
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
        # select suitable position dependent on contiguity
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

        # discretization parameters
        Nx = self.img_shape[0]
        Ny = self.img_shape[1]
        Ns = self.sinogram_shape[0]
        Na = self.sinogram_shape[1]

        # create empty img and sinogram
        img = clarray.zeros(self.queue, self.img_shape, dtype=dtype,
                            order=order)
        sino = forwardprojection(img, self)

        # lists to save values into
        rows = []
        cols = []
        vals = []

        # go through all pixels and put a delta peak at the position x,y and
        # consider the resulting sinogram
        for x in range(Nx):
            if x % int(Nx/100.) == 0:
                sys.stdout.write('\rProgress at {:3.0%}'
                                 .format(float(x)/Nx))
            for y in range(Ny):
                # compute projection for delta peaks in x,y and write onto sino
                projection_from_single_pixel(x, y, sino)
                sinonew = sino.get()
                pos = pos_1(x, y)

                # where non-zero values were added
                index = np.where(sinonew > epsilon)
                for i in range(len(index[0])):
                    s = index[0][i]
                    phi = index[1][i]
                    pos2 = pos_2(s, phi)

                    # save values into corresponding lists
                    rows.append(pos2)
                    cols.append(pos)
                    vals.append(sinonew[s, phi])

        print("\rSparse matrix creation complete")

        # create sparse matrix
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
    :type sino: :class:`pyopencl.array.Array`

    :type img: :class:`pyopencl.array.Array`
    :param projectionsetting: The geometry settings for which the rescaling
        is computed.
    :type projectionsetting: :class:`gratopy.ProjectionSettings`
    :param sino_out: The array in which the result of rescaling
        is saved. If :obj:`None` (per default) is given, a new array
        will be created and returned. When giving the same array as
        **sino**, the values in **sino** will be overwritten.
    :type sino_out: :class:`pyopencl.array.Array` default :obj:`None`

    :param divide: Determines whether the sinogram is
        divided (instead of multiplied) by the angular weights. If :obj:`True`,
        a division is performed, otherwise, the weights are multiplied.
    :type divide: :class:`bool`, default :obj:`False`

    :param wait_for: The events to wait for before performing the
        computation in order to avoid, e.g., race conditions, see
        :class:`pyopencl.Event`.
        This program will always wait for img.events
        and sino.events (so you need not add them to wait_for).
    :type wait_for: :class:`list[pyopencl.Event]`, default :attr:`[]`

    :return: The weighted sinogram.
        If **sino_out** is not :obj:`None`, it
        is returned with the values in its data overwritten. In particular,
        giving the same array for sino and sino_out will overwrite this array.
    :rtype: :class:`pyopencl.array.Array`
    """

    # define type of data considered
    dtype = sino.dtype
    my_order = {0: 'F', 1: 'C'}[sino.flags.c_contiguous]

    # create new sino_out when None is given
    if sino_out is None:
        sino_out = clarray.zeros(sino.queue, sino.shape, dtype=dtype,
                                 order=my_order)

    # choose between the suitable kernel to apply, in particular between divide
    # and multiplication
    float32 = np.dtype("float32")
    float64 = np.dtype("float64")
    if divide is False:
        functions = {
            (float32, "C"): projectionsetting.prg.multiply_float_c,
            (float32, "F"): projectionsetting.prg.multiply_float_f,
            (float64, "C"): projectionsetting.prg.multiply_double_c,
            (float64, "F"): projectionsetting.prg.multiply_double_f
            }
    elif divide is True:
        functions = {
            (float32, "C"): projectionsetting.prg.divide_float_c,
            (float32, "F"): projectionsetting.prg.divide_float_f,
            (float64, "C"): projectionsetting.prg.divide_double_c,
            (float64, "F"): projectionsetting.prg.divide_double_f
            }
    function = functions[dtype, my_order]

    # ensure buffers of the right dtype are uploaded onto the gpu
    projectionsetting.ensure_dtype(dtype)

    # execute weighting by calling the kernel
    myevent = function(sino.queue, sino.shape, None, sino.data,
                       projectionsetting.angle_weights_buf[dtype],
                       sino_out.data,
                       wait_for=wait_for+sino_out.events+sino.events)
    sino_out.add_event(myevent)
    return sino_out


def equ_mul_add(rhs, a, x, projectionsetting, wait_for=[]):
    """
    Executes the calculation rhs+=a*y inside a kernel to avoid memory issues.
    """
    # choose correct kernel to use
    dtype = x.dtype
    function = {np.dtype("float32"): projectionsetting.prg.equ_mul_add_float_c,
                np.dtype("float64"): projectionsetting.prg.equ_mul_add_double_c
                }[dtype]

    # execute operation
    myevent = function(x.queue, [x.size], None, rhs.data, a, x.data,
                       wait_for=x.events+rhs.events+wait_for)
    rhs.add_event(myevent)
    return rhs


def mul_add_add(rhs, a, x, y, projectionsetting, wait_for=[]):
    """
    Executes the calculation rhs=a*x+y inside a kernel to avoid memory issues.
    """
    # choose correct kernel to use
    dtype = x.dtype
    function = {np.dtype("float32"): projectionsetting.prg.mul_add_add_float_c,
                np.dtype("float64"): projectionsetting.prg.mul_add_add_double_c
                }[dtype]

    # execute operation
    myevent = function(x.queue, [x.size], None, rhs.data, a, x.data, y.data,
                       wait_for=x.events+y.events+rhs.events+wait_for)
    rhs.add_event(myevent)
    return rhs


def normest(projectionsetting, number_iterations=50, dtype='float32',
            allocator=None):
    """
    Estimate the spectral norm of the projection operator via power
    iteration, i.e., the operator norm with respect to the
    norms discussed in :ref:`section concerning adjointness <adjointness>`.
    Useful for iterative methods that require such an estimate,
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
    img = clarray.to_device(queue, np.require(np.random.randn(
                                              *projectionsetting.img_shape),
                                              dtype, 'F'),
                            allocator=allocator)

    sino = forwardprojection(img, projectionsetting)

    # power_iteration
    for i in range(number_iterations):
        # rescaling iterate
        normsqr = float(clarray.sum(img).get())
        img /= normsqr
        # compute next iterate
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
               equations of the first kind." Amer. J. Math. 73, 615--624
               (1951). https://doi.org/10.2307/2372313
    """
    # Order to consider solution in
    my_order = {0: 'F', 1: 'C'}[sino.flags.c_contiguous]

    # Set relaxation parameter
    norm_estimate = normest(projectionsetting, allocator=sino.allocator)
    w = sino.dtype.type(w/norm_estimate**2)

    # create required variables
    sinonew = sino.copy()
    U = w*backprojection(sinonew, projectionsetting)
    Unew = clarray.zeros(projectionsetting.queue, U.shape, dtype=sino.dtype,
                         order=my_order, allocator=sino.allocator)

    # execute landweber iteration
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
                Bureau of Standards, 49:409--436 (1952).
                https://doi.org/10.6028/jres.049.044
    """

    # Determine suitable dimensions
    dimensions = projectionsetting.img_shape
    dimensions2 = (1, projectionsetting.n_angles)
    if len(sino.shape) > 2:
        dimensions = dimensions+tuple([sino.shape[2]])
        dimensions2 = dimensions2+tuple([1])

    # order of data
    order = {0: 'F', 1: 'C'}[sino.flags.c_contiguous]

    # create zeros startingpoint if no startingpoint was given
    if x0 is None:
        x0 = clarray.zeros(projectionsetting.queue, dimensions,
                           sino.dtype, order)

    assert(x0.flags.c_contiguous == sino.flags.c_contiguous), (
        "The data sino and initial guess x0 must have the same contiguity!")

    x = x0.copy()

    # preliminary initializations
    d = sino-forwardprojection(x, projectionsetting)
    p = backprojection(d, projectionsetting)
    q = clarray.empty_like(d, projectionsetting.queue)
    q_rescaled = q.copy()
    snew = backprojection(d, projectionsetting)
    sold = snew.copy()

    # Execute conjugate gradients
    for k in range(0, number_iterations):
        sys.stdout.write('\rProgress at {:3.0%}'
                         .format(float(k)/number_iterations))

        forwardprojection(p, projectionsetting, sino=q)  # q=Tp
        alpha = x.dtype.type(projectionsetting.delta_x**2  # alpha=norm(sold)^2
                             / (projectionsetting.delta_s)  # / norm(q)^2
                             * (clarray.vdot(sold, sold)
                                / clarray.vdot(weight_sinogram(q,
                                               projectionsetting, q_rescaled),
                                               q)).get())

        equ_mul_add(x, +alpha, p, projectionsetting)  # x += alpha*p
        equ_mul_add(d, -alpha, q, projectionsetting)  # d -= alpha*q
        backprojection(d, projectionsetting, img=snew)  # snew=T*d
        beta = (clarray.vdot(snew, snew)  # beta = norm(snew)^2/norm(sold)^2
                / clarray.vdot(sold, sold)).get()
        (sold, snew) = (snew, sold)  # swap snew and sold
        mul_add_add(p, beta, p, sold, projectionsetting)  # p = beta*p+sold
        residue = np.sqrt(np.sum(clarray.vdot(d, d).get())
                          / np.sum(clarray.vdot(sino, sino).get()))

        # break if relative residue is smaller than given epsilon
        if residue < epsilon:
            print('\rProgress aborted prematurely as desired'
                  + 'precision is reached')
            break

    print("\rCG reconstruction complete")
    return x


def total_variation(sino, projectionsetting, mu,
                    number_iterations=1000, slice_thickness=1.0,
                    stepsize_weighting=10.):
    """
    Performs a primal-dual algorithm [CP2011]_ to solve a total-variation
    regularized reconstruction problem associated with a given
    projection operator and sinogram. This corresponds to the approximate
    solution of
    :math:`\\min_{u} {\\frac\\mu2}\\|\\mathcal{P}u-f\\|_{L^2}^2+\\mathrm{TV}(u)`
    for :math:`\\mathcal{P}` the projection operator, :math:`f` the sinogram
    and :math:`\\mu` a positive regularization parameter (i.e.,
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
        step sizes :math:`\\sigma` (stepsize for dual update) and :math:`\\tau`
        (stepsize for primal update)
        (with :math:`\\sigma\\tau\\|\\mathcal{P}\\|^2\\leq 1`)
        by multiplication and division, respectively,
        with the given value.
    :type stepsize_weighting: :class:`float`, default 10.0


    :return: Reconstruction gained via primal-dual iteration for the
        total-variation regularized reconstruction problem.
    :rtype:  :class:`pyopencl.array.Array`

    .. [CP2011] Chambolle, A., Pock, T. "A First-Order Primal-Dual Algorithm
                for Convex Problems with Applications to Imaging." J Math
                Imaging Vis 40, 120--145 (2011).
                https://doi.org/10.1007/s10851-010-0251-1
    """
    # Establish queue and context
    queue = projectionsetting.queue
    ctx = queue.context

    # type of data considered
    my_dtype = sino.dtype
    my_order = {0: 'F', 1: 'C'}[sino.flags.c_contiguous]

    # set the shape of the image to be reconstructed
    img_shape = projectionsetting.img_shape
    if len(sino.shape) == 2:
        slice_thickness = 0
    else:
        img_shape = img_shape+tuple(sino.shape[2],)
    extended_img_shape = tuple([4])+img_shape

    # Definitions of suitable kernel functions for primal and dual updates

    # update dual variable to data term
    float32 = np.dtype("float32")
    float64 = np.dtype("float")
    update_lambda_ = {
            (float32, 0): projectionsetting.prg.update_lambda_L2_float_f,
            (float32, 1): projectionsetting.prg.update_lambda_L2_float_c,
            (float64, 0): projectionsetting.prg.update_lambda_L2_double_f,
            (float64, 1): projectionsetting.prg.update_lambda_L2_double_c}
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

    # Modified mu for internal uses
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
