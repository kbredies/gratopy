# -*- coding: utf-8 -*-
# From phantominator: https://github.com/mckib2/phantominator/
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

'''The canonical Shepp-Logan phantom used for CT simulations.'''

import numpy as np
import pyopencl as cl

def ct_shepp_logan(queue, N, modified=True, E=None, ret_E=False,
                   dtype='double', allocator=None):
    '''Generate an OpenCL Shepp-Logan phantom of size (N, N).

    :param queue: The OpenCL command queue.
    :type queue: :class:`pyopencl.CommandQueue`

    :param N: Matrix size, (N, N) or (M, N).
    :type N: :class:`int` or :class:`array_like`

    :param modified: Use original gray-scale values as given in [SL1974]_.
        Most implementations use modified values for better contrast (for
        example, see [2]_ and [3]_).
    :type modified: :class:`bool`

    :param E: :math:`e \\times 6` numeric matrix defining :math:`e` ellipses.  
        The six columns of **E** are:

        - Gray value of the ellipse (in [0, 1])
        - Length of the horizontal semi-axis of the ellipse
        - Length of the vertical semi-axis of the ellipse
        - x-coordinate of the center of the ellipse (in [-1, 1])
        - y-coordinate of the center of the ellipse (in [-1, 1])
        - Angle between the horizontal semi-axis of the ellipse
          and the x-axis of the image (in rad)

    :type E: :class:`array_like` or :obj:`None`

    :param ret_E: Return the matrix **E** used to generate the phantom.
    :type ret_E: :class:`bool`

    :param dtype: The :mod:`pyopencl` data type in which the phantom is created.
    :type dtype: :class:`str` or :class:`numpy.dtype`

    :param allocator: The :mod:`pyopencl` allocator used for memory allocation.
    :type allocator: An implementation of :class:`pyopencl.tools.AllocatorInterface` or :obj:`None`

    :returns: Phantom/parameter pair (**ph** [, **E**]). 
    :var ph: The Shepp-Logan phantom.
    :vartype ph: :class:`pyopencl.array.Array`

    :var E: The ellipse parameters used to generate **ph**.
    :vartype E: :class:`array_like`, optional
    
    This much abused phantom is due to [SL1974]_.  The tabulated values in
    the paper are reproduced in the Wikipedia entry [1]_.  The
    original values do not produce great contrast, so modified values
    are used by default (see Table B.1 in [TS1996]_ or implementations
    [2]_ and [3]_).

    .. [SL1974] Shepp, Lawrence A., and Benjamin F. Logan. "The Fourier
                reconstruction of a head section." IEEE Transactions on
                nuclear science 21.3 (1974): 21-43.
    .. [TS1996] Toft, Peter Aundal, and John Aasted Sørensen. "The Radon
               transform-theory and implementation." (1996).
    .. [1] https://en.wikipedia.org/wiki/Shepp%E2%80%93Logan_phantom
    .. [2] https://sigpy.readthedocs.io/en/latest/_modules/sigpy/sim.html#shepp_logan
    .. [3] http://www.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom
    '''

    # Get size of phantom
    if np.isscalar(N):
        M, N = N, N
        is2D = True
    else:
        if len(N) == 2:
            M, N = N[:]
            is2D = True
        else:
            raise ValueError('Dimension must be scalar or 2D!')

    # Give back either a 2D or 3D phantom
    return ct_shepp_logan_2d(queue, M, N, modified, E, ret_E, dtype, allocator)


def ct_shepp_logan_2d(queue, M, N, modified, E, ret_E, dtype, allocator):
    '''Make a 2D phantom.'''

    # Get the ellipse parameters the user asked for
    if E is None:
        if modified:
            E = ct_modified_shepp_logan_params_2d()
        else:
            E = ct_shepp_logan_params_2d()

    # Extract params
    grey = E[:, 0]
    major = E[:, 1]
    minor = E[:, 2]
    xs = E[:, 3]
    ys = E[:, 4]
    theta = E[:, 5]

    # 2x2 square => FOV = (-1, 1)
    X, Y = np.meshgrid(  # meshgrid needs linspace in opposite order
        np.linspace(-1, 1, N),
        -np.linspace(-1, 1, M))
    ph = np.zeros((M, N))
    ct = np.cos(theta)
    st = np.sin(theta)

    for ii in range(E.shape[0]):
        xc, yc = xs[ii], ys[ii]
        a, b = major[ii], minor[ii]
        ct0, st0 = ct[ii], st[ii]

        # Find indices falling inside the ellipse
        idx = (
            ((X - xc)*ct0 + (Y - yc)*st0)**2/a**2 +
            ((X - xc)*st0 - (Y - yc)*ct0)**2/b**2 <= 1)

        # Sum of ellipses
        ph[idx] += grey[ii]

    ph = ph.astype(dtype)
    ph = cl.array.to_device(queue, ph, allocator)
    if ret_E:
        return(ph, E)
    return ph

def ct_shepp_logan_params_2d():
    '''Return parameters for original Shepp-Logan phantom.

    Returns
    -------
    E : array_like, shape (10, 6)
        Parameters for the 10 ellipses used to construct the phantom.
    '''

    E = np.zeros((10, 6)) # (10, [A, a, b, xc, yc, theta])
    E[:, 0] = [2, -.98, -.02, -.02, .01, .01, .01, .01, .01, .01]
    E[:, 1] = [
        .69, .6624, .11, .16, .21, .046, .046, .046, .023, .023]
    E[:, 2] = [.92, .874, .31, .41, .25, .046, .046, .023, .023, .046]
    E[:, 3] = [0, 0, .22, -.22, 0, 0, 0, -.08, 0, .06]
    E[:, 4] = [0, -.0184, 0, 0, .35, .1, -.1, -.605, -.605, -.605]
    E[:, 5] = np.deg2rad([0, 0, -18, 18, 0, 0, 0, 0, 0, 0])
    return E

def ct_modified_shepp_logan_params_2d():
    '''Return parameters for modified Shepp-Logan phantom.

    Returns
    -------
    E : array_like, shape (10, 6)
        Parameters for the 10 ellipses used to construct the phantom.
    '''
    E = ct_shepp_logan_params_2d()
    E[:, 0] = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    return E

if __name__ == '__main__':
    pass
