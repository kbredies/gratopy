import os
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.array as clarray
import time

import gratopy

# Plots are deactivated by default, can be activated
# by setting 'export GRATOPY_TEST_PLOT=0' in the terminal
plot_parameter = os.environ.get("GRATOPY_TEST_PLOT")
if (plot_parameter is None):
    plot_parameter = '0'
if plot_parameter != '0':
    PLOT = True
else:
    PLOT = False


# Read files relevant for the tests
def curdir(filename):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)


# Names of relevant data
TESTRNG = curdir("rng.txt")

# Dummy Context
ctx = None
queue = None


def evaluate_control_numbers(data, dimensions,
                             expected_result, classified, name):
    # Computes a number from given data, compares with expected value,
    # and raises an error when they do not coincide

    # Extract dimensions
    [Nx, Ny, Ns, Na, Nz] = dimensions

    # Get indices for which to compute control-number
    test_s, test_phi, test_z, factors, test_x, test_y = read_control_numbers(
                                                     Nx, Ny, Ns, Na, Nz)
    m = 1000
    mysum = 0
    # Dependent on classifier 'img' or 'sino' choose which variables to use
    if classified == "img":
        var1 = test_x
        var2 = test_y
        var3 = test_z
    else:
        var1 = test_s
        var2 = test_phi
        var3 = test_z

    # Reshape data to 3-dimensional array
    if Nz == 1:
        data = data.reshape(data.shape[0], data.shape[1], 1)

    # Go through all test_numbers
    for i in range(0, m):
        mysum += factors[i]*data[var1[i], var2[i], var3[i]]

    # Check if control-number coincides with expected value
    precision = abs(expected_result)/(10.**3)
    assert(abs(mysum-expected_result) < precision),\
        "A control sum for the "+name + " did not match the expected value. "\
        + "Expected: "+str(expected_result) + ", received: "+str(mysum) +\
        ". Please observe the visual results to check whether this is " +\
        "a numerical issue or a fundamental error."


def create_control_numbers():
    # This function is not really needed for the user, but was used to create
    # the random values for the control number

    m = 1000
    M = 2000
    rng = np.random.default_rng(1)
    mylist = []

    # Create Radom variables
    mylist.append(rng.integers(0, M, m))  # s
    mylist.append(rng.integers(0, M, m))  # phi
    mylist.append(rng.integers(0, M, m))  # z
    mylist.append(rng.normal(0, 1, m))  # factors
    mylist.append(rng.integers(0, M, m))
    mylist.append(rng.integers(0, M, m))

    # Save random numbers into file
    myfile = open(TESTRNG, "w")
    for j in range(6):
        for i in range(m):
            myfile.write(str(mylist[j][i])+"\n")
    myfile.close()


def read_control_numbers(Nx, Ny, Ns, Na, Nz=1):
    # Read saved random numbers to compute the control-number

    myfile = open(TESTRNG, "r")
    m = 1000
    test_s = []
    test_phi = []
    test_z = []
    factors = []
    test_x = []
    test_y = []

    # Read saved rng
    text = myfile.readlines()
    for i in range(m):
        test_s.append(int(text[i]) % Ns)
        test_phi.append(int(text[i+m]) % Na)
        test_z.append(int(text[i+2*m]) % Nz)
        factors.append(float(text[i+3*m]))
        test_x.append(int(text[i+4*m]) % Nx)
        test_y.append(int(text[i+5*m]) % Ny)
    myfile.close()
    return test_s, test_phi, test_z, factors, test_x, test_y


def create_phantoms(queue, N, dtype='double', order="F"):
    # Create a phantom image which is used in many of the tests that follow

    # use gratopy phantom method to create Shepp-Logan phantom
    A = gratopy.phantom(queue, N, dtype=dtype)
    A *= 255/cl.array.max(A).get()

    # second test image consisting of 2 horizontal bars
    B = cl.array.empty(queue, A.shape, dtype=dtype)
    B[:] = 255-120
    B[int(N/3):int(2*N/3)] = 0
    B[0:int(N/4)] = 0
    B[int(N-N/4):N] = 0

    # stack the two images together
    img = cl.array.to_device(queue, np.require(np.stack([A.get(), B.get()],
                                               axis=-1),
                                               dtype, order))
    return img


def test_projection():
    """
    Basic projection test. Simply computes forward and backprojection
    of the Radon transform for two test images in order to visually confirm
    the correctness of the method.
    """

    print("Projection test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # create test image
    dtype = np.dtype("float32")
    N = 1200
    img_gpu = create_phantoms(queue, N, dtype=dtype)

    # relevant quantities for geometry and discretization
    angles = 360
    detector_width = 4
    image_width = 4
    Ns = int(0.5*N)

    # create projectionsetting with parallel beam setting with 360
    # equi-distant angles, the detector has a width of 4 and the observed
    # object has a diameter of 4 (i.e. is captured by the detector), we
    # consider half the amount of detector pixels as image pixels
    PS = gratopy.ProjectionSettings(queue, gratopy.PARALLEL, img_gpu.shape,
                                    angles,
                                    Ns, image_width=image_width,
                                    detector_width=detector_width,
                                    detector_shift=0,
                                    fullangle=True)

    # compute Radon transform for given test images
    sino_gpu = clarray.zeros(queue, (PS.n_detectors, PS.n_angles, 2),
                             dtype=dtype, order='F')

    # compute backprojection of computed sinogram
    backprojected_gpu = clarray.zeros(queue, (PS.img_shape+(2,)),
                                      dtype=dtype, order='F')

    # test speed of implementation for forward projection
    M = 10
    a = time.perf_counter()
    for i in range(M):
        gratopy.forwardprojection(img_gpu, PS, sino=sino_gpu)
    sino_gpu.get()
    print('Average time required for forward projection',
          (time.perf_counter()-a)/M)

    a = time.perf_counter()
    for i in range(M):
        gratopy.backprojection(sino_gpu, PS, img=backprojected_gpu)
    backprojected_gpu.get()
    print('Average time required for backprojection',
          (time.perf_counter()-a)/M)

    # write data from gpu to the cpu
    img = img_gpu.get()
    sino = sino_gpu.get()
    backprojected = backprojected_gpu.get()

    # plot geometry via show_geometry method to visualize geometry
    if PLOT:
        plt.figure(0)
        PS.show_geometry(0, axes=plt.subplot(2, 2, 1))
        PS.show_geometry(np.pi/8, axes=plt.subplot(2, 2, 2))
        PS.show_geometry(np.pi/4, axes=plt.subplot(2, 2, 3))
        PS.show_geometry(np.pi*3/8., axes=plt.subplot(2, 2, 4))

    # plot results
    if PLOT:
        plt.figure(1)
        plt.imshow(np.hstack([img[:, :, 0], img[:, :, 1]]), cmap=plt.cm.gray)
        plt.figure(2)
        plt.imshow(np.hstack([sino[:, :, 0], sino[:, :, 1]]),
                   cmap=plt.cm.gray)
        plt.figure(3)
        plt.imshow(np.hstack([backprojected[:, :, 0],
                              backprojected[:, :, 1]]), cmap=plt.cm.gray)

        plt.show()

    # Computing controlnumbers to quantitatively verify correctness
    evaluate_control_numbers(img, (N, N, Ns, angles, 2),
                             expected_result=2949.3738,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (N, N, Ns, angles, 2),
                             expected_result=1221.2257,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backprojected, (N, N, Ns, angles, 2),
                             expected_result=7427.7049,
                             classified="img", name="backprojected image")


def test_types_contiguity():
    """
    Types and contiguity test.
    Runs forward and backprojections for parallel beam geometry
    for different precision and contiguity settings,
    checking that they all lead to the same results.
    """
    print("Types and contiguity test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # create test image
    Nx = 600
    phantom = gratopy.phantom(queue, Nx, dtype=np.dtype("float64")).get()

    # define setting for projection
    number_detectors = 300
    angles = 180
    PS = gratopy.ProjectionSettings(queue, gratopy.RADON,
                                    img_shape=(Nx, Nx), angles=angles,
                                    n_detectors=number_detectors)

    # loop through all possible settings for precision and contiguity
    for dtype in [np.dtype("float32"), np.dtype("float64")]:
        for order1 in ["F", "C"]:
            for order2 in ["F", "C"]:
                # Set img to suitable setting
                img = clarray.to_device(queue, np.require(phantom,
                                                          dtype, order1))

                # Create zero arrays to save the results in
                sino_gpu = clarray.zeros(queue,
                                         (PS.n_detectors, PS.n_angles),
                                         dtype=dtype, order=order2)
                backprojected_gpu = clarray.zeros(queue,
                                                  PS.img_shape,
                                                  dtype=dtype, order=order1)

                # test speed of implementation for forward projection
                iterations = 20
                a = time.perf_counter()
                for i in range(iterations):
                    gratopy.forwardprojection(img, PS, sino=sino_gpu)
                img.get()
                print('Average time required for forward projection with '
                      '(precision:'
                      + str(dtype)+"), (image contiguity:"+str(order1)
                      + "), (sinogram contiguity:" + str(order2) + ") is ",
                      f"{(time.perf_counter()-a)/iterations:.3f}")

                # test speed of implementation for backward projection
                a = time.perf_counter()
                for i in range(iterations):
                    gratopy.backprojection(sino_gpu, PS, img=backprojected_gpu)
                sino_gpu.get()
                print('Average time required for backprojection with '
                      + '(precision:'
                      + str(dtype)+"), (image contiguity:"+str(order1)
                      + "), (sinogram contiguity:" + str(order2) + ") is ",
                      f"{(time.perf_counter()-a)/iterations:.3f}", "\n")

                # retrieve data back from gpu to cpu
                sino = sino_gpu.get()
                backprojected = backprojected_gpu.get()

                # Computing controlnumbers to quantitatively verify correctness
                evaluate_control_numbers(img,
                                         (Nx, Nx, number_detectors, angles, 1),
                                         expected_result=7.89220,
                                         classified="img",
                                         name="original image")

                evaluate_control_numbers(sino,
                                         (Nx, Nx, number_detectors, angles, 1),
                                         expected_result=6.9605,
                                         classified="sino",
                                         name="sinogram with "+str(dtype)
                                              + str(order1) + str(order2))

                evaluate_control_numbers(backprojected,
                                         (Nx, Nx, number_detectors, angles, 1),
                                         expected_result=20.2049,
                                         classified="img",
                                         name="backprojected image"
                                              + str(dtype)
                                              + str(order1) + str(order2))


def test_weighting():
    """ Mass preservation test. Check whether the total mass of an image
    (square with side length 4/3 and pixel values, i.e. density, 1)
    is correctly transported into the total mass of a projection, i.e.,
    the scaling is adequate.
    """
    print("Weighting test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # types of data
    dtype = np.dtype("float32")
    order = 'F'

    # geometric quantites
    detector_width = 4
    image_width = 4

    # discretization parameters
    angles = 30
    N = 900
    Ns = 500
    img_shape = (N, N)
    # define projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.PARALLEL, img_shape, angles,
                                    Ns, detector_width=detector_width,
                                    image_width=image_width)

    # consider image as rectangular of side-length (4/3)
    img = np.zeros([N, N])
    img[int(N/3.):int(2*N/3.)][:, int(N/3.):int(2*N/3.)] = 1
    img_gpu = cl.array.to_device(queue, np.require(img, dtype, order))

    # compute corresponding sinogram
    sino_gpu = gratopy.forwardprojection(img_gpu, PS)

    # mass inside the image must correspond to the mass any projection
    mass_image = np.sum(img)*PS.delta_x**2
    mass_sino_rdm = np.sum(sino_gpu.get()[:, np.random.randint(0, angles)])\
        * PS.delta_s

    # Announce how many errors occurred
    print("The mass inside the image is "+str(mass_image)
          + " was carried over in the mass inside an projection is "
          + str(mass_sino_rdm)+" i.e. the relative error is "
          + str(abs(1-mass_image/mass_sino_rdm)))

    assert((abs(1-mass_image/mass_sino_rdm) < 0.001)
           * (abs(1-mass_image/mass_sino_rdm) < 0.001)),\
        "The mass was not carried over correctly into  projections,\
        as the relative difference is "\
        + str(abs(1-mass_image/mass_sino_rdm))


def test_adjointness():
    """ Adjointness test. Creates random images
    and sinograms to check whether forward and backprojection are indeed
    adjoint to one another (by comparing the corresponding dual pairings).
    This comparison is carried out for multiple experiments.
    """
    print("Adjointness test")

    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # types of data_type
    dtype = np.dtype("float32")
    order = 'F'

    # discretization parameters
    Nx = 400
    number_detectors = 230
    angles = 360
    img_shape = (Nx, Nx)

    # define projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.PARALLEL, img_shape, angles,
                                    n_detectors=number_detectors,
                                    fullangle=True)

    # define zero images and sinograms
    sino2_gpu = cl.array.zeros(queue, PS.sinogram_shape, dtype=dtype,
                               order=order)
    img2_gpu = cl.array.zeros(queue, PS.img_shape, dtype=dtype, order=order)

    # preliminary definitions for counting errors
    Error = []
    count = 0
    eps = 0.00001

    # loop through a number of experiments
    for i in range(100):

        # create random image and sinogram
        img1_gpu = cl.array.to_device(queue,
                                      np.require(np.random.random
                                                 (PS.img_shape),
                                                 dtype, order))
        sino1_gpu = cl.array.to_device(queue,
                                       np.require(np.random.random(
                                           PS.sinogram_shape), dtype, order)
                                       )

        # compute corresponding forward and backprojections
        gratopy.forwardprojection(img1_gpu, PS, sino=sino2_gpu)
        gratopy.backprojection(sino1_gpu, PS, img=img2_gpu)

        # dual pairing in image domain (weighted bei delta_x^2)
        pairing_img = cl.array.vdot(img1_gpu, img2_gpu).get()*PS.delta_x**2

        # dual pairing in sinogram domain
        pairing_sino = cl.array.vdot(gratopy.weight_sinogram(sino1_gpu, PS),
                                     sino2_gpu).get() * PS.delta_s

        # check whether an error occurred,
        # i.e., the dual pairings pairing_img and pairing_sino must coincide
        if abs(pairing_img-pairing_sino)/min(abs(pairing_img),
                                             abs(pairing_sino)) > eps:
            count += 1
            Error.append((pairing_img, pairing_sino))

    # Anounce how many errors occurred
    print('Adjointness: Number of Errors: '+str(count)+' out of'
          + " 100 tests adjointness-errors were bigger than" + str(eps))
    assert(len(Error) < 10), 'A large number of experiments for adjointness\
        turned out negative, number of errors: '+str(count)+' out of 100\
        tests adjointness-errors were bigger than '+str(eps)


def test_fullangle():
    """ Full-angle test. Tests and illustrates the impact of the **fullangle**
    parameter, in particular showing artifacts resulting from the incorrect
    use of the limited angle setting.
    """

    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # create test image
    dtype = np.dtype("float32")
    N = 1200
    img_gpu = create_phantoms(queue, N, dtype=dtype)

    # detector parameters
    detector_width = 2
    Ns = int(0.3*N)
    shift = 0

    # angles cover only a part of the angular range, angles is a list of angles
    # while angular_range discribes the intervall covered by it
    angles = np.linspace(0, np.pi*3/4., 180)+np.pi/8
    delta = np.pi*3/4.*0.5/179
    angular_range = (np.pi/8-delta, np.pi*3/4.+np.pi/8+delta)
    # Alternatively, angular_range = [] (or simply not setting the value)
    # yields the same result for these concrete angles

    # create two projecetionsettings, one with the correct "fullangle=False"
    # parameter for limited-angle situation, incorrectly using "fullangle=True"
    PScorrect = gratopy.ProjectionSettings(queue, gratopy.PARALLEL,
                                           img_gpu.shape, angles, Ns,
                                           detector_width=detector_width,
                                           detector_shift=shift,
                                           fullangle=False,
                                           angular_range=angular_range)
    PSincorrect = gratopy.ProjectionSettings(queue, gratopy.PARALLEL,
                                             img_gpu.shape, angles, Ns,
                                             detector_width=detector_width,
                                             detector_shift=shift,
                                             fullangle=True,
                                             )

    # forward and backprojection for the two settings
    sino_gpu_correct = gratopy.forwardprojection(img_gpu, PScorrect)
    sino_gpu_incorrect = gratopy.forwardprojection(img_gpu, PSincorrect)
    backprojected_gpu_correct = gratopy.backprojection(
        sino_gpu_correct, PScorrect)
    backprojected_gpu_incorrect = gratopy.backprojection(sino_gpu_correct,
                                                         PSincorrect)

    # write data from gpu onto cpu
    sino_correct = sino_gpu_correct.get()
    sino_incorrect = sino_gpu_incorrect.get()
    backprojected_correct = backprojected_gpu_correct.get()
    backprojected_incorrect = backprojected_gpu_incorrect.get()
    img = img_gpu.get()

    # plot results
    if PLOT:
        plt.figure(1)
        plt.imshow(np.hstack([img[:, :, 0], img[:, :, 1]]), cmap=plt.cm.gray)
        plt.figure(2)
        plt.title("Sinograms with vs without fullangle")
        plt.imshow(np.vstack([np.hstack([sino_correct[:, :, 0],
                                         sino_correct[:, :, 1]]),
                              np.hstack([sino_incorrect[:, :, 0],
                                         sino_incorrect[:, :, 1]])]),
                   cmap=plt.cm.gray)

        plt.figure(3)
        plt.title("Backprojection with vs without fullangle")
        plt.imshow(np.vstack([np.hstack([backprojected_correct[:, :, 0],
                                         backprojected_correct[:, :, 1]]),
                              np.hstack([backprojected_incorrect[:, :, 0],
                                         backprojected_incorrect[:, :, 1]])]),
                   cmap=plt.cm.gray)

        plt.show()

    # Computing controlnumbers to quantitatively verify correctness
    evaluate_control_numbers(img, (N, N, Ns, len(angles), 2),
                             expected_result=2949.3738,
                             classified="img", name="original image")

    evaluate_control_numbers(sino_correct, (N, N, Ns, len(angles), 2),
                             expected_result=990.3170, classified="sino",
                             name="sinogram with correct fullangle setting")

    evaluate_control_numbers(sino_incorrect, (N, N, Ns, len(angles), 2),
                             expected_result=990.3170, classified="sino",
                             name="sinogram with incorrect fullangle setting")

    evaluate_control_numbers(backprojected_correct, (N, N, Ns, len(angles), 2),
                             expected_result=1357.2650, classified="img",
                             name="backprojected image"
                             + "with correct fullangle setting")

    evaluate_control_numbers(backprojected_incorrect,
                             (N, N, Ns, len(angles), 2),
                             expected_result=2409.5415, classified="img",
                             name="backprojected image with incorrect"
                             + "fullangle setting")


def test_nonquadratic():
    """ Nonquadratic image test. Tests and illustrates the projection
    operator for non-quadratic images. """

    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # create phantom but cut of one side
    dtype = np.dtype("float32")
    N1 = 1200
    img = create_phantoms(queue, N1, dtype=dtype)
    N2 = int(img.shape[0]*2/3.)
    img_gpu = cl.array.to_device(queue, img.get()[:, 0:N2, :].copy())

    # sinogram discretization and projectionsetting
    angles = 360
    Ns = int(0.5*img_gpu.shape[0])
    PS = gratopy.ProjectionSettings(queue, gratopy.PARALLEL, img_gpu.shape,
                                    angles, Ns)

    # show geometry of projectionsetting
    if PLOT:
        PS.show_geometry(1*np.pi/8, show=False)

    # compute forward and backprojection
    sino_gpu = gratopy.forwardprojection(img_gpu, PS)
    backprojected_gpu = gratopy.backprojection(sino_gpu, PS)

    # write data from gpu onto cpu
    img = img_gpu.get()
    sino = sino_gpu.get()
    backprojected = backprojected_gpu.get()

    # plot results
    if PLOT:
        plt.figure(1)
        plt.title("original non square images")
        plt.imshow(np.hstack([img[:, :, 0], img[:, :, 1]]), cmap=plt.cm.gray)
        plt.figure(2)
        plt.title("Radon sinogram for non-square image")
        plt.imshow(np.hstack([sino[:, :, 0], sino[:, :, 1]]),
                   cmap=plt.cm.gray)

        plt.figure(3)
        plt.title("backprojection for non-square image")
        plt.imshow(np.hstack([backprojected[:, :, 0],
                              backprojected[:, :, 1]]), cmap=plt.cm.gray)

        plt.show()

    # Computing a controlnumbers to quantitatively verify correctness
    evaluate_control_numbers(img, (N1, N2, Ns, angles, 2),
                             expected_result=999.4965,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (N1, N2, Ns, angles, 2),
                             expected_result=-782.3489,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backprojected, (N1, N2, Ns, angles, 2),
                             expected_result=3310.3464,
                             classified="img", name="backprojected image")


def test_create_sparse_matrix():
    """
    Tests the :func:`create_sparse_matrix
    <gratopy.ProjectionSettings.create_sparse_matrix>`
    method to create a sparse matrix
    associated with the transform, and tests it by appling forward and
    backprojection by matrix multiplication.
    """
    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # Set types of data
    order = "F"
    dtype = np.dtype("float")

    # discretization quantities
    Nx = 150
    number_detectors = 100
    img = np.zeros([Nx, Nx])
    angles = 30

    # define projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.PARALLEL, img.shape, angles,
                                    n_detectors=number_detectors,
                                    fullangle=True)

    # Create corresponding sparse matrix
    sparsematrix = PS.create_sparse_matrix(dtype=dtype, order=order)

    # create test image and flatten to be suitable for matrix multiplication
    img = gratopy.phantom(queue, Nx, dtype)
    img = img.get()
    img = img.reshape(Nx**2, order=order)

    # Compute forward and backprojection
    sino = sparsematrix*img
    backproj = sparsematrix.T*sino

    # reshape results back to rectangular images
    img = img.reshape(Nx, Nx, order=order)
    sino = sino.reshape(number_detectors, angles, order=order)
    backproj = backproj.reshape(Nx, Nx, order=order)

    # plot results
    if PLOT:
        plt.figure(1)
        plt.title("test image")
        plt.imshow(img, cmap=plt.cm.gray)

        plt.figure(2)
        plt.title("projection via sparse matrix")
        plt.imshow(sino, cmap=plt.cm.gray)

        plt.figure(3)
        plt.title("backprojection via sparse matrix")
        plt.imshow(backproj, cmap=plt.cm.gray)
        plt.show()

    # Computing a controlnumbers to quantitatively verify correctness
    evaluate_control_numbers(img, (Nx, Nx, number_detectors, angles, 1),
                             expected_result=7.1182,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (Nx, Nx, number_detectors, angles, 1),
                             expected_result=-1.06132,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backproj, (Nx, Nx, number_detectors, angles, 1),
                             expected_result=1.03955,
                             classified="img", name="backprojected image")


def test_midpoint_shift():
    """
    Shifted midpoint test.
    Tests and illustrates how the sinogram changes if the midpoint of an
    images is shifted away from the center of rotation.
    """

    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # create phantom for test
    dtype = np.dtype("float32")
    N = 1200
    img_gpu = create_phantoms(queue, N, dtype)

    # geometry and sinogram parameters
    (angles, Detector_width, image_width) = (360, 2, 3)
    midpoint_shift = [0., 0.4]
    Ns = int(0.5*N)

    # define projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.PARALLEL, img_gpu.shape,
                                    angles, Ns, image_width=image_width,
                                    detector_width=Detector_width,
                                    midpoint_shift=midpoint_shift)

    # plot the geometry from various angles
    if PLOT:
        plt.figure(0)
        for k in range(0, 16):
            PS.show_geometry(k*np.pi/16, axes=plt.subplot(4, 4, k+1))

    # compute forward and backprojection
    sino_gpu = gratopy.forwardprojection(img_gpu, PS)
    backprojected_gpu = gratopy.backprojection(sino_gpu, PS)

    # write data from gpu onto cpu
    img = img_gpu.get()
    sino = sino_gpu.get()
    backprojected = backprojected_gpu.get()

    # plot results
    if PLOT:
        plt.figure(1)
        plt.imshow(np.hstack([img[:, :, 0], img[:, :, 1]]), cmap=plt.cm.gray)
        plt.figure(2)
        plt.title("Sinogram with shifted midpoint")
        plt.imshow(np.hstack([sino[:, :, 0], sino[:, :, 1]]),
                   cmap=plt.cm.gray)
        plt.figure(3)
        plt.title("Backprojection with shifted midpoint")
        plt.imshow(np.hstack([backprojected[:, :, 0], backprojected[:, :, 1]]),
                   cmap=plt.cm.gray)
        plt.show()

    # Computing controlnumbers to quantitatively verify correctness
    evaluate_control_numbers(img, (N, N, Ns, angles, 2), expected_result=2949.,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (N, N, Ns, angles, 2),
                             expected_result=1810.5192,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backprojected, (N, N, Ns, angles, 2),
                             expected_result=3570.1789,
                             classified="img", name="backprojected image")


# test
if __name__ == '__main__':
    pass
