import os
import pyopencl as cl
import pyopencl.array as clarray
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import gratopy


# Plots are deactivated by default, can be activated
# by setting 'export GRATOPY_TEST_PLOT=true' in the terminal
plot_parameter = os.environ.get("GRATOPY_TEST_PLOT")
if (plot_parameter is None):
    plot_parameter = '0'
if plot_parameter.lower() not in ['0', 'false']:
    PLOT = True
else:
    PLOT = False


# Read files relevant for the tests
def curdir(filename):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)


# Names of relevant test-data
TESTWALNUT = curdir('walnut.png')
TESTWALNUTSINOGRAM = curdir('walnut_sinogram.png')
TESTRNG = curdir("rng.txt")

# Dummy context
ctx = None
queue = None
INTERACTIVE = False


def evaluate_control_numbers(data, dimensions, expected_result,
                             classified, name):
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

    # Create random variables
    mylist.append(rng.integers(0, M, m))  # s
    mylist.append(rng.integers(0, M, m))  # phi
    mylist.append(rng.integers(0, M, m))  # z
    mylist.append(rng.normal(0, 1, m))  # factors
    mylist.append(rng.integers(0, M, m))  # x
    mylist.append(rng.integers(0, M, m))  # y

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

    # Read saved random numbers
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
    Basic projection test. Computes the forward and backprojection of
    the fanbeam transform for two test images to visually confirm the
    correctness of the method. This projection is repeated 10 times to
    estimate the required time per execution.
    """

    print("Projection test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test image
    Nx = 1200
    dtype = np.dtype("float32")
    img = create_phantoms(queue, Nx, dtype)
    original = img.get()

    # define setting for projection
    number_detectors = 600
    angles = 360
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=img.shape, angles=angles,
                                    detector_width=400.0, RE=200.0,
                                    R=752.0, n_detectors=number_detectors)

    # Create zero arrays to save the results in
    sino_gpu = clarray.zeros(queue, (PS.n_detectors, PS.n_angles, 2),
                             dtype=dtype, order='F')

    backprojected_gpu = clarray.zeros(queue, (PS.img_shape+(2,)),
                                      dtype=dtype, order='F')

    # test speed of implementation for forward projection
    iterations = 10
    a = time.perf_counter()
    for i in range(iterations):
        gratopy.forwardprojection(img, PS, sino=sino_gpu)
    sino_gpu.get()

    print('Average time required for forward projection',
          f"{(time.perf_counter()-a)/iterations:.3f}")

    # test speed of implementation for backward projection
    a = time.perf_counter()
    for i in range(iterations):
        gratopy.backprojection(sino_gpu, PS, img=backprojected_gpu)
    backprojected_gpu.get()
    print('Average time required for backprojection',
          f"{(time.perf_counter()-a)/iterations:.3f}")

    # retrieve data back from gpu to cpu
    sino = sino_gpu.get()
    backprojected = backprojected_gpu.get()

    # plot results
    if PLOT:
        plt.figure(1)
        plt.imshow(np.hstack([original[:, :, 0], original[:, :, 1]]),
                   cmap=plt.cm.gray)
        plt.title('original image')
        plt.figure(2)
        plt.imshow(np.hstack([sino[:, :, 0], sino[:, :, 1]]),
                   cmap=plt.cm.gray)
        plt.title('fanbeam transformed image')
        plt.figure(3)
        plt.imshow(np.hstack([backprojected[:, :, 0],
                   backprojected[:, :, 1]]), cmap=plt.cm.gray)
        plt.title('backprojected image')
        plt.show()

    # Computing controlnumbers to quantitatively verify correctness
    evaluate_control_numbers(img, (Nx, Nx, number_detectors, angles, 2),
                             expected_result=2949.37,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (Nx, Nx, number_detectors, angles, 2),
                             expected_result=94031.43,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backprojected,
                             (Nx, Nx, number_detectors, angles, 2),
                             expected_result=1482240.72,
                             classified="img", name="backprojected image")


def test_types_contiguity():
    """
    Types and contiguity test.
    Types and contiguity test.
    Runs forward and backprojections for fanbeam geometry
    for different precision and contiguity settings,
    checking that they all lead to the same results.
    """
    print("Types and contiguity test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test image
    Nx = 600
    phantom = gratopy.phantom(queue, Nx, dtype=np.dtype("float64")).get()

    # define setting for projection
    number_detectors = 300
    angles = 180
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=(Nx, Nx), angles=angles,
                                    detector_width=400.0, RE=200.0,
                                    R=752.0, n_detectors=number_detectors)

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

                # Computing control numbers to quantitatively verify
                # correctness
                evaluate_control_numbers(img,
                                         (Nx, Nx, number_detectors, angles, 1),
                                         expected_result=7.8922043,
                                         classified="img",
                                         name="original image")

                evaluate_control_numbers(sino,
                                         (Nx, Nx, number_detectors, angles, 1),
                                         expected_result=416.107,
                                         classified="sino",
                                         name="sinogram with "+str(dtype)
                                              + str(order1) + str(order2))

                evaluate_control_numbers(backprojected,
                                         (Nx, Nx, number_detectors, angles, 1),
                                         expected_result=7851.41,
                                         classified="img",
                                         name="backprojected image"
                                              + str(dtype)
                                              + str(order1) + str(order2))


def test_weighting():
    """ Mass preservation test. Checks whether the total mass of an image
    is correctly transported into the total mass of a projection.
    Due to the fan geometry, the width of a projected object on the detector is
    wider than the original object was, as the width of the fan grows linearly
    with the distance it travels. Consequently, also the total mass on the
    detector is roughly the multiplication of the total mass in the
    object by the ratio **R** to **RE**. This estimate is verified
    numerically.
    """
    print("Weighting test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # determine which dtype to use
    dtype = np.dtype("float32")

    # execute for different number of detectors, to ensure
    # resolution independence
    for number_detectors in [50, 100, 200, 400, 800, 1600]:

        # create image with constant density 1
        Nx = 400
        img = np.ones([Nx, Nx])
        angles = 720

        # rescaling parameter for geometry
        rescaling = 1/40.*np.sqrt(2)

        # set the geometry of the system (to change the size of the object)
        detector_width = 400.0*rescaling
        R = 1200.0*rescaling
        RE = 200.0*rescaling
        image_width = 40.0*rescaling

        # create projectionsetting
        PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                        img_shape=img.shape, angles=angles,
                                        detector_width=detector_width, R=R,
                                        RE=RE, n_detectors=number_detectors,
                                        image_width=image_width)

        # forward and backprojection
        img_gpu = clarray.to_device(queue, np.require(img, dtype, 'F'))
        sino_gpu = gratopy.forwardprojection(img_gpu, PS)

        # compute mass inside object and on the detector
        mass_in_image = np.sum(img_gpu.get())*PS.delta_x**2
        mass_in_projcetion = np.sum(sino_gpu.get())\
            * (PS.delta_s) / angles

        # Check results
        print("Mass in original image: " + f"{mass_in_image:.3f}"
              + ", mass in projection with " + f"{number_detectors:.3f}"
              + " detectors: " + f"{mass_in_projcetion:.3f}"
              + ". Ratio " + f"{mass_in_projcetion/mass_in_image:.3f}"
              + ", ratio should be roughly " + f"{(R/RE):.3f}")

        assert(abs(mass_in_projcetion/mass_in_image/(R/RE)-1) < 0.1),\
            "Due to the fan effect the object is enlarged on the detector,\
            roughly by the ratio of the distances, but this was not\
            satisfied in this test."


def test_adjointness():
    """
    Adjointness test. Creates random images
    and sinograms to check whether forward and backprojection are indeed
    adjoint to one another (by comparing the corresponding dual pairings).
    This comparison is carried out for 100 experiments to affirm adjointness
    with some certainty.
    """

    print("Adjointness test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    number_detectors = 230
    img = np.zeros([400, 400])
    angles = 360
    midpoint_shift = [0.0, 0.0]
    dtype = np.dtype("float32")
    order = 'F'

    # define projection setting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM, img.shape, angles,
                                    n_detectors=number_detectors,
                                    detector_width=83.0, detector_shift=0.0,
                                    midpoint_shift=midpoint_shift,
                                    R=900.0, RE=300.0, image_width=None)

    # preliminary definitions for counting errors
    Error = []
    count = 0
    eps = 0.00001

    # create empty arrays for further computation
    img2_gpu = clarray.zeros(queue, PS.img_shape, dtype=dtype,
                             order=order)
    sino2_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=dtype,
                              order=order)

    # loop through a number of experiments
    for i in range(100):

        # Create random image and sinogram
        img1_gpu = clarray.to_device(queue,
                                     np.require(np.random.random(PS.img_shape),
                                                dtype, order))
        sino1_gpu = clarray.to_device(queue, np.require(np.random.random
                                      (PS.sinogram_shape),
                                      dtype, order))

        # compute corresponding forward and backprojections
        gratopy.forwardprojection(img1_gpu, PS, sino=sino2_gpu)
        gratopy.backprojection(sino1_gpu, PS, img=img2_gpu)

        # dual pairing in image domain (weighted by delta_x^2)
        pairing_img = cl.array.dot(img1_gpu, img2_gpu)*PS.delta_x**2
        # dual pairing in sinogram domain (weighted by delta_s)
        pairing_sino = cl.array.dot(gratopy.weight_sinogram(sino1_gpu, PS),
                                    sino2_gpu)*(PS.delta_s)

        # check whether an error occurred,
        # i.e., the dual pairings pairing_img and pairing_sino must coincide
        if abs(pairing_img-pairing_sino) / min(abs(pairing_img),
                                               abs(pairing_sino)) > eps:
            print(pairing_img, pairing_sino, pairing_img/pairing_sino)
            count += 1
            Error.append((pairing_img, pairing_sino))

    # Announce how many errors occurred
    print('Adjointness: For ' + str(count) + ' out of 100'
          + 'tests, the adjointness-errors were bigger than '+str(eps))
    assert (len(Error) < 10), 'A large number of experiments for adjointness '\
        + 'turned out negative, number of errors: '+str(count)+' out of 100 '\
        + 'tests adjointness-errors were bigger than '+str(eps)


def test_limited_angles():
    """ Limited angle test. Tests and illustrates how to set the angles in case
    of limited angle situation, in particular showing artifacts resulting
    from the incorrect use for the limited angle setting
    (leading to undesired angle\\_weights). This can be achieved
    through the format of the **angles** parameter
    or by setting the **angle_weights** directly as shown in the test.
    """

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test phantom
    dtype = np.dtype("float32")
    N = 1200
    img_gpu = create_phantoms(queue, N, dtype)

    # relevant quantities
    Ns = int(0.3*N)
    shift = 0
    (R, RE, Detector_width, image_width) = (5.0, 2.0, 6.0, 2.0)

    # angles cover only part of the angular range, given by two sections
    # the first section moves from a1 to b1. Angular range is the range
    # covered by these (finite number of) angles
    n_angles1 = 180
    a1 = np.pi/8.
    b1 = np.pi/8 + np.pi*3/4.
    angles1 = np.linspace(a1, b1, n_angles1)
    delta1 = (b1-a1) / (n_angles1-1) * 0.5
    angular_range1 = (a1-delta1, b1+delta1)

    # Second angle section
    n_angles2 = 90
    a2 = 5./4*np.pi
    b2 = 2*np.pi
    delta2 = (b2-a2) / (n_angles2-1) * 0.5
    # Careful!, Angle values should be in [0,2pi[, so one does not want the
    # angle 2pi
    angles2 = np.linspace(a2, b2, n_angles2)[:-1]
    angular_range2 = (a2-delta2, b2-delta2)

    # Combine the angle sections to the angle set
    angles_incorrect = list(angles1) + list(angles2)
    angles_correct = [(angles1, angular_range1[0], angular_range1[1]),
                      (angles2, angular_range2[0], angular_range2[1])]

    # Alternatively angular=[] or angular_range=[(),()] will choose
    # automatically some angular range, in this case with the same result.

    # create two projecetionsettings, one with the correct "fullangle=False"
    # parameter for limited-angle situation, one incorrectly using
    # "fullangle=True"
    PScorrect = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                           img_gpu.shape, angles_correct, Ns,
                                           image_width=image_width,
                                           R=R, RE=RE,
                                           detector_width=Detector_width,
                                           detector_shift=shift)

    PSincorrect = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                             img_gpu.shape, angles_incorrect,
                                             Ns, image_width=image_width, R=R,
                                             RE=RE,
                                             detector_width=Detector_width,
                                             detector_shift=shift)

    angle_weights = PScorrect.angle_weights

    PScorrect2 = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                            img_gpu.shape, angles_incorrect,
                                            n_detectors=Ns,
                                            image_width=image_width,
                                            R=R, RE=RE,
                                            detector_width=Detector_width,
                                            detector_shift=shift,
                                            angle_weights=angle_weights)

    # show geometry of the problem
    if PLOT:
        PScorrect.show_geometry(np.pi/4, show=False)

    # forward and backprojection for the two settings
    sino_gpu_correct = gratopy.forwardprojection(img_gpu, PScorrect)
    sino_gpu_incorrect = gratopy.forwardprojection(img_gpu, PSincorrect)
    backprojected_gpu_correct = gratopy.backprojection(sino_gpu_correct,
                                                       PScorrect)
    backprojected_gpu_incorrect = gratopy.backprojection(sino_gpu_correct,
                                                         PSincorrect)
    sino_gpu_correct2 = gratopy.forwardprojection(img_gpu, PScorrect2)
    backprojected_gpu_correct2 = gratopy.backprojection(sino_gpu_correct2,
                                                        PScorrect2)
    # transport results from gpu to cpu
    sino_correct = sino_gpu_correct.get()
    sino_incorrect = sino_gpu_incorrect.get()
    sino_correct2 = sino_gpu_correct2.get()

    backprojected_correct = backprojected_gpu_correct.get()
    backprojected_incorrect = backprojected_gpu_incorrect.get()
    backprojected_correct2 = backprojected_gpu_correct2.get()

    img = img_gpu.get()

    # plot results
    if PLOT:
        plt.figure(1)
        plt.imshow(np.hstack([img[:, :, 0], img[:, :, 1]]), cmap=plt.cm.gray)
        plt.figure(2)
        plt.title("sinograms with vs. without full angle")
        plt.imshow(np.vstack([np.hstack([sino_correct[:, :, 0],
                              sino_correct[:, :, 1]]),
                              np.hstack([sino_incorrect[:, :, 0],
                                         sino_incorrect[:, :, 1]])]),
                   cmap=plt.cm.gray)

        plt.figure(3)
        plt.title("backprojection with vs. without full angle")
        plt.imshow(np.vstack([np.hstack([backprojected_correct[:, :, 0],
                                         backprojected_correct[:, :, 1]]),
                              np.hstack([backprojected_incorrect[:, :, 0],
                                         backprojected_incorrect[:, :, 1]])]),
                   cmap=plt.cm.gray)
        plt.show()

    # Computing control numbers to quantitatively verify correctness
    evaluate_control_numbers(img, (N, N, Ns, len(angles_correct), 2),
                             expected_result=2949.3738,
                             classified="img", name="original image")

    evaluate_control_numbers(sino_correct, (N, N, Ns, len(angles_correct), 2),
                             expected_result=-131.00, classified="sino",
                             name="sinogram with correct fullangle setting")

    evaluate_control_numbers(sino_correct2, (N, N, Ns, len(angles_correct), 2),
                             expected_result=-131.00, classified="sino",
                             name="sinogram with second correct fullangle "
                             + "setting")

    evaluate_control_numbers(sino_incorrect,
                             (N, N, Ns, len(angles_correct), 2),
                             expected_result=-131.00, classified="sino",
                             name="sinogram with incorrect fullangle setting")

    evaluate_control_numbers(backprojected_correct,
                             (N, N, Ns, len(angles_correct), 2),
                             expected_result=19495.96, classified="img",
                             name="backprojected image with correct"
                             + " fullangle setting")

    evaluate_control_numbers(backprojected_correct2,
                             (N, N, Ns, len(angles_correct), 2),
                             expected_result=19495.96, classified="img",
                             name="backprojected image with second correct"
                             + " fullangle setting")

    evaluate_control_numbers(backprojected_incorrect,
                             (N, N, Ns, len(angles_correct), 2),
                             expected_result=23043.191, classified="img",
                             name="backprojected image with incorrect"
                             + " fullangle setting")

    # repair by setting angle_weights Suitable
    PSincorrect.set_angle_weights(PScorrect.angle_weights)
    backprojected_incorrect = gratopy.backprojection(sino_gpu_correct,
                                                     PSincorrect).get()

    evaluate_control_numbers(backprojected_incorrect,
                             (N, N, Ns, len(angles_correct), 2),
                             expected_result=19495.96, classified="img",
                             name="backprojected image with correction on "
                             + "incorrect fullangle setting")


def test_midpoint_shift():
    """
    Shifted midpoint test.
    Tests and illustrates how the sinogram changes if the midpoint of an
    images is shifted away from the center of rotation.
    """

    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create phantom for test
    dtype = np.dtype("float32")
    N = 1200
    img_gpu = create_phantoms(queue, N, dtype)

    # relevant quantities
    (angles, R, RE, Detector_width, image_width, shift) = (360, 5.0,
                                                           3.0, 6.0, 2.0, 0.0)
    midpoint_shift = [0, 0.5]
    Ns = int(0.5*N)

    # define projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM, img_gpu.shape,
                                    angles, Ns, image_width=image_width, R=R,
                                    RE=RE, detector_width=Detector_width,
                                    detector_shift=shift,
                                    midpoint_shift=midpoint_shift)

    # plot the geometry from various angles
    if PLOT:
        plt.figure(0)
        for k in range(0, 16):
            PS.show_geometry(k*np.pi/8, axes=plt.subplot(4, 4, k+1))

    # compute forward and backprojection
    sino_gpu = gratopy.forwardprojection(img_gpu, PS)
    backprojected_gpu = gratopy.backprojection(sino_gpu, PS)

    # transport data from gpu to cpu
    img = img_gpu.get()
    sino = sino_gpu.get()
    backprojected = backprojected_gpu.get()

    # plot results
    if PLOT:
        plt.figure(1)
        plt.imshow(np.hstack([img[:, :, 0], img[:, :, 1]]), cmap=plt.cm.gray)
        plt.figure(2)
        plt.title("sinogram with shifted midpoint")
        plt.imshow(np.hstack([sino[:, :, 0], sino[:, :, 1]]),
                   cmap=plt.cm.gray)
        plt.figure(3)
        plt.title("backprojection with shifted midpoint")
        plt.imshow(np.hstack([backprojected[:, :, 0], backprojected[:, :, 1]]),
                   cmap=plt.cm.gray)
        plt.show()

    # Computing controlnumbers to quantitatively verify correctness
    evaluate_control_numbers(img, (N, N, Ns, angles, 2),
                             expected_result=2949.37,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (N, N, Ns, angles, 2),
                             expected_result=297.880,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backprojected, (N, N, Ns, angles, 2),
                             expected_result=12217.92,
                             classified="img", name="backprojected image")


def test_geometric_orientation():
    """
    Geometric orientation test.
    Considers projections with parallel and fanbeam geometry for very simple
    images in different shifted geometries to illustrate how the geometry of
    the projection work and that they indeed behave analogously
    for parallel and fanbeam setting. Note that the axes of the images shown
    by :func:`matplotlib.pyplot.imshow` are always rotated by 90
    degrees compared to the standard (*x*, *y*)-axes.
    """

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test image, first an image which is only active in the upper left
    # corner, and the second active only on the upper side
    Nx = 600
    img1 = np.zeros([Nx, Nx])
    img1[0:int(Nx*0.3)][:, 0:int(Nx*0.3)] = 1
    img2 = img1.copy()
    img2[0:int(Nx*0.3)][:, 0:int(Nx)] = 1

    image_width = 94.39
    # Sinogram parameters
    Ns = 300
    angles = 360

    # Consider different Parameters for detector_shift and image_shift
    Parameters = [(0, 0, 0), (50, 0, 0), (0, 20, 0), (0, -20, 0),
                  (0, 0, 30), (0, 0, -30)]
    # Corresponding control numbers
    Controlnumbers = [(-22.6691, 1760.90, 131.301, 1157.280),
                      (51.959, -61.3725, 85.7716, 1058.5918),
                      (-703.53, 881.399, 542.11, 1050.41),
                      (390.100, 173.108, 110.7186, 1202.997),
                      (935.812, 1775.13, 578.73, 951.169),
                      (51.3383, 1225.368, -322.945, 579.33)]
    # Go through the parameters
    for j in range(len(Parameters)):
        # Set parameters accordingly
        detector_shift = Parameters[j][0]
        midpoint_shift = [Parameters[j][1], Parameters[j][2]]

        # Define projectionsetting
        PS_fan = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                            img_shape=img1.shape,
                                            angles=angles,
                                            detector_width=170.0, RE=200.0,
                                            R=350, n_detectors=Ns,
                                            detector_shift=detector_shift,
                                            image_width=image_width,
                                            midpoint_shift=midpoint_shift)

        PS_par = gratopy.ProjectionSettings(queue, gratopy.RADON,
                                            img_shape=img1.shape,
                                            angles=angles,
                                            image_width=image_width,
                                            detector_width=image_width,
                                            n_detectors=Ns,
                                            detector_shift=detector_shift,
                                            midpoint_shift=midpoint_shift)

        # compute the projection of the first image in the parallel setting
        img_gpu_par1 = clarray.to_device(queue, img1)
        sino_gpu_par1 = gratopy.forwardprojection(img_gpu_par1, PS_par)
        sino_par1 = sino_gpu_par1.get()

        # compute the projection of the first image in the fanbeam setting
        img_gpu_fan1 = clarray.to_device(queue, img1)
        sino_gpu_fan1 = gratopy.forwardprojection(img_gpu_fan1, PS_fan)
        sino_fan1 = sino_gpu_fan1.get()

        # compute the projection of the second image in the parallel setting
        img_gpu_par2 = clarray.to_device(queue, img2)
        sino_gpu_par2 = gratopy.forwardprojection(img_gpu_par2, PS_par)
        sino_par2 = sino_gpu_par2.get()

        # compute the projection of the second image in the fanbeam setting
        img_gpu_fan2 = clarray.to_device(queue, img2)
        sino_gpu_fan2 = gratopy.forwardprojection(img_gpu_fan2, PS_fan)
        sino_fan2 = sino_gpu_fan2.get()

        # plot results
        if PLOT:
            a = 3
            b = 3
            plt.figure(j)
            plt.suptitle("Results for detector_shift " + str(detector_shift)
                         + ", and midpoint_shift "+str(midpoint_shift) + ".")
            plt.subplot(a, b, 1)
            plt.title("original image 1")
            plt.imshow(img1)

            plt.subplot(a, b, 2)
            plt.title("sinogram of parallel transformed image 1")
            plt.imshow(sino_par1)

            plt.subplot(a, b, 3)
            plt.title("sinogram of fanbeam transformed image 1")
            plt.imshow(sino_fan1)

            PS_par.show_geometry(0, axes=plt.subplot(a, b, 8))

            plt.subplot(a, b, 4)
            plt.title("original image 2")
            plt.imshow(img2)

            plt.subplot(a, b, 5)
            plt.title("sinogram of parallel transformed image 2")
            plt.imshow(sino_par2)

            plt.subplot(a, b, 6)
            plt.title("sinogram of fanbeam transformed image 2")
            plt.imshow(sino_fan2)

            PS_fan.show_geometry(0, axes=plt.subplot(a, b, 9))

            # plt.savefig(os.path.join("orientaionfiles",
            #             str(Parameters[j])+".png"))

        # Compute controlnumbers
        evaluate_control_numbers(sino_par1, (Nx, Nx, Ns, angles, 1),
                                 expected_result=Controlnumbers[j][0],
                                 classified="sino",
                                 name="sinogram of first radon example with "
                                 + "parameters " + str(Parameters[j]))
        evaluate_control_numbers(sino_par2, (Nx, Nx, Ns, angles, 1),
                                 expected_result=Controlnumbers[j][1],
                                 classified="sino",
                                 name="sinogram of second radon example with "
                                 + "parameters " + str(Parameters[j]))
        evaluate_control_numbers(sino_fan1, (Nx, Nx, Ns, angles, 1),
                                 expected_result=Controlnumbers[j][2],
                                 classified="sino",
                                 name="sinogram of first fanbeam example with"
                                 + " parameters " + str(Parameters[j]))
        evaluate_control_numbers(sino_fan2, (Nx, Nx, Ns, angles, 1),
                                 expected_result=Controlnumbers[j][3],
                                 classified="sino",
                                 name="sinogram of second fanbeam example with"
                                 + " parameters " + str(Parameters[j]))

    if PLOT:
        plt.show()


def test_range_check_walnut():
    """ The walnut data set from [HHKKNS2015]_ is considered
    for testing the implementation.
    This test observes that with suitable parameters, the data is
    well-explained by the model defined by gratopy's operators. In particular,
    one can observe that there is a slight imperfection in the data set as
    the detector is not perfectly centered. Indeed, the total
    mass of the upper detector-half theoretically
    needs to coincide with the lower detector-half's total mass
    (up to numerical precision), but these values differ significantly.
    Moreover, this test serves to verify the validity of the conjugate
    gradients (CG) method. It is well-known that the CG algorithm
    approximates the minimal-norm least squares solution to the data,
    and in particular,
    the forward projection of this solution corresponds to the projection
    of data onto the range of the operator. As depicted in the
    plots of the residual data shown by this test,
    the walnut projection data admit, after **detector_shift** correction,
    only slight intensity variations as systematic error.

    .. [HHKKNS2015] Keijo Hämäläinen and Lauri Harhanen and Aki Kallonen and
                    Antti Kujanpää and Esa Niemi and Samuli Siltanen.
                    "Tomographic X-ray data of a walnut".
                    https://arxiv.org/abs/1502.04064
    """

    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # basic parameters
    dtype = np.dtype("float32")
    order = "C"
    reverse = True

    # Geometric and discretization information
    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110.0, 300.0)
    angles = -np.linspace(0, 2*np.pi, 121)[:-1]
    img_shape = (600, 600)

    # read sinogram data and write to device
    sino_gpu = mpimg.imread(TESTWALNUTSINOGRAM)
    sino_gpu = clarray.to_device(queue, np.require(sino_gpu, dtype, order))

    # Create two projectionsettings, one with (the correct) detector shift
    # the other without such correction.
    PS_incorrect = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                              img_shape=img_shape,
                                              angles=angles,
                                              detector_width=Detectorwidth,
                                              R=FDD, RE=FOD,
                                              n_detectors=number_detectors,
                                              detector_shift=0,
                                              reverse_detector=reverse)

    PS_correct = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                            img_shape=img_shape, angles=angles,
                                            detector_width=Detectorwidth,
                                            R=FDD, RE=FOD,
                                            n_detectors=number_detectors,
                                            detector_shift=0.27,
                                            reverse_detector=reverse)

    # Choose random starting point
    mynoise = clarray.to_device(queue,
                                0.*np.random.randn(img_shape[0], img_shape[1]))

    # Execute conjugate gradients. resulting in the minimal norm least squares
    # solution,
    UCG_correct = gratopy.conjugate_gradients(sino_gpu, PS_correct,
                                              number_iterations=50,
                                              x0=mynoise)
    UCG_incorrect = gratopy.conjugate_gradients(sino_gpu, PS_incorrect,
                                                number_iterations=50,
                                                x0=mynoise)
    # The residue is orthogonal to the range of the operator and in particular
    # the projection of the solution is the projection of data onto the range
    best_approximation_correct = gratopy.forwardprojection(
        UCG_correct, PS_correct)
    best_approximation_incorrect = gratopy.forwardprojection(UCG_incorrect,
                                                             PS_incorrect)

    # print approximation error
    print("Approximation error for corrected setting ",
          np.sum((sino_gpu.get()-best_approximation_correct.get())**2)**0.5)
    print("Approximation error for incorrect setting ",
          np.sum((sino_gpu.get()-best_approximation_incorrect.get())**2)**0.5)

    # plot results
    if PLOT:
        plt.figure(1)
        plt.suptitle("reconstructions of walnut data via CG")
        plt.subplot(1, 2, 1)
        plt.title("conjugate gradients reconstruction with shift correction")
        plt.imshow(UCG_correct.get(), cmap=plt.cm.gray)
        plt.subplot(1, 2, 2)
        plt.title("conjugate gradients reconstruction "
                  + "without shift correction")
        plt.imshow(UCG_incorrect.get(), cmap=plt.cm.gray)

        plt.figure(2)
        plt.subplot(1, 3, 1)
        plt.title("best approximation")
        plt.imshow(best_approximation_correct.get(), cmap=plt.cm.gray)
        plt.subplot(1, 3, 2)
        plt.title("given data")
        plt.imshow(sino_gpu.get(), cmap=plt.cm.gray)
        plt.subplot(1, 3, 3)
        plt.title("residual")
        plt.imshow(abs(sino_gpu.get()-best_approximation_correct.get()))
        plt.colorbar()
        plt.suptitle("sinogram associated with reconstruction with shift "
                     + "correction, i.e., best possible approximation "
                     + "with given operator")

        plt.figure(3)
        plt.subplot(1, 3, 1)
        plt.title("best approximation")
        plt.imshow(best_approximation_incorrect.get(), cmap=plt.cm.gray)
        plt.subplot(1, 3, 2)
        plt.title("given data")
        plt.imshow(sino_gpu.get(), cmap=plt.cm.gray)
        plt.subplot(1, 3, 3)
        plt.title("residual")
        plt.imshow(abs(sino_gpu.get()-best_approximation_incorrect.get()))
        plt.colorbar()
        plt.suptitle("sinogram associated with reconstruction without shift "
                     " correction, i.e., best possible approximation with "
                     + "given operator")

        plt.show()


def test_landweber():
    """
    Landweber reconstruction test. Performs the Landweber iteration
    to compute a reconstruction from a sinogram contained in
    the walnut data set of [HHKKNS2015]_, testing the implementation.
    """
    print("Walnut Landweber reconstruction test")

    # create phantom for test
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # Set data types
    dtype = np.dtype("float32")
    order = "F"

    # load sinogram of walnut and rescale
    walnut = mpimg.imread(TESTWALNUTSINOGRAM)
    walnut /= np.mean(walnut)

    # Geometric and discretization quantities
    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110.0, 300.0)
    angles = -np.linspace(0, 2*np.pi, 121)[:-1]
    reverse = True
    img_shape = (600, 600)

    # create projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=img_shape, angles=angles,
                                    detector_width=Detectorwidth,
                                    R=FDD, RE=FOD,
                                    n_detectors=number_detectors,
                                    detector_shift=0.27,
                                    reverse_detector=reverse)

    # create phantom and compute its sinogram for additional data
    my_phantom = gratopy.phantom(queue, img_shape)
    my_phantom_sinogram = gratopy.forwardprojection(my_phantom, PS)

    # Stick together real data and phantom sinogram data
    sino = np.zeros(PS.sinogram_shape+(2,))
    sino[:, :, 0] = walnut/np.max(walnut)
    sino[:, :, 1] = my_phantom_sinogram.get()/np.max(my_phantom_sinogram.get())
    walnut_gpu = clarray.to_device(queue, np.require(sino, dtype, order))

    # execute Landweber method
    a = time.perf_counter()
    ULW = gratopy.landweber(walnut_gpu, PS, 100).get()
    print("Time required "+str(time.perf_counter()-a))

    # plot Landweber reconstruction
    if PLOT:
        plt.figure(4)
        plt.imshow(np.hstack([ULW[:, :, 0], ULW[:, :, 1]]), cmap=plt.cm.gray)
        plt.title("Landweber reconstruction")
        plt.show()

    # Computing control numbers to quantitatively verify correctness
    [Nx, Ny] = img_shape
    evaluate_control_numbers(ULW, (Nx, Ny, number_detectors, len(angles), 2),
                             expected_result=0.971266, classified="img",
                             name=" Landweber-reconstruction")


def test_conjugate_gradients():
    """
    Conjugate gradients reconstruction test.
    Performs the conjugate gradients iteration
    to compute a reconstruction from a sinogram contained in
    the walnut data set of [HHKKNS2015]_, testing the implementation.
    """
    print("Walnut conjugated_gradients reconstruction test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # Set data types
    dtype = np.dtype("float32")
    order = "F"

    # load and rescale image
    walnut = mpimg.imread(TESTWALNUTSINOGRAM)
    walnut /= np.mean(walnut)

    # geometric quantities
    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110.0, 300.0)
    angles = -np.linspace(0, 2*np.pi, 121)[:-1]
    reverse = True
    img_shape = (600, 600)

    # create projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=img_shape,
                                    angles=angles,
                                    detector_width=Detectorwidth,
                                    R=FDD, RE=FOD,
                                    n_detectors=number_detectors,
                                    detector_shift=0.27,
                                    reverse_detector=reverse)

    my_phantom = gratopy.phantom(queue, img_shape)
    my_phantom_sinogram = gratopy.forwardprojection(my_phantom, PS)

    sino = np.zeros(PS.sinogram_shape+tuple([2]))
    sino[:, :, 0] = walnut/np.max(walnut)
    sino[:, :, 1] = my_phantom_sinogram.get()/np.max(my_phantom_sinogram.get())
    walnut_gpu2new = clarray.to_device(queue, np.require(sino, dtype, order))

    # perform conjugate gradients algorithm
    a = time.perf_counter()
    UCG = gratopy.conjugate_gradients(walnut_gpu2new, PS, number_iterations=50)
    print("Time required "+str(time.perf_counter()-a))

    UCG = UCG.get()

    # plot results
    if PLOT:
        plt.figure(1)
        plt.imshow(np.hstack([UCG[:, :, 0], UCG[:, :, 1]]), cmap=plt.cm.gray)
        plt.title("conjugate gradients reconstruction")
        plt.show()

    # Compute control numbers to quantitatively verify correctness
    [Nx, Ny] = img_shape
    evaluate_control_numbers(UCG, (Nx, Ny, number_detectors, len(angles), 2),
                             expected_result=0.75674, classified="img",
                             name=" conjugate gradients reconstruction")


def test_total_variation():
    """
    Total variation reconstruction test.
    Performs the toolbox's total-variation-based approach
    to compute a reconstruction from a sinogram contained in
    the walnut data set of [HHKKNS2015]_, testing the implementation.
    """
    print("Walnut total variation reconstruction test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # Set data types
    dtype = np.dtype("float32")
    order = 'F'

    # relevant quantities
    number_detectors = 328
    (Detectorwidth, FOD, FDD, numberofangles) = (114.8, 110., 300., 120)
    angles = -np.linspace(0, 2*np.pi, 121)[:-1]
    reverse = True
    img_shape = (400, 400)

    # create projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=img_shape, angles=angles,
                                    detector_width=Detectorwidth,
                                    R=FDD, RE=FOD,
                                    n_detectors=number_detectors,
                                    detector_shift=0.27,
                                    reverse_detector=reverse)

    # load and rescale sinogram
    walnut = mpimg.imread(TESTWALNUTSINOGRAM)
    walnut /= np.mean(walnut)

    # Create and add noise to sinograms
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, walnut.shape)*0.2
    walnut2 = walnut+noise

    # Write sinograms to gpu
    walnut_gpu = clarray.to_device(queue, np.require(walnut, dtype, order))
    walnut_gpu2 = clarray.to_device(queue, np.require(walnut2, dtype, order))

    # reconstruction parameters
    number_iterations = 2000
    stepsize_weighting = 10
    mu1 = 100000000
    mu2 = 20

    # perform total_variation reconstruction
    a = time.perf_counter()
    UTV = gratopy.total_variation(walnut_gpu, PS, mu=mu1,
                                  number_iterations=number_iterations,
                                  slice_thickness=0,
                                  stepsize_weighting=stepsize_weighting)
    print("Time required for reconstruction "+str(time.perf_counter()-a))

    a = time.perf_counter()
    UTV2 = gratopy.total_variation(walnut_gpu2, PS, mu=mu2,
                                   number_iterations=number_iterations,
                                   slice_thickness=0,
                                   stepsize_weighting=stepsize_weighting)
    print("Time required for reconstruction "+str(time.perf_counter()-a))

    # Compute sinograms of solutions
    sinoreprojected = gratopy.forwardprojection(UTV, PS).get()
    sinoreprojected2 = gratopy.forwardprojection(UTV2, PS).get()

    # Write solutions from gpu to cpu
    UTV = UTV.get()
    UTV2 = UTV2.get()
    walnut = walnut_gpu.get()
    walnut2 = walnut_gpu2.get()

    # plot results
    if PLOT:
        plt.figure(1)
        plt.imshow(UTV, cmap=plt.cm.gray)
        plt.title("total variation reconstruction with true data")

        plt.figure(2)
        plt.imshow(sinoreprojected, cmap=plt.cm.gray)
        plt.title("reprojected sinogram of solution with true data")

        plt.figure(3)
        plt.imshow(np.hstack(
                [sinoreprojected-walnut, walnut]),
                cmap=plt.cm.gray)
        plt.title("comparison residual (left) with true data (right)")

        plt.figure(4)
        plt.imshow(UTV2, cmap=plt.cm.gray)
        plt.title("total variation reconstruction with noisy data")

        plt.figure(5)
        plt.imshow(sinoreprojected2, cmap=plt.cm.gray)
        plt.title("reprojected sinogram of solution with noisy data")

        plt.figure(6)
        plt.title("comparison residual (left) with noisy data (right)")
        plt.imshow(np.hstack(
            [sinoreprojected2-walnut, walnut]),
            cmap=plt.cm.gray)
        plt.show()

    # Computing control numbers to quantitatively verify correctness
    [Nx, Ny] = img_shape
    evaluate_control_numbers(UTV,
                             (Nx, Ny, number_detectors, numberofangles, 1),
                             expected_result=0.251590, classified="img",
                             name="total-variation reconstruction"
                             + "with true data")

    evaluate_control_numbers(UTV2,
                             (Nx, Ny, number_detectors, numberofangles, 1),
                             expected_result=0.038413, classified="img",
                             name="total-variation reconstruction"
                             + "with noisy data")


def test_nonquadratic():
    """
    Non-quadratic image test. Tests and illustrates the projection
    operator for non-quadratic images.
    """

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create phantom and cut one side out
    N = 1200
    img_gpu = create_phantoms(queue, N)
    N1 = img_gpu.shape[0]
    N2 = int(img_gpu.shape[0]*2/3.)
    img_gpu = cl.array.to_device(queue, img_gpu.get()[:, 0:N2, :].copy())

    # geometric and discretization quantities
    (angles, R, RE, Detector_width, shift) = (360, 5.0, 3.0, 5.0, 0.0)
    image_width = None
    midpoint_shift = [0, 0.]
    Ns = int(0.5*N1)

    # create projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM, img_gpu.shape,
                                    angles, Ns,
                                    image_width=image_width, R=R, RE=RE,
                                    detector_width=Detector_width,
                                    detector_shift=shift,
                                    midpoint_shift=midpoint_shift)

    # compute forward and backprojection
    sino_gpu = gratopy.forwardprojection(img_gpu, PS)
    backprojected_gpu = gratopy.backprojection(sino_gpu, PS)

    # write results from gpu to cpu
    img = img_gpu.get()
    sino = sino_gpu.get()
    backprojected = backprojected_gpu.get()

    # show geometry of projectionsetting
    if PLOT:
        PS.show_geometry(1*np.pi/8, show=False)

        # plot results
        plt.figure(1)
        plt.title("original non-square images")
        plt.imshow(np.hstack([img[:, :, 0], img[:, :, 1]]), cmap=plt.cm.gray)

        plt.figure(2)
        plt.title("fanbeam sinogram for non-square image")
        plt.imshow(np.hstack([sino[:, :, 0], sino[:, :, 1]]),
                   cmap=plt.cm.gray)

        plt.figure(3)
        plt.title("backprojection for non-square image")
        plt.imshow(np.hstack([backprojected[:, :, 0], backprojected[:, :, 1]]),
                   cmap=plt.cm.gray)
        plt.show()

    # Computing a controlnumbers to quantitatively verify correctness
    evaluate_control_numbers(img, (N1, N2, Ns, angles, 2),
                             expected_result=999.4965,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (N1, N2, Ns, angles, 2),
                             expected_result=530.45,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backprojected, (N1, N2, Ns, angles, 2),
                             expected_result=16101.3542,
                             classified="img", name="backprojected image")


def test_create_sparse_matrix():
    """
    Tests the :func:`create_sparse_matrix
    <gratopy.ProjectionSettings.create_sparse_matrix>`
    method to create a sparse matrix
    associated with the transform, and tests it by applying forward and
    backprojection via matrix multiplication.
    """

    # create PyOpenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # define data types
    order = "F"
    dtype = np.dtype("float")

    # discretization quantities
    Nx = 150
    number_detectors = 100
    angles = 30

    # define projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=(Nx, Nx), angles=angles,
                                    detector_width=400.0, R=752.0,
                                    RE=200.0, n_detectors=number_detectors)

    # Create corresponding sparse matrix
    sparsematrix = PS.create_sparse_matrix(dtype=dtype, order=order)

    # create test image and flatten to be suitable for matrix multiplication
    img = gratopy.phantom(queue, Nx, dtype)
    img = img.get()
    img = img.reshape(Nx**2, order=order)

    # Compute forward and backprojection via sparse matrix multiplication
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

    # Computing a control numbers to quantitatively verify correctness
    evaluate_control_numbers(img, (Nx, Nx, number_detectors, angles, 1),
                             expected_result=7.1182017,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (Nx, Nx, number_detectors, angles, 1),
                             expected_result=233.96,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backproj, (Nx, Nx, number_detectors, angles, 1),
                             expected_result=2690.00,
                             classified="img", name="backprojected image")


# test
if __name__ == '__main__':
    pass
