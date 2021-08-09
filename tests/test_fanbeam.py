import os
import pyopencl as cl
import pyopencl.array as clarray
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import gratopy


INTERACTIVE = False


# Read files relevant for the tests
def curdir(filename):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)


# Names of relevant test-data
TESTWALNUT = curdir('walnut.png')
TESTWALNUTSINOGRAM = curdir('walnut_sinogram.png')
TESTRNG = curdir("rng.txt")


ctx = None
queue = None


def evaluate_control_numbers(data, dimensions, expected_result,
                             classified, name):
    # Compites a number from given data, compares with expected value,
    # and raises an error when they do not coincide

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


def create_phantoms(queue, N, dtype='double', order="C"):
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

    img = cl.array.to_device(queue, np.require(np.stack([A.get(), B.get()],
                                               axis=-1),
                                               dtype, order))
    return img


def test_projection():
    """  Basic projection test. Computes the forward and backprojection of
    the fanbeam transform for two test images to visually confirm the
    correctness of the method.    """

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
                                    detector_width=400, RE=200,
                                    R=752, n_detectors=number_detectors)

    # Create zero arrays to save the results in
    sino_gpu = clarray.zeros(queue, (PS.n_detectors, PS.n_angles, 2),
                             dtype=dtype, order='F')

    backprojected_gpu = clarray.zeros(queue, (PS.img_shape+tuple([2])),
                                      dtype=dtype, order='F')

    # test speed of implementation for forward projection
    iterations = 10
    a = time.perf_counter()
    for i in range(iterations):
        gratopy.forwardprojection(img, PS, sino=sino_gpu)
    img.get()

    print('Average time required for forward projection',
           f"{(time.perf_counter()-a)/iterations:.3f}")

    # test speed of implementation for backward projection
    a = time.perf_counter()
    for i in range(iterations):
        gratopy.backprojection(sino_gpu, PS, img=backprojected_gpu)
    sino_gpu.get()
    print('Average time required for backprojection',
           f"{(time.perf_counter()-a)/iterations:.3f}")

    # retrieve data back from gpu to cpu
    sino = sino_gpu.get()
    backprojected = backprojected_gpu.get()

    # plot results
    plt.figure(1)
    plt.imshow(
        np.hstack([original[:, :, 0], original[:, :, 1]]), cmap=plt.cm.gray)
    plt.title('original image')
    plt.figure(2)
    plt.imshow(np.hstack([sino[:, :, 0], sino[:, :, 1]]),
               cmap=plt.cm.gray)

    plt.title('Fanbeam transformed image')
    plt.figure(3)
    plt.imshow(np.hstack([backprojected[:, :, 0],
               backprojected[:, :, 1]]), cmap=plt.cm.gray)
    plt.title('Backprojected image')
    plt.show()

    # Computing controlnumbers to quantitatively verify correctness
    evaluate_control_numbers(img, (Nx, Nx, number_detectors, angles, 2),
                             expected_result=2949.3728,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (Nx, Nx, number_detectors, angles, 2),
                             expected_result=63928.2563,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backprojected,
                             (Nx, Nx, number_detectors, angles, 2),
                             expected_result=1482240.72690,
                             classified="img", name="backprojected image")


def test_types_contiguity():

    print("Projection test")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test image
    Nx = 1200

    # define setting for projection
    number_detectors = 600
    angles = 360
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=(Nx, Nx), angles=angles,
                                    detector_width=400, RE=200,
                                    R=752, n_detectors=number_detectors)

    for dtype in [np.dtype("float32"), np.dtype("float64")]:
        for order1 in ["F", "C"]:
            for order2 in ["F", "C"]:
                img = create_phantoms(queue, Nx, dtype,order=order1)
                original = img.get()

                # Create zero arrays to save the results in
                sino_gpu = clarray.zeros(queue, (PS.n_detectors, PS.n_angles, 2),
                                         dtype=dtype, order=order2)

                backprojected_gpu = clarray.zeros(queue, (PS.img_shape+tuple([2])),
                                                  dtype=dtype, order=order1)

                # test speed of implementation for forward projection
                iterations = 1
                a = time.perf_counter()
                for i in range(iterations):
                    gratopy.forwardprojection(img, PS, sino=sino_gpu)
                img.get()
                print('Average time required for forward projection with (precision:'
                      + str(dtype)+"), (image contiguity:"+str(order1)
                      + "), (sinogram contiguity:" + str(order2) + ") is ",
                      f"{(time.perf_counter()-a)/iterations:.3f}")

                # test speed of implementation for backward projection
                a = time.perf_counter()
                for i in range(iterations):
                    gratopy.backprojection(sino_gpu, PS, img=backprojected_gpu)
                sino_gpu.get()
                print('Average time required for backprojection with (precision:'
                      + str(dtype)+"), (image contiguity:"+str(order1)
                      + "), (sinogram contiguity:" + str(order2) + ") is ",
                      f"{(time.perf_counter()-a)/iterations:.3f}", "\n")

                # retrieve data back from gpu to cpu
                sino = sino_gpu.get()
                backprojected = backprojected_gpu.get()

                # Computing controlnumbers to quantitatively verify correctness
                evaluate_control_numbers(img, (Nx, Nx, number_detectors, angles, 2),
                                         expected_result=2949.3728,
                                         classified="img", name="original image")

                evaluate_control_numbers(sino, (Nx, Nx, number_detectors, angles, 2),
                                         expected_result=63928.2563,
                                         classified="sino", name="sinogram")

                evaluate_control_numbers(backprojected,
                                         (Nx, Nx, number_detectors, angles, 2),
                                         expected_result=1482240.72690,
                                         classified="img", name="backprojected image")


def test_weighting():
    """ Mass preservation test. Checks whether the total mass of an image
    is correctly transported into the total mass of a projection.
    Due to the fan geometry, the width of a projected object on the detector is
    wider than the original object was, as the width of the fan grows linearly
    with the distance it travels. Consequently, also the total mass on the
    detector is rougly the multiplication of the total mass in the
    object by the ratio of **R** and **RE**. This test indicates that
    the scaling of the transform
    is suitable. (???)
    """
    print("Weighting;")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # determine which dtype to use
    dtype = np.dtype("float32")

    # Execute for different number of detectors, to ensure
    # resolution independence
    for number_detectors in [50, 100, 200, 400, 800, 1600]:

        # consider full image
        Nx = 400
        img = np.ones([Nx, Nx])
        angles = 720

        # Rescaling parameter for geometry
        rescaling = 1/40.*np.sqrt(2)

        # Set the geometry of the system
        detector_width = 400*rescaling
        R = 1200.*rescaling
        RE = 200.*rescaling
        image_width = 40.*rescaling

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
            * (PS.delta_ratio*PS.delta_x) / angles

        # Check results
        print("Mass in original image", mass_in_image, "mass in projection",
              mass_in_projcetion, "Ratio", mass_in_projcetion/mass_in_image,
              "Ratio should be roughly"+str(R/RE))

        assert(abs(mass_in_projcetion/mass_in_image/(R/RE)-1) < 0.1),\
            "Due to the fan effect the object is enlarged on the detector,\
            roughly by the ratio of the distances, but this was not\
            satisfied in this test."


def test_adjointness():
    """
    Adjointness test. Creates random images
    and sinograms to check whether forward and backprojection are indeed
    adjoint to one another (by comparing the corresponding dual pairings).
    This comparison is carried out for multiple experiments.
    """

    print("Adjointness:")

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    number_detectors = 230
    img = np.zeros([400, 400])
    angles = 360
    midpoint_shift = [0, 0]
    dtype = np.dtype("float32")

    # define projection setting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM, img.shape, angles,
                                    n_detectors=number_detectors,
                                    detector_width=83, detector_shift=0.0,
                                    midpoint_shift=midpoint_shift,
                                    R=900, RE=300, image_width=None,
                                    fullangle=True)

    # preliminary definitions
    Error = []
    count = 0
    eps = 0.00001

    # create empty arrays for further computation
    img2_gpu = clarray.zeros(queue, PS.img_shape, dtype=dtype,
                             order='F')
    sino2_gpu = clarray.zeros(queue, PS.sinogram_shape, dtype=dtype,
                              order='F')
    for i in range(100):
        # loop through a number of experiments

        # Create random image and sinogram
        img1_gpu = clarray.to_device(queue,
                                     np.require(np.random.random(PS.img_shape),
                                                dtype, 'F'))
        sino1_gpu = clarray.to_device(queue, np.require(np.random.random
                                      (PS.sinogram_shape),
                                      dtype, 'F'))

        # compute corresponding forward and backprojections
        gratopy.forwardprojection(img1_gpu, PS, sino=sino2_gpu)
        gratopy.backprojection(sino1_gpu, PS, img=img2_gpu)

        # dual pairing in image domain (weighted bei delta_x^2)
        a = cl.array.dot(img1_gpu, img2_gpu)*PS.delta_x**2
        # dual pairing in sinogram domain (weighted by delta_s)
        b = cl.array.dot(gratopy.angle_weighting(sino1_gpu, PS),
                         sino2_gpu)*(PS.delta_s)

        # check whether an error occurred, the dual pairings must coincide
        if abs(a-b)/min(abs(a), abs(b)) > eps:
            print(a, b, a/b)
            count += 1
            Error.append((a, b))

    print('Adjointness: Number of Errors: '+str(count)+' out of 100\
        tests adjointness-errors were bigger than '+str(eps))
    assert (len(Error) < 10), 'A large number of experiments for adjointness\
        turned out negative, number of errors: '+str(count)+' out of 100\
        tests adjointness-errors were bigger than '+str(eps)


def test_fullangle():
    """
    Full-angle test. Tests and illustrates the impact of the **fullangle**
    parameter, in particular showing artifacts resulting from the incorrect
    use of the limited angle setting. Also shows how multiple angle sections
    can be employed and how to set the corresponding angular range.
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
    (R, RE, Detector_width, image_width) = (5, 2, 6, 2)

    # angles cover only part of the angular range
    n_angles1 = 180
    a1 = np.pi/8.
    b1 = np.pi/8 + np.pi*3/4.
    angles1 = np.linspace(a1, b1, n_angles1)
    delta1 = (b1-a1) / (n_angles1-1) * 0.5
    angular_range1 = (a1-delta1, b1+delta1)

    n_angles2 = 90
    a2 = 5./4*np.pi
    b2 = 2*np.pi
    delta2 = (b2-a2) / (n_angles2-1) * 0.5
    # Careful!, Angle values should be in [0,2pi[, so one does not want the
    # angle 2pi
    angles2 = np.linspace(a2, b2, n_angles2)[:-1]
    angular_range2 = (a2-delta2, b2-delta2)

    angles = [angles1, angles2]
    angular_range = [angular_range1, angular_range2]
    # Alternatively angular=[] or angular_range=[(),()] will choose
    # automatically some angular range, in this case with the same result.

    # create two projecetionsettings, one with the correct "fullangle=False"
    # parameter for limited-angle situation, incorrectly using "fullangle=True"
    PScorrect = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                           img_gpu.shape, angles, Ns,
                                           image_width=image_width,
                                           R=R, RE=RE,
                                           detector_width=Detector_width,
                                           detector_shift=shift,
                                           fullangle=False,
                                           angular_range=angular_range)

    PSincorrect = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                             img_gpu.shape, angles, Ns,
                                             image_width=image_width, R=R,
                                             RE=RE,
                                             detector_width=Detector_width,
                                             detector_shift=shift,
                                             fullangle=True)

    # show geometry of the problem
    PScorrect.show_geometry(np.pi/4, show=False)

    # forward and backprojection for the two settings
    sino_gpu_correct = gratopy.forwardprojection(img_gpu, PScorrect)
    sino_gpu_incorrect = gratopy.forwardprojection(img_gpu, PSincorrect)
    backprojected_gpu_correct = gratopy.backprojection(
        sino_gpu_correct, PScorrect)
    backprojected_gpu_incorrect = gratopy.backprojection(sino_gpu_correct,
                                                         PSincorrect)

    sino_correct = sino_gpu_correct.get()
    sino_incorrect = sino_gpu_incorrect.get()
    backprojected_correct = backprojected_gpu_correct.get()
    backprojected_incorrect = backprojected_gpu_incorrect.get()
    img = img_gpu.get()

    # plot results
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
                             expected_result=410.598, classified="sino",
                             name="sinogram with correct fullangle setting")

    evaluate_control_numbers(sino_incorrect, (N, N, Ns, len(angles), 2),
                             expected_result=410.598, classified="sino",
                             name="sinogram with incorrect fullangle setting")

    evaluate_control_numbers(backprojected_correct, (N, N, Ns, len(angles), 2),
                             expected_result=14025.02, classified="img",
                             name="backprojected image with correct"
                             + " fullangle setting")

    evaluate_control_numbers(backprojected_incorrect,
                             (N, N, Ns, len(angles), 2),
                             expected_result=21871., classified="img",
                             name="backprojected image with incorrect"
                             + " fullangle setting")


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
    (angles, R, RE, Detector_width, image_width, shift) = (360, 5, 3, 6, 2, 0)
    midpoint_shift = [0, 0.5]
    Ns = int(0.5*N)

    # define projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM, img_gpu.shape,
                                    angles, Ns, image_width=image_width, R=R,
                                    RE=RE, detector_width=Detector_width,
                                    detector_shift=shift,
                                    midpoint_shift=midpoint_shift,
                                    fullangle=True)

    # plot the geometry from various angles
    plt.figure(0)
    for k in range(0, 16):
        PS.show_geometry(k*np.pi/8, axes=plt.subplot(4, 4, k+1))

    # compute forward and backprojection
    sino_gpu = gratopy.forwardprojection(img_gpu, PS)
    backprojected_gpu = gratopy.backprojection(sino_gpu, PS)

    img = img_gpu.get()
    sino = sino_gpu.get()
    backprojected = backprojected_gpu.get()

    # plot results
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
    evaluate_control_numbers(img, (N, N, Ns, angles, 2),
                             expected_result=2949.37386,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (N, N, Ns, angles, 2),
                             expected_result=-613.781,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backprojected, (N, N, Ns, angles, 2),
                             expected_result=14599.8994,
                             classified="img", name="backprojected image")


def test_angle_orientation():
    """
    Angle orientation test
    Considers projections with Radon and Fanbeam setting to illustrate that
    the projection orientation is indeed correct (and not rotated by an angle)
    """
    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create test image
    Nx = 1200
    img1 = np.zeros([Nx, Nx])
    img1[0:int(Nx*0.3)][:, 0:int(Nx*0.3)] = 1
    img2 = img1.copy()
    img2[0:int(Nx*0.3)][:, 0:int(Nx)] = 1

    # Sinogram parameters
    Ns = 300
    reverse = False
    angles = 360

    # Define setting for projection
    PS_fan = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                        img_shape=img1.shape, angles=angles,
                                        detector_width=400, RE=200, R=752,
                                        n_detectors=Ns,
                                        detector_shift=0,
                                        image_width=None,
                                        reverse_detector=reverse)

    PS_par = gratopy.ProjectionSettings(queue, gratopy.RADON,
                                        img_shape=img1.shape, angles=angles,
                                        n_detectors=Ns,
                                        detector_shift=0)

    img_gpu_par1 = clarray.to_device(queue, img1)
    sino_gpu_par1 = gratopy.forwardprojection(img_gpu_par1, PS_par)
    sino_par1 = sino_gpu_par1.get()

    img_gpu_fan1 = clarray.to_device(queue, img1)
    sino_gpu_fan1 = gratopy.forwardprojection(img_gpu_fan1, PS_fan)
    sino_fan1 = sino_gpu_fan1.get()

    img_gpu_par2 = clarray.to_device(queue, img2)
    sino_gpu_par2 = gratopy.forwardprojection(img_gpu_par2, PS_par)
    sino_par2 = sino_gpu_par2.get()

    img_gpu_fan2 = clarray.to_device(queue, img2)
    sino_gpu_fan2 = gratopy.forwardprojection(img_gpu_fan2, PS_fan)
    sino_fan2 = sino_gpu_fan2.get()

    plt.figure(1)
    plt.title("Original image 1")
    plt.imshow(img1)

    plt.figure(2)
    plt.title("Sinogram of parallel setting for image 1")
    plt.imshow(sino_par1)

    plt.figure(3)
    plt.title("Sinogram of fanbeam setting for image 1")
    plt.imshow(sino_fan1)

    plt.figure(4)
    plt.title("Original image 2")
    plt.imshow(img2)

    plt.figure(5)
    plt.title("Sinogram of parallel setting for image 2")
    plt.imshow(sino_par2)

    plt.figure(6)
    plt.title("Sinogram of fanbeam setting for image 2")
    plt.imshow(sino_fan2)

    plt.show()

    evaluate_control_numbers(sino_par1, (Nx, Nx, Ns, angles, 1),
                             expected_result=11.64580,
                             classified="sino",
                             name="sinogram of first radon example")

    evaluate_control_numbers(sino_par2, (Nx, Nx, Ns, angles, 1),
                             expected_result=17.9159, classified="sino",
                             name="sinogram of second radon example")

    evaluate_control_numbers(sino_fan1, (Nx, Nx, Ns, angles, 1),
                             expected_result=184.857, classified="sino",
                             name="sinogram of second fanbeam example")

    evaluate_control_numbers(sino_fan2, (Nx, Nx, Ns, angles, 1),
                             expected_result=558.4474, classified="sino",
                             name="sinogram of second fanbeam example")


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
    of data onto the range of the operator. As depicted in the plots
    shown by this test,
    these projections do not have a systematic error, aside from an apparent
    change of intensity/sensitivity **(???)** for different projection angles.

    .. [HHKKNS2015] Keijo Hämäläinen and Lauri Harhanen and Aki Kallonen and
                    Antti Kujanpää and Esa Niemi and Samuli Siltanen.
                    "Tomographic X-ray data of a walnut".
                    https://arxiv.org/abs/1502.04064
    """

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    dtype = np.dtype("float32")
    order = "C"
    reverse = True
    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110, 300)
    angles = np.linspace(0, 2*np.pi, 121)[:-1]
    img_shape = (600, 600)

#   import pdb;pdb.set_trace()
    sino_gpu = mpimg.imread(TESTWALNUTSINOGRAM)
#   sino_gpu=scipy.io.loadmat(TESTWALNUTSINOGRAM)["m"]

    sino_gpu = clarray.to_device(queue, np.require(sino_gpu, dtype, order))

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
                                            detector_shift=-0.27,
                                            reverse_detector=reverse)

    mynoise = clarray.to_device(

        queue, 0*np.random.randn(img_shape[0], img_shape[1]))
    UCG_correct = gratopy.conjugate_gradients(sino_gpu, PS_correct,
                                              number_iterations=50,
                                              x0=mynoise)
    UCG_incorrect = gratopy.conjugate_gradients(sino_gpu, PS_incorrect,
                                                number_iterations=50,
                                                x0=mynoise)

    best_approximation_correct = gratopy.forwardprojection(
        UCG_correct, PS_correct)
    best_approximation_incorrect = gratopy.forwardprojection(UCG_incorrect,
                                                             PS_incorrect)

    plt.figure(1)
    plt.suptitle("reconstructions of Walnut data via CG")
    plt.subplot(1, 2, 1)
    plt.title("conjugate gradients reconstruction with shift correction")
    plt.imshow(UCG_correct.get(), cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.title("conjugate gradients reconstruction without shift correction")
    plt.imshow(UCG_incorrect.get(), cmap=plt.cm.gray)

    plt.figure(2)
    plt.subplot(1, 3, 1)

    plt.title("Best approximation")
    plt.imshow(best_approximation_correct.get(), cmap=plt.cm.gray)
    plt.subplot(1, 3, 2)
    plt.title("Given data")
    plt.imshow(sino_gpu.get(), cmap=plt.cm.gray)
    plt.subplot(1, 3, 3)
    plt.title("Residue")
    plt.imshow(abs(sino_gpu.get()-best_approximation_correct.get()))
    plt.colorbar()

    plt.suptitle("Sinogram associated to reconstruction with shift correction,"
                 " i.e., best possible approximation with given operator")

    plt.figure(3)
    plt.subplot(1, 3, 1)

    plt.title("Best approximation")
    plt.imshow(best_approximation_incorrect.get(), cmap=plt.cm.gray)
    plt.subplot(1, 3, 2)
    plt.title("Given data")
    plt.imshow(sino_gpu.get(), cmap=plt.cm.gray)
    plt.subplot(1, 3, 3)
    plt.title("Residue")
    plt.imshow(abs(sino_gpu.get()-best_approximation_incorrect.get()))
    plt.colorbar()
    plt.suptitle("Sinogram associated to reconstruction without shift "
                 " correction, i.e., best possible approximation with given"
                 + "operator")

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

    dtype = np.dtype("float32")
    order = "F"

    # load sinogram of walnut and rescale
    # walnut=scipy.io.loadmat(TESTWALNUTSINOGRAM)["m"]
    walnut = mpimg.imread(TESTWALNUTSINOGRAM)
    walnut /= np.mean(walnut)

    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110, 300)
    angles = np.linspace(0, 2*np.pi, 121)[:-1]
    img_shape = (600, 600)

    # create projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=img_shape, angles=angles,
                                    detector_width=Detectorwidth,
                                    R=FDD, RE=FOD,
                                    n_detectors=number_detectors,
                                    detector_shift=-0.27)

    my_phantom = gratopy.phantom(queue, img_shape)
    my_phantom_sinogram = gratopy.forwardprojection(my_phantom, PS)

    sino = np.zeros(PS.sinogram_shape+tuple([2]))
    sino[:, :, 0] = walnut/np.max(walnut)
    sino[:, :, 1] = my_phantom_sinogram.get()/np.max(my_phantom_sinogram.get())

    walnut_gpu = clarray.to_device(queue, np.require(sino, dtype, order))

    # execute Landweber method
    a = time.perf_counter()
    ULW = gratopy.landweber(walnut_gpu, PS, 100).get()
    print("Time required "+str(time.perf_counter()-a))

    # plot Landweber reconstruction
    plt.figure(4)
    plt.imshow(np.hstack([ULW[:, :, 0], ULW[:, :, 1]]), cmap=plt.cm.gray)
    plt.title("Landweber reconstruction")
    plt.show()

    # Computing controlnumbers to quantitatively verify correctness
    [Nx, Ny] = img_shape
    evaluate_control_numbers(ULW, (Nx, Ny, number_detectors, len(angles), 2),
                             expected_result=0.89971, classified="img",
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

    dtype = np.dtype("float32")
    # load and rescale image
#    sinonew=scipy.io.loadmat(TESTWALNUTSINOGRAM)["m"]
    walnut = mpimg.imread(TESTWALNUTSINOGRAM)

    walnut /= np.mean(walnut)

    # geometric quantities
    (number_detectors, Detectorwidth, FOD, FDD) = (328, 114.8, 110, 300)
    angles = np.linspace(0, 2*np.pi, 121)[:-1]
    img_shape = (600, 600)

    # create projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=img_shape,
                                    angles=angles,
                                    detector_width=Detectorwidth,
                                    R=FDD, RE=FOD,
                                    n_detectors=number_detectors,
                                    detector_shift=-0.27)

    my_phantom = gratopy.phantom(queue, img_shape)
    my_phantom_sinogram = gratopy.forwardprojection(my_phantom, PS)

    sino = np.zeros(PS.sinogram_shape+tuple([2]))
    sino[:, :, 0] = walnut/np.max(walnut)
    sino[:, :, 1] = my_phantom_sinogram.get()/np.max(my_phantom_sinogram.get())

    # perform conjugate gradients algorithm
    walnut_gpu2new = clarray.to_device(queue, np.require(sino, dtype, 'C'))

    a = time.perf_counter()
    UCG = gratopy.conjugate_gradients(walnut_gpu2new, PS, number_iterations=50)
    print("Time required "+str(time.perf_counter()-a))

    UCG = UCG.get()
    # plot results
    plt.figure(1)
    plt.imshow(np.hstack([UCG[:, :, 0], UCG[:, :, 1]]), cmap=plt.cm.gray)
    plt.title("conjugate gradients reconstruction")
    plt.show()

    # Compute control numbers to quantitatively verify correctness
    [Nx, Ny] = img_shape
    evaluate_control_numbers(UCG, (Nx, Ny, number_detectors, len(angles), 2),
                             expected_result=0.77610, classified="img",
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

    # relevant quantities
    dtype = np.dtype("float32")
    number_detectors = 328
    (Detectorwidth, FOD, FDD, numberofangles) = (114.8, 110, 300, 120)
    angles = np.linspace(0, 2*np.pi, 121)[:-1]
    img_shape = (400, 400)

    # create projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=img_shape, angles=angles,
                                    detector_width=Detectorwidth,
                                    R=FDD, RE=FOD,
                                    n_detectors=number_detectors,
                                    detector_shift=-0.27)

    # load and rescale sinogram
    walnut = mpimg.imread(TESTWALNUTSINOGRAM)
    walnut /= np.mean(walnut)
    walnut = np.array(walnut, dtype=dtype)

    # Add noise
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, walnut.shape)*0.2
    walnut2 = walnut+noise

    # perform total_variation reconstruction
    walnut_gpu = clarray.to_device(queue, np.require(walnut, dtype, 'C'))
    walnut_gpu2 = clarray.to_device(queue, np.require(walnut2, dtype, 'C'))

    number_iterations = 2000
    stepsize_weighting = 10
    UTV = gratopy.total_variation(walnut_gpu, PS, mu=100000000,
                                  number_iterations=number_iterations,
                                  slice_thickness=0,
                                  stepsize_weighting=stepsize_weighting)

    UTV2 = gratopy.total_variation(walnut_gpu2, PS, mu=20,
                                   number_iterations=number_iterations,
                                   slice_thickness=0,
                                   stepsize_weighting=stepsize_weighting)

    sinoreprojected = gratopy.forwardprojection(UTV, PS)
    sinoreprojected2 = gratopy.forwardprojection(UTV2, PS)

    UTV = UTV.get()
    UTV2 = UTV2.get()
    # plot results
    plt.figure(1)
    plt.imshow(UTV, cmap=plt.cm.gray)

    plt.title("total variation reconstruction with true data")
    plt.figure(2)
    plt.imshow(sinoreprojected.get(), cmap=plt.cm.gray)
    plt.title("reprojected sinogram of solution with true data")
    plt.figure(3)
    plt.imshow(np.hstack(
            [sinoreprojected.get()-walnut_gpu.get(), walnut_gpu.get()]),
            cmap=plt.cm.gray)
    plt.title("comparison residue(left) with true data(right)")
    plt.figure(4)
    plt.imshow(UTV2, cmap=plt.cm.gray)

    plt.title("total variation reconstruction with noisy data")
    plt.figure(5)
    plt.imshow(sinoreprojected2.get(), cmap=plt.cm.gray)
    plt.title("reprojected sinogram of solution with noisy data")
    plt.figure(6)
    plt.title("comparison residue(left) with noisy data(right)")
    plt.imshow(np.hstack(
        [sinoreprojected2.get()-walnut_gpu2.get(), walnut_gpu2.get()]),
        cmap=plt.cm.gray)
    plt.show()

    # Computing controlnumbers to quantitatively verify correctness
    [Nx, Ny] = img_shape

    evaluate_control_numbers(UTV,
                             (Nx, Ny, number_detectors, numberofangles, 1),
                             expected_result=-1.164750, classified="img",
                             name="total-variation reconstruction"
                             + "with true data")

    evaluate_control_numbers(UTV2,
                             (Nx, Ny, number_detectors, numberofangles, 1),
                             expected_result=-1.190727, classified="img",
                             name="total-variation reconstruction"
                             + "with noisy data")


def test_nonquadratic():
    """
    Nonquadratic image test. Tests and illustrates the projection
    operator for non-quadratic images.
    """

    # create PyopenCL context
    ctx = cl.create_some_context(interactive=INTERACTIVE)
    queue = cl.CommandQueue(ctx)

    # create phantom and cut one side
    N = 1200
    img_gpu = create_phantoms(queue, N)
    N1 = img_gpu.shape[0]
    N2 = int(img_gpu.shape[0]*2/3.)
    img_gpu = cl.array.to_device(queue, img_gpu.get()[:, 0:N2, :].copy())

    # geometric quantities
    (angles, R, RE, Detector_width, shift) = (360, 5, 3, 5, 0)
    image_width = None
    midpoint_shift = [0, 0.]
    Ns = int(0.5*N1)

    # create projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM, img_gpu.shape,
                                    angles, Ns,
                                    image_width=image_width, R=R, RE=RE,
                                    detector_width=Detector_width,
                                    detector_shift=shift,
                                    midpoint_shift=midpoint_shift,
                                    fullangle=True)

    # compute forward and backprojection
    sino_gpu = gratopy.forwardprojection(img_gpu, PS)
    backprojected_gpu = gratopy.backprojection(sino_gpu, PS)

    img = img_gpu.get()
    sino = sino_gpu.get()
    backprojected = backprojected_gpu.get()

    # show geometry of projectionsetting
    PS.show_geometry(1*np.pi/8, show=False)

    # plot results
    plt.figure(1)
    plt.title("original non square images")
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
                             expected_result=1078.80,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backprojected, (N1, N2, Ns, angles, 2),
                             expected_result=16101.3542,
                             classified="img", name="backprojected image")


def test_extract_sparse_matrix():
    """
    Tests the :func:`create_sparse_matrix
    <gratopy.ProjectionSettings.create_sparse_matrix>`
    method to create a sparse matrix
    associated with the transform, and tests it by appling forward and
    backprojection by matrix multiplication.
    """

    order = "F"
    dtype = np.dtype("float")
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    # relevant quantities
    Nx = 150
    number_detectors = 100
    img = np.zeros([Nx, Nx])
    angles = 30

    # define projectionsetting
    PS = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                                    img_shape=img.shape, angles=angles,
                                    detector_width=400, R=752,
                                    RE=200, n_detectors=number_detectors)

    # Create corresponding sparse matrix
    sparsematrix = PS.create_sparse_matrix(dtype=dtype, order=order)

    # Test image
    img = gratopy.phantom(queue, Nx, dtype)
    img = img.get()
    img = img.reshape(Nx**2, order=order)

    # Compute forward and backprojection
    sino = sparsematrix*img
    backproj = sparsematrix.T*sino

    # reshape
    img = img.reshape(Nx, Nx, order=order)
    sino = sino.reshape(number_detectors, angles, order=order)
    backproj = backproj.reshape(Nx, Nx, order=order)

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
                             expected_result=7.1182017,
                             classified="img", name="original image")

    evaluate_control_numbers(sino, (Nx, Nx, number_detectors, angles, 1),
                             expected_result=212.594,
                             classified="sino", name="sinogram")

    evaluate_control_numbers(backproj, (Nx, Nx, number_detectors, angles, 1),
                             expected_result=2765.528,
                             classified="img", name="backprojected image")


# test
if __name__ == '__main__':
    pass
