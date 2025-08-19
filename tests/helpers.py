from __future__ import annotations

import gratopy
import numpy as np
import pyopencl.array as clarray

from pathlib import Path


TESTRNG = Path(__file__).parent / "rng.txt"


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
            myfile.write(str(mylist[j][i]) + "\n")
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
        test_phi.append(int(text[i + m]) % Na)
        test_z.append(int(text[i + 2 * m]) % Nz)
        factors.append(float(text[i + 3 * m]))
        test_x.append(int(text[i + 4 * m]) % Nx)
        test_y.append(int(text[i + 5 * m]) % Ny)
    myfile.close()
    return test_s, test_phi, test_z, factors, test_x, test_y


def evaluate_control_numbers(
    data, dimensions, expected_result, classified, name, rtol=1e-3
):
    # Computes a number from given data, compares with expected value,
    # and raises an error when they do not coincide

    # Extract dimensions
    [Nx, Ny, Ns, Na, Nz] = dimensions

    # Get indices for which to compute control-number
    test_s, test_phi, test_z, factors, test_x, test_y = read_control_numbers(
        Nx, Ny, Ns, Na, Nz
    )

    m = 1000
    mysum = 0
    expected_result = data.dtype.type(expected_result)

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
        mysum += factors[i] * data[var1[i], var2[i], var3[i]]

    # Check if control-number coincides with expected value
    precision = abs(expected_result) * rtol
    assert abs(mysum - expected_result) < precision, (
        "A control sum for the "
        + name
        + " did not match the expected value. "
        + "Expected: "
        + str(expected_result)
        + ", received: "
        + str(mysum)
        + ". Please observe the visual results to check whether this is "
        + "a numerical issue or a fundamental error."
    )


def create_phantoms(queue, N, dtype: str | np.typing.DTypeLike = "double", order="F"):
    # Create a phantom image which is used in many of the tests that follow

    # use gratopy phantom method to create Shepp-Logan phantom
    A = gratopy.phantom(queue, N, dtype=dtype)
    A *= 255 / clarray.max(A).get()

    # second test image consisting of 2 horizontal bars
    B = clarray.empty(queue, A.shape, dtype=dtype)
    B[:] = 255 - 120
    B[int(N / 3) : int(2 * N / 3)] = 0
    B[0 : int(N / 4)] = 0
    B[int(N - N / 4) : N] = 0

    # stack the two images together
    img = clarray.to_device(
        queue, np.require(np.stack([A.get(), B.get()], axis=-1), dtype, order)
    )
    return img
