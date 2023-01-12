## Gratopy Docker image

To build the image corresponding to the current version in the local repository,
simply run
```sh
docker build -f docker/Dockerfile -t gratopy:latest .
```
from the repository root.

**Note:** In order to utilize the host GPU, the container needs to be started
with different CLI arguments, depending on the graphics card. 

#### NVIDIA GPUs

For NVIDIA GPUs, make sure to have the `nvidia-container-runtime` package
installed via the package manager of your distribution. See, for example,
[this page](https://linuxhandbook.com/setup-opencl-linux-docker/) for more
instructions.

Given the NVIDIA container runtime is installed, the built image can be
used via
```sh
docker run --rm -it --gpus all gratopy:latest
```
which spawns a Python shell inside of the container from which `gratopy`
can be imported.

#### AMD GPUs

The docker image installs the necessary drivers for AMD GPUs so that
no additional requirements have to be present on the host system. The
container can be started using
```sh
docker run --rm -it --device=/dev/kfd --device=/dev/dri gratopy:latest
```
which spawns a Python shell inside of the container from which `gratopy`
can be imported.


