## Gratopy Docker image

### Basic Usage

**Note:** In order to utilize the host GPU, the container needs to be started
with different CLI arguments, depending on the graphics card. In case you do
not want to utilize the GPU, the command to start the image is
```sh
docker run --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY gratopy:latest
```
The arguments `-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY`
enable X11 forwarding and can be omitted if you do not require it (e.g., because you
are using the image to start a local Jupyter server which you interact with from your
host machine).

It is possible that, in order to allow the container to use your host machine's X server,
you additionally have to run `xhost +local:docker` and/or make your `~/.Xauthority` file
available within the container by adding `-v $HOME/.Xauthority:/root/.Xauthority` to the
suggested `docker run` calls.

#### NVIDIA GPUs

For NVIDIA GPUs, make sure to have the `nvidia-container-runtime` package
installed via the package manager of your distribution. See, for example,
[this page](https://linuxhandbook.com/setup-opencl-linux-docker/) for more
instructions.

Given the NVIDIA container runtime is installed, the built image can be
used via
```sh
docker run --rm -it --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY gratopy:latest
```
which spawns a Python shell inside of the container from which `gratopy`
can be imported.

#### AMD GPUs

The docker image installs the necessary drivers for AMD GPUs so that
no additional requirements have to be present on the host system. The
container can be started using
```sh
docker run --rm -it --device=/dev/kfd --device=/dev/dri -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY gratopy:latest
```
which spawns a Python shell inside of the container from which `gratopy`
can be imported.

#### Intel GPUs

Same as for AMD GPUs, the image ships with the necessary drivers and
no additional dependencies are required for the host system. The container
can be started using
```sh
docker run --rm -it --device=/dev/dri -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY gratopy:latest
```
which spawns a Python shell inside of the container from which `gratopy`
can be imported.

### Jupyter Notebooks

It is also possible to spin up a local Jupyter server in order to
interact with `gratopy` in a Jupyter notebook from a browser on your
host machine. To do so, run the image with
```
docker run --rm [GPU options] -p 8888:8888 gratopy:latest uv run jupyter notebook --ip=0.0.0.0 --allow-root
```
and navigate to the link to `http://127.0.0.1:8888/?token=...` that is
shown in your terminal.

### Building the Image

To build the image corresponding to the current version in the local repository,
simply run
```sh
docker build -f docker/Dockerfile -t gratopy:latest .
```
from the repository root.

### Running Tests

The docker container ships with `gratopy`s test suite, which can be
run via
```sh
docker run --rm -it [GPU options] gratopy:latest uv run pytest
```




