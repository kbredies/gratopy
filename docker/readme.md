## Gratopy Docker image

To build the image corresponding to the current version in the local repository,
simply run
```sh
docker build -f docker/Dockerfile -t Gratopy:latest .
```
from the repository root.

The built image can then be used via
```sh
docker run --rm -it Gratopy:latest
```
which spawns a Python shell inside of the container.

**Note:** Due to restrictions on `opencl` in virtual environments,
only the CPU can be utilized from within the container.
