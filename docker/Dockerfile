FROM ubuntu:22.04
LABEL maintainer="Benjamin Hackl <benjamin.hackl@uni-graz.at>"

ARG AMD_DRIVER_URL=https://repo.radeon.com/amdgpu-install/5.3/ubuntu/jammy/amdgpu-install_5.3.50300-1_all.deb

RUN apt-get update && apt-get -y upgrade \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    initramfs-tools \
    apt-utils \
    unzip \
    tar \
    curl \
    xz-utils \
    ocl-icd-libopencl1 \
    opencl-headers \
    clinfo \
    intel-opencl-icd \
    pocl-opencl-icd \
    python3 \
    python3-pip \
    python3-pyqt5

# Setup AMD OpenCL.

RUN mkdir -p /tmp/amdgpu-install
WORKDIR /tmp/amdgpu-install
RUN curl -O $AMD_DRIVER_URL \
    && dpkg -i ./amdgpu-install_* \
    && DEBIAN_FRONTEND=noninteractive amdgpu-install --opencl=rocr --no-dkms -y \
    && rm -rf /tmp/amdgpu-install

# Setup NVIDIA OpenCL. This assumes that the nvidia-container-runtime
# is installed and available on the host system.

RUN echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install gratopy

RUN pip install -U pip

WORKDIR /usr/local/gratopy
ADD . .

RUN pip install -r requirements.txt && pip install pytest notebook
RUN pip install .

CMD ["/usr/bin/python3"]
