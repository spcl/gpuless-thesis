FROM nvidia/cuda:11.4.2-cudnn8-devel-centos8

RUN yum -y install -y wget curl perl util-linux xz bzip2 git patch which perl zlib-devel
RUN yum -y groupinstall "Development Tools"

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
RUN chmod +x miniconda.sh
RUN ./miniconda.sh -b -p /opt/conda
RUN rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN /opt/conda/bin/conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
RUN conda install -c pytorch magma-cuda115
WORKDIR /opt
RUN git clone --recursive https://github.com/pytorch/pytorch
WORKDIR /opt/pytorch
RUN MAX_JOBS=16 USE_DISTRIBUTED=OFF TORCH_CUDA_ARCH_LIST="8.0+PTX;8.6;" CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} python setup.py develop
