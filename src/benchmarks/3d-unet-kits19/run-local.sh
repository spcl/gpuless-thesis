#!/usr/bin/env bash

EXECUTOR_TYPE=tcp \
CUDA_BINARY=/usr/lib/libtorch_cuda.so \
MANAGER_IP=192.168.1.109 \
LD_PRELOAD=/home/luke/ethz/master/thesis/msc-lutobler-gpuless/src/build/libgpuless.so \
python run.py
# python batched.py
