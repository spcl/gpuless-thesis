#!/usr/bin/env bash

EXECUTOR_TYPE=tcp \
CUDA_BINARY=/usr/lib/libtorch_cuda.so \
MANAGER_IP=127.0.0.1 \
LD_PRELOAD=/home/paul/ETH/HS22/DPHPC/gpuless/src/cmake-build-debug/libgpuless.so  \
python run.py
