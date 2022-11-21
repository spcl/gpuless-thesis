#!/usr/bin/env bash

EXECUTOR_TYPE=tcp \
MANAGER_PORT=8002 \
MANAGER_IP=127.0.0.1 \
CUDA_BINARY=/usr/lib/libtorch_cuda.so \
LD_PRELOAD=/home/paul/ETH/HS22/DPHPC/gpuless/src/cmake-build-debug/libgpuless.so  \
python run.py
