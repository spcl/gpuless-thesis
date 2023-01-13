#!/usr/bin/bash

EXECUTOR_TYPE=tcp \
MANAGER_IP=127.0.0.1 \
MANAGER_PORT=8002 \
CUDA_BINARY=$HOME/libtorch/lib/libtorch_cuda.so \
LD_PRELOAD=$HOME/gpuless/src/build/libgpuless.so \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/conda/lib \
./image-recognition-batched
