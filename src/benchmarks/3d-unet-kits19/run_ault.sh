#!/usr/bin/bash

EXECUTOR_TYPE=tcp \
CUDA_VISIBLE_DEVICES=0 \
MANAGER_IP=127.0.0.1 \
MANAGER_PORT=8002 \
CUDA_BINARY=$HOME/libtorch/lib/libtorch_cuda.so \
LD_PRELOAD=$HOME/gpuless/src/build/libgpuless.so \
python run.py
