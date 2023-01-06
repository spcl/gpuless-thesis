#!/usr/bin/env bash

EXECUTOR_TYPE=tcp \
MANAGER_IP=127.0.0.1 \
CUDA_BINARY=./pathfinder \
LD_PRELOAD=$HOME/gpuless/src/build/libgpuless.so \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64 \
./pathfinder 100000 100 20
