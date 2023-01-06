#!/usr/bin/env bash

EXECUTOR_TYPE=tcp \
	LD_LIBRARY_PATH=$CUDA_HOME/lib64
MANAGER_IP=127.0.0.1 \
MANAGER_PORT=8002 \
CUDA_BINARY=./dwt2d \
LD_PRELOAD=$HOME/gpuless/src/build/libgpuless.so \
./dwt2d 192.bmp -d 192x192 -f -5 -l 3
ls
EXECUTOR_TYPE=tcp \
	LD_LIBRARY_PATH=$CUDA_HOME/lib64
MANAGER_IP=127.0.0.1 \
MANAGER_PORT=8002 \
CUDA_BINARY=./dwt2d \
LD_PRELOAD=$HOME/gpuless/src/build/libgpuless.so \
./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3
