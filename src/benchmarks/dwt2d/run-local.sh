#!/usr/bin/env bash

EXECUTOR_TYPE=tcp \
	LD_LIBRARY_PATH=$CUDA_HOME/lib64
MANAGER_IP=127.0.0.1 \
MANAGER_PORT=8002 \
CUDA_BINARY=./dwt2d \
LD_PRELOAD=$HOME/gpuless/src/build/libgpuless.so \
./dwt2d RGB_color_solid_cube_1.bmp -d 1024x1024 -f -5
