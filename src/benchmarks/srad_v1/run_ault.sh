#!/usr/bin/bash

LD_PRELOAD="$HOME/gpuless/src/build/libgpuless.so" \
EXECUTOR_TYPE=tcp \
LD_LIBRARY_PATH=$CUDA_HOME/lib64 \
  MANAGER_IP=127.0.0.1 \
  MANAGER_PORT=8002 \
  CUDA_BINARY=./srad_v1 \
./run
