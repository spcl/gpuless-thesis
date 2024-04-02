#!/usr/bin/bash

LD_PRELOAD="$HOME/gpuless/src/build/libgpuless.so" \
EXECUTOR_TYPE=tcp \
  MANAGER_IP=127.0.0.1 \
  MANAGER_PORT=8002 \
  CUDA_BINARY=./hotspot \
  LD_PRELOAD="$HOME/gpuless/src/build/libgpuless.so" \
./run
