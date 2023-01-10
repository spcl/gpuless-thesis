#!/usr/bin/bash

LD_PRELOAD="$HOME/gpuless/src/build_trace/libgpuless.so" \
EXECUTOR_TYPE=tcp \
  MANAGER_IP=127.0.0.1 \
  MANAGER_PORT=8002 \
  CUDA_BINARY=./latency \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64 \
./latency
