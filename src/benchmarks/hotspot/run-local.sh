#!/usr/bin/env bash

EXECUTOR_TYPE=tcp \
MANAGER_IP=127.0.0.1 \
MANAGER_PORT=8002 \
CUDA_BINARY=./hotspot \
LD_PRELOAD=/home/paul/ETH/HS22/DPHPC/gpuless/src/cmake-build-debug/libgpuless.so \
./hotspot 512 2 2 ./data/temp_512 ./data/power_512 output.out
