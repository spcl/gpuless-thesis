#!/usr/bin/env bash

EXECUTOR_TYPE=tcp \
MANAGER_IP=127.0.0.1 \
CUDA_BINARY=./bfs \
LD_PRELOAD=/home/paul/ETH/HS22/DPHPC/gpuless/src/cmake-build-debug/libgpuless.so \
./bfs ./graph1MW_6.txt
