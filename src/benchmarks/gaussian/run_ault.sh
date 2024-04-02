#!/usr/bin/env bash

EXECUTOR_TYPE=tcp \
	LD_LIBRARY_PATH=$CUDA_HOME/lib64
MANAGER_IP=127.0.0.1 \
MANAGER_PORT=8002 \
CUDA_BINARY=./gaussian \
LD_PRELOAD=$HOME/gpuless/src/build/libgpuless.so \
./gaussian -f ./matrix1024.txt | tail -n3
