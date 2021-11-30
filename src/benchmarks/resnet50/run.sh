#!/usr/bin/env bash

level=debug
if [[ $1 == "info" ]]; then
    level=info
elif [[ $1 == "trace" ]]; then
    level=trace
elif [[ $1 == "off" ]]; then
    level=off
fi

SPDLOG_LEVEL=${level} \
EXECUTOR_TYPE=tcp \
MANAGER_PORT=8002 \
MANAGER_IP=192.168.1.109 \
CUDA_BINARY=/usr/lib/libtorch_cuda.so \
LD_PRELOAD=/home/luke/ethz/master/thesis/msc-lutobler-gpuless/src/build/libgpuless.so \
./image-recognition
