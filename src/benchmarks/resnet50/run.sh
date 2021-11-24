#!/usr/bin/env bash

if [[ $1 == "debug" ]]; then
    level=debug
elif [[ $1 == "off" ]]; then
    level=off
else
    level=info
fi

SPDLOG_LEVEL=${level} \
EXECUTOR_TYPE=local \
CUDA_BINARY=/usr/lib/libtorch_cuda.so \
LD_PRELOAD=/home/luke/ethz/master/thesis/msc-lutobler-gpuless/src/bin/libgpuless.so \
./image-recognition
