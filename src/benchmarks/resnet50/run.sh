#!/usr/bin/env bash

level=debug
if [[ $1 == "info" ]]; then
    level=info
elif [[ $1 == "off" ]]; then
    level=off
fi

SPDLOG_LEVEL=${level} \
EXECUTOR_TYPE=tcp \
CUDA_BINARY=/usr/lib/libtorch_cuda.so \
LD_PRELOAD=/home/luke/ethz/master/thesis/msc-lutobler-gpuless/src/bin/libgpuless.so \
./image-recognition
