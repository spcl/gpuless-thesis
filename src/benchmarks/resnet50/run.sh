#!/usr/bin/env bash

if [[ $1 == "debug" ]]; then
    level=debug
elif [[ $1 == "off" ]]; then
    level=off
else
    level=info
fi

SPDLOG_LEVEL=${level} \
CUDA_BINARY=/usr/lib/libtorch_cuda.so,/usr/lib/libcudnn_ops_infer.so.8.2.4 \
LD_PRELOAD=/home/luke/ethz/master/thesis/msc-lutobler-gpuless/src/build/libgpuless.so \
./image-recognition
