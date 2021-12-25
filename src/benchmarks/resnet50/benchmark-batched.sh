#!/usr/bin/env bash

sizes=(
    1
    2
    4
    8
    16
    32
    64
    128
    256
    512
    1024
    )

# for i in "${sizes[@]}"; do
#     out_f="image-recognition-batched-total-${i}-$(date --iso-8601=seconds).out"
#     for j in {1..10}; do
#         ./image-recognition-batched-total ${i} >> ${out_f}
#     done
# done

remote_ip=192.168.1.109
cuda_bin=/usr/lib/libtorch_cuda.so
project_dir=/home/luke/ethz/master/thesis/msc-lutobler-gpuless

for i in "${sizes[@]}"; do
    out_f="image-recognition-batched-total-${i}-$(date --iso-8601=seconds).out"
    for j in {1..10}; do
        MANAGER_IP=${remote_ip} \
        MANAGER_PORT=8002 \
        CUDA_BINARY=${cuda_bin} \
        EXECUTOR_TYPE=tcp \
        LD_PRELOAD=${project_dir}/src/build/libgpuless.so \
        ./image-recognition-batched-total ${i} >> ${out_f}
    done
done
