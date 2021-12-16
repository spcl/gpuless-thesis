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
    2048
    4096
    )

for i in "${sizes[@]}"; do
    out_f="image-recognition-batched-total-${i}-$(date --iso-8601=seconds).out"
    for j in {1..10}; do
        ./image-recognition-batched-total ${i} >> ${out_f}
    done
done
