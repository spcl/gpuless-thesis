#!/usr/bin/env bash

sizes=(
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
    8192
    16384
    32768
    65536
    131072
    262144
    524288
    1048576
    2097152
    4194304
    8388608
    16777216
    33554432
    67108864
    134217728
    268435456
)

n_runs=$1
ip=$2

for s in "${sizes[@]}"; do
    # ./benchmark-tcp ${n_runs} ${s} ${ip} > ../bench-tcp-rtt-${n_runs}-${s}-ms.log
    # ./benchmark-tcp ${n_runs} ${s} ${ip} > ../bench-local-rtt-${n_runs}-${s}-ms.log
    ./benchmark-local ${n_runs} ${s} > ../bench-no-net-rtt-${n_runs}-${s}-ms.log
done
