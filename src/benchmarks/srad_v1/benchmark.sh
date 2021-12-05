#!/usr/bin/env bash

out_file="benchmark-srad_v1-$(date --iso-8601=seconds)"
# project_dir=$HOME/ethz/master/thesis/msc-lutobler-gpuless
project_dir=../../../
manager_bin=$project_dir/src/bin/manager_trace
bench_dir=$project_dir/src/benchmarks/srad_v1

if [[ "$1" == "local" ]]; then
    echo 'local native performance'
    pushd .
    cd $bench_dir
    printf '' > $out_file-local.out
    for i in {1..100}; do
        echo run $i
        t=$(./run | tail -n1 | cut -d' ' -f1)
        printf "$t\n"
        printf "$t\n" >> "${out_file}-local.out"
    done
    popd
elif [[ "$1" == "local-net" ]]; then
    echo 'local network performance'
    pushd .
    cd $bench_dir
    printf '' > $out_file-local-network.out
    for i in {1..100}; do
        echo run $i
        SPDLOG_LEVEL=off $manager_bin &
        sleep 0.5
        t=$(SPDLOG_LEVEL=off CUDA_BINARY=./srad EXECUTOR_TYPE=tcp LD_PRELOAD=$project_dir/src/build/libgpuless.so ./run | tail -n1 | cut -d' ' -f1)
        printf "$t\n"
        printf "$t\n" >> "${out_file}-local-network.out"
        pkill manager_trace
        sleep 0.5
    done
    popd
elif [[ "$1" == "remote-net" ]]; then
    echo TODO
    echo 'remote network performance'
fi



