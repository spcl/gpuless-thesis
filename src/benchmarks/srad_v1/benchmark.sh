#!/usr/bin/env bash

out_file="benchmark-srad_v1-$(date --iso-8601=seconds)"
# project_dir=$HOME/ethz/master/thesis/msc-lutobler-gpuless
project_dir=../../../
manager_bin=$project_dir/src/build/manager_trace
bench_dir=$project_dir/src/benchmarks/srad_v1

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

# echo 'local network performance'

# echo 'remote network performance'
