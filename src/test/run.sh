#!/usr/bin/env bash

lib='/home/luke/ethz/master/thesis/msc-lutobler-gpuless/src/build/libgpuless.so'

basic_targets=(
    unit/simple
    unit/objkernel
    unit/manykernel
    unit/manymodule
)

run_unit() {
    cuda_bin="$1"
    target="$2"
    printf "running: ${target}\n\n"
    CUDA_BINARY=./${cuda_bin} LD_PRELOAD=${lib} ./${target}
    printf "\n"
}

run_custom() {
    cuda_bin="$1"
    target="$2"
    dir="$3"
    printf "running custom: ${dir}\n"
    pushd . >/dev/null
    cd "$dir"
    run_unit "$cuda_bin" "$target"
    popd >/dev/null
}

# unit tests
for target in ${basic_targets[@]}; do
    run_unit "$target" "$target"
done

# real benchmarks
run_custom 'hotspot' 'run' 'hotspot'
run_custom 'dwt2d' 'run' 'dwt2d'
run_custom 'srad' 'run' 'srad_v1'
