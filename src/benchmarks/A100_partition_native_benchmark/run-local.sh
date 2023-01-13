#!/usr/bin/env bash


n=$1
file="../A100_partition_native_benchmark/$n"
touch $file


project_dir="${HOME}/gpuless"

mkdir -p results

benchmark="srad_v1"
benchmark_run="./run_batched.sh"

build="${project_dir}/src/build_trace"
#cuda_bin="$HOME/libtorch/lib/libtorch_cuda.so"
cuda_bin="./srad_v1"

export MANAGER_IP=127.0.0.1 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64 
export CUDA_VISIBLE_DEVICES=0 
export SPDLOG_LEVEL=OFF 
export MANAGER_PORT=8002 
export CUDA_BINARY=${cuda_bin} 
export EXECUTOR_TYPE=tcp 
export LD_PRELOAD=${build}/libgpuless.so


result="../A100_partition_native_benchmark/results/a100-mig-partitions-${benchmark}-$n"

cd ../$benchmark


while [ 1 ]
do
	while [ ! -s $file ]
	do
        	sleep 0.01
	done
	t=$(${benchmark_run})
	echo "$t" > $result
	> $file
done
