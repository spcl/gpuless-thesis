#!/usr/bin/env bash


n=$1
file=$n
touch $file


project_dir="${HOME}/gpuless"

mkdir -p results

benchmark="resnet50-py"
benchmark_run="./../${benchmark}/run_batched.sh"

build="${project_dir}/src/build_trace"
cuda_bin="${build}/libgpuless.so"
manager="${build}/manager_trace"

env="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64 CUDA_VISIBLE_DEVICES=0 SPDLOG_LEVEL=OFF MANAGER_PORT=8002 CUDA_BINARY=${cuda_bin} EXECUTOR_TYPE=tcp LD_PRELOAD=${build}/libgpuless.so"

result="results/a100-mig-partitions-${benchmark}-$n"



while [ 1 ]
do
	while [ $(wc -c $file | awk '{print $1}') -eq 0 ]
	do
        	sleep 0.01
	done
	ip=$(cat $file)
	t=$(MANAGER_IP=${ip} ${env} ${benchmark_run})
	printf "$t" > $result
	> $file
done
