#!/usr/bin/env bash
set -x

BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_colour () {
        echo -e "${BLUE}$1${NC}"
}

project_dir="${HOME}/gpuless/"

mkdir -p results

benchmark="resnet50-py"
benchmark_run="../${benchmark}/run_batched.sh"

build="${project_dir}/src/build_trace"
cuda_bin="${build}/libgpuless.so"
manager="${build}/manager_trace"

env="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64 CUDA_VISIBLE_DEVICES=0 SPDLOG_LEVEL=OFF MANAGER_IP=127.0.0.1 MANAGER_PORT=8002 CUDA_BINARY=${cuda_bin} EXECUTOR_TYPE=tcp LD_PRELOAD=${build}/libgpuless.so"

result="results/a100-mig-partitions-${benchmark}"

device=0

sudo nvidia-smi -i ${device} -mig 1
sudo nvidia-smi mig -i ${device} -dgi

bench(){
	n=$(echo $1 | awk -F"," '{print NF}')
	sudo nvidia-smi mig -i ${device} -cgi $1 -C
	SPDLOG_LEVEL=OFF $manager 
	for i in {1..$n}
	do
		$env $benchmark_run > "${result}-temp-$i" &
	done
	wait
	for i in {1..$n}
	do
		cat "$1\n$result-temp-$i\n" >> "${result}"
	done
	sleep 1.0
	killall manager_trace
	sudo nvidia-smi mig -i ${device} -dci
	sudo nvidia-smi mig -i ${device} -dgi
}

partitions="9,9"

for p in $partitons
do 
	echo_colour $p
	bench $p	
done
