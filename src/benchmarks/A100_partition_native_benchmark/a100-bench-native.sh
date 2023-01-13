#!/usr/bin/env bash

BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_colour () {
        echo -e "${BLUE}$1${NC}"
}

project_dir="${HOME}/gpuless"

mkdir -p results

benchmark="resnet50-py"
benchmark_run="./../${benchmark}/run_batched.sh"

build="${project_dir}/src/build_trace"
cuda_bin="${build}/libgpuless.so"
manager="./../../build_trace/manager_trace"
manager_ip="127.0.0.1"

env="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64 CUDA_VISIBLE_DEVICES=0 SPDLOG_LEVEL=OFF MANAGER_IP=${manager_ip} MANAGER_PORT=8002 CUDA_BINARY=${cuda_bin} EXECUTOR_TYPE=tcp LD_PRELOAD=${build}/libgpuless.so"

result="results/a100-mig-partitions-native-${benchmark}"

node_list=(1 2 3 4 5 6 7)

device=0


bench(){
	p=$1
	n=$(echo $1 | awk -F"," '{print NF}')

	#remote benchmarks

	sudo nvidia-smi mig -i ${device} -cgi $1 -C
	ids=($(nvidia-smi -L | grep "MIG " | sed -rn "s/.*MIG-([a-f0-9-]*).*/\\1/p"))
	i=1
	while [ $i -le $n ]
	do
		printf "MIG-${ids[$i]}" > $i
		((++i))
	done
	for i in ${node_list[@]:0:$n}
	do
		while [ -s $i ]
		do
			sleep 0.01
		done
		printf "$p\nnode $i\n" >> "${result}"
		cat $result-$i >> $result
	done
	
	#native benchmarks:
	#set -- $ids
	#i=1
	#while [ $i -le $n ]
	#do 
#		CUDA_VISIBLE_DEVICES=MIG-${ids[$i]} LD_LIBRARY_PATH=$CUDA_HOME/lib64 $benchmark_run > $result_native-$i &
#	done	
#	wait
#	i=1
#	while [ $i -le $n ]
#	do
#		printf "$p\nrun $i\n" >> $result_native	
#		cat $result_native-$i >> $result_native
#	done	
	sleep 2
	sudo nvidia-smi mig -i ${device} -dci
	sudo nvidia-smi mig -i ${device} -dgi
}
partitions="19
19,19
19,19,19
19,19,19,19
19,19,19,19,19
19,19,19,19,19,19
19,19,19,19,19,19,19
14
14,14
14,14,14
9
9,9
5
0
19,14,5
19,14,9
9,5
9,14,14
"
#set -x

sudo nvidia-smi -i ${device} -mig 1
sudo nvidia-smi mig -i ${device} -dgi

export SPDLOG_LEVEL=OFF

rm $result
for p in $partitions
do 
	echo_colour $p
	bench $p	
done


sudo nvidia-smi -i $device -mig 0
