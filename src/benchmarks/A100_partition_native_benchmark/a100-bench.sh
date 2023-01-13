#!/usr/bin/env bash
set -x

BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_colour () {
        echo -e "${BLUE}$1${NC}"
}

project_dir="${HOME}/gpuless"

mkdir -p results
mkdir -p sync

benchmark="resnet50-py"
benchmark_run="../${benchmark}/run_batched.sh"

build="${project_dir}/src/build_trace"
cuda_bin="${build}/libgpuless.so"
manager="${build}/manager_trace"
manager_ip=$(ifconfig 2> /dev/null | grep -E "([0-9]{1,3}\.){3}[0-9]{1,3}" | grep -v 127.0.0.1 | awk '{ print $2 }' | cut -f2 -d:)

env="LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64 CUDA_VISIBLE_DEVICES=0 SPDLOG_LEVEL=OFF MANAGER_IP=127.0.0.1 MANAGER_PORT=8002 CUDA_BINARY=${cuda_bin} EXECUTOR_TYPE=tcp LD_PRELOAD=${build}/libgpuless.so"

result="results/a100-mig-partitions-${benchmark}"
result_native="results/a100-mig-partitions-native-${benchmark}"

node_list=(06 10)

device=0


bench(){
	n=$(echo $1 | awk -F"," '{print NF}')

	#remote benchmarks

	#sudo nvidia-smi mig -i ${device} -cgi $1 -C
	ids=($(nvidia-smi -L | grep "MIG " | sed -rn "s/.*MIG-([a-f0-9-]*).*/\\1/p"))
	MANAGER_IP=$manager_ip SPDLOG_LEVEL=OFF $manager &
	printf "$manager_ip" | tee ${node_list[@]:0:$n}
	for i in ${node_list[@]:0:$n}
	do
		while [ $(wc -c $file | awk '{print $1}') -gt 0 ]
		do
			sleep 0.01
		done
		printf "$p\nnode $i\n" >> "${result}"
		cat $result-$i >> $result
	done
	killall manager_trace
	
	#native benchmarks:
	set -- $ids
	i=1
	while [ $i -le $n ]
	do 
		CUDA_VISIBLE_DEVICES=MIG-${ids[$i]} LD_LIBRARY_PATH=$CUDA_HOME/lib64 $benchmark_run > $result_native-$i &
	done	
	wait
	i=1
	while [ $i -le $n ]
	do
		printf "$p\nrun $i\n" >> $result_native	
		cat $result_native-$i >> $result_native
	done	
	#sudo nvidia-smi mig -i ${device} -dci
	#sudo nvidia-smi mig -i ${device} -dgi
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
14,14,9
"


#sudo nvidia-smi -i ${device} -mig 1
#sudo nvidia-smi mig -i ${device} -dgi


rm $result $result_native
for p in $partitions
do 
	echo_colour $p
	bench $p	
done


#sudo nvidia-smi -i $device -mig 0
