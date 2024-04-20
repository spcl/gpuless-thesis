#!/usr/bin/env bash

n_runs=15

# command line arguments
project_dir="$1"
bench_dir="$2"
bench_type="$3"
cuda_bin="$4"
remote_ip="$5"
note="$6"

bechmark_name=$(basename "$bench_dir")
out_file="${project_dir}/benchmark-results/cuda-analysis/benchmark-${bechmark_name}-${bench_type}-${note}-$(date --iso-8601=seconds)"
manager_bin=$project_dir/src/build/manager_trace

run_bench_native() {
	echo 'native performance'

	pushd .
	cd $bench_dir
	printf '' >"$out_file" # clear output file

	for ((i = 0; i < $n_runs; i++)); do
		t=$(LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$HOME/conda/lib ./run_timed_reordered.sh)
		echo "$t\n" >>"$out_file"
		echo "run ${i}: ${t}"
		sleep 1.0
	done

	popd
	echo ''
}

run_bench_remote() {
	echo 'local network performance'

	pushd .
	cd $bench_dir
	printf '' >"$out_file" # clear output file

	for ((i = 0; i < $n_runs; i++)); do
		t=$(LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64 CUDA_VISIBLE_DEVICES=0 SPDLOG_LEVEL=OFF MANAGER_IP=${remote_ip} MANAGER_PORT=8002 CUDA_BINARY=${cuda_bin} EXECUTOR_TYPE=tcp LD_PRELOAD=${project_dir}/src/build_trace/libgpuless.so ./run_timed_reordered.sh)
		printf "$t\n" >>"$out_file"
		echo "run ${i}: ${t}"
		sleep 1.0
	done

	popd
	echo ''
}

case $bench_type in
native)
	run_bench_native
	;;
remote)
	run_bench_remote
	;;
*)
	echo 'unknown benchmark type' $bench_type
	;;
esac
