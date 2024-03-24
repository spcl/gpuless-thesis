# for mps, TODO: remeber to print SM for each run
# $1 is the benchmark to run.

# put this at the beginning of a mock benchmark
read_stream_output() {
    ident="$1" # First argument: ident
    stream_out="$2" # Second argument: stream_out
    
    IFS=$'\n' # Set the Internal Field Separator to newline
    ls=() # Array to store the lines
    while read -r line; do
        # Customize the parsing logic here
        ls+=("$line")
    done <<< "$stream_out"
    
    printf '%s\n' "${ls[@]}"
}

# START MPS
# program="bfs"
# program_path="/users/pzhou/projects/gpuless/src/benchmarks/bfs/bfs"
# input_path="/users/pzhou/projects/gpuless/src/benchmarks/bfs/graph1MW_6.txt"

program="resnet50-py"
program_path="/users/pzhou/lib/conda/bin/python3 /users/pzhou/projects/gpuless/src/benchmarks/resnet50-py/run.py"
input_path=""
NUM_RUNS=100

echo "start MPS on STREAM and $program"

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY='/users/pzhou/projects/gpuless/src/mps_pipe'
export CUDA_MPS_LOG_DIRECTORY='/users/pzhou/projects/gpuless/src/mps_logs'
echo quit | nvidia-cuda-mps-control

echo "Limit SM to 3/7"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14

echo "Start daemon in background process"
nvidia-cuda-mps-control -d

echo "Limit memory to 1/2 (40G)"
echo set_default_device_pinned_mem_limit 0 5G | nvidia-cuda-mps-control

echo "run $program isolated in mps"
for ((i = 1; i <= NUM_RUNS; i++)); do
    echo "Running single mps $i ================="
    output_isolated="$($program_path $input_path)"
    parsed_output_isolated=$(read_stream_output "" "$output_isolated")
    echo "$parsed_output_isolated" >> "./bench-results/test-mps/mps-$program-isolated14.out"
done

echo "run stream + $program in mps"

for ((i = 1; i <= NUM_RUNS; i++)); do
    echo "Running parallel mps $i ================="
    CUDA_VISIBLE_DEVICES=0 $program_path $input_path >/dev/null &
    CUDA_VISIBLE_DEVICES=0 $program_path $input_path >/dev/null &
    CUDA_VISIBLE_DEVICES=0 $program_path $input_path >/dev/null &
    output_isolated="$(CUDA_VISIBLE_DEVICES=0 $program_path $input_path &)"
    CUDA_VISIBLE_DEVICES=0 $program_path $input_path >/dev/null &
    CUDA_VISIBLE_DEVICES=0 $program_path $input_path >/dev/null &
    CUDA_VISIBLE_DEVICES=0 $program_path $input_path

    parsed_output_isolated=$(read_stream_output "" "$output_isolated")
    echo "$parsed_output_isolated" >> "./bench-results/test-mps/mps-$program-shared14.out"

    sleep 10
done

echo set_default_device_pinned_mem_limit 0 40G | nvidia-cuda-mps-control # works
echo quit | nvidia-cuda-mps-control
echo "finish mps"

echo "start MIG and reset"

echo "enable mig on device 0"
sudo nvidia-smi -i 0 -mig 1
# config the partition into 1g. 5gb, we need config 5.
echo "run $program isolated in mig"
sudo nvidia-smi mig -i 0 -cgi 1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb -C | grep "created GPU instance" | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"
gpu_uuid=$(nvidia-smi -L | grep "GPU 0:" | sed -rn "s/.*GPU-([a-f0-9-]*).*/\\1/p")

for ((i = 1; i <= NUM_RUNS; i++)); do
    echo "Running single mig $i ================="
    output_isolated="$(CUDA_VISIBLE_DEVICES=MIG-GPU-$gpu_uuid/7/0 $program_path $input_path)"
    parsed_output_isolated=$(read_stream_output "" "$output_isolated")
    echo "$parsed_output_isolated" >> "./bench-results/test-mps/mig-$program-isolated1g.out"
done

echo "run stream + $program in mig"


for ((i = 1; i <= NUM_RUNS; i++)); do
    echo "Running parallel mig $i ================="
    CUDA_VISIBLE_DEVICES=MIG-GPU-$gpu_uuid/13/0 $program_path $input_path >/dev/null &
    CUDA_VISIBLE_DEVICES=MIG-GPU-$gpu_uuid/11/0 $program_path $input_path >/dev/null &
    CUDA_VISIBLE_DEVICES=MIG-GPU-$gpu_uuid/12/0 $program_path $input_path >/dev/null &
    output_isolated="$(CUDA_VISIBLE_DEVICES=MIG-GPU-$gpu_uuid/7/0 $program_path $input_path &)"
    CUDA_VISIBLE_DEVICES=MIG-GPU-$gpu_uuid/8/0 $program_path $input_path >/dev/null &
    CUDA_VISIBLE_DEVICES=MIG-GPU-$gpu_uuid/9/0 $program_path $input_path >/dev/null &
    CUDA_VISIBLE_DEVICES=MIG-GPU-$gpu_uuid/10/0 $program_path $input_path 

    parsed_output_isolated=$(read_stream_output "" "$output_isolated")
    echo "$parsed_output_isolated" >> "./bench-results/test-mps/mig-$program-shared1g.out"

    sleep 10
done

echo "delete ci and gi"
sudo nvidia-smi mig -i 0 -dci
sudo nvidia-smi mig -i 0 -dgi

echo "disable mig on device 0"
sudo nvidia-smi -i 0 -mig 0

### OLD ###

# export CUDA_VISIBLE_DEVICES=0
# export CUDA_MPS_PIPE_DIRECTORY='/users/pzhou/projects/gpuless/src/mps_pipe'
# export CUDA_MPS_LOG_DIRECTORY='/users/pzhou/projects/gpuless/src/mps_logs'

# echo quit | nvidia-cuda-mps-control

# # single run
# echo "single gpu"
# output="$(./$1 0)"
# parsed_output=$(read_stream_output "" "$output")
# echo "$parsed_output" >> "./bench-results/test-mps/mps-single.out"

# # single run with 50% of SMs
# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
# echo  "50% isolated thread percentage"
# nvidia-cuda-mps-control -d

# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 output1="$(./$1 0)"
# parsed_output1=$(read_stream_output "" "$output1")
# echo "$parsed_output1" >> "./bench-results/test-mps/mps-50-isolated50.out"

# echo quit | nvidia-cuda-mps-control

# # single run with 25% of SMs
# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
# echo  "25% isolated thread percentage"
# nvidia-cuda-mps-control -d

# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 output1="$(./$1 0)"
# parsed_output1=$(read_stream_output "" "$output1")
# echo "$parsed_output1" >> "./bench-results/test-mps/mps-25-isolated25.out"

# echo quit | nvidia-cuda-mps-control

# # # 100% thread percentage
# # nvidia-cuda-mps-control -d

# # echo "100% thread percentage"
# # sh percentage-helper.sh $1 100 1 &
# # sh percentage-helper.sh $1 100 2 
# # echo quit | nvidia-cuda-mps-control

# # # 75% thread percentage
# # nvidia-cuda-mps-control -d
# # echo set_default_active_thread_percentage 75|nvidia-cuda-mps-control >/dev/null

# # echo "75% thread percentage"
# # sh percentage-helper.sh $1 75 1 &
# # sh percentage-helper.sh $1 75 2 
# # echo quit | nvidia-cuda-mps-control

# # 50% thread percentage
# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

# nvidia-cuda-mps-control -d
# echo  "50% thread percentage"
# echo set_default_active_thread_percentage 50|nvidia-cuda-mps-control >/dev/null

# sh percentage-helper.sh $1 50 shared1 &
# sh percentage-helper.sh $1 50 shared2 

# echo quit | nvidia-cuda-mps-control

# # 25% thread percentage
# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

# nvidia-cuda-mps-control -d
# echo "25% thread percentage"
# echo set_default_active_thread_percentage 25|nvidia-cuda-mps-control >/dev/null

# sh percentage-helper.sh $1 25 shared1 &
# sh percentage-helper.sh $1 25 shared2 &
# sh percentage-helper.sh $1 25 shared3 &
# sh percentage-helper.sh $1 25 shared4 

# echo quit | nvidia-cuda-mps-control

# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

# # # set non-uniform distribution
# # export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
# # export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=True
# # # nvidia-cuda-mps-control -d

# # echo "custom thread percentage, 75% & 25%"
# # sh custom-percentage-helper.sh $1 25 1 &
# # sh custom-percentage-helper.sh $1 75 1

# # echo quit | nvidia-cuda-mps-control
