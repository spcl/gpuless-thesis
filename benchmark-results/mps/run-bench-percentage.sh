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

# START
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY='/users/pzhou/projects/gpuless/src/mps_pipe'
export CUDA_MPS_LOG_DIRECTORY='/users/pzhou/projects/gpuless/src/mps_logs'

echo quit | nvidia-cuda-mps-control

# single run
echo "single gpu"
output="$(./$1 0)"
parsed_output=$(read_stream_output "" "$output")
echo "$parsed_output" >> "./bench-results/test-mps/mps-single.out"

# # 100% thread percentage
# nvidia-cuda-mps-control -d

# echo "100% thread percentage"
# sh percentage-helper.sh $1 100 1 &
# sh percentage-helper.sh $1 100 2 
# echo quit | nvidia-cuda-mps-control

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=True
# nvidia-cuda-mps-control -d

echo "custom thread percentage, 100% & 100%"
sh custom-percentage-helper.sh $1 100 1001 &
sh custom-percentage-helper.sh $1 100 1002

echo quit | nvidia-cuda-mps-control

# # 75% thread percentage
# nvidia-cuda-mps-control -d
# echo set_default_active_thread_percentage 75|nvidia-cuda-mps-control >/dev/null

# echo "75% thread percentage"
# sh percentage-helper.sh $1 75 1 &
# sh percentage-helper.sh $1 75 2 
# echo quit | nvidia-cuda-mps-control

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=True
# nvidia-cuda-mps-control -d

echo "custom thread percentage, 75% & 75%"
sh custom-percentage-helper.sh $1 75 751 &
sh custom-percentage-helper.sh $1 75 752

echo quit | nvidia-cuda-mps-control

# # 50% thread percentage
# nvidia-cuda-mps-control -d
# echo set_default_active_thread_percentage 50|nvidia-cuda-mps-control >/dev/null

# echo "50% thread percentage"
# sh percentage-helper.sh $1 50 1 &
# sh percentage-helper.sh $1 50 2 
# echo quit | nvidia-cuda-mps-control

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=True
# nvidia-cuda-mps-control -d

echo "custom thread percentage, 50% & 50%"
sh custom-percentage-helper.sh $1 50 501 &
sh custom-percentage-helper.sh $1 50 502

echo quit | nvidia-cuda-mps-control

# # 25% thread percentage
# nvidia-cuda-mps-control -d
# echo set_default_active_thread_percentage 25|nvidia-cuda-mps-control >/dev/null

# echo "25% thread percentage"
# sh percentage-helper.sh $1 25 1 &
# sh percentage-helper.sh $1 25 2 
# echo quit | nvidia-cuda-mps-control

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=True
# nvidia-cuda-mps-control -d

echo "custom thread percentage, 100% & 100%"
sh custom-percentage-helper.sh $1 25 251 &
sh custom-percentage-helper.sh $1 25 252

echo quit | nvidia-cuda-mps-control

# set non-uniform distribution
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=True
# nvidia-cuda-mps-control -d

echo "custom thread percentage, 75% & 25%"
sh custom-percentage-helper.sh $1 25 1 &
sh custom-percentage-helper.sh $1 75 1

echo quit | nvidia-cuda-mps-control
