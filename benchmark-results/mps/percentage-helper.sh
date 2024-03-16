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

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY='/users/pzhou/projects/gpuless/src/mps_pipe'
export CUDA_MPS_LOG_DIRECTORY='/users/pzhou/projects/gpuless/src/mps_logs'

output1="$(./$1 0)"
parsed_output1=$(read_stream_output "" "$output1")

echo "$parsed_output1" >> "./bench-results/test-mps/mps-$2-$3.out"