# global variables
current_idx=0

# Function to convert nanoseconds to seconds
calculate_elapsed_time() {
    local start_time=$1
    local end_time=$2
    local diff=$((end_time - start_time))
    local seconds=$(awk "BEGIN {printf \"%.6f\n\", $diff / 1000000000}")
    echo "$seconds seconds"
}

read_stream_output() {
    ident="$1" # First argument: ident
    stream_out="$2" # Second argument: stream_out
    
    IFS=$'\n' # Set the Internal Field Separator to newline
    ls=() # Array to store the lines
    while read -r line; do
        # Customize the parsing logic here
        ls+=("$ident,$line")
    done <<< "$stream_out"
    
    printf '%s\n' "${ls[@]}"
}

# File to write the results
output_file="./bench-results/$1/$1-mps-bench-result-$4-$3.out"

start_time=$(date +%s%N) # Get start time in nanoseconds
output="$(./$2 $current_idx)" # start from the beginning
end_time=$(date +%s%N) # Get end time in nanoseconds

parsed_output=$(read_stream_output "$4" "$output")

current_idx="${parsed_output##* }"

elapsed_time=$(calculate_elapsed_time $start_time $end_time)
echo "$parsed_output" >> "$output_file"
echo "$elapsed_time" >> "$output_file"

while [[ ! $parsed_output =~ 'success' ]]; do # while the parsed output does not contain 'success'
    start_time=$(date +%s%N) # Get start time in nanoseconds
    output="$(./$2 $current_idx)" # resume runing
    end_time=$(date +%s%N) # Get end time in nanoseconds

    parsed_output=$(read_stream_output "$4" "$output")
    # echo "$parsed_output" >> "$output_file" # Display parsed output
    current_idx="${parsed_output##* }"
    
    elapsed_time=$(calculate_elapsed_time $start_time $end_time)
    echo "RESUME $parsed_output" >> "$output_file"
    echo "$elapsed_time" >> "$output_file"
done

# echo quit | nvidia-cuda-mps-control
# in the end, sum up all elapsed_time