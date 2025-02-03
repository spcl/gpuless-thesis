#!/bin/bash

# Number of times to run the script
NUM_RUNS=50

# Path to the shell script to run
SCRIPT_TO_RUN="./run-bench-percentage.sh"

# Loop to run the script multiple times
for ((i = 1; i <= NUM_RUNS; i++)); do
    echo "Running script iteration $i ================="
    # Run the script
    bash "$SCRIPT_TO_RUN"
done