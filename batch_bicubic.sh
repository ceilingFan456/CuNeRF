#!/bin/bash

# Define the base directory where all cases are stored
BASE_DIR="/home/simtech/Qiming/kits19/data"

# Path to the bicubic.py script
BICUBIC_SCRIPT="/home/simtech/Qiming/CuNeRF-mgpu/src/bicubic.py"

# Number of evaluations and scale (you can modify these if needed)
N_EVAL=0
SCALE=2

# Starting index for iteration (change this value to start from a different case)
START_INDEX=95

# Counter to keep track of the iteration
counter=0

# Loop through each subdirectory in BASE_DIR
for CASE_DIR in "$BASE_DIR"/*; do
    if [ -d "$CASE_DIR" ]; then
        # Increment the counter
        counter=$((counter + 1))

        # Only start processing from the 96th iteration onwards
        if [ $counter -ge $START_INDEX ]; then
            # Extract the case name from the directory path (e.g., "case_00000")
            CASE_NAME=$(basename "$CASE_DIR")
            
            # Run bicubic.py with the current case
            echo "Running bicubic.py for case: $CASE_NAME (iteration $counter)"
            python "$BICUBIC_SCRIPT" --case "$CASE_NAME" --n_eval $N_EVAL --scale $SCALE
        fi
    fi
done
