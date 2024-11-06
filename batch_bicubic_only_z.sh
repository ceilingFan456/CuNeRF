#!/bin/bash

# Define the base directory where all cases are stored
BASE_DIR="/home/simtech/Qiming/kits19/data"

# Path to the bicubic.py script
BICUBIC_SCRIPT="/home/simtech/Qiming/CuNeRF-mgpu/src/bicubic-z.py"

# Number of evaluations and scale (you can modify these if needed)
N_EVAL=0
SCALE=4

# Sorted list of specific cases to process
cases=("case_00010" "case_00045" "case_00052" "case_00089" 
       "case_00120" "case_00135" "case_00140" "case_00162" 
       "case_00197" "case_00210" "case_00230" "case_00291" "case_00295")

# Loop through the specified cases
for CASE_NAME in "${cases[@]}"; do
    CASE_DIR="$BASE_DIR/$CASE_NAME"
    if [ -d "$CASE_DIR" ]; then
        # Run bicubic.py with the current case
        echo "Running bicubic.py for case: $CASE_NAME"
        python "$BICUBIC_SCRIPT" --case "$CASE_NAME" --n_eval $N_EVAL --scale $SCALE --save_folder bicubic_only_z
    else
        echo "Directory for case $CASE_NAME does not exist!"
    fi
done
