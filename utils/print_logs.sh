#!/bin/bash

# Define the base directory where all cases are stored
BASE_DIR="/home/simtech/Qiming/CuNeRF-mgpu/bicubic_only_z/CuNeRFx2" 

# Loop through each subdirectory in BASE_DIR that matches "case_00XXX"
for CASE_DIR in "$BASE_DIR"/case_00*; do
    if [ -d "$CASE_DIR" ]; then
        # Extract the case name from the directory path
        CASE_NAME=$(basename "$CASE_DIR")
        
        # Define the path to the logs.txt file
        LOG_FILE="$CASE_DIR/logs.txt"
        
        # Check if the logs.txt file exists
        if [ -f "$LOG_FILE" ]; then
            # Extract the psnr and training psnr from the logs.txt
            PSNR=$(grep -i "psnr" "$LOG_FILE" | grep -v "training" | tail -1 | awk '{print $NF}')
            TRAINING_PSNR=$(grep -i "PSNR Training" "$LOG_FILE" | tail -1 | awk '{print $NF}')
            
            # Print the case name, psnr, and training psnr
            echo "Case: $CASE_NAME"
            echo "PSNR: $PSNR"
            echo "Training PSNR: $TRAINING_PSNR"
            echo "-----------------------------------"
        else
            echo "logs.txt not found in $CASE_NAME"
        fi
    fi
done
