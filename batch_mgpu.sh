#!/bin/bash

## let it run all to finish
set +e

BASE_DIR="/home/simtech/Qiming/kits19/data"
CONFIG_FILE="configs/example.yaml"
SEED=26  # Set your desired seed value here
SCALE=2  # Set your desired scale value here

# Set up a random seed function using the SEED variable
random_source() {
    seed=$SEED
    while :; do
        seed=$(( (1103515245 * seed + 12345) % 2147483648 ))
        printf "%03d" $((seed % 256))
    done
}

# Select 10 random cases from the available directories using the seed
case_dirs=($(find "$BASE_DIR" -maxdepth 1 -type d -name 'case_*'))
selected_dirs=($(shuf -e "${case_dirs[@]}" -n 10 --random-source=<(random_source)))

for case_dir in "${selected_dirs[@]}"; do
    case_name=$(basename "$case_dir")
    imaging_file="${case_dir}/${case_name}.nii.gz"

    # Run the command with the correct options
    python run.py CuNeRFx$SCALE --cfg "$CONFIG_FILE" --scale $SCALE --mode train --file "$imaging_file" --save_map --resume --multi_gpu

    echo "Processed ${imaging_file}"
done
