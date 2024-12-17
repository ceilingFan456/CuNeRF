#!/bin/bash

# Source and destination directories
##source_folder="/home/simtech/Qiming/CuNeRF-mgpu/save_bicubic-512-uneven_256/CuNeRFx4"    # Replace with your source folder path
##destination_folder="/home/simtech/Qiming/CuNeRF-mgpu/save_bicubic-512-uneven_256_small/CuNeRFx4"    # Replace with your destination folder path

source_folder="/home/simtech/Qiming/CuNeRF-mgpu/save_bicubic-512/CuNeRFx4"    # Replace with your source folder path
destination_folder="/home/simtech/Qiming/CuNeRF-mgpu/save_bicubic-512_small/CuNeRFx4"    # Replace with your destination folder path

mkdir -p $destination_folder

# List of folders to move
folders=("case_00010" "case_00120" "case_00135" "case_00140" "case_00197" "case_00210" "case_00230" "case_00295" "case_00089" "case_00045" "case_00052" "case_00162" "case_00291" "case_00244" "case_00264")

# Move each folder from source to destination
for folder in "${folders[@]}"; do
    # Check if the folder exists
    if [ -d "$source_folder/$folder" ]; then
        echo "Moving $folder to $destination_folder"
        mv "$source_folder/$folder" "$destination_folder"
    else
        echo "Folder $folder does not exist in $source_folder"
    fi
done

echo "Folders moved successfully."

