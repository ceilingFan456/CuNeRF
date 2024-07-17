import os
import cv2
import numpy as np

IMG_PATH = "/home/simtech/Qiming/CuNeRF/save/CuNeRFx2/case_00000/eval"

# List all files in IMG_PATH
files = os.listdir(IMG_PATH)

# Filter image files
image_files = [f for f in files if f.endswith('.png') and ('_gt.png' in f or '_ours.png' in f)]

# Sort image files to ensure pairs are matched correctly
image_files.sort()

# Iterate over pairs of _gt.png and _ours.png
for i in range(0, len(image_files), 2):
    gt_filename = image_files[i]
    ours_filename = image_files[i + 1]
    
    # Load images
    gt_image = cv2.imread(os.path.join(IMG_PATH, gt_filename))
    ours_image = cv2.imread(os.path.join(IMG_PATH, ours_filename))
    
    # Compute difference
    difference = cv2.absdiff(gt_image, ours_image)
    
    # Create color-coded difference image (red to blue scale)
    # Normalize difference to 0-255 range
    normalized_difference = cv2.normalize(difference, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap (assuming grayscale difference)
    difference_colormap = cv2.applyColorMap(normalized_difference, cv2.COLORMAP_JET)
    
    # Save difference image
    difference_save_path = os.path.join(IMG_PATH, f"{gt_filename[:-7]}_difference.png")
    cv2.imwrite(difference_save_path, difference_colormap)
    
    print(f"Difference image saved for {gt_filename} and {ours_filename} as {difference_save_path}")

print("Process completed.")
