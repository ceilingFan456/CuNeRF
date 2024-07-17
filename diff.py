import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

IMG_PATH = "/home/simtech/Qiming/CuNeRF/save/CuNeRFx2/case_00069/eval"

# List all files in IMG_PATH
files = os.listdir(IMG_PATH)

# Filter image files
image_files = [f for f in files if f.endswith('.png') and ('_gt.png' in f or '_ours.png' in f)]

# Sort image files to ensure pairs are matched correctly
image_files.sort()

# Function to read image using PIL
def read_image(file_path):
    return Image.open(file_path)

# Function to convert PIL image to NumPy array
def pil_to_np(image):
    return np.array(image)

# Iterate over pairs of _gt.png and _ours.png
for i in range(0, len(image_files), 2):
    gt_filename = image_files[i]
    ours_filename = image_files[i + 1]
    
    # Load images
    gt_image = read_image(os.path.join(IMG_PATH, gt_filename))
    ours_image = read_image(os.path.join(IMG_PATH, ours_filename))
    
    # Downsample and resize ground truth
    gt_downsampled = gt_image.resize((gt_image.width // 2, gt_image.height // 2), Image.Resampling.LANCZOS)
    gt_resized = gt_downsampled.resize((512, 512), Image.Resampling.LANCZOS)

    # Convert images to NumPy arrays for calculation
    gt_image_np = pil_to_np(gt_image)
    ours_image_np = pil_to_np(ours_image)
    gt_resized_np = pil_to_np(gt_resized)

    # Compute differences
    difference_gt_ours = np.abs(gt_image_np - ours_image_np)
    difference_ours_resized = np.abs(ours_image_np - gt_resized_np)

    # Normalize the difference images
    normalized_difference_gt_ours = cv2.normalize(difference_gt_ours, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_difference_ours_resized = cv2.normalize(difference_ours_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Plot images
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 15))
    
    # Plot Ground Truth
    axes[0, 0].imshow(gt_image)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    
    # Plot Ground Truth Resized
    axes[0, 1].imshow(gt_resized)
    axes[0, 1].set_title('GT Resized')
    axes[0, 1].axis('off')
    
    # Plot Ours
    axes[0, 2].imshow(ours_image)
    axes[0, 2].set_title('Ours')
    axes[0, 2].axis('off')
    
    # Plot Difference GT-Ours
    im3 = axes[0, 3].imshow(difference_gt_ours, cmap='coolwarm')
    axes[0, 3].set_title('Diff GT-Ours')
    axes[0, 3].axis('off')
    fig.colorbar(im3, ax=axes[0, 3], orientation='vertical')
    
    # Plot Difference Ours-Resized
    im4 = axes[0, 4].imshow(difference_ours_resized, cmap='coolwarm')
    axes[0, 4].set_title('Diff Ours-Resized')
    axes[0, 4].axis('off')
    fig.colorbar(im4, ax=axes[0, 4], orientation='vertical')

    # Plot histograms
    axes[1, 3].hist(difference_gt_ours.ravel(), bins=256, color='blue', alpha=0.7)
    axes[1, 3].set_title('Histogram of Diff GT-Ours')
    axes[1, 3].set_xlim([0, 255])
    
    axes[1, 4].hist(difference_ours_resized.ravel(), bins=256, color='red', alpha=0.7)
    axes[1, 4].set_title('Histogram of Diff Ours-Resized')
    axes[1, 4].set_xlim([0, 255])

    # Plot normalized differences
    axes[2, 3].imshow(normalized_difference_gt_ours, cmap='coolwarm')
    axes[2, 3].set_title('Normalized Diff GT-Ours')
    axes[2, 3].axis('off')
    
    axes[2, 4].imshow(normalized_difference_ours_resized, cmap='coolwarm')
    axes[2, 4].set_title('Normalized Diff Ours-Resized')
    axes[2, 4].axis('off')

    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    combined_save_path = os.path.join(IMG_PATH, f"combined_image_{i // 2}.png")
    plt.savefig(combined_save_path)
    plt.close()

print("Process completed.")
