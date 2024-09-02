import os
import cv2
import argparse

def smooth_image(image_path, kernel_size=(5, 5)):
    """
    Apply a Gaussian blur to smooth the image.
    :param image_path: Path to the input image.
    :param kernel_size: The size of the Gaussian kernel.
    :return: The smoothed image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    smoothed_image = cv2.GaussianBlur(image, kernel_size, 0)
    return smoothed_image

def process_folder(folder_path):
    """
    Process all XX_ours.png images in the specified folder.
    :param folder_path: Path to the folder containing the images.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith("_ours.png"):
            original_path = os.path.join(folder_path, filename)
            smoothed_image = smooth_image(original_path)

            # Save the original image as XX_ours_original.png
            original_save_path = os.path.join(folder_path, filename.replace("_ours.png", "_ours_original.png"))
            os.rename(original_path, original_save_path)
            print(f"Saved original image as {original_save_path}")

            # Save the smoothed image as XX_ours.png
            smoothed_save_path = os.path.join(folder_path, filename)
            cv2.imwrite(smoothed_save_path, smoothed_image)
            print(f"Saved smoothed image as {smoothed_save_path}")

def main(args):
    process_folder(args.folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smooth XX_ours.png images in the evaluation folder.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the evaluation folder containing the images.")
    args = parser.parse_args()
    main(args)
