import cv2
import numpy as np

def compare_images(image_path1, image_path2, output_path):
    # Load the two grayscale images
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # Check if both images were loaded correctly
    if img1 is None or img2 is None:
        print("Error: One or both image paths are invalid.")
        return

    # Ensure both images are the same size
    if img1.shape != img2.shape:
        print("Error: Images must be the same size for comparison.")
        return

    # Compute the absolute difference between the two images
    difference = cv2.absdiff(img1, img2)
    _, binary_image = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)


    # Save the resulting difference image
    cv2.imwrite(output_path, binary_image)

    # Display the difference image
    # cv2.imshow("Difference Image", difference)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()

# Example usage
image_path1 = "/home/simtech/Qiming/CuNeRF-mgpu/save_bicubic-512-uneven_256/CuNeRFx2/case_00000/eval/000_gt.png"
image_path2 = "/home/simtech/Qiming/CuNeRF-mgpu/save_bicubic-512-uneven_256/CuNeRFx2/case_00000/eval/000_ours.png"
output_path = "output.png"

# image_path1 = "/home/simtech/Qiming/CuNeRF-mgpu/save-vanilla-512-256/CuNeRFx2/case_00010/traineval/000_gt.png"
# image_path2 = "/home/simtech/Qiming/CuNeRF-gt/fix/CuNeRFx2/case_00010/traineval/000_gt.png"
# output_path = "output.png"

compare_images(image_path1, image_path2, output_path)
