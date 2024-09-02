import SimpleITK as sitk
import torch    
import argparse
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_file(filepath):
    data = sitk.GetArrayFromImage(sitk.ReadImage(filepath)).astype(float)
    data = torch.from_numpy(data).float().cuda()
    return data

def load_data(filepath):
    data = load_file(filepath)
    data = nomalize(data)
    data = align(data)
    length, H, W = data.shape
    print(length, H, W)
    return data

def align(data):
    if data.shape[1] != data.shape[2]:
        if data.shape[0] == data.shape[2]:
            data = data.permute(1, 0, 2)
        else:
            data = data.permute(2, 1, 0)
    return data

def nomalize(data):
    return (data - data.min()) / (data.max() - data.min())

def calculate_metrics(current_frame, previous_frame):
    psnr_value = peak_signal_noise_ratio(previous_frame, current_frame)
    ssim_value = structural_similarity(previous_frame, current_frame, data_range=current_frame.max() - current_frame.min())
    return psnr_value, ssim_value

def visualize(data, n_eval, scale):
    length, _, _ = data.shape
    eval_indices = [int(i * (length - 1) / (n_eval - 1)) for i in range(n_eval)]
    train_indices = list(filter(lambda x: x % scale == 0, range(length)))

    previous_frame = None

    for i in range(length):
        # Convert tensor to numpy and scale to [0, 255] for visualization
        frame = data[i].cpu().numpy() * 255
        frame = frame.astype(np.uint8)
        
        # Convert grayscale to BGR for display with colored text
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Determine if the frame is for evaluation or training
        label0 = f"Evaluation @ 50" if i in eval_indices else ""
        label1 = f"Training" if i not in train_indices else ""
        
        # Add the label to the frame
        cv2.putText(frame_bgr, label0, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, label1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Calculate PSNR and SSIM if there is a previous frame
        if previous_frame is not None:
            psnr_value, ssim_value = calculate_metrics(frame, previous_frame)
            psnr_text = f"PSNR: {psnr_value:.2f}"
            ssim_text = f"SSIM: {ssim_value:.2f}"
            cv2.putText(frame_bgr, psnr_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, ssim_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Frame', frame_bgr)

        # Wait for a key press to move to the next frame
        key = cv2.waitKey(0)
        if key == ord('q'):  # Press 'q' to quit early
            break

        # Update the previous frame
        previous_frame = frame

    cv2.destroyAllWindows()

def main(args):
    filepath = os.path.join("/home/simtech/Qiming/kits19/data/", args.case,'imaging.nii.gz')
    print("Loading data from", filepath)
    data = load_data(filepath)
    print("Data loaded successfully")

    visualize(data, args.n_eval, args.scale)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case_00000")
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    main(args)
