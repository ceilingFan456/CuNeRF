import os
import cv2
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_image(folder, filename):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    return None

def add_text_to_image(image, text, position=(10, 30), font_scale=1, color=(255, 255, 255), thickness=2):
    """Adds text to the image at the specified position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def compare_images(folders, des):
    img_folders = [os.path.join(folder, 'eval') for folder in folders]
    gt_images = [f for f in os.listdir(img_folders[0]) if f.endswith('_gt.png')]
    gt_images.sort()

    total_ssims = []
    total_psnrs = []
    
    skip = False
    
    for gt_img_name in gt_images:
        base_name = gt_img_name.replace('_gt.png', '')
        ours_img_name = f"{base_name}_ours.png"

        gt_img = load_image(img_folders[0], gt_img_name)
        our_imgs = []
        for folder in img_folders:
            our_imgs.append(load_image(folder, ours_img_name))

        ## find downsampled image
        tmp = list(filter(lambda x: x.startswith("CuNeRFx") , img_folders[0].split('/')))[0]
        scale = int(tmp.split('x')[-1])
        down_img = gt_img[::scale, ::scale]
        tmp = np.zeros(gt_img.shape, dtype=gt_img.dtype)
        tmp[:down_img.shape[0], :down_img.shape[1]] = down_img
        down_img = tmp

        # Calculate PSNR
        psnrs = []
        for i, folder in enumerate(img_folders):
            psnrs.append(peak_signal_noise_ratio(gt_img, our_imgs[i]))
        total_psnrs.append(psnrs)

        # Calculate ssim
        ssims = []
        for i, folder in enumerate(img_folders):
            # ssims.append(structural_similarity(gt_img, our_imgs[i], data_range=our_imgs[i].max() - our_imgs[i].min()))
            ssims.append(structural_similarity(gt_img, our_imgs[i], win_size=11, data_range=255))
        total_ssims.append(ssims)

        # inference timing
        timing = []
        for i, folder in enumerate(folders):
            timing_file = os.path.join(folder, 'timing.txt')
            with open(timing_file, 'r') as f:
                tf = f.readlines()
                t = [float(t) for t in tf]
                t = sum(t) / len(t) if len(t) > 0 else 0
                timing.append(t)

        # Add PSNR text to images
        our_imgs_texts = []
        for i, img in enumerate(our_imgs):
            a = add_text_to_image(img.copy(), f'PSNR: {psnrs[i]:.2f}', position=(10, 30))
            a = add_text_to_image(a, f'SSIM: {ssims[i]:.2f}', position=(10, 60))
            b = add_text_to_image(a, f'Timing: {timing[i]:.2f}', position=(10, 90))
            c = add_text_to_image(b, f'{des[i]}', position=(10, 120))
            our_imgs_texts.append(c)

        # Concatenate images horizontally
        combined_image = np.hstack([gt_img, down_img] + our_imgs_texts)

        # Show the combined image
        if not skip:
            window_name = f"Comparison: {base_name}"
            cv2.imshow(window_name, combined_image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

        if key == ord('q'):
            skip = True

    # Print the average SSIM values
    print(f"Average SSIM values: {np.mean(total_ssims, axis=0)}")
    print(f"Average PSNR values: {np.mean(total_psnrs, axis=0)}")

if __name__ == "__main__":
    folder1 = "/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx2/case_00004_mgpu" 
    ds1 = "mgpu"
    
    folder2 = "/home/simtech/Qiming/CuNeRF-gt/save/CuNeRFx2/case_00004" 
    ds2 = "original CuNeRF"
    
    folder7 = "/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx2/case_00004_ngp_single_need_post"
    ds7 = "ngp no rand eval"

    folder8 = '/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx1/case_00004_ngp_rand_eval'
    ds8 = "eval w rand & s=1" ## with randomness during sampling in evaluation. scale set to 1 to prevent bad in between frame problem

    folder9 = '/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx1/case_00004_post-smooth'
    ds9 = "eval w rand & s=1 & post smooth" ## with randomness during sampling in evaluation. scale set to 1 to prevent bad in between frame problem

    folder10 = "/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx2/case_00004_ngp_rand_eval"
    ds10 = "ngp rand eval" ## scale = 2




    # folder3 = "/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx2/case_00000_mgpu_comparison/eval"
    folder3 = "/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx2/case_00000_mgpu/"
    ds3 = "mgpu"
    
    # folder4 = '/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx2/case_00000_ngp_coarse_and_fine/eval'
    # folder5 = '/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx2/case_00000_ngp_single/eval'
    folder4 = '/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx2/case_00000_ngp_single_need_post'
    ds4 = "ngp no rand eval"
    
    folder6 = '/home/simtech/Qiming/CuNeRF-gt/save/CuNeRFx2/case_00000'
    ds6 = "original CuNeRF"

    folder11 = "/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx2/case_00000_ngp_rand_eval"
    ds11 = 'eval rand' ## scale = 2

    # folder12 = "/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx1/case_00000_ngp_rand_eval"
    # ds12 = 'eval rand & s=1' ## scale = 1
    ## somehow the training is not working and gives black imaes while works for case 00004.






    folder13 = "/home/simtech/Qiming/CuNeRF-mgpu/save/CuNeRFx1/case_00032"
    ds13 = "ngp eval rand"

    
    
    folder14 = "/home/simtech/Qiming/CuNeRF-mgpu/save-1/CuNeRFx2/seed_26_ngp/case_00089"
    ds14 = "ngp" ## evaluation rand, scale = 2, seed = 2, layer = 3
    
    folder15 = "/home/simtech/Qiming/CuNeRF-mgpu/save_mgpu/CuNeRFx2/case_00089"
    ds15 = "mgpu" 


    case_00004 = [folder1, folder2, folder7, folder8, folder10]
    case_00004_des = [ds1, ds2, ds7, ds8, ds10]
    
    case_00000 = [folder3, folder6, folder4, folder11]
    case_00000_des = [ds3, ds6, ds4, ds11]

    case_00032 = [folder13]
    case_00032_des = [ds13]

    case_00089 = [folder14, folder15]
    case_00089_des = [ds14, ds15]
    
    # compare_images(case_00000, case_00000_des) ## all too white post-processing
    # compare_images(case_00004, case_00004_des) ## black and white -> intpolation

    # compare_images(case_00032, case_00032_des) 

    compare_images(case_00089, case_00089_des) 
