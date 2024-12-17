import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr

def process_cases():
    # Find all cases in the ./cor/ directory
    cor_case_paths = sorted(glob.glob('./save-cor/CuNeRFx2/case_*'))

    for cor_case_path in cor_case_paths:
        case_name = os.path.basename(cor_case_path)
        eval_dir = os.path.join(cor_case_path, 'eval')
        if not os.path.exists(eval_dir):
            continue  # Skip if no eval folder

        # Corresponding sag directory
        sag_eval_dir = eval_dir.replace('./save-cor/', './save-sag/')
        if not os.path.exists(sag_eval_dir):
            continue  # Skip if no corresponding sag folder

        axi_eval_dir = eval_dir.replace('./save-cor/', './save-vcube-size/')
        if not os.path.exists(sag_eval_dir):
            continue  # Skip if no corresponding sag folder

        # Read and stack cor images
        cor_image_files = sorted(glob.glob(os.path.join(eval_dir, '*_ours.png')))
        cor_images = []
        for f in cor_image_files:
            img = np.array(Image.open(f))
            cor_images.append(img)
        cor_volume = np.stack(cor_images, axis=0)  # Stack along last axis
        # Transpose from (y, x, z) to (z, y, x)
        cor_volume = np.transpose(cor_volume, (2, 0, 1))

        # Read and stack sag images
        sag_image_files = sorted(glob.glob(os.path.join(sag_eval_dir, '*_ours.png')))
        sag_images = []
        for f in sag_image_files:
            img = np.array(Image.open(f))
            sag_images.append(img)
        sag_volume = np.stack(sag_images, axis=0)  # Stack along last axis
        # Transpose from (x, y, z) to (z, y, x)
        sag_volume = np.transpose(sag_volume, (2, 0, 1))

        # Read and stack cor images
        axi_image_files = sorted(glob.glob(os.path.join(axi_eval_dir, '*_ours.png')))
        axi_images = []
        for f in axi_image_files:
            img = np.array(Image.open(f))
            axi_images.append(img)
        axi_volume = np.stack(axi_images, axis=0)  # Stack along last axis
        # already (z, y, x)
        
        min_z = min(cor_volume.shape[0], sag_volume.shape[0], axi_volume.shape[0])

        # Compare PSNR between each xy frame
        psnr_values = []
        psnr_values_1 = []
        for z in range(min_z):
            cor_img = cor_volume[z, :, :]
            sag_img = sag_volume[z, :, :]
            axi_img = axi_volume[z, :, :]

            # Compute PSNR
            cmax = np.max(cor_img)
            smax = np.max(sag_img)
            cmin = np.min(cor_img)
            smin = np.min(sag_img)
            amax = np.max(axi_img)
            amin = np.min(axi_img)
            psnr_value = psnr(cor_img, sag_img, data_range=max(cmax - cmin, smax - smin, amax - amin))
            psnr_value_1 = psnr(cor_img, axi_img, data_range=max(cmax - cmin, smax - smin, amax - amin))
            psnr_values.append(psnr_value)
            psnr_values_1.append(psnr_value_1)

            # Compute binary difference image
            diff_img = np.abs(cor_img.astype(np.float32) - sag_img.astype(np.float32))
            threshold = 10  # You can adjust the threshold as needed
            diff_img_bin = (diff_img > 0).astype(np.uint8) * 255
            diff_pil_img = Image.fromarray(diff_img_bin)

            # Save images side by side with PSNR annotation
            cor_pil_img = Image.fromarray(cor_img)
            sag_pil_img = Image.fromarray(sag_img)
            axi_pil_img = Image.fromarray(axi_img)
            combined_width = cor_pil_img.width + sag_pil_img.width + diff_pil_img.width
            combined_height = max(cor_pil_img.height, sag_pil_img.height)
            combined_img = Image.new('RGB', (combined_width, combined_height))
            combined_img.paste(cor_pil_img.convert('RGB'), (0, 0))
            combined_img.paste(sag_pil_img.convert('RGB'), (cor_pil_img.width, 0))
            # combined_img.paste(diff_pil_img.convert('RGB'), (cor_pil_img.width + sag_pil_img.width, 0))
            combined_img.paste(axi_pil_img.convert('RGB'), (cor_pil_img.width + sag_pil_img.width, 0))

            # Annotate PSNR value
            draw = ImageDraw.Draw(combined_img)
            psnr_text = f"PSNR: {psnr_value:.2f}"
            psnr_text_1 = f"PSNR: {psnr_value_1:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                font = ImageFont.load_default()
            draw.text((cor_pil_img.width+ 10, 10), psnr_text, font=font, fill='white')
            draw.text((cor_pil_img.width+ sag_pil_img.width +10, 10), psnr_text_1, font=font, fill='white')

            # Save the combined image
            output_dir = os.path.join('./compare_cor_sag', case_name, 'eval')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{z:03d}.png')
            combined_img.save(output_file)

        # Print average PSNR
        avg_psnr = np.mean(psnr_values)
        print(f"Case {case_name}: Average PSNR = {avg_psnr:.2f}")

if __name__ == "__main__":
    process_cases()
