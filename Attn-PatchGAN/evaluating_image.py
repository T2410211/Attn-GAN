import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io, transform
from tqdm import tqdm
import pandas as pd

def compute_mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

def compute_psnr(imageA, imageB):
    mse = compute_mse(imageA, imageB)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(imageA, imageB):
    return ssim(imageA, imageB, data_range=imageB.max() - imageB.min(), channel_axis=-1)

def compute_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def compute_contrast(image):
    return np.std(image)

def compute_snr(imageA, imageB):
    signal_power = np.mean(imageB ** 2)
    noise_power = np.mean((imageB - imageA) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def process_images(gt_folder, gen_folder):
    gt_images = sorted(os.listdir(gt_folder))
    gen_images = sorted(os.listdir(gen_folder))
    
    results = []
    mse_vals, psnr_vals, ssim_vals, sharpness_gt_vals, sharpness_gen_vals, contrast_gt_vals, contrast_gen_vals, snr_vals = [], [], [], [], [], [], [], []
    
    for gt_img_name, gen_img_name in tqdm(zip(gt_images, gen_images), total=len(gt_images)):
        gt_img_path = os.path.join(gt_folder, gt_img_name)
        gen_img_path = os.path.join(gen_folder, gen_img_name)
        
        gt_img = io.imread(gt_img_path)
        if gt_img.ndim == 3:
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)


        gen_img = io.imread(gen_img_path)

        # Ensure generated image is in 8-bit format
        if gen_img.dtype != np.uint8:
            gen_img = (gen_img * 255).astype(np.uint8)

        

        # Resize GT image to 128x128 to match generated image size
        gt_img_resized = transform.resize(gt_img, (256, 256), anti_aliasing=True, mode='reflect')
        gt_img_resized = (gt_img_resized * 255).astype(np.uint8)  # Convert back to 8-bit

        # gt_img_resized = gt_img  # No resizing — keep original resolution
        if gt_img_resized.shape != gen_img.shape:
            print(f"Skipping {gen_img_name} due to shape mismatch after resizing.")
            continue
        
        mse_val = compute_mse(gt_img_resized, gen_img)
        psnr_val = compute_psnr(gt_img_resized, gen_img)
        ssim_val = compute_ssim(gt_img_resized, gen_img)
        sharpness_gt_val = compute_sharpness(gt_img_resized)
        sharpness_gen_val = compute_sharpness(gen_img)
        contrast_gt_val = compute_contrast(gt_img_resized)
        contrast_gen_val = compute_contrast(gen_img)
        snr_val = compute_snr(gt_img_resized, gen_img)
        
        mse_vals.append(mse_val)
        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        sharpness_gt_vals.append(sharpness_gt_val)
        sharpness_gen_vals.append(sharpness_gen_val)
        contrast_gt_vals.append(contrast_gt_val)
        contrast_gen_vals.append(contrast_gen_val)
        snr_vals.append(snr_val)
        
        results.append([gen_img_name, mse_val, psnr_val, ssim_val, sharpness_gt_val, sharpness_gen_val, contrast_gt_val, contrast_gen_val, snr_val])
    
    avg_mse = np.mean(mse_vals)
    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    avg_sharpness_gt = np.mean(sharpness_gt_vals)
    avg_sharpness_gen = np.mean(sharpness_gen_vals)
    avg_contrast_gt = np.mean(contrast_gt_vals)
    avg_contrast_gen = np.mean(contrast_gen_vals)
    avg_snr = np.mean(snr_vals)
    
    results.append(["Average", avg_mse, avg_psnr, avg_ssim, avg_sharpness_gt, avg_sharpness_gen, avg_contrast_gt, avg_contrast_gen, avg_snr])
    
    return results

# Example usage
gt_folder = r"HDCT"
gen_folder = r"generatingImages_DenseNet121_classification"

results = process_images(gt_folder, gen_folder)

results_df = pd.DataFrame(results, columns=["Image", "MSE", "PSNR", "SSIM", "Sharpness_GT", "Sharpness_Gen", "Contrast_GT", "Contrast_Gen", "SNR"])

# Print results and save as CSV
print(results_df)
results_df.to_csv("withPrePro_CombinedTrained_GeneratedImages.csv", index=False)
print("Results saved to image_metrics.csv")