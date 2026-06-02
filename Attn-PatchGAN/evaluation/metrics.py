import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage import io
from datetime import datetime


def compute_mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

def compute_psnr(imageA, imageB):
    mse = compute_mse(imageA, imageB)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(imageA, imageB):
    return ssim(imageA, imageB, data_range=255, channel_axis=None)

def compute_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def compute_contrast(image):
    return np.std(image)

def compute_snr(reference, comparison):
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - comparison) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def evaluate_sr_image(epoch, sr_image_path, gt_image_path, ldct_image_path, output_csv="epoch_metrics.csv"):
    """
    Evaluate a single SR image against the GT and LDCT image.
    Appends one row per epoch to a growing CSV file.
    """
    if not os.path.exists(sr_image_path) or not os.path.exists(gt_image_path) or not os.path.exists(ldct_image_path):
        print(f"❌ Missing input image(s) for evaluation at epoch {epoch+1}.")
        return

    sr_img = io.imread(sr_image_path)
    gt_img = io.imread(gt_image_path)
    ldct_img = io.imread(ldct_image_path)

    # Convert to grayscale if needed
    if sr_img.ndim == 3:
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY)
    if gt_img.ndim == 3:
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
    if ldct_img.ndim == 3:
        ldct_img = cv2.cvtColor(ldct_img, cv2.COLOR_RGB2GRAY)

    # Convert to uint8 if needed
    if sr_img.dtype != np.uint8:
        sr_img = (sr_img * 255).astype(np.uint8)
    if gt_img.dtype != np.uint8:
        gt_img = (gt_img * 255).astype(np.uint8)
    if ldct_img.dtype != np.uint8:
        ldct_img = (ldct_img * 255).astype(np.uint8)

    # ✅ Resize GT to match SR image
    gt_img_resized = cv2.resize(gt_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_AREA)

    # Metrics
    mse_val = compute_mse(gt_img_resized, sr_img)
    psnr_val = compute_psnr(gt_img_resized, sr_img)
    ssim_val = compute_ssim(gt_img_resized, sr_img)
    sharpness_gt = compute_sharpness(gt_img_resized)
    sharpness_sr = compute_sharpness(sr_img)
    contrast_gt = compute_contrast(gt_img_resized)
    contrast_sr = compute_contrast(sr_img)
    snr_gen = compute_snr(ldct_img, sr_img)
    snr_gt = compute_snr(ldct_img, gt_img_resized)

    # Row to append
    row = {
        "Epoch": epoch + 1,
        "MSE": mse_val,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "Sharpness_GT": sharpness_gt,
        "Sharpness_SR": sharpness_sr,
        "Contrast_GT": contrast_gt,
        "Contrast_SR": contrast_sr,
        "SNR_GT": snr_gt,
        "SNR_SR": snr_gen,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save or append to CSV
    df = pd.DataFrame([row])
    if not os.path.exists(output_csv):
        df.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, mode='a', header=False, index=False)

    print(f"📊 Evaluation metrics saved for epoch {epoch+1}.")
