import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import os


# ramp_scheduler.py
def ramp_weight(epoch: int, ramp_epoch: int, max_weight: float, min_weight: float = 0) -> float:
    """
    Linearly increase weight from 0 to max_weight until ramp_epoch.
    After ramp_epoch, weight stays constant at max_weight.
    """
    progress = min(float(epoch) / ramp_epoch, 1.0)
    return min_weight + (max_weight - min_weight) * progress


# # Test ramping for 25 epochs
# ramp_epoch = 15
# max_weight = 0.10

# print("Epoch | Weight")
# print("------|--------")
# for epoch in range(25):
#     weight = ramp_weight(epoch, ramp_epoch, max_weight)
#     print(f"{epoch:5d} | {weight:.4f}")


def save_comparison(generator, sample_ldct, sample_hdct, epoch, device, save_dir="my_generated_images"):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()

    with torch.no_grad():
        sample_ldct = sample_ldct.to(device)
        sample_hdct = sample_hdct.to(device)

        sr = generator(sample_ldct)
        sr = torch.clamp((sr + 1) / 2.0, 0.0, 1.0)
        ld = torch.clamp((sample_ldct + 1) / 2.0, 0.0, 1.0)
        gt = torch.clamp((sample_hdct + 1) / 2.0, 0.0, 1.0)

        comparison = torch.cat([ld, sr, gt], dim=3)  # Concatenate horizontally
        save_path = os.path.join(save_dir, f"compare_epoch_{epoch+1}.png")
        save_image(comparison, save_path)

    # ---- Add labels using PIL ----
    img = Image.open(save_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    W, H = img.size
    third = W // 3

    draw.text((third * 0 + 10, 10), "LDCT", fill="white", font=font)
    draw.text((third * 1 + 10, 10), "SR",   fill="white", font=font)
    draw.text((third * 2 + 10, 10), "GT",   fill="white", font=font)

    img.save(save_path)


def save_sr_only(generator, sample_ldct, epoch, device, save_dir="sr_only"):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()

    with torch.no_grad():
        sample_ldct = sample_ldct.to(device)
        sr = generator(sample_ldct)
        sr = torch.clamp((sr + 1) / 2.0, 0.0, 1.0)

    save_image(sr, os.path.join(save_dir, f"sr_epoch_{epoch+1}.png"))