import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from models.generator import Generator

# ---------------------------
# Paths
# ---------------------------
test_dir = r"C:\Users\T2410211\Desktop\train_test_split\test\ldct"
output_dir = r"enhanced_images"
model_path = r"generator_epoch_6.pth"

os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Image Transform (must match training)
# ---------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Match training
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Match training
])

# ---------------------------
# Load Generator Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

# Load generator weights from checkpoint
checkpoint = torch.load(model_path, map_location=device)
if "generator_state_dict" in checkpoint:
    generator.load_state_dict(checkpoint["generator_state_dict"])
    print("✅ Loaded generator weights from checkpoint.")
else:
    raise KeyError("❌ 'generator_state_dict' not found in checkpoint.")

generator.eval()

# ---------------------------
# Generate Enhanced Images
# ---------------------------
for img_name in sorted(os.listdir(test_dir)):
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("L")  # Ensure grayscale input
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Denormalize: [-1,1] → [0,1]
    output_tensor = output_tensor.squeeze(0).cpu()
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)

    # Convert to PIL and save
    output_image = transforms.ToPILImage()(output_tensor)
    output_path = os.path.join(output_dir, img_name)
    output_image.save(output_path)

    print(f"✅ Processed {img_name} → {output_path}")

print("\n🎉 All images processed. Check:", output_dir)