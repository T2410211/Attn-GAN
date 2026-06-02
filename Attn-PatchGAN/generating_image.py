import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from models.generator import Generator
 

# Set paths
test_dir = r"ldct"  # Folder containing test images
output_dir = r"generatingImages_DenseNet121_classification"  # Folder to save super-resolved images
model_path = r"checkpoints\generator_epoch_24.pth"  # Path to the trained model

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load trained generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
# Load the entire checkpoint
checkpoint = torch.load(model_path, map_location=device)

# Extract only the generator's state dictionary
# generator.load_state_dict(checkpoint["model_state_dict"])  # Use the correct key
generator.load_state_dict(checkpoint)


generator.eval()

# Process each test image
for img_name in sorted(os.listdir(test_dir)):
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("L")  # Convert to grayscale
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Convert output tensor back to image
    output_tensor = output_tensor.squeeze(0).cpu().detach()
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)  # Denormalize
    output_image = transforms.ToPILImage()(output_tensor)

    # Save output image
    output_path = os.path.join(output_dir, img_name)
    output_image.save(output_path)

    print(f"Processed {img_name} -> {output_path}")

print("Testing complete. Super-resolved images saved in:", output_dir)