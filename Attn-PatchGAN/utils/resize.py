from PIL import Image
import os

# Input and output directories
input_dir = r"C:\Users\T2410211\Desktop\train_test_split\test\hdct"
output_dir = r"C:\Users\T2410211\Desktop\train_test_split\test\hdct\resized"
os.makedirs(output_dir, exist_ok=True)

# Loop over all image files
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Open and resize
        with Image.open(input_path) as img:
            resized = img.resize((256, 256))
            resized.save(output_path)

print(f"All images in '{input_dir}' have been resized to 256x256 and saved to '{output_dir}'.")
