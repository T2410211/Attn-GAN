import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# ---------------------------
# Dataset for Paired CT Images
# ---------------------------
class PairedCTDataset(Dataset):
    def __init__(self, low_dose_dir, high_dose_dir, transform=None):
        self.low_dose_dir = low_dose_dir
        self.high_dose_dir = high_dose_dir
        self.transform = transform
        self.low_filenames = sorted(os.listdir(low_dose_dir))
        self.high_filenames = sorted(os.listdir(high_dose_dir))

        assert len(self.low_filenames) == len(self.high_filenames), "Mismatch in dataset size"

    def __len__(self):
        return len(self.low_filenames)

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_dose_dir, self.low_filenames[idx])
        high_img_path = os.path.join(self.high_dose_dir, self.high_filenames[idx])

        low_img = Image.open(low_img_path).convert("L")
        high_img = Image.open(high_img_path).convert("L")

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img


# ---------------------------
# Image Transformations
# ---------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
