import torch
from torch.utils.data import DataLoader
from PIL import Image
from models.generator import Generator
from models.discriminator import Discriminator
from losses.losses import ContentLoss, AdversarialLoss, SSIMLoss, PerceptualLoss
from data.dataloader import PairedCTDataset, transform
from engine.training import train, load_checkpoint


# Load one fixed LDCT image for visualization
sample_ldct_path = r"train_test_split\test\ldct\Wr02_01.bmp"
sample_hdct_path = r"train_test_split\test\hdct\Wr02_01.bmp"  
sample_ldct_img = Image.open(sample_ldct_path).convert("L")
sample_hdct_img = Image.open(sample_hdct_path).convert("L")
sample_ldct_img = transform(sample_ldct_img).unsqueeze(0)  # (1, 1, H, W)
sample_hdct_img = transform(sample_hdct_img).unsqueeze(0)


if __name__ == "__main__":

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate models
    generator = Generator()
    discriminator = Discriminator()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Losses
    criterion_content = ContentLoss()
    criterion_adv = AdversarialLoss()
    criterion_ssim = SSIMLoss()
    criterion_perceptual = PerceptualLoss()

    # Dataset paths
    low_dose_dir = r"train_test_split\train\ldct"
    high_dose_dir = r"train_test_split\train\hdct"

    # Dataset and DataLoader
    dataset = PairedCTDataset(low_dose_dir, high_dose_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Load checkpoint if available
    start_epoch = load_checkpoint(generator, optimizer_G, discriminator, optimizer_D)

    # Train
    train(generator, discriminator, dataloader, num_epochs=50,
        optimizer_G=optimizer_G, optimizer_D=optimizer_D,
        criterion_content=criterion_content, criterion_adv=criterion_adv,
        criterion_ssim=criterion_ssim, criterion_perceptual=criterion_perceptual,
        device=device, sample_ldct_img=sample_ldct_img, sample_hdct_img=sample_hdct_img,
        patience=50, start_epoch=start_epoch)

