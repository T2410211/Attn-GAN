import torch
import os
import gc
from tqdm import tqdm
from PIL import Image
from data.dataloader import transform
from utils.Utils import save_comparison, save_sr_only
from evaluation.metrics import evaluate_sr_image


class EarlyStopping:
    """Stops training if the validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def save_checkpoint(model, optimizer, epoch, loss, filename="best_checkpoint.pth"):
    """Saves the best model if validation loss improves."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, filename)
    print(f"✅ Best model saved at epoch {epoch+1} with loss: {loss:.4f}")


def load_checkpoint(generator, optimizer_G, discriminator, optimizer_D, filename="last_checkpoint.pth", device="cuda"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)  # Ensure loading to the correct device
        generator.load_state_dict(checkpoint["generator_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

        # Ensure optimizer states are moved to the correct device (same as model)
        for state in optimizer_G.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    v.data = v.data.to(device)

        for state in optimizer_D.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    v.data = v.data.to(device)

        start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    else:
        print("⚠️ No checkpoint found, starting fresh.")
        start_epoch = 0
    return start_epoch


def train(generator, discriminator, dataloader, num_epochs, optimizer_G, optimizer_D,
          criterion_content, criterion_adv, criterion_ssim, criterion_perceptual, device,
          sample_ldct_img, sample_hdct_img,
          patience=5, start_epoch=0):

    generator.to(device)
    discriminator.to(device)
    criterion_perceptual.to(device)

    early_stopping = EarlyStopping(patience=patience)
    best_loss = float("inf")
    torch.cuda.empty_cache()
    gc.collect()

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        discriminator.train()
        epoch_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for i, (low_img, high_img) in enumerate(progress_bar):
            low_img, high_img = low_img.to(device), high_img.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            sr_image = generator(low_img)

            content_loss = criterion_content(sr_image, high_img)
            ssim_loss = criterion_ssim(sr_image, high_img)
            sr_pred = discriminator(sr_image)
            # adv_loss = criterion_adv(sr_pred, torch.ones_like(sr_pred))
            real_labels = torch.empty_like(sr_pred).uniform_(0.8, 1.0)  # Label smoothing for real
            adv_loss = criterion_adv(sr_pred, real_labels)
            perceptual_loss = criterion_perceptual(sr_image, high_img)

            # g_loss = (
            #     content_loss
            #     + 0.10 * adv_loss
            #     + 0.05 * ssim_loss     
            #     + 0.15 * perceptual_loss 
            # )
            g_loss = (
                content_loss
                + 0.12 * adv_loss
                + 0.05 * ssim_loss      
                + 0.16 * perceptual_loss  
            )

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()            

            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(high_img)
            fake_output = discriminator(sr_image.detach())
            # real_loss = criterion_adv(real_output, torch.ones_like(real_output))
            # fake_loss = criterion_adv(fake_output, torch.zeros_like(fake_output))
            real_labels = torch.empty_like(real_output).uniform_(0.8, 1.0)  # Smoothed real
            fake_labels = torch.empty_like(fake_output).uniform_(0.0, 0.2)  # Smoothed fake
            real_loss = criterion_adv(real_output, real_labels)
            fake_loss = criterion_adv(fake_output, fake_labels)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()

            epoch_loss += g_loss.item()
            progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})

        epoch_loss /= len(dataloader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(generator, optimizer_G, epoch, best_loss)

        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
            "loss": epoch_loss
        }, "last_checkpoint.pth")
        # print(f"💾 Checkpoint saved at epoch {epoch+1}")

        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

        torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")
        save_comparison(generator, sample_ldct_img, sample_hdct_img, epoch, device, save_dir="comparisons")
        save_sr_only(generator, sample_ldct_img, epoch, device, save_dir="sr_only")

        # Define image paths
        sr_path = os.path.join("sr_only", f"sr_epoch_{epoch+1}.png")
        gt_path = r"C:\Users\T2410211\Desktop\train_test_split\test\hdct\Wr02_01.bmp"
        ldct_path = r"C:\Users\T2410211\Desktop\train_test_split\test\ldct\Wr02_01.bmp"

        # Evaluate
        evaluate_sr_image(epoch, sr_path, gt_path, ldct_path, output_csv="epoch_metrics.csv")

