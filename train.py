import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # For progress bar
from TriAttNet import TriAttNet
from load_dataset import *
from evaluation import *
from torch.optim.lr_scheduler import CosineAnnealingLR  # Import scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TriAttNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

train_dataloader = processing_dataset()
test_dataloader = processing_dataset('data/DIV2K_valid_HR')
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
# Training loop
num_epochs = 100  # You can adjust this based on your needs
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for batch_idx, (lr_images, hr_images) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)
        sr_images = model(lr_images)
        loss = criterion(sr_images, hr_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    # Validation
    model.eval()
    val_psnr, val_ssim = [], []
    with torch.no_grad():
        for batch_idx, (lr_images, hr_images) in enumerate(tqdm(test_dataloader, desc=f"Validating Epoch {epoch+1}/{num_epochs}")):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            sr_images = model(lr_images)
            psnr_value = calculate_psnr(hr_images, sr_images)
            ssim_value = calculate_ssim(hr_images, sr_images)

            val_psnr.append(psnr_value)
            val_ssim.append(ssim_value)
                    
    avg_psnr = sum(val_psnr) / len(val_psnr)
    avg_ssim = sum(val_ssim) / len(val_ssim)
    print(f"Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    with open("epoch_log_train.txt", "a") as log_file:
        log_file.write(f"{epoch+1}\t{epoch_loss :.4f}\t{avg_psnr:.4f}\t{avg_ssim:.4f}\n")
    scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

print("Training complete!")