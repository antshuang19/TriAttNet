from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, dataset_folder, image_size=(256, 256), scale_factor=4, transform=None):
        self.dataset_folder = dataset_folder
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.transform = transform
        self.image_paths = []

        for root, _, files in os.walk(dataset_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Resize the image to the desired high-resolution size (HR)
        hr_image = image.resize(self.image_size, Image.BICUBIC)
        lr_image = hr_image.resize(
            (self.image_size[0] // self.scale_factor, self.image_size[1] // self.scale_factor),
            Image.BICUBIC
        )

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        else:
            hr_image = transforms.ToTensor()(hr_image)
            lr_image = transforms.ToTensor()(lr_image)

        return lr_image, hr_image

def processing_dataset(type='data/DIV2K_train_HR' ):
    
    dataset_folder = type  

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Custom normalization
    ])
    
    dataset = ImageDataset(dataset_folder, image_size=(512, 512), scale_factor=4, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    
    return dataloader
