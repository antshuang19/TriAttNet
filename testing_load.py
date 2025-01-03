import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_and_resize_image(image_path, target_size):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    return resized_image

def transform_image(image):

    transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor_image = transform(image)
    return tensor_image

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, pixel_hr, pixel_lr):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.pixel_hr = pixel_hr
        self.pixel_lr = pixel_lr
        self.hr_images = sorted(os.listdir(hr_folder))
        self.lr_images = sorted(os.listdir(lr_folder))
        assert len(self.hr_images) == len(self.lr_images), "Mismatch in number of HR and LR images."

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_img_path = os.path.join(self.hr_folder, self.hr_images[idx])
        lr_img_path = os.path.join(self.lr_folder, self.lr_images[idx])
        hr_resized = load_and_resize_image(hr_img_path, self.pixel_hr)
        lr_resized = load_and_resize_image(lr_img_path, self.pixel_lr)
        hr_tensor = transform_image(hr_resized)
        lr_tensor = transform_image(lr_resized)

        return hr_tensor, lr_tensor

# Function to create DataLoader
def create_dataloader(hr_folder, lr_folder, pixel_hr, pixel_lr, batch_size=1, shuffle=True):
    # Create the dataset
    dataset = SuperResolutionDataset(hr_folder, lr_folder, pixel_hr, pixel_lr)
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

