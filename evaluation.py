from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
import numpy as np

def calculate_psnr(hr_image, sr_image):
    hr_image = (hr_image + 1) / 2  
    sr_image = (sr_image + 1) / 2 

    mse = F.mse_loss(sr_image, hr_image)
    if mse == 0:
        return 100  
    max_pixel = 1.0 
    psnr = 10 * torch.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_ssim(hr_image, sr_image):
    hr_image = (hr_image + 1) / 2 
    sr_image = (sr_image + 1) / 2 
    hr_image = hr_image.detach().cpu().numpy().transpose(0, 2, 3, 1) 
    sr_image = sr_image.detach().cpu().numpy().transpose(0, 2, 3, 1)

    ssim_values = []
    for i in range(hr_image.shape[0]):
        ssim_value = ssim(hr_image[i], sr_image[i], data_range=1, multichannel=True)
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)
