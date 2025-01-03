import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from TriAttNet import *
from testing_load import *
from evaluation import *
from utils import *
from load_dataset import *

"""
The purpose of this file is used for testing model with different kinds of images from different dataset.
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    normalized_image = 2 * normalized_image - 1
    return normalized_image

def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    return model

if __name__ == "__main__":
    
    model = TriAttNet()
    model_path = 'model/model.pth'
    model = load_model(model_path, model).to(device)

    # test_loader = processing_dataset(type='data/DIV2K_valid_HR')
    test_loader = create_dataloader('testing/Set5/scale_2_HR','testing/Set5/scale_2_LR',(512,512),(256,256))
    total_psnr,total_ssim=0,0
    maximum_psnr=0
    with torch.no_grad():  
        for i,(hr, lr) in enumerate(test_loader):

            inputs, labels = lr.to(device), hr.to(device)
            outputs = model(inputs)
            print(outputs.shape)
            psnr_value = calculate_psnr(outputs, labels)
            ssim_value = calculate_ssim(outputs, labels)
            
            total_psnr += psnr_value
            total_ssim += ssim_value
            maximum_psnr = max(maximum_psnr,psnr_value)
            # imshow(lr_image=lr,hr_image=hr,epoch=i,title=f"Epoch {i} - Training",generated_image=outputs,save_dir='Testing_4')
            
    average_psnr = total_psnr / len(test_loader)
    average_ssim = total_ssim / len(test_loader)

    # Print result
    print(f'Average PSNR: {average_psnr:.2f} dB')
    print(f'Average SSIM: {average_ssim:.4f}')





#Scale 2
# SET 5
# Average PSNR: 29.70 dB
# Average SSIM: 0.8621
# SET 14
# Average PSNR: 28.38 dB
# Average SSIM: 0.8421

#Scale 3
#SET 5
# Average PSNR: 28.13 dB
# Average SSIM: 0.8118
# SET 14
# Average PSNR: 25.97 dB
# Average SSIM: 0.7396

# Scale 4
# SET 5:
# Average PSNR: 26.44 dB
# Average SSIM: 0.7535
# SET 14:
# Average PSNR: 28.38 dB
# Average SSIM: 0.8421
