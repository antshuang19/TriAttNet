import os
import numpy as np
import matplotlib.pyplot as plt

def imshow(lr_image, hr_image, generated_image, epoch=None, title=None, save_dir="images_training2"):
    # If the images are in a batch (i.e., shape [batch_size, C, H, W]), take the first image
    lr_image = lr_image[0]  # Take the first image from the batch
    hr_image = hr_image[0]  # Take the first image from the batch
    generated_image = generated_image[0]  # Take the first image from the batch

    # Convert tensors to NumPy arrays and move to CPU if necessary
    lr_image = lr_image.detach().cpu().numpy().transpose((1, 2, 0))  # C, H, W -> H, W, C
    hr_image = hr_image.detach().cpu().numpy().transpose((1, 2, 0))  # C, H, W -> H, W, C
    generated_image = generated_image.detach().cpu().numpy().transpose((1, 2, 0))  # C, H, W -> H, W, C

    # Ensure pixel values are in the [0, 1] range
    lr_image = np.clip(lr_image, 0, 1)
    hr_image = np.clip(hr_image, 0, 1)
    generated_image = np.clip(generated_image, 0, 1)

    # Create a figure with 3 subplots (for LR, HR, and generated images)
    plt.figure(figsize=(15, 5))

    # Display the low-resolution image
    plt.subplot(1, 3, 1)
    plt.imshow(lr_image)
    plt.title('Low Resolution')
    plt.axis('off')

    # Display the high-resolution ground truth image
    plt.subplot(1, 3, 2)
    plt.imshow(hr_image)
    plt.title('High Resolution (HR)')
    plt.axis('off')

    # Display the generated image
    plt.subplot(1, 3, 3)
    plt.imshow(generated_image)
    plt.title('Generated Image (SR)')
    plt.axis('off')

    # Optionally, add a title to the entire figure (e.g., for epoch info)
    if title:
        plt.suptitle(title, fontsize=16)

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct the filename based on the epoch
    filename = f"epoch_{epoch+1}.png" if epoch is not None else "generated_image.png"
    
    # Save the figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
