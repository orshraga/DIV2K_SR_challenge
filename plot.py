# ================================================================
# Deep Learning and its Applications to Signal and Image Processing
# FINAL PROJECT - DIV2K Super Resolution Challenge
#
# Course: 361.2.1120
# Authors:   Or Shraga , Gal Apple
# ================================================================

import matplotlib.pyplot as plt
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
import itertools
import cv2
import torch
import math
import matplotlib.pyplot as plt

def show_sample_images(file_list, num=4, title_prefix="Image"):
    """
    Display sample images from a given list.

    Args:
        file_list (list): List of image file paths.
        num (int): Number of images to display.
        title_prefix (str): Prefix for the subplot titles.
    """
    selected = random.sample(file_list, min(num, len(file_list)))
    plt.figure(figsize=(15, 3))
    for i, path in enumerate(selected):
        img = Image.open(path)
        plt.subplot(1, len(selected), i + 1)
        plt.imshow(img)
        plt.title(f"{title_prefix} {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_multiple_training_losses(train_losses_list, val_losses_list):
    plt.figure(figsize=(10, 4))
    for i, (train_losses, val_losses) in enumerate(zip(train_losses_list, val_losses_list)):
        plt.plot(train_losses, label=f'Train Loss Model {i+1}')
        plt.plot(val_losses, label=f'Val Loss Model {i+1}', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss Over Epochs (All Models)")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_ychannel_samples(dataset, num_samples=3):
    """
    Visualize LR and HR Y channel images from a dataset.
    Args:
        dataset: PyTorch dataset with 'lr' and 'hr' keys in each sample.
        num_samples (int): Number of samples to visualize.
    """
    print(f"show ychannel_samples HR and LR from: {dataset}")
    plt.figure(figsize=(10, 4 * num_samples))
    for i in range(num_samples):
        sample = dataset[i]
        lr = sample['lr'].squeeze(0).numpy()  # shape: (H, W)
        hr = sample['hr'].squeeze(0).numpy()

        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(lr, cmap='gray')
        plt.title(f'Sample {i+1} - LR (Y channel)')
        plt.axis('off')

        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(hr, cmap='gray')
        plt.title(f'Sample {i+1} - HR (Y channel)')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_training_loss(train_losses_epoch, val_losses_epoch=None):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses_epoch, label='Training Loss per Epoch')
    if val_losses_epoch is not None:
        plt.plot(val_losses_epoch, label='Validation Loss per Epoch')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

def show_sr_examples(model, dataset, device='cuda', num_examples=4):
    model.eval()
    model.to(device)

    indices = random.sample(range(len(dataset)), num_examples)
    plt.figure(figsize=(15, 6 * num_examples))

    for i, idx in enumerate(indices):
        sample = dataset[idx]

        lr = sample['lr'].unsqueeze(0).to(device)  # [1, 1, H, W]
        hr_y = sample['hr']                        # [1, H, W]

        with torch.no_grad():
            sr_y = model(lr).clamp(0.0, 1.0).squeeze(0).cpu()  # [1, H, W]

        # Convert Y-channel to images
        lr_img = TF.to_pil_image(sample['lr'])
        sr_y_img = TF.to_pil_image(sr_y)
        hr_y_img = TF.to_pil_image(hr_y)

        # If RGB available (CbCr provided), reconstruct SR and HR RGB
        if all(k in sample for k in ['cb_sr', 'cr_sr', 'cb_hr', 'cr_hr']):
            sr_rgb = tensor_to_rgb_image(sr_y, sample['cb_sr'], sample['cr_sr'])
            hr_rgb = tensor_to_rgb_image(hr_y, sample['cb_hr'], sample['cr_hr'])

            plt.subplot(num_examples, 5, i * 5 + 1)
            plt.imshow(lr_img, cmap='gray')
            plt.title('Low-Res Y')
            plt.axis('off')

            plt.subplot(num_examples, 5, i * 5 + 2)
            plt.imshow(sr_y_img, cmap='gray')
            plt.title('SR Y Output')
            plt.axis('off')

            plt.subplot(num_examples, 5, i * 5 + 3)
            plt.imshow(hr_y_img, cmap='gray')
            plt.title('HR Y Ground Truth')
            plt.axis('off')

            plt.subplot(num_examples, 5, i * 5 + 4)
            plt.imshow(sr_rgb)
            plt.title('SR RGB Reconstructed')
            plt.axis('off')

            plt.subplot(num_examples, 5, i * 5 + 5)
            plt.imshow(hr_rgb)
            plt.title('HR RGB Ground Truth')
            plt.axis('off')
        else:
            # fallback to Y-channel only
            plt.subplot(num_examples, 3, i * 3 + 1)
            plt.imshow(lr_img, cmap='gray')
            plt.title('Low-Res Y')
            plt.axis('off')

            plt.subplot(num_examples, 3, i * 3 + 2)
            plt.imshow(sr_y_img, cmap='gray')
            plt.title('SR Y')
            plt.axis('off')

            plt.subplot(num_examples, 3, i * 3 + 3)
            plt.imshow(hr_y_img, cmap='gray')
            plt.title('HR Y')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

def tensor_to_rgb_image(y, cb, cr):
    """Convert Y, Cb, Cr tensors (C,H,W) to RGB np array."""
    y = y.squeeze().cpu().numpy()
    cb = cb.squeeze().cpu().numpy()
    cr = cr.squeeze().cpu().numpy()
    ycbcr = np.stack([y, cb, cr], axis=-1) * 255.0
    ycbcr = ycbcr.astype(np.uint8)
    rgb = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
    return rgb