# ================================================================
# Deep Learning and its Applications to Signal and Image Processing
# FINAL PROJECT - DIV2K Super Resolution Challenge
#
# Course: 361.2.1120
# Authors:   Or Shraga , Gal Apple 
# ================================================================
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import piq
import cv2
import numpy as np
from tqdm import tqdm
from utils import ycbcr_to_rgb_tensor
import matplotlib.pyplot as plt

def evaluate_model_ycbcr(model, dataset, device='cuda', batch_size=4, image_size=299):
    model.eval()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    psnr_list, ssim_list = [], []
    sr_rgb_imgs, hr_rgb_imgs = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            if 'cb_sr' not in batch or 'cb_hr' not in batch:
                  raise ValueError("Dataset must include 'cb_sr', 'cr_sr', 'cb_hr', 'cr_hr' keys. Set add_cbcr=True.")
            if 'cr_sr' not in batch or 'cr_hr' not in batch:
                  raise ValueError("Dataset must include 'cb_sr', 'cr_sr', 'cb_hr', 'cr_hr' keys. Set add_cbcr=True.")

            cb_sr = batch['cb_sr']
            cr_sr = batch['cr_sr']
            cb_hr = batch['cb_hr']
            cr_hr = batch['cr_hr']

            # Y-channel SR output
            sr = model(lr).clamp(0.0, 1.0)

            # PSNR & SSIM (Y only)
            psnr = piq.psnr(sr, hr, reduction='none').mean().item()
            ssim = piq.ssim(sr, hr, reduction='none').mean().item()
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            # RGB Reconstruction for FID
            for i in range(sr.size(0)):
                sr_rgb = ycbcr_to_rgb_tensor(sr[i], cb_sr[i], cr_sr[i])
                hr_rgb = ycbcr_to_rgb_tensor(hr[i], cb_hr[i], cr_hr[i])

                # Resize to 299x299 for FID
                sr_rgb_resized = cv2.resize(sr_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                hr_rgb_resized = cv2.resize(hr_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                # Convert to tensors: (C, H, W)
                sr_rgb_imgs.append(torch.tensor(sr_rgb_resized).permute(2, 0, 1))
                hr_rgb_imgs.append(torch.tensor(hr_rgb_resized).permute(2, 0, 1))

    # Stack and convert to DataLoaders for FID
    sr_dataset = [{'images': img} for img in torch.stack(sr_rgb_imgs)]
    hr_dataset = [{'images': img} for img in torch.stack(hr_rgb_imgs)]

    sr_dl = DataLoader(sr_dataset, batch_size=batch_size)
    hr_dl = DataLoader(hr_dataset, batch_size=batch_size)
    
    print(torch.cuda.is_available())

    fid = piq.FID().to("cpu")
    fid_score = fid(fid.compute_feats(sr_dl), fid.compute_feats(hr_dl)).item()

    # Return
    results = {
        "psnr": sum(psnr_list) / len(psnr_list),
        "ssim": sum(ssim_list) / len(ssim_list),
        "fid": fid_score,
    }

    print("\n📊 Final Evaluation Results:")
    print(f"  PSNR: {results['psnr']:.4f}")
    print(f"  SSIM: {results['ssim']:.4f}")
    print(f"  FID : {results['fid']:.4f}")

    return results

def compare_models_examples(model1, model2, dataset, device='cuda', threshold_good=30, max_images=200):
    """
    Compare two models and plot required four examples.

    Args:
        model1: Residual model (baseline).
        model2: Ablation model (no_global).
        dataset: Test dataset with __getitem__ returning {'lr': .., 'hr': ..}
        device: cuda or cpu.
        threshold_good: PSNR threshold to define 'good performance'.
        max_images: maximum number of images to process
    """

    model1.eval().to(device)
    model2.eval().to(device)

    categories = {'both_good': None, 'both_bad': None, 'model1_good': None, 'model2_good': None}

    for idx in range(min(len(dataset), max_images)):
        if idx % 20 == 0:
            print(f"Processing image {idx}/{min(len(dataset), max_images)}...")

        sample = dataset[idx]
        lr = sample['lr'].unsqueeze(0).to(device)  # add batch dim
        hr = sample['hr'].to(device)

        with torch.no_grad():
            sr1 = model1(lr).clamp(0, 1).cpu().squeeze(0)
            sr2 = model2(lr).clamp(0, 1).cpu().squeeze(0)

        psnr1 = piq.psnr(sr1.unsqueeze(0), hr.cpu().unsqueeze(0), data_range=1.0).item()
        psnr2 = piq.psnr(sr2.unsqueeze(0), hr.cpu().unsqueeze(0), data_range=1.0).item()


        # Category logic
        if psnr1 > threshold_good and psnr2 > threshold_good and categories['both_good'] is None:
            categories['both_good'] = (lr.cpu(), sr1, sr2, hr.cpu(), psnr1, psnr2)
        elif psnr1 < threshold_good and psnr2 < threshold_good and categories['both_bad'] is None:
            categories['both_bad'] = (lr.cpu(), sr1, sr2, hr.cpu(), psnr1, psnr2)
        elif psnr1 > threshold_good and psnr2 < threshold_good and categories['model1_good'] is None:
            categories['model1_good'] = (lr.cpu(), sr1, sr2, hr.cpu(), psnr1, psnr2)
        elif psnr2 > threshold_good and psnr1 < threshold_good and categories['model2_good'] is None:
            categories['model2_good'] = (lr.cpu(), sr1, sr2, hr.cpu(), psnr1, psnr2)

    print("✅ Finished scanning dataset.")

    # Plotting
    for key, data in categories.items():
        if data is None:
            print(f"⚠️ No example found for {key}.")
            continue

        lr, sr1, sr2, hr, psnr1, psnr2 = data

        plt.figure(figsize=(12,4))
        plt.suptitle(f"{key.replace('_', ' ').title()} (PSNR1: {psnr1:.2f}, PSNR2: {psnr2:.2f})")

        plt.subplot(1,4,1)
        plt.imshow(lr.squeeze().numpy(), cmap='gray')
        plt.title('LR Input')
        plt.axis('off')

        plt.subplot(1,4,2)
        plt.imshow(sr1.squeeze().numpy(), cmap='gray')
        plt.title('Residual SR')
        plt.axis('off')

        plt.subplot(1,4,3)
        plt.imshow(sr2.squeeze().numpy(), cmap='gray')
        plt.title('Ablation SR')
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.imshow(hr.squeeze().numpy(), cmap='gray')
        plt.title('HR Ground Truth')
        plt.axis('off')

        plt.show()