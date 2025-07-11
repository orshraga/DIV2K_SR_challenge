# ================================================================
# Deep Learning and its Applications to Signal and Image Processing
# FINAL PROJECT - DIV2K Super Resolution Challenge
#
# Course: 361.2.1120
# Authors:   Or Shraga, Gal Apple
# ================================================================
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

import os

from plot import (
    visualize_ychannel_samples,
    show_sample_images,
    plot_training_loss,
    show_sr_examples,
    plot_multiple_training_losses
)


from model import ESPCNLightning_YChannel
from utils import set_seed, seed_worker
from eval import evaluate_model_ycbcr, compare_models_examples
from data import (
    use_seed_and_split,
    DIV2K_YCbCr_SR_Dataset,
    get_image_files,
    download_kaggle_dataset,
)

def run_model(files, test_files, seed, split_ratio, max_epochs,scale,lr= 1e-4,batch_size_train=16,batch_size_val=4, model_type='base'):
    print(f"\n=== Running Model {model_type.upper()} (Seed {seed}) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset splits
    train_ds, val_ds = use_seed_and_split(
        files,
        scale,
        split_ratio=split_ratio,
        seed=seed,
        dataset_class=DIV2K_YCbCr_SR_Dataset,
        dataset_kwargs={"add_cbcr": False}
    )
    test_ds = DIV2K_YCbCr_SR_Dataset(test_files, scale=scale, mode='test', add_cbcr=True)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

    # DataLoaders
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)

    visualize_ychannel_samples(train_ds, num_samples=3)

    # Model
    model_espcn = ESPCNLightning_YChannel(lr=lr, scale_factor=scale, model_type=model_type)
    print(f"Created model_{model_type}")

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-4, verbose=True, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", callbacks=[early_stop_callback, lr_monitor])
    trainer.fit(model_espcn, train_loader, val_loader)

    # Evaluation
    model_espcn.eval()
    model_espcn.freeze()
    model_espcn.model.to(device)

    results = evaluate_model_ycbcr(model=model_espcn.model, dataset=test_ds, device=device, batch_size=batch_size_val, image_size=299)

    final_train_loss = model_espcn.train_losses_epoch[-1] if model_espcn.train_losses_epoch else None
    final_val_loss = model_espcn.val_losses_epoch[-1] if model_espcn.val_losses_epoch else None

    print(f"📉 Model {model_type.upper()} Final Train Loss: {final_train_loss:.4f}")
    print(f"📉 Model {model_type.upper()} Final Val Loss  : {final_val_loss:.4f}")

    plot_training_loss(model_espcn.train_losses_epoch, model_espcn.val_losses_epoch)
    show_sr_examples(model=model_espcn.model, dataset=test_ds, device=device, num_examples=4)

    return {
        'psnr': results['psnr'],
        'ssim': results['ssim'],
        'fid': results['fid'],
        'train_losses': model_espcn.train_losses_epoch,
        'val_losses': model_espcn.val_losses_epoch,
        'model': model_espcn.model  # add the trained model for inference

    }
    
    
    import numpy as np

def run_model_multiple_seeds(files, test_files, seeds, split_ratio, max_epochs, scale=4, lr= 1e-4,batch_size_train=16,batch_size_val=4,model_type='base'):
    """
    Run a given model multiple times with different seeds and aggregate results.

    Args:
        files: training files
        test_files: testing files
        seeds: list of seeds
        split_ratio: train/val split ratio
        scale: scaling factor for SR
        model_type: 'base', 'residual', or 'no_global'

    Returns:
        dict with mean and std of PSNR, SSIM, FID, and lists of losses
    """
    psnr_list, ssim_list, fid_list = [], [], []
    all_train_losses, all_val_losses = [], []

    for seed in seeds:
        set_seed(seed)  # your utility to set seeds globally
        results = run_model(
            files, test_files,
            seed=seed,
            split_ratio=split_ratio,
            max_epochs=max_epochs,
            scale=scale,
            lr=lr,
            batch_size_train=batch_size_train,
            batch_size_val=batch_size_val,
            model_type=model_type
        )
        psnr_list.append(results['psnr'])
        ssim_list.append(results['ssim'])
        fid_list.append(results['fid'])
        all_train_losses.append(results['train_losses'])
        all_val_losses.append(results['val_losses'])

    # Aggregate metrics
    psnr_mean, psnr_std = np.mean(psnr_list), np.std(psnr_list)
    ssim_mean, ssim_std = np.mean(ssim_list), np.std(ssim_list)
    fid_mean, fid_std   = np.mean(fid_list), np.std(fid_list)

    print(f"\n=== Results for model_type: {model_type.upper()} ===")
    print(f"PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}")
    print(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    print(f"FID : {fid_mean:.4f} ± {fid_std:.4f}")

    # Plot all training/validation losses
    # plot_multiple_training_losses(all_train_losses, all_val_losses, labels=[f"Seed {s}" for s in seeds])
    plot_multiple_training_losses(all_train_losses, all_val_losses)


    return {
        'psnr_mean': psnr_mean,
        'psnr_std': psnr_std,
        'ssim_mean': ssim_mean,
        'ssim_std': ssim_std,
        'fid_mean': fid_mean,
        'fid_std': fid_std,
        'all_train_losses': all_train_losses,
        'all_val_losses': all_val_losses
    }
