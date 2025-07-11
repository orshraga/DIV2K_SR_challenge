# ================================================================
# Deep Learning and its Applications to Signal and Image Processing
# FINAL PROJECT - DIV2K Super Resolution Challenge
#
# Course: 361.2.1120
# Authors:   Or Shraga , Gal Apple
# # ================================================================

import os
import glob
import zipfile
import subprocess
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
from utils import set_seed
import shutil
import os

def download_kaggle_dataset(dataset: str, kaggle_json_path: str, download_dir: str):
    """
    Download a dataset from Kaggle using the Kaggle API.
    Downloads only if the dataset folder is missing or empty.
    """
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    dest = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(dest):
        shutil.copy(kaggle_json_path, dest)
        os.chmod(dest, 0o600)

    # ✅ Check if dataset already exists and is non-empty
    if os.path.exists(download_dir) and len(os.listdir(download_dir)) > 0:
        print(f"✅ Dataset already exists in {download_dir}. Skipping download.")
        return

    # Otherwise, download
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset, "-p", download_dir, "--unzip"
    ], check=True)
    print(f"✅ Dataset downloaded and extracted to {download_dir}.")


def get_image_files(data_dir: str, subset: str = "train"):
    """
    Get list of image files for training or validation.
    Args:
        data_dir (str): Path to the extracted dataset directory
        subset (str): 'train' or 'valid'
    Returns:
        list: List of image file paths
    """
    if subset == "train":
        pattern = os.path.join(data_dir, "DIV2K_train_HR", "DIV2K_train_HR", "*.png")
    elif subset == "valid":
        pattern = os.path.join(data_dir, "DIV2K_valid_HR", "DIV2K_valid_HR", "*.png")
    else:
        raise ValueError("subset must be 'train' or 'valid'")
    files = glob.glob(pattern)
    print(f"Extracted {subset} images: {len(files)}")
    return files

def use_seed_and_split(files, scale, split_ratio=0.9, seed=42, dataset_class=None, dataset_kwargs=None):
    """
    Shuffle and split files into train/val sets, set seed, and return dataset instances.
    Args:
        files (list): List of file paths.
        split_ratio (float): Ratio for train split (e.g., 0.9 for 90% train).
        seed (int): Random seed for reproducibility.
        dataset_class (class): Dataset class to instantiate.
        dataset_kwargs (dict): Additional kwargs for dataset class.
    Returns:
        train_ds, val_ds: Dataset instances for training and validation.
    """
    if dataset_class is None:
        raise ValueError("dataset_class must be provided")
    if dataset_kwargs is None:
        dataset_kwargs = {}
    set_seed(seed)
    files = files.copy()
    random.shuffle(files)
    split_idx = int(split_ratio * len(files))
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    train_ds = dataset_class(train_files, patch_size=384, scale=scale, mode="train", **dataset_kwargs)
    val_ds = dataset_class(val_files,patch_size=384, scale=scale, mode="val", **dataset_kwargs)
    return train_ds, val_ds


class DIV2K_YCbCr_SR_Dataset(Dataset):
    def __init__(self, file_list, patch_size=384, scale=4, mode='train', add_cbcr=False):
        self.files = file_list
        self.patch_size = patch_size
        self.scale = scale
        self.mode = mode.lower()
        self.add_cbcr = add_cbcr
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("YCbCr")
        y, cb, cr = img.split()

        # Crop
        w, h = y.size
        if self.mode == "train":
            x = random.randint(0, w - self.patch_size)
            y0 = random.randint(0, h - self.patch_size)
        else:
            x = (w - self.patch_size) // 2
            y0 = (h - self.patch_size) // 2

        y = y.crop((x, y0, x + self.patch_size, y0 + self.patch_size))
       

        # Downsample
        lr_size = self.patch_size // self.scale
        lr_y = y.resize((lr_size, lr_size), Image.Resampling.BICUBIC)
        
        # Prepare output
        sample = {
            "lr": self.to_tensor(lr_y),
            "hr": self.to_tensor(y)
        }

        if self.add_cbcr:
            cb = cb.crop((x, y0, x + self.patch_size, y0 + self.patch_size))
            cr = cr.crop((x, y0, x + self.patch_size, y0 + self.patch_size))
            lr_cb = cb.resize((lr_size, lr_size), Image.Resampling.BICUBIC)
            lr_cr = cr.resize((lr_size, lr_size), Image.Resampling.BICUBIC)

            # Upscale CbCr to SR size
            sr_cb = lr_cb.resize((self.patch_size, self.patch_size), Image.Resampling.BICUBIC)
            sr_cr = lr_cr.resize((self.patch_size, self.patch_size), Image.Resampling.BICUBIC)

            sample.update({
                "cb_sr": self.to_tensor(sr_cb),
                "cr_sr": self.to_tensor(sr_cr),
                "cb_hr": self.to_tensor(cb),
                "cr_hr": self.to_tensor(cr)
            })

        return sample