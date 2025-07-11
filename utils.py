# ================================================================
# Deep Learning and its Applications to Signal and Image Processing
# FINAL PROJECT - DIV2K Super Resolution Challenge
#
# Course: 361.2.1120
# Authors:   Or Shraga , Gal Apple 
# ================================================================

import torch
import numpy as np
import random
import cv2

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Seed worker for PyTorch DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def ycbcr_to_rgb_tensor(y, cb, cr):
    """Convert YCbCr tensors in [0,1] to RGB np array in [0,1]."""
    y = y.squeeze(0).cpu().numpy()
    cb = cb.squeeze(0).cpu().numpy()
    cr = cr.squeeze(0).cpu().numpy()
    ycbcr = np.stack([y, cb, cr], axis=2) * 255.0
    ycbcr = ycbcr.astype(np.uint8)
    rgb = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
    return rgb
