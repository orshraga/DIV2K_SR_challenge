# ================================================================
# Deep Learning and its Applications to Signal and Image Processing
# FINAL PROJECT - DIV2K Super Resolution Challenge
#
# Course: 361.2.1120
# Authors:   Or Shraga , Gal Apple
# # ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

def make_resblock(ch):
    return nn.Sequential(
        nn.Conv2d(ch, ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, padding=1),
    )

##############################################
# Base ESPCN model (Y channel only)
##############################################
class ESPCN_YChannel(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 1 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        return self.layers(x)



##############################################
# NoGlobal variant for ablation study
##############################################
class ESPCN_YChannel_NoGlobal(nn.Module):
    def __init__(self, scale_factor=4, n_feats=64, n_res=4):
        super().__init__()
        self.head = nn.Conv2d(1, n_feats, 3, padding=1)
        body = []
        for _ in range(n_res):
            body.append(make_resblock(n_feats))
            body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, scale_factor**2, 3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        x0 = F.relu(self.head(x))
        res = self.body(x0) + x0  # local skips only
        sr = self.tail(res)
        return torch.clamp(sr, 0, 1)  # no global add

class ESPCN_YChannel_Residual(nn.Module):
    def __init__(self, scale_factor=4, n_feats=64, n_res=4):
        super().__init__()
        self.scale_factor = scale_factor

        self.head = nn.Conv2d(1, n_feats, kernel_size=3, padding=1)
        body = []
        for _ in range(n_res):
            body.append(make_resblock(n_feats))
            body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, lr):
        upsampled = F.interpolate(lr, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        x = F.relu(self.head(lr))
        res = self.body(x) + x
        sr = self.tail(res)
        return torch.clamp(upsampled + sr, 0, 1)

##############################################
# Updated LightningModule to support ablation
##############################################
class ESPCNLightning_YChannel(pl.LightningModule):
    def __init__(self, lr=1e-4, scale_factor=4, model_type='base', n_feats=64, n_res=4):
        super().__init__()
        self.save_hyperparameters()

        if model_type == 'residual':
            self.model = ESPCN_YChannel_Residual(scale_factor=scale_factor, n_feats=n_feats, n_res=n_res)
        elif model_type == 'no_global':
            self.model = ESPCN_YChannel_NoGlobal(scale_factor=scale_factor, n_feats=n_feats, n_res=n_res)
        else:
            self.model = ESPCN_YChannel(scale_factor=scale_factor)

        self.train_losses_epoch = []
        self.val_losses_epoch = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sr = self(batch['lr']).clamp(0.0, 1.0)
        loss = F.mse_loss(sr, batch['hr'])
        self._train_epoch_losses.append(loss)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        sr = self(batch['lr']).clamp(0.0, 1.0)
        loss = F.mse_loss(sr, batch['hr'])
        self._val_epoch_losses.append(loss)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_start(self):
        self._train_epoch_losses = []

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self._train_epoch_losses).mean()
        self.train_losses_epoch.append(avg_loss.item())

    def on_validation_epoch_start(self):
        self._val_epoch_losses = []

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self._val_epoch_losses).mean()
        self.val_losses_epoch.append(avg_loss.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
