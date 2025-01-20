import torch
from torch import nn
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer
import pytorch_lightning as pl
from builders.models.i3d import InceptionI3d

class FirstLettersDecoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask

class FirstLettersi3dModel(pl.LightningModule):
    def __init__(self, pred_shape, size=256, with_norm=False):
        super(FirstLettersi3dModel, self).__init__()
        self.save_hyperparameters()
        self.backbone = InceptionI3d(in_channels=1, num_classes=512)
        self.decoder = FirstLettersDecoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1, 1, 20, 256, 256))],
                               upscale=1)

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)

        return pred_mask

class TimeSFormerInk(pl.LightningModule):
    def __init__(self, pred_shape, size=256, with_norm=False):
        super(TimeSFormerInk, self).__init__()

        self.save_hyperparameters()
        self.backbone=TimeSformer(
                dim = 512,
                image_size = 64,
                patch_size = 16,
                num_frames = 26,
                num_classes = 16,
                channels=1,
                depth = 8,
                heads = 6,
                dim_head =  64,
                attn_dropout = 0.1,
                ff_dropout = 0.1
            )
        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        x = self.backbone(torch.permute(x, (0, 2, 1, 3, 4)))
        x = x.view(-1, 1, 4, 4)
        return x



