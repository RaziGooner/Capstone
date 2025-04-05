import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

ENCODER = 'efficientnet-b4'
WEIGHTS = 'imagenet'

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        self.arc = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images, mask=None):
        logits = self.arc(images)

        if mask is not None:
            loss1 = DiceLoss(mode='binary')(logits, mask)
            loss2 = FocalLoss(mode='binary', gamma=3)(logits, mask)
            return logits, loss1 + loss2
        return logits
