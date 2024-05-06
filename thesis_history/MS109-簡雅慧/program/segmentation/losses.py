import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        smooth = 1e-5
        pred = torch.sigmoid(pred) > 0.5
        num = target.size(0)
        pred = pred.type(torch.FloatTensor).cuda()
        pred = pred.view(num, -1)
        target = target.view(num, -1)
        intersection = (pred * target)
        dice = (2. * intersection.sum(1) + smooth) / (pred.sum(1) + target.sum(1) + smooth)

        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(pred, target, per_image=True)

        return loss
