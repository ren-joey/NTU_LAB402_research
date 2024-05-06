# -*- coding: utf-8 -*-
# use for evaluate result
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import warnings
from datetime import datetime
from dice_loss import dice_coe
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import joblib
from dataset import Dataset
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import archs 
import losses
from utils import str2bool, count_params
import warnings
warnings.filterwarnings("ignore")

def eval_net(net, loader, device, n_test):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot_dice = 0
    tot_iou = 0

    for i, (input_, target) in enumerate(loader):
        
        imgs = input_.to(device=device, dtype=torch.float32)
        mask_type = torch.float32
        true_masks = target.to(device=device, dtype=mask_type)

        mask_pred = net(imgs)

        for true_mask, pred in zip(true_masks, mask_pred):
            # pred = torch.sigmoid(pred)
            # pred = (pred > 0.5).float()
            # if net.n_classes > 1:
            #     tot_dice += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
            # else:
            # dice, iou = dice_coef(pred.unsqueeze(0), true_mask.unsqueeze(0))
            iou = iou_score(pred, true_mask)
            dice = dice_coef(pred, true_mask)
            # print('Dice Coeff: {:.4f}, Iou: {:.4f}'.format(dice.item(), iou.item()))
            tot_dice += dice.item()
            tot_iou += iou.item()

    return tot_dice / n_test, tot_iou /n_test


def mask_to_255(mask):
    Image = np.asarray((mask * 255).astype(np.uint8))
    return Image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_dice = 0.
    total_iou = 0.
    for fold_number in range(5):
        args = joblib.load(os.path.join('models',f'{fold_number}_woDS','args.pkl' ))

        if fold_number == 0:
            print('Config -----')
            for arg in vars(args):
                print('%s: %s' %(arg, getattr(args, arg)))
            print('------------')

        joblib.dump(args, os.path.join('models',f'{fold_number}_woDS','args.pkl'))

        # create model
        print("=> creating model %s" %args.arch)
        model = archs.__dict__[args.arch](args)

        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join('models',f'{fold_number}_woDS',"best.pth")))
        model.eval()
        img_paths = 'D:/3D_ABUS/folds_xuan/img/'
        mask_paths = 'D:/3D_ABUS/folds_xuan/mask/'   #ground truth
        test_dataset = Dataset(img_paths, mask_paths, fold_number)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_dice, test_iou = eval_net(model, test_loader, device, len(test_dataset))
        print('fold {} test Dice Coeff: {:.4f}, test Iou: {:.4f}'.format(fold_number, test_dice, test_iou))
        total_dice += test_dice
        total_iou += test_iou
    print('average Dice Coeff: {:.4f}, average Iou: {:.4f}'.format(total_dice/5, total_iou/5))


if __name__ == '__main__':
    main()
