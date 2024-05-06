# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
# from lovasz_losses import lovasz_hinge
import warnings
warnings.filterwarnings("ignore")

arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    
    parser.add_argument('--aug', default=True, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('-e', '--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    # parser.add_argument('--early-stop', default=25, type=int,
    #                     metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 2)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer, epoch, global_step, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()
    diceCoe = AverageMeter()
    model.train()

    for i, (train_input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        train_input = train_input.cuda()
        target = target.cuda()
        # compute output
        output = model(train_input)
        loss = criterion(output, target)
        # loss = lovasz_hinge(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        losses.update(loss.item(), train_input.size(0))
        ious.update(iou, train_input.size(0))
        diceCoe.update(dice, train_input.size(0))
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses_avg = losses.avg
    writer.add_scalar('Loss/train', losses_avg, global_step)
    log = OrderedDict([
        ('loss', losses_avg),
        ('iou', ious.avg),
        ('dice', diceCoe.avg),
    ])

    return log


def validate(args, val_loader, model, criterion, global_step):
    losses = AverageMeter()
    ious = AverageMeter()
    diceCoe = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (valid_input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            valid_input = valid_input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(valid_input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                    # loss += lovasz_hinge(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(outputs[-1], target)
            else:
                output = model(valid_input)
                loss = criterion(output, target)
                # loss = lovasz_hinge(output, target)
                iou = iou_score(output, target)
                dice = dice_coef(output, target)
            losses.update(loss.item(), valid_input.size(0))
            ious.update(iou, valid_input.size(0))
            diceCoe.update(dice, valid_input.size(0))
    avgDice = diceCoe.avg
    avgIou = ious.avg
    writer.add_scalar('Dice/Valid', avgDice, global_step)
    writer.add_scalar('Iou/Valid', avgIou, global_step)       
    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', avgIou),
        ('dice', avgDice)
    ])

    return log


def main():
    start_time = time.time()
    cudnn.benchmark = True
    for fold_number in range(5):
        args = parse_args()

        if args.name is None:
            if args.deepsupervision:
                args.name = '%s_wDS' %(fold_number)
            else:
                args.name = '%s_woDS' %(fold_number)
        # if not os.path.exists('models/%s' %args.name):
            # os.makedirs('models/%s' %args.name)

        print('Config -----')
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)))
        print('------------')

        with open('models/%s/args.txt' %args.name, 'w') as f:
            for arg in vars(args):
                print('%s: %s' %(arg, getattr(args, arg)), file=f)

        joblib.dump(args, 'models/%s/args.pkl' %args.name)

        # define loss function (criterion)
        if args.loss == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss().cuda()
        else:
            criterion = losses.__dict__[args.loss]().cuda()
        
        # Data Path
        img_paths = 'D:/3D_ABUS/folds_xuan/img/'
        mask_paths = 'D:/3D_ABUS/folds_xuan/mask_gt/'

        dataset = Dataset(img_paths, mask_paths, fold_number, 'train')
        n_val = int(len(dataset) * 0.2)
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        # create model
        print("=> creating model %s" %args.arch)
        model = archs.__dict__[args.arch](args)

        model = model.cuda()

        print(f'model total parameter: {count_params(model)}')

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1)

        log = pd.DataFrame(index=[], columns=[
            'epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'
        ])
        writer = SummaryWriter(f'./runs/fold_{fold_number}')

        global_step = 0
        best_loss = 50
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, threshold=0.01)
        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' %(epoch, args.epochs))
            current_lr = args.lr * (0.1 ** (epoch // 20))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # train for one epoch
            train_log = train(args, train_loader, model, criterion, optimizer, epoch, global_step)
            global_step += 1
            # evaluate on validation set
            val_log = validate(args, val_loader, model, criterion, global_step)
            # scheduler.step(val_log['loss'])
            print('fold %d loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
                %(fold_number, train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice']))

            tmp = pd.Series([
                epoch,
                optimizer.param_groups[0]['lr'],
                train_log['loss'],
                train_log['iou'],
                train_log['dice'],
                val_log['loss'],
                val_log['iou'],
                val_log['dice'],
            ], index=['epoch', 'lr', 'loss', 'dice', 'iou', 'val_loss', 'val_iou', 'val_dice'])

            log = log.append(tmp, ignore_index=True)
            log.to_excel('models/%s/log.xlsx' %args.name, index=False)


            if val_log['loss'] <= best_loss:
                # torch.save(model.state_dict(), 'models/%s/best.pth' %args.name)
                best_loss = val_log['loss']
                print("=> saved best model")
            torch.cuda.empty_cache()
        writer.close()
    stop_time = time.time()
    print(f"Time elapsed: {(stop_time- start_time)/60:.3f} mins")

if __name__ == '__main__':
    print('cuda' if torch.cuda.is_available() else 'cpu')
    main()
