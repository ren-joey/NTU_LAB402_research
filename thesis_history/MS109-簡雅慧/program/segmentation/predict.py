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

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from sklearn.externals import joblib
from dataset import Dataset

import archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
from skimage import transform

def mask_to_255(mask):
    Image = np.asarray((mask * 255).astype(np.uint8))
    return Image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_number = 1   # 這邊可以換，選最好的
    datafolder = f'{fold_number}_woDS'
    args = joblib.load(os.path.join('models',datafolder,'args.pkl' ))


    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    # joblib.dump(args, os.path.join('models',datafolder,'args.pkl'))

    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)

    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join('models',datafolder,"best.pth")))
    model.eval()
    # 5-fold image path
    img_paths = 'D:/3D_ABUS/folds_xuan/img/'
    save_path = 'D:/3D_ABUS/folds_xuan/pred/'
    fold_path = 'D:/3D_ABUS/folds_xuan/fold_'+str(fold_number)+'.xlsx'   ### fold
    csv = pd.read_excel(fold_path)   ### fold
    ID = np.array(csv['id'])    ### fold


    # if not os.path.exists('output/result'):
        # os.makedirs('output/result')
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            for fn in os.listdir(img_paths):
            # for fn in ID:  ### fold
                print(f'predicting ID {fn}')
                input_ = np.load(img_paths+str(fn))
                # input_ = np.load(img_paths+str(fn)+'.npy')   ### fold
                img = torch.from_numpy(input_).unsqueeze(0).unsqueeze(0)
                img = img.to(device=device, dtype=torch.float32)

                # compute output
                if args.deepsupervision:
                    output = model(img)[-1]
                else:
                    output = model(img)

                output = torch.sigmoid(output).squeeze(1).squeeze(0).data.cpu().numpy()
                # output = output.squeeze(1).squeeze(0).data.cpu().numpy()
                output = output > 0.5
                result = mask_to_255(output)
                # np.save('./output/result/'+str(fn), result)
                # np.save(save_path+str(fn), result)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
