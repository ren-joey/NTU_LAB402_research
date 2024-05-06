import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import argparse
import warnings

import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import TensorDataset, DataLoader
# from timm.data import Mixup #only for 2-D
from mixup_cutmix_3D import Mixup
#from mixup_cutmix_2D import Mixup
# from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler

import config.swint_config as cfg

from model.swin_transformer_3D import SwinTransformer
#from model.swin_transformer_2D import SwinTransformer

## from AdaBelief import AdaBelief
from load_data_3D_new import *
#from load_data_2D_new import *

# from dataset import *
## from loss import *

## try later (speed up)
# from apex import amp

if __name__ == '__main__':

    # cuda = torch.cuda.is_available()
    # device = 'cuda' if cuda else 'cpu'

    valn = [3,4,0,1,2]# valn = [3,2,1,0,4]
    testn = [4,0,1,2,3]# testn = [4,3,2,1,0]
    label_name = 'bm'
    # file_path = '../../data'  # the path you save excel
    file_path = './data'
    torch.cuda.empty_cache()
    for i in range(5):
    # for i in [2]:
        val_num = valn[i]
        test_num = testn[i]
        # save_path = './models/real_resnest/'+label_name+'/fold_'+str(i) # the path you save models
        save_path = './models/SWIRL3D_/'+label_name+'/fold_'+str(i) # the path you save models

        os.makedirs(save_path, exist_ok=True)
        epoch = 200 # 200
        Batch_size = 6 #6::dai/ 36::zhang
        best_loss = 1.
        best_acc = 50. #60.

        train_dataset = load_data(file_dir=file_path, label_name=label_name, val=val_num, test=test_num, mode='train', augmentation=True)
        val_dataset = load_data(file_dir=file_path, label_name=label_name, val=val_num, test=test_num, mode='val')


        # model = resnest50(pretrained=False)
        # model = spp_resnest()
        ###3D
        
        model = SwinTransformer(img_size=cfg.IMG_SIZE,
                                dpatch_size=cfg.DEPTH_PATCH_SIZE,
                                patch_size=cfg.PATCH_SIZE,
                                in_chans=cfg.IN_CHANS,
                                num_classes=cfg.NUM_CLASSES,
                                embed_dim=cfg.EMBED_DIM,
                                depths=cfg.DEPTHS,
                                num_heads=cfg.NUM_HEADS,
                                window_size=cfg.WINDOW_SIZE,
                                mlp_ratio=cfg.MLP_RATIO,
                                qkv_bias=cfg.QKV_BIAS,
                                qk_scale=cfg.QK_SCALE,
                                drop_rate=cfg.DROP_RATE,
                                drop_path_rate=cfg.DROP_PATH_RATE,
                                ape=cfg.APE,
                                patch_norm=cfg.PATCH_NORM) #, use_checkpoint=cfg.USE_CHECKPOINT)
        
        ###2D
        """
        model = SwinTransformer(img_size=cfg.IMG_SIZE,
                                patch_size=cfg.PATCH_SIZE,
                                in_chans=cfg.IN_CHANS,
                                num_classes=cfg.NUM_CLASSES,
                                embed_dim=cfg.EMBED_DIM,
                                depths=cfg.DEPTHS,
                                num_heads=cfg.NUM_HEADS,
                                window_size=cfg.WINDOW_SIZE,
                                mlp_ratio=cfg.MLP_RATIO,
                                qkv_bias=cfg.QKV_BIAS,
                                qk_scale=cfg.QK_SCALE,
                                drop_rate=cfg.DROP_RATE,
                                drop_path_rate=cfg.DROP_PATH_RATE,
                                ape=cfg.APE,
                                patch_norm=cfg.PATCH_NORM)
        """

        # model = resnet50()
        model = model.cuda()


        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("\nTotal number of params of training folder%d = %d" % (i, pytorch_total_params))


        ### load dataset
        train_loader = DataLoader(dataset = train_dataset,
                                batch_size = Batch_size,
                                shuffle = True, num_workers=cfg.NUMBER_WORKERS, drop_last=True)
        val_loader = DataLoader(dataset = val_dataset,
                                batch_size = Batch_size,
                                shuffle = True, num_workers=cfg.NUMBER_WORKERS)

        # setup mixup / cutmix
        
        mixup_fn = None
        
        mixup_active = cfg.AUG["MIXUP"] > 0 or cfg.AUG["CUTMIX"] > 0. or cfg.AUG["CUTMIX_MINMAX"] is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=cfg.AUG["MIXUP"], cutmix_alpha=cfg.AUG["CUTMIX"], cutmix_minmax=cfg.AUG["CUTMIX_MINMAX"],
                prob=cfg.AUG["MIXUP_PROB"], switch_prob=cfg.AUG["MIXUP_SWITCH_PROB"], mode=cfg.AUG["MIXUP_MODE"],
                label_smoothing=cfg.LABEL_SMOOTHING, num_classes=cfg.NUM_CLASSES)
        


        ### train & validation
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        val_sensi_list = []
        val_spec_list = []
        val_f1_list = []

        
        ### optimizer for masppnet
        # optimizer = AdaBelief(model.parameters(), lr=0.0001, eps=1e-8, betas=(0.9, 0.999), weight_decouple = True, weight_decay = 1e-4, fixed_decay = False)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        """
        ### optimizer for original swin-T
        if cfg.TRAIN['OPTIMIZER_NAME'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), eps=cfg.TRAIN['OPTIMIZER_EPS'], betas=cfg.TRAIN['OPTIMIZER_BETAS']
                                    lr=cfg.TRAIN['BASE_LR'], weight_decay=cfg.TRAIN['WEIGHT_DECAY'])
        """
        ### lr_scheduler for swin-T
        """
        num_steps = int(cfg.TRAIN['EPOCHS'] * len(train_loader))
        warmup_steps = int(cfg.TRAIN['WARMUP_EPOCHS'] * len(train_loader))
        if cfg.TRAIN['LR_SCHEDULER_NAME'] == 'cosine':
            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                t_mul=1.,
                lr_min=cfg.TRAIN['MIN_LR'],
                warmup_lr_init=cfg.TRAIN['WARMUP_LR'],
                warmup_t=warmup_steps,
                warmup_prefix=False,
                cycle_limit=1,
                t_in_epochs=False,
            )
            """
        
        if cfg.AUG["MIXUP"] > 0.:
        # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
            print('SoftTargetCrossEntropy')
        elif cfg.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=cfg.LABEL_SMOOTHING)
            print('LabelSmoothingCrossEntropy')
        else:
            # criterion = nn.CrossEntropyLoss()
            criterion = nn.BCEWithLogitsLoss()
            print('BCEWithLogitsLoss')
        """
        #loss_bce = nn.BCEWithLogitsLoss()
        criterion = nn.BCEWithLogitsLoss()
        print('BCEWithLogitsLoss')
        """
        print("Start training")
        t_start = time.time()
        for ep in range(epoch):
            len_data = len(train_loader)
            total_loss = 0.0
            # train
            print('\n---training epoch%3d---' % (ep+1))
            total = 0
            acc = 0
            model.train()
            for i, (id, x, y) in enumerate(tqdm(train_loader, ncols=50)):
                #(b,3,16,128,128)
                total += x.size(0)
                #print('y before mixup:', y)
                
                if mixup_fn is not None:
                    x, y = mixup_fn(x, y)
                    #print('y after mixup:', y)
                    #print(torch.ones(y.size()) - y)
                    #y = torch.cat((y.reshape(1, -1), (torch.ones(y.size()) - y).reshape(1, -1)), 0)
                    #y = torch.transpose(y, 0, 1)
                    #print('y for cross entropy:', y)
                
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                out = model(x)

                # loss = focal_loss(out, y, alpha=0.5, gamma=gamma)

                # print(out.view(-1))
                out = out.view(-1)
                #print('out:', out)
                #out = out.view(-1, 2)
                y = y.view(-1)
                #print('y:', y)
                #y = y.view(-1, 2)
                loss = criterion(out, y)
                #print('loss:', loss)
                # loss = focal_loss(out, y, alpha=0.5, gamma=gamma)
                out= F.sigmoid(out)
                #print('out.sigmoid:', out)
                #loss_s = criterion(out, y)
                #print('loss_s:', loss_s)
                
                #out = F.sigmoid(out[:, 0])
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                #lr_scheduler.step_update(ep * len_data + i)

                pred = (out >= 0.5).type(torch.FloatTensor)
                y = (y >= 0.5).type(torch.FloatTensor)
                #y = (y[:, 0] >= 0.5).type(torch.FloatTensor)
                # type(torch.FloatTensor).cuda()
                acc += torch.sum(pred.cpu().detach().eq(y.cpu().detach().view_as(pred))).item()

                # print('epoch:[%d/%d] iter:[%d/%d] loss: %.4f'
                #     % (ep+1, epoch, i+1, len_data, loss.item()))


            torch.save(model.state_dict(), save_path+'/epoch_'+str(ep+1)+'.pth')

            acc = acc / total * 100
            train_acc_list.append(acc)
            print('train_acc:', acc)
            print('train_loss:', total_loss/len_data)
            total_loss /= len_data
            train_loss_list.append(total_loss)

            # validation
            val_loss = 0.0
            total = 0
            acc = 0.0
            corret = 0
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            criterion = nn.BCEWithLogitsLoss()

            print('---val folder%d---' % (val_num))
            with torch.no_grad():
                model.eval()
                for i, (id, x, y) in enumerate(tqdm(val_loader, ncols=50)):
                    total += x.size(0)
                    x, y = x.cuda(), y.cuda()
                    pred = model(x)

                    # loss = focal_loss(pred, y, alpha=0.5, gamma=gamma)
                    pred = pred.view(-1)
                    #pred = pred.view(-1, 2)
                    #pred = pred[:, 0]
                    y = y.view(-1)
                    loss = criterion(pred, y)
                    #print('pred:',pred)
                    #print('y:', y)
                    #print('loss:', loss)
                    pred = F.sigmoid(pred)
                    #print('pred.sigmoid:', pred)
                    #loss_s = criterion(pred, y)

                    val_loss += loss.item()

                    pred = (pred >= 0.5).type(torch.cuda.FloatTensor)

                    # pred = (pred >= 0.5)
                    pred = pred.view(-1)
                    y = y.view(-1)
                    TP += ((pred==1.0)*(y==1.0)).sum().item()
                    FP += ((pred == 1.0) * (y == 0.0)).sum().item()

                    TN += ((pred == 0.0)*(y == 0.0)).sum().item()
                    FN +=  ((pred == 0.0)*(y == 1.0)).sum().item()


                    acc += torch.sum(pred.cpu().detach().eq(y.cpu().detach().view_as(pred))).item()

                # print(acc, '/', total)

                val_loss /= len(val_loader)
                val_loss_list.append(val_loss)
                acc = acc / total * 100
                val_acc_list.append(acc)
                print('val_acc:', acc)
                print('val_loss:', val_loss)
                sensitivity = TP/(TP+FN)
                specificity = TN/(FP+TN)
                f1 = 2 * TP / (2 * TP + FN + FP)
                print('sensitivity:', sensitivity, '\tspecificity:', specificity, '\tF1:', f1)
                # print('{:.4f}  {:.4f}  {:.4f}'.format(sensitivity, specificity, f1))
                val_sensi_list.append(sensitivity)
                val_spec_list.append(specificity)
                val_f1_list.append(f1)


                # save model
                # save lowerst val loss model
                if val_loss <= best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), save_path+'/model_'+str(test_num)+'_fold_'+str(val_num)+'_loss.pth')
                # save acc highest acc model
                if acc >= best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), save_path+'/model_'+str(test_num)+'.pth')

            print(label_name, ' current best acc & loss:', best_acc, '/', best_loss)
            # scheduler.step()

        t_end = time.time()
        print(t_end - t_start, 'sec')

        name = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'sensitivity', 'specificity', 'f1']
        f = np.stack((train_loss_list, val_loss_list, train_acc_list, val_acc_list, val_sensi_list, val_spec_list, val_f1_list), 1)
        log = pd.DataFrame(columns=name, data=f)
        log.to_csv(save_path+'/model_'+str(test_num)+'.csv')  # learning curve


    print('------ done ------')
