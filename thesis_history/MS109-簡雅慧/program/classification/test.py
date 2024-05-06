#from model.masp_resnest import spp_resnest
from model.swin_transformer_3D import SwinTransformer
#from model.swin_transformer_2D import SwinTransformer
import config.swint_config as cfg

import pandas as pd
import numpy as np
from os.path import join
from load_data_3D_new import load_data
#from load_data_2D_new import load_data
import sys, os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
"""
model_name = 'MASPPNAS-fold1-220d_7'
label_name = 'bm'
file_path = './data'
save_path = './models/'+model_name+'/' + label_name
best_path = './res_data/classification/3D/'  # +model_name+''
"""
modeln = ['SWIRL3D', 'SWIRL3D', 'SWIRL3D', 'SWIRL3D', 'SWIRL3D'] #order: test fold num (best)
label_name = 'bm'
file_path = '../../data'
best_path = '../res_data/classification/3D/'

trainn = [1,2,3,4,0] #folder num
valn =   [4,0,1,2,3]
testn =  [0,1,2,3,4]
epn = [186,120,112,143,149]

all_total = 0
all_correct = 0
all_TP = 0
all_FP = 0
all_TN = 0
all_FN = 0
"""
trainn = [0,1,2,3,4]
valn =   [3,4,0,1,2]
testn =  [4,0,1,2,3]

valn =   [3,2,1,0,4] #dai
testn =  [4,3,2,1,0] #dai

train_num = 1
val_num = 2
test_num = 3
ep = 163
"""
for i in [0,1,2,3,4]:
    model_name = modeln[i]
    train_num = trainn[i]
    val_num = valn[i]
    test_num = testn[i]
    ep = epn[i]
    save_path = './models/'+model_name+'/' + label_name
    test_dataset = load_data(file_dir=file_path, label_name=label_name, val=val_num, test=test_num, mode='test')
    test_loader = DataLoader(dataset = test_dataset, batch_size=1, shuffle=False)

    #model = spp_resnest()
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
                                patch_norm=cfg.PATCH_NORM)
    """
    ###2D
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
    model = model.cuda()
    model_path = save_path+'/fold_'+str(train_num)+'/epoch_'+str(ep)+'.pth'
    model.load_state_dict(torch.load(model_path))
    total = 0
    correct = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    model.eval()
    probs = []
    with torch.no_grad():
        for i, (ids, data, label) in enumerate(test_loader):

            total += data.shape[0]
            data = data.cuda()

            pred = model(data)
            pred = torch.sigmoid(pred[:, 0])
            probs.append([ids[0], pred.cpu().detach().item()])
            pred = (pred >= 0.5).type(torch.FloatTensor)

            pred = pred.view(-1).item()
            label = label.view(-1).item()
            if pred == label:
                correct += 1
            if pred == 1.0 and label == 1.0:
                TP += 1
            if pred == 1.0 and label == 0.0:
                FP += 1
            if pred == 0.0 and label == 0.0:
                TN += 1
            if pred == 0.0 and label == 1.0:
                FN += 1

        acc = correct / total
        print('Test fold:', test_num)
        print('ACC:', acc, 'correct_num/total:', correct, '/', total)
        print('Sensitivity:', TP/(TP+FN), 'TP/FN:', TP, '/', FN)
        print('Specificity:', TN/(FP+TN), 'TN/FP:', TN, '/', FP)
        print(' ')

        all_total += total 
        all_correct += correct
        all_TP += TP
        all_FP += FP
        all_TN += TN
        all_FN += FN
        
        df = pd.DataFrame(probs, columns=['ids', 'probs'])
        os.makedirs(best_path, exist_ok=True)
        df.to_excel(best_path + model_name + '_' + str(test_num) + '_' + str(ep) + '_probs.xlsx')
        
accuracy = all_correct / all_total
print('Result from all 5 folds:')
print('ACC:', accuracy, 'correct/total:', all_correct, '/', all_total)
print('Sensitivity:', all_TP/(all_TP+all_FN), 'TP/FN:', all_TP, '/', all_FN)
print('Specificity:', all_TN/(all_FP+all_TN), 'TN/FP:', all_TN, '/', all_FP)
print('PPV:', all_TP/(all_TP+all_FP), 'TP/FP:', all_TP, '/', all_FP)
print('NPV:', all_TN/(all_TN+all_FN), 'TN/FN:', all_TN, '/', all_FN)   