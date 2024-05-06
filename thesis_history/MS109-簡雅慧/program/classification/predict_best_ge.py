# from model.resnest import *
# from model.masp_resnest import spp_resnest
# swin_transformer
import config.swint_config as cfg
from model.swin_transformer_3D import SwinTransformer
#from model.swin_transformer_2D import SwinTransformer



import pandas as pd
import numpy as np
import os
from os.path import join
# from dataset import *
from load_data_3D_new import load_data
#from load_data_2D_new import load_data
import sys, os
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# 要用哪個指標去挑
INDEX = 1
RANK = 3
REVERSE = [False, True, True, True, True] # 不用動
'''
0 : val loss
1 : val acc
2 : sensitivity
3 : specificity
4 : f1
'''
# model_name = 'only_rurusplit'
model_name = 'SWIRL3D_'

path = "./models/"+model_name+"/bm/fold_"

if __name__ == '__main__':
    
    train = [0,1,2,3,4]
    val = [3,4,0,1,2]
    test = [4,0,1,2,3]
    """
    train = [0, 1, 2, 3, 4]
    ## swin_transformer
    val = [3,4,0,1,2]
    test = [4,0,1,2,3]
    """
    ## dai
    #val = [3, 2, 1, 0, 4]
    #test = [4, 3, 2, 1, 0]
    # train = [0,2,4]
    # test = [4,2,0]
    # val = [3,1,4]
    torch.cuda.empty_cache()
    for train_num, val_num, test_num in zip(train, val, test):

        csv = pd.read_csv(join(path + str(train_num), "Model_" + str(test_num)) + '.csv')
        val_loss = csv['val_loss'].values
        val_acc = csv['val_acc'].values
        sensitivity = csv['sensitivity'].values
        specificity = csv['specificity'].values
        f1 = csv['f1'].values
        targets = [val_loss, val_acc, sensitivity, specificity, f1]
        target = targets[INDEX]

        target = [(score, i+1) for i, score in enumerate(target)]
        target = sorted(target, reverse=REVERSE[INDEX])

        best_5_epoch = []
        best_5_val_acc = []
        for s, e in target:
            if e <= 200:
                best_5_epoch += [e]
                best_5_val_acc += [s]
                if len(best_5_epoch) == 5:
                    break
        # best_5_epoch = target[:RANK]
        # best_5_epoch = [epoch for (score, epoch) in best_5_epoch]
        print('best 5 epoch and val_acc of val folder%d: %s, %s' %(val_num, best_5_epoch, best_5_val_acc))
        label_name = 'bm'
        file_path = './data'  # the path you save excel
        save_path = './models/'+model_name+'/' + label_name  # the path you save models
        # best_path = '../../res_data/classification/3D/'+model_name+''  # the path you save best model
        best_path = './res_data/classification/3D/'  # +model_name+''
        # test_dataset = load_data(mode='test', val=val_num, test=test_num, file_name = '../../data/new/fold')
        test_dataset = load_data(file_dir=file_path, label_name=label_name, val=val_num, test=test_num, mode='test')

        test_loader = DataLoader(dataset = test_dataset, batch_size=1, shuffle=False)

        #model = spp_resnest()
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

        model = model.cuda()

        best = 0.
        print('-' * 50)
        print("\nTest Folder Num : ", test_num)
        for ep in tqdm(best_5_epoch, ncols=50):
            # print("\nTest Folder Num : ", test_num)
            print('Test Epoch : ', ep)
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
                print('ACC:', acc)
                print('Sensitivity:', TP/(TP+FN))
                print('Specificity:', TN/(FP+TN))
                """
                df = pd.DataFrame(probs, columns=['ids', 'probs'])
                os.makedirs(best_path, exist_ok=True)
                # df.to_excel(best_path + model_name + '_' + str(train_num) + '_' + str(ep) + '_probs.xlsx')
                df.to_excel(best_path + model_name + '_' + str(test_num) + '_' + str(ep) + '_probs.xlsx')
                """

            print()
