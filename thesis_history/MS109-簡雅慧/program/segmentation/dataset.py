import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data
from os.path import splitext
import torch.nn.functional as F
import utils
class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_root, mask_root, testNum, mode='test'):
        self.img_root = img_root
        self.mask_root = mask_root
        self.mode = mode
        
        fold_path = 'D:/3D_ABUS/folds_xuan/fold_'
        if mode == 'test':
            csv = pd.read_excel(fold_path+str(testNum)+'.xlsx')
            self.ID = np.array(csv['id'])
        else:
            ID = []
            for i in range(5):
                if i != testNum:
                    csv = pd.read_excel(fold_path+str(i)+'.xlsx')
                    ID.append(np.array(csv['id']))
            self.ID = np.concatenate((ID[0], ID[1], ID[2], ID[3]), axis=0)

    def __len__(self):
        return len(self.ID)
    
    def __getitem__(self, idx):
        index = self.ID[idx]

        img_path = self.img_root + str(index) + '.npy'
        mask_path = self.mask_root + str(index) + '.npy'

        image = np.load(img_path)
        np_max, np_min = np.amax(image), np.amin(image)
        image = (image-np_min)/(np_max-np_min)
        
        mask = np.load(mask_path)
        # mask = mask/255

        image = torch.from_numpy(image).unsqueeze(0).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).unsqueeze(0).type(torch.FloatTensor)

        return image, mask
