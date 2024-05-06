from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import os
import torch
import warnings
import torchvision.transforms.functional as TF
import random
warnings.filterwarnings("ignore")


class load_data(Dataset):
    def __init__(self, file_dir, label_name, val, test, mode, transform=None, augmentation=False):
        self.imgs = []
        self.labels = []
        self.mode = mode
        self.aug = augmentation
        ids = 0
        file_name = '/check_'
        ids = 'id'
        name = label_name
        file_name = '/fold_'

        ID = []
        label = []
        if mode == 'test':
            csv = pd.read_excel(file_dir+file_name+str(test)+'.xlsx')
            self.ID = np.array(csv[ids])
            self.label = np.array(csv[name])
        elif mode == 'val':
            csv = pd.read_excel(file_dir+file_name+str(val)+'.xlsx')
            self.ID = np.array(csv[ids])
            self.label = np.array(csv[name])
        else:
            for i in range(5):
                if i != test and i!=val:
                    csv = pd.read_excel(file_dir+file_name+str(i)+'.xlsx')
                    ID.append(np.array(csv[ids]))
                    label.append(np.array(csv[name]))
            self.ID = np.concatenate((ID[0], ID[1], ID[2]), axis=0)
            self.label = np.concatenate((label[0], label[1], label[2]), axis=0)
        self.label = torch.from_numpy(self.label).type(torch.FloatTensor)

    def transform(self, inputs):
        o, h, m = inputs
        flipPermute = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        #flipping
        isFlip = np.random.randint(0, 8)
        isFlip = flipPermute[isFlip]
        if isFlip[0]:
            o = np.flip(o, axis=0)
            h = np.flip(h, axis=0)
            m = np.flip(m, axis=0)

        if isFlip[1]:
            o = np.flip(o, axis=1)
            h = np.flip(h, axis=1)
            m = np.flip(m, axis=1)

        if isFlip[2]:
            o = np.flip(o, axis=2)
            h = np.flip(h, axis=2)
            m = np.flip(m, axis=2)

        # random crop
        if np.random.randint(0, 2):

            o = np.pad(o, ((0, 0), (16, 16), (16, 16)), mode='constant', constant_values=0)
            h = np.pad(h, ((0, 0), (16, 16), (16, 16)), mode='constant', constant_values=0)
            m = np.pad(m, ((0, 0), (16, 16), (16, 16)), mode='constant', constant_values=0)

            offset_x = np.random.randint(32)
            offset_y = np.random.randint(32)
            o = o[:, offset_x:offset_x+128, offset_y:offset_y+128]
            h = h[:, offset_x:offset_x+128, offset_y:offset_y+128]
            m = m[:, offset_x:offset_x+128, offset_y:offset_y+128]
        # cutout
        if np.random.randint(0, 2):
            for i in range(np.random.randint(1, 11)):
                x = np.random.randint(0, 5)
                y = np.random.randint(0, 110)
                z = np.random.randint(0, 110)
                o[x:x+4, y:y+16, z:z+16] = 0
                h[x:x+4, y:y+16, z:z+16] = 0
                m[x:x+4, y:y+16, z:z+16] = 0




        # if random.random() > 0.5:
        #     noise = np.random.randn(16, 128, 128) * 0.125
        #     o += noise
        #     h += noise

        imgs = np.stack([o, h, m])
        return imgs

    def __getitem__(self, index):
        
        #aug = False
        if index >= len(self.label):
            index = np.random.randint(0, len(self.label))
            #aug = True
        
        fileID = self.ID[index]
        imgs = self.merge_data(fileID)
        if self.aug:
            imgs = self.transform(imgs)

        imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
        # imgs = torch.unsqueeze(imgs, 0)

        labels = self.label[index]

        return fileID, imgs, labels

    def __len__(self):     
        if self.mode == 'train':
            return int(len(self.label) / 0.6)
        return len(self.label)


    def merge_data(self, filename):

        img_o = np.load('../../data/3D/img/' + str(filename) + '.npy').astype(np.float32)   # the path you save resized volume
        # img_o = (img_o - img_o.min()) / (img_o.max() - img_o.min())
        img_h = np.load('../../data/3D/his/' + str(filename) + '.npy').astype(np.float32)   # the path you save higtorgram equlization
        # img_h = (img_h - img_h.min()) / (img_h.max() - img_h.min())
        img_m = np.load('../../data/3D/mask/' + str(filename) + '.npy').astype(np.float32)   # the path you save mask
        # img_m = img_m / 255.0
        # img_o = img_o / 255.0
        # img_h = img_h / 255.0
        # img_m = img_m / 255.0

        img = np.stack((img_o, img_h, img_m))

        return img
