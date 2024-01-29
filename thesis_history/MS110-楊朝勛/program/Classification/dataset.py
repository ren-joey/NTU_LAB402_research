import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchio as tio
import random
import os
import numpy as np
import pandas as pd
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

##### Dataset #####
class Lung_Dataset(Dataset):
    def __init__(self, images_path=None, clinical_data_path=None, train_val_test_list_path=None, mode=None, k=0, predict_mode=False):
        self.images_path = images_path
        self.clinical_data_path = clinical_data_path
        self.train_val_test_list_path = train_val_test_list_path
        self.mode = mode
        self.predict_mode = predict_mode  
        train = 'train' + str(k)
        validation = 'validation' + str(k)
        test = 'test' + str(k)
        
        self.clinical_data_info = pd.read_excel(self.clinical_data_path, dtype=str)
        
        #獲取分組名單
        if self.mode == 'train':  #training set, 
            group = pd.read_excel(self.train_val_test_list_path, dtype=str)[train]
            group = group.astype('str')         
        elif self.mode == 'val':  #validation set, 
            group = pd.read_excel(self.train_val_test_list_path, dtype=str)[validation]
            group = group.astype('str')    
        else:                #testing 
            group = pd.read_excel(self.train_val_test_list_path, dtype=str)[test]
            group = group.astype('str')
        
        self.img_files = [self.images_path / Path(id).with_suffix('.npy')  for id in group if id !='nan']
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img_name = img_path.stem

        # label
        label = self.clinical_data_info[self.clinical_data_info['Patient ID'].isin([img_name])]['是否存活'].values[-1]
        if isinstance(label, str):
            label = int(label)
        label = torch.tensor(label)

        # Image
        ct_img = np.load(img_path).astype(float)
        ct_img = (ct_img - ct_img.min()) / (ct_img.max() - ct_img.min())
        ct_img = torch.from_numpy(ct_img)
        ct_img = ct_img.to(torch.float32)
        ct_img = ct_img.view(1, ct_img.shape[0], ct_img.shape[1], ct_img.shape[2])

        if not self.predict_mode and self.mode == 'train':
            transforms_list = []
            if random.random() > 0.5:
                transforms_list.append(tio.RandomFlip(axes=(0), flip_probability=1))
            if random.random() > 0.5: 
                transforms_list.append(tio.RandomFlip(axes=(1), flip_probability=1))
            if random.random() > 0.5: 
                transforms_list.append(tio.RandomFlip(axes=(2), flip_probability=1))
            if random.random() > 0.5: 
                transforms_list.append(tio.RandomAffine(scales=0, degrees=15))
            if random.random() > 0.5: 
                transforms_list.append(tio.RandomAffine(scales=0, degrees=15))
            if random.random() > 0.5: 
                transforms_list.append(tio.RandomAffine(scales=0, degrees=15))
            transform = tio.Compose(transforms_list)
            ct_img = transform(ct_img)
        
        ct_img = transforms.Normalize(mean=torch.mean(ct_img), std=torch.std(ct_img))(ct_img)

        # clinical features
        clinical_info = (self.clinical_data_info[self.clinical_data_info['Patient ID'].isin([img_name])].drop(['是否存活'], axis=1).values[0])[1:]
        clinical_info = list(map(int, clinical_info))
        clinical_values, voi_size = clinical_info[:-3], clinical_info[-3:]
        clinical_values, voi_size = torch.tensor(clinical_values, dtype=torch.float32), torch.tensor(voi_size, dtype=torch.float32)

        return img_name, ct_img, clinical_values, voi_size, label