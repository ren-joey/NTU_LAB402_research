import numpy as np
import torch
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from skimage import transform
from sklearn.model_selection import KFold
from tqdm import tqdm

# from histogram import histogram

import os
import warnings

warnings.filterwarnings("ignore")

df0 = pd.read_excel('D:/3D_ABUS/case.xlsx', sheet_name='Ben')
df1 = pd.read_excel('D:/3D_ABUS/case.xlsx', sheet_name='Mal')
# print('B', df0.head())
# print('M', df1.head())

filename0 = Series.as_matrix(df0['ID'])
filename1 = Series.as_matrix(df1['ID'])


# 等比例縮放
def rescale_data(filedir):
    for idx, filename in enumerate(os.listdir(filedir)):
        filepath = os.path.join(filedir, filename)
        print(filepath)

        img = np.load(filepath)
        # print('o', img.shape)

        # plt.imshow(img[img.shape[0]//2])
        # plt.show()

        new = 128   ###
        _, x, y = img.shape

        ###  resize
        if x >= y:
            scale = float(new / x)
        else:
            scale = float(new / y)            
        re_x = int(x * scale)
        re_y = int(y * scale)

        img_re = transform.resize(img, (16,re_x,re_y))  # (16,128,128)

        ###  normalize
        img_re = ((img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re)))

        ###  padding
        h = new - re_x
        v = new - re_y
        if h % 2 == 0:
            left = int(h / 2)
            right = left
        else:
            left = int(h / 2) + 1
            right = left - 1
        if v % 2 == 0:
            up = int(v / 2)
            down = up
        else:
            up = int(v / 2) + 1
            down = up -1
        img_re = np.pad(img_re, ((0, 0), (left, right), (up, down)), mode='constant', constant_values=0)
        # img_re = fix_mask(img_re)   # for mask file
        print('pad', img_re.shape)

        #plt.imshow(img_re[20], cmap='gray')
        #plt.show()

        path = os.path.join(save_path, filename)
        # print(path)
        np.save(path, img_re)

    print(str(idx+1))

# 固定size縮放
def resize_data(filedir):
    for idx, filename in enumerate(os.listdir(filedir)):
        filepath = os.path.join(filedir, filename)
        print(filepath)

        img = np.load(filepath)

        # img_re = transform.resize(img, (16,128,128))
        # img_re = ((img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re)))
        # img_re = img_re * 255
        img_re = ndimage.interpolation.zoom(img, [16/img.shape[0], 128/img.shape[1], 128/img.shape[2]]
                                , mode='nearest')
        # # order 1 bilinear, 3 cubic, mode='nearest'

        print(np.max(img),np.max(img_re))
        print(np.min(img),np.min(img_re))

        # img_re = histogram(img_re)
                
        ###  normalize
        # img_re = ((img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re)))
        print('re', img_re.shape)

        path = os.path.join(save_path, filename)
        #print(path)
        np.save(path, img_re)

    print(str(idx+1))

# 提取中間部分，多退少補
def extract_mid_data(filedir):
    for idx, filename in enumerate(os.listdir(filedir)):
        filepath = os.path.join(filedir, filename)
        print(filepath)

        img = np.load(filepath)
        slices = img.shape[0]

        if slices > 15:
            s = int(slices / 2)
            img_re = img[(s-7):(s+8)]
        else:
            if slices % 2 == 0:
                pre = int((15-slices) / 2)
                post = pre + 1
            else:
                pre = int((15-slices) / 2)
                post = pre
            img_re = np.pad(img, ((pre, post), (0, 0), (0, 0)), mode='constant', constant_values=0)

        if img_re.shape[0] != 15:
            print('!!!')

        img_re = transform.resize(img, (15,81,81))  # eff-(80,80,80)  res-(16,112,112)  inc-(43,81,81)
        img_re = ((img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re)))
        img_re = img_re * 255

        #img_re = histogram(img_re)
        # img_re = ((img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re)))
        print('re', img_re.shape)

        path = os.path.join(save_path, filename)
        #print(path)
        np.save(path, img_re)

    print(str(idx+1))

# mask二值化
def fix_mask(imgs):

    for z in range(imgs.shape[0]):
        for y in range(imgs.shape[1]):
            for x in range(imgs.shape[2]):
                if imgs[z,y,x] >= 0.1:
                    imgs[z,y,x] = 1
                else:
                    imgs[z,y,x] = 0

    print(imgs.shape)
    # np.save(save_path+filename, imgs)

    return imgs

# 提取中間的一張2D slide
def save_2D(filedir):
    for idx, filename in enumerate(os.listdir(filedir)):
        filepath = os.path.join(filedir, filename)
        filename = filename[:-4]
        print(filename)

        img = np.load(filepath)
        img_re = img[8]
        # img_re = histogram(img_re)

        print(np.max(img),np.max(img_re))
        print(np.min(img),np.min(img_re))
        print('re', img_re.shape)

        path = os.path.join(save_path, filename)
        np.save(path, img_re)
        # img_re = Image.fromarray(img_re)
        # img_re.save(path + '.png')

    print(str(idx+1))



if __name__ == "__main__":
    # load_path = 'D:/3D_ABUS/folds_xuan/img'
    # save_path = 'D:/3D_ABUS/folds_xuan/2D/img'
    os.makedirs(save_path, exist_ok=True)  #上面的路徑記得改好后開起來

    # rescale_data(load_path)
    # resize_data(load_path)
    # extract_mid_data(load_path)
    # save_2D(load_path)

    print('\nfinish')