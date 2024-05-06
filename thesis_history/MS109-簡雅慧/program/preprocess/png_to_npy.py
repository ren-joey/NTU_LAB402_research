import os
import numpy as np
import cv2

load_path = './img/'
# save_path = 'D:/3D_ABUS/2D_128/'
os.makedirs(save_path, exist_ok=True)

for i, filename in enumerate(os.listdir(load_path)):
    filepath = os.path.join(load_path, filename)
    imgs = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    imgs = cv2.resize(imgs, (128, 128), interpolation=cv2.INTER_CUBIC)

    # imgs = cv2.equalizeHist(imgs)
    # imgs = ((imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs)))

    print(imgs.shape)

    # print(np.max(imgs))
    # print(np.min(imgs))

    np.save(save_path+filename[:-4]+'.npy', imgs)
    
    
print(i+1)
print('\n---done---')