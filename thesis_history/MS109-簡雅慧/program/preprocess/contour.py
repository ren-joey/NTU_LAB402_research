import numpy as np 
import os 
import matplotlib.pyplot as plt
from PIL import Image

def draw_contour(img_src,img_mask):
  
    for k in range(img_mask.shape[0]):
        for i in range(img_mask.shape[1]):
            for j in range(img_mask.shape[2] - 1):
                if (img_mask[k][i][j]<128 and img_mask[k][i][j+1]>=128):
                    img_src[k][i][j+1]=255
                if (img_mask[k][i][j]>=128 and img_mask[k][i][j+1]<128):
                    img_src[k][i][j]=255
    
    for k in range(img_mask.shape[0]):
        for i in range(img_mask.shape[2]):
            for j in range(img_mask.shape[1] - 1):
                if (img_mask[k][j][i]<128 and img_mask[k][j+1][i]>=128):
                    img_src[k][j+1][i]=255
                if (img_mask[k][j][i]>=128 and img_mask[k][j+1][i]<128):
                    img_src[k][j][i]=255

    return img_src

def save_2D(img, save_path):
    img_re = img[8]
    
    print('re', img_re.shape)

    img_re = Image.fromarray(img_re)
    img_re.save(save_path + '.png')


if __name__ == "__main__":
    img_dir = ''  # the fold of original image
    mask_dir = ''  # the fold of mask
    save_dir = ''  # contour save path

    
    for data in os.listdir(img_dir):
        if '.npy' in data:
            img = np.load(img_dir + data)
            mask = np.load(mask_dir + data)
            name = data[:-4]
            # print(np.max(img), np.max(mask))
            #print('img', img.shape)
            #print('mask', mask.shape)

            result = draw_contour(img, mask)
             
            np.save(save_dir + str(name) + '.npy', result)
            
            save_path = save_dir + '2D/' + str(name)
            save_2D(result, save_path)

            print(data, result.shape)
