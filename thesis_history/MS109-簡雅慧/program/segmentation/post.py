import numpy as np 
import cv2
import os 
import matplotlib.pyplot as plt
from skimage import transform
def resizeTo(pil_img):
        mid_slice = pil_img.shape[0]//2
        mid_row = pil_img.shape[1]//2
        mid_col = pil_img.shape[2]//2
        img_trans = pil_img[mid_slice-16:mid_slice+16,: ,:]
        img_trans = transform.resize(img_trans, (32, 64, 128))
        return img_trans

def draw_contour(img_src,img_mask):
  
    for k in range(img_mask.shape[0]):
        for i in range(img_mask.shape[1]):
            for j in range(img_mask.shape[2] - 1):
                if (img_mask[k][i][j]<128 and img_mask[k][i][j+1]>=128):
                    # print('got')
                    img_src[k][i][j+1]=255
                if (img_mask[k][i][j]>=128 and img_mask[k][i][j+1]<128):
                    # print('got')
                    img_src[k][i][j]=255
    
    for k in range(img_mask.shape[0]):
        for i in range(img_mask.shape[2]):
            for j in range(img_mask.shape[1] - 1):
                if (img_mask[k][j][i]<128 and img_mask[k][j+1][i]>=128):
                    # print('got')
                    img_src[k][j+1][i]=255
                if (img_mask[k][j][i]>=128 and img_mask[k][j+1][i]<128):
                    # print('got')
                    img_src[k][j][i]=255

    return img_src


def postprocessing(mask):
    for i in range(mask.shape[0]):
        mask[i] = (mask[i] - np.min(mask[i])) / np.max(mask[i])
        mask[i,:,:] = largestConnectedComponent(mask[i,:,:] * 255).astype(np.uint8)
        mask[i,:,:] = fillhole(mask[i,:,:].astype(np.uint8))
        
    return mask 

def largestConnectedComponent(img):
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = img.astype(np.uint8)
    
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # output = cv2.connectedComponents(binary, connectivity=8)
    # label = output[1]

    # labels = np.zeros((x,y), dtype=np.uint8)
    # for row in range(x):
    #     for col in range(y):
    #         if label[row, col] > 0:
    #             labels[row, col] = 1
    #         else:
    #             labels[row, col] = 0

    numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if(stats.shape[0] == 1):
        return img 

    maxarea = stats[1,cv2.CC_STAT_AREA]
    label = 1
    for i,stat in enumerate(stats[1:]):
        if maxarea < stat[cv2.CC_STAT_AREA]:
            label = i+1
            maxarea = stat[cv2.CC_STAT_AREA]

    bg = labels != label
    fg =  bg != True
    labels[bg] = 0
    labels[fg] = 255

    return labels

def fillhole(img):
    hole = img.copy()
    hole = cv2.floodFill(hole,None,(0,0),255)
    img[hole[1] == 0] = 255
    return img  

if __name__ == "__main__":
    img_dir = 'D:/3D_ABUS/folds_xuan/img/' #'D:/3D_ABUS/data_128/'
    mask_dir = 'D:/3D_ABUS/folds_xuan/pred/' #'./output/result/'
    afterPost_dir = 'D:/3D_ABUS/folds_xuan/mask/' #'./output/after/'
    con_dir = 'D:/3D_ABUS/contour/'
    if not os.path.exists(afterPost_dir):
        os.makedirs(afterPost_dir)
    for data in os.listdir(img_dir):
        if '.npy' in data:
            img = np.load(os.path.join(img_dir, data))
            # img = img * 255
            mask = np.load(os.path.join(mask_dir, data))
            # img = resizeTo(img)
            name = data[:-4]
            print(name)
           
            if img.shape != mask.shape:
                print('ID : {} img : {} mask : {}' .format(name, img.shape, mask.shape))
            mask = postprocessing(mask)
            np.save(os.path.join(afterPost_dir, str(name) + '.npy'), mask)

            result = draw_contour(img, mask)
            np.save(os.path.join(con_dir, str(name) + '.npy'), result)
