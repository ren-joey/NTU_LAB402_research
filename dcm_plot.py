import numpy as np
import matplotlib.pyplot as plt
from plotter import img_plotter
import pydicom as pd

plt.axis('off')

file = np.load(
    '/Users/joey_ren/Desktop/MS/Lab/Research/datasets/l3_dataset.npz',
    allow_pickle=True
)

# ['images_s', 'spacings', 'images_f', 'ydata', 'num_images', 'names']
# folders = file.files

idx = 620

# negative numbers in 2d array
# sagittal CT images
# n=1006
images_s = file['images_s']
img_s = images_s[idx]

# negative numbers in 2d array
# coronal/frontal CT images
# n=1006
images_f = file['images_f']
img_f = images_f[idx]

# numbers in a 1d array
ydata = file['ydata']
ydata = ydata.item()['B']
y = ydata[idx]
img_s[y] = 2048
img_f[y] = 2048

# https://medium.com/analytics-vidhya/how-to-convert-grayscale-dicom-file-to-rgb-dicom-file-with-python-df86ac055bd
# pd.dcmread(img_s)
# img = ds.pixel_array # dtype = uint16
# img = img.astype(float)
# img = img*ds.RescaleSlope + ds.RescaleIntercept


img_plotter([img_s, img_f], ['sagittal', 'frontal'])

# 1006
# num_images = file['num_images']

# file names
# names = file['names']
