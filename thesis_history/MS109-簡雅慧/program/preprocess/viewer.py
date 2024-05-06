import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
import numpy as np
import os

def load_mhd_file(filename):
	print(os.path.isfile(filename))
	itkimage = sitk.ReadImage(filename)
	numpyImage = sitk.GetArrayFromImage(itkimage)
	return numpyImage

### read npy

# imgs = np.load('D:/3D_ABUS/data_128/02020.npy')
imgs = np.load('D:/new/benign/1_1.3.12.2.1107.5.5.2.202964.30000015061506003931200000016/1.3.12.2.1107.5.5.2.202964.30000015061506003931200000016.npy')  

### read npz
# a = np.load('C:/Users/zhang/Desktop/3DAnimation/split_seg3.npz')
# imgs = a['img'][0]

print(imgs.shape)
print(np.min(imgs), np.max(imgs))

def previous_slice(ax):
	volume = ax.volume
	ax.index = (ax.index - 1) % volume.shape[0]
	ax.images[0].set_array(volume[ax.index])


def next_slice(ax):

	volume = ax.volume
	ax.index = (ax.index + 1) % volume.shape[0]
	ax.images[0].set_array(volume[ax.index])


def process_key(event):

	fig = event.canvas.figure
	ax = fig.axes[0]
	#print(event.key)
	if event.key == 'a' or event.key == 'up':
		previous_slice(ax)
	elif event.key == 'd' or event.key == 'down':
		next_slice(ax)

	fig.canvas.draw()


def multi_slice_viewer(volume):

	fig, ax = plt.subplots()
	plt.axis('off')
	ax.volume = volume
	ax.index = volume.shape[0]//2
	ax.imshow(volume[ax.index], cmap='gray')
	# ax.imshow(volume, cmap='gray')   # 2D
	fig.canvas.mpl_connect('key_press_event', process_key)
	plt.show()



multi_slice_viewer(imgs)
