import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import os

def load_mhd_file(filename):
	print(os.path.isfile(filename))
	itkimage = sitk.ReadImage(filename)
	numpyImage = sitk.GetArrayFromImage(itkimage)
	return numpyImage

### read npy
imgs_1 = np.load('D:/賴以尊/賴以尊/data/3D/img/split_0/3_804_0.npy')
imgs_2 = np.load('D:/賴以尊/賴以尊/data/seg_3D/403_mask_gt/split_0/mask_3_804_0.npy')

print(imgs_1.shape, np.min(imgs_1), np.max(imgs_1))
print(imgs_2.shape, np.min(imgs_2), np.max(imgs_2))

def previous_slice(ax1, ax2):
	volume1 = ax1.volume1
	ax1.index = (ax1.index - 1) % volume1.shape[0]
	ax1.images[0].set_array(volume1[ax1.index])

	volume2 = ax2.volume2
	ax2.index = (ax2.index - 1) % volume2.shape[0]
	ax2.images[0].set_array(volume2[ax2.index])


def next_slice(ax1, ax2):

	volume1 = ax1.volume1
	ax1.index = (ax1.index + 1) % volume1.shape[0]
	ax1.images[0].set_array(volume1[ax1.index])

	volume2 = ax2.volume2
	ax2.index = (ax2.index + 1) % volume2.shape[0]
	ax2.images[0].set_array(volume2[ax2.index])



def process_key(event):

	fig = event.canvas.figure
	ax1 = fig.axes[0]
	ax2 = fig.axes[1]
	#print(event.key)
	if event.key == 'a' or event.key == 'up':
		previous_slice(ax1, ax2)
	elif event.key == 'd' or event.key == 'down':
		next_slice(ax1, ax2)

	fig.canvas.draw()


def multi_slice_viewer(volume1, volume2):

	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.volume1 = volume1
	ax1.index = volume1.shape[0]//2
	ax1.imshow(volume1[ax1.index], cmap='gray')
	ax2.volume2 = volume2
	ax2.index = volume2.shape[0]//2
	ax2.imshow(volume2[ax2.index], cmap='gray')

	fig.canvas.mpl_connect('key_press_event', process_key)
	plt.show()

multi_slice_viewer(imgs_1, imgs_2)

