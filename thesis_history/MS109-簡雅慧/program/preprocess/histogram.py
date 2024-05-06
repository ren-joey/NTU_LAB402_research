import numpy as np
import os


def histogram(img):
	table = [0 for i in range(256)]
	z, x, y = img.shape
	print(img.shape)
	
	for zi in range(z):
		for xi in range(x):
			for yi in range(y):
				table[int(img[zi, xi, yi])] += 1

	# cdf
	for i in range(1, 256):
		table[i] = table[i-1] + table[i]

	minValue = np.min(table)
	maxValue = np.max(table)
	#print(minValue, maxValue)

	for zi in range(z):
		for xi in range(x):
			for yi in range(y):
				img[zi, xi, yi] = (table[int(img[zi, xi, yi])] - minValue) / (z * x * y - minValue) * 255
				
	return img
