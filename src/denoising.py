import numpy as np
import cv2
from matplotlib import pyplot as plt

src_dir = "../data/"
dst_dir = "../denoised_images/"
paths = ["Lena01.jpg", "Lena02.jpg", "Lena03.jpg", "Lena04.jpg", "Lena05.jpg", "Lena06.GIF", "Lena07.GIF"]

def gauss(img, ksize, s):
	return cv2.GaussianBlur(img, None, ksize, s, s)

def fastNlMeans(img, is_show, strength, templateWindowSize, searchWindowSize):

	dst = cv2.fastNlMeansDenoisingColored(img, None, strength, strength, templateWindowSize, searchWindowSize)

	b, g, r = cv2.split(img)
	src_show = cv2.merge([r, g, b])
	b, g, r = cv2.split(dst)
	dst_show = cv2.merge([r, g, b])

	if is_show:
		show(src_show, dst_show)

	return dst

def median(img, is_show, ksize = 5):
	dst = cv2.medianBlur(img, ksize)
	if is_show:
		show(img, dst)
	return dst

def show(rgb_img, rgb_dst):
	plt.subplot(211), plt.imshow(rgb_img)
	plt.subplot(212), plt.imshow(rgb_dst)
	plt.show()

def sharpening(img, _type):
	kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
	kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
	                             [-1,2,2,2,-1],
	                             [-1,2,8,2,-1],
	                             [-1,2,2,2,-1],
	                             [-1,-1,-1,-1,-1]]) / 8.0
	kernels = [kernel_sharpen_1, kernel_sharpen_2, kernel_sharpen_3]

	output = cv2.filter2D(img, -1, kernels[_type])
	return output

is_show = False

# Lena01
src = cv2.imread(src_dir + paths[0])
img = src
img = fastNlMeans(img, is_show, 15, 7, 21)
img = np.concatenate((src, img), axis=1)
cv2.imwrite(dst_dir + paths[0], img)

# Lena02
src = cv2.imread(src_dir + paths[1])
img = src
img = fastNlMeans(img, is_show, 20, 7, 21)
img = sharpening(img, 2)
img = np.concatenate((src, img), axis=1)
cv2.imwrite(dst_dir + paths[1], img)

# Lena03
src = cv2.imread(src_dir + paths[2])
img = src
img = sharpening(img, 2)
img = sharpening(img, 2)
img = fastNlMeans(img, is_show, 10, 7, 21)
# img = median(img, is_show)
img = np.concatenate((src, img), axis=1)
cv2.imwrite(dst_dir + paths[2], img)

# Lena04
src = cv2.imread(src_dir + paths[3])
img = src
img = median(img, is_show)
img = np.concatenate((src, img), axis=1)
cv2.imwrite(dst_dir + paths[3], img)

# Lena05
src = cv2.imread(src_dir + paths[4])
img = src
img = median(img, is_show, 11)
img = sharpening(img, 2)
img = gauss(img, 5, 1)
img = sharpening(img, 2)
img = median(img, is_show, 11)
img = sharpening(img, 2)
img = sharpening(img, 2)
img = np.concatenate((src, img), axis=1)
cv2.imwrite(dst_dir + paths[4], img)

# Lena06
# grey = cv2.imread(src_dir + paths[5])
# grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
# src = np.zeros_like(grey)
# src[:,:,0] = grey
# src[:,:,1] = grey
# src[:,:,2] = grey
# img = src
# img = np.concatenate((src, img), axis=1)
# cv2.imwrite(dst_dir + paths[5], img)
