import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.fftpack import ifft2, fft2, fftshift, ifftshift

src_path = "../data/saturn.jpg"
results_path = "results/"

img = cv2.imread(src_path, 0)
f = fft2(img)
fshift = fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.savefig(results_path + "Magnitude_Spectrum.png")

def filter_image(shifted, size, low_freq=True):
	rows, cols = img.shape
	crow, ccol = int(rows/2), int(cols/2)
	if low_freq:
		shifted[crow-size:crow+size, ccol-size:ccol+size] = 0
	else:
		shifted[:crow-size] = 0
		shifted[crow+size:] = 0
		shifted[:, :ccol-size] = 0
		shifted[:, ccol+size:] = 0
	f_ishift = ifftshift(shifted)
	img_back = ifft2(f_ishift)
	img_back = np.abs(img_back)

	plt.subplot(121),plt.imshow(img, cmap = 'gray')
	plt.title(""), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
	f_name = None
	if low_freq:
		plt.title('Removed low frequency content: %dpx'%(size)), plt.xticks([]), plt.yticks([])
		f_name = "removed_low_f_%dpx"%(size)
	else:
		plt.title('Removed high frequency content: keep %dpx low freq'%(size)), plt.xticks([]), plt.yticks([])
		f_name = "removed_high_f_%dpx"%(size)
	plt.savefig(results_path + f_name)

for s in range(1, 11):
	filter_image(np.copy(fshift), s, True)
	filter_image(np.copy(fshift), s, False)