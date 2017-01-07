import cv2
import numpy as np
import scipy
from scipy import ndimage
import argparse
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

data_folder = "../data/seminar_23_10_14/images/"
results_path = "results/"

def drawMatches(img1, kp1, img2, kp2, matches):
	# Create a new output image that concatenates the two images together
	# (a.k.a) a montage
	rows1, cols1 = img1.shape
	rows2, cols2 = img2.shape

	out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

	# Place the first image to the left
	out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])

	cv2.imwrite("result.jpg", out)


def readDir(directory):
	source_files = [f for f in listdir(directory) if isfile(
		join(directory, f))]
	images = []
	for source_name in source_files:
		source_path = directory + source_name
		print(source_path)
		img = cv2.imread(source_path)
		images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	return images


def show(img):
	cv2.imshow("image", img)
	cv2.waitKey()


def pano(images):
	sift = cv2.xfeatures2d.SIFT_create()
	key_points = []
	img_des = []
	for image in images:
		kp, d = sift.detectAndCompute(image, None)
		print(len(kp))
		key_points.append(kp)
		img_des.append(d)

	bf = cv2.BFMatcher()
	obj = None
	scene = images[0]
	matches = [[[] for i in range(len(images))] for j in range(len(images))]
	good = [[[] for i in range(len(images))] for j in range(len(images))]

	# for i in range(len(images)):
	# 	for j in range(len(images)):
	# 		if i == j:
	# 			continue
	# 		matches[i][j] = bf.knnMatch(img_des[i], img_des[j], k=2)
	# 		print("Found matches for " + str(i) + " and " + str(j))
	# 		# Apply ratio test
	# 		for m, n in matches[i][j]:
	# 			if m.distance < 0.7 * n.distance:
	# 				good[i][j].append(m)
	# 		print(len(good[i][j]))

	# sum_len = [None for i in range(len(images))]
	# for i in range(len(images)):
	# 	sum_len[i] = sum([len(good[i][j]) for j in range(len(images))])
	# 	print(sum_len[i])
	# max_index = 0
	# for i in range(len(images)):
	# 	if sum_len[max_index] < sum_len[i]:
	# 		max_index = i
	# scene = images[max_index]
	# print(max_index)

	max_index = 1
	scene = images[max_index]
	for i in range(len(images)):
		if i == max_index:
			continue

		matches[i][max_index] = bf.knnMatch(img_des[i], img_des[max_index], k=2)
		# print("Found matches for " + str(i) + " and " + str(j))
		# Apply ratio test
		for m, n in matches[i][max_index]:
			if m.distance < 0.7 * n.distance:
				good[i][max_index].append(m)

		obj = images[i]
		obj_pts = np.float32(
			[key_points[i][m.queryIdx].pt for m in good[i][max_index]]).reshape(-1, 1, 2)
		scene_pts = np.float32(
			[key_points[max_index][m.trainIdx].pt for m in good[i][max_index]]).reshape(-1, 1, 2)

		# print(len(obj_pts))
		# print(len(scene_pts))
		M, mask = cv2.findHomography(scene_pts, obj_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		# print(M)
		# h,w = images[0].shape
		# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		# dst = cv2.perspectiveTransform(pts, M)
		# print(dst)
		# np.linalg.inv
		rows, cols = scene.shape
		mult = 1
		result = cv2.warpPerspective(obj, np.linalg.inv(M), (cols*(mult+1), rows*(mult+1)))
		
		top = int(0*rows)
		bottom = int(mult*rows)
		left = int(0*cols)
		right = int(mult*cols)
		tmp = cv2.copyMakeBorder(scene, top, bottom, left, right, cv2.BORDER_CONSTANT, 0) 
		print(result.shape)
		print(tmp.shape)
		cv2.imwrite("transformed" + str(i) + ".jpg", result)
		result += tmp
		cv2.imwrite("result" + str(i) + ".jpg", result)

images = readDir(data_folder)
pano(images)
