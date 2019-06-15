#! /user/bin/env python3

import cv2
from skimage.feature import hog
from imutils import face_utils
import dlib

__dlib_landmark_predictor = dlib.shape_predictor("./saved-models/face-landmarks/shape_predictor_68_face_landmarks.dat")

def sk_get_hog(img, orientations=8, pixels_per_cell=(12, 12)):
	"""
	img : 48 * 48 gray scale image
	"""
	features, hog_image = hog(
		img,
		orientations=orientations,
		pixels_per_cell=pixels_per_cell,
		cells_per_block=(4, 4),
		visualize=True,
		transform_sqrt=False,
		feature_vector=True,
		multichannel=False,
		block_norm='L2'
	)

	return features

def get_face_landmarks(img):
	rect = dlib.rectangle(0, 0, img.shape[0] - 1, img.shape[1] -1)
	shape = __dlib_landmark_predictor(img, rect)
	shape = face_utils.shape_to_np(shape)
	return shape

def hog_with_sliding_window(img, window_step=6, orientations=8, pixels_per_cell=(12, 12)):
	"""
	window_step = 6
	img 48*48 gray scale
	-------------------
	orientations = 8
	pixels_per_cell = (12, 12) >> output length : 1152
	pixels_per_cell = (8, 8) >> output length : 2592 <<<<<
	-------------------
	orientations = 12
	pixels_per_cell = (12, 12) >>output length : 1728
	pixels_per_cell = (12, 12) >>output length : 3888
	"""
	window_size = 24
	hog_windows =[]
	for y in range(0, img.shape[0], window_step):
		for x in range(0, img.shape[1], window_step):
			window = img[y:y+window_size, x:x+window_size]
			hog_windows.extend(hog(
			window,
			orientations=orientations,
			pixels_per_cell=pixels_per_cell,
			cells_per_block=(1, 1),
			block_norm='L2-Hys',
			visualize=False
			))
	return hog_windows
