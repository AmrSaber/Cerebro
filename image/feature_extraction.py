#! /user/bin/env python3


# HOG task
# from skimage import io
import cv2
from skimage.feature import hog
from imutils import face_utils
import dlib
# import matplotlib.pyplot as plt

__dlib_landmark_predictor = dlib.shape_predictor("../saved-models/face-landmarks/shape_predictor_68_face_landmarks.dat")

def sk_get_hog(img):
	"""
	img : 48 * 48 gray scale image
	"""
	features, hog_image = hog(
		img,
		orientations=8,
		pixels_per_cell=(12, 12),
		cells_per_block=(4, 4),
		visualize=True,
		transform_sqrt=False,
		feature_vector=True,
		multichannel=False
	)

	# to display
	# io.imshow(hog_image)
	# plt.show()

	return features

def get_face_landmarks(img):
	rect = dlib.rectangle(0, 0, img.shape[0] - 1, img.shape[1] -1)
	shape = __dlib_landmark_predictor(img, rect)
	shape = face_utils.shape_to_np(shape)
	return shape
