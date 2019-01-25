from haar_FaceDetector import FaceDetector
from HOG import sk_get_hog
import cv2


def detect_faces(image):
	"""
	:param: image: np.Mat
	:retruns: list of tuples each of which contains
	1 - image of detected face
	2 - box of detected face as (upper-left-corner, lower-right-corner)
	constraints:
	1 - size 48 * 48
	2 - gray image
	"""
	return FaceDetector.get_faces(image)


def get_features(image):
	"""
	:param: image: np.Mat
	:returns: (landmarks, HOG)
	"""
	# here we used the HOG function provided by skimage
	hog = sk_get_hog(image)
	# face land marks should be ready soon
	landmarks = FaceDetector.get_face_landmarks(image)
	return landmarks, hog


def is_face(image):
	"""
	:param: np.Mat
	:returns : bool if only 1 face is in the image
	"""
	return FaceDetector.is_one_face(image)


def normalise_face(image):
	"""
	:param: np.Mat
	:return: np.Mat
	"""
	# no enhancement done yet
	return image 
