"""
missing work:
1 - landmark detection module
2 - enhancement to the image

dnn_FaceDetector to use DNN based face detection
cascade_FaceDetector to use cascade classifiers like haar and lbp
dlib_FaceDetector to use dlib face detector HOG + SVM

next step will be creating a general interface for all face detection methods
to allow changing throw them all via programming function
"""

from dlib_FaceDetector import FaceDetector
from HOG import sk_get_hog

# uncomment this function when using cascade_FaceDetector
# def set_detection_method(method="haar"):
# 	"""
# 	this method is sed to specify the detection method
# 	:param method: string
# 	possible values {
# 		"haar" > for haar classifier
# 		"lbp"  > for lbp classifier
# 	}
# 	:return: nothing
# 	:raise Exception "unknown type" if another type is entered
# 	"""
# 	FaceDetector.set_classifier(method)


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
	:param: image: np.Mat (the image must contain only the face extracted)
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


def normalize_face(image):
	"""
	:param: np.Mat
	:return: np.Mat
	"""
	# no enhancement done yet
	return image
