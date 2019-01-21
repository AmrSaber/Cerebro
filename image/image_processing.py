from FaceDetector import FaceDetector

def detect_faces(image):
	"""
	:param: image: np.Mat
	:retruns: list of images of detected faces
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
	# HOG is ready and will be uploaded soon
	# face land marks should be ready soon
	raise NotImplementedError("this function logic is not implemented yet")

def is_face(image):
	"""
	:param: np.Mat
	:returns : bool if only 1 face is in the image
	"""
	return len(detect_faces(image)) == 1


def normalise_face(image):
	"""
	:param: np.Mat
	:return: np.Mat
	"""
	# no enhancement done yet
	return image 
