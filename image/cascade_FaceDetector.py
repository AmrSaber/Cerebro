import cv2

# haarcascade_frontalface_alt.xml scale-factor = 1.00849, min-neighbours = 5
# scale factor of 1.03 - 1.05 for haar seems to give good results


class CascadeClassifier:
	def __init__(self, classifier_type):
		if classifier_type == 'haar':
			self.classifier = cv2.CascadeClassifier('../saved-models/face-detection/haarcascade_frontalface_alt.xml')
			self.scale_factor = 1.0303035
			self.min_neighbours = 5
		elif classifier_type == 'lbp':
			self.classifier = cv2.CascadeClassifier('../saved-models/face-detection/lbpcascade_frontalface_improved.xml')
			self.scale_factor = 1.2
			self.min_neighbours = 5
		elif classifier_type == "none":
			self.classifier = cv2.CascadeClassifier()
			self.scale_factor = -1
			self.min_neighbours = -1
		else:
			raise Exception("unknown classifier type")


class FaceDetector:
	cascade_classifier = CascadeClassifier("none")

	def __init__(self):
		raise Exception("you can't make an object of this class")

	# public functions
	@staticmethod
	def set_classifier(classifier_type='haar'):
		FaceDetector.cascade_classifier = CascadeClassifier(classifier_type)

	@staticmethod
	def get_faces(img):
		"""
		this function is supposed to extract faces from image, set them in required format,
		enhance them, pack them with their respective boxes in original image and return them
		:param img:np.mat the image with faces in it
		:returns res:list [ .., (face, box), ...]
		where each face is an image for the detected face
		and each box is a tuple of two points the upper-left-corner and lower-right-corner
		"""
		boxes = FaceDetector.__get_faces_locations(img)
		# FaceDetector.__display(img.copy(), boxes)
		faces = FaceDetector.__extract_faces(img, boxes)
		faces = FaceDetector.__enhance_faces(faces)
		res = []
		for face, box in zip(faces, boxes):
			res.append((face, box))
		return res

	@staticmethod
	def is_one_face(img):
		"""
		this function checks if the given image contains EXACTLY 1 face
		:param img:
		:return: True if only 1 face is found, False otherwise
		"""
		return len(FaceDetector.__get_faces_locations(img)) == 1

	@staticmethod
	def get_face_landmarks(face):
		"""
		this function should return the locations of the landmarks in the face
		:param face:np.mat image with exactly one face in it
		:return: not-known-yet
		"""
		raise Exception("not implemented yet")

	# private functions
	@staticmethod
	def __get_faces_locations(img):
		"""
		this function applies the haar classifier to the image to detect the faces in it
		:param img:np.mat
		:return:boxes:list of boxes that contain faces in the image
		where each box contains a tuple of two points
		the upper left corner and the lower right corner
		"""
		if FaceDetector.cascade_classifier.scale_factor < 0:
			raise Exception("you should initialize the classifier first")
		faces = FaceDetector.cascade_classifier.classifier.detectMultiScale(
			img,
			scaleFactor=FaceDetector.cascade_classifier.scale_factor,
			minNeighbors=FaceDetector.cascade_classifier.min_neighbours
		)
		boxes = []
		for face in faces:
			p1 = (face[0], face[1])
			p2 = (face[2] + face[0], face[3] + face[1])
			boxes.append((p1, p2))
		return boxes

	@staticmethod
	def __extract_faces(img, boxes):
		"""
		this function should extract the faces in the given locations in the given image
		:param img:np.mat the image with faces in it
		:param boxes:list it contains the locations for faces in the image
		:return:faces:list of np.mat images of faces in the given image
		"""
		faces = []
		for box in boxes:
			p1, p2 = box
			face = img[p1[1]:p2[1], p1[0]:p2[0]]
			faces.append(face)
		return faces

	@staticmethod
	def __enhance_faces(faces):
		"""
		this function puts the image for the face in the correct format
		:param faces: a list of faces detected in the image
		:return: faces: a list of faces after applying the required enhancements
		1 - reshaped to 48 * 48
		2 - turn into greyscale
		"""
		for i in range(len(faces)):
			faces[i] = cv2.resize(faces[i], (48, 48))
			faces[i] = cv2.cvtColor(faces[i], cv2.COLOR_RGB2GRAY)
		return faces

	@staticmethod
	def __display(img, boxes):
		for box in boxes:
			cv2.rectangle(img, box[0], box[1], (0, 255, 0), 1)
		cv2.imshow("detected faces", img)
		cv2.waitKey(0)


if __name__ == '__main__':
	im = cv2.imread("examples/FacesOfDarbya.jpg")
	FaceDetector.set_classifier("haar")
	fs = FaceDetector.get_faces(im)
	for i in fs:
		print(i[1])
		cv2.imshow("face", i[0])
		cv2.waitKey(0)
