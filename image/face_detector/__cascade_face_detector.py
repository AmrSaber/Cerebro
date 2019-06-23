import cv2

# haarcascade_frontalface_alt.xml scale-factor = 1.00849, min-neighbours = 5
# scale factor of 1.03 - 1.05 for haar seems to give good results

class cascade_FaceDetector:
	def __init__(self, classifier_type):
		self.min_neighbors = 5
		if classifier_type == 'haar':
			self.classifier = cv2.CascadeClassifier('./saved-models/face-detection/haarcascade_frontalface_alt.xml')
			self.scale_factor = 1.0303035
		elif classifier_type == 'lbp':
			self.classifier = cv2.CascadeClassifier('./saved-models/face-detection/lbpcascade_frontalface_improved.xml')
			self.scale_factor = 1.2
		else:
			raise Exception("unknown classifier type")


	def get_faces(self, img):
		"""
		this function is supposed to extract faces from image, set them in required format,
		enhance them, pack them with their respective boxes in original image and return them
		:param img:np.mat the image with faces in it
		:returns res:list [ .., (face, box), ...]
		where each face is an image for the detected face
		and each box is a tuple of two points the upper-left-corner and lower-right-corner
		"""
		boxes = self.__get_faces_locations(img)
		res = []
		for box in boxes:
			face = self.__extract_face(img, box)
			# face = self.__enhance_face(face)
			res.append((face, box))
		return res

	def is_one_face(self, img):
		"""
		this function checks if the given image contains EXACTLY 1 face
		:param img:
		:return: True if only 1 face is found, False otherwise
		"""
		return len(self.__get_faces_locations(img)) == 1

	# private functions
	def __get_faces_locations(self, img):
		"""
		this function applies the haar classifier to the image to detect the faces in it
		:param img:np.mat
		:return:boxes:list of boxes that contain faces in the image
		where each box contains a tuple of two points
		the upper left corner and the lower right corner
		"""
		faces = self.classifier.detectMultiScale(
			img,
			scaleFactor=self.scale_factor,
			minNeighbors=self.min_neighbors
		)
		boxes = []
		for face in faces:
			p1 = (face[0], face[1])
			p2 = (face[2] + face[0], face[3] + face[1])
			boxes.append((p1, p2))
		return boxes

	def __extract_face(self, img, box):
		"""
		this function should extract the faces in the given locations in the given image
		:param img:np.mat the image with faces in it
		:param boxes:list it contains the locations for faces in the image
		:return:faces:list of np.mat images of faces in the given image
		"""
		p1, p2 = box
		face = img[p1[1]:p2[1], p1[0]:p2[0]]
		return face

	@staticmethod
	def __display(img, boxes):
		for box in boxes:
			cv2.rectangle(img, box[0], box[1], (0, 255, 0), 1)
		cv2.imshow("detected faces", img)
		cv2.waitKey(0)
