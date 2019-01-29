from imutils import face_utils
import numpy as np
import cv2
import dlib


class FaceDetector:
	"""
	this class is responsible for detecting faces in an image
	it should provide a method that p=given an image
	it should return a set of locations for faces inside that image
	"""
	# DNN parameters
	folder_path = '../saved-models/image-dnn/'
	net = cv2.dnn.readNetFromCaffe(
		folder_path + 'deploy.prototxt.txt',
		folder_path + 'res10_300x300_ssd_iter_140000.caffemodel'
	)

	# face landmarks extractor
	landmark_predictor = dlib.shape_predictor(
		'../saved-models/face-landmarks/shape_predictor_68_face_landmarks.dat'
	)
	# significance lvl
	alpha = 0.85

	def __init__(self):
		raise Exception("you can't make an object of this class")

	# public functions
	@staticmethod
	def get_faces(img):
		if img.shape[2] == 1:
			img = img.repeat(3, axis=-1)
		return FaceDetector.__extract_faces(img, FaceDetector.__detect(img))

	@staticmethod
	def is_one_face(img):
		if img.shape[2] == 1:
			img = img.repeat(3, axis=-1)
		return len(FaceDetector.__detect(img)) == 1

	@staticmethod
	def get_face_landmarks(face):
		if len(face.shape) == 2:
			face = face.reshape((face.shape[0], face.shape[1], 1))
		if face.shape[2] == 3:
			face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
		rect = [(0, 0), (face.shape[0], face.shape[1])]
		landmarks = FaceDetector.landmark_predictor(face, rect)
		landmarks = face_utils.shape_to_np(landmarks)
		return landmarks

	# private functions
	@staticmethod
	def __detect(img):
		"""
		this function is supposed to detect the faces in a given image
		and return a list of locations for those faces
		:param img:
		:return:
		"""
		locations = []
		(h, w) = img.shape[:2]
		blob = cv2.dnn.blobFromImage(
			cv2.resize(img, (300, 300)),
			1.0,
			(300, 300),
			(104.0, 177.0, 123.0)
		)
		FaceDetector.net.setInput(blob)
		detections = FaceDetector.net.forward()

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > FaceDetector.alpha:
				box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
				(startX, startY, endX, endY) = box
				# make sure the size of detected face is not 0
				if startX != endX and startY != endY:
					locations.append(box)

		# display the detected faces marked on the image
		# FaceDetector.display(img, locations)
		return locations

	@staticmethod
	def __extract_faces(img, locations):
		faces = []
		for box in locations:
			(startX, startY, endX, endY) = box
			face = img[startY:endY, startX:endX].copy()
			print("face shape : " + str(face.shape))
			face = cv2.resize(face, (48, 48))
			# face = imutils.resize(face, height=48, width=48)
			print("face shape : " + str(face.shape))
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			faces.append(face)
		return faces

	@staticmethod
	def __display(img, locations):
		# put a rectangle around all faces
		for i in range(len(locations)):
			(startX, startY, endX, endY) = locations[i]
			cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 255), 2)

		# show the output image
		cv2.imwrite('sample_out_2.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		cv2.imshow("Output", img)
		cv2.waitKey(0)


if __name__ == '__main__':
	im = cv2.imread('side face.jpg')
	faces = FaceDetector.get_faces(im)
	the_face = faces[0]
	print(FaceDetector.get_face_landmarks(the_face))
	for i in faces:
		cv2.imshow("face", i)
		cv2.waitKey(0)
