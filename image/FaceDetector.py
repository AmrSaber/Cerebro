import numpy as np
import cv2
import imutils


class FaceDetector:
	"""
	this class is responsible for detecting faces in an image
	it should provide a method that p=given an image
	it should return a set of locations for faces inside that image
	"""
	# DNN parameters
	folder_path = '../saved-models/image-dnn/'
	net = cv2.dnn.readNetFromCaffe(folder_path + 'deploy.prototxt.txt', folder_path + 'res10_300x300_ssd_iter_140000.caffemodel')

	# significance lvl
	alpha = 0.85

	def __init__(self):
		raise Exception("you can't make an object of this class")

	@staticmethod
	def detect(img):
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
	def extract_faces(img, locations):
		faces = []
		for box in locations:
			(startX, startY, endX, endY) = box
			face = img[startY:endY, startX:endX].copy()
			print(face.shape)
			# face = cv2.resize(face, (48, 48))
			face = imutils.resize(face, height=48, width=48)
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			faces.append(face)
		return faces

	@staticmethod
	def display(img, locations):
		# put a rectangle around all faces
		for i in range(len(locations)):
			(startX, startY, endX, endY) = locations[i]
			cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 255), 2)

		# show the output image
		cv2.imwrite('sample_out_2.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		cv2.imshow("Output", img)
		cv2.waitKey(0)

	@staticmethod
	def get_faces(img):
		return FaceDetector.extract_faces(img, FaceDetector.detect(img))

	@staticmethod
	def is_one_face(img):
		return len(FaceDetector.detect(img)) == 1


if __name__ == '__main__':
	im = cv2.imread('img.png')
	faces = FaceDetector.get_faces(im)
	for i in faces:
		cv2.imshow("face", i)
		cv2.waitKey(0)
