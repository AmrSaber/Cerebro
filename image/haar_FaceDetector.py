import numpy as np
import cv2


class FaceDetector:
	classifier = cv2.CascadeClassifier('../saved-models/image-dnn/haarcascade_frontalface_alt.xml')

	def __init__(self):
		raise Exception("you can't make an object of this class")

	# public functions
	@staticmethod
	def get_faces(img):
		# get locations for faces
		# extract faces
		boxes = FaceDetector.__get_faces_locations(img)
		faces = FaceDetector.__extract_faces(img, boxes)
		faces = FaceDetector.__enhance_faces(faces)
		res = []
		for face, box in zip(faces, boxes):
			res.append((face, box))
		return res

	@staticmethod
	def is_one_face(img):
		return len(FaceDetector.__get_faces_locations(img)) == 1

	@staticmethod
	def get_face_landmarks(face):
		raise Exception("not implemented yet")

	# private functions
	@staticmethod
	def __get_faces_locations(img):
		faces = FaceDetector.classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
		for i in range(len(faces)):
			faces[i][2] += faces[i][0]
			faces[i][3] += faces[i][1]
		return faces

	@staticmethod
	def __extract_faces(img, boxes):
		faces = []
		for i in boxes:
			x1, y1, x2, y2 = i
			face = img[y1:y2, x1:x2]
			faces.append(face)
		return faces

	@staticmethod
	def __enhance_faces(faces):
		for i in range(len(faces)):
			faces[i] = cv2.resize(faces[i], (48, 48))
			faces[i] = cv2.cvtColor(faces[i], cv2.COLOR_RGB2GRAY)
		return faces


if __name__ == '__main__':
	im = cv2.imread("examples/example_01.jpg")
	fs = FaceDetector.get_faces(im)
	for i in fs:
		print(i[1])
		cv2.imshow("face", i[0])
		cv2.waitKey(0)
