from imutils import face_utils
import cv2
import dlib


class FaceDetector:
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("../saved-models/face-landmarks/shape_predictor_68_face_landmarks.dat")

	def __init__(self):
		raise Exception("you can't make an object of this class")

	# pubic methods
	@staticmethod
	def get_faces(img):
		rects = FaceDetector.detector(img, 2)
		faces = FaceDetector.__extract_faces(img, rects)
		faces = FaceDetector.__enhance_faces(faces)
		res = []
		for face, rect in zip(faces, rects):
			p1 = rect.tl_corner()
			p2 = rect.br_corner()
			box = [(p1.x, p1.y), (p2.x, p2.y)]
			res.append((face, box))
		return res

	@staticmethod
	def is_one_face(img):
		return len(FaceDetector.detector(img, 2)) == 1

	@staticmethod
	def get_face_landmarks(img):
		rect = FaceDetector.detector(img, 2)[0]
		shape = FaceDetector.predictor(img, rect)
		shape = face_utils.shape_to_np(shape)
		return shape

	# private methods
	@staticmethod
	def __extract_faces(img, rects):
		faces = []
		for rect in rects:
			p1 = rect.tl_corner()
			p2 = rect.br_corner()
			face = img[p1.y:p2.y, p1.x:p2.x]
			faces.append(face)
		return faces

	@staticmethod
	def __enhance_faces(faces):
		for i in range(len(faces)):
			faces[i] = cv2.resize(faces[i], (48, 48))
			faces[i] = cv2.cvtColor(faces[i], cv2.COLOR_RGB2GRAY)
		return faces


if __name__ == '__main__':
	im = cv2.imread("examples/FacesOfDarbya.jpg")
	in_image_faces = FaceDetector.get_faces(im)
	some_face = in_image_faces[0][0]
	cv2.imshow("face", some_face)
	cv2.waitKey(0)
	print(FaceDetector.get_face_landmarks(some_face))
