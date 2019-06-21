import numpy as np
import cv2
import dlib

__dlib_detector = dlib.get_frontal_face_detector()

# public interface
def get_faces(img):
	rects = __dlib_detector(img, 2)
	res = []
	for rect in rects:
		face = _extract_face(img, rect)
		face = _enhance_face(face)
		p1 = rect.tl_corner()
		p2 = rect.br_corner()
		box = [(p1.x, p1.y), (p2.x, p2.y)]
		res.append((face, box))
	return res

def is_one_face(img):
	return len(__dlib_detector(img, 2)) == 1

# private methods
def _extract_face(img, rect):
	p1 = rect.tl_corner()
	p2 = rect.br_corner()
	face = img[p1.y:p2.y, p1.x:p2.x]
	return face

def _enhance_face(face):
	return face
	face = cv2.resize(face, (48, 48))
	face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
	return face


if __name__ == '__main__':
	im = cv2.imread('y.jpg')
	cv2.imshow("title", im)
	cv2.waitKey(0)
	res = get_faces(im)
	for i in res:
		cv2.imshow("df", i[0])
		cv2.waitKey(0)
