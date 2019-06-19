import numpy as np
import cv2, time
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
	face = cv2.resize(face, (48, 48))
	face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
	return face

def __display(img, boxes):
		for box in boxes:
			cv2.rectangle(img, box[0], box[1], (0, 255, 0), 1)
		cv2.imwrite("result.jpg", img)
		cv2.imshow("detected faces", img)
		cv2.waitKey(0)

if __name__ == '__main__':
	img = cv2.imread("example1.jpg")
	t1 = time.time()	
	out = get_faces(img)
	print("execution time : %f\n" %(time.time() - t1))
	boxes = []
	for face in out:
		boxes.append(face[1])
	__display(img, boxes)
	
