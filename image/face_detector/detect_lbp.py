from cascade_face_detector import cascade_FaceDetector
import cv2, time

__lbp_facedetector = cascade_FaceDetector("lbp")

def get_faces(img):
    return __lbp_facedetector.get_faces(img)

def is_one_face(img):
    return __lbp_facedetector.is_one_face(img)

if __name__ == '__main__':
	img = cv2.imread("example.jpg")
	t1 = time.time()	
	out = get_faces(img)
	print("execution time : %f\n" %(time.time() - t1))
	boxes = []
	for face in out:
		boxes.append(face[1])
	cascade_FaceDetector._cascade_FaceDetector__display(img, boxes)
