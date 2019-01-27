import cv2
import dlib
import time

image = cv2.imread("examples/FacesOfDarbya.jpg")
detector = dlib.get_frontal_face_detector()
t1 = time.time()
rects = detector(image, 2)
dt = time.time() - t1
print(dt)

for (i, rect) in enumerate(rects):
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	p1 = rect.tl_corner()
	p2 = rect.br_corner()
	cv2.rectangle(image, (p1.x, p1.y), (p2.x, p2.y), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (p1.x - 10, p1.y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)