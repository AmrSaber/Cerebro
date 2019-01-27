
# Description
#image should Be
#1-centered in the image.
#2-rotated that such the eyes lie on a horizontal line (i.e., the face is rotated such that the eyes lie along the same y-coordinates).
#3-scaled such that the size of the faces are approximately identical.

#align face is a funtion depend on landmark(68) and dlib image extraction

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
#import cv2
#import uuid


#image = cv2.imread("input.png")
#f = str(uuid.uuid4())


#send face rectangle
def align_faces(image):
    #initialize dlib detector && landmarks Predictor && FaceAligner class
    detector = dlib.get_frontal_face_detector()
    pred = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(pred)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #detect faces in gray scale
    rects = detector(gray,2)
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

    #to display the output images
    # cv2.imwrite("foo/" + f + ".png", faceAligned)
    # cv2.imshow("Original", faceOrig)
    # cv2.imshow("Aligned", faceAligned)
    # cv2.waitKey(0)
    return faceAligned
