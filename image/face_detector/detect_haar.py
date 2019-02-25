from image.face_detector.__cascade_face_detector import cascade_FaceDetector


__haar_facedetector = cascade_FaceDetector("haar")

def get_faces(img):
    return __haar_facedetector.get_faces(img)

def is_one_face(img):
    return __haar_facedetector.is_one_face(img)
