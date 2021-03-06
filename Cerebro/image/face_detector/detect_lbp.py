from Cerebro.image.face_detector.__cascade_face_detector import cascade_FaceDetector

__lbp_facedetector = cascade_FaceDetector("lbp")

def get_faces(img):
    return __lbp_facedetector.get_faces(img)

def is_one_face(img):
    return __lbp_facedetector.is_one_face(img)
