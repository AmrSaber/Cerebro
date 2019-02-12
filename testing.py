

import cv2

def extract_faces_emotion(image, detector_type):
    """
    detector_type >> ('dlib, haar, lbp')
    each item has [face_image, corner_coordinate, emotion]
    faces of size 48*48 , gray scale
    corners coordinates as ()
    """
    item = []
    items= []

    if detector_type == 'dlib':
        from image.face_detector import dlib as detector

    faces = detector.get_faces(image)


    return faces

if __name__ == '__main__':
    im = cv2.imread("y.jpg")
    x = extract_faces_emotion(im,'dlib')
    for i in range(len(x)):
        cv2.rectangle(im,x[i][1][0],x[i][1][1],(66,206,244),2)

    cv2.imshow("detected faces",im)
    cv2.waitKey(0)
