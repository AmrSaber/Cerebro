

import cv2

def extract_faces_emotion(image, detector_type):
    if detector_type == 'dlib':
        from image.face_detector import dlib as detector
    elif detector_type =='haar':
        from image.face_detector import haar as detector
    elif detector_type =='lbp':
        from image.face_detector import lbp as detector
    else :
        raise Exception("invalid detector")
    faces = detector.get_faces(image)

    return faces

if __name__ == '__main__':
    im = cv2.imread("y.jpg")
    x = extract_faces_emotion(im,'haar')

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (77, 121, 255)
    emotion = []
    for i in range(len(x)):
        emotion.append('face')
    for i in range(len(x)):
        cv2.rectangle(im,x[i][1][0],x[i][1][1],(66,206,244),2)
        cv2.putText(im,
            emotion[i],
            x[i][1][0]+(0, 2),
            font,
            font_scale,
            font_color,
            cv2.LINE_AA)

    cv2.imshow("detected faces",im)
    cv2.waitKey(0)
