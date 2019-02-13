

import cv2

def extract_faces_emotion(image, detector_type  = 'dlib'):
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
    im = cv2.imread("b.jpg")
    x = extract_faces_emotion(im)

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    box_color = (0, 255, 255)
    text_color = (0, 0, 0)
    emotion = []
    for i in range(len(x)):
        emotion.append('face')
    for i in range(len(x)):
        im = cv2.rectangle(im,x[i][1][0],x[i][1][1],box_color,1)
        y = (x[i][1][0][0]-2, x[i][1][0][1]-5)
        im = cv2.putText(im,
            emotion[i],
            y,
            font,
            font_scale,
            text_color,
            2)
    cv2.imshow("detected faces",im)
    cv2.waitKey(0)
    """
    x[i] >> item (face, corners(topright,bottomleft), emotion)
    x[i][1] >> corners
    x[i][1][0] >> topright
    """
