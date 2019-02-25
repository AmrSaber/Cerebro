import process_image as pi
import numpy as np
import cv2
if __name__ == '__main__':
    im = cv2.imread("y.jpg")
    im = np.array(im)
    items = pi.extract_faces_emotions(im)
    # print (items)

    im = pi.mark_faces_emotions(items)
    cv2.imshow("detected emotions",im)
    cv2.waitKey(0)
