import process_image as pi
import cv2
if __name__ == '__main__':
    im = cv2.imread("index.jpg")
    items = pi.extract_faces_emotions(im, 'haar')
    print (items)

    im = pi.mark_faces_emotions(items)
    cv2.imshow("detected emotions",image)
    cv2.waitKey(0)
