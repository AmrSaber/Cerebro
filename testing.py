import process_image as pi
if __name__ == '__main__':
    im = cv2.imread("b.jpg")
    items = pi.extract_faces_emotions(im)
    print (items)

    im = pi.mark_faces_emotions(items)
    cv2.imshow("detected emotions",image)
    cv2.waitKey(0)
