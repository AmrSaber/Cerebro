
import sys
import dlib


def main():
    if len(sys.argv) < 3:
        print(
            "Call this program like this:\n"
            "   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
            "You can get the mmod_human_face_detector.dat file from:\n"
            "    http://dlib.net/files/mmod_human_face_detector.dat.bz2")
        exit()

    cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])
    win = dlib.image_window()

    for f in sys.argv[2:]:
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        dets = cnn_face_detector(img, 1)
        '''
        This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
        These objects can be accessed by simply iterating over the mmod_rectangles object
        The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.

        It is also possible to pass a list of images to the detector.
            - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

        In this case it will return a mmod_rectangless object.
        This object behaves just like a list of lists and can be iterated over.
        '''
        print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

        rects = dlib.rectangles()
        rects.extend([d.rect for d in dets])

        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(rects)
        dlib.hit_enter_to_continue()

if __name__=="__main__":
    main()
