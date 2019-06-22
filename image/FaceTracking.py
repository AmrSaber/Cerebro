import cv2
import dlib
import numpy as np

def faceTracking(frames, framesNumber):

    currentFaceID = -1
    frameCounter = -1

    nones = [None for i in range(framesNumber)]
    faces = []
    cords = []
    faceTrackers = {}

    # Loop over all faces and check if the area for this
    # face is the largest so far
    # We need to convert it to int here because of the
    # requirement of the dlib tracker. If we omit the cast to
    # int here, you will get cast errors since the detector
    # returns numpy.int32 and the tracker requires an int
    for (frame, croppedFaces) in frames:
        frameCounter += 1
        # frame = cv2.resize(frame, (320, 240))
        for ((_x1, _y1), (_x2, _y2)) in croppedFaces:
            x = int(_x1)
            y = int(_y1)
            w = int(abs(_x1-_x2))
            h = int(abs(_y1-_y2))

            # calculate the centerpoint
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            matchedFid = False

            # Now loop over all the trackers and check if the
            # centerpoint of the face is within the box of a
            # tracker
            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[fid].update(frame)
                if trackingQuality < 7:
                    fidsToDelete.append(fid)
                    continue

                tracked_position = faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                # calculate the centerpoint
                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                # check if the centerpoint of the face is within the
                # rectangleof a tracker region. Also, the centerpoint
                # of the tracker region must be within the region
                # detected as a face. If both of these conditions hold
                # we have a match
                if ((t_x <= x_bar <= (t_x + t_w)) and
                        (t_y <= y_bar <= (t_y + t_h)) and
                        (x <= t_x_bar <= (x + w)) and
                        (y <= t_y_bar <= (y + h))):
                    matchedFid = True
                    faces[fid][frameCounter] = frame[y: (y + h), x: (x + w)]
                    cords[fid][frameCounter] = ((x, y), (x + w, y + h))
                    break

            for fid in fidsToDelete:
                faceTrackers.pop(fid, None)
            # If no matched fid, then we have to create a new tracker
            if not matchedFid:
                # Increase the currentFaceID counter
                currentFaceID += 1
                # Create and store the tracker
                tracker = dlib.correlation_tracker()
                tracker.start_track(frame, dlib.rectangle(x - 10, y - 20, x + w + 10, y + h + 20))
                faceTrackers[currentFaceID] = tracker
                faces.append(list(nones))
                cords.append(list(nones))
                faces[currentFaceID][frameCounter] = frame[y: (y + h), x: (x + w)]
                cords[currentFaceID][frameCounter] = ((x, y), (x + w, y + h))
    #print(np.array(faces).shape)
    #for i in faces[0]: print(i)
    return faces, cords

"""
if __name__ == '__main__':
    faceCascade = cv2.CascadeClassifier('C:/Users/Aref/PycharmProjects/trial/venv\Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    frames = []
    for i in range(100):
        rc, x = capture.read()
        x = cv2.resize(x, (320, 240))
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        y = faceCascade.detectMultiScale(gray, 1.3, 5)
        if i%10 == 0:
            frames.append((x, y))
        cv2.imshow('frame', x)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
    z = faceTracking(frames)
    print(len(z))
    for i in range(len(z)):
        for j in range(len(z[i])):
            if z[i][j] is None:
                print(j)
        print("\n")
"""
