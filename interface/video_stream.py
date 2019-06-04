import cv2
import time
import process_image as pi
from imutils.video import VideoStream

def detect_stream_emotions(fps):
    frame_counter = 0
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        if frame_counter % fps :
            frame_counter += 1
            continue
        frame_counter += 1
        frame = vs.read()
        marked_frame = pi.mark_faces_emotions(frame)
        cv2.imshow("Frame", marked_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
	        break

    vs.stop()
    print("vd stopped")
    cv2.destroyAllWindows()
    print("windows closed")
