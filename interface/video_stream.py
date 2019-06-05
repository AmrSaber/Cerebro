import cv2
import time
import process_image as pi
from imutils.video import VideoStream
import threading

def marke_frame(frame):
    frame_data = pi.extract_faces_emotions(frame)
    frame = pi.mark_faces_emotions(frame, "None", frame_data)
    print("thread working")

def detect_stream_emotions(fps):
    frame_counter = 0
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        res, current_frame= vs.read()
        frame = current_frame
        if res:
            frame_counter += 1
            if frame_data != None:
                frame = pi.mark_faces_emotions(frame, "None", frame_data)
            cv2.imshow("Frame", frame)
            print("mian thread")
            if not (frame_counter % fps) :
                thread = threading.Thread(target=marke_frame, args=(frame,))
                thread.start()
                continue

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vs.release()
    cv2.destroyAllWindows()

frame =[]
frame_data = []
