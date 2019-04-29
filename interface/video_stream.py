import cv2
import time
import process_image as pi
from imutils.video import VideoStream
import threading
from queue import Queue

def marke_frame(frame, out_queue):
    marked_frame = pi.mark_faces_emotions(frame)
    out_queue.put(marked_frame)

def detect_stream_emotions(fps):
    frame_counter = 0
    my_queue = Queue()
    global frame

    vs = VideoStream(1).start()
    time.sleep(2.0)

    while True:
        frame_counter += 1
        frame = vs.read()
        if frame_counter % fps :
            thread = threading.Thread(target=marke_frame, args=(frame,my_queue))
            thread.start()
            continue

        cv2.imshow("Frame", frame)

        thread.join()
        marked_frame = my_queue.get()
        cv2.imshow("Frame", marked_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.stop()
    print("vd stopped")
    cv2.destroyAllWindows()
    print("windows closed")
