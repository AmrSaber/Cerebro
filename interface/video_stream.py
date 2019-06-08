import cv2
import time
from imutils.video import VideoStream
#import threading
from threading import Thread, Semaphore

from interface import process_image as pi

# frame =[]
frame_data = None

def mark_frame(frame):
    global frame_data
    print('HERE')
    frame_data = pi.extract_faces_emotions(frame)
    # frame = pi.mark_faces_emotions(frame, None, frame_data)
    print(frame_data)

def detect_stream_emotions(fps):
    global frame_data
    
    frame_counter = 0
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        res, frame = vs.read()
        if res:
            frame_counter += 1
            if frame_data != None:
                #print(frame_data)
                frame = pi.mark_faces_emotions(frame, None, frame_data)

            cv2.imshow("Frame", frame)
            # print("main thread")
            if not (frame_counter % fps) :
                thread = threading.Thread(target=mark_frame, args=(frame,))
                thread.start()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vs.release()
    cv2.destroyAllWindows()

class WorkerThread(Thread):
    def __init__(self):
        self.semaphore = Semaphore(0)
    
    def run(self):
        while True:
            self.semaphore.aquire()
            # get frame from queue
    
    def processFrame(frame):
        # put frame in queue
        self.semaphore.release()
















