from threading import Thread
from queue import Queue
import time

import cv2
from imutils.video import VideoStream

from interface import process_image as pi

def detect_stream_emotions(fps):
    frame_counter = 0
    frame_data = None
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    worker_thread = WorkerThread(frame_data)
    worker_thread.start()

    while True:
        success, frame = vs.read()
        if not success: continue
        
        frame_counter += 1

        if (frame_counter % fps) == 0:
            worker_thread.add_task(frame)

        if frame_data != None:
            frame = pi.mark_faces_emotions(frame, None, frame_data)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): break
    
    vs.release()
    cv2.destroyAllWindows()
    worker_thread.kill()

class WorkerThread(Thread):
    def __init__(self, result_container):
        self.queue = Queue()
        self.result_container = result_container
    
    def run(self):
        while True:
            frame = self.queue.get()
            if frame == -1: break
            self.result_container = pi.extract_faces_emotions(frame)
    
    def add_task(self, frame):
        self.queue.put(frame)

    def kill(self):
        self.queue.put(-1)
