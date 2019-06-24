from threading import Thread, get_ident, Condition
from queue import Queue
import time

import cv2
from imutils.video import VideoStream

from Cerebro.interface import process_image as pi

frame_data = []
def detect_stream_emotions(skip=10):
    frame_counter = 0
    vs = cv2.VideoCapture(0)

    worker_thread = WorkerThread()
    worker_thread.start()

    while True:
        success, frame = vs.read()
        if not success: continue
        
        frame_counter += 1

        if (frame_counter % skip) == 0:
            worker_thread.add_task(frame)
            frame_counrer = 0

        if len(frame_data) > 0:
            frame = pi.mark_faces_emotions(frame, None, frame_data)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): break
    
    vs.release()
    cv2.destroyAllWindows()
    worker_thread.kill()
    worker_thread.join()

class WorkerThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.condition = Condition()
        self.saved_frame = None
        self.should_close = False
    
    def run(self):
        global frame_data
        while True:
            # get saved frame if any and clear the saved value
            self.condition.acquire()
            try:
                while type(self.saved_frame) == type(None): 
                    if self.should_close: return
                    self.condition.wait()
                
                frame = self.saved_frame
                self.saved_frame = None
            finally:
                self.condition.release()
            
            if type(frame) == type(-1) and frame == -1: break
            frame_data = pi.extract_faces_emotions(frame, 'haar')
    
    def add_task(self, frame):
        self.condition.acquire()
        try:
            self.saved_frame = frame
            self.condition.notify()
        finally:
            self.condition.release()

    def kill(self):
        self.condition.acquire()
        try:
            self.should_close = True
            self.condition.notify()
        finally:
            self.condition.release()
