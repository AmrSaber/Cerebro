import cv2
import time
from imutils.video import VideoStream
from threading import Thread
from interface import process_image as pi
from queue import Queue

task_queue = Queue(1)
result_queue = Queue()

def worker( task_queue, result_queue):
        print("running")
        while True:
            if not task_queue.empty():
                result_queue.put(pi.extract_faces_emotions(task_queue.get()))


def detect_stream_emotions(fps):
    frame_counter = 0
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        res, frame = vs.read()
        if res:
            frame_counter += 1
            if not result_queue.empty():
                frame_data = result_queue.get()
                frame = pi.mark_faces_emotions(frame, None, frame_data)
            cv2.imshow("Frame", frame)
            if not (frame_counter % fps) :
                if not task_queue.full():
                    task_queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vs.release()
    cv2.destroyAllWindows()

mythread = Thread(target = worker, args=(task_queue, result_queue))
mythread.start()
