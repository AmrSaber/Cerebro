import cv2
import time
from imutils.video import VideoStream
from threading import Thread
from interface import process_image as pi
from queue import Queue

task_queue = Queue(1)
result_queue = Queue()

def worker():
        global task_queue
        global result_queue
        print("running")
        while True:
            while(not task_queue.empty()):
                result_queue.put(pi.extract_faces_emotions(task_queue.get()))


def detect_stream_emotions():
    fps = 40
    global task_queue
    global result_queue
    frame_counter = 0
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        res, frame = vs.read()
        if res:
            frame_counter += 1
            if not result_queue.empty():
                frame = pi.mark_faces_emotions(frame, None, result_queue.get())
            cv2.imshow("Frame", frame)
            if not (frame_counter % fps) :
                if not task_queue.full():
                    task_queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vs.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    producer = Thread(target = worker)
    consumer = Thread(target = detect_stream_emotions)

    producer.start()
    consumer.start()
