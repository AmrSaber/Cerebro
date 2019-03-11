import sys
sys.path.insert(1,'../image/face_detector')
sys.path.insert(1, '../image')
import detect_dlib as detector
import cv2
import numpy as np
#import process_image as pi
"""
def __extract_frames(video_path):

    vidObj = cv2.VideoCapture(video_path)

    # checks whether frames were extracted
    success = 1
    frames = []

    while success:
        success, image = vidObj.read()
        frames.append(image)

    return frames

def detect_video_emotions(video_path, fps=100):

    # TODO: how to control fps

    frames = __extract_frames(video_path)
    for i in range(len(frames)-1):
        frames_info = detector.get_faces(frames[i])
        for frame_info in frames_info:
        	frames[i] = cv2.rectangle(frames[i],
                                    frame_info[1][0],
                                    frame_info[1][1],
                                    (0, 255, 0),
                                     1)
        print("process frame :", i, " of ", len(frames))

    display_video(frames)
    cv2.waitKey(0)
    https://www.techbeamers.com/python-multithreading-concepts/#python-multithreading-modules
"""
def display_video(frames, video_name="video"):

    #take frames and display it at particular rate
    #exit by pressing q key
    # TODO: how to control fps display through waitKey

    for i in range(len(frames)-1):
        cv2.imshow(video_name,frames[i])
        if cv2.waitKey(2) & 0xFF == ord('q'):
             break

def detect_video_emotions(video_path):
    vidObj = cv2.VideoCapture(video_path)

    # checks whether frames were extracted
    success = 1
    counter = 1
    frames = []
    while success:
        success, image = vidObj.read()
        if image.size != 0:
            frame_info = detector.get_faces(image)
            for fi in frame_info:
                image = cv2.rectangle(image,fi[1][0],fi[1][1],(0, 255, 0),1)
            frames.append(image)
            print("process frame: ", counter)
            counter += 1
    return frames

if __name__ == '__main__':
    """
    frames = __extract_frames("x.mp4")
    for i in range(len(frames)-1):
        cv2.imshow("zzz",frames[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    """
    frames = detect_video_emotions("x.mp4")
    display_video(frames)
