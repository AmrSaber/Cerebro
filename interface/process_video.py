#!usr/env/bin python3

import cv2
import numpy as np
from moviepy.editor import *

from interface import process_image as pi

import ffmpeg 

def check_rotation(path_video_file):

    print (path_video_file)
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

def detect_video_emotions_tracking (video_path, output_path, tracked_frames=50):
    #saving audio 
    video = VideoFileClip(video_path)
    audio = video.audio

    #processing frames
    vidObj = cv2.VideoCapture(video_path)
    fps = vidObj.get(5) #fps
    rotateCode = check_rotation(video_path)

    success = 1 #checks whether frames were extracted
    it = 0
    img_frames = []
    to_be_tracked = []
    #getting frames
    while success:
        success, image = vidObj.read()

        if rotateCode is not None:
            image = cv2.rotate(image, rotateCode)

        if not success :
            break
        if it == tracked_frames :
            to_be_tracked = []
            it = 0
            #calling aref function >>returns 2 lists [imgs ,persons], [boxes, persons]
            #calling amr detection function >> returns [emotions, persons]
            #combing frame by frame image,[face, corners, emotion]
            #for loop to marks emotions on it (with length n)
            #from BGR to RGB :image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            #append frames to image frames

        face_boxes = []
        frame_data = dt.get_faces(image)
        for fd in frame_data :
            face_boxes.append(fd[1])
        to_be_tracked.append([image, get_faces])
        it += 1

    #making output
    clips = [ImageClip(m).set_duration(1/fps) for m in img_frames]
    concat_clip = concatenate_videoclips(clips, method="chain")
    concat_clip_edited = concat_clip.set_audio(audio)
    concat_clip_edited.write_videofile(output_path, fps=fps)

