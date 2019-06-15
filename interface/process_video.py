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


def detect_video_emotions(video_path, output_path, skip = 50, verbose=False):
    """
    skip >> determines number of skipped frames during procerssing
    in normal video speed skip=50 means detect emotions every 2 seconds
    higher skip means faster processing but less accurate
    -----------------------------------------
    output_path >> include output name & extension
    the output video extension should be mp4
    """
    #saving audio 
    video = VideoFileClip(video_path)
    audio = video.audio

    #processing frames
    vidObj = cv2.VideoCapture(video_path)
    fps = vidObj.get(5) #fps
    rotateCode = check_rotation(video_path)

    success = 1 #checks whether frames were extracted
    real_frame_counter = 1 #to check with sampling
    sampled_frame_counter = 1 #to print frame number for user
    prev_frame_data = None

    img_frames = []

    #getting frames
    while success:
        success, image = vidObj.read()

        if rotateCode is not None:
            image = cv2.rotate(image, rotateCode)

        if not success :
            break
        #processing frames
        if not(real_frame_counter % skip):
            prev_frame_data = pi.extract_faces_emotions(image)
            if verbose: print("process frame: ", sampled_frame_counter)
            sampled_frame_counter += 1

        image = pi.mark_faces_emotions(image,None, prev_frame_data)
        real_frame_counter += 1
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img_frames.append(image)
    
    #making output video
    if verbose: print("making output video...")
    clips = [ImageClip(m).set_duration(1/fps) for m in img_frames]
    concat_clip = concatenate_videoclips(clips, method="chain")
    concat_clip_edited = concat_clip.set_audio(audio)
    concat_clip_edited.write_videofile(output_path, fps=fps)