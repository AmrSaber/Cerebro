import sys
sys.path.insert(1,'../image/face_detector')
sys.path.insert(1, '../image')
import detect_dlib as detector
import cv2
import numpy as np
import process_image as pi
from moviepy.editor import *

def detect_video_emotions(video_path, output_path, skip = 50):
    """
    skip >> detemines number of skipped frames during procerssing
    in normal video speed skip=50 means detect emotions every 2 seconds
    higher skip means faster processing but less accurate
    -----------------------------------------
    output_path >> include output name & extension
    the output video extension should be mp4
    """
    #saving audio 
    video = VideoFileClip("x.mp4")
    audio = video.audio

    #processing frames
    vidObj = cv2.VideoCapture(video_path)
    fps = vidObj.get(5) #fps

    success = 1 #checks whether frames were extracted
    real_frame_counter = 1 #to check with sampling
    sampled_frame_counter = 1 #to print frame number for user
    prev_frame_data = None

    img_frames = []

    #getting frames
    while success:
        success, image = vidObj.read()

        if not success :
            break
        #processing frames
        if not(real_frame_counter % skip):
            prev_frame_data = pi.extract_faces_emotions(image)
            print("process frame: ", sampled_frame_counter)
            sampled_frame_counter += 1

        image = pi.mark_faces_emotions(image,"None", prev_frame_data)
        real_frame_counter += 1
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img_frames.append(image)
    
    #making output video
    print("making output video...")
    clips = [ImageClip(m).set_duration(1/fps)
         for m in img_frames]
    concat_clip = concatenate_videoclips(clips, method="chain")
    concat_clip_edited = concat_clip.set_audio(audio)
    concat_clip_edited.write_videofile("output.mp4", fps=fps)
    return 
