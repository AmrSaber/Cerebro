#!usr/env/bin python3

import cv2
import numpy as np
from moviepy.editor import *
import ffmpeg 

from Cerebro.interface import process_image as pi
from Cerebro.image import FaceTracking as tracker
from Cerebro.model.emotions_model import EmotionsModel

from Cerebro.image.face_detector import detect_dlib
from Cerebro.image.face_detector import detect_haar
from Cerebro.image.face_detector import detect_lbp

model = EmotionsModel()

def check_rotation(path_video_file):

    print (path_video_file)
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    print(meta_dict['streams'][0]['tags'])

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

def detect_video_emotions_with_tracking (video_path, output_path, batch_size=125, detector_type='dlib', verbose=False):
    #saving audio 
    video = VideoFileClip(video_path)
    audio = video.audio

    #detector
    if detector_type == 'dlib':
        detector = detect_dlib
    elif detector_type =='haar':
        detector = detect_haar
    elif detector_type =='lbp':
        detector = detect_lbp

    #processing frames
    vidObj = cv2.VideoCapture(video_path)
    fps = vidObj.get(5) #fps
    rotateCode = check_rotation(video_path)

    success = 1 #checks whether frames were extracted
    it = 0
    img_frames = []
    to_be_tracked = []
    to_be_printed = 0
    #getting frames
    while success:
        success, image = vidObj.read()

        if rotateCode is not None:
            image = cv2.rotate(image, rotateCode)

        if not success :
            break

        #after calling track function that is what is supposed to be done
        if it == batch_size :
            returned_faces, returned_cords = tracker.faceTracking(to_be_tracked, batch_size)
            
            print(len(returned_faces))

            emotions = model.predict_with_vote(returned_faces)
            
            if len(returned_cords) == 0:
                for i in range(125):
                    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)
                    img_frames.append(tmp)
            else:
                for j in range(len(returned_cords[0])): 
                    tmp = to_be_tracked[j][0]
                    extracted_faces_emotions = []
                    for i in range(len(returned_cords)):
                        if returned_cords[i][j] != None:
                            em = emotions[i]
                            extracted_faces_emotions.append((returned_faces[i][j], returned_cords[i][j], em))
                            
                    tmp = pi.mark_faces_emotions(tmp, extracted_faces_emotions=extracted_faces_emotions)
                    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)
                    img_frames.append(tmp)
            
            to_be_tracked = []
            it = 0
            to_be_printed += 1
            if verbose:
                print("batch: ", to_be_printed)


        face_boxes = []
        frame_data = detector.get_faces(image)
        for fd in frame_data :
            face_boxes.append(fd[1])
        to_be_tracked.append([image, face_boxes])
        it += 1

    #making output
    clips = [ImageClip(m).set_duration(1/fps) for m in img_frames]
    concat_clip = concatenate_videoclips(clips, method="chain")
    concat_clip_edited = concat_clip.set_audio(audio)
    concat_clip_edited.write_videofile(output_path, fps=fps)

def detect_video_emotions(video_path, output_path, skip = 50, detector_type = 'dlib', verbose=False):
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
            prev_frame_data = pi.extract_faces_emotions(image, detector_type)
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
