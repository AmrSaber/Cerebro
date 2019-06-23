#!usr/env/bin python3

import cv2
import numpy as np
from moviepy.editor import *

from interface import process_image as pi
from image import FaceTracking as tracker
from model.emotions_model import EmotionsModel
import ffmpeg 
from image.face_detector import detect_dlib
from image.face_detector import detect_haar
from image.face_detector import detect_lbp

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
                            #tmp = mark_emotion(tmp, returned_cords[i][j], em)
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

def mark_emotion(image, cords, emotion):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (72, 1, 68)
    offset_x = 5
    offset_y = 0


    tmp = (cords[0][0] + offset_x, cords[0][1] - offset_y)
    size = cv2.getTextSize(emotion, font, fontScale=font_scale, thickness=1)[0]
    box_coords = (tmp, (tmp[0] + size[0] - 2, tmp[1] - size[1] - 2))
    cv2.rectangle(image, box_coords[0], box_coords[1], color, cv2.FILLED)

    #selected face box
    image = cv2.rectangle(image,cords[0],cords[1],color,2)

    #text
    image = cv2.putText(
        image,
        emotion,
        tmp,
        font,
        font_scale,
        (255, 255, 255),
        1
    )
    return image

