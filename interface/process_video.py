import sys
sys.path.insert(1,'../image/face_detector')
sys.path.insert(1, '../image')
import detect_dlib as detector
import cv2
import numpy as np
import process_image as pi
from moviepy.editor import *


def display_video(frames, video_name="video"):

    for i in range(len(frames)-1):
        cv2.imshow(video_name,frames[i])
        if cv2.waitKey(200) & 0xFF == ord('q'):
             break

def detect_video_emotions(video_path, skip = 50):

    #audio part 
    video = VideoFileClip("x.mp4")
    audio = video.audio

    #processing frames
    vidObj = cv2.VideoCapture(video_path)
    video_fps = vidObj.get(5)#fps

    success = 1 # checks whether frames were extracted
    real_frame_counter = 1 #to check with sampling
    sampled_frame_counter = 1 #to print frame number for user
    prev_frame_data = None

    img_frames = []

    while success:
        success, image = vidObj.read()

        if not success :
            break

        if not(real_frame_counter % skip):
            prev_frame_data = pi.extract_faces_emotions(image)
            print("process frame: ", sampled_frame_counter)
            sampled_frame_counter += 1

        image = pi.mark_faces_emotions(image,"None", prev_frame_data)
        real_frame_counter += 1

        img_frames.append(image)
        
    return img_frames, audio, video_fps

def save_video(img_frames, audio, video_name,fps = 25): 
    """
    
    output will be .avi
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width, layers = img_frames[0].shape
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))
    for i in range(len(img_frames)):
        video.write(img_frames[i])
        
    cv2.destroyAllWindows()
    video.release()
    """
    clips = [ImageClip(m).set_duration(1/fps)
         for m in img_frames]
    concat_clip = concatenate_videoclips(clips, method="chain")
    concat_clip_edited = concat_clip.set_audio(audio)
    concat_clip_edited.write_videofile("test.mp4", fps=fps)
    

if __name__ == '__main__':
    
    img_frames, audio, fps = detect_video_emotions("x.mp4")
    save_video(img_frames, audio,"xx.avi", fps)

    """
    video2 = VideoFileClip("xx.avi")
    edited = video2.set_audio(audio)
    edited.write_videofile("test.mp4", fps = fps)
    """
    #display_video(frames)
    
    