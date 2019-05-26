import sys
sys.path.insert(1,'../image/face_detector')
sys.path.insert(1, '../image')
import detect_dlib as detector
import cv2
import numpy as np
import process_image as pi

def display_video(frames, video_name="video"):

    #take frames and display it at particular rate
    #exit by pressing q key
    # TODO: how to control fps display through waitKey

    for i in range(len(frames)-1):
        cv2.imshow(video_name,frames[i])
        if cv2.waitKey(200) & 0xFF == ord('q'):
             break

def detect_video_emotions(video_path, skip = 50):
    vidObj = cv2.VideoCapture(video_path)

    success = 1 # checks whether frames were extracted
    real_frame_counter = 1 #to check with sampling
    sampled_frame_counter = 1 #to print frame number for user
    prev_frame_data = None
    frames = []
    while success:
        success, image = vidObj.read()
        if not success :
            break
        #print(success,">>>",real_frame_counter % fps)
        if not(real_frame_counter % skip):
            prev_frame_data = pi.extract_faces_emotions(image)
            #frame_info = detector.get_faces(image)
            #for fi in frame_info:
            #    image = cv2.rectangle(image,fi[1][0],fi[1][1],(0, 255, 0),1)
            print("process frame: ", sampled_frame_counter)
            sampled_frame_counter += 1


        image = pi.mark_faces_emotions(image,"None", prev_frame_data)
        frames.append(image)
        real_frame_counter += 1

    return frames

def save_video(frames, video_name, fps = 25):
    """
    video_extension should be .avi .mp4
    """
    height, width, layers = frame[0].shape
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))
    for frame in frames:
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    """
    frames = __extract_frames("x.mp4")
    for i in range(len(frames)-1):
        cv2.imshow("zzz",frames[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    """
    frames = detect_video_emotions("x.mp4")
    save_video(frames, "xx")
    #display_video(frames)
