#! /usr/bin/env python3

import os

from model.reader.utils import *

path_all_faces = './model/dataset/CK+/faces'
path_all_emotions = './model/dataset/CK+/emotions'
path_training = './model/dataset/ck_training.bin'
path_testing = './model/dataset/ck_testing.bin'

# 0: neutral, 1: anger, 2: contempt, 3: disgust, 4: fear, 5: happy, 6: sadness, 7: surprise
emotions = ['Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

def read_training(limit=-1):
	return read_file(path_training, limit)

def read_testing(limit=-1):
    return read_file(path_testing, limit)

def split_data(quite, filter):
    xs, ys = parse_all_data(quite, filter)
    save_parsed_data(xs, ys, path_testing, path_training, emotions)

def parse_all_data(quite, filter):
    xs, ys = [], []

    # read dataset subjects
    subjects = os.listdir(path_all_emotions)
    for subject in subjects:
        subjectFacesDirectoryPath = os.path.join(path_all_faces, subject)
        subjectEmotionsDirectoryPath = os.path.join(path_all_emotions, subject)

        # read all sequences for each subject
        sequences = os.listdir(subjectEmotionsDirectoryPath)
        for sequence in sequences:
            emotion = os.path.join(subjectEmotionsDirectoryPath, sequence)
            facesPath = os.path.join(subjectFacesDirectoryPath, sequence)
            
            emotionsContent = os.listdir(emotion)

            # if there is no emotions, skip sequence
            if len(emotionsContent) == 0: continue

            # get the emotion value for this sequence
            emotionPath = os.path.join(emotion, emotionsContent[0])
            with open(emotionPath) as f:
                emotionValue = int(f.read().strip()[0])
            
            # read, sort and format faces pathes
            facesFiles = [face for face in os.listdir(facesPath) if not face.startswith('.')]
            facesFiles = sorted(facesFiles, key=faceNameIntoNumber)
            facesFiles = [os.path.join(facesPath, face) for face in facesFiles]
            
            emotionFacesCount = 2 * len(facesFiles) // 3
            neutralFacesCount = 1
            
            # read and convert face images
            emotionFaces = [read_face_from_path(face, filter) for face in facesFiles[-emotionFacesCount::2]]
            neutralFaces = [read_face_from_path(face, filter) for face in facesFiles[:neutralFacesCount:2]]

            xs += emotionFaces
            xs += neutralFaces

            ys += [emotionValue for face in emotionFaces]
            ys += [0 for face in neutralFaces]

            if not quite: print('Sequence Done')

        if not quite: print('Subject Done\n')
    return (xs, ys)

# faces files names are of format 'S064_003_00000009.png' is changed into '9'
def faceNameIntoNumber(name):
    name = name.split('_')[-1]
    name = name.split('.')[0]
    return int(name)
