#! /usr/bin/env python3

import os, random, pickle
import cv2
import numpy as np

from image.utils import normalize_image

path_all_faces = './model/dataset/CK+/faces'
path_all_emotions = './model/dataset/CK+/emotions'
path_training = './model/dataset/ctk_training.bin'
path_testing = './model/dataset/ctk_testing.bin'

# 0: neutral, 1: anger, 2: contempt, 3: disgust, 4: fear, 5: happy, 6: sadness, 7: surprise
emotions = ['Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

def read_training(limit=-1):
	return read_file(path_training, limit)

def read_testing(limit=-1):
    return read_file(path_testing, limit)

def read_file(path, limit):
    with open(path, 'rb') as f: 
        xs, ys = pickle.load(f)
    
    if limit != -1:
        xs = xs[:limit]
        yx = ys[:limit]
    return (xs, ys)

def split_data(quite, filter):
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
            emotionFaces = [readFaceFromPath(face, filter) for face in facesFiles[-emotionFacesCount::2]]
            neutralFaces = [readFaceFromPath(face, filter) for face in facesFiles[:neutralFacesCount:2]]

            xs += emotionFaces
            xs += neutralFaces

            ys += [emotionValue for face in emotionFaces]
            ys += [0 for face in neutralFaces]

            if not quite: print('Sequence Done')

        if not quite: print('Subject Done\n')
        
    # shuffle xs, and ys
    allData = [(x, y) for x, y in zip(xs, ys)]
    random.shuffle(allData)
    xs = [e[0] for e in allData]
    ys = [e[1] for e in allData]

    # split xs, ys into training and testing (80-20 split)
    testingCount = len(xs) // 5
    xs_training = xs[:-testingCount]
    xs_testing = xs[-testingCount:]
    
    ys_training = ys[:-testingCount]
    ys_testing = ys[-testingCount:]

    # save training and testing data into binary files
    with open(path_training, 'wb') as f: pickle.dump((xs_training, ys_training), f)
    with open(path_testing, 'wb') as f: pickle.dump((xs_testing, ys_testing), f)

    # print stats
    emotionsCount = [0] * len(emotions)
    for y in ys: emotionsCount[y] += 1

    print(f'Dataset Size: {len(xs)}')
    print(f'Training Size: {len(xs_training)}')
    print(f'Testing Size: {len(xs_testing)}')
    print()

    print('Emotions Stats:')
    print('===============')
    for i, emotion in enumerate(emotions):
        print('%8s: %3d' % (emotion, emotionsCount[i]))

# faces files names are of format 'S064_003_00000009.png' is changed into '9'
def faceNameIntoNumber(name):
    name = name.split('_')[-1]
    name = name.split('.')[0]
    return int(name)

# reads and transforms images
def readFaceFromPath(path, filter):
    image = cv2.imread(path)

    image_size = 150

    if filter:
        image = normalize_image(image, image_size, True, 'haar')
    else:
        image = normalize_image(image, image_size, True)

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = np.resize(image, (image_size, image_size, 1))

    return image