#! /usr/bin/env python3

import os

from model.reader.utils import *

path_all_data = './model/dataset/RafD'
path_training = './model/dataset/rafd_training.bin'
path_testing = './model/dataset/rafd_testing.bin'

# 0: neutral, 1: anger, 2: contempt, 3: disgust, 4: fear, 5: happy, 6: sadness, 7: surprise
emotions = ['Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
reduced_emotions = ['Neutral', 'Unsatisfied', 'Unsatisfied', 'Unsatisfied', 'Unsatisfied', 'Satisfied', 'Unsatisfied', 'Neutral']

def read_training(limit=-1):
	return read_file(path_training, limit)

def read_testing(limit=-1):
    return read_file(path_testing, limit)

def split_data(quite, filter):
    xs, ys = parse_all_data(quite, filter)
    save_parsed_data(xs, ys, path_testing, path_training, emotions)

def parse_all_data(quite, filter):
    xs, ys = [], []
    
    image_names = os.listdir(path_all_data)

    if not quite:
        done_count = 0
        print()

    for image_name in image_names:
        image_full_path = os.path.join(path_all_data, image_name)
        y = get_emotion_from_name(image_name)
        x = read_face_from_path(image_full_path, filter)

        xs.append(x)
        ys.append(y)

        if not quite:
            done_count += 1
            print(f'\rIn progress: {done_count}/{len(image_names)}', end='')
        
    if not quite:
        print('\rDone.')

    return (xs, ys)

'''
our_emotions =   ['Neutral', 'Anger', 'Contempt',     'Disgust',   'Fear',    'Happy', 'Sadness', 'Surprise' ]
their_emotions = ['neutral', 'angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'sad',     'surprised']
'''
def get_emotion_from_name(file_name):
    their_emotions = ['neutral', 'angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
    file_name = file_name.lower()

    for i, emotion in enumerate(their_emotions):
        if emotion in file_name:
            return i

    raise Exception(f'Cannot find emotion in name {file_name}!!')
