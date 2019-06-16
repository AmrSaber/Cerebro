#!/usr/bin/env python3

import pickle, random
import numpy as np
import cv2

from image.utils import normalize_channels, normalize_image

def read_file(path, limit):
    with open(path, 'rb') as f: 
        xs, ys = pickle.load(f)
    
    if limit != -1:
        xs = xs[:limit]
        yx = ys[:limit]
    return (xs, ys)

def read_face_from_path(path, filter):
    image = cv2.imread(path)

    image_size = 150

    if filter:
        image = normalize_image(image, image_size, True, 'haar')
    else:
        image = normalize_image(image, image_size, True)

    image = normalize_channels(image)
    
    image = np.resize(image, (image_size, image_size, 1))

    return image

def save_parsed_data(xs, ys, path_testing, path_training, emotions):
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