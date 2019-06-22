#! /usr/bin/env python3

import os

from model.reader import ck, rafd
from model.reader.utils import *

path_training = './model/dataset/combined_training.bin'
path_testing = './model/dataset/combined_testing.bin'

# 0: neutral, 1: anger, 2: contempt, 3: disgust, 4: fear, 5: happy, 6: sadness, 7: surprise
emotions = ['Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
reduced_emotions = ['Neutral', 'Unsatisfied', 'Unsatisfied', 'Unsatisfied', 'Unsatisfied', 'Satisfied', 'Unsatisfied', 'Neutral']

def read_training(limit=-1):
	return read_file(path_training, limit)

def read_testing(limit=-1):
    return read_file(path_testing, limit)

def split_data(quite, filter):
    xs, ys = [], []
    
    # parse CK+ dataset
    if not quite: print('Parsing CK+ dataset')
    try:
        ck_xs_training, ck_ys_training = ck.read_training()
        ck_xs_testing, ck_ys_testing = ck.read_testing()

        xs += ck_xs_training
        xs += ck_xs_testing

        ys += ck_ys_training
        ys += ck_ys_testing
    except:
        raise Exception('CK+ dataset is not parsed!')
    
    # parse RafD dataset
    if not quite: print('Parsing RafD dataset')
    try:
        rafd_xs_training, rafd_ys_training = rafd.read_training()
        rafd_xs_testing, rafd_ys_testing = rafd.read_testing()

        xs += rafd_xs_training
        xs += rafd_xs_testing

        ys += rafd_ys_training
        ys += rafd_ys_testing
    except:
        raise Exception('RafD dataset is not parsed!')
    
    if not quite: print('Saving combined datasets')
    save_parsed_data(xs, ys, path_testing, path_training, emotions)