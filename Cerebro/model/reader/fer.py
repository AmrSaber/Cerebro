#! /user/bin/env python3

import cv2
import numpy as np

from image.face_detector import detect_dlib as dlib

data_all = './model/dataset/fer2013.csv'
data_test = './model/dataset/fer2013_testing.csv'
data_training = './model/dataset/fer2013_train.csv'

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def read_testing(limit=-1):
	return read_from_file(data_test, limit)

def read_training(limit=-1):
	return read_from_file(data_training, limit)

def read_from_file(path, limit=-1):
	xs, ys = [], []
	with open(path) as file:
		count = 0
		for line in file:
			if limit != -1 and count >= limit: break
			count += 1
			emotion, image = parse_line(line)
			xs.append(image)
			ys.append(emotion)
	return np.array(xs), np.array(ys)

def parse_line(line):
	emotion, pxls, usage = line.split(',')
	emotion, pxls = int(emotion), list(map(int, pxls.split()))

	image = np.reshape(pxls, (48, 48))
	image = np.array(image, dtype=np.uint8)

	return emotion, image

def split_data(quite, filter):
	tests, trains = [], []
	with open(data_all) as input:
		for line in input:
			# ignore header
			if line[0].isalpha(): continue

			_, image = parse_line(line)

			if filter:
				# filter non-face images
				if dlib.is_one_face(image):
					if not quite: print('Face')
				else:
					if not quite: print('Not Face')
					# cv2.imshow('', image)
					# cv2.waitKey(0)
					continue

			# add image to its category
			if 'test' in line.lower(): tests.append(line)
			if 'train' in line.lower(): trains.append(line)

	print('Training size:', len(trains))
	print('Testing size:', len(tests))

	with open(data_test, 'w') as f: f.writelines(tests)
	with open(data_training, 'w') as f: f.writelines(trains)