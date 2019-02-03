#! /user/bin/env python3

import sys; sys.path.insert(1, '../image')
from face_detector import dlib

import cv2
import numpy as np
import argparse

data_all = 'dataset/fer2013.csv'
data_test = 'dataset/fer2013_testing.csv'
data_training = 'dataset/fer2013_train.csv'

# old_emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# new_emotions = ["Fear", "Neutral", "Satisfied", "Surprise", "Unsatisfied"]
emotions_map = [4, 4, 0, 2, 4, 3, 1]

def read_testing():
	return read_from_file(data_test)

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

	image = np.reshape(pxls, (48, 48, 1))
	image = np.array(image, dtype=np.uint8)
	emotion = emotions_map[emotion]

	return emotion, image

def enhance(image):
	return image

def split_data(quite, filter, leave):
	tests, trains = [], []
	with open(data_all) as input:
		for line in input:
			# ignore header
			if line[0].isalpha(): continue

			_, image = parse_line(line)

			if not leave: image = enhance(image)

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

	if not quite:
		print('Training size:', len(trains))
		print('Testing size:', len(tests))

	with open(data_test, 'w') as f: f.writelines(tests)
	with open(data_training, 'w') as f: f.writelines(trains)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Read, filter and separate the data to testing and training data (filtering is optional)')
	parser.add_argument('-q', action='store_true', help='Quite mode, no ouput')
	parser.add_argument('-f', action='store_true', help='Filter non-face images')
	parser.add_argument('-l', action='store_true', help='Leave image without enhancement')
	args = parser.parse_args()

	split_data(args.q, args.f, args.l)
