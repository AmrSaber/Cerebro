#! /user/bin/env python3

import numpy as np

from Cerebro.image.utils import normalize_channels, normalize_image
from Cerebro.model.reader.utils import save_parsed_data, read_file

path_all = './model/dataset/fer2013.csv'
path_testing = './model/dataset/fer2013_testing.csv'
path_training = './model/dataset/fer2013_train.csv'

emotions = ['Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
reduced_emotions = ['Neutral', 'Unsatisfied', 'Unsatisfied', 'Unsatisfied', 'Unsatisfied', 'Satisfied', 'Unsatisfied', 'Neutral']

# their_emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# our_emotions = ['Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
emotions_map = [1, 3, 4, 5, 6, 7, 0]

def read_testing(limit=-1):
	return read_file(path_testing, limit)

def read_training(limit=-1):
	return read_file(path_training, limit)

def split_data(quite, filter):
	xs, ys = [], []

	if not quite:
		print()
		done_count = 0

	with open(path_all) as input:
		for line in input:
			# ignore header
			if line[0].isalpha(): continue
			
			try:
				x, y = parse_line(line, filter)
			except IndexError as error:
				continue

			y = emotions_map[y]

			xs.append(x)
			ys.append(y)

			if not quite:
				done_count += 1
				print(f'\rProcessed: {done_count}', end='')
			
	if not quite: print()
	save_parsed_data(xs, ys, path_testing, path_training, emotions)

def parse_line(line, filter):
	emotion, pxls, usage = line.split(',')
	emotion, pxls = int(emotion), list(map(int, pxls.split()))

	image = np.reshape(pxls, (48, 48))
	image = np.array(image, dtype=np.uint8)

	image_size = 150

	if filter:
		image = normalize_image(image, image_size, True, 'haar')
	else:
		image = normalize_image(image, image_size, False)

	image = normalize_channels(image)
	
	image = np.resize(image, (image_size, image_size, 1))

	return image, emotion