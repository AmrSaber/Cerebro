#! /user/bin/env python3

import numpy as np

# old_emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# new_emotions = ["Fear", "Neutral", "Satisfied", "Surprise", "Unsatisfied"]
emotions_map = [4, 4, 0, 2, 4, 3, 1]

def read_data(file_name = 'dataset/fer2013.csv', limit = 1000):
	x_train, y_train, x_test, y_test = [], [], [], []
	with open(file_name) as file:
		for i, line in enumerate(file):
			if i == 0: continue
			emotion, pxls, usage = line.split(',')
			emotion = int(emotion)
			pxls = [int(e) for e in pxls.split(' ')]
			image = np.reshape(pxls, (48, 48, 1))

			emotion = emotions_map[emotion]

			if 'training' in usage.lower():
				x_train.append(image)
				y_train.append(emotion)
			elif 'test' in usage.lower():
				x_test.append(image)
				y_test.append(emotion)

	return ((np.array(x_train[:limit]), np.array(y_train[:limit])), (np.array(x_test[:limit]), np.array(y_test[:limit])))
