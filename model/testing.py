#! /user/bin/env python3

from model import *
from reader import read_testing, read_training, emotions_map
from  keras.utils import  to_categorical

def main():
	print('Reading data...')
	x_test, y_test = read_testing()
	x_train, y_train = read_training()

	emotions_count = len(set(emotions_map))
	y_train = to_categorical(y_train, emotions_count)
	y_test = to_categorical(y_test, emotions_count)

	model = EmotionsModel(emotions_count)
	print('\nCreated Model.')

	print('Training...')
	history = model.fit(x_train, y_train)

	print('\nTesting...')
	result = model.test(x_test, y_test)
	print("Result:", result)

if __name__ == '__main__': main()
