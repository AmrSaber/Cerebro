#! /user/bin/env python3

from model import *
from reader import read_testing, read_training, emotions_map
from  keras.utils import  to_categorical
import argparse

def main():

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-t', action='store_true', help='Force train model')
	parser.add_argument('-g', action='store_true', help='Use HOG for training')
	parser.add_argument('-n', action='store_true', help='Create new model, reset old weights (if any)')
	args = parser.parse_args()
	must_train = args.t

	print('Reading data...')
	x_test, y_test = read_testing()
	x_train, y_train = read_training()

	emotions_count = len(set(emotions_map))
	y_train = to_categorical(y_train, emotions_count)
	y_test = to_categorical(y_test, emotions_count)

	model = EmotionsModel(emotions_count, use_hog=args.g, create_new=args.n)
	print('\nCreated Model.')

	print('Training...')
	if not model.is_trained or args.t:
		history = model.fit(x_train, y_train)

	print('\nTesting...')
	result = model.test(x_test, y_test)
	print("Result:", result)

if __name__ == '__main__': main()
