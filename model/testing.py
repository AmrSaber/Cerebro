#! /user/bin/env python3

from model import *
from reader import read_testing, read_training, emotions_map
from  keras.utils import  to_categorical
import argparse

def main():

	parser = argparse.ArgumentParser(description='')

	parser.add_argument('-g', action='store_true', help='Use HOG for training')
	parser.add_argument('-m', action='store_true', help='Use LM (land marks) for training')
	parser.add_argument('-c', action='store_true', help='Use CNN for training')

	parser.add_argument('-t', action='store_true', help='Force train model')
	parser.add_argument('-n', action='store_true', help='Create new model, reset old weights (if any)')

	parser.add_argument('--tr-count', metavar='int', type=int, default=-1, help='Set the count of the training data')
	parser.add_argument('--ts-count', metavar='int', type=int, default=-1, help='Set the count of the testing data')

	args = parser.parse_args()
	must_train = args.t

	print('Reading data...')
	x_test, y_test = read_testing(args.ts_count)
	x_train, y_train = read_training(args.tr_count)

	emotions_count = len(set(emotions_map))

	model = EmotionsModel(
				emotions_count,
				use_hog=args.g,
				use_lm=args.m,
				use_cnn=args.c,
				create_new=args.n
			)
	print('\nCreated Model.')

	if not model.is_trained or args.t:
		print('Training...')
		history = model.fit(x_train, y_train)
		# print(history.history['val_acc'])

	print('\nTesting...')
	result = model.test(x_test, y_test)
	print("Result:", result)

if __name__ == '__main__': main()
