#! /user/bin/env python3

import argparse

from model.emotions_model import *
from model.reader import read_testing, read_training, get_emotions, get_reduced_emotions, set_dataset, available_datasets

def main():
	parser = argparse.ArgumentParser(description='')

	parser.add_argument('-g', action='store_true', help='Use HOG for training')
	parser.add_argument('-m', action='store_true', help='Use LM (land marks) for training')
	parser.add_argument('-c', action='store_true', help='Use CNN for training')

	parser.add_argument('-t', action='store_true', help='Force train model')
	parser.add_argument('-n', action='store_true', help='Create new model, reset old weights (if any)')

	parser.add_argument('--tr-count', metavar='int', type=int, default=-1, help='Set the count of the training data')
	parser.add_argument('--ts-count', metavar='int', type=int, default=-1, help='Set the count of the testing data')
	
	parser.add_argument('-d', type=str, required=True, choices=available_datasets, help='Dataset to use')

	args = parser.parse_args()
	must_train = args.t

	set_dataset(args.d)

	print('Reading data...')
	x_test, y_test = read_testing(args.ts_count)
	x_train, y_train = read_training(args.tr_count)

	model = EmotionsModel(
				use_lm=args.m,
				use_hog=args.g,
				use_cnn=args.c,
				create_new=args.n,
				emotions=get_emotions(),
				reduced_emotions=get_reduced_emotions(),
				verbose=True,
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