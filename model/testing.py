#! /user/bin/env python3

from model import *
from reader import read_data, emotions_map
from  keras.utils import  to_categorical

def main():
	model = create_model()
	print('Created Model.')

	print('Reading data...')
	(x_train, y_train), (x_test, y_test) = read_data(limit=-1)

	emotions_count = len(set(emotions_map))
	print(emotions_count)
	y_train = to_categorical(y_train, emotions_count)
	y_test = to_categorical(y_test, emotions_count)

	print('Training...')
	history = model.fit(x_train, y_train, epochs=1)

	print('Testing...')
	result = model.evaluate(x_test, y_test)
	print("Result:", result)

#if __name__ == '___main__': main()
main()
