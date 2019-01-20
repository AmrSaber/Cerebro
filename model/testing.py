#! /user/bin/env python3

from model import create_model
from reader import read_data
from  keras.utils import  to_categorical

def main():
	model = create_model()
	print('Created Model.')

	print('Reading data...')
	(x_train, y_train), (x_test, y_test) = read_data(limit=1000)

	y_train = to_categorical(y_train, 7)
	y_test = to_categorical(y_test, 7)

	#print(x_train)
	#print(y_train)

	#print(x_test)
	#print(y_test)

	print('Training...')
	history = model.fit(x_train, y_train)

	print('Testing...')
	result = model.evaluate(x_test, y_test)
	print("Result:", result)

#if __name__ == '___main__': main()
main()
