#! /user/bin/env python3

import sys; sys.path.insert(1, '../image')
from image_processing import get_features

import numpy as np

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, concatenate ,Dropout
from keras.models import Model

class EmotionsModel(object):

	def __init__(self, targets_count, create_new=False):
		if not create_new and self.has_saved_model():
			self.model = self.load_model()
			self.is_trained = True
		else:
			self.model = self._create_model(targets_count)
			self.is_trained = False

	def fit(self, xs, ys, should_save_model=True):
		xs = self.transform_input(xs)

		# try with 13 epochs as the github example
		history = self.model.fit(xs, ys, epochs=1)

		self.is_trained = True
		if should_save_model: self.save_model()

		return history

	def test(self, faces, targets):
		if not self.is_trained: raise Exception("Model not trained yet")
		faces = self.transform_input(faces)
		return self.model.evaluate(faces, targets)

	def predict(self, faces):
		if not self.is_trained: raise Exception("Model not trained yet")
		faces = slef.transform_input(faces)
		pass

	def transform_input(self, images):
		lms, hogs = [], []
		for i, image in enumerate(images):
			landmarks, hog = get_features(image)
			lms.append(landmarks)
			hogs.append(hog)
		return [np.array(images), np.array(hogs), np.array(lms)]


	# TODO
	def save_model(self):
		pass

	# TODO
	def load_model(self):
		pass

	# TODO
	def has_saved_model(self):
		return False

	def _create_model(self, targets_count):
		conv_activation = 'relu'
		dense_activation = 'relu'
		Batch_Normalization = False #should be True
		keep_prob = 0.956

		# ========================== CNN Part ==========================
		input_image = Input(batch_shape=(None, 48, 48, 1), dtype='float32', name='input_image')
		x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=conv_activation)(input_image)

		if (Batch_Normalization):
				x = BatchNormalization(axis=-1)(x)

		x = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last")(x)
		x = Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation=conv_activation)(x)

		if (Batch_Normalization):
				x = BatchNormalization(axis=-1)(x)

		x = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation=conv_activation)(x)

		if (Batch_Normalization):
				x = BatchNormalization(axis=-1)(x)

		x = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last")(x)
		x = Dropout(rate=keep_prob)(x)
		x=  Dense(units=4096 , activation=dense_activation)(x)

		x = Dropout(rate=keep_prob)(x)
		x=  Dense(units=1024 , activation=dense_activation)(x)

		if (Batch_Normalization):
				x = BatchNormalization(axis=-1)(x)

		x = Flatten()(x)
		outputCNN = x

		# ========================== Image features part ==========================
		#inputHOG = Input(batch_shape=(None, 128), dtype='float32', name='input_HOG')
		inputLandmarks = Input(batch_shape=(None, 68, 2), dtype='float32', name='input_landmarks')
		outputImage = Flatten()(inputLandmarks)

		outputImage = Dense(units=1024, activation=dense_activation)(outputImage)
		if (Batch_Normalization):
				outputImage = BatchNormalization(axis=-1)(outputImage)

		outputImage = Dense(units=128, activation= dense_activation)(outputImage)
		if (Batch_Normalization):
				outputImage = BatchNormalization(axis=-1)(outputImage)

		outputImage = Dense(units=128 ,activation=dense_activation)(outputImage)
		#outputImage = Flatten()(outputImage)
		concat_output = concatenate([outputCNN, outputImage])

		output = Dense(units=targets_count, activation='softmax')(concat_output)


		model = Model(inputs=[input_image,inputLandmarks], outputs=[output])
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		return model
