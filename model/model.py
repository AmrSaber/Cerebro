#! /user/bin/env python3

import sys; sys.path.insert(1, '../image')
import feature_extraction

from pathlib import Path

import numpy as np

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, concatenate ,Dropout
from keras.models import Model
from keras.optimizers import SGD

class EmotionsModel(object):

	def __init__(self, targets_count, force_train=False):
		self.model_path = '../saved-models/emotions_model.f5'
		self.batch_size = 32
		self.epochs = 13
		if not force_train and self.has_saved_model():
			self.load_model()
			self.is_trained = True
		else:
			self.model = self._create_model(targets_count)
			self.is_trained = False

	def fit(self, xs, ys, should_save_model=True):
		xs = self.transform_input(xs)

		history = self.model.fit(xs, ys, batch_size=self.batch_size, epochs=self.epochs)

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
		return self.model.predict(faces, batch_size=self.batch_size)

	def transform_input(self, images):
		lms, hogs = [], []
		for i, image in enumerate(images):
			lms.append(feature_extraction.get_face_landmarks(image))
			# hogs.append(feature_extraction.sk_get_hog(image))
		return [np.array(images), np.array(lms)]

	def save_model(self):
		self.model.save(self.model_path)

	def load_model(self):
		self.model = keras.models.load_model(self.model_path)

	def has_saved_model(self):
		model_path = Path(self.model_path)
		return model_path.is_file()

	def _create_model(self, targets_count):
		conv_activation = 'relu'
		dense_activation = 'relu'
		Batch_Normalization = True
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

		flatLandmarks = Flatten()(inputLandmarks)
		#outputImage = concatenate([inputHOG, flatLandmarks])

		outputImage = Dense(units=1024, activation=dense_activation)(flatLandmarks)
		if (Batch_Normalization):
			outputImage = BatchNormalization(axis=-1)(outputImage)

		outputImage = Dense(units=128, activation= dense_activation)(outputImage)
		if (Batch_Normalization):
			outputImage = BatchNormalization(axis=-1)(outputImage)

		outputImage = Dense(units=128 ,activation=dense_activation)(outputImage)
		#outputImage = Flatten()(outputImage)
		concat_output = concatenate([outputCNN, outputImage])

		output = Dense(units=targets_count, activation='softmax')(concat_output)


		model = Model(inputs=[input_image, inputLandmarks], outputs=[output])

		sgd = SGD(lr=0.016, decay=0.864, momentum=0.95, nesterov=True)
		model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

		return model
